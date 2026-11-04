#include "models.h"
#include "llama-impl.h"
#include "llama-ft-moe.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <cinttypes>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn > 1 || hparams.n_layer_nextn >= hparams.n_layer_all) {
        throw std::runtime_error("qwen4exp requires at most one MTP block after the trunk");
    }
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, 0);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    // MTP tensors sit in the trailing blocks; skip them entirely unless a draft head was asked for
    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        // the MTP block is structurally a trunk block: is_recr()/is_ple() are both false past
        // the trunk, so it takes the full-attention + MoE path below with no special casing
        const int flags = il < n_layer ? 0 : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        // NextN/MTP head. enorm/hnorm gate the two inputs; eh_proj is the checkpoint's
        // fc_embedding and fc_hidden fused side by side, so one matmul over
        // concat(e, h) computes fc_embedding@e + fc_hidden@h.
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        // the head's own output mixer, mirroring the trunk's hc_head_*: it collapses the
        // hc streams and stands in for the output norm, of which qwen4exp has none
        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);

        // qwen4exp sets mtp_use_dedicated_embeddings=false, so these are absent and the
        // head falls back to the trunk's embedding table and LM head
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
    }

    // A7 P1: host structs for GGML_OP_A7_GDN_FFN (trunk layers only; the tensors may be null, the graph checks them)
    a7_ffn.assign(n_layer, ggml_a7_ffn_layer{});
    for (int il = 0; il < (int) n_layer; ++il) {
        const auto & layer = layers[il];
        ggml_a7_ffn_layer & a = a7_ffn[il];
        a.hc_norm = layer.hc_ffn_norm;
        a.hc_down = layer.hc_ffn_down;
        a.hc_up = layer.hc_ffn_up;
        a.hc_inject = layer.hc_ffn_inject;
        a.gate_inp = layer.ffn_gate_inp;
        a.gate_inp_shexp = layer.ffn_gate_inp_shexp;
        a.gate_exps = layer.ffn_gate_exps;
        a.up_exps = layer.ffn_up_exps;
        a.down_exps = layer.ffn_down_exps;
        a.gate_shexp = layer.ffn_gate_shexp;
        a.up_shexp = layer.ffn_up_shexp;
        a.down_shexp = layer.ffn_down_shexp;
        a.hc_norm_next = il + 1 < (int) n_layer ? layers[il + 1].hc_attn_norm : nullptr;  // the op skips it when that layer is on another GPU
        a.il = il;
    }

    // A7 P2: host structs for GGML_OP_A7_GDN_ATTN (same rules as a7_ffn: filled once, never resized)
    a7_attn.assign(n_layer, ggml_a7_attn_layer{});
    for (int il = 0; il < (int) n_layer; ++il) {
        const auto & layer = layers[il];
        ggml_a7_attn_layer & a = a7_attn[il];
        a.hc_norm = layer.hc_attn_norm;
        a.hc_down = layer.hc_attn_down;
        a.hc_up = layer.hc_attn_up;
        a.hc_inject = layer.hc_attn_inject;
        a.wqkv = layer.wqkv;
        a.wqkv_gate = layer.wqkv_gate;
        a.ssm_alpha = layer.ssm_alpha;
        a.ssm_beta = layer.ssm_beta;
        a.ssm_dt = layer.ssm_dt;
        a.ssm_a = layer.ssm_a;
        a.ssm_conv1d = layer.ssm_conv1d;
        a.ssm_norm = layer.ssm_norm;
        a.ssm_out = layer.ssm_out;
        a.hc_ffn_norm = layer.hc_ffn_norm;
        a.il = il;
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// LLAMA_QWEN4EXP_HC_FUSION=1: run the hyper-connection mix and combine as the fused CUDA kernels
// of the FreeToken dense API even without the native MoE path, which needs a single device. Each
// CUDA device gets one weightless owner; the kernels only need to know which device they run on.
static const ggml_cuda_ft_dense_api * qwen4exp_hc_api = nullptr;

static void * qwen4exp_hc_owner(const ggml_tensor * w) {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QWEN4EXP_HC_FUSION");
        return env && atoi(env) != 0;
    }();
    if (!enabled || w == nullptr || w->buffer == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(w->buffer));
    if (dev == nullptr) {
        return nullptr;
    }
    // called for every mix of every graph build: look the device up before anything that may
    // query the driver (the CUDA device type getter calls cudaGetDeviceProperties)
    static std::mutex mutex;
    static std::map<ggml_backend_dev_t, void *> owners;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = owners.find(dev);
    if (it != owners.end()) {
        return it->second;
    }
    void * owner = nullptr;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const auto * api = ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ? nullptr :
        static_cast<const ggml_cuda_ft_dense_api *>(ggml_backend_reg_get_proc_address(reg, "ggml_cuda_ft_dense_api_v4"));
    if (api != nullptr) {
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
            if (ggml_backend_reg_dev_get(reg, i) == dev) {
                owner = api->create((int) i, nullptr, 0);
                qwen4exp_hc_api = api;
                break;
            }
        }
    }
    owners.emplace(dev, owner);
    return owner;
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // LLAMA_HC_FUSED=2: the whole mix as two kernels (ggml_hc_mix_down/up), for up to 8 tokens. Slower than the
    //     int8 GEMV path for now (f32 activations, few blocks), so only 1 (fused combine) is on by default.
    static const bool hc_fused_mix = [] { const char * e = getenv("LLAMA_HC_FUSED"); return e && atoi(e) >= 2; }();
    if (hc_fused_mix && !ft_moe && loras->empty() && nt <= 8 && ggml_is_contiguous(x) && n_embd % 512 == 0 &&
            w_down->type == GGML_TYPE_Q8_0 && w_up->type == GGML_TYPE_Q8_0 && w_norm->type == GGML_TYPE_F32 &&
            (!w_inject || w_inject->type == GGML_TYPE_F32)) {
        ggml_tensor * down = ggml_hc_mix_down(ctx0, x, w_norm, w_down, inject ? w_inject : nullptr, hparams.f_norm_rms_eps);
        cb(down, "hc_down", il);
        ggml_tensor * mixed = ggml_hc_mix_up(ctx0, x, w_norm, w_up, down);
        cb(mixed, "hc_mixed", il);
        if (inject) {
            *inject = down; // build_hc_combine reads the inject rows of this layout
        }
        return mixed;
    }

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = ft_moe && !cparams.warmup && loras->empty()
        ? ft_moe->build_hc(ctx0, xn, gate, n_embd) : nullptr;
    hc_owner = !ft_moe && loras->empty() ? qwen4exp_hc_owner(w_down) : nullptr;
    if (!mixed && hc_owner) {
        mixed = qwen4exp_hc_api->build_hc(hc_owner, ctx0, xn, gate, n_embd);
    }
    if (!mixed && cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else if (!mixed) {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // LLAMA_HC_FUSED=1: the combine as one kernel
    static const bool hc_fused_combine = [] { const char * e = getenv("LLAMA_HC_FUSED"); return e && atoi(e) >= 1; }();
    if (hc_fused_combine && il >= 0 && nt <= 8 && ggml_is_contiguous(residual) && ggml_is_contiguous(block_out) &&
            ggml_is_contiguous(inject) && inject->type == GGML_TYPE_F32) {
        ggml_tensor * cur = ggml_hc_combine(ctx0, block_out, residual, inject);
        cb(cur, "hc_combine", il);
        return cur;
    }
    if (inject->ne[0] != hc) {
        // inject is the ggml_hc_mix_down layout: [lr, inject (hc), 1/rms (hc)] per token
        if (ggml_is_contiguous(residual) && ggml_is_contiguous(block_out) && ggml_is_contiguous(inject)) {
            ggml_tensor * cur = ggml_hc_combine(ctx0, block_out, residual, inject);
            cb(cur, "hc_combine", il);
            return cur;
        }
        const int64_t lr = inject->ne[0] - 2*hc;
        inject = ggml_cont(ctx0, ggml_view_2d(ctx0, inject, hc, inject->ne[1], inject->nb[1], lr*ggml_element_size(inject)));
    }

    ggml_tensor * cur = ft_moe && !cparams.warmup && loras->empty()
        ? ft_moe->build_hc_combine(ctx0, residual, block_out, inject) : nullptr;
    if (!cur && hc_owner) {
        cur = qwen4exp_hc_api->build_hc_combine(hc_owner, ctx0, residual, block_out, inject);
    }
    if (!cur) {
        // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
        ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
        w = ggml_scale(ctx0, w, 2.0f);

        if (cparams.fused_dsv4_hc_post && il >= 0) {
            // identity comb: every stream adds the same block output, scaled by its own weight
            cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
            res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
        } else {
            w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

            ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
            b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

            cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
        }
    }
    cb(cur, "hc_combine", il);

    return cur;
}

// A7 P2: the layer numbers of a comma separated env list (LLAMA_A7_ATTN_LAYERS), empty when the variable is unset
static std::vector<int> qwen4exp_a7_env_layers(const char * name) {
    std::vector<int> v;
    const char * e = getenv(name);
    while (e && *e) {
        char * end = nullptr;
        const long x = strtol(e, &end, 10);
        if (end == e) {
            break;
        }
        v.push_back((int) x);
        e = *end == ',' ? end + 1 : end;
    }
    return v;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_qsa_base(model, params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    // tensor split: a view of the CPU embedding can not be split, use a real copy
    ggml_tensor * inp_3d = model.split_mode() == LLAMA_SPLIT_MODE_TENSOR
            ? ggml_cont_3d(ctx0, inpL, n_embd, 1, n_tokens)
            : ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0, inp_3d, n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    // A7 P1 (env LLAMA_A7_FFN_LAYERS=il,il,...): the ffn half of the listed GDN layers (hc_mix(ffn), router, MoE, shared expert,
    // hc_combine(ffn)) as one GGML_OP_A7_GDN_FFN, for the T = 4 verify window only. LLAMA_A7_CHECK=1: build both paths, the op compares
    // with the production result (A7CHECK line) and continues with the production values.
    static const std::vector<int> a7_layers = [] {
        std::vector<int> v;
        const char * e = getenv("LLAMA_A7_FFN_LAYERS");
        while (e && *e) {
            char * end = nullptr;
            const long x = strtol(e, &end, 10);
            if (end == e) {
                break;
            }
            v.push_back((int) x);
            e = *end == ',' ? end + 1 : end;
        }
        return v;
    }();
    static const bool a7_check = [] { const char * e = getenv("LLAMA_A7_CHECK"); return e && atoi(e) == 1; }();
    const auto & a7_model = static_cast<const llama_model_qwen4exp &>(model);
    auto a7_ffn_on = [&](int il) -> bool {
        if (a7_layers.empty() || std::find(a7_layers.begin(), a7_layers.end(), il) == a7_layers.end()) {
            return false;
        }
        // [a7-ffn-fullattn] LLAMA_A7_FFN_FULLATTN=1: the ffn half of the full-attention layers is the same block (hc_mix, router, MoE,
        // shared expert, hc_combine), so the listed full-attention layers take the op too (the q8 of R * hc_ffn_norm is made by its rquant kernel)
        static const bool a7_ffn_fullattn = [] { const char * e = getenv("LLAMA_A7_FFN_FULLATTN"); return e && atoi(e) != 0; }();
        if ((!hparams.is_recr(il) && !a7_ffn_fullattn) || il == n_layer - 1 || il >= (int) a7_model.a7_ffn.size()) {
            return false;
        }
        if (n_tokens != 4 || params.ubatch.n_seqs != 1 || n_outputs != 4 || params.gtype != LLM_GRAPH_TYPE_DEFAULT) {
            return false;
        }
        const auto & ly = model.layers[il];
        return ly.ffn_gate_exps && ly.ffn_gate_exps->type == GGML_TYPE_IQ2_S && ly.ffn_up_exps && ly.ffn_up_exps->type == GGML_TYPE_IQ2_S &&
            ly.ffn_down_exps && ly.ffn_down_exps->type == GGML_TYPE_IQ4_NL && !ly.ffn_gate_up_exps &&
            !ly.ffn_gate_exps_s && !ly.ffn_up_exps_s && !ly.ffn_down_exps_s && !ly.ffn_gate_shexp_s && !ly.ffn_up_shexp_s && !ly.ffn_down_shexp_s &&
            ly.ffn_gate_inp && ly.ffn_gate_inp_shexp && ly.ffn_gate_shexp && ly.ffn_up_shexp && ly.ffn_down_shexp &&
            ly.hc_ffn_norm && ly.hc_ffn_down && ly.hc_ffn_up && ly.hc_ffn_inject &&
            ft_moe == nullptr && moe_cache == nullptr && loras->empty() && model.split_mode() != LLAMA_SPLIT_MODE_TENSOR &&
            n_expert == 512 && n_expert_used == 10 && hparams.swiglu_clamp_exp[il] <= 1e-6f && hparams.dsv4_hc_mult == 4 && n_embd == 2560;
    };

    // A7 P2 (env LLAMA_A7_ATTN_LAYERS=il,il,...): the attn half of the listed GDN layers (hc_mix(attn), qkv / z / alpha / beta, conv, l2 norm, delta rule,
    // gated norm, ssm_out, hc_combine(attn)) as one GGML_OP_A7_GDN_ATTN, for the T = 4 verify window only; the op reads and writes the recurrent state
    // buffers r_l / s_l itself. LLAMA_A7_CHECK=1: the op runs on work copies of the state, then the production path runs on the real state, and a comparison
    // node (A7CHECK_ATTN line) compares result and state and passes the production result on. The conditions (a7_attn_why) contain those of supports_op except the GPU checks (SM count 80 / 72, shared memory): another GPU aborts in the scheduler.
    static const std::vector<int> a7_attn_layers = qwen4exp_a7_env_layers("LLAMA_A7_ATTN_LAYERS");
    static std::atomic<bool> a7_attn_off_logged{false};
    llm_graph_input_rs * a7_rs = inp->get_recr();
    auto a7_buf_dev = [](const ggml_tensor * t) -> ggml_backend_dev_t {
        return t && t->buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) : nullptr;
    };
    // the reason a listed layer cannot use the op in this T = 4 graph (nullptr: it can)
    auto a7_attn_why = [&](int il) -> const char * {
        if (!hparams.is_recr(il) || il < 0 || il == n_layer - 1 || il >= (int) a7_model.a7_attn.size()) {
            return "not a GDN layer / last layer";
        }
        if (cparams.n_rs_seq != 3) {
            return "n_rs_seq != 3 (K = 4 state slots)";
        }
        const auto & ly = model.layers[il];
        auto is2d = [](const ggml_tensor * t, ggml_type ty, int64_t ne0, int64_t ne1) {
            return t && t->type == ty && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == 1 && t->ne[3] == 1;
        };
        if (!(is2d(ly.hc_attn_norm, GGML_TYPE_F32, 2560, 4) && is2d(ly.hc_attn_down, GGML_TYPE_Q8_0, 10240, 320) && is2d(ly.hc_attn_up, GGML_TYPE_Q8_0, 320, 10240) &&
              is2d(ly.hc_attn_inject, GGML_TYPE_F32, 10240, 4) && is2d(ly.hc_ffn_norm, GGML_TYPE_F32, 2560, 4) &&
              is2d(ly.wqkv, GGML_TYPE_Q8_0, 2560, 10240) && is2d(ly.wqkv_gate, GGML_TYPE_Q8_0, 2560, 6144) &&
              is2d(ly.ssm_alpha, GGML_TYPE_F32, 2560, 48) && is2d(ly.ssm_beta, GGML_TYPE_F32, 2560, 48) &&
              is2d(ly.ssm_dt, GGML_TYPE_F32, 48, 1) && is2d(ly.ssm_a, GGML_TYPE_F32, 48, 1) && is2d(ly.ssm_conv1d, GGML_TYPE_F32, 4, 10240) &&
              is2d(ly.ssm_norm, GGML_TYPE_F32, 128, 1) && is2d(ly.ssm_out, GGML_TYPE_Q8_0, 6144, 2560))) {
            return "weight types / shapes";
        }
        if (ly.wqkv_s || ly.wqkv_gate_s || ly.ssm_alpha_s || ly.ssm_beta_s || ly.ssm_out_s) {
            return "weight scales";
        }
        if (!(hparams.ssm_d_state == 128 && hparams.ssm_n_group == 16 && hparams.ssm_dt_rank == 48 && hparams.ssm_d_inner == 6144 &&
              hparams.n_embd_r() == 3 * 10240 && hparams.n_embd_s() == 128 * 128 * 48 && n_embd == 2560 && hparams.dsv4_hc_mult == 4)) {
            return "hparams";
        }
        // (no hc_owner condition: the op replaces both the mix and the combine of the attn half, so the fused hc path is not used there)
        if (ft_moe != nullptr || !loras->empty() || model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            return "ft_moe / lora / tensor split";
        }
        const ggml_tensor * r_l = a7_rs->mctx->get_r_l(il);
        const ggml_tensor * s_l = a7_rs->mctx->get_s_l(il);
        const int64_t rows = (int64_t) a7_rs->mctx->get_size() * 4;
        if (!(is2d(r_l, GGML_TYPE_F32, 3 * 10240, rows) && is2d(s_l, GGML_TYPE_F32, 128 * 128 * 48, rows) && r_l->buffer && s_l->buffer &&
              !ggml_backend_buffer_is_host(r_l->buffer) && !ggml_backend_buffer_is_host(s_l->buffer))) {
            return "state buffers";
        }
        // supports_op of the CUDA backend wants every weight and both state buffers on the device of the op: same device as hc_attn_norm
        const ggml_backend_dev_t dev = a7_buf_dev(ly.hc_attn_norm);
        const ggml_tensor * ws[] = { ly.hc_attn_down, ly.hc_attn_up, ly.hc_attn_inject, ly.hc_ffn_norm, ly.wqkv, ly.wqkv_gate, ly.ssm_alpha, ly.ssm_beta,
                                     ly.ssm_dt, ly.ssm_a, ly.ssm_conv1d, ly.ssm_norm, ly.ssm_out, r_l, s_l };
        if (dev == nullptr) {
            return "device of hc_attn_norm";
        }
        for (const ggml_tensor * w : ws) {
            if (a7_buf_dev(w) != dev || w->extra != nullptr) {
                return "a weight or state buffer on another device (or with extra)";
            }
        }
        // extra states (n_rs > n_seqs; full-size graphs and cells in use beyond the sequence's)
        if (a7_rs->mctx->get_n_rs() != (uint32_t) params.ubatch.n_seqs) {
            return "n_rs != n_seqs";
        }
        return nullptr;
    };
    auto a7_attn_on = [&](int il) -> bool {
        if (a7_attn_layers.empty() || std::find(a7_attn_layers.begin(), a7_attn_layers.end(), il) == a7_attn_layers.end()) {
            return false;
        }
        // other graphs (prompt batches, reserve graphs of other sizes, MTP) never use the op: no log
        if (n_tokens != 4 || params.ubatch.n_seqs != 1 || n_outputs != 4 || params.gtype != LLM_GRAPH_TYPE_DEFAULT) {
            return false;
        }
        const char * why = a7_attn_why(il);
        if (why != nullptr) {
            if (!a7_attn_off_logged.exchange(true)) {
                fprintf(stderr, "A7: attn op off on layer %d: %s (n_rs=%u n_seqs=%d n_rs_seq=%u)\n", il, why, a7_rs->mctx->get_n_rs(),
                        (int) params.ubatch.n_seqs, cparams.n_rs_seq);
            }
            return false;
        }
        return true;
    };

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        // A7 P2: the attn half as one op (a7_attn_use); in the check mode the production path below also runs, on the real state
        const bool a7_attn_use = a7_attn_on(il);
        ggml_tensor * a7_attn_res = nullptr;   // check mode: the A7 result
        ggml_tensor * a7_conv_w   = nullptr;   // check mode: the work copies of the state the A7 op ran on
        ggml_tensor * a7_ssm_w    = nullptr;
        ggml_tensor * a7_res_in   = res_hc;
        if (a7_attn_use) {
            const llama_memory_recurrent_context * rmctx = a7_rs->mctx;
            ggml_tensor * r_l = rmctx->get_r_l(il);
            ggml_tensor * s_l = rmctx->get_s_l(il);
            const ggml_a7_attn_layer * LA = &a7_model.a7_attn[il];
            // the clearing of the state row that build_rs does (the gather and the extra copy are not needed: the op reads the rows itself)
            build_rs_zero(a7_rs, r_l, (int32_t) r_l->ne[0], (int32_t) params.ubatch.n_seqs);
            build_rs_zero(a7_rs, s_l, (int32_t) s_l->ne[0], (int32_t) params.ubatch.n_seqs);
            ggml_tensor * conv_src = r_l;
            ggml_tensor * ssm_src  = s_l;
            if (a7_check) {  // work copies (whole tensors, after the clearing), the op runs on them
                a7_conv_w = ggml_cpy(ctx0, r_l, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, r_l->ne[0], r_l->ne[1]));
                a7_ssm_w  = ggml_cpy(ctx0, s_l, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, s_l->ne[0], s_l->ne[1]));
                ggml_build_forward_expand(gf, a7_conv_w);
                ggml_build_forward_expand(gf, a7_ssm_w);
                conv_src = a7_conv_w;
                ssm_src  = a7_ssm_w;
            }
            // rq_ready: the q8 of R * hc_attn_norm was made by the down stage of the ffn op of the previous layer (see ggml-cuda/a7-gdn.cu, work-area tag)
            const bool rq_ready = !a7_check && il > 0 && a7_ffn_on(il - 1) && !hparams.is_ple(il) &&
                a7_buf_dev(model.layers[il - 1].hc_ffn_norm) != nullptr &&
                a7_buf_dev(model.layers[il - 1].hc_ffn_norm) == a7_buf_dev(model.layers[il].hc_attn_norm) &&
                a7_model.a7_ffn[il - 1].hc_norm_next == model.layers[il].hc_attn_norm;
            ggml_tensor * op = ggml_a7_gdn_attn(ctx0, res_hc, model.layers[il].hc_attn_norm, a7_rs->s_copy_main, conv_src, ssm_src, LA);
            ggml_a7_gdn_attn_params(op, (int) rmctx->get_head(), (int) rmctx->get_size(), (int) cparams.n_rs_seq + 1, rq_ready ? 1 : 0,
                    hparams.f_norm_rms_eps, a7_check ? 1 : 0);
            cb(op, "a7_attn_out", il);
            ggml_build_forward_expand(gf, op);  // after the clearing nodes above: keeps the order
            if (a7_check) {
                a7_attn_res = op;
            } else {
                res_hc = op;
            }
        }

        ggml_tensor * inject = nullptr;   // also used by the ffn half below
        ggml_tensor * cur    = nullptr;
        if (!a7_attn_use || a7_check) {
        cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // an unmasked MTP export needs a hidden row for every token, so in that case the
        // gather is deferred until after t_h_nextn is taken below
        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        if (a7_attn_use && a7_check) {
            // the comparison node, after the production combine and after the production state writes (they are side-effect nodes expanded before it);
            // the model continues with the production result (the dst of this node is a copy of it)
            const llama_memory_recurrent_context * rmctx = a7_rs->mctx;
            res_hc = ggml_a7_gdn_attn_cmp(ctx0, a7_attn_res, res_hc, a7_res_in, a7_rs->s_copy_main, a7_conv_w, a7_ssm_w,
                    rmctx->get_r_l(il), rmctx->get_s_l(il), &a7_model.a7_attn[il], (int) rmctx->get_head(), (int) rmctx->get_size(), (int) cparams.n_rs_seq + 1);
            cb(res_hc, "a7_attn_cmp", il);
        }
        }

        // the ffn half of the layer: rq_ready = the q8 of R * hc_ffn_norm was made by the ssmout stage of the attn op of this layer
        if (a7_ffn_on(il) && !a7_check) {
            res_hc = ggml_a7_gdn_ffn(ctx0, res_hc, model.layers[il].hc_ffn_norm, nullptr, &a7_model.a7_ffn[il], !a7_check && a7_attn_use ? 1 : 0);
        } else {
            ggml_tensor * res_hc_in = res_hc;

            cur = build_hc_mix(res_hc,
                    model.layers[il].hc_ffn_norm,
                    model.layers[il].hc_ffn_down,
                    model.layers[il].hc_ffn_up,
                    model.layers[il].hc_ffn_inject,
                    &inject, il);

            cur = build_layer_ffn(cur, il);
            cb(cur, "ffn_out", il);

            res_hc = build_hc_combine(res_hc, cur, inject, il);

            if (a7_check && a7_ffn_on(il)) {
                // the op reads the same input as the production path, compares with its result and passes that result on
                res_hc = ggml_a7_gdn_ffn(ctx0, res_hc_in, model.layers[il].hc_ffn_norm, res_hc, &a7_model.a7_ffn[il], 0);
            }
        }

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // The MTP head consumes the wide residual, before the head mixer collapses it. Export the
    // combine result itself rather than a reshape of it: a pure view gets no backend assignment
    // from the scheduler, and the readback in llama_context looks one up. It is contiguous, so
    // [n_embd, hc, rows] already has the [n_embd_out, rows] layout the reader expects, and it
    // carries exactly the right rows either way -- gathered above when masked, ungathered when not.
    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        // deferred from the last layer: collapse to the output rows now that the export is taken
        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// The draft consumes and returns the wide HC residual.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing nextn.hc_head_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = cparams.mtp_kv_only ? nullptr : build_inp_out_ids(); // [mtp-kvonly] no outputs

    auto * inp_mem = build_inp_mem_hybrid();
    inp_mem->uses_recr = false; // the MTP block is an attention layer
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_mem->mctx);
    auto * inp_attn = inp_mem->get_attn();

    // grouped RMSNorm over the wide stream: normalise each hc stream, then scale with the
    // head's [n_embd, hc] gamma, exactly as build_hc_mix does
    ggml_tensor * h_norm = ggml_mul(ctx0, ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps), layer.nextn.hnorm);
    cb(h_norm, "mtp_hnorm", il);

    // the token embedding is shared across the streams, so broadcast it to hc copies
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // eh_proj holds fc_embedding and fc_hidden side by side, so this one matmul is
    // fc_embedding @ e_norm + fc_hidden @ h_norm, applied to each stream independently.
    // Keeping the streams distinct here is the point of the hyper-connection residual:
    // pooling them before the projection would throw that away.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    // as one [2*n_embd, hc*n_tokens] matrix: a 3-D src1 is multiplied per token and re-reads the weight every time
    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj,
            ggml_reshape_2d(ctx0, concat, concat->ne[0], concat->ne[1]*concat->ne[2]), layer.nextn.eh_proj_s);
    res_hc = ggml_reshape_3d(ctx0, res_hc, res_hc->ne[0], concat->ne[1], concat->ne[2]);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    if (cparams.mtp_kv_only) {
        // [mtp-kvonly] prompt catch-up: these rows only fill the MTP block's caches for later drafts, nothing is read back
        build_layer_kv_only(inp_attn, mctx_hyb, cur, inp_pos, sections, il);
        return;
    }

    cur = build_layer_attn(inp_attn, mctx_hyb, cur, inp_pos, sections, il);

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    // ---- MoE, identical to the trunk's build_layer_ffn ----
    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // The next draft step re-enters here, so export the wide stream before it is collapsed.
    // As in the trunk, export the combine result rather than a reshape view of it.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    // the head's own mixer collapses the streams and doubles as the output norm
    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // deliberately no res->t_embd: it would be n_embd wide while the context sizes its
    // embedding buffer by n_embd_out (the wide stream). The driver reads t_h_nextn instead.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN4EXP MTP: missing LM head (nextn.shared_head_head or model.output)");

    if (model.mtp_dv_head && model.mtp_dv_d2t && model.mtp_dv_src == head_w) {
        // [mtp-draft-vocab] reduced-vocabulary head: only the listed rows are multiplied, the rest of the logits are -INF
        cur = build_lora_mm(model.mtp_dv_head, cur, head_s);
        const int64_t n_draft = cur->ne[0];
        const int64_t n_out   = cur->ne[1];
        const int64_t n_voc   = (int64_t) model.vocab.n_tokens();
        GGML_ASSERT(model.mtp_dv_d2t->type == GGML_TYPE_I64 && model.mtp_dv_d2t->ne[0] == n_draft);
        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_voc, n_out), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,                 1,       n_draft, n_out),
                ggml_reshape_3d(ctx0, model.mtp_dv_d2t,    n_draft, 1,       1));
        cur = ggml_reshape_2d(ctx0, cur, n_voc, n_out);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// Block-sparse attention kernel (CUDA, Q8_0 K/V, head size 256): each query reads its selected
// blocks straight from the cache. It lets the fast path take prompt batches too.
// LLAMA_QSA_SPARSE_KERNEL=0 keeps the gather + flash attention path (and its 16-token limit).
static bool qwen4exp_qsa_sparse_kernel_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QSA_SPARSE_KERNEL");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
static bool qwen4exp_qsa_blk_fast(
        const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        uint32_t ratio);

class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        // the pooled block cache adds its update to this graph, or replaces this path outright
        const bool blk_now = mctx->blk_ok();
        if (blk_now != had_blk || (blk_now && qwen4exp_qsa_blk_fast(mctx, params.ubatch, cparams, *hparams, ratio))) { // [v4bits]
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        res &= bias->ne[0]      == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // whether the graph also updates the pooled block cache, and what decides the fast path
    bool had_blk = false;
    llama_cparams cparams = {};   // [v4bits] a copy: the graph context's cparams live in the caller's llm_graph_params
    const llama_hparams * hparams = nullptr;
};

// Fixed-cost decode path over the pooled block-key cache. Taken for small causal text batches
// once every query sees more complete blocks than the budget: each query then attends to its
// indexer_top_k/ratio best whole blocks plus its incomplete tail block, the reference selection.
static bool qwen4exp_qsa_blk_fast(
        const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        uint32_t ratio) {
    static const bool disabled = [] {
        const char * env = getenv("LLAMA_QSA_DENSE");
        return env && atoi(env) != 0;
    }();
    if (disabled || ratio == 0 || !mctx->blk_ok() || !cparams.causal_attn || !cparams.flash_attn || hparams.use_alibi) {
        return false;
    }
    // the gather path materializes every query's selected rows: small batches only
    const uint32_t max_tokens = qwen4exp_qsa_sparse_kernel_enabled() ? UINT32_MAX : 16;
    if (ubatch.n_tokens == 0 || ubatch.n_tokens > max_tokens || mctx->get_n_stream() != 1) {
        return false;
    }
    // blocks cut the cell array (blk_ok: cell = rank), so the first query's cell says how many complete blocks it has
    const int64_t c0    = mctx->get_idx()->ubatch_cell0();
    const int64_t n_sel = hparams.indexer_top_k / ratio;
    return (c0 + 1) / ratio >= n_sel + 1;
}

// [qsa-reserve-blk] LLAMA_QSA_RESERVE_BLK=1: a reserve graph (init_full context: no ubatch is ever applied, so blk_ok() is false) takes
// the pooled-block path that prompt ubatches past the first indexer_top_k + ratio cells take at run time, instead of the masked
// dense path whose [n_kv, n_tokens] tensors (~14 bytes per cell and token) would otherwise set the compute buffer. Only the
// conditions of qwen4exp_qsa_blk_fast that do not depend on the cells (see patch-mt-qsareserve.py)
static bool qwen4exp_qsa_reserve_blk(
        const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams) {
    static const bool on = [] {
        const char * env = getenv("LLAMA_QSA_RESERVE_BLK");
        return env && atoi(env) != 0;
    }();
    static const bool dense = [] {
        const char * env = getenv("LLAMA_QSA_DENSE");
        return env && atoi(env) != 0;
    }();
    if (!on || dense || mctx->get_attn()->has_ubatch() || mctx->get_blk() == nullptr || mctx->get_idx() == nullptr) {
        return false;
    }
    if (!cparams.causal_attn || !cparams.flash_attn || hparams.use_alibi || (!cparams.kv_unified && cparams.n_seq_max > 1)) {
        return false;
    }
    const uint32_t max_tokens = qwen4exp_qsa_sparse_kernel_enabled() ? UINT32_MAX : 16;
    return ubatch.n_tokens > 0 && ubatch.n_tokens <= max_tokens;
}

bool llm_build_qsa_base::qsa_block_fast(const llama_memory_hybrid_idx_context * ctx, uint32_t ratio) const {
    if (getenv("LLAMA_QSA27B_REFERENCE")) { return false; }
    return qwen4exp_qsa_blk_fast(ctx, ubatch, cparams, hparams, ratio) ||
           qwen4exp_qsa_reserve_blk(ctx, ubatch, cparams, hparams);
}

class llm_graph_input_qsa_blk : public llm_graph_input_i {
public:
    llm_graph_input_qsa_blk(const llama_memory_hybrid_idx_context * mctx, const llama_cparams & cparams,
            const llama_hparams & hparams, uint32_t ratio, bool fast, bool rebuild, bool own_always) : // [blk-rebuild]
        mctx(mctx), cparams(cparams), hparams(hparams), ratio(ratio), fast(fast), rebuild(rebuild), own_always(own_always) {}
    virtual ~llm_graph_input_qsa_blk() = default;

    static int64_t n_touch(int64_t n_tokens, uint32_t ratio) {
        return (n_tokens + ratio - 1)/ratio + 1;
    }

    // [blk-rebuild] rows re-pooled by one ubatch whose first token sits in cell c0: the blocks it touches, or with rebuild every
    // block from 0 through the one holding its last token
    static int64_t n_touch(int64_t n_tokens, int64_t c0, uint32_t ratio, bool rebuild) {
        return rebuild ? (c0 + n_tokens - 1)/ratio + 1 : n_touch(n_tokens, ratio);
    }

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t r = ratio;
        const int64_t n = ubatch->n_tokens;

        // blk_ok() holds: token i sits in cell c0 + i, and blocks cut the cell array
        const auto *  idx = mctx->get_idx();
        const int64_t c0  = idx->ubatch_cell0();

        // every block holding a token of this ubatch is re-pooled; the rest of the rows pad
        // into the scratch row
        const int64_t nbt     = touch_src->ne[0];
        const int64_t b0      = rebuild ? 0 : c0/r;   // [blk-rebuild] after a restore every block is re-pooled, from block 0
        const int64_t b1      = (c0 + n - 1)/r;
        const int32_t scratch = (int32_t) mctx->get_blk()->get_size() - 1;

        GGML_ASSERT(ggml_backend_buffer_is_host(touch_src->buffer));
        int32_t * src = (int32_t *) touch_src->data;
        int32_t * dst = (int32_t *) touch_dst->data;
        int32_t * pos = (int32_t *) touch_pos->data;
        for (int64_t k = 0; k < nbt; ++k) {
            const int64_t b  = b0 + k;
            const bool    ok = b <= b1;
            src[k] = ok ? (int32_t) b : 0;
            dst[k] = ok ? (int32_t) b : scratch;
            // a pooled key is rotated to the position of its first token: b*r for text, (t, y, x) for an image
            int32_t sec_pos[4] = { 0, 0, 0, 0 };
            if (ok) {
                idx->cell_rope_pos((uint32_t) (b*r), sec_pos);
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                pos[sec*nbt + k] = sec_pos[sec];
            }
        }

        if (!fast) {
            return;
        }

        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);

        // [mtp-kvonly] a cache-write-only graph uses k_idxs and the touch rows but never selects: the rest is unallocated
        if (vis->buffer == nullptr) {
            return;
        }

        const int64_t n_sel  = hparams.indexer_top_k/r;
        const int64_t n_tail = tail_ids->ne[0];          // the tail block, then padding blocks
        const int64_t width  = sel_mask->ne[0];          // (n_sel + n_tail)*r cells per query

        const ggml_fp16_t zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

        float       * v  = (float       *) vis->data;
        int32_t     * tb = (int32_t     *) tail_ids->data;
        ggml_fp16_t * m  = (ggml_fp16_t *) sel_mask->data;
        for (int64_t i = 0; i < n; ++i) {
            // complete blocks up to the query's cell are visible; the rest of its own
            // block, up to and including the query, is the always-visible tail
            const int64_t q   = c0 + i;
            const int64_t c   = (q + 1)/r;
            const int64_t rem = own_always ? q%r + 1 : (q + 1)%r;
            v[i] = (float) c;
            for (int64_t j = 0; j < n_tail; ++j) {
                tb[i*n_tail + j] = own_always ? (int32_t) (q/r) : (rem ? (int32_t) c : 0);
            }
            // selected blocks are complete and visible, the tail is visible up to the query,
            // the padding blocks are off
            if (m) {
                ggml_fp16_t * mi = m + i*width;
                for (int64_t cell = 0; cell < width; ++cell) {
                    const int64_t t = cell - n_sel*r;
                    mi[cell] = t < 0 ? zero : (t < rem ? zero : neg_inf);
                }
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        bool res = mctx->get_idx() != nullptr && mctx->blk_ok();
        res = res && qwen4exp_qsa_blk_fast(mctx, params.ubatch, cparams, hparams, ratio) == fast;
        // [blk-rebuild] a rebuild graph re-pools blocks 0..last, a normal one only the touched blocks: never reuse one for the other,
        // and the row count must come from the same formula as the graph being reused
        res = res && mctx->blk_rebuild() == rebuild;
        res = res && touch_src->ne[0] == n_touch(params.ubatch.n_tokens, (res && rebuild) ? (int64_t) mctx->get_idx()->ubatch_cell0() : 0, ratio, rebuild);
        if (res && fast) {
            res = vis->ne[1] == params.ubatch.n_tokens && k_idxs->ne[0] == params.ubatch.n_tokens &&
                  sel_mask->ne[3] == params.ubatch.n_tokens;
        }
        return res;
    }

    ggml_tensor * touch_src = nullptr;  // I32 [n_touch]    block rows pooled from the indexer cache
    ggml_tensor * touch_dst = nullptr;  // I32 [n_touch]    rows written in the pooled cache
    ggml_tensor * touch_pos = nullptr;  // I32 [4*n_touch]  rope positions of those blocks

    // fast selection only
    ggml_tensor * k_idxs    = nullptr;  // I32 [n_tokens]         indexer cache cells of this ubatch
    ggml_tensor * vis       = nullptr;  // F32 [1, n_tokens]      complete blocks each query sees
    ggml_tensor * tail_ids  = nullptr;  // I32 [n_tail, n_tokens]           tail block, repeated as padding
    ggml_tensor * sel_mask  = nullptr;  // F16 [width, 1, 1, n_tokens]      attention mask over the gathered cells

    // graph node, not an input: the visibility bias built from vis, shared by the layers
    ggml_tensor * vis_bias  = nullptr;

    const llama_memory_hybrid_idx_context * mctx;
    const llama_cparams   cparams;  // [v4bits] a copy: the graph context's cparams live in the caller's llm_graph_params
    const llama_hparams & hparams;  // the model's
    const uint32_t ratio;
    const bool fast;

    // [blk-rebuild] this graph re-pools every block from 0 (set from the memory context when the graph is built)
    const bool rebuild;
    const bool own_always;
};

llm_graph_input_qsa_blk * llm_build_qsa_base::get_qsa_blk_input(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        uint32_t                                ratio,
        bool                                    fast) {
    const auto it = qsa_blk_inps.find(ratio);
    if (it != qsa_blk_inps.end()) {
        GGML_ASSERT(it->second->fast == fast);
        return it->second;
    }

    // [qsa-reserve-blk] a reserve graph (no ubatch placed) sizes the worst re-pool: every block of a full cache
    const bool rsv = !mctx_hyb->get_attn()->has_ubatch();
    auto inp = std::make_unique<llm_graph_input_qsa_blk>(mctx_hyb, cparams, model.hparams, ratio, fast, rsv || mctx_hyb->blk_rebuild(), model.arch == LLM_ARCH_QWEN35); // [v4bits] [blk-rebuild]

    // [blk-rebuild] blk_ok() holds here, so the ubatch has a first cell
    const int64_t c0r = rsv ? std::max<int64_t>(0, (int64_t) mctx_hyb->get_idx()->get_n_kv() - n_tokens)
                            : (inp->rebuild ? (int64_t) mctx_hyb->get_idx()->ubatch_cell0() : 0);
    const int64_t nbt = llm_graph_input_qsa_blk::n_touch(n_tokens, c0r, ratio, inp->rebuild);

    inp->touch_src = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, nbt);
    inp->touch_dst = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, nbt);
    inp->touch_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*nbt);
    ggml_set_input(inp->touch_src);
    ggml_set_input(inp->touch_dst);
    ggml_set_input(inp->touch_pos);

    if (fast) {
        inp->k_idxs    = mctx_hyb->get_idx()->build_input_k_idxs(ctx0, ubatch);
        inp->vis       = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        // the selected blocks plus the tail block, padded to a FATTN_KQ_STRIDE multiple of cells
        // so the vector FA kernels stay eligible; the padding repeats the tail block, masked off
        const int64_t n_sel  = hparams.indexer_top_k/ratio;
        const int64_t n_wblk = GGML_PAD((n_sel + 1)*ratio, 256)/ratio;

        inp->tail_ids  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_wblk - n_sel, n_tokens);
        inp->sel_mask  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_wblk*ratio, 1, 1, n_tokens);
        ggml_set_input(inp->vis);
        ggml_set_input(inp->tail_ids);
        ggml_set_input(inp->sel_mask);
    }

    auto * res_inp = inp.get();
    res->add_input(std::move(inp));
    qsa_blk_inps.emplace(ratio, res_inp);

    return res_inp;
}

// Re-pool the blocks this ubatch touched into the pooled cache: the same mean, norm and
// rotation build_qsa_top_k applies to every block, applied to these rows only. The members
// of block b are cells b*ratio.. (cell = rank, see blk_ok), so they are one row of a block view.
ggml_tensor * llm_build_qsa_base::build_qsa_blk_update(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_blk *               inp,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t r       = inp->ratio;
    const int64_t n_kv    = mctx_idx->get_n_kv();

    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    ggml_tensor * k_blk = ggml_view_2d(ctx0, k_all, idx_dim*r, n_kv/r, r*k_all->nb[2], 0);

    ggml_tensor * members = ggml_get_rows(ctx0, k_blk, inp->touch_src);
    const int64_t nbt = members->ne[1];
    members = ggml_reshape_3d(ctx0, members, idx_dim, r, nbt);

    // start the sum from two strided views: no separate copy of the first slice
    const auto member = [&](int64_t i) {
        return ggml_view_2d(ctx0, members, idx_dim, nbt, members->nb[2], i*members->nb[1]);
    };
    ggml_tensor * pooled = r > 1 ? ggml_add(ctx0, member(0), member(1)) : ggml_cont(ctx0, member(0));
    for (int64_t i = 2; i < r; ++i) {
        pooled = ggml_add(ctx0, pooled, member(i));
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);

    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, nbt, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, nbt);
    pooled = ggml_rope_multi(ctx0, pooled, inp->touch_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_2d(ctx0, pooled, idx_dim, nbt);
    cb(pooled, "indexer_k_blk", il);

    ggml_tensor * cache = mctx_hyb->get_blk()->get_k_storage(il);
    cache = ggml_view_2d(ctx0, cache, idx_dim, cache->ne[1], cache->nb[1], 0);

    ggml_tensor * upd = ggml_set_rows(ctx0, cache, pooled, inp->touch_dst);
    ggml_build_forward_expand(gf, upd);

    return upd;
}

// Score the pooled blocks and pick each query's best indexer_top_k/ratio blocks.
ggml_tensor * llm_build_qsa_base::build_qsa_blk_select(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_blk *               inp,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t r       = inp->ratio;
    const int64_t n_kv    = mctx_idx->get_n_kv();
    const int64_t n_blk   = n_kv/r;
    const int64_t n_sel   = hparams.indexer_top_k/r;

    // cache this ubatch's raw indexer keys, then re-pool the blocks they touch
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    if (model.layers[il].index_k_pre_norm) {
        k_raw = build_norm(k_raw, model.layers[il].index_k_pre_norm, nullptr, LLM_NORM_RMS, il);
    }
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    ggml_tensor * blk = build_qsa_blk_update(mctx_hyb, inp, sections, il);
    blk = ggml_view_2d(ctx0, blk, idx_dim, n_blk, blk->nb[1], 0);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // same score as build_qsa_top_k: rectified per-head dot products, summed over the heads.
    // Prompt batches on a long cache go through in token chunks, so the [n_blk, heads, tokens]
    // score stays around 64 MB (selection is per token, so chunking changes nothing)
    int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(n_tokens,
            (int64_t) (64ull << 20) / (int64_t) (n_blk*n_idx_h*sizeof(float))));
    // [qsa-fixchunk] LLAMA_QSA_CHUNK_TOKENS=N: a token chunk that does not depend on n_kv (while N is within the budget above),
    // so a graph for a larger n_kv only has larger tensors and pp plan-ahead can plan past one chunk (see patch-mt-qsafixchunk.py)
    static const int64_t fix_chunk = [] {
        const char * env = getenv("LLAMA_QSA_CHUNK_TOKENS");
        return env ? (int64_t) std::max(0, atoi(env)) : (int64_t) 0;
    }();
    if (fix_chunk > 0) {
        chunk = std::min(chunk, fix_chunk);
    }

    // [qsa-fixcount] LLAMA_QSA_CHUNK_FIXCOUNT=1: a prompt ubatch shorter than n_ubatch (the last one of a call) gets a full ubatch's
    // chunk count in nearly equal chunks, so its graph has the full ubatch's node count and fits the planned reservation (a
    // re-reserve synchronizes all GPUs and serializes the pipeline's tail; see patch-mt-qsafixcount.py)
    static const bool fix_count = [] {
        const char * env = getenv("LLAMA_QSA_CHUNK_FIXCOUNT");
        return env != nullptr && atoi(env) != 0;
    }();
    int64_t n_eq = 0;
    if (fix_count && fix_chunk > 0 && chunk == fix_chunk && n_tokens >= 64 && n_tokens < (int64_t) cparams.n_ubatch) {
        const int64_t n_full = ((int64_t) cparams.n_ubatch + chunk - 1)/chunk;
        if (n_tokens >= n_full && n_full > (n_tokens + chunk - 1)/chunk) {
            n_eq = n_full;
        }
    }

    ggml_tensor * top = nullptr;
    int64_t i_chunk = 0;
    for (int64_t t0 = 0, nt = 0; t0 < n_tokens; t0 += nt, ++i_chunk) {
        nt = n_eq > 0 ? (i_chunk + 1)*n_tokens/n_eq - t0 : std::min(chunk, n_tokens - t0);

        ggml_tensor * qc = ggml_view_2d(ctx0, q, idx_dim, n_idx_h*nt, q->nb[1], t0*q->nb[2]);
        ggml_tensor * score = ggml_mul_mat(ctx0, blk, qc);
        score = ggml_reshape_3d(ctx0, score, n_blk, n_idx_h, nt);
        score = ggml_relu(ctx0, score);

        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_2d(ctx0, score, n_blk, nt, score->nb[2], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }
        cb(summed, "indexer_score", il);

        // blocks at or past a query's tail are out of reach: 0 below vis, -1e30 from there on.
        // it depends on the query cells only, so a single-chunk graph builds it once for all layers
        ggml_tensor * vis_bias = nullptr;
        if (nt == n_tokens && inp->vis_bias != nullptr && inp->vis_bias->ne[0] == n_blk) {
            vis_bias = inp->vis_bias;
        } else {
            ggml_tensor * vis = ggml_view_2d(ctx0, inp->vis, 1, nt, inp->vis->nb[1], t0*inp->vis->nb[1]);
            ggml_tensor * idx = ggml_repeat_4d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_blk, 1.0f), n_blk, nt, 1, 1);
            ggml_tensor * seen = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, idx, vis), -1.0f, -0.5f));
            vis_bias = ggml_scale_bias(ctx0, seen, 1e30f, -1e30f);
            if (nt == n_tokens) {
                inp->vis_bias = vis_bias;
            }
        }
        summed = ggml_add(ctx0, summed, vis_bias);

        ggml_tensor * tc = ggml_top_k(ctx0, summed, n_sel);
        if (!ggml_is_contiguous(tc)) {
            tc = ggml_cont(ctx0, tc);
        }
        top = top ? ggml_concat(ctx0, top, tc, 1) : tc;
    }
    cb(top, "indexer_top_blk", il);

    return top;
}

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias);
        qsa->had_blk = mctx_hyb->blk_ok();
        qsa->cparams = cparams;          // [v4bits]
        qsa->hparams = &model.hparams; // params.hparams is a copy that dies with the caller's llm_graph_params

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        ggml_set_input(qsa->bias);

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // gathers per stream: blk_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                        members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    cb(pooled, "indexer_k_pooled", il);

    // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks*n_stream, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
    pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream));
    score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
    score = ggml_relu(ctx0, score);

    // the heads sit side by side on ne[1] and there are only a few of them
    ggml_tensor * summed = nullptr;
    for (int64_t h = 0; h < n_idx_h; ++h) {
        ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream,
                score->nb[2], score->nb[3], h*score->nb[1]);
        summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
    }

    score = summed;
    cb(score, "indexer_score", il);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, inp->bias);
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, inp->bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    // Small batches (decode, speculative verify) on a long cache: gather the top_k cells'
    // K/V rows and mask values and attend over those only. This is the masked dense path
    // below restricted to its finite entries, so the result is the same up to summation
    // order, but the cost no longer grows with the cache: each token becomes its own
    // stream (ne[3]) holding its own gathered cells, padded to a FATTN_KQ_STRIDE multiple
    // with -inf mask entries so the vector kernels stay eligible.
    {
        const int64_t width     = top_k->ne[0];
        const int64_t n_tok     = top_k->ne[1];
        const int64_t width_pad = GGML_PAD(width, 256);

        ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);

        static const bool force_dense = [] {
            const char * env = getenv("LLAMA_QSA_DENSE");
            return env && atoi(env) != 0;
        }();

        const bool v_trans = v_all->nb[1] > v_all->nb[2];
        const bool sparse  = !force_dense && cparams.flash_attn && !v_trans &&
            top_k->ne[3] == 1 && k_all->ne[3] == 1 && n_tok <= 16 &&
            kq_mask->ne[0] == k_all->ne[2] && kq_mask->ne[1] == n_tok && kq_mask->ne[3] == 1 &&
            k_all->ne[2] >= 2*width_pad;

        if (sparse) {
            const int64_t n_embd_k = k_all->ne[0]*k_all->ne[1];
            const int64_t n_embd_v = v_all->ne[0]*v_all->ne[1];

            // [width*n_tok] cell ids, token-major, for gathering whole K/V rows
            ggml_tensor * ids = ggml_reshape_1d(ctx0, ggml_is_contiguous(top_k) ? top_k : ggml_cont(ctx0, top_k), width*n_tok);

            auto gather = [&](ggml_tensor * cache, int64_t n_embd) {
                ggml_tensor * rows = ggml_view_2d(ctx0, cache, n_embd, cache->ne[2], cache->nb[2], 0);
                rows = ggml_get_rows(ctx0, rows, ids);                        // F32 [n_embd, width*n_tok]
                rows = ggml_reshape_3d(ctx0, rows, n_embd, width, n_tok);
                rows = ggml_pad(ctx0, rows, 0, width_pad - width, 0, 0);       // zero rows, masked below
                return ggml_reshape_4d(ctx0, rows, cache->ne[0], cache->ne[1], width_pad, n_tok);
            };
            ggml_tensor * k = gather(k_all, n_embd_k);
            ggml_tensor * v = gather(v_all, n_embd_v);

            // per-token mask values at the selected cells: view the mask as one-element rows
            ggml_tensor * m = ggml_view_3d(ctx0, kq_mask, 1, kq_mask->ne[0], n_tok,
                    ggml_element_size(kq_mask), kq_mask->nb[1], 0);
            m = ggml_get_rows(ctx0, m, ggml_is_contiguous(top_k) ? ggml_reshape_2d(ctx0, top_k, width, n_tok)
                                                                 : ggml_cont(ctx0, ggml_reshape_2d(ctx0, top_k, width, n_tok)));
            m = ggml_reshape_4d(ctx0, m, width, 1, 1, n_tok);
            if (width_pad > width) {
                ggml_tensor * pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, width_pad - width, 1, 1, n_tok);
                m = ggml_concat(ctx0, m, ggml_fill(ctx0, pad, -INFINITY), 0);
            }
            m = ggml_cast(ctx0, m, GGML_TYPE_F16);

            // build_attn_mha splits q into k->ne[3] == n_tok streams: one query per stream
            ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, m, nullptr, nullptr, 0, kq_scale, il);
            cb(cur, "kqv_out", il);

            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }

            return cur;
        }
    }

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

// Attend each query to its selected blocks plus its tail block. With cell == position a block
// is one row of a [ratio cells] view of the K/V cache, so whole blocks are gathered at once.
ggml_tensor * llm_build_qsa_base::build_attn_qsa_blk(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_qsa_blk * blk_inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_blk,
        float                     kq_scale,
        int                       il) {
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    const int64_t r      = blk_inp->ratio;
    const int64_t n_tok  = top_blk->ne[1];
    const int64_t n_wblk = top_blk->ne[0] + blk_inp->tail_ids->ne[0];

    // per query: its selected blocks, then its tail block (repeated as padding)
    ggml_tensor * ids = ggml_concat(ctx0, top_blk, blk_inp->tail_ids, 0);
    static const int64_t masked_prefill_max = [] {
        const char * env = getenv("LLAMA_QSA27B_MASKED_PREFILL_MAX");
        return env ? std::max<int64_t>(0, atoll(env)) : int64_t(65536);
    }();
    if (model.arch == LLM_ARCH_QWEN35 && n_tok > 16 && inp->mctx->get_n_kv() <= masked_prefill_max) {
        const int64_t nkv = inp->mctx->get_n_kv();
        auto * offsets = ggml_reshape_3d(ctx0, ggml_arange(ctx0, 0, float(r), 1), r, 1, 1);
        auto * starts = ggml_scale(ctx0, ggml_cast(ctx0, ids, GGML_TYPE_F32), float(r));
        starts = ggml_reshape_3d(ctx0, starts, 1, n_wblk, n_tok);
        starts = ggml_repeat_4d(ctx0, starts, r, n_wblk, n_tok, 1);
        auto * cells = ggml_cast(ctx0, ggml_add(ctx0, starts, offsets), GGML_TYPE_I32);
        cells = ggml_reshape_3d(ctx0, cells, n_wblk*r, n_tok, 1);
        auto * mask = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, nkv, n_tok), -INFINITY);
        auto * zero = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_wblk*r, n_tok), 0.0f);
        mask = ggml_reshape_2d(ctx0, ggml_set_rows(ctx0, mask, zero, cells), nkv, n_tok);
        auto * causal = ggml_cast(ctx0, inp->get_kq_mask(), GGML_TYPE_F32);
        causal = ggml_view_2d(ctx0, causal, nkv, n_tok, causal->nb[1], 0);
        mask = ggml_add(ctx0, mask, causal);
        if (inp->get_kq_mask()->ne[1] > n_tok) {
            mask = ggml_pad(ctx0, mask, 0, inp->get_kq_mask()->ne[1] - n_tok, 0, 0);
        }
        mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
        auto * cur = build_attn_mha(q_cur, inp->mctx->get_k(ctx0, il), inp->mctx->get_v(ctx0, il),
                nullptr, mask, nullptr, nullptr, 0, kq_scale, il);
        if (inp->self_v_rot) { cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot); }
        return cur;
    }

    ggml_tensor * selected_mask = blk_inp->sel_mask;
    if (model.arch == LLM_ARCH_QWEN35) {
        auto * own = ggml_view_2d(ctx0, blk_inp->tail_ids, 1, n_tok, blk_inp->tail_ids->nb[1], 0);
        auto * delta = ggml_abs(ctx0, ggml_sub(ctx0, ggml_cast(ctx0, top_blk, GGML_TYPE_F32), ggml_cast(ctx0, own, GGML_TYPE_F32)));
        auto * unique = ggml_step(ctx0, ggml_scale_bias(ctx0, delta, 1.0f, -0.5f));
        auto * penalty = ggml_scale_bias(ctx0, unique, 1e30f, -1e30f);
        penalty = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, penalty, 1, top_blk->ne[0], n_tok), r, top_blk->ne[0], n_tok, 1);
        penalty = ggml_reshape_4d(ctx0, penalty, top_blk->ne[0]*r, 1, 1, n_tok);
        penalty = ggml_pad(ctx0, penalty, blk_inp->tail_ids->ne[0]*r, 0, 0, 0);
        selected_mask = ggml_add(ctx0, selected_mask, ggml_cast(ctx0, penalty, GGML_TYPE_F16));
    }


    {
        ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);
        // decode-sized batches keep the gather + flash attention path (as fast there); prompt batches,
        // which the gather path cannot take, use the sparse kernel
        // (q and the cached K/V are already Hadamard-rotated alike when the cache rotates them)
        const bool sparse_ok = qwen4exp_qsa_sparse_kernel_enabled() &&
            ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            (n_tok > 16 || model.arch == LLM_ARCH_QWEN35) &&
            (k_all->type == GGML_TYPE_Q8_0 || k_all->type == GGML_TYPE_F16) && v_all->type == k_all->type && k_all->ne[0] == 256 && v_all->ne[0] == 256 &&
            k_all->ne[3] == 1 && v_all->nb[1] <= v_all->nb[2] && r == 4 && q_cur->type == GGML_TYPE_F32;
        if (sparse_ok) {
            // block-sparse kernel: reads each query's blocks straight from the Q8_0 cache
            ggml_tensor * q = ggml_is_contiguous(q_cur) ? q_cur : ggml_cont(ctx0, q_cur);
            q = ggml_reshape_3d(ctx0, q, q->ne[0], q->ne[1], n_tok);
            ggml_tensor * k3 = ggml_view_3d(ctx0, k_all, k_all->ne[0], k_all->ne[1], k_all->ne[2], k_all->nb[1], k_all->nb[2], 0);
            ggml_tensor * v3 = ggml_view_3d(ctx0, v_all, v_all->ne[0], v_all->ne[1], v_all->ne[2], v_all->nb[1], v_all->nb[2], 0);
            ggml_tensor * out = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 256, q->ne[1], n_tok);
            // FLASH_ATTN_EXT params: scale, max_bias, softcap, prec; [6] the block size
            const float   fparams[3] = { kq_scale, 0.0f, 0.0f };
            const int32_t prec       = GGML_PREC_F32;
            const int32_t ratio_i32  = (int32_t) r;
            memcpy((char *) out->op_params + 0*sizeof(int32_t), fparams, sizeof(fparams));
            memcpy((char *) out->op_params + 3*sizeof(int32_t), &prec, sizeof(prec));
            memcpy((char *) out->op_params + 6*sizeof(int32_t), &ratio_i32, sizeof(ratio_i32));
            out->op     = GGML_OP_FLASH_ATTN_EXT;
            out->src[0] = q;
            out->src[1] = k3;
            out->src[2] = v3;
            out->src[3] = selected_mask;
            out->src[5] = ids;
            cb(out, "kqv_out", il);
            ggml_tensor * cur = ggml_reshape_2d(ctx0, out, 256*q->ne[1], n_tok);
            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }
            return cur;
        }
    }

    ids = ggml_reshape_1d(ctx0, ids, n_wblk*n_tok);

    // one gather per cache, straight to the F16 that flash attention consumes
    auto gather = [&](ggml_tensor * cache) {
        GGML_ASSERT(cache->ne[3] == 1);
        const int64_t n_embd = cache->ne[0]*cache->ne[1];
        ggml_tensor * rows = ggml_view_2d(ctx0, cache, n_embd*r, cache->ne[2]/r, r*cache->nb[2], 0);

        ggml_tensor * f32 = ggml_get_rows(ctx0, rows, ids);
        ggml_tensor * g = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, f32->ne[0], f32->ne[1], f32->ne[2], f32->ne[3]);
        g->op     = GGML_OP_GET_ROWS;
        g->src[0] = rows;
        g->src[1] = ids;

        return ggml_reshape_4d(ctx0, g, cache->ne[0], cache->ne[1], n_wblk*r, n_tok);
    };

    ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);
    GGML_ASSERT(v_all->nb[1] <= v_all->nb[2] && "block attention needs a non-transposed V cache");

    ggml_tensor * k = gather(mctx_cur->get_k(ctx0, il));
    ggml_tensor * v = gather(v_all);
    ggml_tensor * m = selected_mask;

    // build_attn_mha splits q into k->ne[3] == n_tok streams: one query per stream
    ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, m, nullptr, nullptr, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    // with the pooled block cache valid, every graph keeps it current; small decode batches
    // also select and attend through it at a cost independent of the context length
    const uint32_t ratio = qsa ? hparams.dsv4_compress_ratios[il] : 0;
    const bool     rsv   = qsa && qwen4exp_qsa_reserve_blk(mctx_hyb, ubatch, cparams, hparams); // [qsa-reserve-blk]
    const bool     blk   = qsa && (rsv || mctx_hyb->blk_ok());
    const bool     fast  = blk && (rsv || qwen4exp_qsa_blk_fast(mctx_hyb, ubatch, cparams, hparams, ratio));

    llm_graph_input_qsa_blk * blk_inp = blk ? get_qsa_blk_input(mctx_hyb, ratio, fast) : nullptr;

    ggml_tensor * top_k   = nullptr;
    ggml_tensor * top_blk = nullptr;

    if (fast) {
        top_blk = build_qsa_blk_select(mctx_hyb, blk_inp, cur, inp_pos, sections, il);
    } else if (qsa) {
        top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il);
        if (blk) {
            build_qsa_blk_update(mctx_hyb, blk_inp, sections, il);
        }
    }

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_blk) {
        cur = build_attn_qsa_blk(inp, blk_inp, Qcur, Kcur, Vcur, top_blk, kq_scale, il);
    } else if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

// [mtp-kvonly] the indexer cache's k_idxs alone, for a graph that stores raw indexer keys without scoring them
// (llm_graph_input_qsa also fills the scoring inputs, which such a graph leaves unallocated). Like llm_graph_input_qsa it
// refuses reuse when the pooled block cache comes or goes (the graph then gains or loses the block update).
class qwen4exp_input_idx_k_idxs : public llm_graph_input_i {
public:
    qwen4exp_input_idx_k_idxs(const llama_memory_hybrid_idx_context * mctx, bool had_blk) : mctx(mctx), had_blk(had_blk) {}
    virtual ~qwen4exp_input_idx_k_idxs() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        return mctx->get_idx() != nullptr && mctx->blk_ok() == had_blk && k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    ggml_tensor * k_idxs = nullptr; // I32 [n_tokens]

    const llama_memory_hybrid_idx_context * mctx;
    const bool had_blk;
};

// [mtp-kvonly] the cache writes of build_layer_attn for the same ubatch, in the same order, without the selection, the
// attention and the output projection: what a later draft reads from this block's caches is the same either way.
void llama_model_qwen4exp::graph::build_layer_kv_only(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // the path decision of build_layer_attn
    const bool     qsa   = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;
    const uint32_t ratio = qsa ? hparams.dsv4_compress_ratios[il] : 0;
    const bool     rsv   = qsa && qwen4exp_qsa_reserve_blk(mctx_hyb, ubatch, cparams, hparams); // [qsa-reserve-blk]
    const bool     blk   = qsa && (rsv || mctx_hyb->blk_ok());
    const bool     fast  = blk && (rsv || qwen4exp_qsa_blk_fast(mctx_hyb, ubatch, cparams, hparams, ratio));

    if (qsa) {
        const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
        llm_graph_input_qsa_blk * blk_inp = blk ? get_qsa_blk_input(mctx_hyb, ratio, fast) : nullptr;

        ggml_tensor * k_idxs = nullptr;
        if (fast) {
            k_idxs = blk_inp->k_idxs;
        } else {
            auto kin = std::make_unique<qwen4exp_input_idx_k_idxs>(mctx_hyb, blk);
            kin->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);
            k_idxs = kin->k_idxs;
            res->add_input(std::move(kin));
        }

        // raw keys, as build_qsa_top_k / build_qsa_blk_select cache them (pooling precedes norm and rotation)
        ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
        k_raw = ggml_reshape_3d(ctx0, k_raw, hparams.indexer_head_size, 1, n_tokens);
        cb(k_raw, "indexer_k_raw", il);
        ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, k_idxs, il));

        if (blk) {
            build_qsa_blk_update(mctx_hyb, blk_inp, sections, il);
        }
    }

    // K and V exactly as build_layer_attn makes them (no Q)
    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // the stores of build_attn / build_attn_qsa / build_attn_qsa_blk
    if (inp->self_k_rot) {
        Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp->self_k_rot);
    }
    if (inp->self_v_rot) {
        Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp->self_v_rot);
    }

    ggml_build_forward_expand(gf, Vcur);
    ggml_build_forward_expand(gf, Kcur);

    const auto * mctx_cur = inp->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp->get_v_idxs(), il));
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * start = ft_moe ? ft_moe->begin(ctx0,moe_out) : nullptr;
        if (start) { ggml_build_forward_expand(gf,start); }
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        if (start) { moe_out = ft_moe->finish(ctx0,start,ffn_shexp); }
        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_qwen4exp_ple : public llm_graph_input_i {
public:
    llm_graph_input_qwen4exp_ple(const llama_model & model,
                        const llama_kv_cache_context * mctx) : model(model), mctx(mctx) {}
    virtual ~llm_graph_input_qwen4exp_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) model.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model & model;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_qwen4exp_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hparams = model.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hparams.ple_image_token_id != 0
        ? (llama_token) hparams.ple_image_token_id
        : (llama_token) hparams.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hparams.ple_ngram_size;
    const int64_t n_heads  = hparams.ple_n_heads;
    const int64_t per_gram = hparams.ple_heads_per_ngram;
    const int64_t eos      = hparams.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hparams.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hparams.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hparams.ple_head_vocab_sizes[h_i] + hparams.ple_head_offsets[h_i]);
            }
        }
    }

    {
        ggml_tensor * ple = model.per_layer_tok_embd;

        const bool prefetch = model.can_prefetch.count(ple);
        if (prefetch) {
            llama_prefetch_rows(ple, idx.data(), idx.size());
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        // one strided copy per slot; a ggml_cont first would add a second full copy
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_qwen4exp_ple>(model, mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
