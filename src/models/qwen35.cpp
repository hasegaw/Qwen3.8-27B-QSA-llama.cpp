#include "models.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-impl.h"

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head, false);
    if (hparams.indexer_n_head > 0) {
        ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K, hparams.indexer_top_k);
        ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, true);
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
            if (hparams.indexer_head_size > 0) {
                const int64_t d = hparams.indexer_head_size;
                layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), {n_embd, d*hparams.indexer_n_head}, flags);
                layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), {n_embd, d}, flags);
                layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), {d}, flags);
                layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), {d}, flags);
                layer.index_k_pre_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "pre_weight", il), {d}, flags);
            }
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_qsa_base(model, params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
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

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
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

    // Apply MRoPE
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

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    auto * original_mask = inp->self_kq_mask_cnv;
    ggml_tensor * top_blocks = nullptr;
    llm_graph_input_qsa_blk * block_input = nullptr;
    if (hparams.indexer_head_size > 0 && cparams.qsa_enabled && !getenv("LLAMA_QSA_DENSE")) {
        const auto * hybrid = static_cast<const llama_memory_hybrid_idx_context *>(mctx);
        const auto ratio = hparams.dsv4_compress_ratios[il];
        const bool cached = hybrid->get_blk() && (hybrid->blk_ok() || !hybrid->get_attn()->has_ubatch());
        const bool fast = cached && qsa_block_fast(hybrid, ratio);
        if (fast) {
            block_input = get_qsa_blk_input(hybrid, ratio, true);
            top_blocks = build_qsa_blk_select(hybrid, block_input, cur, inp_pos, sections, il);
        } else {
            inp->self_kq_mask_cnv = build_qsa_mask(inp, cur, inp_pos, sections, il);
            if (cached) { build_qsa_blk_update(hybrid, get_qsa_blk_input(hybrid, ratio, false), sections, il); }
        }
    }
    if (top_blocks) {
        cur = build_attn_qsa_blk(inp, block_input, Qcur, Kcur, Vcur, top_blocks, kq_scale, il);
    } else {
        cur = build_attn(inp, nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    inp->self_kq_mask_cnv = original_mask;
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
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
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

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

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
// Build only the selected query's mask, using the cache's existing visibility
// rules (including holes, sequence membership, position offsets and M-RoPE).
// Every padded token still writes its key/value through the ordinary KV input.
class qwen35_mtp_dense_mask : public llm_graph_input_i {
public:
    const llama_kv_cache_context * mctx;
    ggml_tensor * mask;
    const bool causal;

    qwen35_mtp_dense_mask(const llama_kv_cache_context * kv, ggml_tensor * mask, bool causal) :
        mctx(kv), mask(mask), causal(causal) {}

    void set_input(const llama_ubatch * ubatch) override {
        uint32_t selected = 0, count = 0;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (ubatch->output[i]) { selected = i; ++count; }
        }
        GGML_ASSERT(count == 1);
        llama_ubatch one = *ubatch;
        std::vector<llama_pos> positions(ubatch->n_pos);
        for (uint32_t axis = 0; axis < ubatch->n_pos; ++axis) {
            positions[axis] = ubatch->pos[axis*ubatch->n_tokens + selected];
        }
        one.n_tokens = one.n_seq_tokens = one.n_seqs = one.n_seqs_unq = 1;
        one.pos = positions.data();
        one.n_seq_id = ubatch->n_seq_id + selected;
        one.seq_id = ubatch->seq_id + selected;
        mctx->set_input_kq_mask(mask, &one, causal);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_context *>(params.mctx);
        return params.n_outputs == 1 && params.ubatch.n_seqs_unq == 1 && mask->ne[0] == mctx->get_n_kv();
    }
};

llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);
    inp->input_embeddings = cparams.mtp_input_embeddings;

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = cparams.mtp_kv_only ? nullptr : build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    if (cparams.mtp_kv_only) {
        ggml_tensor * key;
        ggml_tensor * value;
        if (layer.wk && layer.wv) {
            key = build_lora_mm(layer.wk, cur, layer.wk_s);
            value = build_lora_mm(layer.wv, cur, layer.wv_s);
        } else {
            auto qkv = build_qkv(layer, cur, n_embd_head*2, n_head,
                    n_embd_head, n_head_kv, n_embd_head, n_head_kv, il, false);
            key = qkv.k;
            value = qkv.v;
        }
        key = ggml_reshape_3d(ctx0, key, n_embd_head, n_head_kv, n_tokens);
        key = build_norm(key, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        key = ggml_rope_multi(ctx0, key, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        value = ggml_reshape_3d(ctx0, value, n_embd_head, n_head_kv, n_tokens);
        if (inp_attn->self_k_rot) { key = llama_mul_mat_hadamard(ctx0, key, inp_attn->self_k_rot); }
        if (inp_attn->self_v_rot) { value = llama_mul_mat_hadamard(ctx0, value, inp_attn->self_v_rot); }
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, key, inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, value, inp_attn->get_v_idxs(), il));
        return;
    }

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // The MTP checkpoint has no trained indexer: retain every key. For a
    // masked draft output only one query is needed, although all padded rows
    // must still write their K/V. The block kernel can also read an exhaustive
    // list of blocks, sharing each KV head across its query heads.
    static const int64_t dense_block_min = [] {
        const char * env = getenv("LLAMA_MTP_DENSE_BLOCK_MIN");
        return env ? atoll(env) : int64_t(65536);
    }();
    const auto * kv = inp_attn->mctx;
    const bool selected_dense = dense_block_min >= 0 && kv->get_n_kv() >= dense_block_min &&
        cparams.embeddings_nextn_masked && inp_out_ids && inp_out_ids->ne[0] == 1 && n_tokens <= 4 &&
        ubatch.n_seqs_unq == 1 && inp_attn->get_kq_mask()->ne[3] == 1 &&
        cparams.flash_attn && hparams.indexer_n_head > 0 && n_embd_head == 256 && n_head == 24 && n_head_kv == 4 &&
        kv->type_k() == GGML_TYPE_F16 && kv->type_v() == GGML_TYPE_F16 &&
        !inp_attn->self_k_rot && !inp_attn->self_v_rot &&
        ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (selected_dense) {
        ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        auto * key = kv->get_k(ctx0, il);
        auto * value = kv->get_v(ctx0, il);
        GGML_ASSERT(key->ne[3] == 1 && value->nb[1] <= value->nb[2] && key->ne[2] % 4 == 0);
        Qcur = ggml_get_rows(ctx0, ggml_cont_2d(ctx0, Qcur, n_embd_head*n_head, n_tokens), inp_out_ids);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, 1);
        gate = ggml_get_rows(ctx0, gate, inp_out_ids);
        auto * mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, key->ne[2], 1, 1, 1);
        ggml_set_input(mask);
        res->add_input(std::make_unique<qwen35_mtp_dense_mask>(kv, mask, cparams.causal_attn));
        auto * ids = ggml_cast(ctx0, ggml_arange(ctx0, 0, float(key->ne[2]/4), 1), GGML_TYPE_I32);
        auto * out = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, n_head, 1);
        const float fparams[3] = {kq_scale, 0.0f, 0.0f};
        const int32_t prec = GGML_PREC_F32, ratio = 4;
        memcpy(out->op_params, fparams, sizeof(fparams));
        memcpy((char *) out->op_params + 3*sizeof(int32_t), &prec, sizeof(prec));
        memcpy((char *) out->op_params + 6*sizeof(int32_t), &ratio, sizeof(ratio));
        out->op = GGML_OP_FLASH_ATTN_EXT;
        out->src[0] = Qcur;
        out->src[1] = ggml_view_3d(ctx0, key, key->ne[0], key->ne[1], key->ne[2], key->nb[1], key->nb[2], 0);
        out->src[2] = ggml_view_3d(ctx0, value, value->ne[0], value->ne[1], value->ne[2], value->nb[1], value->nb[2], 0);
        out->src[3] = mask;
        out->src[5] = ids;
        cb(out, "mtp_dense_blocks", il);
        cur = ggml_reshape_2d(ctx0, out, n_embd_head*n_head, 1);
    } else {
        cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    if (inp_out_ids && cparams.embeddings_nextn_masked) {
        if (!selected_dense) { cur = ggml_get_rows(ctx0, cur, inp_out_ids); }
        inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    }
    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

class qsa27b_input : public llm_graph_input_i {
public:
    const llama_memory_hybrid_idx_context * mctx;
    int64_t nkv, nq, ratio;
    ggml_tensor * key_ids = nullptr;
    ggml_tensor * token_blocks = nullptr;
    ggml_tensor * block_pos = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * query_block = nullptr;

    qsa27b_input(const llama_memory_hybrid_idx_context * m, int64_t k, int64_t q, int64_t r) : mctx(m), nkv(k), nq(q), ratio(r) {}

    void set_input(const llama_ubatch * ub) override {
        mctx->get_idx()->set_input_k_idxs(key_ids, ub);
        const int64_t nb = nkv / ratio;
        if (!nb || !token_blocks->buffer) { return; }
        const int64_t offset = mctx->get_idx()->ubatch_cell0();
        std::vector<int32_t> ids(nkv), positions(4*nb);
        std::vector<float> visible(nb*nq, -INFINITY), current(nq);
        for (int64_t i = 0; i < nkv; ++i) { ids[i] = std::min(i / ratio, nb - 1); }
        for (int64_t b = 0; b < nb; ++b) {
            int32_t rope[4] = {0, 0, 0, 0};
            mctx->get_idx()->cell_rope_pos(b*ratio, rope);
            for (int axis = 0; axis < 4; ++axis) { positions[axis*nb + b] = rope[axis]; }
        }
        for (int64_t q = 0; q < nq; ++q) {
            current[q] = (offset + q) / ratio;
            for (int64_t b = 0; b < nb && (b + 1)*ratio - 1 <= offset + q; ++b) { visible[q*nb + b] = 0; }
        }
        ggml_backend_tensor_set(token_blocks, ids.data(), 0, ids.size()*sizeof(int32_t));
        ggml_backend_tensor_set(block_pos, positions.data(), 0, positions.size()*sizeof(int32_t));
        ggml_backend_tensor_set(bias, visible.data(), 0, visible.size()*sizeof(float));
        ggml_backend_tensor_set(query_block, current.data(), 0, current.size()*sizeof(float));
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        const bool fast = !getenv("LLAMA_QSA27B_REFERENCE") && params.cparams.flash_attn && mctx->blk_ok() &&
                (mctx->get_idx()->ubatch_cell0() + 1)/ratio > params.hparams.indexer_top_k/ratio;
        return !fast && params.cparams.qsa_enabled && params.ubatch.n_tokens == nq && mctx->get_idx()->get_n_kv() == nkv;
    }
};

ggml_tensor * llama_model_qwen35::graph::build_qsa_mask(llm_graph_input_attn_kv * attn, ggml_tensor * hidden, ggml_tensor * pos, int * sections, int il) {
    const auto * hybrid = static_cast<const llama_memory_hybrid_idx_context *>(mctx);
    const auto * idx = hybrid->get_idx();
    GGML_ASSERT(idx && hybrid->get_n_stream() == 1);
    const int64_t nkv = idx->get_n_kv();
    const int64_t r = hparams.dsv4_compress_ratios[il];
    const int64_t nb = nkv / r;
    const int64_t d = hparams.indexer_head_size;
    const int64_t heads = hparams.indexer_n_head;
    auto * input = static_cast<qsa27b_input *>(qsa_input);
    if (!input) {
        auto owned = std::make_unique<qsa27b_input>(hybrid, nkv, n_tokens, r);
        input = owned.get();
        input->key_ids = idx->build_input_k_idxs(ctx0, ubatch);
        if (nb) {
            input->token_blocks = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, nkv);
            input->block_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*nb);
            input->bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nb, n_tokens);
            input->query_block = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
            for (auto * t : {input->token_blocks, input->block_pos, input->bias, input->query_block}) { ggml_set_input(t); }
        }
        qsa_input = input;
        res->add_input(std::move(owned));
    }
    auto & layer = model.layers[il];
    auto * raw = build_lora_mm(layer.index_k_proj, hidden);
    raw = build_norm(raw, layer.index_k_pre_norm, nullptr, LLM_NORM_RMS, il);
    raw = ggml_reshape_3d(ctx0, raw, d, 1, n_tokens);
    ggml_build_forward_expand(gf, idx->cpy_k(ctx0, raw, input->key_ids, il));
    if (!nb || nkv <= hparams.indexer_top_k) { return attn->get_kq_mask(); }
    auto * cached = idx->get_k(ctx0, il);
    auto * members = ggml_view_3d(ctx0, cached, d, r, nb, cached->nb[2], r*cached->nb[2], 0);
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        auto * part = ggml_cont(ctx0, ggml_view_2d(ctx0, members, d, nb, members->nb[2], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, part) : part;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/r);
    pooled = build_norm(ggml_reshape_3d(ctx0, pooled, d, nb, 1), layer.index_k_norm, nullptr, LLM_NORM_RMS, il);
    pooled = ggml_rope_multi(ctx0, ggml_reshape_3d(ctx0, pooled, d, 1, nb), input->block_pos, nullptr,
                            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_2d(ctx0, pooled, d, nb);
    auto * query = build_lora_mm(layer.index_q_proj, hidden);
    query = build_norm(ggml_reshape_3d(ctx0, query, d, heads, n_tokens), layer.index_q_norm, nullptr, LLM_NORM_RMS, il);
    query = ggml_rope_multi(ctx0, query, pos, nullptr, n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                           ext_factor, attn_factor, beta_fast, beta_slow);
    auto * scores = ggml_relu(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, pooled, ggml_reshape_2d(ctx0, query, d, heads*n_tokens)), nb, heads, n_tokens));
    ggml_tensor * sum = nullptr;
    for (int64_t h = 0; h < heads; ++h) {
        auto * part = ggml_view_2d(ctx0, scores, nb, n_tokens, scores->nb[2], h*scores->nb[1]);
        sum = sum ? ggml_add(ctx0, sum, part) : ggml_cont(ctx0, part);
    }
    sum = ggml_add(ctx0, ggml_scale(ctx0, sum, 1.0f/sqrtf(float(d))), input->bias);
    const int64_t kb = std::min<int64_t>(nb, (hparams.indexer_top_k + r - 1)/r);
    auto * top = ggml_cont(ctx0, ggml_top_k(ctx0, sum, kb));
    cb(top, "qsa27b_top_blocks", il);
    auto * selected = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, nb, n_tokens), 0.0f);
    auto * ones = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, kb, n_tokens), 1.0f);
    selected = ggml_reshape_2d(ctx0, ggml_set_rows(ctx0, selected, ones, ggml_reshape_3d(ctx0, top, kb, n_tokens, 1)), nb, n_tokens);
    auto * block_ids = ggml_repeat_4d(ctx0, ggml_arange(ctx0, 0.0f, float(nb), 1.0f), nb, n_tokens, 1, 1);
    auto * own = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_abs(ctx0, ggml_sub(ctx0, block_ids, input->query_block)), 1.0f, -0.5f));
    own = ggml_scale_bias(ctx0, own, -1.0f, 1.0f);
    selected = ggml_clamp(ctx0, ggml_add(ctx0, selected, own), 0.0f, 1.0f);
    selected = ggml_scale_bias(ctx0, selected, 1e30f, -1e30f);
    auto * expanded = ggml_get_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, selected)), input->token_blocks);
    expanded = ggml_cont(ctx0, ggml_transpose(ctx0, expanded));
    auto * original = attn->get_kq_mask();
    auto * original_f32 = original->type == GGML_TYPE_F32 ? original : ggml_cast(ctx0, original, GGML_TYPE_F32);
    auto * causal = ggml_view_2d(ctx0, original_f32, nkv, n_tokens, original_f32->nb[1], 0);
    auto * mask = ggml_add(ctx0, expanded, causal);
    if (original->ne[1] > n_tokens) { mask = ggml_pad(ctx0, mask, 0, original->ne[1] - n_tokens, 0, 0); }
    return ggml_cast(ctx0, mask, original->type);
}
