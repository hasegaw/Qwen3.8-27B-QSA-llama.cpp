#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-moe-cache.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <mutex>  // [mtp-async]
#include <thread> // [mtp-async]
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_mtp_kv_only(bool value); // [mtp-kvonly]
    void set_mtp_input_embeddings(const float * data, uint32_t n_tokens);
    void set_pp_plan_hint(uint32_t n_tokens); // [pp-plan-hint]
    void set_embeddings_nextn_lag(bool value); // [mtp-lag]
    void    decode_kv_only_async(const llama_batch & batch); // [mtp-async]
    int32_t async_join(); // [mtp-async] waits for a pending decode_kv_only_async, returns its status
    bool    async_rm_is_noop(llama_seq_id seq_id, llama_pos p0, llama_pos p1); // [mtp-async-rmskip]
    uint64_t      get_embeddings_nextn_call() const; // [mtp-lag]
    const float * get_embeddings_nextn_prev(uint64_t call_id); // [mtp-lag]
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_qsa_enabled(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    // [la-s0] stage-0 lookahead (llama-ext.h)
    int  decode_lookahead(const llama_batch & batch_inp);
    int  decode_resume(const llama_batch & batch_inp);
    void decode_discard();
    bool lookahead_pending() const { return la_pending; }

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // LLAMA_GPU_ARGMAX: logits rows whose device-to-host copy is deferred until someone reads them
    struct logits_pending_copy {
        ggml_tensor *  t;
        ggml_backend_t backend;
        float *        dst;
        size_t         nbytes;
    };
    std::vector<logits_pending_copy> logits_pending;
    std::vector<int32_t>             argmax_ids; // per output row, -1 when not computed
    // [draft-ahead] LLAMA_DRAFT_AHEAD>0 or LLAMA_ARGMAX_PIN=1: the argmax rows of a decode are copied into pinned memory and moved
    // into argmax_ids at synchronize(), so llama_decode returns before the GPUs finish (a pageable copy waits for them)
    ggml_backend_buffer_ptr buf_argmax_pin;
    int32_t * argmax_pin     = nullptr;
    int64_t   argmax_pin_cap = 0;
    int64_t   argmax_pin_n   = 0; // rows of the last decode not yet moved into argmax_ids

    // [la-s0] stage-0 lookahead: la_mode 1 inside decode_lookahead, 2 inside decode_resume; la_pending = a stage 0 waits for resume/discard
    int            la_mode    = 0;
    bool           la_pending = false;
    int            la_split   = -1;      // first split of the second GPU
    llama_pos      la_pos0    = -1;
    llama_seq_id   la_seq     = 0;
    ggml_backend_t la_be0     = nullptr; // GPU0's backend (not waited for while la_pending)
    bool           la_restore_inflight = false; // [la-sync] a discard queued g_restore on GPU0; the next synchronize() waits for it
    std::unique_ptr<llama_batch_allocr> balloc_la;
    llama_memory_context_ptr            mctx_la;
    void logits_flush(bool sync);

    // [gpu-topk] device top-K of every output row (LLAMA_GPU_TOPK): ids then values in one pinned host buffer, rows of topk_k entries
    ggml_backend_buffer_ptr buf_topk;
    int32_t *            topk_ids  = nullptr;
    float *              topk_vals = nullptr;
    int64_t              topk_k    = 0;
    int64_t              topk_cap  = 0; // rows
    std::vector<uint8_t> topk_ok;       // per output row: 1 when its top-K was copied by the current decode
    bool topk_reserve(int64_t n_rows);

public:
    // greedy token of output i computed on the device, or LLAMA_TOKEN_NULL
    llama_token get_argmax_ith(int32_t i);

    // [gpu-topk] K and the device top-K (ids, raw logits; unordered) of output i, or 0 when not available; no logits copy
    int32_t get_topk_ith(int32_t i, const llama_token ** ids, const float ** vals);

private:

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);
    ggml_status graph_compute_la(ggml_cgraph * gf); // [la-s0] the splits of the current stage

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    // GPU-resident cache for host-offloaded MoE experts (null when --moe-expert-cache is 0)
    std::unique_ptr<llama_moe_cache> moe_cache;
    std::unique_ptr<class llama_ft_moe> ft_moe;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // false while work enqueued on the backends may still be running (see synchronize())
    bool synced = true;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // [reserve-heal] env: LLAMA_RESERVE_HEAL=1 (custom tree; see patch-mt-reserveheal.py)
    bool reserve_heal      = false;
    bool reserve_trace     = false; // [reserve-trace] env: LLAMA_RESERVE_TRACE=1
    int  trace_nf_seen     = 0;     // [reserve-trace2] per context: each context has its own scheduler
    bool heal_pending      = false;
    int  heal_nf_prev      = 0;   // scheduler fallback re-reserves seen so far
    int  heal_last_nodes   = -1;  // node count of the last graph that fell back
    int  heal_done_nodes   = -1;  // node count the last heal was for
    int  n_heal            = 0;
    std::vector<int> heal_failed; // node counts a heal did not help
    void heal_reserve(const llama_ubatch & ubatch, llm_graph_type gtype);

    // [pp-plan-ahead] env: LLAMA_PP_PLAN_AHEAD=1 (custom tree; see patch-mt-planahead.py). With pipeline parallelism, reserve the
    // scheduler once per decode call with the n_kv of the call's last ubatch, so the later ubatches do not re-reserve
    bool     pp_plan_ahead     = false;
    uint32_t pp_plan_remaining = 0;     // tokens of this decode call after the ubatch being processed (0 outside decode)
    bool     pp_plan_done      = false; // already planned in this decode call
    int      pp_plan_count     = 0;     // plans made in this decode call (at most 4)
    uint32_t pp_plan_max_kv    = 32768; // env LLAMA_PP_PLAN_AHEAD_MAX_KV, 0 = no cap; see patch-mt-planahead.py
    int      n_plan            = 0;
    // 1 = reserved with a plan graph, 0 = no graph of the real graph's node count was found, -1 = reserve failed
    int plan_reserve(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, int target_nodes);

    // [pp-plan-hint] env LLAMA_PP_PLAN_HINT=1 (see patch-mt-mtplag.py): the server passes the prompt tokens left after the batch, so the
    // first plan covers the prompt's end and later calls skip the plan (a reserve synchronizes all GPUs) while it still covers them
    bool     pp_plan_hint_on   = false;
    uint32_t pp_plan_hint      = 0; // set before a decode call, cleared at its end
    int64_t  pp_plan_cover_pos = 0; // cells (last pos + 1 + planned tokens) the last successful plan covered, 0 = none
    int64_t  pp_plan_cur_end   = 0; // last pos + 1 of the ubatch being processed
    int64_t  pp_plan_dense_end = 0; // [pp-plan-hint-dense] v2: a prompt ubatch whose first pos + 1 is below this takes the dense QSA path
    uint32_t pp_plan_extra_limit = UINT32_MAX; // [pp-plan-hint-dense] v2: cap on the planned tokens for the plan being made

    // [mtp-lag] unmasked nextn rows of consecutive decode calls alternate between the buf_output view (0) and nextn_alt_buf (1);
    // each call records an event after its last row copy (see llama_get_embeddings_nextn_prev)
    bool                    nextn_lag     = false;

    // [mtp-async] the worker of decode_kv_only_async (see patch-mt-mtpasync.py)
    std::thread async_th;
    std::mutex  async_mtx;
    int32_t     async_rc = 0;
    // [mtp-async-rmskip] what the running catch-up can touch: one sequence, positions up to async_hi (incl. the memory's end before it)
    llama_seq_id async_seq = -1;
    llama_pos    async_hi  = -1;
    int                     nextn_flip    = 0;
    ggml_backend_buffer_ptr nextn_alt_buf;
    float *                 nextn_ptr[2]  = { nullptr, nullptr };
    uint64_t                nextn_call[2] = { 0, 0 }; // call id whose rows each buffer holds, 0 = none
    uint64_t                nextn_calls   = 0;
    ggml_backend_event_t    nextn_ev[2]   = { nullptr, nullptr };
    ggml_backend_t          nextn_backend = nullptr; // backend of the current call's last row copy

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
