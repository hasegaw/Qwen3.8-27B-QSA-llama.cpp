#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // Pooled indexer keys, one F32 row per block of `ratio` cells (row n_blocks is a
    // scratch row), kept up to date incrementally while blk_valid holds. blk_valid means a
    // single sequence written in order, every cell holding the token of the same rank (for
    // text that is the position of the same number), so a block's members are ratio
    // consecutive cells and a finished block's pooled key never changes. Anything that breaks
    // that (copies, shifts, restores, other placements) clears it until the cache is emptied
    // again.
    llama_kv_cache * get_mem_blk() const;   // nullptr when the model carries no indexer
    uint32_t get_blk_ratio() const { return blk_ratio; }
    bool     get_blk_valid() const { return blk_valid; }
    // [blk-rebuild] dropping validity also drops a pending rebuild: there is no valid layout to rebuild into
    void     set_blk_valid(bool v) const { blk_valid = v; if (!v) { blk_rebuild = false; blk_rebuild_tries = 0; } }   // [blk-rebuild2]

    // [blk-rebuild] LLAMA_QSA_BLK_REBUILD=1 only. A restored state whose cell layout passes blk_layout_ok() is marked valid again
    // with this flag set: the pooled rows are stale (they are not part of the saved state), so the next ubatch re-pools every
    // block from the restored raw indexer keys, and the flag is cleared once that ubatch went through
    bool     get_blk_rebuild() const { return blk_rebuild; }
    void     set_blk_rebuild(bool v) const { blk_rebuild = v; blk_rebuild_tries = 0; }   // [blk-rebuild2] the counter restarts with the flag

    // [blk-rebuild2] apply() calls that captured the pending rebuild since the last next() or invalidation. apply() runs once per
    // ubatch attempt, so finding this > 0 at the next apply() means the attempt that captured it never reached next(): it failed
    uint32_t get_blk_rebuild_tries() const { return blk_rebuild_tries; }
    void     set_blk_rebuild_tries(uint32_t n) const { blk_rebuild_tries = n; }

    // [blk-rebuild2] "target" for a memory with recurrent layers, "draft" for the attention-only MTP draft; for log lines
    const char * get_blk_who() const { return has_recurrent ? "target" : "draft"; }

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // [blk-rebuild] true if the cells of the attention and the indexer cache (stream 0) are the layout the pooled block keys assume:
    // cells 0..n-1 hold seq 0 only, in strictly increasing (pos, y, x) order, position 0 first
    // [blk-rebuild2] it does not promise one position per text cell: the check meant for that is inert here, see blk_layout_ok
    // on failure *why points to a static string naming the first condition that failed
    bool blk_layout_ok(const char ** why) const;

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;
    llama_hparams hparams_blk;
    bool has_recurrent = false;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    uint32_t blk_ratio = 0;
    mutable bool blk_valid = true;
    mutable bool blk_rebuild = false;   // [blk-rebuild] valid, but the pooled rows are stale until the next ubatch re-pools them
    mutable uint32_t blk_rebuild_tries = 0;   // [blk-rebuild2] see get_blk_rebuild_tries()
    std::unique_ptr<llama_kv_cache> mem_blk;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // the pooled block-key cache and whether the current ubatch may use / must update it
    llama_kv_cache * get_blk() const;
    bool blk_ok() const;

    // [blk-rebuild] the current ubatch must re-pool every block, not only the ones it touches (decided in apply())
    bool blk_rebuild() const { return blk_rebuild_cur; }

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // [blk-rebuild] set by apply() for the ubatch being processed, confirmed by next()
    bool blk_rebuild_cur = false;
};
