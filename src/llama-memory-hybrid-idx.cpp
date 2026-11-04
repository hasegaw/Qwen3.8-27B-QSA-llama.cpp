#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <cstdlib>

static bool qsa_blk_debug() {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QSA_BLK_DEBUG"); return e && atoi(e) != 0; }();
    return enabled;
}
// [blk-rebuild] LLAMA_QSA_BLK_REBUILD: after a state restore, rebuild the pooled block keys from the restored raw indexer keys
// instead of leaving the layout invalid. Off by default.
// [blk-rebuild2] unset or 0: off. 2: TEST ONLY negative control, the layout is marked valid again but the stale pooled keys are kept.
// Any other value: rebuild. Read on every call - only state_read asks, which is rare - so a test can switch modes within one process.
static int qsa_blk_rebuild_mode() {
    const char * e = getenv("LLAMA_QSA_BLK_REBUILD");
    return e ? atoi(e) : 0;
}
// [blk-rebuild] WARN, not INFO: libllama's INFO is hidden at the server's default verbosity (the line only prints with LLAMA_QSA_BLK_DEBUG=1)
#define QSA_BLK_INVALIDATE(why) do { if (blk_valid && qsa_blk_debug()) { LLAMA_LOG_WARN("qsa-blk: layout invalidated by %s\n", why); } blk_valid = false; blk_rebuild = false; blk_rebuild_tries = 0; } while (0)

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, model.arch == LLM_ARCH_QWEN35 ? GGML_TYPE_F32 : type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        has_recurrent |= model.hparams.is_recr(il) && (!filter_recr || filter_recr(il));
    }

    if (mem_idx) {
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            if (filter_idx(il)) {
                blk_ratio = std::max(blk_ratio, model.hparams.dsv4_compress_ratios[il]);
            }
        }
    }

    if (mem_idx && blk_ratio > 0 && n_seq_max == 1) {
        hparams_blk = hparams_idx;

        // one row per block plus a scratch row for padded updates
        const uint32_t n_pad_blk = std::max<uint32_t>(n_pad, 1);
        const uint32_t size_blk  = GGML_PAD(kv_size/blk_ratio + 1, n_pad_blk);

        LLAMA_LOG_INFO("%s: creating pooled indexer key cache, size = %u blocks\n", __func__, size_blk);

        mem_blk.reset(new llama_kv_cache(
            model, hparams_blk, GGML_TYPE_F32, GGML_TYPE_F32, false, offload, unified,
            size_blk, n_seq_max, n_pad_blk, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, filter_idx, nullptr, nullptr, "blk_"));
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    // an empty cache is trivially laid out by position; blocks are rebuilt as they fill
    blk_valid = true;
    blk_rebuild = false;   // [blk-rebuild] nothing left to rebuild
    blk_rebuild_tries = 0; // [blk-rebuild2]
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    // An attention-only draft has bookkeeping cells but no recurrent state to restore.
    if (!get_mem_recr()->seq_rm(seq_id, has_recurrent ? p0 : -1, has_recurrent ? p1 : -1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    const bool res = get_mem_attn()->seq_rm(seq_id, p0, p1);

    // removal keeps every remaining cell at its position; a later misplaced insert is caught
    // when its ubatch is applied. Emptying the cache makes the layout valid again.
    if (res && get_mem_attn()->seq_pos_max(seq_id < 0 ? 0 : seq_id) < 0 && (seq_id <= 0)) {
        if (!blk_valid && qsa_blk_debug()) {
            LLAMA_LOG_WARN("qsa-blk: layout valid again (cache emptied)\n");   // [blk-rebuild] INFO -> WARN
        }
        blk_valid = true;
        blk_rebuild = false;   // [blk-rebuild] nothing left to rebuild
        blk_rebuild_tries = 0; // [blk-rebuild2]
    }

    return res;
}

llama_pos llama_memory_hybrid_idx::seq_pos_min(llama_seq_id seq_id) const {
    return has_recurrent ? llama_memory_hybrid::seq_pos_min(seq_id) : get_mem_attn()->seq_pos_min(seq_id);
}

llama_pos llama_memory_hybrid_idx::seq_pos_max(llama_seq_id seq_id) const {
    return has_recurrent ? llama_memory_hybrid::seq_pos_max(seq_id) : get_mem_attn()->seq_pos_max(seq_id);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
    QSA_BLK_INVALIDATE("seq_cp");

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);
    QSA_BLK_INVALIDATE("seq_keep");

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);
    QSA_BLK_INVALIDATE("seq_add");

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
    QSA_BLK_INVALIDATE("seq_div");

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    if (mem_blk) {
        for (const auto & buft_size : mem_blk->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // pooled block keys are not part of the saved state; a partial (recurrent-only) restore,
    // as the server's context checkpoints use, leaves the attention and indexer cells alone
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        QSA_BLK_INVALIDATE("state_read");
    }

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    // [blk-rebuild] the pooled keys are not part of the saved state, but the raw indexer keys are. If the restored cells are laid out
    // the way the pooled cache assumes, mark it valid again and have the next ubatch re-pool every block from those keys.
    // Only reached when the restore went through (the catch above rethrows).
    // [blk-rebuild2] the mode is read per restore: 2 is the TEST ONLY negative control
    const int blk_mode = qsa_blk_rebuild_mode();
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0 && mem_blk && blk_mode != 0) {
        const char * why = nullptr;
        const char * who = has_recurrent ? "target" : "draft";

        if (blk_layout_ok(&why)) {
            blk_valid = true;
            blk_rebuild_tries = 0;

            if (blk_mode == 2) {
                // [blk-rebuild2] TEST ONLY: valid again, but blk_rebuild stays false (QSA_BLK_INVALIDATE above cleared it), so the next
                // ubatches select blocks by the stale pooled keys left in mem_blk. For checking that a test can tell a rebuilt cache from a stale one.
                LLAMA_LOG_WARN("qsa-blk: TEST ONLY: restored layout ok, stale pooled keys kept (%s)\n", who);
            } else {
                blk_rebuild = true;

                LLAMA_LOG_WARN("qsa-blk: restored layout ok (%s), pooled keys will be rebuilt on the next ubatch\n", who);
            }
        } else {
            LLAMA_LOG_WARN("qsa-blk: restored layout not rebuildable: %s (%s)\n", why ? why : "unknown", who);
        }
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

// [blk-rebuild] The layout the incremental pooled cache assumes, checked on the restored cells of both caches (stream 0): cell j holds
// the token of rank j of seq 0, so block b is cells b*ratio..; cells run in the (pos, y, x) order set_input_qsa ranks them by; the first
// position is 0. Check 7 is meant to also require a text cell to be one position after the previous one, a jump being allowed only next
// to a 2-D cell. [blk-rebuild2] It is inert for this model: IMROPE stores ext=(p,p) for text cells (the text position is broadcast to every
// section in llama-batch.cpp and apply_ubatch() keeps it as ext.x/ext.y), so is_2d() holds for every cell with p > 0 and no jump is ever
// rejected. What is guaranteed: cells 0..n-1 packed, seq 0 only, strictly increasing (pos, y, x), first position 0. Text positions may
// still jump (for example a draft context that skipped an image's embeddings).
bool llama_memory_hybrid_idx::blk_layout_ok(const char ** why) const {
    // check() returns 0 for ok, else an index into these
    static const char * const why_attn[] = {
        nullptr,
        "attention cache missing",
        "attention cache has more than one stream",
        "attention cells are not packed from cell 0",
        "attention cell belongs to another sequence",
        "attention cells are not in strictly increasing position order",
        "attention cache does not start at position 0",
        "attention cache has a position jump away from an image",
    };
    static const char * const why_idx[] = {
        nullptr,
        "indexer cache missing",
        "indexer cache has more than one stream",
        "indexer cells are not packed from cell 0",
        "indexer cell belongs to another sequence",
        "indexer cells are not in strictly increasing position order",
        "indexer cache does not start at position 0",
        "indexer cache has a position jump away from an image",
    };

    const auto check = [](const llama_kv_cache * kv) -> int {
        if (kv == nullptr) {
            return 1;
        }

        if (kv->get_n_stream() != 1) {
            return 2;
        }

        const auto & cells = kv->get_cells(0);

        // used count == used_max_p1 == n means the used cells are exactly 0..n-1
        const uint32_t n = cells.used_max_p1();
        if (n == 0 || cells.get_used() != n) {
            return 3;
        }

        for (uint32_t j = 0; j < n; ++j) {
            if (cells.seq_count(j) != 1 || !cells.seq_has(j, 0)) {
                return 4;
            }
        }

        if (cells.pos_get(0) != 0) {
            return 6;
        }

        const auto is_2d = [&cells](uint32_t j) {
            const auto & e = cells.ext_get(j);

            return e.x != 0 || e.y != 0;
        };

        for (uint32_t j = 1; j < n; ++j) {
            const llama_pos pa = cells.pos_get(j - 1);
            const llama_pos pb = cells.pos_get(j);
            const auto &    ea = cells.ext_get(j - 1);

            // same total order as set_input_qsa: pos, then ext.y, then ext.x; strictly increasing along the cells
            if (pa > pb || (pa == pb && !cells.ext_get(j).is_2d_gt(ea.x, ea.y))) {
                return 5;
            }

            // meant: a jump only next to a 2-D cell. [blk-rebuild2] With IMROPE text cells carry ext=(p,p), so this never fires (see above)
            if (pb != pa + 1 && !is_2d(j - 1) && !is_2d(j)) {
                return 7;
            }
        }

        return 0;
    };

    const char * dummy = nullptr;
    if (why == nullptr) {
        why = &dummy;
    }

    const llama_kv_cache * const caches[2] = { get_mem_attn(), get_mem_idx() };

    for (int c = 0; c < 2; ++c) {
        const int code = check(caches[c]);

        if (code != 0) {
            *why = (c == 0 ? why_attn : why_idx)[code];

            return false;
        }
    }

    return true;
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_blk() const {
    return mem_blk.get();
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // rank cells instead of using the position when positions do not number the cells:
        // - mrope repeats one position across an image;
        // - a context that skipped an image's embeddings (the MTP draft context) keeps the target's positions, which
        //   jump past the image and so run ahead of its cells, beyond the cell window
        if (((dup && ubatch->is_pos_2d()) || oor) && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            const auto before = [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            };

            // one sequence written in order leaves its cells sorted already: O(n_kv) instead of a sort per ubatch
            if (!std::is_sorted(order.begin(), order.end(), before)) {
                std::sort(order.begin(), order.end(), before);
            }

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const bool      q2 = ubatch->is_pos_2d();
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = q2 ? ubatch->pos[i + n_tokens]   : 0;
                const llama_pos qx = q2 ? ubatch->pos[i + n_tokens*2] : 0;

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    // [blk-rebuild] next() runs only after the ubatch went through, so a failed one retries the rebuild on the next decode
    // read the cursor-dependent values before ctx_idx moves on
    if (blk_rebuild_cur) {
        const int64_t r  = mem->get_blk_ratio();
        const int64_t n  = (int64_t) get_attn()->get_ubatch().n_tokens;
        const int64_t c0 = (int64_t) get_idx()->ubatch_cell0();

        mem->set_blk_rebuild(false);

        LLAMA_LOG_WARN("qsa-blk: pooled keys rebuilt (%d blocks)\n", r > 0 ? (int) ((c0 + n - 1)/r + 1) : 0);
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    // a token placed anywhere but in the next cell invalidates the pooled block keys
    if (mem && mem->get_mem_blk() && mem->get_blk_valid() && ctx_idx &&
        get_attn()->has_ubatch() && !get_attn()->ubatch_cells_follow()) {
        if (qsa_blk_debug()) {
            const auto & ub = get_attn()->get_ubatch();
            LLAMA_LOG_WARN("qsa-blk: layout invalidated by a misplaced ubatch (n_tokens=%u, pos0=%d)\n", ub.n_tokens, ub.n_tokens ? ub.pos[0] : -1);   // [blk-rebuild] INFO -> WARN
        }
        mem->set_blk_valid(false);
    }

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // [blk-rebuild] the cells of this ubatch are in place now, so blk_ok() can say whether the pooled cache is usable for it;
    // if it is and a rebuild is pending, this ubatch re-pools every block
    blk_rebuild_cur = mem && mem->get_blk_rebuild() && blk_ok();

    // [blk-rebuild2] apply() runs once per ubatch attempt (llama_context::process_ubatch; the reserve paths build their graphs on init_full()
    // contexts and never apply, and an update context has no ubatch so blk_ok() is false for it). A pending rebuild that an earlier apply()
    // already captured and that next() never confirmed means that attempt failed: do not retry the rebuild forever, fall back to the dense
    // path. set_blk_valid(false) also clears blk_rebuild and the counter.
    if (blk_rebuild_cur) {
        if (mem->get_blk_rebuild_tries() > 0) {
            LLAMA_LOG_WARN("qsa-blk: rebuild failed, falling back to dense (%s)\n", mem->get_blk_who());

            mem->set_blk_valid(false);
            blk_rebuild_cur = false;
        } else {
            mem->set_blk_rebuild_tries(1);
        }
    }

    return res;
}

llama_kv_cache * llama_memory_hybrid_idx_context::get_blk() const {
    return mem ? mem->get_mem_blk() : nullptr;
}

bool llama_memory_hybrid_idx_context::blk_ok() const {
    if (get_blk() == nullptr || !mem->get_blk_valid() || !get_attn()->ubatch_cells_follow()) {
        return false;
    }
    // the pooled blocks name cells of both caches, so they must agree; if they do not, the layout is gone for good
    if (get_idx() == nullptr || !get_idx()->ubatch_cells_follow() || get_attn()->ubatch_cell0() != get_idx()->ubatch_cell0()) {
        mem->set_blk_valid(false);
        return false;
    }
    return true;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
}
