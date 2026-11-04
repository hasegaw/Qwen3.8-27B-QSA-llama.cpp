#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// [mtp-kvonly] QWEN4EXP MTP context: while set, a decode builds only the cache writes of the MTP block (attention K/V,
// raw indexer keys, pooled block keys) - no attention, FFN, head or outputs. For catch-up rows that request nothing.
LLAMA_API void llama_set_mtp_kv_only(struct llama_context * ctx, bool value);

// Qwen35 MTP: separate image token embeddings from the hidden rows in batch.embd.
// The input is copied and may be released by the caller. Clear after the decode.
LLAMA_API void llama_set_mtp_input_embeddings(struct llama_context * ctx, const float * data, uint32_t n_tokens);

// [la-s0] LLAMA_LOOKAHEAD: queue the GPU0 part of the next verify window while the current one finishes on GPU1 (two GPUs, pipeline
// parallel); then either resume it with the same batch (only its remaining part runs) or discard it. 0 = queued, 1 = not queued
LLAMA_API int32_t llama_decode_lookahead(struct llama_context * ctx, struct llama_batch batch);
LLAMA_API int32_t llama_decode_resume   (struct llama_context * ctx, struct llama_batch batch);
LLAMA_API void    llama_decode_discard  (struct llama_context * ctx);
LLAMA_API bool    llama_lookahead_pending(const struct llama_context * ctx);

// [pp-plan-hint] prompt tokens left after the batch of the next decode call (LLAMA_PP_PLAN_HINT=1 with LLAMA_PP_PLAN_AHEAD=1): pp plan-ahead
// reserves for the prompt's end once instead of at every call. Cleared by each decode call.
LLAMA_API void llama_set_pp_plan_hint(struct llama_context * ctx, uint32_t n_tokens);

// [mtp-lag] the unmasked nextn rows of consecutive decode calls alternate between two host buffers. _call: id of the most recent
// call's rows (0 = off). _prev(id): the rows of the call before the most recent one if that call is `id`, after waiting only for
// its copies; NULL if they are gone.
LLAMA_API void          llama_set_embeddings_nextn_lag (struct llama_context * ctx, bool value);
LLAMA_API uint64_t      llama_get_embeddings_nextn_call(struct llama_context * ctx);
LLAMA_API const float * llama_get_embeddings_nextn_prev(struct llama_context * ctx, uint64_t call_id);

// [mtp-async] the MTP context's kv-only catch-up decode on a worker thread (LLAMA_MTP_ASYNC=1). Every llama_* call on that context
// and every llama_memory_* call on its memory waits for it first; llama_async_join waits and returns the decode status.
LLAMA_API void    llama_decode_kv_only_async(struct llama_context * ctx, struct llama_batch batch);
LLAMA_API int32_t llama_async_join          (struct llama_context * ctx);
LLAMA_API int32_t llama_async_join_by       (struct llama_context * ctx, const char * who); // [mtp-async-trace]

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
