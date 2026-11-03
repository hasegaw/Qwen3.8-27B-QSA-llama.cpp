#pragma once

// A7 P1: weights of the second half (ffn) of one GDN layer, for GGML_OP_A7_GDN_FFN (ggml_a7_gdn_ffn).
// The struct is filled once by the model loader and lives as long as the model; the op keeps only its address in op_params.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_a7_ffn_layer {
    const struct ggml_tensor * hc_norm, * hc_down, * hc_up, * hc_inject;     // hc_ffn_*
    const struct ggml_tensor * gate_inp, * gate_inp_shexp;                    // router (F32), shared gate (F32)
    const struct ggml_tensor * gate_exps, * up_exps, * down_exps;             // IQ2_S, IQ2_S, IQ4_NL
    const struct ggml_tensor * gate_shexp, * up_shexp, * down_shexp;          // Q8_0
    const struct ggml_tensor * hc_norm_next;                                  // hc_attn_norm of layer il+1 (F32), or NULL: the op then skips the q8 of the new R for the next layer
    int il;
};

// A7 P2: weights of the first half (attn) of one GDN layer, for GGML_OP_A7_GDN_ATTN (ggml_a7_gdn_attn).
// Same lifetime rule as ggml_a7_ffn_layer: the array of these structs is filled once and never reallocated.
struct ggml_a7_attn_layer {
    const struct ggml_tensor * hc_norm, * hc_down, * hc_up, * hc_inject;     // hc_attn_*
    const struct ggml_tensor * wqkv, * wqkv_gate;                            // Q8_0 (2560 -> 10240, 2560 -> 6144)
    const struct ggml_tensor * ssm_alpha, * ssm_beta;                        // F32 [2560, 48]
    const struct ggml_tensor * ssm_dt, * ssm_a;                              // F32 [48]
    const struct ggml_tensor * ssm_conv1d;                                   // F32 [4, 10240]
    const struct ggml_tensor * ssm_norm;                                     // F32 [128]
    const struct ggml_tensor * ssm_out;                                      // Q8_0 (6144 -> 2560)
    const struct ggml_tensor * hc_ffn_norm;                                  // for the ffn-side q8 made by the ssmout stage
    int il;
};

#ifdef __cplusplus
}
#endif
