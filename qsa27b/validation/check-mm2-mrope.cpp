#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int count = 29, hidden = 128, vocab = 128;
static void fill(llama_batch & batch, const std::vector<float> & embeddings, int begin, int n) {
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        const int token = begin + i;
        std::copy_n(embeddings.data() + token*hidden, hidden, batch.embd + i*hidden);
        const bool image = token >= 3 && token < 19;
        const int t = token < 3 ? token : image ? 3 : token - 12;
        batch.pos[i] = t;
        batch.pos[n + i] = image ? 3 + (token - 3)/4 : t;
        batch.pos[2*n + i] = image ? 3 + (token - 3)%4 : t;
        batch.pos[3*n + i] = t;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true;
    }
}
int main(int argc, char ** argv) {
    if (argc != 5) { return 2; }
    const int mode = std::atoi(argv[3]);
    if (mode == 2) { setenv("LLAMA_QSA27B_REFERENCE", "1", 1); }
    setenv("LLAMA_QSA_RESERVE_BLK", "1", 1);
    setenv("LLAMA_QSA_BLK_REBUILD", "1", 1);
    ggml_backend_load_all(); llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp); if (!model) { return 3; }
    if (model->tok_embd->type != GGML_TYPE_F32) { return 4; }
    std::vector<float> embeddings(count*hidden);
    for (int i = 0; i < count; ++i) {
        ggml_backend_tensor_get(model->tok_embd, embeddings.data() + i*hidden,
                (size_t) (i + 1)*hidden*sizeof(float), hidden*sizeof(float));
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = std::atoi(argv[4]);
    cp.n_threads = cp.n_threads_batch = 2;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx = llama_init_from_model(model, cp); if (!ctx) { return 5; }
    llama_set_qsa_enabled(ctx, mode != 0);
    auto batch = llama_batch_init(count, hidden, 1);
    std::free(batch.pos); batch.pos = (llama_pos *) std::malloc(4*count*sizeof(llama_pos));
    fill(batch, embeddings, 0, count);
    if (llama_decode(ctx, batch)) { return 6; }
    auto * file = std::fopen(argv[2], "wb");
    std::fwrite(llama_get_logits(ctx), sizeof(float), count*vocab, file); std::fclose(file);
    llama_memory_clear(llama_get_memory(ctx), true);
    fill(batch, embeddings, 0, 19); if (llama_decode(ctx, batch)) { return 6; }
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    std::vector<uint8_t> checkpoint(llama_state_seq_get_size_ext(ctx, 0, flags));
    if (llama_state_seq_get_data_ext(ctx, checkpoint.data(), checkpoint.size(), 0, flags) != checkpoint.size()) { return 7; }
    fill(batch, embeddings, 19, 10); if (llama_decode(ctx, batch)) { return 6; }
    std::vector<float> expected(llama_get_logits(ctx), llama_get_logits(ctx) + 10*vocab);
    if (llama_state_seq_set_data_ext(ctx, checkpoint.data(), checkpoint.size(), 0, flags) != checkpoint.size()) { return 7; }
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, 7, -1)) { return 7; }
    fill(batch, embeddings, 19, 10); if (llama_decode(ctx, batch)) { return 6; }
    float error = 0;
    for (int i = 0; i < 10*vocab; ++i) {
        const float got = llama_get_logits(ctx)[i]; if (!std::isfinite(got)) { return 8; }
        error = std::max(error, std::abs(got - expected[i]));
    }
    std::printf("M-RoPE cache restore mode=%d ubatch=%d max_logit_error=%.9g\n", mode, cp.n_ubatch, error);
    if (error > 0.001f) { return 9; }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
}
