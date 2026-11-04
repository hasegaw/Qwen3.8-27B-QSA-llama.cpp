#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
static bool inspect(ggml_tensor * t, bool ask, void *) {
    bool wanted = t->type == GGML_TYPE_I32 && (std::strstr(t->name, "qsa27b_top_blocks") != nullptr || std::strstr(t->name, "indexer_top_blk") != nullptr);
    if (ask) return wanted;
    if (wanted) {
        std::vector<int32_t> ids(ggml_nelements(t));
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size()*sizeof(int32_t));
        std::fprintf(stderr, "SELECT %s:", t->name);
        for (auto id : ids) std::fprintf(stderr, " %d", id);
        std::fprintf(stderr, "\n");
    }
    return true;
}
int main(int argc, char **argv) {
    if (argc < 4 || argc > 7) return 2;
    const int count = argc >= 7 ? std::atoi(argv[6]) : 29;
    if (std::atoi(argv[3]) == 2) setenv("LLAMA_QSA27B_REFERENCE", "1", 1);
    setenv("LLAMA_QSA_RESERVE_BLK", "1", 1);
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = argc >= 5 ? 99 : 0;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.cb_eval = inspect;
    cp.n_ctx = 256; cp.n_batch = std::max(32, count); cp.n_ubatch = argc >= 6 ? std::atoi(argv[5]) : 7;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (argc >= 5) {
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    llama_set_qsa_enabled(ctx, std::atoi(argv[3]) != 0);
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = i+1; batch.pos[i] = i; batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0; batch.logits[i] = true;
    }
    if (llama_decode(ctx, batch)) return 5;
    auto * file = std::fopen(argv[2], "wb");
    std::fwrite(llama_get_logits(ctx), sizeof(float), count*128, file);
    std::fclose(file);
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
}
