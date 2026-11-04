#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "llama-model.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int hidden=128, vocab=128;
static bool cache_only = false, saw_query = false;
static bool inspect(ggml_tensor * t, bool ask, void *) {
    const bool query = std::strstr(t->name, "mtp_Qcur") != nullptr;
    if (ask) { return query; }
    if (cache_only && query) { saw_query = true; }
    return true;
}
static void fill(llama_batch & b,int n,int offset,bool outputs=true) {
    b.n_tokens=n;
    for(int i=0;i<n;++i) {
        b.token[i]=i+offset+1;b.pos[i]=i+offset;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=outputs;
        for(int d=0;d<hidden;++d)b.embd[i*hidden+d]=std::sin(float((i+offset)*hidden+d)*0.07f)*0.2f;
    }
}
static float compare(const std::vector<float> & a,const float * b) {
    float err=0;for(size_t i=0;i<a.size();++i){if(!std::isfinite(b[i]))std::exit(7);err=std::max(err,std::abs(a[i]-b[i]));}return err;
}
int main(int argc,char ** argv) {
    if(argc!=2)return 2;
    ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.load_mtp=true;
    auto * model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
    auto cp=llama_context_default_params();cp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx=256;cp.n_batch=16;cp.n_ubatch=8;cp.n_threads=2;cp.n_threads_batch=2;
    cp.cb_eval=inspect;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;
    auto * ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
    auto batch=llama_batch_init(8,hidden,1);batch.token=(llama_token*)std::malloc(8*sizeof(llama_token));
    llama_set_embeddings_nextn(ctx,true,false);
    fill(batch,7,0);if(llama_decode(ctx,batch))return 5;
    float * all=llama_get_embeddings_nextn(ctx);std::vector<float> full(all,all+7*hidden);
    llama_memory_clear(llama_get_memory(ctx),true);llama_set_embeddings_nextn(ctx,true,true);
    fill(batch,7,0);for(int i=0;i<7;++i)batch.logits[i]=(i==1||i==6);
    if(llama_decode(ctx,batch))return 5;
    std::vector<float> want;for(int i : {1,6})want.insert(want.end(),full.begin()+i*hidden,full.begin()+(i+1)*hidden);
    float row_error=compare(want,llama_get_embeddings_nextn(ctx));
    std::printf("masked hidden rows max_abs_error=%.9g\n",row_error);if(row_error>2e-4)return 6;
    // One selected query from a padded batch, with nonconsecutive positions.
    // With LLAMA_MTP_DENSE_BLOCK_MIN=0 this exercises the exhaustive block
    // kernel and its one-column mask; the unmasked output uses normal FA.
    llama_memory_clear(llama_get_memory(ctx),true);llama_set_embeddings_nextn(ctx,true,false);
    fill(batch,4,0);for(int i=0;i<4;++i)batch.pos[i]=3+i*i*2;
    if(llama_decode(ctx,batch))return 5;
    auto * padded=llama_get_embeddings_nextn(ctx);
    std::vector<float> one_want(padded+2*hidden,padded+3*hidden);
    auto * padded_logits=llama_get_logits(ctx);
    std::vector<float> one_logits(padded_logits+2*vocab,padded_logits+3*vocab);
    llama_memory_clear(llama_get_memory(ctx),true);llama_set_embeddings_nextn(ctx,true,true);
    fill(batch,4,0);for(int i=0;i<4;++i){batch.pos[i]=3+i*i*2;batch.logits[i]=(i==2);}
    if(llama_decode(ctx,batch))return 5;
    float one_error=compare(one_want,llama_get_embeddings_nextn(ctx));
    float one_logit_error=compare(one_logits,llama_get_logits(ctx));
    int expected=int(std::max_element(one_logits.begin(),one_logits.end())-one_logits.begin());
    int actual=int(std::max_element(llama_get_logits(ctx),llama_get_logits(ctx)+vocab)-llama_get_logits(ctx));
    std::printf("padded selected query hidden_error=%.9g logit_error=%.9g argmax=%d/%d\n",one_error,one_logit_error,expected,actual);
    if(one_error>0.005f||one_logit_error>0.005f||expected!=actual)return 6;
    // Full prompt catch-up followed by the same next-token input.
    llama_memory_clear(llama_get_memory(ctx),true);fill(batch,7,0);if(llama_decode(ctx,batch))return 5;
    fill(batch,1,7);if(llama_decode(ctx,batch))return 5;
    auto * logits=llama_get_logits(ctx);std::vector<float> next(logits,logits+vocab);
    llama_memory_clear(llama_get_memory(ctx),true);fill(batch,7,0,false);llama_set_mtp_kv_only(ctx,true);
    cache_only=true;
    if(llama_decode(ctx,batch))return 5;
    cache_only=false;
    if(saw_query){std::fprintf(stderr,"KV-only path evaluated query projection\n");return 8;}
    llama_set_mtp_kv_only(ctx,false);fill(batch,1,7);if(llama_decode(ctx,batch))return 5;
    float cache_error=compare(next,llama_get_logits(ctx));
    std::printf("KV-only catch-up next-token logits max_abs_error=%.9g\n",cache_error);if(cache_error>2e-4)return 6;
    // The same token embeddings supplied as image rows must produce the same KV.
    cp.n_ubatch=3;
    auto * image_ctx=llama_init_from_model(model,cp);if(!image_ctx)return 4;
    llama_set_embeddings_nextn(image_ctx,true,true);
    fill(batch,7,0,false);
    std::vector<float> embeddings(7*hidden);
    if(model->tok_embd->type!=GGML_TYPE_F32)return 9;
    for(int i=0;i<7;++i)ggml_backend_tensor_get(model->tok_embd,embeddings.data()+i*hidden,
            (size_t) batch.token[i]*hidden*sizeof(float),hidden*sizeof(float));
    std::vector<llama_pos> positions(4*7);
    for(int a=0;a<4;++a)for(int i=0;i<7;++i)positions[a*7+i]=i;
    llama_batch image=batch;image.token=nullptr;image.pos=positions.data();
    llama_set_mtp_input_embeddings(image_ctx,embeddings.data(),7);
    llama_set_mtp_kv_only(image_ctx,true);cache_only=true;saw_query=false;
    if(llama_decode(image_ctx,image))return 5;
    cache_only=false;if(saw_query)return 8;
    llama_set_mtp_kv_only(image_ctx,false);
    llama_set_mtp_input_embeddings(image_ctx,nullptr,0);
    fill(batch,1,7);if(llama_decode(image_ctx,batch))return 5;
    float image_error=compare(next,llama_get_logits(image_ctx));
    std::printf("image embedding catch-up next-token logits max_abs_error=%.9g\n",image_error);
    if(image_error>0.001f)return 6;
    llama_free(image_ctx);
    llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
}
