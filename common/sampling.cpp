#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <atomic>
#include <cstdlib>
#include <vector>

// [sample-topk] env LLAMA_SAMPLE_TOPK (patch-mt-sampletopk.py): 0/unset = off, 1 = on, 2 = on + check against the full array
static int common_sampler_topk_mode() {
    static const int mode = [] {
        const char * e = getenv("LLAMA_SAMPLE_TOPK");
        const int m = e ? atoi(e) : 0;
        if (m > 0) {
            LOG_WRN("[sample-topk] LLAMA_SAMPLE_TOPK=%d: top-k candidates without the full-vocabulary array%s\n", m, m >= 2 ? " (check mode)" : "");
        }
        return m;
    }();
    return mode;
}

// [sample-topk] k when top_k is the first sampler of the chain that changes anything, else 0
static int common_sampler_topk_prefilter_k(const common_params_sampling & p) {
    if (common_sampler_topk_mode() <= 0 || p.mirostat != 0 || p.n_probs > 0 || p.top_k <= 0 || p.top_k > 128) {
        return 0;
    }
    for (const auto & s : p.samplers) {
        switch (s) {
            case COMMON_SAMPLER_TYPE_TOP_K:
                return p.top_k;
            case COMMON_SAMPLER_TYPE_PENALTIES: // llama_sampler_penalties::is_disabled
                if (p.penalty_last_n == 0 || (p.penalty_repeat == 1.0f && p.penalty_freq == 0.0f && p.penalty_present == 0.0f)) {
                    continue;
                }
                return 0;
            case COMMON_SAMPLER_TYPE_DRY: // llama_sampler_init_dry returns an empty sampler
                if (p.dry_multiplier == 0.0f || p.dry_base < 1.0f || p.dry_penalty_last_n <= 0) {
                    continue;
                }
                return 0;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (p.top_n_sigma <= 0.0f) {
                    continue;
                }
                return 0;
            default:
                return 0;
        }
    }
    return 0;
}

struct common_sampler_topk_stats {
    std::atomic<int64_t> checked {0};
    std::atomic<int64_t> bad_id  {0};
    std::atomic<int64_t> bad_cand{0};
};

static common_sampler_topk_stats g_topk_stats;

// [gpu-topk] set_logits from the device top-K (llama_get_topk_ith, LLAMA_GPU_TOPK, patch-mt-gputopk.py): uses, fallbacks to the CPU scan
// of the row by reason, and the check-mode results of the device path
struct common_sampler_gputopk_stats {
    std::atomic<int64_t> used    {0};
    std::atomic<int64_t> fb_none {0}; // no device top-K for the row (LLAMA_GPU_TOPK unset, tensor split, ...)
    std::atomic<int64_t> fb_k    {0}; // K < k
    std::atomic<int64_t> fb_bias {0}; // a logit_bias entry > 0 / NaN / out of range / duplicated, or a bad device entry (id, NaN)
    std::atomic<int64_t> fb_exact{0}; // the k-th best is not strictly above the smallest device value: exactness not provable
    std::atomic<int64_t> checked {0};
    std::atomic<int64_t> bad     {0};
};

static common_sampler_gputopk_stats g_gputopk_stats;

// [gpu-topk] the last set_logits of this thread took the device path (read by the check in common_sampler_sample)
static thread_local bool tl_gputopk_used = false;

static void common_sampler_gputopk_log(const char * what) {
    const auto & g = g_gputopk_stats;
    LOG_WRN("[gpu-topk] %s: device path %lld, CPU scan: no device top-K %lld, K < k %lld, bias %lld, not provable %lld; checked %lld, mismatches %lld\n",
        what, (long long) g.used.load(), (long long) g.fb_none.load(), (long long) g.fb_k.load(), (long long) g.fb_bias.load(),
        (long long) g.fb_exact.load(), (long long) g.checked.load(), (long long) g.bad.load());
}

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    // [sample-topk] allow_topk: the caller runs the chain on cur without applying the grammar or a forcing reasoning budget first;
    // returns true when cur holds only the top-k candidates
    bool set_logits(struct llama_context * ctx, int idx, bool allow_topk = false) {
        bool used_topk = false;
        tl_gputopk_used = false; // [gpu-topk]
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            // [gpu-topk] the device top-K first: when it decides, the logits row is neither read nor copied to the host
            used_topk = allow_topk && set_logits_topk_device(ctx, idx, n_vocab, vocab);
            if (!used_topk) {
                const auto * logits = llama_get_logits_ith(ctx, idx);
                GGML_ASSERT(logits != nullptr);
                used_topk = allow_topk && set_logits_topk(logits, n_vocab, vocab);
                if (!used_topk) {
                    cur.resize(n_vocab);
                    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
                    }
                }
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };

        return used_topk;
    }

    // [gpu-topk] the selection of set_logits_topk from the device top-K of output idx (llama_get_topk_ith): the K largest raw logits and
    // their ids, without reading the logits row (a deferred row is then never copied to the host). All biases must be <= 0: a token outside
    // the K then has biased <= raw <= the smallest device value, so when the k-th best biased value among the K is strictly greater than
    // that value, the k best of the K are the k best of the vocabulary. false (cur untouched): the caller scans the row (v2) as before
    bool set_logits_topk_device(struct llama_context * ctx, int idx, int n_vocab, const llama_vocab * vocab) {
        const int k = common_sampler_topk_prefilter_k(params);
        if (k <= 0) {
            return false;
        }

        const llama_token * d_ids  = nullptr;
        const float       * d_vals = nullptr;
        const int K = llama_get_topk_ith(ctx, idx, &d_ids, &d_vals);
        if (K <= 0 || d_ids == nullptr || d_vals == nullptr) {
            ++g_gputopk_stats.fb_none;
            return false;
        }
        if (K < k) {
            ++g_gputopk_stats.fb_k;
            return false;
        }

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);

        static thread_local std::vector<llama_logit_bias> dbias;
        dbias.clear();
        for (const auto & lb : params.logit_bias) {
            dbias.push_back(lb);
        }
        for (int32_t i = 0; i < n_suppress; ++i) {
            dbias.push_back({ suppress[i], -INFINITY });
        }
        for (size_t i = 0; i < dbias.size(); ++i) {
            if (dbias[i].token < 0 || dbias[i].token >= n_vocab || !(dbias[i].bias <= 0.0f)) {
                ++g_gputopk_stats.fb_bias;
                return false;
            }
            for (size_t j = 0; j < i; ++j) {
                if (dbias[j].token == dbias[i].token) {
                    ++g_gputopk_stats.fb_bias;
                    return false;
                }
            }
        }

        // the K device entries with their biased logit in .p
        static thread_local std::vector<llama_token_data> dcand;
        dcand.resize(K);
        float min_raw = INFINITY;
        for (int i = 0; i < K; ++i) {
            const llama_token t   = d_ids[i];
            const float       raw = d_vals[i];
            if (t < 0 || t >= n_vocab || raw != raw) {
                ++g_gputopk_stats.fb_bias;
                return false;
            }
            float adj = raw;
            for (const auto & lb : dbias) {
                if (lb.token == t) {
                    adj = raw + lb.bias;
                }
            }
            dcand[i] = llama_token_data{ t, raw, adj };
            min_raw = std::min(min_raw, raw);
        }
        std::partial_sort(dcand.begin(), dcand.begin() + k, dcand.end(),
            [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
        if (!(dcand[k - 1].p > min_raw)) {
            ++g_gputopk_stats.fb_exact;
            return false;
        }
        std::sort(dcand.begin(), dcand.begin() + k,
            [](const llama_token_data & a, const llama_token_data & b) { return a.id < b.id; });

        cur.resize(k);
        for (int i = 0; i < k; ++i) {
            cur[i] = llama_token_data{ dcand[i].id, dcand[i].logit, 0.0f };
        }

        tl_gputopk_used = true;
        const int64_t n = ++g_gputopk_stats.used;
        if (n == 1) {
            LOG_WRN("[gpu-topk] top-k candidates from the device top-%d (k = %d): the logits row is not read\n", K, k);
        }
        if (common_sampler_topk_mode() < 2 && n % 20000 == 0) {
            common_sampler_gputopk_log("progress");
        }
        return true;
    }

    // [sample-topk] cur = the k tokens with the largest biased logit (raw logit + logit_bias entries, as the chain's logit_bias sampler
    // adds them), with their raw logits, in ascending token id; false = not applicable, cur untouched
    bool set_logits_topk(const float * logits, int n_vocab, const llama_vocab * vocab) {
        const int k = common_sampler_topk_prefilter_k(params);
        if (k <= 0) {
            return false;
        }

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);

        static thread_local std::vector<llama_logit_bias> bias;
        bias.clear();
        for (const auto & lb : params.logit_bias) {
            bias.push_back(lb);
        }
        for (int32_t i = 0; i < n_suppress; ++i) {
            bias.push_back({ suppress[i], -INFINITY });
        }
        for (size_t i = 0; i < bias.size(); ++i) {
            if (bias[i].token < 0 || bias[i].token >= n_vocab) {
                return false;
            }
            for (size_t j = 0; j < i; ++j) {
                if (bias[j].token == bias[i].token) {
                    return false; // duplicates: logit_bias applies them differently to a full and a partial array
                }
            }
        }

        const int kk = k + (int) bias.size();
        if (kk >= n_vocab) {
            return false;
        }

        // the kk largest raw logits: min-heap on the logit, its root is the smallest kept
        static thread_local std::vector<llama_token_data> heap;
        heap.resize(kk);
        for (int t = 0; t < kk; ++t) {
            heap[t] = llama_token_data{ t, logits[t], 0.0f };
        }
        const auto cmp_min = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
        std::make_heap(heap.begin(), heap.end(), cmp_min);
        float thr = heap.front().logit;
        const auto push = [&](int t, float l) {
            std::pop_heap(heap.begin(), heap.end(), cmp_min);
            heap.back() = llama_token_data{ t, l, 0.0f };
            std::push_heap(heap.begin(), heap.end(), cmp_min);
            thr = heap.front().logit;
        };
        // blocks of 32 logits with none above the smallest kept one are skipped with a 4-wide compare (SSE2 / NEON baseline);
        // the tokens are still visited in ascending order, so the result is that of the plain scan (52 -> 17 us per row, main PC)
        typedef float topk_v4f __attribute__((vector_size(16)));
        typedef int   topk_v4i __attribute__((vector_size(16)));
        int t = kk;
        for (; t < n_vocab && (t & 31); ++t) {
            if (logits[t] > thr) { push(t, logits[t]); }
        }
        for (; t + 32 <= n_vocab; t += 32) {
            const topk_v4f vt = { thr, thr, thr, thr };
            topk_v4i acc = { 0, 0, 0, 0 };
            for (int j = 0; j < 32; j += 4) {
                topk_v4f v;
                memcpy(&v, logits + t + j, sizeof(v));
                acc |= (v > vt);
            }
            if ((acc[0] | acc[1] | acc[2] | acc[3]) == 0) {
                continue;
            }
            for (int u = t; u < t + 32; ++u) {
                if (logits[u] > thr) { push(u, logits[u]); }
            }
        }
        for (; t < n_vocab; ++t) {
            if (logits[t] > thr) { push(t, logits[t]); }
        }

        // candidates by biased logit (in .p): unbiased tokens of the heap + every biased token
        static thread_local std::vector<llama_token_data> cand;
        cand.clear();
        for (const auto & e : heap) {
            bool biased = false;
            for (const auto & lb : bias) {
                biased |= lb.token == e.id;
            }
            if (!biased) {
                cand.push_back(llama_token_data{ e.id, e.logit, e.logit });
            }
        }
        for (const auto & lb : bias) {
            cand.push_back(llama_token_data{ lb.token, logits[lb.token], logits[lb.token] + lb.bias });
        }
        if ((int) cand.size() < k) {
            return false;
        }
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
            [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
        std::sort(cand.begin(), cand.begin() + k,
            [](const llama_token_data & a, const llama_token_data & b) { return a.id < b.id; });

        cur.resize(k);
        for (int i = 0; i < k; ++i) {
            cur[i] = llama_token_data{ cand[i].id, cand[i].logit, 0.0f };
        }

        return true;
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        // mix it, the chain and the draft are seeded from this one too
        /* .rng     = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .rng     = */ gsmpl->rng,
    };
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->t_total_us = src->t_total_us;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

// Greedy sampling whose chain cannot move the maximum reduces to an argmax over the logits: the
// temperature stage keeps only the largest logit, and nothing configured here can reorder or remove
// it. Scanning the logits directly skips building (and partially sorting) a candidate array over
// the whole vocabulary, which costs about a millisecond per token for a 248K vocabulary.
static bool common_sampler_greedy_eligible(const common_sampler * gsmpl) {
    static const bool disabled = [] {
        const char * env = getenv("LLAMA_SAMPLER_NO_GREEDY_FAST");
        return env && atoi(env) != 0;
    }();
    const auto & p = gsmpl->params;
    if (disabled || !(p.temp <= 0.0f) || p.mirostat != 0 || p.n_probs > 0 || !p.logit_bias.empty() || gsmpl->grmr) {
        return false;
    }
    if (p.penalty_last_n != 0 && (p.penalty_repeat != 1.0f || p.penalty_freq != 0.0f || p.penalty_present != 0.0f)) {
        return false;
    }
    if (p.dry_multiplier != 0.0f && p.dry_penalty_last_n != 0) {
        return false;
    }
    if (p.xtc_probability > 0.0f && p.xtc_threshold <= 0.5f) {
        return false;
    }
    for (const auto s : p.samplers) {
        if (s == COMMON_SAMPLER_TYPE_INFILL || s == COMMON_SAMPLER_TYPE_ADAPTIVE_P) {
            return false;
        }
    }
    return true;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    if (common_sampler_greedy_eligible(gsmpl) &&
            llama_get_sampled_token_ith(ctx, idx) == LLAMA_TOKEN_NULL &&
            llama_get_sampled_logits_ith(ctx, idx) == nullptr && llama_get_sampled_probs_ith(ctx, idx) == nullptr) {
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
        const int n_vocab = llama_vocab_n_tokens(vocab);

        // the model's suppressed tokens carry a -inf bias in the chain
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);

        llama_token best = -1;
        float best_logit = -INFINITY;

        // the device already picked the token (LLAMA_GPU_ARGMAX): no logits copy, no scan.
        // Only without suppressed tokens, which the device argmax does not know about.
        const llama_token device_best = n_suppress == 0 ? llama_get_argmax_ith(ctx, idx) : LLAMA_TOKEN_NULL;
        if (device_best != LLAMA_TOKEN_NULL) {
            best       = device_best;
            best_logit = 0.0f; // not needed by greedy callers
        } else {
            const float * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            for (llama_token t = 0; t < n_vocab; ++t) {
                if (logits[t] > best_logit) {
                    bool suppressed = false;
                    for (int32_t i = 0; i < n_suppress; ++i) {
                        suppressed |= suppress[i] == t;
                    }
                    if (!suppressed) {
                        best = t;
                        best_logit = logits[t];
                    }
                }
            }
        }

        bool ok = best >= 0;
        if (ok && gsmpl->rbudget) {
            // a reasoning budget that is forcing a token rejects everything else: fall back if so
            llama_token_data       single       = { best, best_logit, 0.0f };
            llama_token_data_array single_array = { &single, 1, -1, false };
            llama_sampler_apply(gsmpl->rbudget, &single_array);
            ok = single_array.data[0].logit != -INFINITY;
        }

        if (ok) {
            gsmpl->cur.resize(1);
            gsmpl->cur[0] = llama_token_data{ best, best_logit, 1.0f };
            gsmpl->cur_p  = { gsmpl->cur.data(), 1, 0, true };
            return best;
        }
    }

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // [sample-topk] only when the chain is the first thing applied to cur (the grammar resampling below always uses the full array)
    const bool topk_allow = common_sampler_topk_mode() > 0 &&
        !(grammar_first && grammar_should_apply(gsmpl)) &&
        !(rbudget && common_reasoning_budget_get_state(rbudget) == REASONING_BUDGET_FORCING);

    const bool topk_used = gsmpl->set_logits(ctx, idx, topk_allow);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    // [sample-topk] check mode: a clone of the chain (same state, same RNG) samples the full array
    llama_sampler * topk_chk = topk_used && common_sampler_topk_mode() >= 2 ? llama_sampler_clone(chain) : nullptr;

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (topk_chk) {
        const float * logits = llama_get_logits_ith(ctx, idx);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        static thread_local std::vector<llama_token_data> full;
        full.resize(n_vocab);
        for (llama_token t = 0; t < n_vocab; t++) {
            full[t] = llama_token_data{ t, logits[t], 0.0f };
        }
        llama_token_data_array full_p = { full.data(), full.size(), -1, false };
        llama_sampler_apply(rbudget, &full_p);
        llama_sampler_apply(topk_chk, &full_p);
        llama_sampler_free(topk_chk);

        const bool same_id = full_p.selected >= 0 && full_p.data[full_p.selected].id == id;
        bool same_cand = full_p.size == cur_p.size;
        for (size_t i = 0; same_cand && i < cur_p.size; ++i) {
            same_cand = full_p.data[i].id == cur_p.data[i].id && full_p.data[i].p == cur_p.data[i].p;
        }
        const int64_t n = ++g_topk_stats.checked;
        const int64_t bad_id = same_id ? g_topk_stats.bad_id.load() : ++g_topk_stats.bad_id;
        const int64_t bad_cand = same_cand ? g_topk_stats.bad_cand.load() : ++g_topk_stats.bad_cand;
        if (tl_gputopk_used) { // [gpu-topk] the device path's own count
            ++g_gputopk_stats.checked;
            if (!same_id || !same_cand) {
                const int64_t nb = ++g_gputopk_stats.bad;
                if (nb <= 20) {
                    LOG_WRN("[gpu-topk] check mismatch on the device path #%lld: id %d vs full %d, candidates %zu vs full %zu\n", (long long) nb,
                        id, full_p.selected >= 0 ? full_p.data[full_p.selected].id : -1, cur_p.size, full_p.size);
                }
            }
        }
        if ((!same_id || !same_cand) && bad_id + bad_cand <= 20) {
            LOG_WRN("[sample-topk] check mismatch #%lld: id %d vs full %d, candidates %zu vs full %zu\n", (long long) (bad_id + bad_cand),
                id, full_p.selected >= 0 ? full_p.data[full_p.selected].id : -1, cur_p.size, full_p.size);
        }
        if (n % 2000 == 0) {
            LOG_WRN("[sample-topk] check: %lld checked, %lld id mismatches, %lld candidate mismatches\n", (long long) n, (long long) bad_id, (long long) bad_cand);
            common_sampler_gputopk_log("check"); // [gpu-topk]
        }
    }

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

// [spec-acclog] LLAMA_SPEC_ACCLOG=<file>: one line per verified window (measurement only; patch-mt-specacclog.py)
static FILE * spec_acclog_file() {
    static FILE * f = [] {
        const char * p = getenv("LLAMA_SPEC_ACCLOG");
        return p != nullptr && *p != '\0' ? fopen(p, "a") : (FILE *) nullptr;
    }();
    return f;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    if (FILE * f = spec_acclog_file()) {
        fprintf(f, "G %zu %zu\n", draft.size(), result.size() - 1);
        fflush(f);
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// [spec-acclog] the window replayed on a clone of the sampler (taken before the window): target distribution at every draft
// position along the drafted path, token-wise and block-verification expected accepted drafts
static void spec_acclog_rejection(common_sampler * clone, llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft,
        const std::vector<std::vector<llama_token_data>> & draft_q, const size_t n_acc) {
    FILE * f = spec_acclog_file();
    const size_t n = draft.size();
    std::vector<double> px(n, 0.0), qx(n, 0.0), S(n, 0.0), w(n + 1, 1.0);
    int gram = 0;
    for (size_t k = 0; k < n; ++k) {
        common_sampler_sample(clone, ctx, idxs[k], false);
        if (grammar_should_apply(clone)) {
            gram = 1;
        }
        const auto * cur = common_sampler_get_candidates(clone, true);
        std::unordered_map<llama_token, float> qm;
        for (const auto & e : draft_q[k]) {
            qm[e.id] = e.p;
        }
        const double a = w[k];
        double sum = 0.0;
        for (size_t j = 0; j < cur->size; ++j) {
            const auto it = qm.find(cur->data[j].id);
            const double qv = it == qm.end() ? 0.0 : it->second;
            const double r  = a*cur->data[j].p - qv;
            if (r > 0.0) {
                sum += r;
            }
            if (cur->data[j].id == draft[k]) {
                px[k] = cur->data[j].p;
            }
        }
        const auto itq = qm.find(draft[k]);
        qx[k]    = itq == qm.end() ? 0.0 : itq->second;
        S[k]     = sum;
        w[k + 1] = qx[k] > 0.0 ? std::min(1.0, a*px[k]/qx[k]) : 0.0;
        common_sampler_accept(clone, draft[k], false);
    }
    double tw = 0.0, prod = 1.0;
    for (size_t k = 0; k < n; ++k) {
        prod *= qx[k] > 0.0 ? std::min(1.0, px[k]/qx[k]) : 0.0;
        tw   += prod;
    }
    std::vector<double> h(n + 1, 0.0);
    for (size_t i = 1; i < n; ++i) {
        const double den = S[i] + 1.0 - w[i];
        h[i] = den > 0.0 ? S[i]/den : 1.0;
    }
    if (n > 0) {
        h[n] = w[n];
    }
    double bv = 0.0;
    for (size_t i = 1; i <= n; ++i) {
        double pi = h[i];
        for (size_t j = i + 1; j <= n; ++j) {
            pi *= 1.0 - h[j];
        }
        bv += i*pi;
    }
    fprintf(f, "R %zu %zu %d %.5f %.5f", n, n_acc, gram, tw, bv);
    for (size_t k = 0; k < n; ++k) {
        fprintf(f, " %.5g/%.5g/%.5g", px[k], qx[k], S[k]);
    }
    fputc('\n', f);
    fflush(f);
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first, bool bonus) {
    GGML_ASSERT(idxs.size()    == draft.size() + (bonus ? 1 : 0) && "idxs.size() must be draft.size() + 1 (draft.size() without a bonus) [la-s0b]");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // [spec-acclog] the sampler as it was before this window
    common_sampler * acl_clone = spec_acclog_file() != nullptr ? common_sampler_clone(gsmpl) : nullptr;

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        break;
    }

    if (bonus && i == draft.size()) { // [la-s0b]
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    if (acl_clone != nullptr) {
        if (bonus) {
            spec_acclog_rejection(acl_clone, ctx, idxs, draft, draft_q, result.size() - 1);
        }
        common_sampler_free(acl_clone);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}
