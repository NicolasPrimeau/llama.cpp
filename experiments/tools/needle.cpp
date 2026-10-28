// Needle-in-a-haystack retrieval test that exercises the single-token decode path.
//
// A normal prompt is processed as one batch, so a sparse decode kernel (Quest) never sees the question. Here the
// haystack is prefilled in batches, then the question is fed one token at a time and the answer generated greedily,
// so every token that has to find the needle goes through decode.
//
// usage: needle <model.gguf> <haystack.txt> <n_ctx> <kv type: f16|q8_0|q4_0> <trials per depth> [state dir]
// Quest is configured through its GGML_QUEST_* environment variables. The prefill is dense whatever the Quest
// settings, so with a state dir each trial's prefilled KV is saved once and reloaded by later runs.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, false, false);
    std::vector<llama_token> out(n);
    llama_tokenize(vocab, text.c_str(), text.size(), out.data(), n, false, false);
    return out;
}

static bool decode(llama_context * ctx, llama_batch_ext * batch, const llama_token * tok, int n, llama_pos pos0) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < n; i++) {
        const int32_t   idx = llama_batch_ext_add_token(batch, 0, tok[i]);
        const llama_pos pos = pos0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n - 1, true);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.gguf haystack.txt n_ctx f16|q8_0|q4_0 trials_per_depth\n", argv[0]);
        return 1;
    }
    const int         n_ctx  = atoi(argv[3]);
    const std::string kvt    = argv[4];
    const int         trials = atoi(argv[5]);
    const ggml_type   type_kv = kvt == "q8_0" ? GGML_TYPE_Q8_0 : kvt == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
    const std::string state_dir = argc > 6 ? argv[6] : "";

    std::ifstream     f(argv[2]);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = 2048;
    cp.n_ubatch        = 512;
    cp.n_threads       = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k          = type_kv;
    cp.type_v          = type_kv;
    llama_context * ctx   = llama_init_from_model(model, cp);
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    const std::vector<llama_token> hay = tokenize(vocab, text);
    const char * colors[] = { "blue", "red", "green", "yellow", "purple", "orange", "silver", "brown" };
    const double depths[] = { 0.05, 0.25, 0.5, 0.75, 0.95 };

    std::mt19937 rng(1234);
    int n_ok = 0, n_all = 0;
    for (double depth : depths) {
        for (int t = 0; t < trials; t++) {
            const std::string color = colors[rng() % 8];
            const int         code  = 10000 + rng() % 90000;
            const auto needle = tokenize(vocab, " The secret code for the " + color + " door is " + std::to_string(code) + ". ");
            const auto question = tokenize(vocab, "\n\nQuestion: What is the secret code for the " + color +
                                                  " door?\nAnswer: The secret code for the " + color + " door is");

            const int n_hay = n_ctx - (int) needle.size() - (int) question.size() - 16;
            const int off   = rng() % ((int) hay.size() - n_hay);
            const int at    = (int) (depth * n_hay);

            std::vector<llama_token> prompt(hay.begin() + off, hay.begin() + off + at);
            prompt.insert(prompt.end(), needle.begin(), needle.end());
            prompt.insert(prompt.end(), hay.begin() + off + at, hay.begin() + off + n_hay);

            llama_memory_clear(llama_get_memory(ctx), true);
            const int64_t t0 = ggml_time_us();
            const std::string state = state_dir.empty() ? "" :
                state_dir + "/kv-" + kvt + "-" + std::to_string(n_ctx) + "-" + std::to_string(n_all) + ".bin";
            std::vector<llama_token> saved(prompt.size());
            size_t n_saved = 0;
            if (state.empty() || !llama_state_load_file(ctx, state.c_str(), saved.data(), saved.size(), &n_saved) ||
                    n_saved != prompt.size() || saved != prompt) {
                llama_memory_clear(llama_get_memory(ctx), true);
                for (int p0 = 0; p0 < (int) prompt.size(); p0 += 2048) {
                    const int n = std::min<int>(2048, prompt.size() - p0);
                    if (!decode(ctx, batch, prompt.data() + p0, n, p0)) { fprintf(stderr, "prefill failed\n"); return 1; }
                }
                if (!state.empty()) {
                    llama_state_save_file(ctx, state.c_str(), prompt.data(), prompt.size());
                }
            }
            int pos = prompt.size();
            const int64_t t1 = ggml_time_us();

            // the question one token at a time: this is the decode path
            for (llama_token tk : question) {
                if (!decode(ctx, batch, &tk, 1, pos++)) { fprintf(stderr, "decode failed\n"); return 1; }
            }
            std::string answer;
            for (int i = 0; i < 8; i++) {
                const float * lg = llama_get_logits_ith(ctx, -1);
                llama_token best = 0;
                for (int v = 1; v < llama_vocab_n_tokens(vocab); v++) {
                    if (lg[v] > lg[best]) best = v;
                }
                if (llama_vocab_is_eog(vocab, best)) break;
                char buf[64];
                const int n = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
                answer.append(buf, n);
                if (!decode(ctx, batch, &best, 1, pos++)) { fprintf(stderr, "decode failed\n"); return 1; }
            }
            const bool ok = answer.find(std::to_string(code)) != std::string::npos;
            n_ok += ok;
            n_all++;
            for (char & c : answer) if (c == '\n') c = ' ';
            printf("depth=%.2f code=%d ok=%d answer=\"%s\" prefill=%.0fs total=%.0fs\n", depth, code, ok, answer.c_str(),
                   (t1 - t0)/1e6, (ggml_time_us() - t0)/1e6);
            fflush(stdout);
        }
    }
    printf("RESULT %d/%d\n", n_ok, n_all);

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
