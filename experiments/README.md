# Quest experiments

Experiments run while building Quest sparse decode ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)) on this branch. Each experiment has its own folder: a `README.md` write-up (question, commands, results, conclusion), plus the original scripts, their summary output and the raw logs. `tools/` holds what several experiments share. This directory is not meant for upstream.

| # | Experiment | Outcome |
|---|---|---|
| 01 | [Budget sweep at 4k](01-budget-sweep-4k/README.md) | Budget 2048 gives KLD 0.006; budget 512 is 4.4× faster at 32k |
| 02 | [Quality at 32k, shared selection](02-long-context-32k/README.md) | Quality drops at 32k: KLD 0.114 at budget 2048 |
| 03 | [Per-query-head selection](03-per-head-selection/README.md) | Halves KLD at 32k (0.053) for about 5% speed; made the default |
| 04 | [Needle test and q8_0/q4_0 KV (Qwen)](04-needle-and-kv-quant/README.md) | Per-head b2048 needle 15/15; q4_0 KV breaks Qwen |
| 05 | [Second model: Llama-3.2-1B](05-llama-3.2-1b/README.md) | Holds up: KLD 0.057, needle 15/15, 4.2× faster at 32k |
| 06 | [Quest on a quantized KV cache](06-quest-on-quantized-kv/README.md) | Quest + q8_0: half the cache memory, 4.0–5.3× faster than dense f16 at 32k |
| 07 | [CUDA port on an RTX 4080](07-cuda-4080/README.md) | Not run yet |
| 08 | [Code-aware KV stitching (kvlink)](08-kvlink-code-aware-stitching/README.md) | Negative result: no better than random token selection |

## Setup shared by all experiments

- **Machine:** Beelink, AMD Ryzen 5 5500U (6 cores / 12 threads), 26 GB RAM, CPU only, `-t 8` unless noted.
- **Models:** `qwen2.5-0.5b-instruct-q8_0.gguf` (14 query heads / 2 KV heads, GQA 7:1) and `Llama-3.2-1B-Instruct-Q8_0.gguf` (bartowski; 32 / 8, GQA 4:1).
- **Text:** `tools/pride.txt` (Pride and Prejudice from Project Gutenberg, public domain).
- **Quality metric:** KL divergence of the next-token distribution against a dense f16 run, from `llama-perplexity --kl-divergence`. `-ub 1` is needed so every token goes through single-token decode, which is the only path Quest touches (this relies on the perplexity commit in this branch). Dense attention gives the same result either way, so the 32k dense baselines were run without `-ub 1` to save time.
- **Speed metric:** `llama-bench -p 0 -n 64 -d 8192,32768`, i.e. tg64 at depths 8k and 32k.
- **Needle test:** `tools/needle.cpp` hides a 5-digit code at depths 5/25/50/75/95% of a 32k haystack (3 trials per depth). The question is decoded one token at a time.

Quest is configured through environment variables: `GGML_QUEST_BUDGET` (tokens attended per step; 0 = off) and `GGML_QUEST_PER_HEAD` (1 = per query head, the default since experiment 03; 0 = shared per KV head).

Build, from the repo root:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-perplexity llama-bench
g++ -O2 -std=c++17 experiments/tools/needle.cpp -Iinclude -Iggml/include -Lbuild/bin \
    -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/bin -o build/bin/needle
```

The commands in each write-up assume `M=<model.gguf>` and `T=experiments/tools/pride.txt`, and are run from the repo root. They are cleaned-up versions of the scripts in each folder. The checked-in scripts are the originals as run on the Beelink, so they still contain its absolute paths. The large files are not checked in: `.kld` baselines (GBs) and saved needle prefill states.
