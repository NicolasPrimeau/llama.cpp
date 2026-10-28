# 03: Per-query-head selection

**Question:** Under GQA, does letting each query head choose its own pages recover the quality lost in experiment 02, and what does it cost?

Shared mode takes the maximum score over the group's query heads and picks one page set per KV head. Per-head mode scores and picks pages for each query head separately. It reads more distinct pages, but each head gets the pages it actually attends to.

**Setup:** Qwen2.5-0.5B, 32k, same baseline as experiment 02.

```sh
GGML_QUEST_PER_HEAD=1 GGML_QUEST_BUDGET=2048 build/bin/llama-perplexity -m $M -f $T -c 32768 -fa on -t 8 --chunks 2 \
    -ub 1 --kl-divergence-base dense-32k.kld --kl-divergence
for cfg in "0 0" "2048 0" "2048 1" "1024 1"; do set -- $cfg
  GGML_QUEST_BUDGET=$1 GGML_QUEST_PER_HEAD=$2 build/bin/llama-bench -m $M -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 3
done
```

## Results

| 32k, budget 2048 | KLD | KLD p99 | Same top-1 | PPL ratio |
|---|---|---|---|---|
| shared (exp. 02) | 0.114 | 0.71 | 79.4% | 1.101 |
| per-head | **0.053** | 0.31 | 85.5% | 1.028 |

Per-head at budget 2048 matches shared at budget 4096.

| tg64 (t/s) | 8k | 32k |
|---|---|---|
| dense | 26.9 | 6.59 |
| shared b2048 | 41.2 | 26.4 |
| per-head b2048 | 39.0 | 24.9 (3.8×) |
| per-head b1024 | 43.5 | 26.6 |

## Conclusion

Per-head selection halves KLD at 32k for about 5% of decode speed, so it became the default (commit "Quest selects pages per query head by default"). `GGML_QUEST_PER_HEAD=0` keeps the shared mode.

## Files

- `quest-perhead.sh` produces `results.txt` (KLD); `quest-bench2.sh` produces `bench-results.txt` (speed). Both are the original scripts.
- `logs/`: raw output.
