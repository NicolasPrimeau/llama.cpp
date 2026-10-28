# 02: Quality at 32k, shared selection

**Question:** Does quality hold when the budget is a small fraction of a long context?

**Setup:** Qwen2.5-0.5B, 32k context, 2 chunks (32k scored tokens), shared per-KV-head selection.

```sh
P="build/bin/llama-perplexity -m $M -f $T -c 32768 -fa on -t 8 --chunks 2"
$P --kl-divergence-base dense-32k.kld
for b in 1024 2048 4096; do
  GGML_QUEST_PER_HEAD=0 GGML_QUEST_BUDGET=$b $P -ub 1 --kl-divergence-base dense-32k.kld --kl-divergence
done
```

## Results

Dense PPL is 17.72.

| Budget | KLD | KLD p99 | Same top-1 | PPL ratio |
|---|---|---|---|---|
| 1024 | 0.199 | 1.26 | 73.4% | 1.185 |
| 2048 | 0.114 | 0.71 | 79.4% | 1.101 |
| 4096 | 0.052 | 0.32 | 85.4% | 1.047 |

## Conclusion

At 32k, a fixed budget loses much more than it did at 4k: KLD at budget 2048 is 0.114 here versus 0.006 in experiment 01. With 7 query heads sharing each KV head, the shared page selection is a compromise between heads that want different pages. Experiment 03 tests giving each query head its own selection.
