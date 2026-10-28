# 01: Budget sweep at 4k

**Question:** Does the Quest CPU implementation work, how does quality fall as the budget shrinks, and how much faster is decode?

**Setup:** Qwen2.5-0.5B, 4k context, 4 chunks (16k scored tokens). Selection is shared per KV head, which was the only mode at the time (`GGML_QUEST_PER_HEAD=0` reproduces it now).

```sh
P="build/bin/llama-perplexity -m $M -f $T -c 4096 -fa on -t 8 --chunks 4"
$P -ub 1 --kl-divergence-base dense-4k.kld                       # dense baseline
for b in 256 512 1024 2048; do
  GGML_QUEST_PER_HEAD=0 GGML_QUEST_BUDGET=$b $P -ub 1 --kl-divergence-base dense-4k.kld --kl-divergence
done
for b in 0 512; do
  GGML_QUEST_PER_HEAD=0 GGML_QUEST_BUDGET=$b build/bin/llama-bench -m $M -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 2
done
```

## Results

Dense PPL is 15.98.

| Budget | KLD | Same top-1 | PPL ratio |
|---|---|---|---|
| 256 | 0.154 | 78.1% | 1.115 |
| 512 | 0.077 | 83.9% | 1.062 |
| 1024 | 0.030 | 89.4% | 1.024 |
| 2048 | 0.006 | 95.2% | 1.006 |

| tg64 (t/s) | 8k | 32k |
|---|---|---|
| dense | 27.8 | 6.70 |
| Quest b512 | 48.7 | 29.2 (4.4×) |

## Conclusion

The kernel is correct: quality degrades smoothly with the budget, and at budget 2048 of a 4k context it is close to dense. Decode speed now depends on the budget instead of the context length. A 4k context says little about long contexts, so experiment 02 measures quality at 32k.
