# 05: Second model, Llama-3.2-1B

**Question:** Are the Qwen results specific to Qwen? Llama-3.2-1B has a different GQA ratio (4:1 instead of 7:1), head size 64 and RoPE scaling.

**Setup:** Llama-3.2-1B-Instruct Q8_0, 32k, with the same three measurements as experiments 03 and 04. Per-head is the default from here on.

```sh
needle() { env "$@" build/bin/needle $M $T 32768 f16 3 state; }
needle GGML_QUEST_BUDGET=0;  needle GGML_QUEST_BUDGET=2048;  needle GGML_QUEST_BUDGET=1024
needle GGML_QUEST_BUDGET=512; needle GGML_QUEST_BUDGET=2048 GGML_QUEST_PER_HEAD=0
P="build/bin/llama-perplexity -m $M -f $T -c 32768 -fa on -t 8 --chunks 2"
$P --kl-divergence-base dense-32k.kld
GGML_QUEST_BUDGET=2048 $P -ub 1 --kl-divergence-base dense-32k.kld --kl-divergence
GGML_QUEST_BUDGET=1024 $P -ub 1 --kl-divergence-base dense-32k.kld --kl-divergence
$P -ctk q8_0 -ctv q8_0 --kl-divergence-base dense-32k.kld --kl-divergence
B="build/bin/llama-bench -m $M -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 3"
GGML_QUEST_BUDGET=0 $B; GGML_QUEST_BUDGET=2048 $B; GGML_QUEST_BUDGET=1024 $B; $B -ctk q8_0 -ctv q8_0
```

## Results

Dense PPL is 16.64.

| Config | Needle | KLD | KLD p99 | Same top-1 | PPL ratio | tg64 8k | tg64 32k |
|---|---|---|---|---|---|---|---|
| dense f16 | 15/15 | — | — | — | — | 7.17 | 1.82 |
| per-head b2048 | 15/15 | 0.057 | 0.31 | 86.4% | 0.991 | 14.5 | 7.62 (4.2×) |
| per-head b1024 | 14/15 | 0.099 | 0.57 | 82.3% | 1.019 | 16.7 | 8.22 |
| per-head b512 | 13/15 | | | | | | |
| shared b2048 | 15/15 | | | | | | |
| q8_0 KV | | 0.0025 | 0.016 | 97.0% | 1.002 | 9.56 | 3.00 |

The needle misses drop a digit (for example 26316 → "2634", 38182 → "3812").

## Conclusion

The results carry over: at budget 2048 and 32k, KLD is 0.057 (Qwen: 0.053), retrieval is perfect, and decode is 4.2× faster. Llama tolerates q8_0 KV even better than Qwen, which leads to experiment 06.

## Files

- `run.sh`: the original script. `results.txt`: its summary output.
- `logs/`: raw output, including needle transcripts.
