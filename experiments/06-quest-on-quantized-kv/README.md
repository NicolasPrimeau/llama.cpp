# 06: Quest on a quantized KV cache

**Question:** Quest originally required an f16/f32 K cache. Can it read q8_0 (or any type with `to_float`), and do the memory and speed gains stack?

**Change:** Page summaries dequantize each K row into thread scratch before updating min/max. Selected pages are read through the type's `vec_dot` (K) and `to_float` (V). Commit "Quest sparse decode on quantized K/V".

**Setup:** Correctness uses Qwen2.5-0.5B at 4k with `-t 4`, against the dense f16 4k baseline from experiment 01. "old" is the branch before this change and "new" is after it. Speed is measured on both models at 8k/32k.

```sh
A="-m $M -f $T -c 4096 -ub 1 -fa on -t 4 --chunks 4 --kl-divergence-base dense-4k.kld --kl-divergence"
GGML_QUEST_BUDGET=0    old/bin/llama-perplexity $A -ctk q8_0 -ctv q8_0
GGML_QUEST_BUDGET=0    build/bin/llama-perplexity $A -ctk q8_0 -ctv q8_0
GGML_QUEST_BUDGET=1024 build/bin/llama-perplexity $A -ctk q8_0 -ctv q8_0
GGML_QUEST_BUDGET=2048 build/bin/llama-perplexity $A -ctk q8_0 -ctv q8_0
GGML_QUEST_BUDGET=1024 build/bin/llama-perplexity $A
B="build/bin/llama-bench -m $M -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 3"
GGML_QUEST_BUDGET=0 $B -ctk q8_0 -ctv q8_0; GGML_QUEST_BUDGET=2048 $B -ctk q8_0 -ctv q8_0
GGML_QUEST_BUDGET=2048 $B; GGML_QUEST_BUDGET=0 $B
```

## Results

**Correctness (Qwen, 4k, KLD vs dense f16):**

| Config | KLD | Same top-1 |
|---|---|---|
| q8_0 dense, old | 0.0147 | 92.9% |
| q8_0 dense, new | 0.0150 | 93.0% |
| q8_0 + Quest b1024 | 0.0376 | 88.1% |
| q8_0 + Quest b2048 | 0.0191 | 92.1% |
| f16 + Quest b1024 | 0.0136 | 92.5% |

The dense q8_0 results match between old and new. A regression check of f16 shared b1024 gives KLD 0.02964 against 0.02953 before the change.

**Speed (tg64, t/s):**

| Config | Qwen 8k | Qwen 32k | Llama 8k | Llama 32k |
|---|---|---|---|---|
| f16 dense | 26.7 | 6.69 | 7.52 | 1.82 |
| q8_0 dense | 24.6 | 8.44 | 9.63 | 3.11 |
| f16 + Quest b2048 | 38.0 | 24.4 | 14.7 | 7.52 |
| q8_0 + Quest b2048 | **39.3** | **26.8 (4.0×)** | **17.0** | **7.96 (4.4×)** |

**A mistake along the way:** The first version also sent dense quantized decode through the split-KV path. That was faster at 8k but made Llama q8_0 slower at 32k (3.11 → 1.95 t/s). The final commit keeps dense quantized decode on the existing per-head path, and only Quest takes the new path. Re-measured on the final code, Llama at 32k: q8_0 dense 3.15 t/s, q8_0 + Quest b2048 **9.60 t/s (5.3× vs f16 dense)**. The rest of the speed table is from the earlier build. Dense f16 and Quest-on-f16 paths are unchanged by the fix, and Qwen was not re-measured.

## Conclusion

The gains stack. q8_0 halves cache memory, and Quest on q8_0 is slightly faster than Quest on f16 because the selected pages are half the bytes. This is the recommended setting: `-ctk q8_0 -ctv q8_0` with `GGML_QUEST_BUDGET=2048`.
