# 04: Needle test and q8_0/q4_0 KV (Qwen)

**Questions:**
1. KLD is an average. Does Quest still retrieve one specific fact from deep in a 32k context?
2. How does Quest compare with the existing way of making long contexts cheaper, a quantized KV cache?

**Setup:** Qwen2.5-0.5B, 32k context. The needle test uses 5 depths × 3 trials, and its prefill is cached in a state dir so later runs only redo decode. KV quantization quality is measured against the dense f16 baseline from experiment 02.

```sh
needle() { env "$@" build/bin/needle $M $T 32768 f16 3 needle-state; }
needle GGML_QUEST_BUDGET=0
needle GGML_QUEST_BUDGET=2048 GGML_QUEST_PER_HEAD=1
needle GGML_QUEST_BUDGET=1024 GGML_QUEST_PER_HEAD=1
needle GGML_QUEST_BUDGET=512  GGML_QUEST_PER_HEAD=1
needle GGML_QUEST_BUDGET=2048 GGML_QUEST_PER_HEAD=0
for kv in q8_0 q4_0; do
  build/bin/llama-perplexity -m $M -f $T -c 32768 -fa on -t 8 --chunks 2 -ctk $kv -ctv $kv \
      --kl-divergence-base dense-32k.kld --kl-divergence
  build/bin/llama-bench -m $M -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 3 -ctk $kv -ctv $kv
done
```

## Results

| Needle, 32k | Found |
|---|---|
| dense | 15/15 |
| per-head b2048 | 15/15 |
| per-head b1024 | 14/15 |
| per-head b512 | 14/15 |
| shared b2048 | 14/15 |

| KV cache | KLD | Same top-1 | tg64 8k | tg64 32k |
|---|---|---|---|---|
| f16 dense | — | — | 26.9 | 6.59 |
| q8_0 | 0.011 | 93.8% | 24.7 | 8.95 |
| q4_0 | **3.27** | 18.3% | 23.9 | 8.49 |
| f16 + Quest per-head b2048 | 0.053 | 85.5% | 39.0 | 24.9 |

## Conclusion

Per-head b2048 keeps exact retrieval at 32k. Smaller budgets fail occasionally, and the failures are dropped or garbled digits rather than wrong facts.

q8_0 KV is nearly lossless but only 1.4× faster at 32k. q4_0 KV breaks this model outright (PPL ×25). So quantization and Quest address different costs: q8_0 halves the bytes per token, while Quest cuts the number of tokens read. Experiment 06 combines them.

## Files

- `verify.sh`: the original script. `results.txt`: its summary output.
- `logs/`: needle transcripts with each trial's answer, KV-quant KLD and bench output.
