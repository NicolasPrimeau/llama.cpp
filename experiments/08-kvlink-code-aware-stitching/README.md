# 08: Code-aware KV stitching (kvlink), negative result

**Question:** Coding agents re-send the same files over and over. Suppose each file's KV cache is computed once on its own, and then caches are concatenated (A then B). B's cache never saw A, so it is wrong in places. CacheBlend fixes this by recomputing the B tokens whose KV deviates most. **Can we use the code itself to choose which tokens to recompute, namely B's lines that reference symbols defined in A?** That would need no deviation pass at all.

**Setup:** This is a Python/transformers experiment that doesn't use this llama.cpp branch (`kvlink.py`, needs `torch` and `transformers`).

Everything is computed in a single forward pass with a custom attention mask:
- A is causal within itself.
- B is causal within B only, at its final positions.
- The selected B tokens are re-run as fresh copies that can see A.
- A 256-token probe at the end sees everything.

The ground truth is a plain causal pass over A + B + probe, and the metric is KL(truth ‖ stitched) on the probe. A sanity check confirmed that the masked pass matches the plain pass (max |Δlogit| 5e-5 with all tokens selected). The model was Qwen2.5-0.5B (fp32, eager attention), on 8 import-linked file pairs (6 from Nimbus, 1 each from Longshot and Artel), with A ≤ 1500 tokens and B 400–1500 tokens.

```sh
python kvlink.py --check --repos ~/projects/Artel     # masked pass == plain pass
python kvlink.py --pairs 15 --repos ~/projects/{Artel,Longshot,Yeet,Polaris,Nimbus,serving-lab,certified-attention}
# 8 pairs met the minimum of 3 reference lines in B
```

## Results

Mean over 8 pairs. Every method recomputes the same number of tokens (about 7.6% of B, the size of the code-referencing lines), except where noted.

| Recompute | KLD | Top-1 |
|---|---|---|
| nothing | 0.0295 | 94.4% |
| random tokens | 0.0281 | 94.5% |
| **code-referencing lines** | 0.0274 | 94.4% |
| code-referencing identifiers only (1.3%) | 0.0291 | 94.5% |
| CacheBlend (KV deviation) | 0.0151 | 96.2% |
| oracle (best possible) | 0.0117 | 96.8% |
| sink: the first tokens of B (0.4%) | 0.0179 | 96.0% |
| sink + code lines | 0.0161 | 96.0% |
| CacheBlend at 15% | 0.0119 | 96.8% |
| oracle at 15% | 0.0064 | 97.6% |

## Conclusion

Selecting tokens from the code structure is no better than random. The error does not sit on the lines that mention A. Most of it is spread out, and much of what remains is the attention sink at B's start: recomputing 0.4% of tokens there recovers more than recomputing every code-referencing line. KV deviation (CacheBlend) remains the right signal. We dropped the idea and went back to Quest.

## Files

- `kvlink.py`: the experiment. `check.log`: the masked-pass sanity check.
- `results.jsonl`: one row per file pair (the table above averages it).
- `results-run1.jsonl`: an earlier partial run, from before the sink controls were added.
