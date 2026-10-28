# 07: CUDA port on an RTX 4080 (not run yet)

**Question:** Is the CUDA port (commit "ggml-cuda : Quest sparse decode (opt-in)") correct, and how much faster is decode on a GPU?

**Design:** For each decode step:
1. Score kernel: bounds every page per query head and refreshes stale summaries (F16, Q8_0 and Q4_0 K are supported).
2. Select kernel: a top-k threshold search per head.
3. Gather kernel: copies each head's selected pages into a dense buffer.
4. The existing vec flash-attention kernel runs on those buffers.

The port has been compiled for sm_89 with CUDA 13.4 but has never run on a GPU.

**Run:**

```sh
experiments/tools/run-cuda.sh ~/models/Llama-3.2-1B-Instruct-Q8_0.gguf [~/models/<8B>.gguf]
```

The script builds `build-cuda/` and then runs, in order:
1. compute-sanitizer memcheck at budget 512, if `compute-sanitizer` is installed.
2. The 1B KLD and needle cross-check against the CPU numbers.
3. 1B speed for dense, Quest, q8_0 and Quest + q8_0.
4. If an 8B model is given, the same speed runs, KLD and needle on it.

The summary goes to `experiments/out/results.txt`. Logs and KLD files (about 20 GB) go to `experiments/out/`, which is gitignored.

## Expected

| Check | Pass if |
|---|---|
| memcheck | 0 errors |
| 1B KLD, b2048 | ≈ 0.057 (CPU value); a selection or gather bug shows up as > 0.1 |
| 1B needle, b2048 | 15/15 |
| 8B tg64 at 32k | roughly 1.3–2× over dense |

The GPU speedup should be smaller than on the CPU. GPU decode is bandwidth-bound and already fast. The gather step also re-reads K once per query head, which hurts more at high GQA ratios. If the numbers come out low, the next step is to read pages in place inside the vec kernel instead of gathering them.

## Results

(to fill in from `experiments/out/results.txt`)
