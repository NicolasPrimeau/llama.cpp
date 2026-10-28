#!/bin/bash
# Decode speed at 8k/32k: q8_0 split-KV vs the old per-head path, and Quest on f16 vs q8_0. Runs after the Llama suite.
while systemctl --user is-active -q quest-llama1b; do sleep 60; done
D=/home/nprimeau/llama-eval/qq; R=$D/bench-results.txt
OLD=/home/nprimeau/llama.cpp/build/bin/llama-bench; NEW=/home/nprimeau/llama.cpp-qq/build/bin/llama-bench
bench() { local l=$1 bin=$2 m=$3 e=$4; shift 4
  env $e $bin -m $m -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 3 -o md "$@" > $D/bench-$l.log 2>&1
  echo "$l: $(grep tg64 $D/bench-$l.log | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr '\n' ';')" >> $R; }
for m in qwen2.5-0.5b-instruct-q8_0 Llama-3.2-1B-Instruct-Q8_0; do
  M=/home/nprimeau/models/$m.gguf
  bench $m-q8-old   $OLD $M GGML_QUEST_BUDGET=0    -ctk q8_0 -ctv q8_0
  bench $m-q8-new   $NEW $M GGML_QUEST_BUDGET=0    -ctk q8_0 -ctv q8_0
  bench $m-q8-q2048 $NEW $M GGML_QUEST_BUDGET=2048 -ctk q8_0 -ctv q8_0
  bench $m-f16-q2048 $NEW $M GGML_QUEST_BUDGET=2048
  bench $m-f16-dense $NEW $M GGML_QUEST_BUDGET=0
done
echo done >> $R
