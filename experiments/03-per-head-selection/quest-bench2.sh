#!/bin/bash
cd ~/llama.cpp
B="build/bin/llama-bench -m /home/nprimeau/models/qwen2.5-0.5b-instruct-q8_0.gguf -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 3 -o md"
for cfg in "0 0" "2048 0" "2048 1" "1024 1"; do
  set -- $cfg
  GGML_QUEST_BUDGET=$1 GGML_QUEST_PER_HEAD=$2 $B > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/bench2-$1-$2.log 2>&1
  echo "budget=$1 per_head=$2 rc=$?"; grep tg64 /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/bench2-$1-$2.log
done > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/bench2-results.txt
