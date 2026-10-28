#!/bin/bash
cd ~/llama.cpp
M=/home/nprimeau/models/qwen2.5-0.5b-instruct-q8_0.gguf; T=/home/nprimeau/projects/certified-attention/data/pride.txt; N=/home/nprimeau/projects/kvlink/needle/needle
run_needle() { # label, then env assignments
  local label=$1; shift
  env "$@" $N $M $T 32768 f16 3 /home/nprimeau/llama-eval/needle-state > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/needle-$label.log 2>&1
  echo "needle $label: $(tail -1 /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/needle-$label.log) fails: $(grep -c 'ok=0' /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/needle-$label.log)" >> /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/verify-results.txt
}
run_needle dense GGML_QUEST_BUDGET=0
run_needle perhead2048 GGML_QUEST_BUDGET=2048 GGML_QUEST_PER_HEAD=1
run_needle perhead1024 GGML_QUEST_BUDGET=1024 GGML_QUEST_PER_HEAD=1
run_needle perhead512 GGML_QUEST_BUDGET=512 GGML_QUEST_PER_HEAD=1
run_needle shared2048 GGML_QUEST_BUDGET=2048 GGML_QUEST_PER_HEAD=0
P="build/bin/llama-perplexity -m $M -f $T -c 32768 -fa on -t 8 --chunks 2"
for kv in q8_0 q4_0; do
  $P -ctk $kv -ctv $kv --kl-divergence-base /home/nprimeau/llama-eval/dense-32k.kld --kl-divergence > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/kvq-$kv.log 2>&1
  echo "kv $kv quality: $(grep -E 'Mean PPL\(Q\)/|Mean    KLD|Same top p' /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/kvq-$kv.log | tr -s ' ' | tr '\n' ';')" >> /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/verify-results.txt
  build/bin/llama-bench -m $M -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 3 -ctk $kv -ctv $kv -o md > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/kvbench-$kv.log 2>&1
  echo "kv $kv speed: $(grep tg64 /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/kvbench-$kv.log | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr '\n' ';')" >> /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/verify-results.txt
done
echo "verify done" >> /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/verify-results.txt
