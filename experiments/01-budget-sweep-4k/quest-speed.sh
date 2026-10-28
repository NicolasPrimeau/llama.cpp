#!/bin/bash
cd ~/llama.cpp
GGML_QUEST_BUDGET=2048 build/bin/llama-perplexity -m /home/nprimeau/models/qwen2.5-0.5b-instruct-q8_0.gguf -f /home/nprimeau/projects/certified-attention/data/pride.txt -c 4096 -ub 1 -fa on -t 8 --chunks 4 --kl-divergence-base /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/dense-4k.kld --kl-divergence > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/sweep-2048.log 2>&1
echo "budget 2048: $(grep -E 'Mean PPL\(Q\)/|Mean    KLD|Same top p' /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/sweep-2048.log | tr -s ' ' | tr '\n' ';')"
for b in 0 512 1024 2048; do
  echo "== bench budget $b"
  GGML_QUEST_BUDGET=$b build/bin/llama-bench -m /home/nprimeau/models/qwen2.5-0.5b-instruct-q8_0.gguf -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 2 -o md 2>&1 | grep -E "tg64|error"
done
