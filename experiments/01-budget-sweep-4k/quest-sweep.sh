#!/bin/bash
cd ~/llama.cpp
P="build/bin/llama-perplexity -m /home/nprimeau/models/qwen2.5-0.5b-instruct-q8_0.gguf -f /home/nprimeau/projects/certified-attention/data/pride.txt -c 4096 -ub 1 -fa on -t 8 --chunks 4"
$P --kl-divergence-base /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/dense-4k.kld > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/sweep-dense.log 2>&1
echo "dense: $(grep 'Final estimate' /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/sweep-dense.log)"
for b in 256 512 1024; do
  GGML_QUEST_BUDGET=$b $P --kl-divergence-base /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/dense-4k.kld --kl-divergence > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/sweep-$b.log 2>&1
  echo "budget $b: $(grep -E 'Mean PPL\(Q\)|Mean    KLD|Same top p' /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/sweep-$b.log | tr -s ' ' | tr '\n' ';')"
done
