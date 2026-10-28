#!/bin/bash
cd ~/llama.cpp
P="build/bin/llama-perplexity -m /home/nprimeau/models/qwen2.5-0.5b-instruct-q8_0.gguf -f /home/nprimeau/projects/certified-attention/data/pride.txt -c 32768 -fa on -t 8 --chunks 2"
$P --kl-divergence-base /home/nprimeau/llama-eval/dense-32k.kld > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-dense.log 2>&1
echo "dense rc=$? $(grep 'Final estimate' /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-dense.log)" >> /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-results.txt
for b in 1024 2048 4096; do
  GGML_QUEST_BUDGET=$b $P -ub 1 --kl-divergence-base /home/nprimeau/llama-eval/dense-32k.kld --kl-divergence > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-$b.log 2>&1
  echo "budget $b rc=$? $(grep -E 'Mean PPL\(Q\)/|Mean    KLD|99.0%   KLD|Same top p' /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-$b.log | tr -s ' ' | tr '\n' ';')" >> /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-results.txt
done
echo "sweep done" >> /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-results.txt
