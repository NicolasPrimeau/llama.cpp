#!/bin/bash
cd ~/llama.cpp
GGML_QUEST_PER_HEAD=1 GGML_QUEST_BUDGET=2048 build/bin/llama-perplexity -m /home/nprimeau/models/qwen2.5-0.5b-instruct-q8_0.gguf -f /home/nprimeau/projects/certified-attention/data/pride.txt -c 32768 -fa on -t 8 --chunks 2 -ub 1 --kl-divergence-base /home/nprimeau/llama-eval/dense-32k.kld --kl-divergence > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-perhead-2048.log 2>&1
echo "per-head budget 2048 rc=$? $(grep -E 'Mean PPL\(Q\)/|Mean    KLD|Same top p' /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/q32-perhead-2048.log | tr -s ' ' | tr '\n' ';')" > /tmp/claude-1000/-home-nprimeau-Steward/e207fc45-526a-4b61-a37e-698c0a2b785c/scratchpad/perhead-result.txt
