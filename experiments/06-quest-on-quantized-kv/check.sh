#!/bin/bash
# Correctness of split-KV decode on quantized KV and of Quest over q8_0, against the dense f16 4k baseline.
D=/home/nprimeau/llama-eval/qq; R=$D/results.txt
A="-m /home/nprimeau/models/qwen2.5-0.5b-instruct-q8_0.gguf -f /home/nprimeau/projects/certified-attention/data/pride.txt -c 4096 -ub 1 -fa on -t 4 --chunks 4 --kl-divergence-base /home/nprimeau/llama-eval/dense-4k.kld --kl-divergence"
run() { local l=$1 bin=$2 e=$3; shift 3
  local t0=$(date +%s)
  env $e $bin/bin/llama-perplexity $A "$@" > $D/$l.log 2>&1
  echo "$l rc=$? $(( $(date +%s)-t0 ))s: $(grep -E 'Mean PPL\(Q\)/|Mean    KLD|Same top p' $D/$l.log | tr -s ' ' | tr '\n' ';')" >> $R; }
run q8-old  /home/nprimeau/llama.cpp/build    GGML_QUEST_BUDGET=0    -ctk q8_0 -ctv q8_0
run q8-new  /home/nprimeau/llama.cpp-qq/build GGML_QUEST_BUDGET=0    -ctk q8_0 -ctv q8_0
run q8-q1024 /home/nprimeau/llama.cpp-qq/build GGML_QUEST_BUDGET=1024 -ctk q8_0 -ctv q8_0
run q8-q2048 /home/nprimeau/llama.cpp-qq/build GGML_QUEST_BUDGET=2048 -ctk q8_0 -ctv q8_0
run f16-q1024 /home/nprimeau/llama.cpp-qq/build GGML_QUEST_BUDGET=1024
echo done >> $R
