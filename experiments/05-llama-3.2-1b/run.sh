#!/bin/bash
# Quest verification suite on Llama-3.2-1B-Instruct: needle, KLD vs dense at 32k, decode speed.
cd ~/llama.cpp
D=/home/nprimeau/llama-eval/llama1b; R=$D/results.txt
M=/home/nprimeau/models/Llama-3.2-1B-Instruct-Q8_0.gguf; T=/home/nprimeau/projects/certified-attention/data/pride.txt
N=/home/nprimeau/projects/kvlink/needle/needle
mkdir -p $D/state
needle() { local l=$1; shift
  env "$@" $N $M $T 32768 f16 3 $D/state > $D/needle-$l.log 2>&1
  echo "needle $l: $(tail -1 $D/needle-$l.log) fails: $(grep 'ok=0' $D/needle-$l.log | tr '\n' ' ')" >> $R; }
needle dense GGML_QUEST_BUDGET=0
needle perhead2048 GGML_QUEST_BUDGET=2048
needle perhead1024 GGML_QUEST_BUDGET=1024
needle perhead512 GGML_QUEST_BUDGET=512
needle shared2048 GGML_QUEST_BUDGET=2048 GGML_QUEST_PER_HEAD=0
P="build/bin/llama-perplexity -m $M -f $T -c 32768 -fa on -t 8 --chunks 2"
$P --kl-divergence-base $D/dense-32k.kld > $D/kld-dense.log 2>&1
echo "dense: $(grep 'Final estimate' $D/kld-dense.log)" >> $R
kld() { local l=$1 e=$2; shift 2
  env $e $P "$@" --kl-divergence-base $D/dense-32k.kld --kl-divergence > $D/kld-$l.log 2>&1
  echo "kld $l: $(grep -E 'Mean PPL\(Q\)/|Mean    KLD|99.0%   KLD|Same top p' $D/kld-$l.log | tr -s ' ' | tr '\n' ';')" >> $R; }
kld perhead2048 GGML_QUEST_BUDGET=2048 -ub 1
kld perhead1024 GGML_QUEST_BUDGET=1024 -ub 1
kld q8_0 GGML_QUEST_BUDGET=0 -ctk q8_0 -ctv q8_0
bench() { local l=$1 e=$2; shift 2
  env $e build/bin/llama-bench -m $M -fa 1 -t 8 -p 0 -n 64 -d 8192,32768 -r 3 -o md "$@" > $D/bench-$l.log 2>&1
  echo "speed $l: $(grep tg64 $D/bench-$l.log | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr '\n' ';')" >> $R; }
bench dense GGML_QUEST_BUDGET=0
bench perhead2048 GGML_QUEST_BUDGET=2048
bench perhead1024 GGML_QUEST_BUDGET=1024
bench q8_0 GGML_QUEST_BUDGET=0 -ctk q8_0 -ctv q8_0
echo "done" >> $R
