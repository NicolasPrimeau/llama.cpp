#!/bin/bash
# Quest on CUDA: correctness, quality and speed. Each step appends one line to results.txt.
#
# usage: experiments/tools/run-cuda.sh <Llama-3.2-1B-Instruct-Q8_0.gguf> [8B model .gguf]
# Builds this checkout into build-cuda/ for the local GPU (CUDA_ARCH=75 etc. to override). Logs go to experiments/out/, the summary to experiments/out/results.txt.
#
# Step 1 checks the CUDA port against the CPU numbers measured on the same 1B model (32k context, budget 2048):
#   KLD vs dense 0.057, same top-1 86.4%, needle 15/15. A broken selection or gather shows up as KLD >> 0.1.
set -u
D=$(dirname "$(realpath "$0")"); L=$(realpath "$D/../.."); W=$L/experiments/out; R=$W/results.txt; T=$D/pride.txt
M1=$(realpath "$1"); M8=${2:+$(realpath "$2")}
mkdir -p "$W"

build() {
    cmake -S "$L" -B "$L/build-cuda" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-native} -DCMAKE_BUILD_TYPE=Release > "$W/cmake.log" 2>&1 &&
    cmake --build "$L/build-cuda" -j --target llama-perplexity llama-bench >> "$W/cmake.log" 2>&1 &&
    g++ -O2 -std=c++17 "$D/needle.cpp" -I"$L/include" -I"$L/ggml/include" -L"$L/build-cuda/bin" \
        -lllama -lggml -lggml-base -Wl,-rpath,"$L/build-cuda/bin" -o "$W/needle"
}
build || { echo "build failed, see $W/cmake.log"; exit 1; }
B=$L/build-cuda/bin

log() { echo "$*" | tee -a "$R"; }

kld() { # label model base-file env... -- extra args
    local l=$1 m=$2 base=$3 e=$4; shift 4
    env $e "$B/llama-perplexity" -m "$m" -f "$T" -c 32768 --chunks 2 -fa on -ngl 99 "$@" \
        --kl-divergence-base "$base" --kl-divergence > "$W/kld-$l.log" 2>&1
    log "kld $l: $(grep -E 'Mean PPL\(Q\)/|Mean    KLD|Same top p' "$W/kld-$l.log" | tr -s ' ' | tr '\n' ';')"
}
base() { # label model
    "$B/llama-perplexity" -m "$2" -f "$T" -c 32768 --chunks 2 -fa on -ngl 99 --kl-divergence-base "$W/dense-$1.kld" \
        > "$W/base-$1.log" 2>&1
    log "dense $1: $(grep 'Final estimate' "$W/base-$1.log")"
}
needle() { # label model env...
    local l=$1 m=$2; shift 2
    mkdir -p "$W/state-$(basename "$m" .gguf)"
    env "$@" "$W/needle" "$m" "$T" 32768 f16 3 "$W/state-$(basename "$m" .gguf)" > "$W/needle-$l.log" 2>&1
    log "needle $l: $(tail -1 "$W/needle-$l.log") fails: $(grep 'ok=0' "$W/needle-$l.log" | tr '\n' ' ')"
}
bench() { # label model env... -- extra args
    local l=$1 m=$2 e=$3; shift 3
    env $e "$B/llama-bench" -m "$m" -fa 1 -ngl 99 -p 0 -n 64 -d 8192,32768 -r 3 -o md "$@" > "$W/bench-$l.log" 2>&1
    log "speed $l: $(grep tg64 "$W/bench-$l.log" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr '\n' ';')"
}

log "== $(date) $(git -C "$L" log --oneline -1)"

# 0. memory errors in the new kernels, on a short decode (needs compute-sanitizer from the CUDA toolkit)
if command -v compute-sanitizer > /dev/null; then
    GGML_QUEST_BUDGET=512 compute-sanitizer --tool memcheck --error-exitcode 1 \
        "$B/llama-bench" -m "$M1" -fa 1 -ngl 99 -p 0 -n 4 -d 4096 -r 1 > "$W/memcheck.log" 2>&1
    log "memcheck: rc=$? $(grep -c 'Invalid\|ERROR SUMMARY: [1-9]' "$W/memcheck.log") problem lines"
fi

# 1. cross-check against the CPU results on the 1B model (-ub 1 so every token goes through decode)
base 1b "$M1"
kld 1b-quest2048     "$M1" "$W/dense-1b.kld" GGML_QUEST_BUDGET=2048 -ub 1
kld 1b-quest2048-q8  "$M1" "$W/dense-1b.kld" GGML_QUEST_BUDGET=2048 -ub 1 -ctk q8_0 -ctv q8_0
kld 1b-q8            "$M1" "$W/dense-1b.kld" GGML_QUEST_BUDGET=0    -ub 1 -ctk q8_0 -ctv q8_0
needle 1b-dense      "$M1" GGML_QUEST_BUDGET=0
needle 1b-quest2048  "$M1" GGML_QUEST_BUDGET=2048

# 2. decode speed
bench 1b-dense       "$M1" GGML_QUEST_BUDGET=0
bench 1b-quest2048   "$M1" GGML_QUEST_BUDGET=2048
bench 1b-q8          "$M1" GGML_QUEST_BUDGET=0    -ctk q8_0 -ctv q8_0
bench 1b-quest2048-q8 "$M1" GGML_QUEST_BUDGET=2048 -ctk q8_0 -ctv q8_0

# 3. the same on a model where the KV cache is a large share of decode traffic
if [ -n "$M8" ]; then
    bench 8b-dense        "$M8" GGML_QUEST_BUDGET=0
    bench 8b-quest2048    "$M8" GGML_QUEST_BUDGET=2048
    bench 8b-q8           "$M8" GGML_QUEST_BUDGET=0    -ctk q8_0 -ctv q8_0
    bench 8b-quest2048-q8 "$M8" GGML_QUEST_BUDGET=2048 -ctk q8_0 -ctv q8_0
    base 8b "$M8"
    kld 8b-quest2048    "$M8" "$W/dense-8b.kld" GGML_QUEST_BUDGET=2048 -ub 1
    kld 8b-quest2048-q8 "$M8" "$W/dense-8b.kld" GGML_QUEST_BUDGET=2048 -ub 1 -ctk q8_0 -ctv q8_0
    needle 8b-dense     "$M8" GGML_QUEST_BUDGET=0
    needle 8b-quest2048 "$M8" GGML_QUEST_BUDGET=2048
fi
log "done"
