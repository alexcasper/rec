#!/usr/bin/env bash
set -u
cd ~/rec-ve/xh7
export OUT=bench_xh7_q4.jsonl LOG=bench_xh7_q4.log  # set before sourcing: the sourced block truncates $OUT
source <(sed -n "/^H=/,/^mid_bench() {/p" bench_xh7.sh | head -n -1)
mid_bench() { gate || { rec "$S" "\"run\":\"mid\",\"model\":\"$1\",\"depth\":$2,\"error\":\"gate timeout\""; return; }
  local pre post j; pre=$(busy_slots 8080 8090 8091)
  j=$(timeout 600 $BIN/llama-bench -m "$1" -ngl 99 -fa on -p 0 -n 32 -d "$2" -r 5 -o jsonl 2>>$LOG | tail -1)
  post=$(busy_slots 8080 8090 8091)
  rec "$S" "\"run\":\"mid\",\"model\":\"$1\",\"depth\":$2,\"slots_busy_pre\":$pre,\"slots_busy_post\":$post,\"bench\":${j:-null}"; }
trap "restart_qwen; echo EXIT $(date -u +%T) qwen=$(systemctl --user is-active llama-qwen) >> $LOG" EXIT
systemctl --user stop llama-qwen; sleep 3; S=S0; W=6
for m in mid-pure-q4_0.gguf mid-ud-iq3xxs-lmq4.gguf mid-ud-iq3xxs-lmq4.gguf mid-pure-q4_0.gguf; do for d in 512 8192; do mid_bench $m $d; done; done
echo DONE >> $LOG
