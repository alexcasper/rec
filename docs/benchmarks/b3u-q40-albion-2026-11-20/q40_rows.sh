#!/usr/bin/env bash
# rec-b3u pure-q4_0 rows on R9700 from the flint-int4 f16 GGUF; existing albion build (~/llama.cpp/build), p1 command pattern.
set -u
cd ~/rec-ve/b3u; BIN=~/llama.cpp/build/bin
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
B=/sys/class/drm/card1/device
busy(){ cat $B/gpu_busy_percent 2>/dev/null || { [ "$(cat $B/power/runtime_status)" = suspended ] && echo 0 || echo 100; }; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(busy)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; echo "GATE TIMEOUT"; return 1; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(busy) vram_used=$(( $(cat $B/mem_info_vram_used)/1048576 ))MiB gpu_procs=$(pgrep -fc "[l]lama-(server|bench|cli)|[f]lint_full|[m]id_mk")"; }
st "quantize --pure q4_0"
/usr/bin/env time -f "quantize wall %e s" $BIN/llama-quantize --pure flint-int4-f16.gguf flint-int4-Q4_0-pure.gguf q4_0 14 > quantize_q40.log 2>&1
st "quantize exit $? $(grep -E "quant size|wall" quantize_q40.log | tr -s " " | tr "\n" ";")"
for r in d512 p512; do
  gate || exit 1
  st "bench $r start"
  if [ $r = d512 ]; then $BIN/llama-bench -m flint-int4-Q4_0-pure.gguf -ngl 99 -fa on -p 0 -n 32 -d 512 -r 5 -o jsonl > q40_$r.jsonl 2> q40_$r.err
  else $BIN/llama-bench -m flint-int4-Q4_0-pure.gguf -ngl 99 -fa on -p 512 -n 0 -r 3 -o jsonl > q40_$r.jsonl 2> q40_$r.err; fi
  st "bench $r exit $? $(python3 -c "import json; [print(\"n_prompt=%s n_gen=%s depth=%s avg_ts=%.2f stddev=%.2f build=%s/%s dev=%s\" % (j[\"n_prompt\"], j[\"n_gen\"], j.get(\"n_depth\"), j[\"avg_ts\"], j[\"stddev_ts\"], j[\"build_commit\"], j[\"build_number\"], j.get(\"devices\", j.get(\"gpu_info\")))) for j in map(json.loads, open(\"q40_$r.jsonl\"))]")"
done
sha256sum flint-int4-f16.gguf flint-int4-Q4_0-pure.gguf > gguf.sha256
st ALL_DONE
