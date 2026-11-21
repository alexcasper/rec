#!/usr/bin/env bash
# flint-y02/r5-reconcile: full-Flint :8080 decode provenance sweep, step-0 R5 protocol.
# Request shape per docs/benchmarks/xh7-step0/README.md §2: /completion, random
# token-id prompt, cache_prompt off, 64 greedy tokens, timings from response.
# Gate: gpu_busy < 5 on 3 samples @0.3s; 0 busy slots on :8080/:8090 (:8091 is
# down per alex's directive — recorded as such). Cotenant state logged per line.
set -uo pipefail
OUT="$HOME/flint-mid/r5-reconcile-$(date -u +%Y%m%dT%H%M%S).jsonl"
TS=100.98.73.29

snapshot () {  # cotenant state line
  date -u +%FT%TZ
  echo "  gpu_busy: $(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | tr '\n' ' ')"
  echo "  vram_used_MiB: $(awk '/vram_used/ {print $2}' /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)"
  echo "  MemAvailable_MiB: $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
  echo "  llama-servers: $(pgrep -ax llama-server | sed -E 's/.*(--port [0-9]+).*-?(host [0-9.]+)?/-\1/' | tr '\n' ' ')"
  echo "  8990_cpu_pct: $(ps -o %cpu= -p $(pgrep -f 'port 8990' | head -1) 2>/dev/null | tr -d ' ')"
}

gate () {
  for i in 1 2 3; do
    b=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)
    [ "${b:-100}" -lt 5 ] || { echo "GATE-FAIL gpu_busy=$b"; exit 1; }
    sleep 0.3
  done
  for port in 8080 8090; do
    busy=$(curl -s -m 3 "http://$TS:$port/slots" | grep -c '"is_processing": true')
    [ "$busy" = "0" ] || { echo "GATE-FAIL slot busy on $port"; exit 1; }
  done
  curl -s -m 3 "http://$TS:8091/slots" >/dev/null 2>&1 && echo "NOTE 8091 answering (should be down)" || true
}

gen_prompt () {  # $1 depth, fixed seed per depth -> identical prompt across runs
  python3 -c "
import json,random,sys
random.seed($1)
print(json.dumps([random.randrange(3,150144) for _ in range($1)]))"
}

run_row () {  # $1 label, $2 depth, $3 n_runs
  d=$2
  ids=$(gen_prompt "$d")
  for i in $(seq 1 "$3"); do
    gate || { echo "$1 run$i GATE-FAIL"; return 1; }
    python3 - "$ids" "$d" <<'EOF' > /tmp/r5-payload.json
import json,sys
ids=json.loads(sys.argv[1])
print(json.dumps({"prompt":ids,"n_predict":64,"temperature":0,"cache_prompt":False,"stream":False}))
EOF
    ts=$(date -u +%FT%TZ)
    resp=$(curl -s -m 600 "http://$TS:8080/completion" -H "Content-Type: application/json" -d @/tmp/r5-payload.json)
    echo "$resp" | python3 -c "
import json,sys
r=json.load(sys.stdin)
t=r.get('timings',{})
print(json.dumps({'label':'$1','depth':$d,'run':$i,'utc':'$ts',
 'prompt_n':t.get('prompt_n'),'prompt_tps':round(t.get('prompt_per_second',0),2),
 'pred_n':t.get('predicted_n'),'pred_tps':round(t.get('predicted_per_second',0),2)}))
" >> "$OUT"
    echo "  $1 run$i: $(tail -1 "$OUT" | head -c 200)"
    sleep 2
  done
}

{
echo "=== R5 reconcile start $(date -u +%FT%TZ) ==="
snapshot
echo "--- rows while 8990 CPU qwen3vl lives (decode-focused per coordinator) ---"
run_row full8k_s0cpuqwen 8192 2
run_row full512_depth512 512 3
echo "--- second 8k window (later, same cotenant state if unchanged) ---"
run_row full8k_s0cpuqwen_late 8192 2
echo "=== done $(date -u +%FT%TZ) ==="
snapshot
} | tee "${OUT%.jsonl}.log"
echo "RESULTS $OUT"
cat "$OUT"
