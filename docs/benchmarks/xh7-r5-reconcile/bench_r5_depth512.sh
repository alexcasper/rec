#!/usr/bin/env bash
# r5-reconcile depth-512 row(s) with wait-for-idle. Appends to the existing jsonl.
set -uo pipefail
OUT="$HOME/flint-mid/r5-reconcile-20261121T214803.jsonl"
TS=100.98.73.29

waitidle () {
  for t in $(seq 1 90); do
    b=$(head -1 /sys/class/drm/card*/device/gpu_busy_percent | head -1)
    [ "${b:-100}" -lt 5 ] && return 0
    sleep 5
  done
  return 1
}

gate () {
  waitidle || return 1
  for i in 1 2 3; do
    b=$(head -1 /sys/class/drm/card*/device/gpu_busy_percent | head -1)
    [ "${b:-100}" -lt 5 ] || return 1
    sleep 0.3
  done
  for port in 8080 8090; do
    n=$(curl -s -m 3 "http://$TS:$port/slots" | grep -c '"is_processing": true')
    [ "$n" = "0" ] || return 1
  done
}

runrow () {  # $1 label, $2 depth, $3 run number
  gate || { echo "$1 run$3 GATE-FAIL after wait"; return 1; }
  python3 -c "
import json, random
random.seed($2)
json.dump({'prompt':[random.randrange(3,150144) for _ in range($2)],
           'n_predict':64,'temperature':0,'cache_prompt':False,'stream':False},
          open('/tmp/r5-p.json','w'))"
  ts=$(date -u +%FT%TZ)
  curl -s -m 600 "http://$TS:8080/completion" -H "Content-Type: application/json" \
    -d @/tmp/r5-p.json | python3 -c "
import json, sys
t = json.load(sys.stdin)['timings']
print(json.dumps({'label':'$1','depth':$2,'run':$3,'utc':'$ts',
 'prompt_tps':round(t['prompt_per_second'],2),'pred_tps':round(t['predicted_per_second'],2)}))" >> "$OUT"
  echo "$1 run$3: $(tail -1 "$OUT")"
  sleep 2
}

runrow depth512 512 1
runrow depth512 512 2
runrow depth512 512 3
echo "COTENANT: 8990=$(pgrep -f 'port 8990' >/dev/null && echo up || echo down) gpu_busy=$(head -1 /sys/class/drm/card*/device/gpu_busy_percent | head -1) mem_avail_MiB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
echo "=== ALL RUNS ==="
cat "$OUT"
