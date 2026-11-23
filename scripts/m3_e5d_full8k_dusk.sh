#!/usr/bin/env bash
# rec-e5d iter 2 full-config confirmation window: ctx8192 production (bf16-KV) ABBA rows for lop4 and gqa vs incumbent 26.80,
# then one fp32-KV depth8192 emit (lop4) for dawn scoring. Script hard stop 19 min (coordinator cap ~20 min :8080 down).
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
H=~/rec-ve/gepa/h; G="$PY $H/scripts/m3_depth_gate.py --bank $HOME/xh7/flint-int4.bank --ckpt $HOME/xh7/flint-int4"
LOP=~/rec-ve/bisect/e5d/full_lop4.so; GQA=~/rec-ve/bisect/e5d/full_gqa.so
OUT=~/rec-ve/m3window/e5d8k-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s); HARD=1140
log(){ echo "$(date -u +%H:%M:%SZ) +$(( $(date +%s) - T0 ))s $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl:$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8990/health)/$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8991/health) llama-qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) display:$(systemctl is-active display-manager 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
left(){ echo $(( HARD - ($(date +%s) - T0) )); }
step(){ local name=$1 budget=$2; shift 2; local l=$(left); if [ $l -lt $budget ]; then log "SKIP $name: ${l}s left < ${budget}s"; return 9; fi
  log "START $name (budget ${budget}s)"; timeout $((l - 15)) "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  grep -E "ROW|GATE|INVALID|cap|deadline|REFUS|Traceback" "$OUT/$name.log" | cut -c1-220 | tee -a "$OUT/window.log"; log "END $name rc=$rc"; }
log "preflight e5d full8k"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $LOP $GQA | tee -a "$OUT/window.log"
step lop4_bench 330 $G --lib $LOP --out "$OUT/lop4_bench" --case depth8192 --builds bf16kv --repeats 1 --pairs 12 --budget-s 420
step gqa_bench 330 $G --lib $GQA --out "$OUT/gqa_bench" --case depth8192 --builds bf16kv --repeats 1 --pairs 12 --budget-s 420
step lop4_fp32kv 300 $G --lib $LOP --out "$OUT/lop4_fp32kv" --case depth8192 --builds fp32kv --repeats 1 --budget-s 400
log "done -> signal ops_flint to restart :8080"
echo "$OUT"
