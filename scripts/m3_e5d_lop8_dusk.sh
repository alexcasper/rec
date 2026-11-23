#!/usr/bin/env bash
# rec-e5d lop8 full-config confirmation: 8192 ABBA row, fp32-KV emit, wrap512 fresh-session determinism, ppl eval_set_full_v1.
# Priority order with per-step budgets; script hard stop 17 min (watchdog at pause+20 min).
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
H=~/rec-ve/gepa/h; G="$PY $H/scripts/m3_depth_gate.py --bank $HOME/xh7/flint-int4.bank --ckpt $HOME/xh7/flint-int4"
LOP8=~/rec-ve/bisect/e5d/full_lop8.so
OUT=~/rec-ve/m3window/e5dlop8-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s); HARD=1020
log(){ echo "$(date -u +%H:%M:%SZ) +$(( $(date +%s) - T0 ))s $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl:$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8990/health)/$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8991/health) llama-qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) display:$(systemctl is-active display-manager 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
left(){ echo $(( HARD - ($(date +%s) - T0) )); }
step(){ local name=$1 budget=$2; shift 2; local l=$(left); if [ $l -lt $budget ]; then log "SKIP $name: ${l}s left < ${budget}s"; return 9; fi
  log "START $name (budget ${budget}s)"; timeout $((l - 15)) "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  grep -E "ROW|GATE|INVALID|cap|deadline|REFUS|Traceback" "$OUT/$name.log" | cut -c1-220 | tee -a "$OUT/window.log"; log "END $name rc=$rc"; }
log "preflight e5d lop8"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $LOP8 | tee -a "$OUT/window.log"
step lop8_bench 330 $G --lib $LOP8 --out "$OUT/lop8_bench" --case depth8192 --builds bf16kv --repeats 1 --pairs 12 --budget-s 420
step lop8_fp32kv 290 $G --lib $LOP8 --out "$OUT/lop8_fp32kv" --case depth8192 --builds fp32kv --repeats 1 --budget-s 400
step lop8_det512 150 $G --lib $LOP8 --out "$OUT/lop8_det512" --case wrap512 --builds fp32kv,bf16kv --repeats 2 --budget-s 200
step lop8_ppl 230 $G --lib $LOP8 --out "$OUT/lop8_ppl" --ppl-set $H/engine/validation/megakernel_mid/qgate/eval_set_full_v1.json --ppl-builds fp32kv,bf16kv --budget-s 230
log "done -> signal ops_flint to restart :8080"
echo "$OUT"
