#!/usr/bin/env bash
# Bedivere M3 window 3 (display session stopped, Alex option b): workers sweep + ctx512/short re-measure in the new state.
# Run ONLY after ops_flint confirms :8080 DOWN. No override. Each invocation streams the bank (9 s) and frees it on exit.
#   w16: ctx512 + short, 12 pairs  (~2 min)   w12: ctx512, 8 pairs (~70 s)   w8: ctx512, 8 pairs (~80 s)
# Total budget check: abort the remaining steps if the elapsed time passes 420 s.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
MK=~/rec-ve/gepa/h/engine/native/megakernel_hip
OUT=~/rec-ve/m3window/sweep-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s)
log(){ echo "$(date -u +%H:%M:%SZ) $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) kwin=$(pgrep -c kwin_wayland) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
log "preflight sweep"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $MK/flint_full.so | tee -a "$OUT/window.log"
rc_all=0
for spec in "16 ctx512,short 12" "12 ctx512 8" "8 ctx512 8"; do
  set -- $spec; el=$(( $(date +%s) - T0 ))
  [ $el -gt 420 ] && { log "budget: skipping w$1 (elapsed ${el}s)"; continue; }
  log "start w$1 rows=$2 pairs=$3"
  $PY $MK/flint_full_driver.py --lib $MK/flint_full.so --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 \
      --workers $1 --mode 0 --window-bench "$OUT/w$1.json" --rows $2 --pairs $3 --budget-s 300 \
      > "$OUT/driver_w$1.log" 2>&1; rc=$?
  grep -E "ROW|window bench|ABORT|REFUS" "$OUT/driver_w$1.log" | tee -a "$OUT/window.log"
  log "w$1 rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
log "done rc=$rc_all elapsed $(( $(date +%s) - T0 ))s -> signal ops_flint to restart :8080"
exit $rc_all
