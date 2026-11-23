#!/usr/bin/env bash
# rec-04j: re-baseline the @512 decode row on lop8f (SHRINK=1+fence, single build for both depths). Self-waiting: polls until
# :8080 is down (ops_flint pause, seat-to-seat STOP from ops_rec), then runs speed rows short+ctx512, 12 ABBA pairs, mode 0
# (production bf16-KV): same protocol as R11r / gate512. det512 already PASS (stageC). Bench hard stop 6 min after window open.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
MK=~/rec-ve/gepa/h/engine/native/megakernel_hip; LIB=~/rec-ve/fut11b/lop8f/flint_mid.so; SHA=ec34a71153bae74b
OUT=~/rec-ve/m3window/rec04j-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s); WAIT_MAX=1800; HARD=360
log(){ echo "$(date -u +%H:%M:%SZ) +$(( $(date +%s) - T0 ))s $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl:$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8990/health)/$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8991/health) llama-qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) display:$(systemctl is-active display-manager 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
log "preflight rec-04j lop8f @512 re-baseline (waiting for :8080 pause, max ${WAIT_MAX}s)"
s=$(sha256sum $LIB | cut -c1-16); [ "$s" = "$SHA" ] || { log "ABORT: lib sha $s != $SHA"; exit 4; }
while :; do c=$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health 2>/dev/null)
  [ "$c" != 200 ] && ! ss -ltn | grep -q 100.98.73.29:8080 && break
  [ $(( $(date +%s) - T0 )) -ge $WAIT_MAX ] && { log "ABORT: window never opened"; exit 3; }; sleep 10; done
log "window open :8080 down"; TW=$(date +%s)
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
D=/sys/class/drm/card1/device; for i in $(seq 1 10); do f=$(( ($(cat $D/mem_info_vram_total)-$(cat $D/mem_info_vram_used))/1073741824 )); [ $f -ge 30 ] && break; sleep 2; done
[ $f -ge 25 ] || { log "ABORT: vram_free ${f}GiB"; exit 5; }
b=0; for i in 1 2 3; do b=$(( b + $(cat $D/gpu_busy_percent) )); sleep 1; done; [ $b -lt 15 ] || { log "ABORT: foreign GPU busy (sum $b)"; exit 6; }
sha256sum $LIB | tee -a "$OUT/window.log"
log "START speed512 (short + ctx512, 12 ABBA pairs, mode 0; budget 200s, hard ${HARD}s)"
timeout $HARD $PY $MK/flint_full_driver.py --lib $LIB --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 --workers 16 --mode 0 \
    --window-bench "$OUT/window_bench.json" --rows short,ctx512 --pairs 12 --budget-s 200 > "$OUT/driver.log" 2>&1; rc=$?
grep -E "ROW|INVALID|cap|deadline|REFUS|Traceback|window bench" "$OUT/driver.log" | cut -c1-230 | tee -a "$OUT/window.log"
log "END speed512 rc=$rc (bench $(( $(date +%s) - TW ))s after window open)"
log "done -> signal ops_flint to restart :8080"
echo "$OUT"
