#!/usr/bin/env bash
# Bedivere M3.4 depth-8192 slot (coordinator 2026-11-22): full-Flint megakernel ctx8192 row only, no override, no spot.
# Run ONLY after ops_flint confirms :8080 DOWN. Hard limits: prefill deadline 420 s, bench budget 480 s.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
MK=~/rec-ve/gepa/h/engine/native/megakernel_hip
OUT=~/rec-ve/m3window/8k-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
log(){ echo "$(date -u +%H:%M:%SZ) $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
log "preflight 8k slot"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $MK/flint_full.so | tee -a "$OUT/window.log"
$PY $MK/flint_full_driver.py --lib $MK/flint_full.so --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 \
    --workers 16 --mode 0 --window-bench "$OUT/window_bench.json" --rows ctx8192 --pairs 12 \
    --budget-s 480 --prefill-deadline-s 420 > "$OUT/driver.log" 2>&1
rc=$?
log "done rc=$rc -> signal ops_flint to restart :8080"
exit $rc
