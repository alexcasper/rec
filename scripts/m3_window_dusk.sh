#!/usr/bin/env bash
# Bedivere M3.4 dusk window run (full Flint megakernel). Pre-staged; run ONLY on the coordinator's go, AFTER ops_flint
# has stopped :8080 for the window (granted 2026-11-22 09:00Z-22:00Z, restart SLA ~10 min).
#   default (cap-compliant): every launch <= 0.25 s via M3.1 KV continuation; aborts on any longer launch.
#   OVERRIDE=1 adds --override-approved: only once Alex's banned-launch override is granted. It still uses
#   continuation and only lifts the abort.
# Logs: ~/rec-ve/m3window/<UTC>/ (driver log with cotenants + MemAvailable per line, json, spot npz).
set -euo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
MK=~/rec-ve/gepa/h/engine/native/megakernel_hip
OUT=~/rec-ve/m3window/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
log(){ echo "$(date -u +%H:%M:%SZ) $* | $(for p in 8080 8090 8091; do printf "%s:%s " $p "$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:$p/health)"; done) busy=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card*/device/mem_info_vram_total|head -1) -v u=$(cat /sys/class/drm/card*/device/mem_info_vram_used|head -1) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
log "preflight (expect 8080 down=000, llama-qwen inactive, vram_free >= 23 GiB)"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $MK/flint_full.so ~/xh7/flint-int4.off.npy | tee -a "$OUT/window.log"
EXTRA=""; [ "${OVERRIDE:-0}" = 1 ] && EXTRA="--override-approved"
log "start: spot (fp32kv short4, 2 continued launches) + rows ctx512, short (12 ABBA pairs each), budget 420 s"
$PY $MK/flint_full_driver.py --lib $MK/flint_full.so --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 \
    --workers 16 --mode 0 --emit-spot "$OUT/spot" --window-bench "$OUT/window_bench.json" \
    --rows ctx512,short --pairs 12 --budget-s 420 $EXTRA 2>&1 | tee -a "$OUT/driver.log" && rc=0 || rc=${PIPESTATUS[0]}
log "done rc=$rc -> signal ops_flint to restart :8080"
exit $rc
