#!/usr/bin/env bash
# rec-3cr R11 re-run (coordinator ruling 2026-11-22): HEAD default full build (no GEPA_ATTN_SPLIT; soaked deterministic+PASS in
# rec-6ek), w16 decode32@512 + short, 12 ABBA pairs, plus a 5-repeat determinism soak spot. Run ONLY after ops_flint confirms DOWN.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
MK=~/rec-ve/gepa/h/engine/native/megakernel_hip; SO=~/rec-ve/bisect/full_k_5363ad5.so
OUT=~/rec-ve/m3window/rerun-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s)
log(){ echo "$(date -u +%H:%M:%SZ) $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) kwin=$(pgrep -c kwin_wayland) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
log "preflight rerun"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $SO | tee -a "$OUT/window.log"
log "soak: 5 repeats fp32kv spot"
cd ~/rec-ve/bisect && LD_PRELOAD=$PWD/shim.so BANK_STREAM_FILE=$HOME/xh7/flint-int4.bank BANK_STREAM_BYTES=$(stat -c %s ~/xh7/flint-int4.bank) \
  $PY spot_bisect.py --so $SO --off ~/xh7/flint-int4.off.npy --ckpt ~/xh7/flint-int4 --out $OUT/soak.npz --repeat 5 2>&1 | grep -v "^bank_stream" | tee -a "$OUT/window.log"
log "rows w16 ctx512,short pairs 12"
$PY $MK/flint_full_driver.py --lib $SO --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 --workers 16 --mode 0 \
  --window-bench "$OUT/w16.json" --rows ctx512,short --pairs 12 --budget-s 300 > "$OUT/driver_w16.log" 2>&1; rc=$?
grep -E "ROW|window bench|ABORT|REFUS" "$OUT/driver_w16.log" | tee -a "$OUT/window.log"
log "done rc=$rc elapsed $(( $(date +%s) - T0 ))s -> signal ops_flint to restart :8080"
exit $rc
