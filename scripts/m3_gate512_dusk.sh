#!/usr/bin/env bash
# rec-3cr / M3: full-config split-K + GEPA_SHRINK=0 (288f4fa default) depth-512 row, GATED.
# (1) gate emits: wrap512 x {fp32kv, fp32_bf16kv, bf16kv} x 2 fresh sessions (determinism + rec-pp2 npz for dawn)
# (2) speed rows: short + ctx512, 12 ABBA pairs, mode 0 (production bf16-KV), same protocol as R11r.
# Run ONLY after ops_flint confirms :8080 DOWN. 0.25 s launch cap everywhere; gate budget 300 s, bench budget 200 s.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
H=~/rec-ve/gepa/h; MK=$H/engine/native/megakernel_hip; LIB=~/rec-ve/bisect/flint_full_split_288f4fa.so
OUT=~/rec-ve/m3window/gate512-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
log(){ echo "$(date -u +%H:%M:%SZ) $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl:$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8990/health)/$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8991/health) llama-qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) display:$(systemctl is-active display-manager 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
log "preflight gate512"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $LIB | tee -a "$OUT/window.log"
$PY $H/scripts/m3_depth_gate.py --lib $LIB --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 --out "$OUT/gate" \
    --case wrap512 --repeats 2 --budget-s 300 > "$OUT/gate.log" 2>&1; g=$?
grep -E "GATE|DETERMINISM" "$OUT/gate.log" | cut -c1-200 | tee -a "$OUT/window.log"
log "gate emits rc=$g"
$PY $MK/flint_full_driver.py --lib $LIB --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 --workers 16 --mode 0 \
    --window-bench "$OUT/window_bench.json" --rows short,ctx512 --pairs 12 --budget-s 200 > "$OUT/driver.log" 2>&1; b=$?
grep -E "ROW |window bench" "$OUT/driver.log" | cut -c1-220 | tee -a "$OUT/window.log"
log "done gate_rc=$g bench_rc=$b -> signal ops_flint to restart :8080"
echo "$OUT"
