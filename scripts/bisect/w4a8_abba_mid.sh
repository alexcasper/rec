#!/usr/bin/env bash
# rec-21a step-4: W4A8 ABBA 12/12 vs INT4-fp32-A baseline (mid, mode 2 both sides), cotenant-gated.
# A = w8base (flag-off INT4-fp32-A), B = w8on (GEPA_W4A8=1). ABBA order A,B,A,B x2 rounds.
# Cotenant hygiene: flint :8080 / E4B :8090 left UP (mid benches run gated-with-flint-up; busy<15 required),
# state recorded per round in window.log. Waits for any running m3_depth_gate (lib exclusivity) first.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1
OUT=~/rec-ve/i8/abba_w4a8_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
log(){ echo "$(date -u +%H:%M:%SZ) $* 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent)" | tee -a "$OUT/window.log"; }
log "preflight: waiting for depth-gate lib exclusivity"
while pgrep -f "m3_depth_gate" >/dev/null; do sleep 30; done
BANK=$(ls ~/rec-ve/ckpt-pp2/mid-prod/*.bank 2>/dev/null | head -1)
[ -z "$BANK" ] && BANK=$(find ~/rec-ve -maxdepth 3 -name "*mid*.bank" 2>/dev/null | head -1)
log "bank=$BANK"
[ -z "$BANK" ] && { log "ABORT: no mid bank found"; exit 2; }
DRV=~/rec-ve/gepa/h/engine/native/megakernel_hip/flint_full_driver.py
for round in 1 2; do
  for lib in w8base w8on; do
    log "ABBA start $lib r$round"
    $PY $DRV --lib ~/rec-ve/i8/$lib/flint_mid.so --bank "$BANK" --ckpt ~/rec-ve/ckpt-pp2/mid-prod \
      --workers 16 --mode 2 --window-bench "$OUT/bench_${lib}_r${round}.json" \
      --rows short,ctx8192 --pairs 12 --budget-s 200 > "$OUT/driver_${lib}_r${round}.log" 2>&1
    rc=$?
    grep -E "ROW |window bench" "$OUT/driver_${lib}_r${round}.log" | cut -c1-220 | tee -a "$OUT/window.log"
    log "ABBA $lib r$round rc=$rc"
  done
done
log "done -> $OUT"
