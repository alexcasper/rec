#!/usr/bin/env bash
# rec-21a step-4: W4A8 ABBA 12/12 vs INT4-fp32-A baseline (mid-8L, mode 2 both sides).
# A = w8base (flag-off INT4-fp32-A), B = w8on (GEPA_W4A8=1). ABBA order A,B,A,B.
# Adapted from i8_bench.sh (rec-ywy mid protocol, rec-bhz lesson): extended per-row cotenant gate —
# no foreign drivers, qwen3vl :8990/:8991 zero ESTAB, gpu_busy sum<15 over 3 s; :8080/:8090 health per row.
# Cotenant hygiene: flint :8080 + E4B :8090 left UP (mid protocol is gated, not paused); state logged per row.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
MK=~/rec-ve/i8/h/engine/native/megakernel_hip; B=~/rec-ve/i8
OUT=$B/abba_w4a8_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
D=/sys/class/drm/card0/device; [ -d "$D" ] || D=/sys/class/drm/card1/device
gate(){ for try in 1 2 3 4 5 6; do
    f=$(ps -eo args | grep -E "[f]lint_full_driver|[m]3_depth_gate|[m]3_emit_wrap512|[g]epa_remote.py (bench|emit)" | wc -l)
    q=$(ss -tn state established "( sport = :8990 or sport = :8991 )" 2>/dev/null | tail -n +2 | wc -l)
    b=0; for i in 1 2 3; do b=$(( b + $(cat $D/gpu_busy_percent) )); sleep 1; done
    h=$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health)
    st="foreign_drivers=$f qwen3vl_estab=$q busy3s=$b 8080=$h 8090=$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) llama-qwen=$(systemctl --user is-active llama-qwen 2>/dev/null) MemAvail=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo)"
    if [ $f -eq 0 ] && [ $q -eq 0 ] && [ $b -lt 15 ]; then echo "GATE OK try=$try $st"; return 0; fi
    echo "GATE WAIT try=$try $st"; sleep 30; done; return 1; }
# wait for any running depth gate / emit (lib exclusivity)
while ps -eo args | grep -q "[m]3_depth_gate"; do sleep 30; done
i=0
for e in w8base:2 w8on:2 w8base:2 w8on:2; do
  n=${e%%:*}; m=${e#*:}
  i=$((i+1))
  g=$(gate); echo "[$i $n m$m] $(date -u +%H:%M:%SZ) $g" | tee -a $OUT/summary.log
  echo "$g" | grep -q "GATE OK" || { echo "[$i $n m$m] SKIPPED (gate)" | tee -a $OUT/summary.log; continue; }
  $PY $MK/flint_full_driver.py --rehearsal --lib $B/$n/flint_mid.so --bank ~/rec-ve/m3prep/mid-prod.bank \
    --off ~/rec-ve/m3prep/mid-prod.off.npy --ckpt ~/rec-ve/ckpt-pp2/mid-prod --workers 16 --mode $m \
    --window-bench $OUT/r${i}_${n}_m$m.json --rows short,ctx8192 --pairs 12 --budget-s 600 --prefill-deadline-s 500 > $OUT/r${i}_${n}_m$m.log 2>&1
  grep -E "ROW |window bench|ABORT|Error" $OUT/r${i}_${n}_m$m.log | sed "s/^/[$i $n m$m] /" | cut -c1-230 | tee -a $OUT/summary.log
  echo "[$i $n m$m] post $(gate | tail -1)" | tee -a $OUT/summary.log
done
echo "done $OUT" | tee -a $OUT/summary.log
