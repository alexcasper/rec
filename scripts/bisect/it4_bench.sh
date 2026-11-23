#!/usr/bin/env bash
# rec-e5d iter 4: mid-8L ABBA at ctx8192 + short (rehearsal driver, :8080 up, cotenant gated per pair)
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
MK=~/rec-ve/gepa/h/engine/native/megakernel_hip; B=~/rec-ve/bisect/it4; OUT=$B/bench-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
i=0
for n in lop8 lop8pf lop8u2 lop8 lop8u2 lop8pf lop8; do
  i=$((i+1))
  $PY $MK/flint_full_driver.py --rehearsal --lib $B/$n/flint_mid.so --bank ~/rec-ve/m3prep/mid-prod.bank \
    --off ~/rec-ve/m3prep/mid-prod.off.npy --ckpt ~/rec-ve/ckpt-pp2/mid-prod --workers 16 --mode 0 \
    --window-bench $OUT/r${i}_$n.json --rows short,ctx8192 --pairs 12 --budget-s 600 --prefill-deadline-s 500 \
    > $OUT/r${i}_$n.log 2>&1
  grep -E "ROW |window bench|ABORT|Error" $OUT/r${i}_$n.log | sed "s/^/[$i $n] /" | cut -c1-230 | tee -a $OUT/summary.log
done
echo "done $OUT" | tee -a $OUT/summary.log
