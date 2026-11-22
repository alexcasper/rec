#!/usr/bin/env bash
# rec-6ek bisect window (dusk, :8080 DOWN by ops_flint): 7 full-config builds, one 7-position fp32kv spot launch each
# (~0.2 s, under the cap), bank streamed via shim. ~80-120 s total. Logs cotenant/MemAvailable per build.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
cd ~/rec-ve/bisect; OUT=win-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { echo "ABORT llama-qwen active"; exit 2; }
BYTES=$(stat -c %s ~/xh7/flint-int4.bank); T0=$(date +%s)
for b in e3c5ed6 2fd83cb 39629e7 5d36dc9 24f167d 2b98737 5363ad5; do
  [ $(( $(date +%s) - T0 )) -gt 360 ] && { echo "budget: stop before $b"; break; }
  LD_PRELOAD=$PWD/shim.so BANK_STREAM_FILE=$HOME/xh7/flint-int4.bank BANK_STREAM_BYTES=$BYTES \
    $PY spot_bisect.py --so full_k_$b.so --off ~/xh7/flint-int4.off.npy --ckpt ~/xh7/flint-int4 --out $OUT/full_$b.npz \
    2>&1 | tee -a $OUT/bisect.log | grep -v "^bank_stream_shim"
done
echo "done $(( $(date +%s) - T0 ))s $(date -u +%H:%M:%SZ) -> signal ops_flint restart" | tee -a $OUT/bisect.log
