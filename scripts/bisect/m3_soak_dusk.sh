#!/usr/bin/env bash
# rec-3cr full-config soak (dusk, :8080 DOWN by ops_flint): racer control + remedy candidates, fp32kv spot x repeats via shim.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
cd ~/rec-ve/bisect; OUT=soak-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; T0=$(date +%s)
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { echo "ABORT llama-qwen active"; exit 2; }
BYTES=$(stat -c %s ~/xh7/flint-int4.bank)
for spec in "full_split_ctl 3" "full_split_shr0 8" "full_split_inl 5" "full_def_shr0 3"; do
  set -- $spec; [ $(( $(date +%s) - T0 )) -gt 240 ] && { echo "budget: skip $1"; continue; }
  sha256sum $1.so | cut -c1-16 | sed "s/^/$1 sha /" | tee -a $OUT/soak.log
  LD_PRELOAD=$PWD/shim.so BANK_STREAM_FILE=$HOME/xh7/flint-int4.bank BANK_STREAM_BYTES=$BYTES \
    $PY spot_bisect.py --so $1.so --off ~/xh7/flint-int4.off.npy --ckpt ~/xh7/flint-int4 --out $OUT/$1.npz --repeat $2 2>&1 \
    | grep -v "^bank_stream" | tee -a $OUT/soak.log
done
echo "done $(( $(date +%s) - T0 ))s $(date -u +%H:%M:%SZ) -> signal ops_flint restart" | tee -a $OUT/soak.log
