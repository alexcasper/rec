#!/usr/bin/env bash
# rec-6ek discriminator window: (1) reproduce the failing window config (flint_full.so = HEAD + GEPA_ATTN_SPLIT=1, driver
# create2/load_bank path), (2) same .so via shim x3, (3) HEAD default build via shim x3 (determinism), (4) 2fd83cb via shim x3
# (race vs defect), (5) HEAD default via driver path (load path at 21 GiB). ~90 s. Single 7-position launches (~0.2 s).
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
cd ~/rec-ve/bisect; OUT=win2-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; T0=$(date +%s)
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { echo "ABORT llama-qwen active"; exit 2; }
MK=~/rec-ve/gepa/h/engine/native/megakernel_hip; BYTES=$(stat -c %s ~/xh7/flint-int4.bank)
sha256sum $MK/flint_full.so | tee -a $OUT/bisect.log
shim(){ LD_PRELOAD=$PWD/shim.so BANK_STREAM_FILE=$HOME/xh7/flint-int4.bank BANK_STREAM_BYTES=$BYTES \
  $PY spot_bisect.py --so $1 --off ~/xh7/flint-int4.off.npy --ckpt ~/xh7/flint-int4 --out $OUT/$2.npz --repeat 3 2>&1 | grep -v "^bank_stream_shim" | tee -a $OUT/bisect.log; }
drv(){ $PY $MK/flint_full_driver.py --lib $1 --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 --workers 16 --mode 0 \
  --emit-spot $OUT/$2 2>&1 | grep -E "spot wrote|REFUS|INVALID|Error" | tee -a $OUT/bisect.log; }
drv $MK/flint_full.so drv_split
shim $MK/flint_full.so shim_split
shim full_k_5363ad5.so shim_head
shim full_k_2fd83cb.so shim_2fd83cb
drv $PWD/full_k_5363ad5.so drv_head
echo "done $(( $(date +%s) - T0 ))s $(date -u +%H:%M:%SZ) -> signal ops_flint restart" | tee -a $OUT/bisect.log
