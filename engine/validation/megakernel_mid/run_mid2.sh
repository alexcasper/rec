#!/usr/bin/env bash
set -u
source ~/rec-ve/e2e-env.sh
export PYTHONPATH=$HOME/rec-ve/pp2/engine/src
cd ~/rec-ve/pp2; H=engine/validation/megakernel_mid/mid_mk_check.py; CK=$HOME/rec-ve/ckpt-pp2/mid-prod
t0=$(date +%s); stamp(){ echo "[$(( $(date +%s)-t0 ))s] $*"; }
for c in short4 chat wrap512; do for f in no-silu softmax-mix; do
  $PY $H produce --ckpt $CK --case $c --impl torch-fp32 --fault $f --device cuda --out out/${c}_fp32_fault-${f}.npz >/dev/null 2>>out/err.log && stamp "produced $c fp32 fault $f"
done; done
$PY $H matrix --ckpt $CK out --ref-device cuda > out/matrix2.txt 2>>out/err.log; echo "matrix exit $?" >> out/matrix2.txt
stamp DONE
