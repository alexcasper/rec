#!/usr/bin/env bash
# rec-pp2 mid-8L validation run on dusk (Radeon 890M). Log: ~/rec-ve/pp2/run_mid.log
set -u
source ~/rec-ve/e2e-env.sh
export PYTHONPATH=$HOME/rec-ve/pp2/engine/src
cd ~/rec-ve/pp2
H=engine/validation/megakernel_mid/mid_mk_check.py
CK=$HOME/rec-ve/ckpt-pp2/mid-prod
t0=$(date +%s); stamp(){ echo "[$(( $(date +%s)-t0 ))s] $*"; }
stamp "build ckpt"; $PY -m rec_mlx.synthetic --preset mid --format production --out $CK 2>&1 | grep wrote
mkdir -p out; $PY $H cases --ckpt $CK > out/cases.json 2>/dev/null; stamp "cases: $(head -c 120 out/cases.json)"
for c in short4 chat wrap512; do
  for impl in torch-fp32 torch-fp32-bf16kv torch-bf16; do
    $PY $H produce --ckpt $CK --case $c --impl $impl --device cuda --out out/${c}_${impl}.npz >/dev/null 2>>out/err.log && stamp "produced $c $impl" || stamp "PRODUCE FAILED $c $impl"
  done
  for f in no-silu softmax-mix; do
    $PY $H produce --ckpt $CK --case $c --impl torch-fp32-bf16kv --fault $f --device cuda --out out/${c}_fault-${f}.npz >/dev/null 2>>out/err.log && stamp "produced $c fault $f" || stamp "PRODUCE FAILED $c $f"
  done
done
stamp "matrix check (ref: fp32 compute on cuda)"
$PY $H matrix --ckpt $CK out --ref-device cuda > out/matrix.txt 2>>out/err.log; echo "matrix exit $?" >> out/matrix.txt
stamp "DONE"
