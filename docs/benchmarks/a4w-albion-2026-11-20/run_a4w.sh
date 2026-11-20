#!/usr/bin/env bash
# rec-a4w correctness legs on albion: tiny megakernel 11/11 + mid-8L rec-pp2 kernel matrix,
# gfx1201 (R9700, HIP device 0, card1) then gfx906 (MI50, HIP device 1, card2), sequential.
# Tree: rec/gemini-kernels 4d6e6ec (d1f6208 kernels + f0698a5 CPU sign-off reference).
# Toolchain: /opt/rocm HIP 7.2 + nix rocm-toolchain clang. Torch-free drivers; the matrix
# reference runs torch fp32 on the CPU (SIGNOFF_REF_DEVICE) via ~/rec-ve/.venv.
set -u
R=~/rec-ve/a4w
SRC=$R/gk/engine
CK=~/rec-ve/r109/ckpt/mid-prod
export LD_LIBRARY_PATH=$(ls -d /nix/store/*-gcc-14.2.1.20250322-lib | head -1)/lib:$(ls -d /nix/store/*-zstd-1.5.7 | head -1)/lib:$(ls -d /nix/store/*-zlib-1.3.2 | head -1)/lib
unset HSA_OVERRIDE_GFX_VERSION
PY=~/rec-ve/.venv/bin/python
TC=$(dirname "$(readlink ~/rec-ve/rocm10/lib/llvm/bin/clang++)")
t0=$(date +%s); stamp(){ echo "[$(( $(date +%s)-t0 ))s $(date -u +%T)Z] $*"; }
busy(){ local B=/sys/class/drm/$CARD/device; cat $B/gpu_busy_percent 2>/dev/null || { [ "$(cat $B/power/runtime_status 2>/dev/null)" = suspended ] && echo 0 || echo 100; }; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do if [ "$(busy)" -lt 5 ]; then s=$((s+1)); [ $s -ge 3 ] && return 0; else s=0; fi; sleep 0.3; done; stamp "GATE TIMEOUT $CARD busy=$(busy)"; return 1; }
hb(){ env ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_CLANG_PATH=$TC hipcc "$@"; }

stamp "host $(hostname) $(uptime)"
sha256sum $SRC/native/megakernel_hip/{tiny_megakernel.hip,flint_mid_megakernel.hip,tiny_mk_driver.py,tiny_mk_np_reference.py,mid_mk_driver.py} $SRC/validation/megakernel_mid/mid_mk_check.py $CK/model-00001.safetensors | sed 's/  / /'

for ARM in "gfx1201 0 card1" "gfx906 1 card2"; do
  set -- $ARM; ARCH=$1; DEV=$2; CARD=$3
  OUT=$R/out-$ARCH; rm -rf $OUT; mkdir -p $OUT
  MK=$R/mk-$ARCH; rm -rf $MK; cp -r $SRC/native/megakernel_hip $MK   # per-arch .so next to its driver
  export HIP_VISIBLE_DEVICES=$DEV CUDA_VISIBLE_DEVICES=$DEV
  cd $MK
  stamp "===== $ARCH (HIP device $DEV, $CARD)"
  hb -shared -fPIC tiny_megakernel.hip -o flint_tiny.so --offload-arch=$ARCH > $OUT/build_tiny.log 2>&1; echo "tiny build exit $? errors=$(grep -c ' error' $OUT/build_tiny.log)"
  hb -shared -fPIC -O2 -Wall --offload-arch=$ARCH flint_mid_megakernel.hip -o flint_mid.so > $OUT/build_mid.log 2>&1; echo "mid build exit $? errors=$(grep -c ' error' $OUT/build_mid.log)"
  OCC=$($PY -c "
import sys; sys.path.insert(0, '.')
import mid_mk_driver as D
o = D.query_occupancy(D.load_lib()); print(o['max_co_resident_workers'], o)" 2>&1 | tail -1)
  echo "occupancy: $OCC"; WMAX=${OCC%% *}

  stamp "== tiny matrix $ARCH"
  run(){ name=$1; seed=$2; prompt=$3; data=$4
    gate || return
    $PY tiny_mk_driver.py --seed="$seed" --prompt="$prompt" --n-out=24 --data="$data" --out="$OUT/io_$name.npz" > $OUT/tiny_$name.log 2>&1; de=$?
    $PY tiny_mk_np_reference.py "$OUT/io_$name.npz" >> $OUT/tiny_$name.log 2>&1; re=$?
    echo "$ARCH $name seed=$seed prompt=$prompt data=$data driver_exit=$de ref_exit=$re :: $(grep -E 'MATCH|MISMATCH|rel|tie|rror' $OUT/tiny_$name.log | tr '\n' ' ' | cut -c1-260)"; }
  for s in 0 1 2 3 4; do run "$(printf 'cfg%02d' $((s+1)))" "$s" "1,5,6,7" signed; done
  for s in 0 1 2 3 4; do run "$(printf 'cfg%02d' $((s+6)))" "$s" "1,100,200,300,400,511" signed; done
  run cfg11 0 "1,5,6,7" legacy
  echo "$ARCH tiny: $(grep -l '^MATCH' $OUT/tiny_cfg*.log 2>/dev/null | wc -l)/11 MATCH"

  export PYTHONPATH=$SRC/src
  for W in 16 $WMAX; do
    [ "$W" -eq 16 ] 2>/dev/null || [ -n "$W" ] || continue
    stamp "== mid suite $ARCH workers=$W"
    gate || continue
    timeout 1500 $PY mid_mk_driver.py --ckpt $CK --emit-suite $OUT/suite_w$W --builds fp32kv,fp32_bf16kv --emit-experts --workers=$W > $OUT/emit_w$W.log 2>&1; echo "emit exit $?"; grep -E "wrote|rror" $OUT/emit_w$W.log | cut -c1-140
    timeout 1500 $PY $SRC/validation/megakernel_mid/mid_mk_check.py matrix --ckpt $CK $OUT/suite_w$W > $OUT/signoff_w$W.txt 2> $OUT/signoff_w$W.err; echo "matrix exit $?" >> $OUT/signoff_w$W.txt
    cat $OUT/signoff_w$W.txt
    [ "$W" = "$WMAX" ] && break
  done
done
stamp DONE
