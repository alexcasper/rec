#!/usr/bin/env bash
# rec-a4w: tiny megakernel 11/11 re-run after the width-32 warp_sum fix, gfx906 then gfx1201.
set -u
R=~/rec-ve/a4w; SRC=$R/gk/engine
export LD_LIBRARY_PATH=$(ls -d /nix/store/*-gcc-14.2.1.20250322-lib | head -1)/lib:$(ls -d /nix/store/*-zstd-1.5.7 | head -1)/lib:$(ls -d /nix/store/*-zlib-1.3.2 | head -1)/lib
unset HSA_OVERRIDE_GFX_VERSION; PY=~/rec-ve/.venv/bin/python
TC=$(dirname "$(readlink ~/rec-ve/rocm10/lib/llvm/bin/clang++)")
busy(){ local B=/sys/class/drm/$CARD/device; cat $B/gpu_busy_percent 2>/dev/null || { [ "$(cat $B/power/runtime_status 2>/dev/null)" = suspended ] && echo 0 || echo 100; }; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do if [ "$(busy)" -lt 5 ]; then s=$((s+1)); [ $s -ge 3 ] && return 0; else s=0; fi; sleep 0.3; done; return 1; }
sha256sum $SRC/native/megakernel_hip/tiny_megakernel.hip
for ARM in "gfx906 1 card2" "gfx1201 0 card1"; do
  set -- $ARM; ARCH=$1; DEV=$2; CARD=$3; export HIP_VISIBLE_DEVICES=$DEV
  OUT=$R/out-$ARCH/tiny-fix; rm -rf $OUT; mkdir -p $OUT; MK=$R/mk-$ARCH
  cp -f $SRC/native/megakernel_hip/tiny_megakernel.hip $MK/
  (cd $MK && env ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_CLANG_PATH=$TC hipcc -shared -fPIC tiny_megakernel.hip -o flint_tiny.so --offload-arch=$ARCH > $OUT/build_tiny.log 2>&1); echo "$ARCH build exit $?"
  cd $MK
  run(){ name=$1; seed=$2; prompt=$3; data=$4; gate || { echo "$name GATE TIMEOUT"; return; }
    $PY tiny_mk_driver.py --seed="$seed" --prompt="$prompt" --n-out=24 --data="$data" --out="$OUT/io_$name.npz" > $OUT/tiny_$name.log 2>&1; de=$?
    $PY tiny_mk_np_reference.py "$OUT/io_$name.npz" >> $OUT/tiny_$name.log 2>&1; re=$?
    echo "$ARCH $name seed=$seed prompt=$prompt data=$data driver_exit=$de ref_exit=$re :: $(grep -E 'MATCH|MISMATCH|rel|tie|rror' $OUT/tiny_$name.log | tr '\n' ' ' | cut -c1-200)"; }
  for s in 0 1 2 3 4; do run "$(printf 'cfg%02d' $((s+1)))" "$s" "1,5,6,7" signed; done
  for s in 0 1 2 3 4; do run "$(printf 'cfg%02d' $((s+6)))" "$s" "1,100,200,300,400,511" signed; done
  run cfg11 0 "1,5,6,7" legacy
  echo "$ARCH tiny: $(grep -l '^MATCH' $OUT/tiny_cfg*.log | wc -l)/11 MATCH"
done
