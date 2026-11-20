#!/usr/bin/env bash
# rec-u9q on albion R9700 (gfx1201, HIP device 0): builds + G0 (refactored FLINT_CFG_FULL=0 build is
# byte-identical to the untouched d1f6208 source on the mid-8L suite).
set -u
R=~/rec-ve/u9q; MK=$R/gk/engine/native/megakernel_hip
export LD_LIBRARY_PATH=$(ls -d /nix/store/*-gcc-14.2.1.20250322-lib | head -1)/lib:$(ls -d /nix/store/*-zstd-1.5.7 | head -1)/lib:$(ls -d /nix/store/*-zlib-1.3.2 | head -1)/lib
unset HSA_OVERRIDE_GFX_VERSION
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
PY=~/rec-ve/.venv/bin/python
TC=$(dirname "$(readlink ~/rec-ve/rocm10/lib/llvm/bin/clang++)")
hb(){ env ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_CLANG_PATH=$TC hipcc "$@"; }
CARD=card1
busy(){ local B=/sys/class/drm/$CARD/device; cat $B/gpu_busy_percent 2>/dev/null || { [ "$(cat $B/power/runtime_status 2>/dev/null)" = suspended ] && echo 0 || echo 100; }; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(busy)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; echo GATE TIMEOUT; return 1; }
stamp(){ echo "[$(date -u +%T)Z] $*"; }
CK=~/rec-ve/r109/ckpt/mid-prod
for v in orig refac full; do
  d=$R/b-$v; rm -rf $d; mkdir -p $d; cp $MK/{mid_mk_driver.py,flint_full_driver.py} $d/
  if [ $v = orig ]; then cp $MK/flint_mid_megakernel.hip.d1f6208 $d/flint_mid_megakernel.hip; else cp $MK/flint_mid_megakernel.hip $d/; fi
  cfg=0; so=flint_mid.so; [ $v = full ] && { cfg=1; so=flint_full.so; }
  (cd $d && hb -shared -fPIC -O2 -Wall -DFLINT_CFG_FULL=$cfg --offload-arch=gfx1201 flint_mid_megakernel.hip -o $so > build.log 2>&1); echo "$v build exit $? errors=$(grep -c ' error' $d/build.log) warnings=$(grep -c warning $d/build.log)"
done
for v in orig refac; do
  gate || exit 1
  stamp "G0 emit $v"
  (cd $R/b-$v && timeout 600 $PY mid_mk_driver.py --ckpt $CK --emit-suite out --builds fp32kv,fp32_bf16kv --emit-experts --workers=16 > emit.log 2>&1); echo "$v emit exit $?"
  grep -E "wrote|rror" $R/b-$v/emit.log | cut -c1-110
done
$PY - <<'EOF'
import numpy as np, glob, os
R = os.path.expanduser("~/rec-ve/u9q")
fs = sorted(glob.glob(f"{R}/b-orig/out/*.npz")); ok = 0
for f in fs:
    a, b = np.load(f), np.load(f.replace("b-orig", "b-refac"))
    keys = [k for k in a.files if k not in ("impl",)]
    diff = [k for k in keys if not np.array_equal(a[k], b[k])]
    ok += not diff
    if diff: print(os.path.basename(f), "DIFF", diff)
print(f"G0 gfx1201: {ok}/{len(fs)} mid npz byte-identical (all arrays except impl string) refactored vs d1f6208")
EOF
stamp done
