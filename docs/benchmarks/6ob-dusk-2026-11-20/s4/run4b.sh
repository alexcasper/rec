#!/usr/bin/env bash
# rec-6ob step 4b on dusk: VCgen megakernel gate (suite byte-identity vs base + spec (a) matrix vs CPU ref),
# then ABBA decode32 short4/chat vs base. Launches <= ~0.24 s; busy-gated; cotenant state logged.
set -u
R=~/rec-ve/6ob/s4; cd $R
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
CK=~/rec-ve/ckpt-pp2/mid-prod
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || echo idle),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || echo idle),qwen:$(systemctl --user is-active llama-qwen)"; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(cat $B)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; return 1; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(cat $B) | $(cot)"; }
for d in base vcgen; do
  cp -f mid_mk_driver.py $d/
  gate || { st "GATE TIMEOUT"; exit 1; }
  st "emit $d short4,chat x 3 builds"
  (cd $d && rm -rf out && timeout 300 $PY mid_mk_driver.py --ckpt $CK --emit-suite out --builds fp32kv,fp32_bf16kv,bf16kv --cases short4,chat --emit-experts --workers=16 > emit.log 2>&1); st "emit $d exit $?"
done
$PY - <<'EOF'
import numpy as np, glob, os
fs = sorted(glob.glob("base/out/*.npz")); ok = 0
for f in fs:
    a, b = np.load(f), np.load(f.replace("base/", "vcgen/"))
    d = [k for k in a.files if k != "impl" and not np.array_equal(a[k], b[k])]
    ok += not d
    print(" ", os.path.basename(f), "IDENTICAL" if not d else f"DIFF {d}")
print(f"SUITE byte-identity vcgen vs base(2fd83cb): {ok}/{len(fs)}")
open("SUITE_OK", "w").write("1") if ok == len(fs) == 6 else None
EOF
st "spec (a) matrix vcgen vs torch fp32 CPU ref"
CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONPATH=$R/../s3/gk/engine/src timeout 1800 $PY ../s3/gk/engine/validation/megakernel_mid/mid_mk_check.py matrix --ckpt $CK vcgen/out > matrix_vcgen.txt 2> matrix_vcgen.err; echo "matrix exit $?" >> matrix_vcgen.txt
cat matrix_vcgen.txt
[ -f SUITE_OK ] || { st "SUITE NOT IDENTICAL -> no timing"; exit 1; }
cp -f ../s3/abba_short.py ../s3/mk_prof.py ../s3/q4_lab.py .
for c in short4 chat; do
  st "ABBA $c: A=base(2fd83cb) B=vcgen, mode 0, 8 rounds"
  timeout 600 $PY abba_short.py --a base/flint_mid.so --b vcgen/flint_mid.so --ckpt $CK --case $c --rounds 8 --mode 0 --out abba_${c}_vcgen.json 2>&1 | grep -v amdgpu.ids
done
st DONE
journalctl -k --since "-30 min" --no-pager 2>/dev/null | grep -iE "ring.*timeout|wedged|gpu reset" | tail -3
