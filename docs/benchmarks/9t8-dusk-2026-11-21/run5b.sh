#!/usr/bin/env bash
# rec-6ob step 5b (rec-6ob.1) on dusk: VD megakernel gate G1 (short4 + chat x 3 builds, npz byte identity vs
# base 2fd83cb), then ABBA decode32 short-ctx x8 rounds: base(2fd83cb) vs VD, and vcgen(39629e7) vs VD.
# Launches <= ~0.24 s (short4/chat only; no wrap512/ctx512). Busy-gated; cotenant state logged.
set -u
S4=~/rec-ve/6ob/s4; R=~/rec-ve/6ob/s5; cd $R
V=${V:-vd}; AS=${AS:-"base vcgen"}
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
CK=~/rec-ve/ckpt-pp2/mid-prod
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || (curl -s -m2 -o /dev/null $H:8080/health && echo idle || echo down)),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || (curl -s -m2 -o /dev/null $H:8090/health && echo idle || echo down)),qwen:$(systemctl --user is-active llama-qwen)"; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(cat $B)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; return 1; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(cat $B) | $(cot)"; }
cp -f $S4/mid_mk_driver.py $V/
gate || { st "GATE TIMEOUT"; exit 1; }
st "emit $V short4,chat x 3 builds"
(cd $V && rm -rf out && timeout 300 $PY mid_mk_driver.py --ckpt $CK --emit-suite out --builds fp32kv,fp32_bf16kv,bf16kv --cases short4,chat --emit-experts --workers=16 > emit.log 2>&1); st "emit $V exit $?"
rm -f SUITE_OK
V=$V $PY - <<'EOF'
import numpy as np, glob, os
V = os.environ["V"]
fs = sorted(glob.glob("base/out/*.npz")); ok = 0
for f in fs:
    a, b = np.load(f), np.load(f.replace("base/", V + "/"))
    d = [k for k in a.files if k != "impl" and not np.array_equal(a[k], b[k])]
    ok += not d
    print(" ", os.path.basename(f), "IDENTICAL" if not d else f"DIFF {d}")
print(f"G1 SUITE byte-identity {V} vs base(2fd83cb): {ok}/{len(fs)}")
open("SUITE_OK", "w").write("1") if ok == len(fs) == 6 else None
EOF
[ -f SUITE_OK ] || { st "SUITE NOT IDENTICAL -> no timing"; exit 1; }
cp -f $S4/abba_short.py $S4/mk_prof.py $S4/q4_lab.py .
for A in $AS; do
  for c in short4 chat; do
    st "ABBA $c: A=$A B=$V, mode 0, 8 rounds"
    timeout 600 $PY abba_short.py --a $A/flint_mid.so --b $V/flint_mid.so --ckpt $CK --case $c --rounds 8 --mode 0 --out abba_${c}_${A}_$V.json 2>&1 | grep -v amdgpu.ids
  done
done
st DONE
journalctl -k --since "-30 min" --no-pager 2>/dev/null | grep -iE "ring.*timeout|wedged|gpu reset" | tail -3
