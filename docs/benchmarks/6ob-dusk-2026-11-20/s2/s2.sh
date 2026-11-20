#!/usr/bin/env bash
# rec-6ob step 2 on dusk: batched-loads gemv. G0 (short4 + chat ONLY), then timing + re-profile.
# Every launch <= ~0.25 s (display iGPU; the 16:03Z wedge was a 3.77 s mid-8L wrap512 launch).
set -u
R=~/rec-ve/6ob/s2; cd $R
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
CK=~/rec-ve/ckpt-pp2/mid-prod
REC=~/rec-ve/megakernel_hip/pp2_aip_exp
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || echo idle),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || echo idle),qwen:$(systemctl --user is-active llama-qwen)"; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(cat $B)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; return 1; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(cat $B) | $(cot)"; }
for v in old new; do
  (cd $v && hipcc -O2 -Wall -shared -fPIC --offload-arch=gfx1150 flint_mid_megakernel.hip -o flint_mid.so > build.log 2>&1); st "build $v exit $? sha=$(sha256sum $v/flint_mid_megakernel.hip | cut -c1-12)"
done
for v in old new; do
  gate || { st "GATE TIMEOUT"; exit 1; }
  st "G0 emit $v (short4,chat x fp32kv,fp32_bf16kv,bf16kv)"
  (cd $v && rm -rf out && timeout 300 $PY mid_mk_driver.py --ckpt $CK --emit-suite out --builds fp32kv,fp32_bf16kv,bf16kv --cases short4,chat --emit-experts --workers=16 > emit.log 2>&1); st "emit $v exit $?"
  grep -E "wrote|rror" $v/emit.log | cut -c1-90
done
$PY - <<'EOF'
import numpy as np, glob, os
R = os.path.expanduser("~/rec-ve/6ob/s2"); REC = os.path.expanduser("~/rec-ve/megakernel_hip/pp2_aip_exp")
fs = sorted(glob.glob(f"{R}/old/out/*.npz")); ok = okr = nr = 0
for f in fs:
    a, b = np.load(f), np.load(f.replace("/old/", "/new/"))
    d = [k for k in a.files if k != "impl" and not np.array_equal(a[k], b[k])]
    ok += not d
    if d: print("old vs new DIFF", os.path.basename(f), d)
    r = os.path.join(REC, os.path.basename(f))
    if os.path.exists(r):
        nr += 1; c = np.load(r)
        dr = [k for k in ("tokens", "step_logits", "last_logits", "chosen_experts", "step_experts") if k in c.files and k in b.files and not np.array_equal(b[k], c[k])]
        okr += not dr
        if dr: print("new vs run-of-record DIFF", os.path.basename(f), dr)
print(f"G0 new vs rebuilt d1f6208 control: {ok}/{len(fs)} npz byte-identical (all arrays except impl)")
print(f"G0 new vs stored run-of-record npz ({REC}): {okr}/{nr} identical")
open(f"{R}/G0_PASS", "w").write("1") if ok == len(fs) and len(fs) == 6 and okr == nr else None
EOF
[ -f G0_PASS ] || { st "G0 FAILED -> no timing"; exit 1; }
st "G0 PASS -> timing (decode32 short-ctx differential, ABBA x8, mode 0 production build)"
timeout 600 $PY abba_short.py --a old/flint_mid.so --b new/flint_mid.so --ckpt $CK --rounds 8 --mode 0 --out abba_short_mode0.json
st "re-profile (MK_PROFILE build of new source)"
(cd prof && hipcc -O2 -Wall -shared -fPIC -DMK_PROFILE --offload-arch=gfx1150 flint_mid_megakernel_prof.hip -o flint_mid_prof.so > build.log 2>&1); st "prof build exit $?"
gate && timeout 300 $PY mk_prof.py --so prof/flint_mid_prof.so --src prof/flint_mid_megakernel_prof.hip --ckpt $CK --prod new/flint_mid.so --n-out 36 --workers 16 --reps 3 --out prof_new_w16.json
st DONE
journalctl -k --since "-15 min" --no-pager 2>/dev/null | grep -iE "ring.*timeout|wedged|gpu reset" | tail -3
