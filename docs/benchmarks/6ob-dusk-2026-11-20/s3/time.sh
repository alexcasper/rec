#!/usr/bin/env bash
# rec-6ob step 4 on dusk: ABBA short-ctx decode timing HEAD vs v2 (short4, chat; mode 0), then
# re-profile v2 (MK_PROFILE build). Every launch <= ~0.26 s; busy-gated; cotenant state per line.
set -u
R=~/rec-ve/6ob/s3; cd $R
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
CK=~/rec-ve/ckpt-pp2/mid-prod
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || echo idle),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || echo idle),qwen:$(systemctl --user is-active llama-qwen)"; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(cat $B) | $(cot)"; }
for c in short4 chat; do
  st "ABBA $c: A=HEAD(headd) B=v2, mode 0, 8 rounds"
  timeout 600 $PY abba_short.py --a headd/flint_mid.so --b v2/flint_mid.so --ckpt $CK --case $c --rounds 8 --mode 0 --out abba_${c}_mode0.json 2>&1 | grep -v amdgpu.ids
done
st "re-profile v2 (MK_PROFILE)"
(cd prof2 && hipcc -O2 -Wall -shared -fPIC -DMK_PROFILE --offload-arch=gfx1150 flint_mid_megakernel_prof.hip -o flint_mid_prof.so > build.log 2>&1); st "prof build exit $?"
timeout 300 $PY mk_prof.py --so prof2/flint_mid_prof.so --src prof2/flint_mid_megakernel_prof.hip --ckpt $CK --prod v2/flint_mid.so --n-out 36 --workers 16 --reps 3 --out prof_v2_w16.json 2>&1 | grep -v amdgpu.ids
st DONE
journalctl -k --since "-20 min" --no-pager 2>/dev/null | grep -iE "ring.*timeout|wedged|gpu reset" | tail -3
