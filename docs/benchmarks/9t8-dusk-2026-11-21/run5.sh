#!/usr/bin/env bash
# rec-6ob step 5 (rec-6ob.1) on dusk: VD = double-buffered LDS staging on the K=2048 path (rec-6u2 fix 3),
# generic path = VCgen (39629e7). G0: byte identity vs base 2fd83cb (gate7: 7 shapes x 40 runs) + rows%4 /
# iteration histograms (lmh_dbg) at lm_head and q_proj; per-op timing only if G0 passes.
# Standalone gemv launches only (ms). Busy-gated; cotenant state logged; nothing on dusk is touched.
set -u
S4=~/rec-ve/6ob/s4; R=~/rec-ve/6ob/s5; mkdir -p $R; cd $R
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || (curl -s -m2 -o /dev/null $H:8080/health && echo idle || echo down)),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || (curl -s -m2 -o /dev/null $H:8090/health && echo idle || echo down)),qwen:$(systemctl --user is-active llama-qwen)"; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(cat $B) | $(cot)"; }
cp -f $S4/gate7.py $S4/q4_lab.py $S4/mid_mk_driver.py $S4/res.sh .   # lmh_dbg.py is scp'd in from the repo
ln -sfn $S4/base base; ln -sfn $S4/vcgen vcgen
VARS=${VARS:-vd}
for v in $VARS; do
  (cd $v && hipcc -O2 -Wall -shared -fPIC --offload-arch=gfx1150 flint_mid_megakernel.hip -o flint_mid.so > build.log 2>&1); st "build $v exit $?"
  grep -m3 -E " error|warning" $v/build.log
  T=$(mktemp -d); cp $v/flint_mid_megakernel.hip $T/; (cd $T && hipcc -O2 -shared -fPIC --offload-arch=gfx1150 --save-temps flint_mid_megakernel.hip -o k.so > /dev/null 2>&1); cp -f $T/*gfx1150.s $R/${v}_gfx1150.s 2>/dev/null; rm -rf $T
done
./res.sh base/flint_mid_megakernel.hip vcgen/flint_mid_megakernel.hip $(for v in $VARS; do echo $v/flint_mid_megakernel.hip; done)
for v in $VARS; do
  st "G0 + timing: $v vs base(2fd83cb)"
  timeout 900 $PY gate7.py $R/base/flint_mid.so $R/$v/flint_mid.so --time --rounds 6 2>&1 | grep -v amdgpu.ids
  st "rows%4 / iteration histograms: $v vs base (lm_head 150144, q_proj 4096)"
  timeout 300 $PY lmh_dbg.py $R/base/flint_mid.so $R/$v/flint_mid.so 2>&1 | grep -v amdgpu.ids
  timeout 300 $PY lmh_dbg.py $R/base/flint_mid.so $R/$v/flint_mid.so 4096 2>&1 | grep -v amdgpu.ids
done
st DONE
journalctl -k --since "-30 min" --no-pager 2>/dev/null | grep -iE "ring.*timeout|wedged|gpu reset" | tail -3
