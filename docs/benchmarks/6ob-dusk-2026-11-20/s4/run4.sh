#!/usr/bin/env bash
# rec-6ob step 4 on dusk: build variants, resources, byte-identity gate vs base (2fd83cb), per-op timing
# for gate passers. Standalone gemv launches only (ms). Busy-gated; cotenant state logged.
set -u
R=~/rec-ve/6ob/s4; cd $R
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || echo idle),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || echo idle),qwen:$(systemctl --user is-active llama-qwen)"; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(cat $B) | $(cot)"; }
VARS=${VARS:-"va vc2048 vcgen"}
for d in base $VARS; do
  (cd $d && hipcc -O2 -Wall -shared -fPIC --offload-arch=gfx1150 flint_mid_megakernel.hip -o flint_mid.so > build.log 2>&1); st "build $d exit $?"
  grep -m2 " error" $d/build.log
done
./res.sh base/flint_mid_megakernel.hip $(for v in $VARS; do echo $v/flint_mid_megakernel.hip; done)
for v in $VARS; do
  st "GATE + timing: $v vs base"
  timeout 900 $PY gate7.py $R/base/flint_mid.so $R/$v/flint_mid.so --time --rounds 6 2>&1 | grep -v amdgpu.ids
done
st DONE
