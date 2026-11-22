#!/usr/bin/env bash
# rec-1f4: run `qgate.py emit` on dusk for one or more kernel builds. Launches <= 40 positions (<= 0.25 s,
# enforced per launch inside qgate.py); busy-gated before each build; cotenant state logged; flint :8080 /
# E4B :8090 / llama-qwen are only probed, never touched.
#   SOS="name=/path/flint_mid.so ..." ./qgate_emit.sh
set -u
G=~/rec-ve/1f4/gk/engine/validation/megakernel_mid/qgate; OUT=~/rec-ve/1f4/emit; mkdir -p $OUT
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
CK=~/rec-ve/ckpt-pp2/mid-prod
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || (curl -s -m2 -o /dev/null $H:8080/health && echo idle || echo down)),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || (curl -s -m2 -o /dev/null $H:8090/health && echo idle || echo down)),qwen:$(systemctl --user is-active llama-qwen)"; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(cat $B)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; return 1; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(cat $B) | $(cot)"; }
for kv in $SOS; do
  name=${kv%%=*}; so=${kv#*=}
  gate || { st "GATE TIMEOUT"; exit 1; }
  st "emit $name ($so)"
  timeout 900 $PY $G/qgate.py emit --so $so --ckpt $CK --out $OUT/$name 2>&1 | grep -v amdgpu.ids
  st "emit $name exit ${PIPESTATUS[0]}"
done
journalctl -k --since "-30 min" --no-pager 2>/dev/null | grep -iE "ring.*timeout|wedged|gpu reset" | tail -3
st DONE
