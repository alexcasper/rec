#!/usr/bin/env bash
# rec-6ob phase profile on dusk: short-ctx launches only (prompt4 + 36 tokens, ~0.3 s each).
set -u
cd ~/rec-ve/6ob/lab
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
B=/sys/class/drm/card1/device/gpu_busy_percent
H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || echo idle),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || echo idle),qwen:$(systemctl --user is-active llama-qwen)"; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(cat $B)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; return 1; }
for W in ${WORKERS:-16}; do
  gate || { echo "GATE TIMEOUT"; exit 1; }
  echo "[$(date -u +%T)Z] workers=$W | busy=$(cat $B) | $(cot)"
  timeout 300 $PY mk_prof.py --so flint_mid_prof.so --src flint_mid_megakernel_prof.hip --ckpt ~/rec-ve/ckpt-pp2/mid-prod \
      --prod flint_mid.so --n-out 36 --workers $W --reps 3 --out prof_w$W.json
  echo "[$(date -u +%T)Z] done exit $? | busy=$(cat $B) | $(cot)"
done
journalctl -k --since "-5 min" --no-pager 2>/dev/null | grep -iE "ring.*timeout|wedged|gpu reset" | tail -3
