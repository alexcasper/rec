#!/usr/bin/env bash
# rec-ywy mid-8L int8-KV gate emits (dusk, :8080 up): ppl fp32kv+int8kv on eval_set_full_v1 per read-path build,
# then 2 fresh-session int8kv depth8192 emits (determinism) on the lop8 build. Usage: i8_gate.sh
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
R=~/rec-ve/i8; H=$R/h; O=$R/gate-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $O
CK=~/rec-ve/ckpt-pp2/mid-prod; MB="--bank $HOME/rec-ve/m3prep/mid-prod.bank --off $HOME/rec-ve/m3prep/mid-prod.off.npy --ckpt $CK"
log(){ echo "$(date -u +%H:%M:%SZ) $* | busy=$(cat /sys/class/drm/card1/device/gpu_busy_percent) 8080=$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090=$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl_estab=$(ss -tn state established '( sport = :8990 or sport = :8991 )' 2>/dev/null | tail -n +2 | wc -l) MemAvail=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo)" | tee -a $O/gate.log; }
cd $H
for n in c6s0 d0 s0; do
  log "[$n] ppl start sha $(sha256sum $R/$n/flint_mid.so | cut -c1-16)"
  $PY scripts/m3_depth_gate.py --rehearsal --lib $R/$n/flint_mid.so $MB --out $O/ppl_$n --ppl-set engine/validation/megakernel_mid/qgate/eval_set_full_v1.json --ppl-builds fp32kv,int8kv --budget-s 500 2>&1 | grep -E "PPL|Error|error|REFUS" | cut -c1-170 | tee -a $O/gate.log
done
log "[c6s0] depth8192 int8kv x2 fresh"
$PY scripts/m3_depth_gate.py --rehearsal --lib $R/c6s0/flint_mid.so $MB --out $O/d8k_c6s0 --case depth8192 --builds int8kv --repeats 2 --budget-s 400 2>&1 | grep -E "GATE|DETERMINISM|Error" | cut -c1-170 | tee -a $O/gate.log
log "done $O"
