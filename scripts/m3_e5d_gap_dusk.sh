#!/usr/bin/env bash
# rec-e5d gap closure window: gqa depth8192 fp32-KV emit + full-config fresh-session determinism
# (wrap512, fp32kv+bf16kv, 2 fresh sessions) for lop4 and gqa. Script hard stop 14 min (cap ~20 min :8080 down incl. pre-start).
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
H=~/rec-ve/gepa/h; G="$PY $H/scripts/m3_depth_gate.py --bank $HOME/xh7/flint-int4.bank --ckpt $HOME/xh7/flint-int4"
LOP=~/rec-ve/bisect/e5d/full_lop4.so; GQA=~/rec-ve/bisect/e5d/full_gqa.so
OUT=~/rec-ve/m3window/e5dgap-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s); HARD=840
log(){ echo "$(date -u +%H:%M:%SZ) +$(( $(date +%s) - T0 ))s $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl:$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8990/health)/$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8991/health) llama-qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) display:$(systemctl is-active display-manager 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
left(){ echo $(( HARD - ($(date +%s) - T0) )); }
step(){ local name=$1 budget=$2; shift 2; local l=$(left); if [ $l -lt $budget ]; then log "SKIP $name: ${l}s left < ${budget}s"; return 9; fi
  log "START $name (budget ${budget}s)"; timeout $((l - 15)) "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  grep -E "ROW|GATE|INVALID|cap|deadline|REFUS|Traceback" "$OUT/$name.log" | cut -c1-220 | tee -a "$OUT/window.log"; log "END $name rc=$rc"; }
log "preflight e5d gap closure"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $LOP $GQA | tee -a "$OUT/window.log"
step gqa_fp32kv 290 $G --lib $GQA --out "$OUT/gqa_fp32kv" --case depth8192 --builds fp32kv --repeats 1 --budget-s 400
step lop4_det512 150 $G --lib $LOP --out "$OUT/lop4_det512" --case wrap512 --builds fp32kv,bf16kv --repeats 2 --budget-s 200
step gqa_det512 150 $G --lib $GQA --out "$OUT/gqa_det512" --case wrap512 --builds fp32kv,bf16kv --repeats 2 --budget-s 200
log "done -> signal ops_flint to restart :8080"
echo "$OUT"
