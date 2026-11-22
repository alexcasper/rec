#!/usr/bin/env bash
# rec-kcb approved window (~25-30 min :8080 down), revised full-config tiers (docs/gates/rec-kcb-full-config-tiers-v1.md).
#  A  DEFAULT full build (split off, SHRINK=0): wrap512 x {fp32kv, fp32_bf16kv, bf16kv} x1  -> baseline bf16-KV divergence
#  B  split+SHRINK=0: teacher-forced ppl on eval_set_full_v1 (fp32kv, bf16kv)
#  D  split+SHRINK=0: depth8192 fp32kv x1 (binding fp32-KV tier)
#  C  split+SHRINK=0: depth8192 bf16kv (production) x2 fresh sessions (determinism) + 12 ABBA speed pairs after r0
#  D2 split+SHRINK=0: depth8192 fp32kv second fresh repeat, only if >= 7.5 min remain of the 27 min hard stop
# Run ONLY after ops_flint confirms :8080 DOWN. Every launch <= 0.25 s (tool-enforced).
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
H=~/rec-ve/gepa/h; G="$PY $H/scripts/m3_depth_gate.py --bank $HOME/xh7/flint-int4.bank --ckpt $HOME/xh7/flint-int4"
SPLIT=~/rec-ve/bisect/flint_full_split_288f4fa.so; DEF=~/rec-ve/bisect/flint_full_def_288f4fa.so
OUT=~/rec-ve/m3window/gate8k-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s); HARD=1620
log(){ echo "$(date -u +%H:%M:%SZ) +$(( $(date +%s) - T0 ))s $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl:$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8990/health)/$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8991/health) llama-qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) display:$(systemctl is-active display-manager 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
left(){ echo $(( HARD - ($(date +%s) - T0) )); }
step(){ local name=$1 budget=$2; shift 2; local l=$(left); if [ $l -lt $budget ]; then log "SKIP $name: ${l}s left < ${budget}s"; return 9; fi
  log "START $name (budget ${budget}s)"; "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  grep -E "GATE|PPL|ROW|DETERMINISM|INVALID|cap|deadline|REFUS|Traceback" "$OUT/$name.log" | cut -c1-210 | tee -a "$OUT/window.log"; log "END $name rc=$rc"; }
log "preflight gate8k"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $SPLIT $DEF | tee -a "$OUT/window.log"
step A_def512 160 $G --lib $DEF --out "$OUT/A_def512" --case wrap512 --builds fp32kv,fp32_bf16kv,bf16kv --repeats 1 --budget-s 160
step B_ppl 220 $G --lib $SPLIT --out "$OUT/B_ppl" --ppl-set $H/engine/validation/megakernel_mid/qgate/eval_set_full_v1.json --ppl-builds fp32kv,bf16kv --budget-s 220
step D_fp32kv8k 480 $G --lib $SPLIT --out "$OUT/D_8k" --case depth8192 --builds fp32kv --repeats 1 --budget-s 480
step C_bf16kv8k 800 $G --lib $SPLIT --out "$OUT/C_8k" --case depth8192 --builds bf16kv --repeats 2 --pairs 12 --budget-s 800
step D2_fp32kv8k 450 $G --lib $SPLIT --out "$OUT/D_8k" --case depth8192 --builds fp32kv --repeats 1 --rep-offset 1 --budget-s 450
log "done -> signal ops_flint to restart :8080"
echo "$OUT"
