#!/usr/bin/env bash
# Discriminator window: DEFAULT full build (split-K OFF, GEPA_SHRINK=0 pinned, 288f4fa) at depth 8192, rec-kcb v1 tiers.
#  D1 fp32kv x1 (binding fp32-KV tier vs torch fp32 CPU)   C bf16kv (production) x1 + 12 cap-safe ABBA speed pairs
#  D2 fp32kv second fresh session (determinism). Hard stop 42 min; every launch <= 0.25 s (tool-enforced).
# Run ONLY after ops_flint confirms :8080 DOWN.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
H=~/rec-ve/gepa/h; G="$PY $H/scripts/m3_depth_gate.py --bank $HOME/xh7/flint-int4.bank --ckpt $HOME/xh7/flint-int4"
DEF=~/rec-ve/bisect/flint_full_def_288f4fa.so
OUT=~/rec-ve/m3window/gate8kdef-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s); HARD=2520
log(){ echo "$(date -u +%H:%M:%SZ) +$(( $(date +%s) - T0 ))s $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl:$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8990/health)/$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8991/health) llama-qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) display:$(systemctl is-active display-manager 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
left(){ echo $(( HARD - ($(date +%s) - T0) )); }
step(){ local name=$1 budget=$2; shift 2; local l=$(left); if [ $l -lt $budget ]; then log "SKIP $name: ${l}s left < ${budget}s"; return 9; fi
  log "START $name (budget ${budget}s)"; "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  grep -E "GATE|ROW|DETERMINISM|INVALID|cap|deadline|REFUS|Traceback|speed" "$OUT/$name.log" | cut -c1-210 | tee -a "$OUT/window.log"; log "END $name rc=$rc"; }
log "preflight gate8k-default"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $DEF | tee -a "$OUT/window.log"
step D1_fp32kv8k 900 $G --lib $DEF --out "$OUT/D_8k" --case depth8192 --builds fp32kv --repeats 1 --budget-s 900
step C_bf16kv8k 900 $G --lib $DEF --out "$OUT/C_8k" --case depth8192 --builds bf16kv --repeats 1 --pairs 12 --budget-s 900
step D2_fp32kv8k 800 $G --lib $DEF --out "$OUT/D_8k" --case depth8192 --builds fp32kv --repeats 1 --rep-offset 1 --budget-s 800
log "done -> signal ops_flint to restart :8080"
echo "$OUT"
