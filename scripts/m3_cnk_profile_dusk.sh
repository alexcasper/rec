#!/usr/bin/env bash
# rec-cnk profile window: fp32kv emits straddling the first fp32 routing tie (pos 3232 on the seeded prompt):
# depth3072 (before) and depth4096 (after), DEFAULT build and split-K+SHRINK=0. Hard stop 15 min; launches <= 0.25 s.
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
H=~/rec-ve/gepa/h; G="$PY $H/scripts/m3_depth_gate.py --bank $HOME/xh7/flint-int4.bank --ckpt $HOME/xh7/flint-int4"
DEF=~/rec-ve/bisect/flint_full_def_288f4fa.so; SPLIT=~/rec-ve/bisect/flint_full_split_288f4fa.so
OUT=~/rec-ve/m3window/cnk-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"; T0=$(date +%s); HARD=900
log(){ echo "$(date -u +%H:%M:%SZ) +$(( $(date +%s) - T0 ))s $* | 8080:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) 8090:$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8090/health) qwen3vl:$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8990/health)/$(curl -s -m1 -o /dev/null -w %{http_code} http://127.0.0.1:8991/health) llama-qwen:$(systemctl --user is-active llama-qwen 2>/dev/null) display:$(systemctl is-active display-manager 2>/dev/null) busy=$(head -1 /sys/class/drm/card1/device/gpu_busy_percent) MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo) vram_free=$(awk -v t=$(cat /sys/class/drm/card1/device/mem_info_vram_total) -v u=$(cat /sys/class/drm/card1/device/mem_info_vram_used) 'BEGIN{printf "%.1fGiB",(t-u)/2^30}')" | tee -a "$OUT/window.log"; }
left(){ echo $(( HARD - ($(date +%s) - T0) )); }
step(){ local name=$1 budget=$2; shift 2; local l=$(left); if [ $l -lt $budget ]; then log "SKIP $name: ${l}s left < ${budget}s"; return 9; fi
  log "START $name (budget ${budget}s)"; "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  grep -E "GATE|INVALID|cap|deadline|REFUS|Traceback" "$OUT/$name.log" | cut -c1-200 | tee -a "$OUT/window.log"; log "END $name rc=$rc"; }
log "preflight cnk profile"
[ "$(systemctl --user is-active llama-qwen || true)" != active ] || { log "ABORT: llama-qwen active"; exit 2; }
sha256sum $DEF $SPLIT | tee -a "$OUT/window.log"
step def3072 220 $G --lib $DEF --out "$OUT/def3072" --case depth3072 --builds fp32kv --repeats 1 --budget-s 220
step def4096 300 $G --lib $DEF --out "$OUT/def4096" --case depth4096 --builds fp32kv --repeats 1 --budget-s 300
step spl3072 150 $G --lib $SPLIT --out "$OUT/spl3072" --case depth3072 --builds fp32kv --repeats 1 --budget-s 150
step spl4096 200 $G --lib $SPLIT --out "$OUT/spl4096" --case depth4096 --builds fp32kv --repeats 1 --budget-s 200
log "done -> signal ops_flint to restart :8080"
echo "$OUT"
