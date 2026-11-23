#!/usr/bin/env bash
# flint-y02: llama.cpp arm on ops_rec mid-8L shape, dusk 890M gfx1150 Vulkan.
# Protocol per /mnt/d/ops/rec/docs/benchmarks/run-definitions.md:
#   prefill512: 3 warmup + 3 timed   |   decode32: after same prefill, temp 0, 32 tokens
# llama-cli timing window note: llama.cpp's "eval time" covers all 32 decode steps
# (no separate decode warmup steps) and prompt-eval time covers all 512 (no warmup);
# this is the closest sync-bounded window llama.cpp exposes externally.
set -euo pipefail
GGUF="$HOME/flint-mid/flint-mid-8L-f16.gguf"
BIN="$HOME/llama.cpp/build-vulkan/bin/llama-cli"
LOG="$HOME/flint-mid/bench-f16-$(date -u +%Y%m%dT%H%M%S).log"
OUT="$HOME/flint-mid/bench-f16-summary.txt"
CTX=1024
# 512-token prompt: repeated BOS special tokens (synthetic vocab has no text tokens;
# work is shape-identical to the ops_rec 512-token prefill)
PROMPT=$(python3 -c "print('<|begin_of_text|>' * 512)")
TPL='{% for m in messages %}{{ m.content }}{% endfor %}'

echo "=== cotenant gate $(date -u +%FT%TZ) ===" | tee -a "$LOG"
for port in 8080 8091; do
  slots=$(curl -s "http://127.0.0.1:$port/slots" || true)
  busy=$(echo "$slots" | grep -c '"is_processing": true' || true)
  echo "port $port busy slots: $busy" | tee -a "$LOG"
  if [ "${busy:-0}" != "0" ]; then echo "ABORT: cotenant busy on $port" | tee -a "$LOG"; exit 1; fi
done
gpu=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | tr '\n' ' ')
echo "gpu_busy: $gpu" | tee -a "$LOG"

run_once () {  # $1 = label
  echo "--- $1 $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$BIN" -m "$GGUF" -ngl 99 -c $CTX -fa on --temp 0 -s 0 -n 32 -st \
    --chat-template "$TPL" --no-display-prompt -p "$PROMPT" 2>&1 \
    | grep -E "prompt eval time|eval time|Prompt:|Generation:" | tee -a "$LOG"
}

for i in 1 2 3; do run_once "warmup$i"; done
for i in 1 2 3; do run_once "timed$i"; done
echo "=== done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
grep -E "timed[0-9]|prompt eval time|eval time" "$LOG" | tail -12 > "$OUT"
cat "$OUT"
