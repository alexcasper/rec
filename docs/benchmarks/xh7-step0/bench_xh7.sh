#!/usr/bin/env bash
# rec-xh7 incumbent row: llama.cpp-Vulkan decode on dusk 890M, hygiene protocol.
# States: S0 = llama-qwen :8091 STOPPED (ours), flint :8080 + E4B :8090 up/idle (untouched)
#         S1 = llama-qwen :8091 up/idle too.  ABBA over states: S0 S1 S1 S0.
# Every line records cotenant state; runs are gated busy<5 x3 @0.3 s and on idle slots.
set -u
cd ~/rec-ve/xh7
H=100.98.73.29
BIN=~/llama.cpp/build-vulkan/bin
B=/sys/class/drm/card1/device/gpu_busy_percent
OUT=${OUT:-bench_xh7_main.jsonl}
LOG=${LOG:-bench_xh7_main.log}
: > "$OUT"
restart_qwen() { systemctl --user start llama-qwen; for i in $(seq 1 60); do curl -s -m2 $H:8091/health | grep -q ok && return 0; sleep 2; done; echo "WARN qwen health not ok" >> $LOG; }
trap 'restart_qwen; echo "EXIT $(date -u +%T) qwen=$(systemctl --user is-active llama-qwen)" >> $LOG' EXIT

busy_slots() { local n=0; for p in "$@"; do n=$((n + $(curl -s -m3 $H:$p/slots | grep -o '"is_processing":true' | wc -l))); done; echo $n; }
cot() { # cotenant state string
  local q; q=$(systemctl --user is-active llama-qwen)
  echo "flint:$(curl -s -m2 $H:8080/health | grep -q ok && echo up || echo DOWN),e4b:$(curl -s -m2 $H:8090/health | grep -q ok && echo up || echo DOWN),qwen:$([ "$q" = active ] && echo up || echo stopped)"
}
gate() { local s=0 t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 600 ]; do
    if [ "$(cat $B)" -lt 5 ] && [ "$(busy_slots 8080 8090 8091)" = 0 ]; then s=$((s+1)); [ $s -ge 3 ] && return 0; else s=0; fi
    sleep 0.3; done; return 1; }
rec() { # state tag json-fragment
  echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"window\":$W,\"state\":\"$1\",\"cotenants\":\"$(cot)\",\"vram_used_gib\":$(awk '{printf "%.2f",$1/2^30}' /sys/class/drm/card1/device/mem_info_vram_used),$2}" >> "$OUT"; }

mid_bench() { # model depth
  gate || { rec "$S" "\"run\":\"mid\",\"model\":\"$1\",\"depth\":$2,\"error\":\"gate timeout\""; return; }
  local pre post j
  pre=$(busy_slots 8080 8090 8091)
  j=$(timeout 600 $BIN/llama-bench -m "$1" -ngl 99 -fa on -p 0 -n 32 -d "$2" -r 5 -o jsonl 2>>$LOG | tail -1)
  post=$(busy_slots 8080 8090 8091)
  rec "$S" "\"run\":\"mid\",\"model\":\"$1\",\"depth\":$2,\"slots_busy_pre\":$pre,\"slots_busy_post\":$post,\"bench\":${j:-null}"
}
full_flint() { # seed  -- incumbent full model through the running :8080 server
  gate || { rec "$S" "\"run\":\"full\",\"error\":\"gate timeout\""; return; }
  local pre post r
  pre=$(busy_slots 8090 8091)
  r=$(python3 - "$1" <<'EOF'
import json, random, sys, urllib.request
rng = random.Random(int(sys.argv[1]))
prompt = [150000] + [rng.randrange(1000, 149000) for _ in range(8191)]
req = {"prompt": prompt, "n_predict": 64, "temperature": 0, "top_k": 1, "cache_prompt": False, "ignore_eos": True}
d = json.loads(urllib.request.urlopen(urllib.request.Request("http://100.98.73.29:8080/completion",
    json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=900).read())
t = d["timings"]
print(json.dumps({k: t[k] for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second")}))
EOF
)
  post=$(busy_slots 8090 8091)
  rec "$S" "\"run\":\"full\",\"model\":\"flint-preview-UD-Q3_K_XL (:8080 server)\",\"depth\":8192,\"other_slots_busy_pre\":$pre,\"other_slots_busy_post\":$post,\"timings\":${r:-null}"
}

echo "START $(date -u +%T) $(cot)" > $LOG
W=0
for S in S0 S1 S1 S0; do
  W=$((W+1))
  if [ $S = S0 ]; then systemctl --user stop llama-qwen; sleep 3; else [ "$(systemctl --user is-active llama-qwen)" = active ] || restart_qwen; fi
  echo "window $W $S $(date -u +%T) $(cot)" >> $LOG
  if [ $((W % 2)) = 1 ]; then M1=mid-ud-iq3xxs.gguf; M2=mid-ud-iq3s.gguf; else M1=mid-ud-iq3s.gguf; M2=mid-ud-iq3xxs.gguf; fi
  for d in 8192 512; do mid_bench $M1 $d; mid_bench $M2 $d; done
  full_flint $((1000 + W))
done
echo "DONE $(date -u +%T)" >> $LOG
