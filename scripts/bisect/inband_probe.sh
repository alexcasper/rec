#!/usr/bin/env bash
# on-box cotenant gate: :8080 health + in-band decode tok/s (llama-server /completion timings), tailscale IP bind
H=http://100.98.73.29:8080
echo "$(date -u +%H:%M:%SZ) health=$(curl -s -m3 -o /dev/null -w %{http_code} $H/health)"
for i in 1 2; do
curl -s -m 60 $H/completion -H 'Content-Type: application/json' \
 -d '{"prompt":"Write a short paragraph about the history of the printing press.","n_predict":64,"temperature":0,"cache_prompt":false}' \
 | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("probe decode %.2f tok/s (%d tok), prefill %.1f tok/s" % (t["predicted_per_second"], t["predicted_n"], t["prompt_per_second"]))'
done
