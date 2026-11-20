#!/usr/bin/env bash
# rec-6ob step 3: repeat the standalone q4 bench per build, interleaved, busy-gated; count FAILs
# (nondeterminism hunt) and collect per-shape latencies. ms launches only.
set -u
cd ~/rec-ve/6ob/s3
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || echo idle),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || echo idle),qwen:$(systemctl --user is-active llama-qwen)"; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(cat $B)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; return 1; }
BUILDS=${BUILDS:-"headd v1 v2"}; REPS=${REPS:-6}
for d in $BUILDS; do (cd $d && [ -f flint_mid.so ] || hipcc -O2 -Wall -shared -fPIC --offload-arch=gfx1150 flint_mid_megakernel.hip -o flint_mid.so > build.log 2>&1); done
: > rep.log
for i in $(seq 1 $REPS); do
  for d in $BUILDS; do
    gate || { echo GATE TIMEOUT; exit 1; }
    pre="busy=$(cat $B) $(cot)"
    (cd $d && timeout 120 $PY mid_mk_driver.py --q4-bench 2>&1 | grep q4_gemv) | sed "s/^/$d rep$i | /" >> rep.log
    echo "[$(date -u +%T)Z] $d rep$i | pre $pre | post busy=$(cat $B) $(cot)" >> rep.log
  done
done
python3 - <<'EOF'
import re, collections, statistics
rows = collections.defaultdict(list); fails = collections.Counter(); runs = collections.Counter()
for l in open("rep.log"):
    m = re.match(r"(\S+) rep\d+ \| q4_gemv (\S+)\s*:\s*([\d.]+) ms .*-> (PASS|FAIL)", l)
    if m:
        b, shape, ms, st = m.groups(); rows[(b, shape)].append(float(ms)); runs[b] += 1
        if st == "FAIL": fails[(b, shape)] += 1
for (b, shape), v in sorted(rows.items()):
    print(f"{b:6s} {shape:22s} median {statistics.median(v):7.3f} ms  min {min(v):7.3f}  max {max(v):7.3f}  n={len(v)}  FAIL={fails[(b, shape)]}")
EOF
