#!/usr/bin/env bash
# rec-u9q G1 (conformance) + G2 (decode bench) on albion R9700 gfx1201, HIP device 0 (card1).
# R9700 drives no display (albion's HDMI is on the gfx1036 iGPU): long persistent launches are safe here.
set -u
R=~/rec-ve/u9q; cd $R
export LD_LIBRARY_PATH=/opt/rocm/lib:$(ls -d /nix/store/*-gcc-14.2.1.20250322-lib | head -1)/lib:$(ls -d /nix/store/*-zstd-1.5.7 | head -1)/lib:$(ls -d /nix/store/*-zlib-1.3.2 | head -1)/lib
unset HSA_OVERRIDE_GFX_VERSION
export HIP_VISIBLE_DEVICES=0
PY=~/rec-ve/.venv/bin/python
D=$R/b-full/flint_full_driver.py
BK=$R/flint-int4.bank; CK=$R/flint-int4
CHK=$R/gk/engine/validation/megakernel_mid/mid_mk_check.py
B=/sys/class/drm/card1/device
busy(){ cat $B/gpu_busy_percent 2>/dev/null || { [ "$(cat $B/power/runtime_status)" = suspended ] && echo 0 || echo 100; }; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(busy)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; echo "GATE TIMEOUT busy=$(busy)"; return 1; }
stamp(){ echo "[$(date -u +%T)Z] $*"; }

stamp "== G1 emit (fp32kv, fp32_bf16kv, bf16kv) x (short4, chat, wrap512)"
for W in 16 64; do
  gate || exit 1
  rm -rf suite_w$W
  timeout 3000 $PY $D --bank $BK --ckpt $CK --emit-suite suite_w$W --builds fp32kv,fp32_bf16kv,bf16kv --workers $W 2>&1 | grep -v amdgpu.ids
  stamp "emit w$W exit ${PIPESTATUS[0]}"
done
$PY - <<'EOF'
import numpy as np, glob, os
fs = sorted(glob.glob(os.path.expanduser("~/rec-ve/u9q/suite_w16/*.npz"))); same = 0
for f in fs:
    a, b = np.load(f), np.load(f.replace("suite_w16", "suite_w64"))
    d = [k for k in a.files if k != "impl" and not np.array_equal(a[k], b[k])]
    same += not d
    if d: print("workers 16 vs 64 DIFF", os.path.basename(f), d)
print(f"worker invariance: {same}/{len(fs)} npz byte-identical at workers 16 vs 64")
EOF
# sign-off matrix: fp32kv + fp32_bf16kv only (bf16kv = production build, smoke, checked separately)
mkdir -p signoff_in && rm -f signoff_in/*.npz && cp suite_w16/*_fp32kv.npz suite_w16/*_fp32_bf16kv.npz signoff_in/
stamp "== G1 sign-off matrix (CPU reference)"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$R/gk/engine/src /usr/bin/env time -f "matrix wall %e s, peak RSS %M KB" \
  timeout 7200 $PY $CHK matrix --ckpt $CK signoff_in > signoff_matrix.txt 2> signoff_matrix.err; echo "matrix exit $?" >> signoff_matrix.txt
cat signoff_matrix.txt; grep "matrix wall" signoff_matrix.err
stamp "== production bf16kv smoke vs CPU reference (diagnostic; bf16 activations are not a sign-off build)"
for c in short4 chat wrap512; do
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$R/gk/engine/src timeout 1800 $PY $CHK check --ckpt $CK --ref-kv bf16 suite_w16/${c}_bf16kv.npz 2>/dev/null \
    | $PY -c "import sys,json; t=sys.stdin.read(); j=json.loads(t[:t.rindex('}')+1]); print('${c}_bf16kv', 'tokens_exact', j['tokens_exact'], 'last_rel %.2e step_rel %.2e' % (j['last_logits_rel'], j.get('step_logits_max_rel') or float('nan')), 'pass', j['pass'])"
done

stamp "== G2 bench (production bf16-KV build, mode 0), R1-R8 differential ABBA"
for W in 16 32 64; do
  gate || continue
  timeout 3000 $PY $D --bank $BK --ckpt $CK --bench bench_full_w$W.json --mode 0 --workers $W 2>&1 | grep -E "RESULT|pair|warmup|REFUS|rror"
done
stamp "== G2 context: mid-8L megakernel on the same R9700 (mode 0)"
for W in 16 64; do
  gate || continue
  (cd $R/b-refac && timeout 1500 $PY mid_mk_driver.py --bench --ckpt ~/rec-ve/r109/ckpt/mid-prod --workers=$W --mode=0 > $R/bench_mid_w$W.json 2> $R/bench_mid_w$W.err)
  $PY -c "import json; d=json.load(open('$R/bench_mid_w$W.json')); [print('mid w%s' % r['workers'], r.get('decode32_after_512_differential_tok_s'), r.get('decode32_after_512_pair_min_tok_s'), r.get('decode32_after_512_pair_max_tok_s'), 'valid', r.get('valid')) for r in d['runs']]"
done
stamp DONE
