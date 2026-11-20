#!/usr/bin/env bash
# rec-6ob step 3 on dusk: (a) contraction-independence of the v2 gemv, (b) v2 suite emit (short4+chat
# only, launches <= ~0.26 s), (c) spec (a) sign-off matrix vs the pinned torch fp32 CPU reference.
set -u
R=~/rec-ve/6ob/s3; cd $R
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1
unset HSA_OVERRIDE_GFX_VERSION
CK=~/rec-ve/ckpt-pp2/mid-prod
B=/sys/class/drm/card1/device/gpu_busy_percent; H=100.98.73.29
cot(){ echo "flint:$(curl -s -m2 $H:8080/slots | grep -q '"is_processing":true' && echo busy || echo idle),e4b:$(curl -s -m2 $H:8090/slots | grep -q '"is_processing":true' && echo busy || echo idle),qwen:$(systemctl --user is-active llama-qwen)"; }
gate(){ local s=0 t=$(date +%s); while [ $(( $(date +%s)-t )) -lt 300 ]; do [ "$(cat $B)" -lt 5 ] && s=$((s+1)) || s=0; [ $s -ge 3 ] && return 0; sleep 0.3; done; return 1; }
st(){ echo "[$(date -u +%T)Z] $* | busy=$(cat $B) | $(cot)"; }

st "(a) contraction independence: v2 built with -ffp-contract=off vs default"
mkdir -p v2off && cp -f v2/flint_mid_megakernel.hip v2/mid_mk_driver.py v2/det.py v2off/
(cd v2off && hipcc -O2 -Wall -shared -fPIC -ffp-contract=off --offload-arch=gfx1150 flint_mid_megakernel.hip -o flint_mid.so > build.log 2>&1); st "v2off build exit $?"
gate; $PY - <<'EOF'
import ctypes, numpy as np, sys
sys.path.insert(0, "v2")
from mid_mk_driver import f32_to_bf16
vp = ctypes.c_void_p
libs = []
for d in ("v2", "v2off"):
    lb = ctypes.CDLL(f"{d}/flint_mid.so"); lb.flint_q4_gemv_host.restype = ctypes.c_int
    lb.flint_q4_gemv_host.argtypes = [vp] * 6 + [ctypes.c_int] * 5 + [vp]; libs.append(lb)
ok = True
for n, k in [(512, 2048), (2048, 512), (2048, 1024), (2048, 4096), (150144, 2048)]:
    rng = np.random.default_rng(7)
    w = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    s = f32_to_bf16(((rng.random((n, k // 64)) * 0.5 + 0.75) * 0.003).astype(np.float32))
    b = f32_to_bf16((-0.02 * (1 + 0.05 * rng.standard_normal((n, k // 64)))).astype(np.float32))
    x = f32_to_bf16(rng.standard_normal(k).astype(np.float32))
    outs = []
    for lb in libs:
        o = np.zeros(n, np.float32); ms = ctypes.c_float()
        assert lb.flint_q4_gemv_host(w.ctypes.data, s.ctypes.data, b.ctypes.data, x.ctypes.data, None, o.ctypes.data, n, k, 0, 0, 1, ctypes.byref(ms)) == 0
        outs.append(o.view(np.uint32))
    same = np.array_equal(*outs); ok &= same
    print(f"  {n}x{k}: fp-contract=fast vs off -> {'IDENTICAL' if same else 'DIFF (%d rows)' % int((outs[0] != outs[1]).sum())}")
print("CONTRACTION-INDEPENDENT:", ok)
EOF

st "(b) emit v2 suite: short4,chat x fp32kv,fp32_bf16kv,bf16kv (+experts), workers=16"
gate || { st "GATE TIMEOUT"; exit 1; }
(cd v2 && rm -rf out && timeout 300 $PY mid_mk_driver.py --ckpt $CK --emit-suite out --builds fp32kv,fp32_bf16kv,bf16kv --cases short4,chat --emit-experts --workers=16 > emit.log 2>&1); st "emit exit $?"
grep -E "wrote|rror" v2/emit.log | cut -c1-100
st "(c) spec (a) matrix vs pinned torch fp32 CPU reference (CPU only)"
CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONPATH=$R/gk/engine/src timeout 3000 $PY gk/engine/validation/megakernel_mid/mid_mk_check.py matrix --ckpt $CK v2/out > matrix_v2.txt 2> matrix_v2.err; echo "matrix exit $?" >> matrix_v2.txt
cat matrix_v2.txt
st DONE
journalctl -k --since "-20 min" --no-pager 2>/dev/null | grep -iE "ring.*timeout|wedged|gpu reset" | tail -3
