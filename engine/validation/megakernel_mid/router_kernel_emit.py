"""rec-tan.3 step 1 (torch-free): run the mid megakernel with step_experts emission
over the calibration prompt set, writing one rec-mk-out/1 npz per (prompt, build).

Uses Gemini's driver (mid_mk_driver.py + flint_mid.so @ rec/gemini-kernels 4c1cb9a)
from --driver-dir. Never imports torch, because torch's bundled libhsa breaks the HIP
loader (rec-ju9). Run router_kernel_analyze.py as a separate process afterwards.

  python router_kernel_emit.py --driver-dir ~/rec-ve/megakernel_hip --ckpt CK --out DIR
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_tie_calibrate import prompts  # noqa: E402  (numpy-only import chain)

BUILDS = {"fp32kv": (2, "fp32"), "fp32_bf16kv": (1, "bf16")}  # driver mode, kv_dtype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--builds", default="fp32kv,fp32_bf16kv")
    ap.add_argument("--n-out", type=int, default=24)
    a = ap.parse_args()
    assert "torch" not in sys.modules
    sys.path.insert(0, str(a.driver_dir))
    import mid_mk_driver as D

    a.out.mkdir(parents=True, exist_ok=True)
    sha = D.shard_sha256(a.ckpt)
    bank, off, _ = D.pack_mid_bank(a.ckpt)
    lib = D.load_lib()
    sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, ctypes.c_int(600))
    try:
        for bname in a.builds.split(","):
            mode, kv = BUILDS[bname]
            for name, prompt in prompts(D.MID_VOCAB, a.seeds).items():
                p = np.array(prompt, np.int32)
                n = a.n_out
                tokens = np.zeros(n, np.int32)
                step_logits = np.zeros((n, D.MID_VOCAB), np.float32)
                last = np.zeros(D.MID_VOCAB, np.float32)
                experts = np.full((len(p) + n, D.MID_NL, D.MID_TOPK), -1, np.int32)
                ctrl = np.zeros(4, np.uint32)
                rc = lib.mid_session_run_ex2(sess, p.ctypes.data, ctypes.c_int(len(p)), tokens.ctypes.data,
                                             ctypes.c_int(n), step_logits.ctypes.data, last.ctypes.data,
                                             ctypes.c_int(a.workers), ctypes.c_int(mode), ctrl.ctypes.data,
                                             experts.ctypes.data)
                if rc != 0 or int(ctrl[2]) != 0:
                    print(f"FAIL {name} {bname}: rc={rc} control[2]={int(ctrl[2])}", flush=True)
                    return 1
                # A kernel that never launched also returns rc=0 / control[2]=0, leaving the
                # outputs at their memset values. Seen on dusk with HSA_OVERRIDE_GFX_VERSION=11.0.0
                # (hides the gfx1150 code object): tokens all 0, experts all -1.
                if (experts[:, 1:] < 0).any() or not np.isfinite(last).all() or not last.any():
                    print(f"FAIL {name} {bname}: kernel did not run (unwritten outputs); "
                          "unset HSA_OVERRIDE_GFX_VERSION for the HIP driver", flush=True)
                    return 1
                np.savez(a.out / f"{name}_{bname}.npz", schema=np.array("rec-mk-out/1"), case=np.array(name),
                         ckpt_sha256=np.array(sha),
                         impl=np.array(f"flint_mid_megakernel mode={mode} workers={a.workers} (4c1cb9a)"),
                         kv_dtype=np.array(kv), prompt=p, tokens=tokens, step_logits=step_logits,
                         last_logits=last, step_experts=experts)
                print(f"wrote {name}_{bname}.npz tokens[:4]={tokens[:4].tolist()}", flush=True)
    finally:
        lib.mid_session_free(sess)
    return 0


if __name__ == "__main__":
    sys.exit(main())
