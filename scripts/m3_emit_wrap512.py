"""Bedivere M3.2 gate support: emit the rec-pp2 wrap512 case (512-token prompt + 24 decode steps, sliding-window wrap)
on dusk via KV CONTINUATION (M3.1), so no launch exceeds ~0.2 s (display-iGPU rule). M3.1 proved continued launches
bit-identical to one launch, so the npz is the same rec-mk-out/1 payload the conformance suite writes and
mid_mk_check.py scores it unchanged (CPU torch fp32 reference, on dawn).

    python scripts/m3_emit_wrap512.py --so flint_mid.so --ckpt ~/rec-ve/ckpt-pp2/mid-prod --out DIR [--builds ...]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

vp = ctypes.c_void_p
MK = Path(__file__).resolve().parent.parent / "engine" / "native" / "megakernel_hip"
sys.path.insert(0, str(MK))
from mid_mk_driver import MID_NL, MID_TOPK, MID_VOCAB, case_matrix, pack_mid_bank, shard_sha256  # noqa: E402

BUILDS = {"fp32kv": (2, "fp32", "flint_mid_megakernel.hip:mid_generate_fp32kv (fp32-compute, fp32-KV)"),
          "fp32_bf16kv": (1, "bf16", "flint_mid_megakernel.hip:mid_generate_fp32_bf16kv (fp32-compute, bf16-KV)"),
          "bf16kv": (0, "bf16", "flint_mid_megakernel.hip:mid_generate (production bf16-KV build)")}
TARGET_S, CAP_S = 0.20, 0.25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--builds", default="fp32kv,fp32_bf16kv,bf16kv")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    lib = ctypes.CDLL(str(Path(a.so).resolve()))
    lib.mid_session_create.restype = vp
    lib.mid_session_create.argtypes = [vp, vp, ctypes.c_int]
    lib.mid_session_run_ex3.restype = ctypes.c_int
    lib.mid_session_run_ex3.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, vp, vp, ctypes.c_int, ctypes.c_int,
                                        vp, vp, ctypes.c_int]
    lib.mid_session_free.argtypes = [vp]
    ck = Path(a.ckpt)
    bank, off, _ = pack_mid_bank(ck)
    sha = shard_sha256(ck)
    case = case_matrix(MID_VOCAB)["wrap512"]
    prompt, n_out = [int(t) for t in case["prompt"]], int(case["n_out"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for b in a.builds.split(","):
        mode, kv_dtype, impl = BUILDS[b]
        sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, 600)
        longest, launches = 0.0, 0

        def run(pr, n, start):
            nonlocal longest, launches
            p = np.array(pr, np.int32)
            tok = np.zeros(max(n, 1), np.int32)
            step = np.zeros((max(n, 1), MID_VOCAB), np.float32)
            last = np.zeros(MID_VOCAB, np.float32)
            exp = np.full((len(p) + n, MID_NL, MID_TOPK), -1, np.int32)
            ctrl = np.zeros(4, np.uint32)
            t0 = time.perf_counter()
            rc = lib.mid_session_run_ex3(sess, p.ctypes.data, len(p), tok.ctypes.data, n,
                                         step.ctypes.data if n else None, last.ctypes.data, a.workers, mode,
                                         ctrl.ctypes.data, exp.ctypes.data, start)
            dt = time.perf_counter() - t0
            longest, launches = max(longest, dt), launches + 1
            if rc != 0 or ctrl[2] != 0 or ctrl[1] == 0:
                raise SystemExit(f"launch failed rc={rc} ctrl={ctrl.tolist()}")
            if dt > CAP_S:
                raise SystemExit(f"launch {dt:.3f} s > cap {CAP_S} s (start={start}, positions={len(p) + n})")
            return dt, tok[:n], step[:n], last, exp

        # prefill: chunks re-sized from the measured per-position time (n_out = 0 except the last chunk)
        start, i, tp, exps = 0, 0, 0.004, []
        tokens, steps = [], []
        while i < len(prompt):
            c = int(max(1, min(64, TARGET_S / tp - 1)))
            chunk = prompt[i:i + c]
            lastc = i + len(chunk) >= len(prompt)
            dt, tok, st, last, ex = run(chunk, 1 if lastc else 0, start)
            tp = max(tp, dt / (len(chunk) + (1 if lastc else 0)))
            exps.append(ex[:len(chunk)])                 # positions start .. start+len-1 (the generated one is re-run)
            if lastc:
                tokens.append(tok)
                steps.append(st)
            start += len(chunk)
            i += len(chunk)
        # decode: continued launches {last} + n (protocol: start = prev_start + 1 + n - 1)
        got = 1
        while got < n_out:
            n = int(max(1, min(n_out - got, TARGET_S / tp - 1, 16)))
            dt, tok, st, last, ex = run([int(tokens[-1][-1])], n, start)
            tp = max(tp, dt / (1 + n))
            exps.append(ex[:n])                          # positions start .. start+n-1
            tokens.append(tok)
            steps.append(st)
            start += n
            got += n
        # final position (the last generated token) is processed by the final launch: its experts row
        exps.append(ex[n:n + 1])
        lib.mid_session_free(sess)
        tok = np.concatenate(tokens)[:n_out]
        step = np.concatenate(steps)[:n_out]
        allexp = np.concatenate(exps)[:len(prompt) + n_out]
        p = len(prompt)
        np.savez(out / f"wrap512_{b}.npz", schema=np.array("rec-mk-out/1"), case=np.array("wrap512"),
                 ckpt_sha256=np.array(sha),
                 impl=np.array(f"{impl} workers={a.workers} [M3 continuation emit, {launches} launches]"),
                 kv_dtype=np.array(kv_dtype), prompt=np.array(prompt, np.int32), tokens=tok, step_logits=step,
                 last_logits=last, chosen_experts=allexp[p - 1:p - 1 + n_out], step_experts=allexp)
        summary[b] = {"launches": launches, "longest_s": round(longest, 4), "tokens6": tok[:6].tolist(),
                      "positions": int(allexp.shape[0])}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
