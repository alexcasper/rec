"""Bedivere M3.1: KV continuation byte-identity check (runs ON dusk, mid-8L, short launches only).

For each case (short4, chat) and mode (0 bf16kv, 1 fp32_bf16kv, 2 fp32kv): ONE launch of prompt + N tokens vs
`chunks` continued launches (mid_session_run_ex3 with start_pos; protocol: prompt = {last token of the previous
launch}, start_pos = prev_start + prev_prompt_len + prev_n_out - 1). Tokens, every step's logits and the final
last-logits must be bit-identical. Prints one JSON line.
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
from mid_mk_driver import MID_VOCAB, case_matrix, pack_mid_bank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-out", type=int, default=32)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    lib = ctypes.CDLL(str(Path(a.so).resolve()))
    lib.mid_session_create.restype = vp
    lib.mid_session_create.argtypes = [vp, vp, ctypes.c_int]
    lib.mid_session_run_ex3.restype = ctypes.c_int
    lib.mid_session_run_ex3.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, vp, vp, ctypes.c_int, ctypes.c_int,
                                        vp, vp, ctypes.c_int]
    lib.mid_session_free.argtypes = [vp]
    bank, off, _ = pack_mid_bank(Path(a.ckpt))
    cm = case_matrix(MID_VOCAB)
    per = a.n_out // a.chunks
    assert per * a.chunks == a.n_out
    res, ok_all, longest = {}, True, 0.0

    def run(sess, prompt, n_out, mode, start):
        nonlocal longest
        p = np.array(prompt, np.int32)
        tok = np.zeros(n_out, np.int32)
        step = np.zeros((n_out, MID_VOCAB), np.float32)
        last = np.zeros(MID_VOCAB, np.float32)
        ctrl = np.zeros(4, np.uint32)
        t0 = time.perf_counter()
        rc = lib.mid_session_run_ex3(sess, p.ctypes.data, len(p), tok.ctypes.data, n_out, step.ctypes.data,
                                     last.ctypes.data, a.workers, mode, ctrl.ctypes.data, None, start)
        longest = max(longest, time.perf_counter() - t0)
        if rc != 0 or ctrl[2] != 0:
            raise SystemExit(f"launch failed rc={rc} ctrl={ctrl.tolist()}")
        return tok, step, last

    for case in ("short4", "chat"):
        prompt = cm[case]["prompt"]
        for mode in (0, 1, 2):
            s1 = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, 64)
            t1, st1, l1 = run(s1, prompt, a.n_out, mode, 0)
            lib.mid_session_free(s1)
            s2 = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, 64)
            toks, steps, start, pr = [], [], 0, list(prompt)
            for c in range(a.chunks):
                t, st, l2 = run(s2, pr, per, mode, start)
                toks.append(t)
                steps.append(st)
                start = start + len(pr) + per - 1
                pr = [int(t[-1])]
            lib.mid_session_free(s2)
            t2, st2 = np.concatenate(toks), np.concatenate(steps)
            r = {"tokens": bool(np.array_equal(t1, t2)), "step_logits": bool(np.array_equal(st1, st2)),
                 "last_logits": bool(np.array_equal(l1, l2))}
            r["pass"] = all(r.values())
            ok_all &= r["pass"]
            res[f"{case}/mode{mode}"] = r
    print(json.dumps({"pass": ok_all, "items": res, "n_out": a.n_out, "chunks": a.chunks,
                      "longest_launch_s": round(longest, 4)}))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
