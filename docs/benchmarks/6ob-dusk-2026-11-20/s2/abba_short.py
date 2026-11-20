"""rec-6ob short-context decode timing on display GPUs: old vs new mid .so, ABBA, launches <= ~0.25 s.

decode32_short_ctx (run-definitions R1-R8 differential, short-ctx variant): per launch pair,
t(prompt4, n_out=36) - t(prompt4, n_out=4) -> 32 / dt tok/s. Builds alternate A B B A per round;
each pair is validated (rc, barrier abort, tokens nonzero, 4-token prefix of the 36-run == 4-run) and
busy-gated; cotenant state is recorded per round.

    python abba_short.py --a old/flint_mid.so --b new/flint_mid.so --ckpt <mid-prod> [--rounds 8] [--mode 0]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mid_mk_driver import pack_mid_bank  # noqa: E402
from mk_prof import load, run  # noqa: E402
from q4_lab import busy, cotenants, gate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("abba_short.json"))
    a = ap.parse_args()
    bank, off, _ = pack_mid_bank(a.ckpt)
    prompt = np.array([1, 5, 6, 7], np.int32)
    sess = {}
    for tag, so in (("A", a.a), ("B", a.b)):
        lib = load(so)
        s = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, 64)
        for _ in range(2):
            run(lib, s, prompt, 8, a.workers, a.mode)  # warm
        sess[tag] = (lib, s)
    rows = {"A": [], "B": []}
    for rnd in range(a.rounds):
        if not gate():
            print("GATE TIMEOUT", flush=True)
            return 2
        pre = f"busy={busy()} {cotenants()}"
        order = "AB" if rnd % 2 == 0 else "BA"
        for tag in order:
            lib, s = sess[tag]
            t4, tok4 = run(lib, s, prompt, 4, a.workers, a.mode)
            t36, tok36 = run(lib, s, prompt, 36, a.workers, a.mode)
            ok = tok36[:4].tolist() == tok4.tolist() and t36 > t4
            rows[tag].append({"round": rnd, "t4": round(t4, 5), "t36": round(t36, 5),
                              "tok_s": round(32 / (t36 - t4), 2) if ok else None, "valid": ok,
                              "tokens36": tok36.tolist()})
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} round {rnd} {order} | A {rows['A'][-1]['tok_s']} "
              f"B {rows['B'][-1]['tok_s']} | pre {pre} | post busy={busy()} {cotenants()}", flush=True)
    for lib, s in sess.values():
        lib.mid_session_free(s)
    res = {"mode": a.mode, "workers": a.workers, "a": str(a.a), "b": str(a.b), "rows": rows}
    for tag in "AB":
        v = [r for r in rows[tag] if r["valid"]]
        t4m, t36m = statistics.mean(r["t4"] for r in v), statistics.mean(r["t36"] for r in v)
        res[tag] = {"valid": len(v), "differential_tok_s": round(32 / (t36m - t4m), 2),
                    "pair_min": min(r["tok_s"] for r in v), "pair_max": max(r["tok_s"] for r in v),
                    "t36_mean_s": round(t36m, 5)}
    res["tokens_identical_A_B"] = all(x["tokens36"] == y["tokens36"] for x, y in zip(rows["A"], rows["B"]))
    res["speedup_B_over_A"] = round(res["B"]["differential_tok_s"] / res["A"]["differential_tok_s"], 3)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("A", "B", "speedup_B_over_A", "tokens_identical_A_B")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
