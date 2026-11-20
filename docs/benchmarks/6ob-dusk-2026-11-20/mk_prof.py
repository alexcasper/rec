"""rec-6ob: per-barrier phase profile of the mid-8L megakernel (MK_PROFILE build), short context only.

Runs ONE launch of prompt4 + n_out tokens (default 36: ~0.3 s on dusk, far below the display-iGPU
wedge threshold; never run long launches on dusk). Worker 0 / thread 0 records per SYNC_ALL the time
its workgroup finished the phase (ta) and the device-wide release (tr). Per call site over decode
positions: phase = tr[k] - tr[k-1], w0_work = ta[k] - tr[k-1], wait = tr[k] - ta[k] (time until the
slowest worker arrived + barrier latency).

    python mk_prof.py --so flint_mid_prof.so --src flint_mid_megakernel_prof.hip --ckpt <mid-prod>
        [--prod flint_mid.so] [--n-out 36] [--workers 16] [--mode 0] [--reps 3] [--out prof.json]
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mid_mk_driver import MID_VOCAB, pack_mid_bank  # noqa: E402

vp = ctypes.c_void_p


def load(so: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so.resolve()))
    lib.mid_session_create.restype = vp
    lib.mid_session_create.argtypes = [vp, vp, ctypes.c_int]
    lib.mid_session_run_ex2.restype = ctypes.c_int
    lib.mid_session_run_ex2.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, vp, vp, ctypes.c_int,
                                        ctypes.c_int, vp, vp]
    lib.mid_session_free.argtypes = [vp]
    return lib


def labels(src: Path) -> dict[int, str]:
    out, sec = {}, "?"
    for i, line in enumerate(src.read_text().splitlines(), 1):
        m = re.search(r"// ---- (\d+\.[^-]*)", line)
        if m:
            sec = m.group(1).strip()[:48]
        if re.match(r"\s+SYNC_ALL\(\);", line):
            out[i] = sec
    return out


def run(lib, sess, prompt, n_out, workers, mode):
    tok = np.zeros(n_out, np.int32)
    last = np.zeros(MID_VOCAB, np.float32)
    ctrl = np.zeros(4, np.uint32)
    t0 = time.perf_counter()
    rc = lib.mid_session_run_ex2(sess, prompt.ctypes.data, len(prompt), tok.ctypes.data, n_out, None,
                                 last.ctypes.data, workers, mode, ctrl.ctypes.data, None)
    dt = time.perf_counter() - t0
    if rc != 0 or ctrl[2] != 0 or ctrl[1] == 0 or not tok.any():
        raise SystemExit(f"invalid launch rc={rc} ctrl={ctrl.tolist()} tokens={tok[:4]}")
    return dt, tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--prod", type=Path)
    ap.add_argument("--n-out", type=int, default=36)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("prof.json"))
    a = ap.parse_args()
    if a.n_out > 40:
        raise SystemExit("refusing n_out > 40: keep launches short on display GPUs")
    bank, off, _ = pack_mid_bank(a.ckpt)
    prompt = np.array([1, 5, 6, 7], np.int32)
    lab = labels(a.src)
    res = {"n_out": a.n_out, "workers": a.workers, "mode": a.mode, "prompt": prompt.tolist()}

    if a.prod:  # instrumentation overhead: same launch on the production build
        p = load(a.prod)
        s = p.mid_session_create(bank.ctypes.data, off.ctypes.data, 64)
        run(p, s, prompt, a.n_out, a.workers, a.mode)
        res["prod_launch_s"] = [round(run(p, s, prompt, a.n_out, a.workers, a.mode)[0], 5) for _ in range(a.reps)]
        res["prod_tokens"] = run(p, s, prompt, a.n_out, a.workers, a.mode)[1].tolist()
        p.mid_session_free(s)

    lib = load(a.so)
    lib.mk_prof_reset.restype = ctypes.c_int
    lib.mk_prof_read.restype = ctypes.c_int
    lib.mk_prof_read.argtypes = [vp, vp, vp, ctypes.c_int, vp]
    sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, 64)
    run(lib, sess, prompt, a.n_out, a.workers, a.mode)  # warm
    agg = collections.defaultdict(lambda: {"phase": [], "work": [], "wait": []})
    per_tok, launches, total_pos = [], [], len(prompt) + a.n_out
    for _ in range(a.reps):
        lib.mk_prof_reset()
        dt, tok = run(lib, sess, prompt, a.n_out, a.workers, a.mode)
        launches.append(round(dt, 5))
        res.setdefault("prof_tokens", tok.tolist())
        M = 1 << 16
        line, ta, tr = np.zeros(M, np.int32), np.zeros(M, np.uint64), np.zeros(M, np.uint64)
        khz = ctypes.c_int(0)
        n = lib.mk_prof_read(line.ctypes.data, ta.ctypes.data, tr.ctypes.data, M, ctypes.byref(khz))
        if n > M or n % total_pos:
            raise SystemExit(f"{n} records is not a multiple of {total_pos} positions")
        per = n // total_pos
        us = 1e3 / khz.value  # ticks -> us
        line, ta, tr = line[:n].reshape(total_pos, per), ta[:n].astype(np.int64), tr[:n].astype(np.int64)
        ta, tr = ta.reshape(total_pos, per), tr.reshape(total_pos, per)
        for pos in range(len(prompt), total_pos):  # decode positions only
            prev = tr[pos - 1, -1]
            per_tok.append((tr[pos, -1] - prev) * us)
            for k in range(per):
                key = (int(line[pos, k]), k)
                start = prev if k == 0 else tr[pos, k - 1]
                agg[key]["phase"].append((tr[pos, k] - start) * us)
                agg[key]["work"].append((ta[pos, k] - start) * us)
                agg[key]["wait"].append((tr[pos, k] - ta[pos, k]) * us)
        res["wall_clock_khz"] = khz.value
        res["barriers_per_token"] = per
    lib.mid_session_free(sess)
    res["prof_launch_s"] = launches
    res["token_us_median"] = round(statistics.median(per_tok), 2)
    res["tokens_match_prod"] = res.get("prod_tokens") == res.get("prof_tokens") if a.prod else None
    rows, by_sec = [], collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for (ln, k), d in sorted(agg.items(), key=lambda kv: kv[0][1]):
        r = {"k": k, "line": ln, "section": lab.get(ln, "?"),
             **{f"{m}_us": round(statistics.median(v), 2) for m, v in d.items()}}
        rows.append(r)
        b = by_sec[r["section"]]
        for i, m in enumerate(("phase_us", "work_us", "wait_us")):
            b[i] += r[m]
        b[3] += 1
    res["per_barrier"] = rows
    res["by_section"] = {s: {"barriers": v[3], "phase_us": round(v[0], 1), "work_us": round(v[1], 1),
                             "wait_us": round(v[2], 1)} for s, v in by_sec.items()}
    tot = sum(r["phase_us"] for r in rows)
    res["sum_phase_us"], res["sum_wait_us"] = round(tot, 1), round(sum(r["wait_us"] for r in rows), 1)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"barriers/token {res['barriers_per_token']} | token {res['token_us_median']} us "
          f"({1e6 / res['token_us_median']:.1f} tok/s in-kernel) | sum phase {tot:.0f} us, sum wait "
          f"{res['sum_wait_us']} us ({100 * res['sum_wait_us'] / tot:.0f}%)")
    print(f"launch s prof {launches} prod {res.get('prod_launch_s')} | tokens match prod: {res['tokens_match_prod']}")
    for s, v in sorted(res["by_section"].items(), key=lambda kv: -kv[1]["phase_us"]):
        print(f"  {s:48s} x{v['barriers']:3d} phase {v['phase_us']:8.1f} us  work {v['work_us']:8.1f}  wait {v['wait_us']:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
