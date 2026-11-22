"""rec-3cr / M3 depth-row gate (full config, dusk): emit the rec-pp2 gate case at depth via KV continuation, per
KV build, R times in FRESH sessions (independent bank streams) so the repeat-determinism check compares independent
emits. Every launch is held to the 0.25 s display-iGPU cap. Output npz = rec-mk-out/1 (scored on dawn by
scripts/bisect/score_soak.py / mid_mk_check.check_arrays against the torch fp32 CPU reference).

    python scripts/m3_depth_gate.py --lib flint_full.so --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 \
        --out DIR [--case wrap512|depth8192] [--builds fp32kv,fp32_bf16kv,bf16kv] [--repeats 2] [--budget-s 300]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

MK = Path(__file__).resolve().parent.parent / "engine" / "native" / "megakernel_hip"
sys.path.insert(0, str(MK))
import flint_full_driver as fd  # noqa: E402

TARGET_S = 0.20


def payload_md5(tok, step, last, exp) -> str:
    h = hashlib.md5()
    for a in (tok, step, last, exp):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def emit(lib, cfg, bank, off, prompt, n_out, mode, workers, max_total, deadline, pairs=0):
    """(An in-session decode replay is NOT a valid determinism check: the sliding-window layers' 513-slot KV ring is
    overwritten by the first decode, so a replay attends to different KV. Use fresh-session repeats only.)
    pairs: after the emit, run ABBA speed pairs from the end state (A = {last}+1, B = {last}+k), window_bench protocol."""
    sess = fd.create_session(lib, cfg, bank, off, max_total=max_total)
    longest, launches, extra = 0.0, 0, {}
    try:
        def run(pr, n, start):
            nonlocal longest, launches
            dt, tok, last, step, exp, ctrl, bad = fd.run(lib, sess, cfg, np.asarray(pr, np.int32), n, workers, mode,
                                                         step_logits=n > 0, experts=True, start=start)
            longest, launches = max(longest, dt), launches + 1
            if bad:
                raise SystemExit(f"INVALID launch start={start}: {bad}")
            if dt > fd.CAP_S:
                raise SystemExit(f"launch {dt:.3f} s > cap {fd.CAP_S} s (start={start}, positions={len(pr) + n})")
            if time.time() > deadline:
                raise SystemExit(f"budget deadline reached at start={start}")
            return dt, tok, last, step, exp

        start, i, tp, exps, tokens, steps = 0, 0, 0.03, [], [], []
        while i < len(prompt):
            c = int(max(1, min(64, TARGET_S / tp - 1)))
            chunk = prompt[i:i + c]
            lastc = i + len(chunk) >= len(prompt)
            dt, tok, last, st, ex = run(chunk, 1 if lastc else 0, start)
            tp = max(tp, dt / (len(chunk) + (1 if lastc else 0)))
            exps.append(ex[:len(chunk)])
            if lastc:
                tokens.append(tok)
                steps.append(st)
            start += len(chunk)
            i += len(chunk)
        got, n, sched, dec0 = 1, 1, [], start
        while got < n_out:
            n = int(max(1, min(n_out - got, TARGET_S / tp - 1, 16)))
            sched.append((int(tokens[-1][-1]), n, start))
            dt, tok, last, st, ex = run([int(tokens[-1][-1])], n, start)
            tp = max(tp, dt / (1 + n))
            exps.append(ex[:n])
            tokens.append(tok)
            steps.append(st)
            start += n
            got += n
        exps.append(ex[n:n + 1])
        if pairs:
            # cap-safe ABBA: probe tA (single-position launch at depth); B(k) <= k * tA (shared fixed cost), so
            # k = floor(0.9 * cap / tA) can never exceed the cap. k < 2 -> speed not measurable under the cap.
            # (A single-launch-minus-overhead estimate is NOT used: at depth a launch carries depth-dependent fixed
            # cost the shallow overhead misses; on mid it understated decode ~2x vs the ABBA differential.)
            lt, rows, d0 = int(tokens[-1][-1]), [], start
            try:
                probe = []
                for _ in range(2):
                    dt, tk, _, _, _ = run([lt], 1, start)
                    probe.append(dt)
                    start += 1
                    lt = int(tk[-1])
                tA0 = max(probe)
                k = int(min(31, 0.9 * fd.CAP_S / tA0))
                extra["speed_probe"] = {"tA_probe_s": [round(x, 5) for x in probe], "k": k}
                if k < 2:
                    extra["speed"] = {"decode_tok_s": None, "k": k, "tA_probe_s": round(tA0, 5),
                                      "note": "ABBA not measurable under the 0.25 s cap (2*tA > 0.9*cap); 1/tA is a lower "
                                              "bound on decode tok/s", "lower_bound_tok_s": round(1.0 / tA0, 2)}
                    fd.log(f"ROW depth{len(prompt)}: speed NOT measurable under cap (tA {tA0 * 1e3:.1f} ms -> k={k}); "
                           f"lower bound 1/tA = {1.0 / tA0:.2f} tok/s")
                else:
                    for i in range(pairs):
                        if not fd.wait_idle(60):
                            raise SystemExit("gate timeout: busy>=5")
                        pre, r = fd.cotenants(), {}
                        for tag in (("A", "B") if i % 2 == 0 else ("B", "A")):
                            nn = 1 if tag == "A" else k
                            dt, tk, _, _, _ = run([lt], nn, start)
                            r[tag] = dt
                            start += nn
                            lt = int(tk[-1])
                        post = fd.cotenants()
                        rows.append({"pair": i, "tA_s": round(r["A"], 5), "tB_s": round(r["B"], 5), "depth": start,
                                     "stable": pre.split(",gpu_busy")[0] == post.split(",gpu_busy")[0], "pre": pre,
                                     "post": post})
                    v = [x for x in rows if x["stable"] and x["tB_s"] > x["tA_s"]]
                    tA = sum(x["tA_s"] for x in v) / max(1, len(v))
                    tB = sum(x["tB_s"] for x in v) / max(1, len(v))
                    extra["speed"] = {"method": "ABBA differential (k-1)/(tB-tA), cap-safe k", "k": k,
                                      "valid_pairs": len(v), "pairs_n": pairs, "depth_range": [d0, start],
                                      "decode_tok_s": round((k - 1) / (tB - tA), 2) if v and tB > tA else None,
                                      "pairs": rows}
                    fd.log(f"ROW depth{len(prompt)}: decode {extra['speed']['decode_tok_s']} tok/s ({len(v)}/{pairs} "
                           f"stable pairs, depth {d0}..{start}, k={k}, tA probe {tA0 * 1e3:.1f} ms) | {fd.cotenants()}")
            except SystemExit as e:   # never lose the gate emit to a speed-phase failure
                extra["speed"] = {"decode_tok_s": None, "error": str(e)}
                fd.log(f"ROW depth{len(prompt)}: speed phase aborted: {e}")
    finally:
        lib.mid_session_free(sess)
    tok = np.concatenate(tokens)[:n_out]
    step = np.concatenate(steps)[:n_out]
    allexp = np.concatenate(exps)[:len(prompt) + n_out]
    extra["t_pos_s"] = tp
    return tok, step, last, allexp, launches, longest, extra


def emit_ppl(lib, cfg, bank, off, items, mode, workers, deadline):
    """Teacher-forced ppl via continuation: launch [t_i] at start=i, n_out=1 -> the logit row predicting t_{i+1}."""
    import math
    n = max(len(it["tokens"]) for it in items)
    sess = fd.create_session(lib, cfg, bank, off, max_total=n + 64)
    logp, top1, rowmax, longest = [], [], [], 0.0
    try:
        for it in items:
            t = it["tokens"]
            lp, t1, mx = [], [], []
            for i in range(len(t) - 1):
                dt, tok, last, step, exp, ctrl, bad = fd.run(lib, sess, cfg, np.asarray([t[i]], np.int32), 1, workers,
                                                             mode, step_logits=True, start=i)
                longest = max(longest, dt)
                if bad or dt > fd.CAP_S or time.time() > deadline:
                    raise SystemExit(f"ppl launch failed/over cap/deadline at {it['id']} pos {i}: {bad} {dt:.3f}")
                row = step[0].astype(np.float64)
                m = row.max()
                lp.append(float(row[t[i + 1]] - (m + math.log(np.exp(row - m).sum()))))
                t1.append(int(row.argmax()))
                mx.append(float(np.abs(row).max()))
            logp.append(lp), top1.append(t1), rowmax.append(mx)
    finally:
        lib.mid_session_free(sess)
    return np.asarray(logp), np.asarray(top1, np.int32), np.asarray(rowmax), longest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", default="wrap512")
    ap.add_argument("--builds", default="fp32kv,fp32_bf16kv,bf16kv")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--rep-offset", type=int, default=0, help="first repeat index (file naming across invocations)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--off", type=Path, default=None)
    ap.add_argument("--ppl-set", type=Path, default=None, help="eval_set_full_v1.json: emit teacher-forced ppl instead")
    ap.add_argument("--ppl-builds", default="fp32kv,bf16kv")
    ap.add_argument("--pairs", type=int, default=0, help="ABBA speed pairs after the emit (production mode row)")
    ap.add_argument("--rehearsal", action="store_true", help="mid-8L dry run: skip the FLINT_CFG_FULL check")
    a = ap.parse_args()
    lib, cfg = fd.load_lib(a.lib)
    fd._CARD = fd.hip_card(0)   # enables the driver's busy gate + VRAM preflight (only set by the driver's own main())
    if fd._CARD is None:
        raise SystemExit("cannot resolve the HIP device's sysfs card: refusing (busy gate + VRAM preflight need it)")
    if not cfg["full"] and not a.rehearsal:
        raise SystemExit(f"{a.lib} is not a FLINT_CFG_FULL build: {cfg}")
    bank, off = fd.open_bank(a.bank, cfg, a.off)
    if a.ppl_set:
        ev = json.loads(a.ppl_set.read_text())
        a.out.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + a.budget_s
        for b in a.ppl_builds.split(","):
            mode, kv_dtype, impl = fd.BUILDS[b]
            t0 = time.time()
            logp, top1, rowmax, longest = emit_ppl(lib, cfg, bank, off, ev["ppl_items"], mode, a.workers, deadline)
            np.savez(a.out / f"ppl__{b}.npz", impl=np.array(f"{impl} build={b}"), logp=logp, top1=top1, rowmax=rowmax,
                     evalset_sha256=np.array(ev["sha256"]), kv_dtype=np.array(kv_dtype))
            fd.log(f"PPL {b}: {logp.shape} nll {-logp.mean():.6f} ppl {np.exp(-logp.mean()):.4f} longest {longest:.3f} s, "
                   f"{time.time() - t0:.1f} s | {fd.cotenants()}")
        return 0
    if a.case.startswith("depth"):   # depthN: N-token prefix of the seeded 8192-token prompt (nested across N)
        depth = int(a.case[5:])
        prompt = np.random.default_rng(8192).integers(3, fd.VOCAB, size=max(8192, depth), dtype=np.int32)[:depth].tolist()
        n_out = 24
    else:
        c = fd.case_matrix(fd.VOCAB)[a.case]
        prompt, n_out = [int(t) for t in c["prompt"]], int(c["n_out"])
    sha = fd.shard_sha256(a.ckpt)
    a.out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.budget_s
    summary = {"case": a.case, "positions": len(prompt) + n_out, "lib": str(a.lib), "builds": {}}
    for b in a.builds.split(","):
        mode, kv_dtype, impl = fd.BUILDS[b]
        md5s, extras = [], []
        for r in range(a.rep_offset, a.rep_offset + a.repeats):
            t0 = time.time()
            tok, step, last, exp, launches, longest, ex = emit(lib, cfg, bank, off, prompt, n_out, mode, a.workers,
                                                               len(prompt) + n_out + 64 + 34 * a.pairs, deadline,
                                                               pairs=a.pairs if r == a.rep_offset else 0)
            tp = ex["t_pos_s"]
            extras.append(ex)
            p = len(prompt)
            name = f"{a.case}_{b}" + (f"_r{r}" if r else "") + ".npz"
            np.savez(a.out / name, schema=np.array("rec-mk-out/1"), case=np.array(a.case), ckpt_sha256=np.array(sha),
                     impl=np.array(f"{impl} workers={a.workers} [depth gate, fresh session r{r}, {launches} launches]"),
                     kv_dtype=np.array(kv_dtype), prompt=np.array(prompt, np.int32), tokens=tok, step_logits=step,
                     last_logits=last, chosen_experts=exp[p - 1:p - 1 + n_out], step_experts=exp)
            md5s.append(payload_md5(tok, step, last, exp))
            fd.log(f"GATE {name}: tokens[:6]={tok[:6].tolist()} md5 {md5s[-1][:12]} launches {launches} longest "
                   f"{longest:.3f} s, {tp * 1e3:.1f} ms/pos, {time.time() - t0:.1f} s | {fd.cotenants()}")
        det = len(set(md5s)) == 1 if len(md5s) > 1 else None
        summary["builds"][b] = {"md5": md5s, "fresh_repeats": len(md5s),
                                "deterministic": det, "speed": next((e["speed"] for e in extras if "speed" in e), None),
                                "t_pos_s": [e["t_pos_s"] for e in extras]}
        fd.log(f"DETERMINISM {b}: {'PASS' if det else ('FAIL' if det is False else 'not tested (1 emit)')} "
               f"({len(md5s)} fresh-session emits)")
    (a.out / "gate_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0 if all(v["deterministic"] is not False for v in summary["builds"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
