#!/usr/bin/env python3
"""rec-ohs: GEPA-style hill-climb driver for the mid-8L megakernel decode path (dusk, 24f167d lineage).

Loop (one candidate at a time, everything appended to the pool JSON):
  propose  single-knob mutation of the current best along a pool mutation axis (KNOBS below; order = the
           pool's axis order), or an external candidate (--candidate JSON: knobs and/or a patch file, for
           reflective / LLM-proposed mutations)
  build    on dusk (scripts/gepa_remote.py build): hipcc -D knobs, resources (VGPR/scratch/LDS) and
           hot-code instruction count recorded per candidate (rec-9t8 I-cache observation)
  bench    on dusk: R1-R8 differential decode32 short ctx, ABBA x rounds vs the CURRENT BEST .so, short4 + chat,
           busy<5 + wait-for-idle, cotenant state per round, any launch > 0.25 s aborts the candidate
  gates    HARD, auto-reject outside them (pool gates.hard_constraints):
             rec-pp2 two-tier: fp32kv @ 1e-4 and fp32_bf16kv @ 1e-3 on short4 + chat vs torch fp32 CPU
               (mid_mk_check.check, run HERE on the CPU host -- dusk has ~8 GB free next to flint/E4B);
             ppl: |ppl(cand, bf16kv) / ppl(incumbent, bf16kv) - 1| < 1 % on the held-out set = the frozen
               rec-1f4 ppl items (eval_set_v1.json, sha 72209796...f464).
           If every suite/ppl npz is byte-identical to the incumbent's, the incumbent's verdict is inherited.
  update   pool: evaluations[] (+ rejected[] with reason), Pareto front (short4, chat; after-512 = albion-gated),
           current best for the hill-climb. Promotion to run-of-record is NEVER automatic: `--promote ID`
           only after gates passed; the kernel default change is then committed by hand.
Stop: --max-evals, or --patience consecutive candidates without improvement (farm plateau lesson).

    python scripts/gepa_kernel_search.py --iter0              # harness calibration: incumbent vs itself
    python scripts/gepa_kernel_search.py --max-evals 12       # hill-climb
    python scripts/gepa_kernel_search.py --candidate cand.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
KERNEL = "engine/native/megakernel_hip/flint_mid_megakernel.hip"
POOL_DEFAULT = Path("/mnt/d/ops/rec-aip-prep/docs/benchmarks/rec-perf-candidates.json")
DUSK = "dusk@dusk"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-i", str(Path.home() / ".ssh/dusk")]
REMOTE = "/home/dusk/rec-ve/gepa"
CK_REMOTE = "/home/dusk/rec-ve/ckpt-pp2/mid-prod"
CK_LOCAL = Path("/home/dawn/rec-data/mid-prod")
STATE = Path("/home/dawn/rec-data/gepa")
ENV = "source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION"
PPL_GATE = 0.01
CASES = ("short4", "chat")

# knob -> (pool axis, values, default). Defaults reproduce the incumbent (24f167d) exactly.
KNOBS = {
    "GEPA_GEN_UNROLL": ("unroll_factors", [1, 2, 4], 1),
    "GEPA_K512_PIPE": ("lds_staging", [0, 1], 0),
    "GEPA_GEN_PREFETCH_X": ("lds_staging", [1, 0], 1),
    "workers": ("tile_shapes", [16, 8], 16),
    "GEPA_TOPK_PER_WG": ("barrier_structure", [0, 1], 0),
    "GEPA_RESID_FUSE": ("barrier_structure", [0, 1], 0),
}
DEFAULTS = {k: v[2] for k, v in KNOBS.items()}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(STATE / "search.log", "a") as f:
        f.write(line + "\n")


def remote(args: str, timeout: int = 1800) -> dict:
    cmd = f"{ENV}; cd {REMOTE}/h && $PY scripts/gepa_remote.py {args}"
    p = subprocess.run(SSH + [DUSK, cmd], capture_output=True, text=True, timeout=timeout)
    with open(STATE / "remote.log", "a") as f:
        f.write(f"\n### {now()} {args}\n{p.stderr}")
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"status": f"remote_error rc={p.returncode}", "stderr_tail": p.stderr[-800:]}
    return json.loads(lines[-1])


def sync_harness(pool: Path) -> None:
    files = ["scripts/gepa_remote.py", "engine/native/megakernel_hip/mid_mk_driver.py",
             "engine/validation/megakernel_mid/qgate/qgate.py",
             "engine/validation/megakernel_mid/qgate/eval_set_v1.json",
             "engine/validation/megakernel_mid/qgate/gate_spec_v1.json"]
    subprocess.run(["rsync", "-a", "-R", "-e", " ".join(SSH), *files, f"{DUSK}:{REMOTE}/h/"], cwd=REPO, check=True)
    sync_pool(pool)


def sync_pool(pool: Path) -> None:
    subprocess.run(SSH + [DUSK, f"mkdir -p {REMOTE}/pool"], check=True)
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), str(pool), f"{DUSK}:{REMOTE}/pool/"], check=True)


def fetch(rel: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), f"{DUSK}:{REMOTE}/{rel}/", str(dest) + "/"], check=True)


def load_pool(p: Path) -> dict:
    return json.loads(p.read_text())


def save_pool(p: Path, pool: dict) -> None:
    pool["meta"]["updated"] = now()
    p.write_text(json.dumps(pool, indent=2) + "\n")
    sync_pool(p)


def knob_key(k: dict) -> str:
    return ",".join(f"{n}={k[n]}" for n in sorted(k))


def source_of(rev: str | None) -> tuple[str, str]:
    """Kernel source text + provenance. rev=None: working tree (must be committed, i.e. clean)."""
    if rev is None:
        dirty = subprocess.run(["git", "status", "--porcelain", KERNEL], cwd=REPO, capture_output=True,
                               text=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
        return (REPO / KERNEL).read_text(), f"{head}{'+dirty' if dirty else ''}"
    txt = subprocess.run(["git", "show", f"{rev}:{KERNEL}"], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout
    return txt, rev


def build(cid: str, src_text: str, knobs: dict, patch: str | None = None) -> dict:
    d = STATE / "src" / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "flint_mid_megakernel.hip").write_text(src_text)
    if patch:
        r = subprocess.run(["patch", "-s", "-p1", "-d", str(d), "-i", str(Path(patch).resolve())],
                           capture_output=True, text=True)
        if r.returncode:
            return {"status": "patch_failed", "stderr": r.stderr[-400:]}
    subprocess.run(SSH + [DUSK, f"mkdir -p {REMOTE}/c/{cid}"], check=True)
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH), str(d / "flint_mid_megakernel.hip"),
                    f"{DUSK}:{REMOTE}/c/{cid}/"], check=True)
    defs = " ".join(f"--define {n}={v}" for n, v in sorted(knobs.items()) if n.startswith("GEPA_"))
    res = remote(f"build --src {REMOTE}/c/{cid}/flint_mid_megakernel.hip --out {REMOTE}/c/{cid} {defs}")
    res["source_sha256"] = hashlib.sha256((d / "flint_mid_megakernel.hip").read_bytes()).hexdigest()
    return res


def bench(a_id: str, a_w: int, b_id: str, b_w: int, rounds: int) -> dict:
    out = {}
    for c in CASES:
        out[c] = remote(f"bench --a {REMOTE}/c/{a_id}/flint_mid.so --b {REMOTE}/c/{b_id}/flint_mid.so "
                        f"--wa {a_w} --wb {b_w} --case {c} --rounds {rounds} --ckpt {CK_REMOTE}", timeout=1200)
    return out


def emit(cid: str, workers: int) -> dict:
    return remote(f"emit --so {REMOTE}/c/{cid}/flint_mid.so --out {REMOTE}/c/{cid}/emit --workers {workers} "
                  f"--ckpt {CK_REMOTE}", timeout=1200)


def ppl_of(npz: Path) -> float:
    return math.exp(-float(np.load(npz)["logp"].mean()))


def run_gates(cid: str, em: dict, inc: dict | None) -> dict:
    """Hard gates. inc = incumbent record (emit md5 + gate verdict) for inheritance."""
    if em.get("status") != "ok":
        return {"pass": False, "reason": f"emit: {em.get('status')}"}
    local = STATE / "emit" / cid
    fetch(f"c/{cid}/emit/ppl", local / "ppl")
    ppl = {b: ppl_of(local / "ppl" / f"ppl__{b}.npz") for b in ("fp32kv", "bf16kv")}
    g = {"ppl": ppl, "ppl_longest_launch_s": em.get("ppl_longest_launch_s")}
    if inc is not None:
        rel = ppl["bf16kv"] / inc["gates"]["ppl"]["bf16kv"] - 1.0
        g["ppl_rel_vs_incumbent_bf16kv"] = rel
        g["ppl_gate"] = abs(rel) < PPL_GATE
    else:
        g["ppl_gate"] = True
        g["ppl_rel_vs_incumbent_bf16kv"] = 0.0
    suite_md5 = {k: v for k, v in em["npz_md5"].items() if k.startswith("suite/")}
    if inc is not None and suite_md5 == {k: v for k, v in inc["emit"]["npz_md5"].items() if k.startswith("suite/")}:
        g["rec_pp2"] = dict(inc["gates"]["rec_pp2"], inherited_from=inc["id"],
                            note="all 6 suite npz byte-identical to the incumbent's")
    else:
        fetch(f"c/{cid}/emit/suite", local / "suite")
        sys.path.insert(0, str(REPO / "engine" / "src"))
        sys.path.insert(0, str(REPO / "engine" / "validation" / "megakernel_mid"))
        import mid_mk_check as mk
        rp = {}
        for c in CASES:
            for b in ("fp32kv", "fp32_bf16kv", "bf16kv"):
                r = mk.check(CK_LOCAL, local / "suite" / f"{c}_{b}.npz", "cpu", verbose=False)
                rp[f"{c}/{b}"] = {"pass": bool(r["pass"]), "step_rel": r.get("step_logits_max_rel"),
                                  "last_rel": r.get("last_logits_rel"), "tokens_exact": r.get("tokens_exact"),
                                  "problems": r.get("problems", [])[:3]}
        tier_ok = all(rp[f"{c}/{b}"]["pass"] for c in CASES for b in ("fp32kv", "fp32_bf16kv"))
        g["rec_pp2"] = {"pass": tier_ok, "items": rp,
                        "rule": "fp32kv @1e-4 + fp32_bf16kv @1e-3 must PASS on short4+chat; bf16kv production recorded"}
    g["pass"] = bool(g["rec_pp2"]["pass"] and g["ppl_gate"])
    if not g["pass"]:
        why = []
        if not g["rec_pp2"]["pass"]:
            why.append("rec-pp2 tier: " + "; ".join(f"{k} {v['problems'][:1]}" for k, v in g["rec_pp2"].get("items", {}).items()
                                                    if not v["pass"] and "bf16kv" != k.split("/")[1]))
        if not g["ppl_gate"]:
            why.append(f"ppl delta {g['ppl_rel_vs_incumbent_bf16kv']:+.3%} >= 1%")
        g["reason"] = " | ".join(why)
    return g


def search_state(pool: dict) -> dict:
    s = pool.setdefault("search", {})
    s.setdefault("driver", "scripts/gepa_kernel_search.py (rec/gemini-kernels)")
    s.setdefault("host", "dusk gfx1150 mid-8L, launches <= 0.25 s, busy<5")
    pool.setdefault("evaluations", [])
    pool.setdefault("rejected", [])
    return s


def evaluate(pool: dict, pool_path: Path, cid: str, knobs: dict, axis: str, parent: str | None,
             src: tuple[str, str], patch: str | None, rounds: int, best: dict | None) -> dict:
    rec = {"id": cid, "time": now(), "axis": axis, "parent": parent, "knobs": knobs, "patch": patch,
           "source": src[1], "after512": None, "after512_note": "albion-gated (dusk launches <= 0.25 s)"}
    log(f"{cid}: build {knob_key(knobs)}" + (f" + patch {patch}" if patch else ""))
    rec["build"] = build(cid, src[0], knobs, patch)
    if rec["build"].get("status") != "ok":
        rec["outcome"] = "rejected"
        rec["reason"] = f"build: {rec['build'].get('status')} {rec['build'].get('errors', '')}"
        return finish(pool, pool_path, rec)
    k = rec["build"].get("kernels", {}).get("mid_generate", {})
    log(f"{cid}: built in {rec['build']['build_s']} s; mid_generate vgpr {k.get('vgpr')} scratch {k.get('scratch')} "
        f"lds {k.get('lds')}; hot instr {rec['build'].get('hot_instr')}")
    a_id, a_w = (best["id"], best["knobs"]["workers"]) if best else (cid, knobs["workers"])
    if best is None:  # iter0: incumbent vs a copy of itself
        subprocess.run(SSH + [DUSK, f"rm -rf {REMOTE}/c/{cid}_copy && cp -r {REMOTE}/c/{cid} {REMOTE}/c/{cid}_copy"],
                       check=True)
        rec["bench"] = bench(cid, knobs["workers"], f"{cid}_copy", knobs["workers"], rounds)
    else:
        rec["bench"] = bench(a_id, a_w, cid, knobs["workers"], rounds)
    bad = [c for c in CASES if rec["bench"][c].get("status") != "ok"]
    if bad:
        rec["outcome"] = "rejected"
        rec["reason"] = "bench: " + "; ".join(f"{c} {rec['bench'][c].get('status')}" for c in bad)
        return finish(pool, pool_path, rec)
    rec["tok_s"] = {c: rec["bench"][c]["B"]["differential_tok_s"] for c in CASES}
    rec["speedup_vs_best"] = {c: rec["bench"][c]["speedup_B_over_A"] for c in CASES}
    log(f"{cid}: tok/s {rec['tok_s']} speedup vs {a_id} {rec['speedup_vs_best']} (max launch "
        f"{max(rec['bench'][c]['max_launch_s'] for c in CASES)} s)")
    rec["emit"] = emit(cid, knobs["workers"])
    rec["gates"] = run_gates(cid, rec["emit"], best)
    log(f"{cid}: gates {'PASS' if rec['gates']['pass'] else 'FAIL'}" +
        (f" ({rec['gates'].get('reason')})" if not rec["gates"]["pass"] else "") +
        f"; ppl bf16kv {rec['gates']['ppl']['bf16kv']:.6g} ({rec['gates']['ppl_rel_vs_incumbent_bf16kv']:+.2e} vs best)")
    rec["outcome"] = "gates_pass" if rec["gates"]["pass"] else "rejected"
    if not rec["gates"]["pass"]:
        rec["reason"] = "gates: " + rec["gates"].get("reason", "?")
    return finish(pool, pool_path, rec)


def finish(pool: dict, pool_path: Path, rec: dict) -> dict:
    slim = json.loads(json.dumps(rec))
    for c in CASES:  # keep per-round rows out of the pool (they stay in remote.log); keep summaries
        if isinstance(slim.get("bench", {}).get(c), dict):
            slim["bench"][c].pop("rows", None)
    pool["evaluations"].append(slim)
    if rec["outcome"] == "rejected":
        pool["rejected"].append({"id": rec["id"], "knobs": rec["knobs"], "time": rec["time"], "reason": rec["reason"]})
        log(f"{rec['id']}: REJECTED - {rec['reason']}")
    (STATE / "records").mkdir(parents=True, exist_ok=True)
    (STATE / "records" / f"{rec['id']}.json").write_text(json.dumps(rec, indent=1))
    save_pool(pool_path, pool)
    return rec


def update_front(pool: dict, rec: dict, min_gain: float) -> bool:
    """Hill-climb acceptance: gates pass AND both cases faster than the current best by > min_gain."""
    sp = rec.get("speedup_vs_best", {})
    better = rec["outcome"] == "gates_pass" and sp and all(sp[c] > 1.0 + min_gain for c in CASES)
    front = pool["pareto_front"]["current_front"]
    if better:
        front.append({"candidate": rec["id"], "knobs": rec["knobs"], "decode32_short4": rec["tok_s"]["short4"],
                      "decode32_chat": rec["tok_s"]["chat"], "after512": None, "hot_instr": rec["build"].get("hot_instr"),
                      "speedup_vs_parent": sp, "parent": rec["parent"], "gates": "pass", "promoted": False})
    return bool(better)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=POOL_DEFAULT)
    ap.add_argument("--iter0", action="store_true", help="calibrate: incumbent (git) vs itself + gates, no mutation")
    ap.add_argument("--max-evals", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=None, help="default: 2x iter0 |speedup - 1| noise, >= 0.005")
    ap.add_argument("--candidate", type=Path, help='JSON {"knobs": {...}, "patch": "file.patch", "axis": "..."}')
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    pool = load_pool(a.pool)
    s = search_state(pool)
    sync_harness(a.pool)
    inc_commit = pool["incumbent"]["commit"]

    if a.iter0:
        src = source_of(inc_commit)
        rec = evaluate(pool, a.pool, "iter0", dict(DEFAULTS), "none (calibration)", None, src, None, a.rounds, None)
        sp = rec.get("speedup_vs_best", {})
        s["iter0"] = {"id": "iter0", "commit": inc_commit, "tok_s": rec.get("tok_s"), "self_speedup": sp,
                      "noise": max(abs(v - 1) for v in sp.values()) if sp else None,
                      "gates": rec.get("gates", {}).get("pass"), "time": rec["time"]}
        s["best"] = {"id": "iter0", "knobs": dict(DEFAULTS), "commit": inc_commit, "tok_s": rec.get("tok_s")}
        save_pool(a.pool, pool)
        log(f"iter0 done: {s['iter0']}")
        return 0 if rec["outcome"] == "gates_pass" else 1

    if "best" not in s:
        raise SystemExit("run --iter0 first (harness calibration)")
    src = source_of(None)
    if src[1].endswith("+dirty"):
        raise SystemExit("kernel working tree is dirty: commit the knob source first (provenance)")
    min_gain = a.min_gain if a.min_gain is not None else max(0.005, 2 * (s["iter0"]["noise"] or 0))
    s["min_gain"] = min_gain
    best = next(r for r in pool["evaluations"] if r["id"] == s["best"]["id"])
    # the incumbent build directory for the A side (iter0 = git incumbent; later = winning candidate)
    done = {knob_key(r["knobs"]) + (r.get("patch") or "") for r in pool["evaluations"] if r["id"] != "iter0"}
    done.add(knob_key(DEFAULTS))
    n_eval, stale = 0, 0
    ext = json.loads(a.candidate.read_text()) if a.candidate else None
    while n_eval < a.max_evals and stale < a.patience:
        if ext is not None:
            if n_eval:  # an external candidate is evaluated exactly once
                break
            knobs = dict(best["knobs"], **ext.get("knobs", {}))
            axis, patch = ext.get("axis", "external"), ext.get("patch")
        else:
            cands = []
            axis_order = [x["axis"] for x in pool["mutation_axes"]]
            for name, (ax, vals, _) in sorted(KNOBS.items(), key=lambda kv: axis_order.index(kv[1][0])
                                              if kv[1][0] in axis_order else 99):
                for v in vals:
                    k = dict(best["knobs"], **{name: v})
                    if v != best["knobs"][name] and knob_key(k) not in done:
                        cands.append((ax, k))
            if not cands:
                log("neighbourhood exhausted")
                break
            axis, knobs = cands[0]
            patch = None
        cid = f"c{len([r for r in pool['evaluations'] if r['id'].startswith('c')]) + 1:03d}"
        rec = evaluate(pool, a.pool, cid, knobs, axis, best["id"], src, patch, a.rounds, best)
        done.add(knob_key(knobs) + (patch or ""))
        n_eval += 1
        if update_front(pool, rec, min_gain):
            log(f"{cid}: ACCEPTED as new best (gain > {min_gain:.3f} on both cases)")
            best = rec
            s["best"] = {"id": cid, "knobs": knobs, "tok_s": rec["tok_s"], "source": rec["source"]}
            stale = 0
        else:
            stale += 1
        save_pool(a.pool, pool)
    s.setdefault("runs", []).append({"time": now(), "evals": n_eval, "stopped": "patience" if stale >= a.patience
                                     else "max_evals/exhausted", "best": s["best"]["id"], "min_gain": min_gain})
    save_pool(a.pool, pool)
    log(f"search stopped after {n_eval} evals; best {s['best']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
