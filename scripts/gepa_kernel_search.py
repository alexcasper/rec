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
import re
import subprocess
import sys
import time
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
    # run 3: per-warp depth on the slow phases (O-proj = generic K=4096, MoE down = K=512 VCgen + K=1024 generic)
    "GEPA_K512_DUAL": ("tile_shapes", [0, 1], 0),
    "GEPA_GEN_DIST2": ("lds_staging", [0, 1], 0),
}
DEFAULTS = {k: v[2] for k, v in KNOBS.items()}
ALLOWED: set = set(KNOBS)   # --knobs restriction (proposals only; reflection prompt + parser honour it)
CONTEXT = ""                # extra reflection context for this run (--context)


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


# ------------------------------------------------------------------------------------------- optimizer
KIMI_WINDOW_S = 180.0         # max 1 kimi call per ~180 s (RUN 2: 6/8 first attempts were 429 at ~30 s spacing)
KIMI_THROTTLE_HOUR_S = 3600.0  # consecutive 429s spanning this long -> kimi disabled for an hour, then re-probe
RETRYABLE = lambda http: http == 429 or 500 <= http < 600  # noqa: E731


class Optimizer:
    """Reflection/mutation LLM: kimi-first (Cerebras), one backoff retry on 429/5xx, then glm-5.3-flash for that
    iteration. State persists in STATE/optimizer_state.json across runs (rate window, 429 streak, disable-until).
    Keys never touch this host: calls execute on dusk (gepa_remote.py reflect) reading mode-600 key files."""

    def __init__(self):
        self.path = STATE / "optimizer_state.json"
        self.st = {"kimi_last_call": 0.0, "kimi_429_total": 0, "kimi_429_consecutive": 0,
                   "kimi_429_streak_start": None, "kimi_disabled_until": 0.0, "primary": "kimi",
                   "calls": {"kimi": 0, "glm": 0}}
        if self.path.exists():
            self.st.update(json.loads(self.path.read_text()))

    def save(self):
        self.path.write_text(json.dumps(self.st, indent=1))

    def _call(self, model: str, prompt_remote: str) -> dict:
        if model == "kimi":
            wait = self.st["kimi_last_call"] + KIMI_WINDOW_S - time.time()
            if wait > 0:
                time.sleep(wait)
            self.st["kimi_last_call"] = time.time()
        self.st["calls"][model] += 1
        r = remote(f"reflect --model {model} --prompt {prompt_remote}", timeout=200)
        r.setdefault("http", 0)
        if model == "kimi":
            if r["http"] == 429:
                self.st["kimi_429_total"] += 1
                self.st["kimi_429_consecutive"] += 1
                self.st["kimi_429_streak_start"] = self.st["kimi_429_streak_start"] or time.time()
            elif r.get("status") == "ok":
                self.st["kimi_429_consecutive"] = 0
                self.st["kimi_429_streak_start"] = None
        self.save()
        return r

    def ask(self, messages: list, tag: str) -> dict:
        """Returns {"optimizer": "kimi"|"fallback-glm"|"none", "content", "attempts": [...], "kimi_429_total"}."""
        p = STATE / "prompts" / f"{tag}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"messages": messages, "max_tokens": 8000, "temperature": 0.4}))
        subprocess.run(SSH + [DUSK, f"mkdir -p {REMOTE}/prompts"], check=True)
        subprocess.run(["rsync", "-a", "-e", " ".join(SSH), str(p), f"{DUSK}:{REMOTE}/prompts/"], check=True)
        pr = f"{REMOTE}/prompts/{tag}.json"
        attempts = []
        now_t = time.time()
        if self.st["kimi_disabled_until"] > now_t:
            attempts.append({"model": "kimi", "status": "skipped_throttle_hour",
                             "until": dt.datetime.fromtimestamp(self.st["kimi_disabled_until"], dt.timezone.utc).isoformat()})
        else:
            if self.st["kimi_disabled_until"]:
                log("optimizer: kimi throttle hour elapsed - re-probing")
                self.st["kimi_disabled_until"] = 0.0
            for i in range(2):  # first try + ONE exponential-backoff retry on 429/5xx
                r = self._call("kimi", pr)
                attempts.append({k: r.get(k) for k in ("model", "status", "http", "latency_s", "error")})
                if r.get("status") == "ok":
                    self.st["primary"] = "kimi"
                    self.save()
                    return self._out("kimi", r, attempts)
                if not RETRYABLE(r["http"]) or i == 1:
                    break
                time.sleep(max(KIMI_WINDOW_S, 15 * 2 ** i))  # backoff, never inside the 30 s window
            s0 = self.st["kimi_429_streak_start"]
            if s0 and time.time() - s0 >= KIMI_THROTTLE_HOUR_S:
                self.st["kimi_disabled_until"] = time.time() + KIMI_THROTTLE_HOUR_S
                log(f"optimizer: kimi 429 for a full hour ({self.st['kimi_429_consecutive']} consecutive) - "
                    "disabled for 1 h")
        self.st["primary"] = "glm"
        r = self._call("glm", pr)
        attempts.append({k: r.get(k) for k in ("model", "status", "http", "latency_s", "error")})
        self.save()
        if r.get("status") == "ok":
            return self._out("fallback-glm", r, attempts)
        return self._out("none", {}, attempts)

    def _out(self, who: str, r: dict, attempts: list) -> dict:
        return {"optimizer": who, "content": r.get("content", ""), "usage": r.get("usage"), "attempts": attempts,
                "finish_reason": r.get("finish_reason"), "content_chars": r.get("content_chars"),
                "kimi_429_total": self.st["kimi_429_total"], "kimi_429_consecutive": self.st["kimi_429_consecutive"]}


REFLECT_SYS = """You are the reflection/mutation step of a GEPA hill-climb over a persistent HIP decode megakernel
(Flint mid-8L MoE, INT4-affine g64 weights, gfx1150 RDNA3.5 iGPU, ~82 GB/s, one launch, device-wide spin barrier
~1.8 us). Metric: decode32 tok/s on two short-context cases (higher is better); a mutation is accepted only if
>= min_gain faster on BOTH and it passes hard correctness gates. You may ONLY choose values from the knob space
given. Reflect on the evaluation history (speedups, VGPR/scratch spill, hot-code instruction count: larger hot code
has correlated with slower decode via I-cache pressure) and propose ONE untried knob assignment most likely to win.
Reply with a single JSON object and nothing else:
{"knobs": {<knob>: <value>, ...}, "axis": "<pool mutation axis>", "rationale": "<= 60 words"}"""


def reflect_prompt(pool: dict, best: dict, done: set, min_gain: float) -> list:
    hist = []
    for r in pool["evaluations"]:
        b = r.get("build", {})
        hist.append({"id": r["id"], "knobs": r["knobs"], "outcome": r["outcome"],
                     "speedup_vs_parent": r.get("speedup_vs_best"), "parent": r.get("parent"),
                     "hot_instr": b.get("hot_instr"), "scratch": b.get("kernels", {}).get("mid_generate", {}).get("scratch"),
                     "reason": r.get("reason")})
    space = {k: {"axis": ax, "values": vals, "default": d} for k, (ax, vals, d) in KNOBS.items() if k in ALLOWED}
    axes = [{"axis": x.get("axis"), "prior": x.get("prior_evidence", x.get("notes"))} for x in pool["mutation_axes"]]
    user = {"run_context": CONTEXT, "knob_space": space, "pool_axes": axes, "current_best": {"id": best["id"], "knobs": best["knobs"]},
            "min_gain": min_gain, "history": hist, "already_tried": sorted(done)}
    return [{"role": "system", "content": REFLECT_SYS}, {"role": "user", "content": json.dumps(user)}]


def parse_proposal(text: str, best: dict, done: set) -> tuple[dict | None, str, str]:
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None, "", "no JSON object in reply"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, "", f"bad JSON: {e}"
    k = dict(best["knobs"])
    for n, v in (d.get("knobs") or {}).items():
        if n not in KNOBS or n not in ALLOWED or v not in KNOBS[n][1]:
            return None, "", f"outside knob space: {n}={v}"
        k[n] = v
    if knob_key(k) in done:
        return None, "", f"already tried: {knob_key(k)}"
    return k, str(d.get("axis") or "llm"), str(d.get("rationale", ""))[:400]


def headroom(g: dict, inc: dict | None) -> dict:
    def mx(gg):
        it = (gg or {}).get("rec_pp2", {}).get("items", {})
        v = [x["step_rel"] for k, x in it.items() if k.endswith("/fp32_bf16kv") and x.get("step_rel") is not None]
        return max(v) if v else float("nan")
    c, i = mx(g), mx(inc)
    return {"cand": c, "inc": i, "margin": 1e-3 - c, "flag": bool(c > i * 1.001)}


def search_state(pool: dict) -> dict:
    s = pool.setdefault("search", {})
    s.setdefault("driver", "scripts/gepa_kernel_search.py (rec/gemini-kernels)")
    s.setdefault("host", "dusk gfx1150 mid-8L, launches <= 0.25 s, busy<5")
    pool.setdefault("evaluations", [])
    pool.setdefault("rejected", [])
    return s


def evaluate(pool: dict, pool_path: Path, cid: str, knobs: dict, axis: str, parent: str | None,
             src: tuple[str, str], patch: str | None, rounds: int, best: dict | None,
             opt_rec: dict | None = None) -> dict:
    rec = {"id": cid, "time": now(), "axis": axis, "parent": parent, "knobs": knobs, "patch": patch,
           "name": (opt_rec or {}).get("name"), "optimizer": (opt_rec or {}).get("optimizer", "grid"), "kimi_429_total": (opt_rec or {}).get("kimi_429_total"),
           "optimizer_detail": opt_rec,
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
    if best is not None:
        # gates FIRST (rec-ohs): a candidate outside the hard gates is rejected without spending a bench window;
        # fp32_bf16kv headroom vs the incumbent is flagged before any timing.
        rec["emit"] = emit(cid, knobs["workers"])
        rec["gates"] = run_gates(cid, rec["emit"], best)
        hr = headroom(rec["gates"], best.get("gates"))
        rec["headroom"] = hr
        log(f"{cid}: gates {'PASS' if rec['gates']['pass'] else 'FAIL'} BEFORE bench; fp32_bf16kv max {hr['cand']:.4e} "
            f"(incumbent {hr['inc']:.4e}, margin to 1e-3 {hr['margin']:.2e})" + ("  ** HEADROOM FLAG: error up **"
            if hr["flag"] else "") + (f" ({rec['gates'].get('reason')})" if not rec["gates"]["pass"] else ""))
        if not rec["gates"]["pass"]:
            rec["outcome"] = "rejected"
            rec["reason"] = "gates: " + rec["gates"].get("reason", "?")
            return finish(pool, pool_path, rec)
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
    if best is None:
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
        pool["rejected"].append({"id": rec["id"], "knobs": rec["knobs"], "time": rec["time"], "reason": rec["reason"],
                                 "optimizer": rec.get("optimizer"), "kimi_429_total": rec.get("kimi_429_total")})
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
    ap.add_argument("--proposer", choices=["llm", "grid"], default="llm",
                    help="llm: kimi-first reflection w/ glm fallback (grid neighbour if both fail); grid: deterministic")
    ap.add_argument("--kimi-every", type=int, default=3, help="ask kimi on every Nth reflection, grid otherwise")
    ap.add_argument("--knobs", default="", help="comma list: restrict proposals to these knobs (others stay at best)")
    ap.add_argument("--context", default="", help="run context string for the reflection prompt")
    ap.add_argument("--probe-optimizer", action="store_true", help="one tiny reflect call through the fallback chain")
    ap.add_argument("--candidate", type=Path, help='JSON {"knobs": {...}, "patch": "file.patch", "axis": "..."}')
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    global ALLOWED, CONTEXT
    if a.knobs:
        ALLOWED = set(a.knobs.split(","))
    CONTEXT = a.context
    pool = load_pool(a.pool)
    s = search_state(pool)
    sync_harness(a.pool)
    opt = Optimizer()
    if a.probe_optimizer:
        r = opt.ask([{"role": "user", "content": 'Reply with exactly {"ok": true}'}], f"probe_{int(time.time())}")
        print(json.dumps({k: r[k] for k in ("optimizer", "attempts", "kimi_429_total", "kimi_429_consecutive")}))
        print("reply:", (r["content"] or "")[:80])
        return 0 if r["optimizer"] != "none" else 1
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
    # external candidates dedupe on knobs + patch + kernel source commit (a fixed implementation is a new eval)
    done_src = {knob_key(r["knobs"]) + (r.get("patch") or "") + "@" + str(r.get("source")) for r in pool["evaluations"]}
    n_eval, stale = 0, 0
    ext = json.loads(a.candidate.read_text()) if a.candidate else None
    if isinstance(ext, dict):
        ext = [ext]
    while n_eval < a.max_evals and stale < a.patience:
        if ext is not None:  # external list: each entry evaluated once, relative to the CURRENT best
            todo = [c for c in ext if knob_key(dict(best["knobs"], **c.get("knobs", {}))) + (c.get("patch") or "") + "@" + src[1]
                    not in done_src]
            if not todo:
                log("external candidate list exhausted")
                break
            c = todo[0]
            knobs = dict(best["knobs"], **c.get("knobs", {}))
            axis, patch = c.get("axis", "external"), c.get("patch")
            ext_name = c.get("name", "external")
        else:
            cands, opt_rec = [], None
            axis_order = [x["axis"] for x in pool["mutation_axes"]]
            allowed = set(a.knobs.split(",")) if a.knobs else set(KNOBS)
            for name, (ax, vals, _) in sorted(((k, v) for k, v in KNOBS.items() if k in allowed),
                                              key=lambda kv: axis_order.index(kv[1][0])
                                              if kv[1][0] in axis_order else 99):
                bk = dict(DEFAULTS, **best["knobs"])        # older records predate newer knobs
                for v in vals:
                    k = dict(bk, **{name: v})
                    if v != bk[name] and knob_key(k) not in done:
                        cands.append((ax, k))
            if not cands:
                log("neighbourhood exhausted")
                break
            axis, knobs = cands[0]
            patch = None
            n_reflect = s.setdefault("reflections", 0)
            s["reflections"] = n_reflect + 1
            if a.proposer == "llm" and n_reflect % max(1, a.kimi_every) != 0:
                opt_rec = {"optimizer": "grid", "kimi_429_total": opt.st["kimi_429_total"],
                           "note": f"kimi asked every {a.kimi_every}th reflection"}
                log(f"optimizer=grid (kimi every {a.kimi_every}) -> {knob_key(knobs)}")
            elif a.proposer == "llm":
                cid_next = f"c{len([r for r in pool['evaluations'] if r['id'].startswith('c')]) + 1:03d}"
                r = opt.ask(reflect_prompt(pool, best, done, min_gain), cid_next)
                opt_rec = {k: r[k] for k in ("optimizer", "attempts", "kimi_429_total", "kimi_429_consecutive", "usage",
                                              "finish_reason", "content_chars")}
                if r["optimizer"] != "none":
                    k2, ax2, why = parse_proposal(r["content"], best, done)
                    opt_rec["rationale"] = why if k2 else None
                    if k2:
                        knobs, axis = k2, ax2
                    else:
                        opt_rec["proposal_rejected"] = why
                        opt_rec["used"] = "grid neighbour (invalid LLM proposal)"
                        log(f"optimizer {r['optimizer']}: invalid proposal ({why}) - grid neighbour used")
                else:
                    opt_rec["used"] = "grid neighbour (no optimizer answered)"
                log(f"optimizer={r['optimizer']} kimi_429_total={r['kimi_429_total']} -> {knob_key(knobs)}")
        cid = f"c{len([r for r in pool['evaluations'] if r['id'].startswith('c')]) + 1:03d}"
        rec = evaluate(pool, a.pool, cid, knobs, axis, best["id"], src, patch, a.rounds, best,
                       opt_rec if ext is None else {"optimizer": "external", "name": ext_name,
                                                    "kimi_429_total": opt.st["kimi_429_total"]})
        done.add(knob_key(knobs) + (patch or ""))
        done_src.add(knob_key(knobs) + (patch or "") + "@" + src[1])
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
