"""rec-ohs GEPA driver -- dusk-side worker (runs ON dusk under ~/rec-ve/e2e-env.sh + ~/opt/rocm-env.sh).
Called over ssh by scripts/gepa_kernel_search.py; prints ONE JSON line on stdout as its result.

  build  --src K.hip --out DIR [--define K=V ...]   hipcc + --save-temps; resources + hot-code instruction count
  bench  --a A.so --b B.so --wa 16 --wb 16 --case short4 --rounds 8 --ckpt CK
         R1-R8 differential (decode32 short ctx: t(prompt+36) - t(prompt+4)), ABBA per round, busy-gated
         (busy < 5 x3, wait-for-idle up to 300 s) with cotenant state per round. Any launch > 0.25 s
         ABORTS (dusk display-GPU rule) -> status "launch_cap".
  emit   --so DIR/flint_mid.so --out DIR --workers W --ckpt CK
         rec-pp2 suite (short4, chat) x {fp32kv, fp32_bf16kv, bf16kv} with step_experts, plus the frozen
         qgate held-out ppl items (fp32kv, bf16kv) as one-token prefix launches; md5 of every npz.
Cotenants (flint :8080, E4B :8090, llama-qwen) are probed read-only, never touched.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

H = "100.98.73.29"
HERE = Path(__file__).resolve().parent              # <remote>/h/scripts (repo layout mirrored)
MK_DIR = HERE.parent / "engine" / "native" / "megakernel_hip"
QGATE = HERE.parent / "engine" / "validation" / "megakernel_mid" / "qgate" / "qgate.py"
MAX_LAUNCH_S = 0.25
vp = ctypes.c_void_p


def busy() -> int:
    for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        try:
            return int(open(p).read())
        except OSError:
            pass
    return 100


def wait_idle(timeout=300.0) -> bool:
    t0, ok = time.time(), 0
    while time.time() - t0 < timeout:
        ok = ok + 1 if busy() < 5 else 0
        if ok >= 3:
            return True
        time.sleep(0.3)
    return False


def cotenants() -> str:
    st = []
    for name, port in (("flint", 8080), ("e4b", 8090)):
        try:
            slots = subprocess.run(["curl", "-s", "-m", "2", f"http://{H}:{port}/slots"], capture_output=True,
                                   text=True, timeout=4).stdout
            health = subprocess.run(["curl", "-s", "-m", "2", "-o", "/dev/null", "-w", "%{http_code}",
                                     f"http://{H}:{port}/health"], capture_output=True, text=True, timeout=4).stdout
        except Exception:
            slots, health = "", ""
        state = "busy" if '"is_processing":true' in slots else ("idle" if health == "200" else "down")
        st.append(f"{name}:{state}")
    q = subprocess.run(["systemctl", "--user", "is-active", "llama-qwen"], capture_output=True, text=True).stdout.strip()
    st.append(f"qwen:{q or '?'}")
    return ",".join(st)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg} | busy={busy()} | {cotenants()}", file=sys.stderr,
          flush=True)


# ---------------------------------------------------------------------------------------------- build
def cmd_build(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    defs = [f"-D{d}" for d in (a.define or [])]
    tmp = Path(tempfile.mkdtemp(prefix="gepa_b_"))
    try:
        shutil.copy(a.src, tmp / "k.hip")
        t0 = time.time()
        p = subprocess.run(["hipcc", "-O2", "-shared", "-fPIC", "--offload-arch=gfx1150", "--save-temps", *defs,
                            "k.hip", "-o", "k.so"], cwd=tmp, capture_output=True, text=True)
        dt = time.time() - t0
        if p.returncode != 0:
            errs = [l for l in (p.stderr + p.stdout).splitlines() if "error" in l][:5]
            return {"status": "build_failed", "errors": errs, "build_s": round(dt, 1)}
        shutil.copy(tmp / "k.so", out / "flint_mid.so")
        s_files = glob.glob(str(tmp / "*gfx1150*.s"))
        res = {"status": "ok", "build_s": round(dt, 1), "defines": a.define or [],
               "so_sha256": hashlib.sha256((out / "flint_mid.so").read_bytes()).hexdigest()}
        if s_files:
            txt = Path(s_files[0]).read_text()
            shutil.copy(s_files[0], out / "k_gfx1150.s")
            kern = {}
            for m in re.finditer(r"\.group_segment_fixed_size:\s+(\d+).*?\.name:\s+(\S+).*?"
                                 r"\.private_segment_fixed_size:\s+(\d+).*?\.vgpr_count:\s+(\d+)", txt, flags=re.S):
                kern[m.group(2)] = {"lds": int(m.group(1)), "scratch": int(m.group(3)), "vgpr": int(m.group(4))}
            res["kernels"] = kern
            funcs, cur = {}, None
            for line in txt.splitlines():
                mm = re.match(r"^([_A-Za-z][\w.]*):", line)
                if mm:
                    cur = mm.group(1)
                    funcs[cur] = 0
                    continue
                if cur and line.startswith("\t") and not line.strip().startswith((".", ";")):
                    funcs[cur] += 1
            hot = {k: v for k, v in funcs.items() if v > 300 and not k.startswith(("mid_generate_fp32",
                                                                                    "flint_q4_gemv_kernel"))}
            res["hot_instr"] = sum(hot.values())
            res["hot_funcs"] = hot
            res["gemv_vgpr_scratch"] = {
                m.group(1): {"vgpr": int(m.group(2))}
                for m in re.finditer(r"\.set\s+\.L(_Z\w*gemv_op\w*)\.num_vgpr,\s+(\d+)", txt)}
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- bench
def _load(so: Path):
    lib = ctypes.CDLL(str(so.resolve()))
    lib.mid_session_create.restype = vp
    lib.mid_session_create.argtypes = [vp, vp, ctypes.c_int]
    lib.mid_session_run_ex2.restype = ctypes.c_int
    lib.mid_session_run_ex2.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, vp, vp, ctypes.c_int,
                                        ctypes.c_int, vp, vp]
    lib.mid_session_free.argtypes = [vp]
    return lib


def _run(lib, sess, prompt, n_out, workers, mode, vocab):
    tok = np.zeros(n_out, np.int32)
    last = np.zeros(vocab, np.float32)
    ctrl = np.zeros(4, np.uint32)
    t0 = time.perf_counter()
    rc = lib.mid_session_run_ex2(sess, prompt.ctypes.data, len(prompt), tok.ctypes.data, n_out, None,
                                 last.ctypes.data, workers, mode, ctrl.ctypes.data, None)
    dt = time.perf_counter() - t0
    if rc != 0 or ctrl[2] != 0 or ctrl[1] == 0 or not tok.any():
        raise RuntimeError(f"invalid launch rc={rc} ctrl={ctrl.tolist()}")
    return dt, tok


def cmd_bench(a) -> dict:
    sys.path.insert(0, str(MK_DIR))
    from mid_mk_driver import MID_VOCAB, case_matrix, pack_mid_bank

    if a.case not in ("short4", "chat"):
        return {"status": "refused", "reason": "only short-launch cases on dusk"}
    bank, off, _ = pack_mid_bank(Path(a.ckpt))
    prompt = np.array(case_matrix(MID_VOCAB)[a.case]["prompt"], np.int32)
    sides = {"A": (Path(a.a), a.wa), "B": (Path(a.b), a.wb)}
    sess = {}
    for tag, (so, w) in sides.items():
        lib = _load(so)
        s = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, 64)
        for _ in range(2):
            _run(lib, s, prompt, 8, w, a.mode, MID_VOCAB)
        sess[tag] = (lib, s, w)
    rows = {"A": [], "B": []}
    status, max_launch = "ok", 0.0
    try:
        for rnd in range(a.rounds):
            if not wait_idle():
                status = "gate_timeout"
                break
            pre = f"busy={busy()} {cotenants()}"
            order = "AB" if rnd % 2 == 0 else "BA"
            for tag in order:
                lib, s, w = sess[tag]
                t4, tok4 = _run(lib, s, prompt, 4, w, a.mode, MID_VOCAB)
                t36, tok36 = _run(lib, s, prompt, 36, w, a.mode, MID_VOCAB)
                max_launch = max(max_launch, t36)
                ok = tok36[:4].tolist() == tok4.tolist() and t36 > t4
                rows[tag].append({"round": rnd, "t4": round(t4, 5), "t36": round(t36, 5),
                                  "tok_s": round(32 / (t36 - t4), 2) if ok else None, "valid": ok,
                                  "tokens36": tok36.tolist()})
                if t36 > MAX_LAUNCH_S:
                    status = "launch_cap"
                    break
            print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {a.case} round {rnd} {order} | A "
                  f"{rows['A'][-1]['tok_s'] if rows['A'] else None} B {rows['B'][-1]['tok_s'] if rows['B'] else None}"
                  f" | pre {pre} | post busy={busy()} {cotenants()}", file=sys.stderr, flush=True)
            if status != "ok":
                break
    except RuntimeError as e:
        status = f"launch_failed: {e}"
    for lib, s, _ in sess.values():
        lib.mid_session_free(s)
    res = {"status": status, "case": a.case, "rounds": a.rounds, "workers": {"A": a.wa, "B": a.wb},
           "max_launch_s": round(max_launch, 4)}
    if status == "ok":
        for tag in "AB":
            v = [r for r in rows[tag] if r["valid"]]
            if len(v) < max(2, a.rounds // 2):
                res["status"] = f"too_few_valid_pairs_{tag}"
                return res
            t4m, t36m = statistics.mean(r["t4"] for r in v), statistics.mean(r["t36"] for r in v)
            res[tag] = {"valid": len(v), "differential_tok_s": round(32 / (t36m - t4m), 2),
                        "pair_min": min(r["tok_s"] for r in v), "pair_max": max(r["tok_s"] for r in v),
                        "t36_mean_s": round(t36m, 5)}
        res["speedup_B_over_A"] = round(res["B"]["differential_tok_s"] / res["A"]["differential_tok_s"], 4)
        res["tokens_identical_A_B"] = all(x["tokens36"] == y["tokens36"] for x, y in zip(rows["A"], rows["B"]))
    return res


# ----------------------------------------------------------------------------------------------- emit
def cmd_emit(a) -> dict:
    sys.path.insert(0, str(MK_DIR))
    import mid_mk_driver as md

    so = Path(a.so).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not wait_idle():
        return {"status": "gate_timeout"}
    log(f"emit suite {so}")
    md.HERE = so.parent                       # load_lib() opens HERE/flint_mid.so
    t0 = time.time()
    md.emit_conformance_suite(Path(a.ckpt), out / "suite", workers=a.workers,
                              builds=("fp32kv", "fp32_bf16kv", "bf16kv"), emit_experts=True,
                              only_cases=("short4", "chat"))
    t_suite = time.time() - t0
    if not wait_idle():
        return {"status": "gate_timeout"}
    log(f"emit ppl {so}")
    q = subprocess.run([sys.executable, str(QGATE), "emit", "--so", str(so), "--ckpt", a.ckpt,
                        "--out", str(out / "ppl"), "--workers", str(a.workers), "--parts", "ppl"],
                       capture_output=True, text=True)
    if q.returncode != 0:
        return {"status": "ppl_emit_failed", "stderr": q.stderr[-600:], "stdout": q.stdout[-600:]}
    md5 = {}
    for f in sorted(glob.glob(str(out / "suite" / "*.npz"))) + sorted(glob.glob(str(out / "ppl" / "*.npz"))):
        d = np.load(f)
        h = hashlib.md5()
        for k in sorted(d.files):
            if k != "impl":
                h.update(k.encode())
                h.update(np.ascontiguousarray(d[k]).tobytes())
        md5[os.path.relpath(f, out)] = h.hexdigest()
    emit_meta = json.loads((out / "ppl" / "emit.json").read_text())
    log("emit done")
    return {"status": "ok", "suite_s": round(t_suite, 1), "npz_md5": md5,
            "ppl_longest_launch_s": emit_meta["longest_launch_s"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--define", action="append")
    c = sub.add_parser("bench")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--wa", type=int, default=16)
    c.add_argument("--wb", type=int, default=16)
    c.add_argument("--case", default="short4")
    c.add_argument("--rounds", type=int, default=8)
    c.add_argument("--mode", type=int, default=0)
    c.add_argument("--ckpt", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--so", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    res = {"build": cmd_build, "bench": cmd_bench, "emit": cmd_emit}[a.cmd](a)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
