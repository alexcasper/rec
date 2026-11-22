"""Full-Flint persistent megakernel driver (rec-u9q / rec-xh7 step 1). Torch-free.

Loads the prebuilt zero-copy bank (pack_flint_bank.py: <bank> + <bank>.off.npy, 17*NL+4
entries) by memmap -- never materialised in host RAM -- into `flint_full.so`
(flint_mid_megakernel.hip built with -DFLINT_CFG_FULL=1) and either

  --emit-suite DIR   rec-mk-out/1 npz per rec-pp2 case (short4, chat, wrap512) and build
                     (fp32kv | fp32_bf16kv | bf16kv), for mid_mk_check.py matrix (CPU reference)
  --bench OUT.json   decode32 differential ABBA (run-definitions R1-R8): warmup 516, then 4
                     pairs (516, 548) in AB BA AB BA order; also the short-ctx pair (8, 40)

    python flint_full_driver.py --bank ~/xh7/flint-int4.bank --ckpt ~/xh7/flint-int4 \\
        (--emit-suite out/ [--builds fp32kv,fp32_bf16kv] [--cases short4,chat,wrap512]
         | --bench bench.json [--mode 0]) [--workers 16] [--lib flint_full.so]

Never run under HSA_OVERRIDE_GFX_VERSION (rec-tan.5: the launch no-ops and returns 0); the
driver refuses. A VRAM preflight reads sysfs (hipMemGetInfo cannot see other processes'
Vulkan allocations on dusk) and refuses to start if bank + session would spill into GTT.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mid_mk_driver import case_matrix, shard_sha256  # noqa: E402

VOCAB, HID, NKV, HDIM, WIN = 150144, 2048, 4, 128, 513
VRAM_SLACK = int(1.5 * 2**30)
STREAM_CHUNK = 256 * 2**20      # host staging for the bank stream (the full bank is 21 GiB; dusk has ~6 GiB available)
CAP_S = 0.25                    # dusk display-iGPU launch cap (rec-u9q wedge rule)
TARGET_S = 0.20                 # continuation chunks are sized to this
BUILDS = {  # name -> (mode, kv_dtype, impl)
    "fp32kv": (2, "fp32", "flint_mid_megakernel.hip[FULL]:mid_generate_fp32kv (fp32-compute, fp32-KV)"),
    "fp32_bf16kv": (1, "bf16", "flint_mid_megakernel.hip[FULL]:mid_generate_fp32_bf16kv (fp32-compute, bf16-KV)"),
    "bf16kv": (0, "bf16", "flint_mid_megakernel.hip[FULL]:mid_generate (production bf16-KV build)"),
}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _local_addrs() -> list[str]:
    try:
        return subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split() or ["127.0.0.1"]
    except Exception:
        return ["127.0.0.1"]


_ADDRS = None
_CARD: str | None = None  # /sys/class/drm/cardN/device of the HIP device, set in main()


def cotenants() -> str:
    """Cotenant state for every log line (benchmark hygiene protocol): llama servers on THIS host
    (dusk binds them to its tailscale address, not localhost) + this GPU's busy counter."""
    global _ADDRS
    _ADDRS = _ADDRS or _local_addrs()
    st = []
    for name, port in (("flint", 8080), ("e4b", 8090), ("qwen", 8091)):
        state = "down"
        for ip in _ADDRS:
            try:
                out = subprocess.run(["curl", "-s", "-m", "1", f"http://{ip}:{port}/slots"],
                                     capture_output=True, text=True, timeout=3).stdout
            except Exception:
                out = ""
            if out:
                state = "busy" if '"is_processing":true' in out else "idle"
                break
        st.append(f"{name}:{state}")
    return ",".join(st) + f",gpu_busy={_busy()},MemAvailable={mem_available_gib():.1f}GiB"


def mem_available_gib() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


def _busy() -> str:
    if _CARD is None:
        return "?"
    try:
        return open(f"{_CARD}/gpu_busy_percent").read().strip()
    except OSError:  # amdgpu refuses busy reads (EBUSY) on a runtime-suspended card
        try:
            return "0(suspended)" if open(f"{_CARD}/power/runtime_status").read().strip() == "suspended" else "?"
        except OSError:
            return "?"


def hip_card(device: int = 0) -> str | None:
    """sysfs device dir of HIP device `device` (respects HIP_VISIBLE_DEVICES), via its PCI bus id."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        buf = ctypes.create_string_buffer(64)
        if hip.hipDeviceGetPCIBusId(buf, 64, device) != 0:
            return None
        bus = buf.value.decode().lower()
    except OSError:
        return None
    for d in glob.glob("/sys/class/drm/card[0-9]*/device"):
        if os.path.basename(os.path.realpath(d)).lower().endswith(bus[-7:]) and os.path.exists(f"{d}/mem_info_vram_total"):
            return d
    return None


def vram_free_bytes() -> int | None:
    if _CARD is None:
        return None
    try:
        return int(open(f"{_CARD}/mem_info_vram_total").read()) - int(open(f"{_CARD}/mem_info_vram_used").read())
    except OSError:
        return None


def load_lib(path: Path) -> tuple[ctypes.CDLL, dict]:
    lib = ctypes.CDLL(str(path.resolve()))   # dlopen needs a path, not a bare name
    cfg = (ctypes.c_int * 8)()
    lib.mid_config(cfg)
    c = dict(zip(("nl", "nexp", "topk", "n_dense", "dense_i", "n_offsets", "n_full", "full"), list(cfg)))
    lib.mid_session_create.restype = ctypes.c_void_p
    lib.mid_session_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.mid_session_run_ex2.restype = ctypes.c_int
    lib.mid_session_run_ex2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.mid_session_run_ex3.restype = ctypes.c_int
    lib.mid_session_run_ex3.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.mid_session_create2.restype = ctypes.c_void_p
    lib.mid_session_create2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mid_session_load_bank.restype = ctypes.c_int
    lib.mid_session_load_bank.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.mid_session_free.restype = None
    lib.mid_session_free.argtypes = [ctypes.c_void_p]
    lib.mid_query_occupancy.restype = ctypes.c_int
    lib.mid_query_occupancy.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4
    return lib, c


def occupancy(lib, mode: int) -> dict:
    v = [ctypes.c_int(0) for _ in range(4)]
    assert lib.mid_query_occupancy(mode, *[ctypes.byref(x) for x in v]) == 0
    return dict(zip(("cus", "max_blocks_per_cu", "max_co_resident_workers", "wave_size"), [x.value for x in v]))


def read_offsets(path: Path) -> np.ndarray:
    """pack_flint_bank.py writes the offset table as raw little-endian int64 (tofile), despite the
    .off.npy name; accept a real .npy as well."""
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        return np.load(path).astype(np.int64)
    return np.fromfile(path, dtype="<i8")


def find_offsets(bank_path: Path, explicit: Path | None = None) -> Path:
    """Offset table next to the bank. pack_flint_bank.py names it <bank>.off.npy; the dusk copy is
    <bank-stem>.off.npy (~/xh7/flint-int4.off.npy next to flint-int4.bank, rec-are D). Accept both, or --off."""
    cands = [explicit] if explicit else [Path(str(bank_path) + ".off.npy"), bank_path.with_suffix(".off.npy")]
    for c in cands:
        if c and c.exists():
            return c
    raise SystemExit(f"offset table not found; tried {[str(c) for c in cands]}")


class Bank:
    """The bank FILE (never mapped or buffered whole): path + size; streamed into the session."""
    def __init__(self, path: Path):
        self.path, self.size = path, os.path.getsize(path)

    def __len__(self) -> int:
        return self.size


def open_bank(bank_path: Path, cfg: dict, off_path: Path | None = None) -> tuple[Bank, np.ndarray]:
    op = find_offsets(bank_path, off_path)
    log(f"offset table {op}")
    off = read_offsets(op)
    if len(off) != cfg["n_offsets"]:
        raise SystemExit(f"offset table has {len(off)} entries, {cfg['n_offsets']} expected by the .so (NL={cfg['nl']})")
    size = os.path.getsize(bank_path)
    if int(off[-1]) != size or bool((np.diff(off) < 0).any()):
        raise SystemExit(f"bad offset table: sentinel {int(off[-1])} vs bank {size} bytes, monotonic={not (np.diff(off) < 0).any()}")
    return Bank(bank_path), off


def session_bytes(bank_size: int, cfg: dict, max_total: int) -> int:
    n_sl = cfg["nl"] - cfg["n_full"]
    kv = (n_sl * 2 * NKV * WIN * HDIM + cfg["n_full"] * 2 * NKV * max_total * HDIM) * (2 + 4)  # bf16 + fp32 copies
    return bank_size + kv + 64 * VOCAB * 4 + 2 * (60000 + VOCAB) * 4 + max_total * cfg["nl"] * cfg["topk"] * 4


def create_session(lib, cfg, bank, off, max_total: int):
    need = session_bytes(len(bank), cfg, max_total)
    free = vram_free_bytes()
    log(f"VRAM preflight: need {need / 2**30:.2f} GiB + {VRAM_SLACK / 2**30:.1f} slack, sysfs free "
        f"{'?' if free is None else f'{free / 2**30:.2f}'} GiB | {cotenants()}")
    if free is not None and need + VRAM_SLACK > free:
        raise SystemExit("REFUSING: session would spill past the VRAM carve-out into GTT/host RAM "
                         "(stop llama-qwen :8091 first; see rec-xh7 step 0)")
    t0 = time.perf_counter()
    sess = lib.mid_session_create2(off.ctypes.data, ctypes.c_int(max_total))
    if not sess:
        raise SystemExit("mid_session_create2 failed (allocation)")
    # stream the bank file through ONE reusable 256 MiB host buffer; drop each chunk from the page cache
    chunk = STREAM_CHUNK
    buf = np.empty(chunk, dtype=np.uint8)
    done, next_log = 0, 0
    fd = os.open(bank.path, os.O_RDONLY)
    try:
        with os.fdopen(fd, "rb", buffering=0) as f:
            while done < bank.size:
                n = min(chunk, bank.size - done)
                got = f.readinto(memoryview(buf)[:n])
                if got != n:
                    raise SystemExit(f"short read at {done}: {got} of {n}")
                if lib.mid_session_load_bank(sess, ctypes.c_size_t(done), buf.ctypes.data, ctypes.c_size_t(n)) != 0:
                    lib.mid_session_free(sess)
                    raise SystemExit(f"mid_session_load_bank failed at {done}")
                try:
                    os.posix_fadvise(f.fileno(), done, n, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
                done += n
                if done >= next_log or done == bank.size:
                    log(f"bank stream {done / 2**30:6.2f}/{bank.size / 2**30:.2f} GiB | host buffer {chunk >> 20} MiB "
                        f"| {cotenants()}")
                    next_log += 2 * 2**30
    finally:
        del buf
    dt = time.perf_counter() - t0
    log(f"session created in {dt:.1f} s (bank {bank.size / 2**30:.3f} GiB streamed H2D, "
        f"{bank.size / 2**30 / max(dt, 1e-9):.2f} GiB/s) | {cotenants()}")
    return sess


def run(lib, sess, cfg, prompt: np.ndarray, n_out: int, workers: int, mode: int,
        step_logits: bool = False, experts: bool = False, start: int = 0):
    tokens = np.zeros(n_out, dtype=np.int32)
    last = np.zeros(VOCAB, dtype=np.float32)
    step = np.zeros((n_out, VOCAB), dtype=np.float32) if step_logits else None
    exp = np.full((len(prompt) + n_out, cfg["nl"], cfg["topk"]), -1, dtype=np.int32) if experts else None
    ctrl = np.zeros(4, dtype=np.uint32)
    t0 = time.perf_counter()
    rc = lib.mid_session_run_ex3(sess, prompt.ctypes.data, len(prompt), tokens.ctypes.data, n_out,
                                 step.ctypes.data if step is not None else None, last.ctypes.data,
                                 workers, mode, ctrl.ctypes.data, exp.ctypes.data if exp is not None else None,
                                 start)
    dt = time.perf_counter() - t0
    # rec-tan.5 validity: rc, barrier abort, executed, written outputs, routed layers written
    bad = []
    if rc != 0 or int(ctrl[2]) != 0 or int(ctrl[1]) == 0:
        bad.append(f"rc={rc} control={ctrl.tolist()}")
    if n_out and not tokens.any():
        bad.append("all-zero tokens")
    if not last.any():
        bad.append("all-zero last_logits")
    if step is not None and not step.any():
        bad.append("all-zero step_logits")
    if exp is not None and bool((exp[:, cfg["n_dense"]:, :] < 0).any()):
        bad.append("unwritten -1 entries in MoE layers of step_experts")
    if bool(((tokens < 0) | (tokens >= VOCAB)).any()):
        bad.append("token id outside vocab")
    return dt, tokens, last, step, exp, ctrl, bad


def emit_suite(lib, cfg, bank, off, ckpt: Path, out: Path, workers: int, builds, cases) -> int:
    out.mkdir(parents=True, exist_ok=True)
    sha = shard_sha256(ckpt)
    log(f"ckpt {ckpt} shard sha256 {sha[:16]} | bank {len(bank)} B | cfg {cfg} | {cotenants()}")
    matrix = case_matrix(VOCAB)
    sess = create_session(lib, cfg, bank, off, max_total=600)
    fails = 0
    try:
        for b in builds:
            mode, kv_dtype, impl = BUILDS[b]
            for cname in cases:
                c = matrix[cname]
                prompt = np.array(c["prompt"], dtype=np.int32)
                dt, tok, last, step, exp, ctrl, bad = run(lib, sess, cfg, prompt, int(c["n_out"]), workers, mode,
                                                         step_logits=True, experts=True)
                if bad:
                    fails += 1
                    log(f"INVALID {cname}_{b}: {bad} | {cotenants()}")
                    continue
                p = len(prompt)
                assert step is not None and exp is not None  # requested above
                np.savez(out / f"{cname}_{b}.npz", schema=np.array("rec-mk-out/1"), case=np.array(cname),
                         ckpt_sha256=np.array(sha), impl=np.array(f"{impl} workers={workers}"),
                         kv_dtype=np.array(kv_dtype), prompt=prompt, tokens=tok, step_logits=step,
                         last_logits=last, chosen_experts=exp[p - 1:p - 1 + len(tok)], step_experts=exp)
                log(f"wrote {cname}_{b}.npz | {dt:7.2f} s | {(p + len(tok)) / dt:6.1f} pos/s | "
                    f"tokens[:6]={tok[:6].tolist()} | {cotenants()}")
    finally:
        lib.mid_session_free(sess)
    return fails


def bench(lib, cfg, bank, off, ckpt: Path, out_json: Path, workers: int, mode: int) -> int:
    sess = create_session(lib, cfg, bank, off, max_total=600)
    prompt512 = np.random.default_rng(0).integers(3, VOCAB, size=512, dtype=np.int32)
    prompt4 = np.array([1, 5, 6, 7], dtype=np.int32)
    occ = occupancy(lib, mode)
    res = {"protocol": "run-definitions R1-R8 differential ABBA (4 pairs), rec-u9q full-Flint",
           "ckpt": str(ckpt), "ckpt_sha256": shard_sha256(ckpt), "bank_bytes": int(len(bank)),
           "config": cfg, "mode": mode, "workers": workers, "occupancy": occ,
           "prompt512_int32_sha256": hashlib.sha256(prompt512.tobytes()).hexdigest(),
           "host": os.uname().nodename, "pairs": {}}
    try:
        def pairs(prompt, na, nb, label):
            rows = []
            for i, order in enumerate(("AB", "BA", "AB", "BA")):
                seq = [(na, "a"), (nb, "b")] if order == "AB" else [(nb, "b"), (na, "a")]
                r = {}
                for n, tag in seq:
                    pre = cotenants()
                    dt, tok, _, _, _, ctrl, bad = run(lib, sess, cfg, prompt, n, workers, mode)
                    r[tag] = (dt, tok, bad, pre, cotenants())
                (ta, toka, bada, prea, posta), (tb, tokb, badb, preb, postb) = r["a"], r["b"]
                d = tb - ta
                row = {"pair": i, "order": order, f"t{len(prompt) + na}_s": round(ta, 5),
                       f"t{len(prompt) + nb}_s": round(tb, 5), "dt32_s": round(d, 5),
                       "decode32_tok_s": round(32.0 / d, 3) if d > 0 else 0.0,
                       "prefix4_match": toka.tolist() == tokb[:na].tolist(), "invalid": bada + badb,
                       "cotenants": [prea, posta, preb, postb]}
                row["valid"] = not row["invalid"] and row["prefix4_match"] and d > 0
                log(f"{label} pair {i} {order}: ta={ta:.3f}s tb={tb:.3f}s -> {row['decode32_tok_s']} tok/s "
                    f"valid={row['valid']} | {postb}")
                rows.append(row)
            ta_m = sum(r[f"t{len(prompt) + na}_s"] for r in rows) / 4
            tb_m = sum(r[f"t{len(prompt) + nb}_s"] for r in rows) / 4
            rates = [r["decode32_tok_s"] for r in rows]
            return {"valid": all(r["valid"] for r in rows), "differential_tok_s": round(32.0 / (tb_m - ta_m), 3),
                    "pair_min": min(rates), "pair_max": max(rates), "ta_mean_s": round(ta_m, 4),
                    "tb_mean_s": round(tb_m, 4), "pairs": rows}

        for _ in range(2):
            run(lib, sess, cfg, prompt4, 8, workers, mode)  # warmup
        res["pairs"]["decode32_short_ctx"] = pairs(prompt4, 4, 36, "short")
        w = run(lib, sess, cfg, prompt512, 4, workers, mode)
        log(f"warmup 516: {w[0]:.3f}s invalid={w[6]}")
        res["pairs"]["decode32_after_512"] = pairs(prompt512, 4, 36, "ctx512")
    finally:
        lib.mid_session_free(sess)
    out_json.write_text(json.dumps(res, indent=1))
    a = res["pairs"]["decode32_after_512"]
    s = res["pairs"]["decode32_short_ctx"]
    log(f"RESULT decode32_after_512 {a['differential_tok_s']} tok/s (pairs {a['pair_min']}..{a['pair_max']}, "
        f"valid={a['valid']}); short_ctx {s['differential_tok_s']} (valid={s['valid']}); "
        f"t516 {a['ta_mean_s']}s -> end_to_end_516 {516 / a['ta_mean_s']:.2f} pos/s")
    return 0 if a["valid"] and s["valid"] else 1


def wait_idle(timeout: float = 300.0) -> bool:
    t0, ok = time.time(), 0
    while time.time() - t0 < timeout:
        b = _busy()
        ok = ok + 1 if b.split("(")[0].isdigit() and int(b.split("(")[0]) < 5 else 0
        if ok >= 3:
            return True
        time.sleep(0.3)
    return False


class CapExceeded(Exception):
    pass


def window_bench(lib, cfg, bank, off, out_json: Path, workers: int, mode: int, pairs_n: int, override: bool,
                 rows: list[str], budget_s: float, label: str) -> int:
    """Bedivere M3.4 decode rows via KV CONTINUATION (M3.1): no launch exceeds CAP_S unless --override-approved.

    Per row: build the context (short: prompt4; ctx512: the rec-u9q seeded 512-token prompt) in continued
    launches of <= TARGET_S, then `pairs_n` ABBA pairs of A = {last}+1 token and B = {last}+k tokens
    (k sized from a probe so B <= TARGET_S); decode tok/s = (k-1) / (mean tB - mean tA): launch overhead cancels.
    Each launch continues from the previous one (start_pos protocol), so depth drifts by ~pairs_n*(k+1); recorded."""
    t_begin = time.time()
    rng_prompt512 = np.random.default_rng(0).integers(3, VOCAB, size=512, dtype=np.int32)
    res = {"protocol": "M3 continuation-chunked decode differential (A={last}+1, B={last}+k), ABBA", "label": label,
           "config": cfg, "mode": mode, "workers": workers, "occupancy": occupancy(lib, mode), "cap_s": CAP_S,
           "override_approved": override, "bank_bytes": len(bank), "host": os.uname().nodename,
           "prompt512_int32_sha256": hashlib.sha256(rng_prompt512.tobytes()).hexdigest(), "rows": {}}
    sess = create_session(lib, cfg, bank, off, max_total=1024)
    launches = []

    def launch(prompt, n_out, start, tag):
        dt, tok, last, _, _, ctrl, bad = run(lib, sess, cfg, np.asarray(prompt, np.int32), n_out, workers, mode,
                                             start=start)
        launches.append({"tag": tag, "start": start, "positions": len(prompt) + n_out, "dt_s": round(dt, 5)})
        if bad:
            raise SystemExit(f"INVALID launch {tag} start={start}: {bad} | {cotenants()}")
        if dt > CAP_S and not override:
            raise CapExceeded(f"{tag}: {dt:.3f} s > cap {CAP_S} s at start={start}")
        return dt, tok

    try:
        if not wait_idle():
            raise SystemExit("gate timeout: busy>=5")
        # probe: 2 positions from a scratch context -> per-position time (upper bound, includes launch overhead)
        dt, _ = launch([1, 5], 0, 0, "probe2")
        dt5, _ = launch([1, 5, 6, 7], 1, 0, "probe5")
        t_pos = max(dt5 / 5.0, dt / 2.0)
        k = int(max(2, min(31, TARGET_S / t_pos - 1)))
        proj = 516 * t_pos + 2 * pairs_n * (k + 3) * t_pos * len(rows) + 10.0
        res.update(t_pos_probe_s=round(t_pos, 5), k=k, projected_bench_s=round(proj, 1))
        log(f"probe: {t_pos * 1e3:.1f} ms/position -> k={k} (B launch ~{(k + 1) * t_pos:.3f} s); projected bench span "
            f"{proj:.0f} s (budget {budget_s:.0f} s) | {cotenants()}")
        if proj > budget_s:
            raise SystemExit(f"projected {proj:.0f} s > budget {budget_s:.0f} s: refusing to start the timed rows")
        for row in rows:
            if time.time() - t_begin > budget_s:
                log(f"budget reached, skipping row {row}")
                break
            prompt = [1, 5, 6, 7] if row == "short" else rng_prompt512.tolist()
            # prefill in continued launches, each re-sized from the previous one's measured per-position time
            # (per-position cost grows with depth: attention is serial over positions)
            start, t_pre, i, tp = 0, 0.0, 0, t_pos
            while i < len(prompt):
                c = int(max(1, min(64, TARGET_S / tp - 1)))
                chunk = prompt[i:i + c]
                last_chunk = i + len(chunk) >= len(prompt)
                n = 1 if last_chunk else 0
                dt, tok = launch(chunk, n, start, f"{row}/prefill")
                tp = max(tp, dt / (len(chunk) + n))
                t_pre += dt
                start += len(chunk)
                i += len(chunk)
            last = int(tok[0])                  # generated at position start-1 ... its KV is written at `start`
            # k for THIS row from the per-position time at this depth, with 20 % headroom for the pairs' drift
            k = int(max(2, min(31, 0.8 * TARGET_S / tp - 1)))
            log(f"{row}: prefill {len(prompt)} pos in {t_pre:.2f} s; {tp * 1e3:.2f} ms/position at depth {start} -> "
                f"k={k} | {cotenants()}")
            rows_ab, depth0 = [], start
            for i in range(pairs_n):
                if not wait_idle(60):
                    raise SystemExit("gate timeout: busy>=5")
                pre = cotenants()
                order = ("A", "B") if i % 2 == 0 else ("B", "A")
                r = {}
                for tag in order:
                    n = 1 if tag == "A" else k
                    dt, tok = launch([last], n, start, f"{row}/{tag}")
                    r[tag] = dt
                    start += n                  # next launch re-runs the last position (protocol)
                    last = int(tok[-1])
                post = cotenants()
                d = r["B"] - r["A"]
                rows_ab.append({"pair": i, "order": "".join(order), "tA_s": round(r["A"], 5), "tB_s": round(r["B"], 5),
                                "tok_s": round((k - 1) / d, 2) if d > 0 else None, "depth": start, "pre": pre,
                                "post": post, "stable": pre.split(",gpu_busy")[0] == post.split(",gpu_busy")[0]})
                log(f"{row} pair {i} {''.join(order)}: tA={r['A']:.4f} tB={r['B']:.4f} -> "
                    f"{rows_ab[-1]['tok_s']} tok/s depth {start} | {post}")
            v = [x for x in rows_ab if x["stable"] and x["tok_s"]]
            tA = sum(x["tA_s"] for x in v) / max(1, len(v))
            tB = sum(x["tB_s"] for x in v) / max(1, len(v))
            res["rows"][row] = {"valid_pairs": len(v), "pairs": rows_ab, "depth_range": [depth0, start], "k": k,
                                "t_pos_at_depth_s": round(tp, 5),
                                "decode_tok_s": round((k - 1) / (tB - tA), 2) if v and tB > tA else None,
                                "prefill_positions": len(prompt), "prefill_s": round(t_pre, 3),
                                "prefill_pos_s": round(len(prompt) / t_pre, 2)}
            log(f"ROW {row}: decode {res['rows'][row]['decode_tok_s']} tok/s ({len(v)}/{pairs_n} stable pairs, depth "
                f"{depth0}..{start}); prefill {len(prompt)} pos in {t_pre:.2f} s | {cotenants()}")
        res["status"] = "ok"
    except CapExceeded as e:
        res["status"] = f"ABORTED cap: {e}"
        log(res["status"])
    finally:
        lib.mid_session_free(sess)
        res["launches"] = launches
        res["max_launch_s"] = max((x["dt_s"] for x in launches), default=0.0)
        res["bench_span_s"] = round(time.time() - t_begin, 1)
        out_json.write_text(json.dumps(res, indent=1))
    log(f"window bench {res['status']} | max launch {res['max_launch_s']} s | span {res['bench_span_s']} s")
    return 0 if res["status"] == "ok" else 1


def emit_spot(lib, cfg, bank, off, ckpt: Path, out: Path, workers: int, n_out: int = 8) -> int:
    """Correctness spot inside the window (coordinator ruling 3): fp32kv short4 via continuation (2 launches,
    bit-identical to one launch by M3.1), rec-mk-out/1 npz for an offline streamed/spot CPU reference."""
    out.mkdir(parents=True, exist_ok=True)
    mode, kv_dtype, impl = BUILDS["fp32kv"]
    prompt = np.array(case_matrix(VOCAB)["short4"]["prompt"], dtype=np.int32)
    sess = create_session(lib, cfg, bank, off, max_total=64)
    try:
        h = n_out // 2
        dt1, t1, _, s1, e1, _, b1 = run(lib, sess, cfg, prompt, h, workers, mode, step_logits=True, experts=True)
        st2 = len(prompt) + h - 1
        dt2, t2, l2, s2, e2, _, b2 = run(lib, sess, cfg, np.array([t1[-1]], np.int32), n_out - h, workers, mode,
                                         step_logits=True, experts=True, start=st2)
        if b1 or b2 or max(dt1, dt2) > CAP_S:
            log(f"spot INVALID/over cap: {b1} {b2} {dt1:.3f} {dt2:.3f}")
            return 1
        tok, step = np.concatenate([t1, t2]), np.concatenate([s1, s2])
        exp = np.concatenate([e1[:len(prompt) + h - 1], e2])       # per position (continued launch re-runs st2)
        p = len(prompt)
        np.savez(out / f"short4_fp32kv.npz", schema=np.array("rec-mk-out/1"), case=np.array("short4"),
                 ckpt_sha256=np.array(shard_sha256(ckpt)), impl=np.array(f"{impl} workers={workers} [M3 spot, 2 continued launches]"),
                 kv_dtype=np.array(kv_dtype), prompt=prompt, tokens=tok, step_logits=step, last_logits=l2,
                 chosen_experts=exp[p - 1:p - 1 + len(tok)], step_experts=exp)
        log(f"spot wrote short4_fp32kv.npz tokens {tok.tolist()} | launches {dt1:.3f}/{dt2:.3f} s | {cotenants()}")
    finally:
        lib.mid_session_free(sess)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="HF flint-int4 dir (shard sha, recorded in npz)")
    ap.add_argument("--lib", type=Path, default=HERE / "flint_full.so")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--emit-suite", type=Path)
    ap.add_argument("--builds", default="fp32kv,fp32_bf16kv")
    ap.add_argument("--cases", default="short4,chat,wrap512")
    ap.add_argument("--bench", type=Path)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--off", type=Path, help="offset table (default: <bank>.off.npy or <bank-stem>.off.npy)")
    ap.add_argument("--window-bench", type=Path, help="M3 continuation-chunked decode rows (cap-compliant)")
    ap.add_argument("--rows", default="ctx512,short")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--budget-s", type=float, default=480.0, help="bench span budget (ops_flint restart SLA)")
    ap.add_argument("--emit-spot", type=Path, help="fp32kv short4 spot npz for an offline CPU reference")
    ap.add_argument("--override-approved", action="store_true",
                    help="Alex's banned-launch override: allow launches > 0.25 s (window-bench only)")
    ap.add_argument("--rehearsal", action="store_true",
                    help="dry rehearsal on a mid-8L bank: skip the full-config + :8080-down preconditions")
    a = ap.parse_args(argv)
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise SystemExit("refusing: HSA_OVERRIDE_GFX_VERSION is set (rec-tan.5: native launch no-ops)")
    lib, cfg = load_lib(a.lib)
    global _CARD
    _CARD = hip_card(0)
    if _CARD is None:
        raise SystemExit("cannot map HIP device 0 to its sysfs card (VRAM preflight impossible)")
    if not cfg["full"] and not a.rehearsal:
        raise SystemExit(f"{a.lib} is not a FLINT_CFG_FULL build: {cfg}")
    if not 1 <= a.workers <= 64:
        raise SystemExit("workers must be in [1, 64]")
    occ = occupancy(lib, a.mode)
    if a.workers > occ["max_co_resident_workers"]:
        raise SystemExit(f"workers={a.workers} > max co-resident {occ['max_co_resident_workers']} ({occ})")
    bank, off = open_bank(a.bank, cfg, a.off)
    log(f"lib {a.lib.name} cfg {cfg} | occupancy {occ} | workers {a.workers} | card {_CARD} | {cotenants()}")
    if (a.window_bench or a.emit_spot) and not a.rehearsal:
        st = cotenants()
        q = subprocess.run(["systemctl", "--user", "is-active", "llama-qwen"], capture_output=True, text=True).stdout.strip()
        if "flint:down" not in st or q == "active":
            raise SystemExit(f"window preconditions not met (need flint :8080 DOWN by ops_flint, llama-qwen inactive): "
                             f"{st} qwen={q}")
    if a.emit_spot:
        rc = emit_spot(lib, cfg, bank, off, a.ckpt, a.emit_spot, a.workers)
        if rc or not a.window_bench:
            return rc
    if a.window_bench:
        return window_bench(lib, cfg, bank, off, a.window_bench, a.workers, a.mode, a.pairs, a.override_approved,
                            a.rows.split(","), a.budget_s, "REHEARSAL mid-8L" if a.rehearsal else "M3 window full Flint")
    if a.emit_suite:
        return 1 if emit_suite(lib, cfg, bank, off, a.ckpt, a.emit_suite, a.workers,
                               a.builds.split(","), a.cases.split(",")) else 0
    if a.bench:
        return bench(lib, cfg, bank, off, a.ckpt, a.bench, a.workers, a.mode)
    ap.error("one of --emit-suite / --bench / --window-bench / --emit-spot")


if __name__ == "__main__":
    sys.exit(main())
