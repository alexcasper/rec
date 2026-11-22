"""rec-6ek bisect: one fp32kv short4 spot (single launch, prompt 4 + 3 tokens = 7 positions, ~0.2 s) per megakernel
build, bank streamed via bank_stream_shim (LD_PRELOAD). Writes rec-mk-out/1 npz scored on dawn with mid_mk_check
spot=True. Usage (on dusk): LD_PRELOAD=shim.so BANK_STREAM_FILE=.. BANK_STREAM_BYTES=.. python spot_bisect.py
  --so X.so --off OFF.npy --ckpt CKPTDIR --out OUT.npz --vocab 150144 --nl 49 --topk 6 [--rehearsal]"""
import argparse, ctypes, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
for a in ("--so", "--off", "--ckpt", "--out"): ap.add_argument(a, required=True)
ap.add_argument("--vocab", type=int, default=150144); ap.add_argument("--nl", type=int, default=49)
ap.add_argument("--topk", type=int, default=6); ap.add_argument("--n-out", type=int, default=3)
ap.add_argument("--workers", type=int, default=16); ap.add_argument("--rehearsal", action="store_true")
ap.add_argument("--repeat", type=int, default=1, help="re-run the same spot N times in one session (determinism)")
ap.add_argument("--dump", action="store_true", help="GEPA_DUMP build: capture per-layer tensors per repeat (rec-3cr)")
a = ap.parse_args()
C = "/sys/class/drm/card1/device/"
def vfree(): return (int(open(C + "mem_info_vram_total").read()) - int(open(C + "mem_info_vram_used").read())) / 2**30
def memavail(): return [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0] / 2**20
def health(p): return subprocess.run(["curl", "-s", "-m1", "-o", "/dev/null", "-w", "%{http_code}", f"http://100.98.73.29:{p}/health"], capture_output=True, text=True).stdout
def cot(): return f"8080:{health(8080)} 8090:{health(8090)} busy={open(C + 'gpu_busy_percent').read().strip()} MemAvailable={memavail():.1f}GiB vram_free={vfree():.1f}GiB"
bank_bytes = int(os.environ["BANK_STREAM_BYTES"])
if not a.rehearsal and health(8080) == "200":
    sys.exit(f"REFUSING: :8080 up | {cot()}")
t = time.time()
while vfree() < bank_bytes / 2**30 + 1.5 and time.time() - t < 15: time.sleep(0.5)
if vfree() < bank_bytes / 2**30 + 1.5: sys.exit(f"REFUSING: VRAM | {cot()}")
for _ in range(60):
    if int(open(C + "gpu_busy_percent").read()) < 5: break
    time.sleep(1)
lib = ctypes.CDLL(str(Path(a.so).resolve())); vp = ctypes.c_void_p
lib.mid_session_create.restype = vp; lib.mid_session_create.argtypes = [vp, vp, ctypes.c_int]
lib.mid_session_run_ex2.restype = ctypes.c_int
lib.mid_session_run_ex2.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, vp, vp, ctypes.c_int, ctypes.c_int, vp, vp]
lib.mid_session_free.argtypes = [vp]
off = np.load(a.off) if a.off.endswith(".npy") and open(a.off, "rb").read(6) == b"\x93NUMPY" else np.fromfile(a.off, dtype="<i8")
assert int(off[-1]) == bank_bytes, (int(off[-1]), bank_bytes)
dummy = np.zeros(16, np.uint8)
t0 = time.perf_counter(); sess = lib.mid_session_create(dummy.ctypes.data, off.ctypes.data, 64); tl = time.perf_counter() - t0
if not sess: sys.exit("create failed")
prompt = np.array([1, 43, 1006, 52006], np.int32)
h = hashlib.sha256(open(Path(a.ckpt) / "model-00001.safetensors", "rb").read()).hexdigest()
n = a.n_out; p = len(prompt); prev = None; dts = []
stride = 0
if a.dump:
    lib.mid_dump_setup.restype = ctypes.c_int; lib.mid_dump_setup.argtypes = [ctypes.c_int]
    lib.mid_dump_fetch.restype = ctypes.c_int; lib.mid_dump_fetch.argtypes = [vp, ctypes.c_size_t]
    stride = lib.mid_dump_setup(p + n)
    if stride <= 0: sys.exit(f"mid_dump_setup -> {stride} (not a GEPA_DUMP build?)")
for rr in range(a.repeat):
    tok = np.zeros(n, np.int32); step = np.zeros((n, a.vocab), np.float32); last = np.zeros(a.vocab, np.float32)
    exp = np.full((len(prompt) + n, a.nl, a.topk), -1, np.int32); ctrl = np.zeros(4, np.uint32)
    t0 = time.perf_counter()
    rc = lib.mid_session_run_ex2(sess, prompt.ctypes.data, len(prompt), tok.ctypes.data, n, step.ctypes.data, last.ctypes.data, a.workers, 2, ctrl.ctypes.data, exp.ctypes.data)
    dt = time.perf_counter() - t0; dts.append(round(dt, 4))
    ok = rc == 0 and ctrl[2] == 0 and ctrl[1] != 0
    outp = a.out if rr == 0 else a.out.replace(".npz", f"_r{rr}.npz")
    extra = {}
    if a.dump:
        dmp = np.empty((p + n) * a.nl * stride, np.float32)
        if lib.mid_dump_fetch(dmp.ctypes.data, dmp.size) != 0: sys.exit("mid_dump_fetch failed")
        extra["dump"] = dmp.reshape(p + n, a.nl, stride)
    np.savez(outp, **extra, schema=np.array("rec-mk-out/1"), case=np.array("short4"), ckpt_sha256=np.array(h),
             impl=np.array(f"{Path(a.so).name} workers={a.workers} [rec-6ek bisect spot r{rr}]"), kv_dtype=np.array("fp32"),
             prompt=prompt, tokens=tok, step_logits=step, last_logits=last, chosen_experts=exp[p - 1:p - 1 + n], step_experts=exp)
    same = None if prev is None else bool(np.array_equal(prev[0], step) and np.array_equal(prev[1], last) and np.array_equal(prev[2], exp))
    print(json.dumps({"so": a.so, "rep": rr, "ok": bool(ok), "rc": rc, "launch_s": round(dt, 4), "tokens": tok.tolist(),
                      "identical_to_prev": same, "cot": cot()}))
    prev = (step.copy(), last.copy(), exp.copy())
lib.mid_session_free(sess)
print(json.dumps({"so": a.so, "load_s": round(tl, 1), "launches": dts}))
if max(dts) > 0.25: print("WARNING launch over 0.25 s cap")
