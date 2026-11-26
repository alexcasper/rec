"""rec-7oc VRAM delta probe: fp32kv vs bf16kv vs int8kv KV footprint at max_total=8192.

Creates one 8192-cap session per KV dtype, materializes the lazy KV buffers with a
4-token prefill, samples sysfs mem_info_vram_used at 100 ms, peaks per phase.
Memory probe only — NOT a timing benchmark (cotenant state recorded for the log).
"""
import importlib.util, json, sys, threading, time
from pathlib import Path

H = Path("/home/dusk/rec-ve/i8/h/engine/native/megakernel_hip")
sys.path.insert(0, str(H))
spec = importlib.util.spec_from_file_location("ffd", H / "flint_full_driver.py")
ffd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ffd)

LIB = Path("/home/dusk/rec-ve/i8/fulls0/flint_mid.so")
BANK = Path.home() / "xh7/flint-int4.bank"
CKPT = Path.home() / "xh7/flint-int4"
CARD = ffd.hip_card(0)
assert CARD is not None, "cannot map HIP device 0 to sysfs card"
ffd._CARD = CARD

def vram_used():
    return int(open(f"{CARD}/mem_info_vram_used").read())

lib, cfg = ffd.load_lib(LIB)
occ = ffd.occupancy(lib, 3)
ffd.log(f"probe start | cfg full={cfg.get('full')} nl={cfg['nl']} n_full={cfg['n_full']} | occ {occ} | {ffd.cotenants()}")
bank, off = ffd.open_bank(BANK, cfg, None)

PHASES = [("fp32kv", 2), ("bf16kv", 0), ("int8kv", 3)]
prompt = __import__("numpy").array(ffd.case_matrix(ffd.VOCAB)["short4"]["prompt"], dtype=__import__("numpy").int32)
results = {}

for name, mode in PHASES:
    # settle: wait until vram_used stabilizes near current level
    base_samples = [vram_used()]
    t0 = time.time()
    while time.time() - t0 < 8:
        time.sleep(0.4)
        base_samples.append(vram_used())
    baseline = min(base_samples)

    stop = threading.Event()
    peak = [0]
    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], vram_used())
            time.sleep(0.1)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    sess = ffd.create_session(lib, cfg, bank, off, 8192)
    t0 = time.perf_counter()
    dt, tok, last, _, _, ctrl, bad = ffd.run(lib, sess, cfg, prompt, 2, 16, mode)
    run_s = time.perf_counter() - t0
    stop.set(); th.join()
    peak_incl_bank = max(peak[0], vram_used())
    ffd.lib.mid_session_free(sess)

    # wait for async release (driver notes: released within ~2 s)
    t0 = time.time()
    while time.time() - t0 < 12 and vram_used() > baseline + 300 * 2**20:
        time.sleep(0.5)
    footprint = peak_incl_bank - baseline
    results[name] = {
        "mode": mode, "bad": bad, "run_s": round(run_s, 2),
        "baseline_GiB": round(baseline / 2**30, 3),
        "peak_incl_bank_GiB": round(peak_incl_bank / 2**30, 3),
        "session_footprint_MiB": round(footprint / 2**20, 1),
        "tokens": tok.tolist() if tok is not None else None,
    }
    ffd.log(f"phase {name}: footprint {footprint/2**20:.1f} MiB (baseline {baseline/2**30:.2f} GiB, peak {peak_incl_bank/2**30:.2f} GiB) run {run_s:.1f}s bad={bad}")

sb8192 = ffd.session_bytes(len(bank), cfg, 8192)
sb0 = ffd.session_bytes(len(bank), cfg, 0)
results["_meta"] = {
    "card": str(CARD), "lib": str(LIB),
    "session_bytes_8192_GiB": round(sb8192 / 2**30, 3),
    "session_bytes_nonkv_GiB": round(sb0 / 2**30, 3),
    "kv_upper_bound_8192_MiB": round((sb8192 - sb0) / 2**20, 1),
    "cotenants_end": ffd.cotenants(),
}
fp = {k: v["session_footprint_MiB"] for k, v in results.items() if not k.startswith("_")}
for a, b in [("fp32kv", "bf16kv"), ("fp32kv", "int8kv"), ("bf16kv", "int8kv")]:
    if a in fp and b in fp:
        results["_meta"][f"delta_{a}_minus_{b}_MiB"] = round(fp[a] - fp[b], 1)
print(json.dumps(results, indent=1))
out = Path.home() / "rec-ve/i8/vram_probe"
out.mkdir(parents=True, exist_ok=True)
(out / "vram_probe.json").write_text(json.dumps(results, indent=1))
ffd.log("probe done")
