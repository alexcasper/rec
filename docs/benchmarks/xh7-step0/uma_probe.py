"""rec-xh7 step 0(b): bounded UMA co-residency probe on dusk (torch-free, HIP via ctypes).

Needs (computed from the real GGUF shapes, see sizes()):
  kernel bank    requant affine INT4 g64 (4.5 bpw, embed 8.5 bpw) or GGUF-as-is bytes
  KV             fp32 (validation build) or bf16, 14 full layers x ctx + 35 sliding x 513
  reference      torch fp32 layer on CPU: packed weights resident in host RAM (+ transient)

Probe: hold VRAM in 1 GiB hipMalloc+hipMemset chunks up to a target, then (while holding)
host RAM in 1 GiB touched numpy chunks up to a target. Every step logs pool state. Stops at
target, first failure, or a safety floor (VRAM free < VRAM_FLOOR, MemAvailable < HOST_FLOOR,
GTT growth > GTT_SPILL = allocation spilling into host RAM). Everything is freed on exit.
"""
import ctypes
import json
import os
import sys
import time

GiB = 1 << 30
VRAM_FLOOR = int(1.5 * GiB)   # headroom left for the resident llama-servers
HOST_FLOOR = 8 * GiB
GTT_SPILL = int(1.5 * GiB)
DEV = "/sys/class/drm/card1/device/"


def rd(name):
    return int(open(DEV + name).read())


def mem_avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError


def sizes():
    H, Q, KV, D, F, E, EI, SI, V, L = 2048, 4096, 512, 128, 6144, 256, 512, 1024, 150144, 49
    dense, moe = 2, 47
    attn = H * Q + 2 * H * KV + Q * H
    q4 = L * attn + dense * 3 * H * F + moe * (E * 3 * H * EI + 3 * H * SI)
    bank_requant = q4 * 4.5 / 8 + V * H * 8.5 / 8 + V * H * 4.5 / 8 + moe * H * E * 2
    bank_requant_lm8 = bank_requant + V * H * 4 / 8
    full, slid = 14, 35

    def kv(ctx, b):
        return 2 * 4 * D * b * (full * ctx + slid * min(ctx, 513))
    return {"params_q4": q4, "bank_requant_int4": bank_requant, "bank_requant_int4_lm8": bank_requant_lm8,
            "bank_gguf_as_is": 18494977472 - 8 * 2**20,  # file minus header/tokenizer (approx); tensors 17.22 GiB
            **{f"kv_{t}_{c}": kv(c, b) for c in (8192, 131072, 262144) for t, b in (("fp32", 4), ("bf16", 2))}}


def state(tag, extra=None):
    s = {"t": round(time.time() - T0, 1), "tag": tag, "vram_used": rd("mem_info_vram_used") / GiB,
         "gtt_used": rd("mem_info_gtt_used") / GiB, "mem_avail": mem_avail() / GiB,
         "busy": rd("gpu_busy_percent"), "cotenants": COT}
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    if hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
        s["hip_free"] = free.value / GiB
    if extra:
        s.update(extra)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    return s


def main():
    global hip, T0, COT
    vram_target = float(sys.argv[1]) * GiB
    host_target = float(sys.argv[2]) * GiB
    COT = sys.argv[3]
    single = len(sys.argv) > 4 and sys.argv[4] == "single"
    T0 = time.time()
    hip = ctypes.CDLL(os.path.expanduser("~/opt/rocm/lib/libamdhip64.so"))
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    hip.hipFree.argtypes = [ctypes.c_void_p]
    print(json.dumps({"sizes_GiB": {k: round(v / GiB, 3) for k, v in sizes().items()}}), flush=True)
    s0 = state("start")
    gtt0 = s0["gtt_used"] * GiB
    ptrs, held, stop = [], 0, "target"
    if single:  # one allocation of the full target, like the driver's single hipMalloc(bank)
        p = ctypes.c_void_p()
        rc = hip.hipMalloc(ctypes.byref(p), int(vram_target))
        if rc == 0:
            rc2 = hip.hipMemset(p, 0, int(vram_target))
            hip.hipDeviceSynchronize()
            ptrs.append(p)
            held = int(vram_target)
            state("single", {"rc": rc, "memset_rc": rc2, "held": held / GiB})
        else:
            stop = f"single hipMalloc rc={rc}"
    else:
        while held + GiB <= vram_target:
            s = state("vram", {"held": held / GiB})
            if s.get("hip_free", 99) * GiB - GiB < VRAM_FLOOR:
                stop = "vram floor"
                break
            if s["mem_avail"] * GiB < HOST_FLOOR or s["gtt_used"] * GiB - gtt0 > GTT_SPILL:
                stop = "host floor / gtt spill"
                break
            p = ctypes.c_void_p()
            rc = hip.hipMalloc(ctypes.byref(p), GiB)
            if rc != 0:
                stop = f"hipMalloc rc={rc}"
                break
            if hip.hipMemset(p, 0, GiB) != 0 or hip.hipDeviceSynchronize() != 0:
                hip.hipFree(p)
                stop = "memset failed"
                break
            ptrs.append(p)
            held += GiB
    sv = state("vram-done", {"held": held / GiB, "stop": stop})
    import numpy as np
    hostbufs, hheld, hstop = [], 0, "target"
    while hheld + GiB <= host_target:
        s = state("host", {"held": held / GiB, "host_held": hheld / GiB})
        if s["mem_avail"] * GiB - GiB < HOST_FLOOR:
            hstop = "host floor"
            break
        b = np.ones(GiB // 8, dtype=np.float64)  # touched
        hostbufs.append(b)
        hheld += GiB
    sh = state("host-done", {"held": held / GiB, "host_held": hheld / GiB, "stop": hstop})
    time.sleep(2)
    for p in ptrs:
        hip.hipFree(p)
    hostbufs.clear()
    hip.hipDeviceSynchronize()
    time.sleep(1)
    se = state("freed")
    print(json.dumps({"RESULT": {"cotenants": COT, "vram_held_GiB": held / GiB, "vram_stop": stop,
                                 "vram_free_at_start_GiB": s0.get("hip_free"),
                                 "vram_free_while_held_GiB": sv.get("hip_free"),
                                 "host_held_GiB": hheld / GiB, "host_stop": hstop,
                                 "mem_avail_while_both_GiB": sh["mem_avail"],
                                 "gtt_growth_GiB": sv["gtt_used"] - s0["gtt_used"],
                                 "restored_vram_used_GiB": se["vram_used"]}}), flush=True)


if __name__ == "__main__":
    main()
