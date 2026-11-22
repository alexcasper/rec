"""rec-3cr: scan gfx11 asm for sunny's class-B VMEM ordering corners (memo rec3cr-k4096-row0-divergence-isa-answer.md).

Per kernel function, linear scan (labels/branches do NOT reset state: conservative, may over-report across paths):
  (b)  VMEM load writes a VGPR that is the dest of a still-pending VMEM load of a DIFFERENT type (global/scratch/flat)
  (b') VMEM load writes a VGPR still being read by a pending VMEM store (store data, vscnt not waited)
  (c)  scratch_load from a slot with a pending scratch_store to the same slot (no s_waitcnt_vscnt between)
vmcnt model: loads retire in order on s_waitcnt vmcnt(N) (keep the N most recent); s_waitcnt_vscnt null, N likewise for stores.
Usage: python asm_vmem_scan.py file.s [function-substring ...]"""
import re
import sys

REG = re.compile(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]")


def regs(tok):
    out = set()
    for m in REG.finditer(tok):
        if m.group(1) is not None:
            out.add(int(m.group(1)))
        else:
            out.update(range(int(m.group(2)), int(m.group(3)) + 1))
    return out


def kind(op):
    for k in ("global", "scratch", "flat", "buffer"):
        if op.startswith(k + "_"):
            return k
    return None


def scan(lines, name):
    loads, stores, hits = [], [], {"b": [], "b'": [], "c": []}
    for n, raw in lines:
        l = raw.split(";")[0].strip()
        if not l or l.endswith(":") or l.startswith("."):
            continue
        op, _, rest = l.partition(" ")
        if op == "s_waitcnt":
            m = re.search(r"vmcnt\((\d+)\)", rest)
            if m:
                k = int(m.group(1))
                loads = loads[len(loads) - k:] if k else []
            continue
        if op == "s_waitcnt_vscnt":
            m = re.search(r",\s*(0x[0-9a-f]+|\d+)", rest)
            k = int(m.group(1), 0) if m else 0
            stores = stores[len(stores) - k:] if k else []
            continue
        if op.startswith("s_swappc") or op.startswith("s_setpc"):
            continue
        kd = kind(op)
        if kd is None or "atomic" in op:
            continue
        ops = [o.strip() for o in rest.split(",")]
        if "_load" in op:
            dst = regs(ops[0])
            for (pn, pk, pd) in loads:
                if pk != kd and dst & pd:
                    hits["b"].append((n, pn, f"{op} {ops[0]} over pending {pk} load (line {pn})"))
            for (sn, sk, sd, soff) in stores:
                if dst & sd:
                    hits["b'"].append((n, sn, f"{op} {ops[0]} overwrites data of pending {sk} store (line {sn})"))
            if kd == "scratch":
                m = re.search(r"offset:(\d+)", rest)
                off = int(m.group(1)) if m else 0
                for (sn, sk, sd, soff) in stores:
                    if sk == "scratch" and soff == off:
                        hits["c"].append((n, sn, f"{op} slot {off} after scratch_store (line {sn}) with no vscnt wait"))
            loads.append((n, kd, dst))
        elif "_store" in op:
            data = regs(ops[1]) if len(ops) > 1 else set()
            m = re.search(r"offset:(\d+)", rest)
            stores.append((n, kd, data, int(m.group(1)) if m else 0))
    return hits


def main():
    path, subs = sys.argv[1], sys.argv[2:] or ["mid_generate"]
    funcs, cur, buf = {}, None, []
    for i, line in enumerate(open(path), 1):
        m = re.match(r"^([A-Za-z_][\w.$]*):", line)
        if m and not m.group(1).startswith(".L"):
            if cur:
                funcs[cur] = buf
            cur, buf = m.group(1), []
            continue
        if cur:
            buf.append((i, line))
            if "s_endpgm" in line or line.strip().startswith("s_setpc_b64"):
                pass
    if cur:
        funcs[cur] = buf
    for f, lines in funcs.items():
        if not any(s in f for s in subs):
            continue
        h = scan(lines, f)
        print(f"{f[:70]:70s} b={len(h['b'])} b'={len(h[chr(98) + chr(39)])} c={len(h['c'])}")
        for k in ("b", "b'", "c"):
            for x in h[k][:4]:
                print(f"    [{k}] line {x[0]}: {x[2]}")


if __name__ == "__main__":
    main()
