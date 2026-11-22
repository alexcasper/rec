"""rec-3cr: for each quad-row output store (global_store_b32 <addr>, vD, off offset:-12/-8/-4/0) in mid_generate,
follow the fall-through/branch path and report the first instruction that writes vD again, with the waits seen on the way.
Usage: python storewar.py file.s"""
import re
import sys

path = sys.argv[1]
L = open(path).read().split("\n")
s = next(i for i, l in enumerate(L) if l.startswith("mid_generate:"))
e = next(i for i in range(s, len(L)) if "s_endpgm" in L[i])
B = L[s:e]
label = {re.match(r"^(\.LBB\w+):", l).group(1): i for i, l in enumerate(B) if re.match(r"^\.LBB\w+:", l)}
REG = re.compile(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]")


def regs(t):
    out = set()
    for m in REG.finditer(t):
        out |= {int(m.group(1))} if m.group(1) else set(range(int(m.group(2)), int(m.group(3)) + 1))
    return out


def dst(l):
    op, _, rest = l.strip().partition(" ")
    if not rest or op.startswith(("s_", "global_store", "scratch_store", "flat_store", "ds_store", "buffer_store")):
        return op, set()
    if op.startswith("v_dual_"):
        a, _, b = rest.partition("::")
        d = regs(a.split(",")[0]) | regs(b.strip().split(" ", 1)[1].split(",")[0] if " " in b.strip() else "")
        return op, d
    return op, regs(rest.split(",")[0])


pat = re.compile(r"global_store_b32 v\[17:18\], v(\d+), off( offset:(-?\d+))?$")
for i, l in enumerate(B):
    m = pat.search(l.strip())
    if not m:
        continue
    off = m.group(3) or "0"
    if off not in ("-12", "-8", "-4", "0"):
        continue
    d = int(m.group(1))
    j, steps, waits, path_ = i + 1, 0, [], []
    while steps < 400 and j < len(B):
        t = B[j].split(";")[0].strip()
        steps += 1
        if not t or t.startswith(".") and not t.endswith(":"):
            j += 1
            continue
        if t.endswith(":"):
            j += 1
            continue
        if t.startswith("s_cbranch_scc0") and "--taken" in sys.argv:
            tgt = t.split()[1]; path_.append(tgt + "(taken)"); j = label[tgt] + 1; continue
        if t.startswith("s_branch"):
            tgt = t.split()[1]
            path_.append(tgt)
            j = label[tgt] + 1
            continue
        if t.startswith("s_waitcnt"):
            waits.append(t)
        op, w = dst(t)
        if d in w:
            print(f"row off {off:>3} store v{d} (body {i}): first rewrite after {steps} instr: '{t[:70]}' "
                  f"| waits between: {[x for x in waits if 'vscnt' in x or 'vmcnt' in x] or 'NONE'} | via {path_}")
            break
        j += 1
    else:
        print(f"row off {off:>3} store v{d} (body {i}): no rewrite within 400")
