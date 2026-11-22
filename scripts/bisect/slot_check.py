"""rec-3cr: class (c) check for the O-proj quad-loop reload slots: scratch_store -> first scratch_load of the same slot,
reporting vscnt waits / barriers in between (linear order). Usage: python slot_check.py file.s [...]"""
import re
import sys

for f in sys.argv[1:]:
    L = open(f).read().split("\n")
    s = next(i for i, l in enumerate(L) if l.startswith("mid_generate:"))
    e = next(i for i in range(s, len(L)) if "s_endpgm" in L[i])
    B = L[s:e]
    heads = [i for i, l in enumerate(B) if "v_add_nc_u32_e32 v1, v3, v28" in l]
    slots = set()
    for h in heads:
        for k in range(h - 20, h):
            m = re.search(r"scratch_load_b32 v[356], off, off offset:(\d+)", B[k])
            if m:
                slots.add(int(m.group(1)))
    print("=====", f, "O-proj loop heads", heads, "reload slots", sorted(slots))
    for sl in sorted(slots):
        st = [i for i, l in enumerate(B) if re.search(r"scratch_store_b32 off, v\d+, off offset:%d\b" % sl, l)]
        ld = [i for i, l in enumerate(B) if re.search(r"scratch_load_b32 v\d+, off, off offset:%d\b" % sl, l)]
        for s_ in st:
            nxt = [x for x in ld if x > s_]
            if not nxt:
                continue
            n = nxt[0]
            seg = B[s_:n]
            vs = [l.strip() for l in seg if "s_waitcnt_vscnt" in l]
            nb = sum(1 for l in seg if "s_barrier" in l)
            print(f"  slot {sl}: store@{s_} [{B[s_].strip()[:44]}] -> reload@{n} [{B[n].strip()[:40]}] "
                  f"gap {n - s_}, vscnt between: {vs[:1] or 'NONE'}, s_barrier between: {nb}")
