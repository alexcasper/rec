"""rec-3cr: extract the generic-K pipelined quad loop (the block(s) with the 4 row reads ds_load_b64 from the warp pool)
from mid_generate and print it with a per-instruction VMEM/LDS wait annotation, for diffing racy vs deterministic builds.
Usage: python quadloop.py file.s [--ctx 140]"""
import re
import sys

path = sys.argv[1]
ctx = int(sys.argv[sys.argv.index("--ctx") + 1]) if "--ctx" in sys.argv else 140
lines = open(path).read().split("\n")
start = next(i for i, l in enumerate(lines) if re.match(r"^mid_generate:", l))
end = next(i for i in range(start, len(lines)) if "s_endpgm" in lines[i])
body = lines[start:end]
reads = [i for i, l in enumerate(body) if "ds_load_b64" in l]
print(f"{path}: ds_load_b64 at body lines {reads}")
if not reads:
    sys.exit()
# loop = from the nearest label at/before the first row read back to the backward branch after the last one
lo = max(i for i in range(reads[0], -1, -1) if re.match(r"^\.LBB", body[i]))
hi = next(i for i in range(reads[-1], len(body)) if re.search(r"s_cbranch_\w+\s+\.LBB|s_branch\s+\.LBB", body[i]))
print(f"loop body lines {lo}..{hi} ({hi - lo} lines)")
for i in range(lo, min(hi + 1, lo + ctx)):
    l = body[i]
    if not l.strip() or l.strip().startswith(";") or l.strip().startswith("."):
        if not re.match(r"^\.LBB", l):
            continue
    print(f"{i:6d} {l.rstrip()[:110]}")
