"""Scan hipcc --save-temps output: per-kernel resources (from -Rpass remarks) and instruction mix of
each kernel's body in the device .s (global/scratch/LDS ops, waitcnts, shuffles/dpp).
Usage: python3 isa_scan.py build.log <device.s>"""
import collections
import re
import sys

log, asm = open(sys.argv[1]).read(), open(sys.argv[2]).read().splitlines()

res = collections.OrderedDict()
fn = None
for line in log.splitlines():
    m = re.search(r"Function Name: (\S+)", line)
    if m:
        fn = m.group(1)
        res[fn] = {}
        continue
    m = re.search(r"remark: \S+\s+(VGPRs|ScratchSize \[bytes/lane\]|Occupancy \[waves/SIMD\]|LDS Size \[bytes/block\]|TotalSGPRs|VGPRs Spill|SGPRs Spill): (\d+)", line)
    if m and fn:
        res[fn][m.group(1).replace(" [bytes/lane]", "").replace(" [waves/SIMD]", "").replace(" [bytes/block]", "")] = int(m.group(2))
for f, r in res.items():
    print(f"{f[:60]:60s} {r}")

bodies, cur = {}, None
for line in asm:
    m = re.match(r"^([A-Za-z_][\w.$]*):\s*(;.*)?$", line)
    if m and (m.group(1) in res or m.group(1).startswith("_Z")):
        cur = m.group(1)
        bodies[cur] = []
        continue
    if cur:
        if re.match(r"^\s*s_endpgm", line):
            bodies[cur].append(line)
            cur = None
        else:
            bodies[cur].append(line)
print()
for f, body in bodies.items():
    ops = collections.Counter()
    for line in body:
        m = re.match(r"^\s+([a-z_0-9]+)", line)
        if m:
            ops[m.group(1)] += 1
    sel = {k: v for k, v in sorted(ops.items()) if re.match(r"(global_|scratch_|buffer_|ds_|s_waitcnt|v_permlane|.*dpp|s_barrier|s_sleep|flat_)", k)}
    print(f"{f[:60]}: {sum(ops.values())} instrs; {sel}")
