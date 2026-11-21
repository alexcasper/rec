"""rec-9t8: byte identity of base vs variant gemv on generic-K shapes with n % 4 != 0 (partial trailing quad)
and odd grids. Uses gate7's data/call helpers.  python gtail.py base.so var.so"""
import sys

import numpy as np

sys.path.insert(0, ".")
from gate7 import call, data, load  # noqa: E402

lb, lv = load(sys.argv[1]), load(sys.argv[2])
ok = True
for n, k in ((2046, 4096), (2047, 512), (1025, 1024), (6, 4096), (3, 512), (4098, 2048)):
    d = data(n, k)
    ref, _ = call(lb, d, n, k, 0, 0, 1)
    bad = 0
    for grid in (0, 1, 3, 16):
        for _ in range(5):
            o, _ = call(lv, d, n, k, grid, 0, 1)
            bad += not np.array_equal(o.view(np.uint32), ref.view(np.uint32))
    ok &= bad == 0
    print(f"  n={n:5d} K={k:4d}: {'IDENTICAL x20' if not bad else f'DIFF in {bad}/20 runs'}", flush=True)
print("TAIL GATE:", "PASS" if ok else "FAIL")
