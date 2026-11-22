"""rec-3cr: diff GEPA_DUMP repeats. First (local step, layer, phase) where any repeat differs from repeat 0, and per-phase
max |diff| for the first few divergent cells. Phases in kernel order: q (after QKV+RoPE), attn, x_attn (after O-proj
residual), scores (router), x_mlp (after MLP). Usage: python dump_diff.py r0.npz r1.npz [r2.npz ...] [--qdim 4096 --hid 2048 --nexp 256]"""
import argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+")
ap.add_argument("--qdim", type=int, default=4096); ap.add_argument("--hid", type=int, default=2048); ap.add_argument("--nexp", type=int, default=256)
a = ap.parse_args()
ws = max(a.nexp, 32)
sl = [("q", 0, a.qdim), ("attn", a.qdim, a.qdim), ("o", 2 * a.qdim + 2 * a.hid + ws, a.hid), ("x_attn", 2 * a.qdim, a.hid), ("scores", 2 * a.qdim + a.hid, a.nexp),
      ("x_mlp", 2 * a.qdim + a.hid + ws, a.hid)]
D = [np.load(f)["dump"] for f in a.files]
steps, nl, _ = D[0].shape
first = None; rows = []
for lp in range(steps):
    for layer in range(nl):
        for name, o, n in sl:
            ref = D[0][lp, layer, o:o + n]
            for r, d in enumerate(D[1:], 1):
                x = d[lp, layer, o:o + n]
                if not np.array_equal(ref.view(np.uint32), x.view(np.uint32)):
                    diff = np.abs(x - ref); idx = np.nonzero(ref.view(np.uint32) != x.view(np.uint32))[0]
                    rows.append((lp, layer, name, r, float(diff.max()), float(np.abs(ref).max()), len(idx), idx[:8].tolist()))
                    if first is None: first = rows[-1]
        if len(rows) > 12: break
    if len(rows) > 12: break
if first is None:
    print("ALL REPEATS BIT-IDENTICAL across", steps, "steps x", nl, "layers x", [s[0] for s in sl])
else:
    print("FIRST DIVERGENCE: step %d layer %d phase %s (repeat %d): max|d| %.3e (absmax %.3e), %d elems differ, idx %s" % first)
    for r in rows[1:12]: print("  then: step %d layer %d %-6s r%d max|d| %.3e (absmax %.3e) n=%d idx %s" % r)
