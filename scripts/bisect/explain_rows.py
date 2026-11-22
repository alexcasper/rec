"""rec-3cr: explain the wrong O-proj rows of the mid dump build (row 0 of quads) by data substitution."""
import numpy as np
import torch
from pathlib import Path
from rec_mlx.torch_rocm import load_model

m = load_model(Path('/home/dawn/rec-data/mid-prod'), 'cpu', torch.float32)
q = m.q['model.layers.0.self_attn.o_proj']
Q, H = 4096, 2048
packed = q.packed.numpy().view(np.uint32)            # [2048, 512]
S = q.scales.float().numpy().astype(np.float64)
B = q.biases.float().numpy().astype(np.float64)       # [2048, 64]
shifts = np.arange(8, dtype=np.uint32) * 4
nib = np.bitwise_and(np.right_shift(packed[:, :, None], shifts), 15).reshape(H, Q).astype(np.float64)
D = [np.load(f'/home/dawn/rec-data/bisect/rehd/mid_dump{"" if r == 0 else "_r%d" % r}.npz')['dump'] for r in range(8)]
emb = m.embed.rows(torch.tensor([1]), torch.float32).double().numpy()[0]


def blk(row_q, row_sb, j, jq, x):
    """block j computed with q-bits of (row_q, block jq) and scales/biases of (row_sb, block j)"""
    qs = nib[row_q, jq * 512:(jq + 1) * 512]
    g = np.arange(j * 8, j * 8 + 8)
    s = np.repeat(S[row_sb, g], 64)
    b = np.repeat(B[row_sb, g], 64)
    return float(((qs * s + b) * x[j * 512:(j + 1) * 512]).sum())


for r in range(1, 7):
    a0 = D[0][0, 0, 2 * Q:2 * Q + H].astype(np.float64)
    ar = D[r][0, 0, 2 * Q:2 * Q + H].astype(np.float64)
    x = D[r][0, 0, Q:2 * Q].astype(np.float64)
    for i in np.nonzero(a0 != ar)[0][:3]:
        err = ar[i] - a0[i]
        cands = []
        for j in range(8):
            base = blk(i, i, j, j, x)
            for jq in range(8):
                if jq != j:
                    cands.append((abs(blk(i, i, j, jq, x) - base - err), f"block {j} used q-bits of block {jq}"))
            cands.append((abs(-base - err), f"block {j} dropped"))
            for dr in (1, 2, 3):
                if i + dr < H:
                    cands.append((abs(blk(i + dr, i, j, j, x) - base - err), f"block {j} used q-bits of row +{dr}"))
        cands.sort()
        print(f"r{r} row {i} (quad {i // 4}, warp {(i // 4) % 128}): err {err:+.6f} | best: {cands[0][1]} "
              f"(resid {cands[0][0]:.2e}); 2nd: {cands[1][1]} ({cands[1][0]:.2e})")


print("--- single writer-lane (16 B = 32 nibbles) substitution ---")
def seg(row_q, jq, l, row_sb, j, x):
    e0 = j * 512 + l * 32; q0 = jq * 512 + l * 32
    qs = nib[row_q, q0:q0 + 32]; gidx = (e0 + np.arange(32)) // 64
    return float(((qs * S[row_sb, gidx] + B[row_sb, gidx]) * x[e0:e0 + 32]).sum())
for r in range(1, 7):
    a0 = D[0][0, 0, 2 * Q:2 * Q + H].astype(np.float64)
    ar = D[r][0, 0, 2 * Q:2 * Q + H].astype(np.float64)
    x = D[r][0, 0, Q:2 * Q].astype(np.float64)
    for i in np.nonzero(a0 != ar)[0][:3]:
        err = ar[i] - a0[i]; c = []
        for j in range(8):
            for l in range(16):
                base = seg(i, j, l, i, j, x)
                for src_row in (i, i - 512, i + 512, i + 2, i - 2):
                    if not 0 <= src_row < H: continue
                    for jq in range(8):
                        if src_row == i and jq == j: continue
                        c.append((abs(seg(src_row, jq, l, i, j, x) - base - err), f"blk {j} wlane {l} <- row {src_row - i:+d} blk {jq}"))
        c.sort()
        print(f"r{r} row {i}: err {err:+.6f} | best: {c[0][1]} (resid {c[0][0]:.1e}); 2nd {c[1][1]} ({c[1][0]:.1e})")
