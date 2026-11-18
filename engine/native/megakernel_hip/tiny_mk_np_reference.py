"""Portable numpy reference for the tiny megakernel (no torch, no MLX).

Re-implements rec_mlx.model.Model semantics for the tiny Flint config in fp64
numpy from the driver's dequantized w_* arrays, teacher-forces along the
kernel's token trajectory and checks every generated token plus the last-step
logits. Runs on any host straight after tiny_mk_driver.py.

Ground truth is tiny_mk_mlx_reference.py (the real model code); this file is
cross-checked against it (identical free-running tokens on the seeds in
docs/rocm-model-layer.md). If the two ever disagree, trust the MLX one.

    python tiny_mk_np_reference.py [tiny_mk_io.npz]     # exit 0 = MATCH
"""
import sys
from pathlib import Path

import numpy as np

HID, HEADS, NKV, HDIM = 128, 4, 2, 64
NEXP, TOPK, MOE_I, SH_I, DENSE_I, VOCAB = 8, 2, 64, 64, 256, 512
EPS, SCALE, THETA = 1e-6, 0.125, 10000.0
TIE_TOL = 5e-3   # accepted argmax margin for bf16-vs-fp64 near-ties
LOGIT_RTOL = 2e-2


class Weights:
    def __init__(self, d):
        self.d = d

    def bf(self, n):   # bf16 tensors are stored as raw bits
        return (self.d["w_" + n].astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    def q(self, n, rows, k):   # q4 tensors are stored dequantized (fp32)
        return self.d["w_" + n].reshape(rows, k).astype(np.float64)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def silu(x):
    return 0.5 * x * (1.0 + np.tanh(0.5 * x))   # = x*sigmoid(x), overflow-free


def rope(t, positions):   # NeoX halves (nn.RoPE traditional=False)
    inv = THETA ** (-2.0 * np.arange(HDIM // 2) / HDIM)
    th = positions[:, None] * inv[None]
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    a, b = t[..., :HDIM // 2], t[..., HDIM // 2:]
    return np.concatenate([a * c - b * s, a * s + b * c], -1)


def forward(w, seq):
    """Full causal forward over seq; returns last-position logits (fp64)."""
    T = len(seq)
    x = w.bf("embed").reshape(VOCAB, HID)[seq]
    for layer in range(2):
        p = f"l{layer}"
        h = rms(x, w.bf(p + "_ln"))
        q = rms((h @ w.q(p + "_q", HEADS * HDIM, HID).T).reshape(T, HEADS, HDIM), w.bf(p + "_qn"))
        k = rms((h @ w.q(p + "_k", NKV * HDIM, HID).T).reshape(T, NKV, HDIM), w.bf(p + "_kn"))
        v = (h @ w.q(p + "_v", NKV * HDIM, HID).T).reshape(T, NKV, HDIM)
        if layer == 0:   # L0 sliding+rope, L1 full no-rope (window 513 > tiny seqs)
            q, k = rope(q, np.arange(T)), rope(k, np.arange(T))
        sinks = w.bf(p + "_sinks")
        att = np.zeros((T, HEADS, HDIM))
        causal = np.tril(np.ones((T, T), dtype=bool))
        for head in range(HEADS):
            kv = head // (HEADS // NKV)
            sc = np.where(causal, q[:, head] @ k[:, kv].T * SCALE, -np.inf)
            sc = np.concatenate([sc, np.full((T, 1), sinks[head])], 1)   # sink logit
            pr = np.exp(sc - sc.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            att[:, head] = pr[:, :T] @ v[:, kv]
        x = x + att.reshape(T, -1) @ w.q(p + "_o", HID, HEADS * HDIM).T
        hn = rms(x, w.bf(p + "_post"))
        if layer == 0:
            g, u = hn @ w.q("l0_gate", DENSE_I, HID).T, hn @ w.q("l0_up", DENSE_I, HID).T
            x = x + (silu(g) * u) @ w.q("l0_down", HID, DENSE_I).T
            continue
        scores = 1.0 / (1.0 + np.exp(-(hn @ w.bf("router_w").reshape(NEXP, HID).T)))
        top = np.argsort(-(scores + w.bf("router_b")), 1, kind="stable")[:, :TOPK]
        mix = np.take_along_axis(scores, top, 1)
        mix /= mix.sum(1, keepdims=True) + 1e-20
        G = w.q("l1_sw_gate", NEXP * MOE_I, HID).reshape(NEXP, MOE_I, HID)
        U = w.q("l1_sw_up", NEXP * MOE_I, HID).reshape(NEXP, MOE_I, HID)
        D = w.q("l1_sw_down", NEXP * HID, MOE_I).reshape(NEXP, HID, MOE_I)
        moe = np.zeros_like(x)
        for t in range(T):
            for j, e in enumerate(top[t]):
                moe[t] += mix[t, j] * (D[e] @ (silu(G[e] @ hn[t]) * (U[e] @ hn[t])))
        sg, su = hn @ w.q("l1_sh_gate", SH_I, HID).T, hn @ w.q("l1_sh_up", SH_I, HID).T
        x = x + moe + (silu(sg) * su) @ w.q("l1_sh_down", HID, SH_I).T
    return rms(x[-1], w.bf("fn_norm")) @ w.q("lm_head", VOCAB, HID).T


def check(d, verbose=True):
    w = Weights(d)
    seq, kt = d["prompt"].tolist(), d["tokens"].tolist()
    ref_tokens, bad, ties = [], [], 0
    for step, ktok in enumerate(kt):   # teacher-forced on the kernel trajectory
        lg = forward(w, seq)
        ref_tokens.append(int(lg.argmax()))
        margin = float(lg.max() - lg[ktok])
        if ref_tokens[-1] != ktok:
            if margin > TIE_TOL:
                bad.append((step, ref_tokens[-1], ktok, margin))
            else:
                ties += 1
        seq.append(ktok)
    ok = not bad
    rel = None
    if "last_logits" in d.files:   # kernel's last pos logits -> token n_out+1
        ref = forward(w, seq)
        rel = float(np.abs(d["last_logits"] - ref).max() / max(np.abs(ref).max(), 1e-6))
        ok = ok and rel < LOGIT_RTOL
    if verbose:
        print("kernel tokens:", kt)
        print("ref    tokens:", ref_tokens, "(teacher-forced)")
        for step, want, got, margin in bad:
            print(f"step {step}: reference {want} kernel {got} margin {margin:.3e} > {TIE_TOL}")
        if ties:
            print(f"near-ties accepted: {ties} (margin <= {TIE_TOL})")
        if rel is not None:
            print(f"last-step logits rel maxabs err {rel:.3e} (tol {LOGIT_RTOL})")
        if len(set(ref_tokens)) == 1:
            print("WARNING: single repeated token -- degenerate test data")
        print("MATCH" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "tiny_mk_io.npz"
    sys.exit(0 if check(np.load(path)) else 1)
