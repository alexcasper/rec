"""Diff two GGUFs: metadata KV + per-tensor (type, shape, max|a-b|). Usage: gguf_diff.py A.gguf B.gguf"""
import sys

import numpy as np
from gguf import GGUFReader


def kv(r):
    out = {}
    for f in r.fields.values():
        if f.name.startswith("tokenizer.ggml.") and f.name not in ("tokenizer.ggml.model", "tokenizer.ggml.pre",
                                                                    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
            continue
        try:
            v = f.contents()
        except Exception:
            v = "?"
        out[f.name] = v
    return out


a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ka, kb = kv(a), kv(b)
for k in sorted(set(ka) | set(kb)):
    if ka.get(k) != kb.get(k):
        print(f"KV {k}: A={str(ka.get(k))[:120]} | B={str(kb.get(k))[:120]}")
ta = {t.name: t for t in a.tensors}
tb = {t.name: t for t in b.tensors}
print(f"tensors A={len(ta)} B={len(tb)} onlyA={sorted(set(ta) - set(tb))[:8]} onlyB={sorted(set(tb) - set(ta))[:8]}")
nbad = 0
for n in sorted(set(ta) & set(tb)):
    x, y = ta[n], tb[n]
    if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape):
        print(f"T {n}: type/shape A={x.tensor_type.name}{list(x.shape)} B={y.tensor_type.name}{list(y.shape)}")
        nbad += 1
        continue
    d = float(np.max(np.abs(np.asarray(x.data, dtype=np.float32) - np.asarray(y.data, dtype=np.float32))))
    s = float(np.max(np.abs(np.asarray(y.data, dtype=np.float32)))) or 1.0
    if d / s > 1e-3:
        nbad += 1
        if nbad <= 25:
            print(f"T {n}: {x.tensor_type.name}{list(x.shape)} max|A-B| {d:.4g} (rel {d / s:.3g})")
print(f"tensors differing (rel > 1e-3): {nbad}")
