import sys
from collections import Counter

from gguf import GGUFReader

r = GGUFReader(sys.argv[1])
for k, f in r.fields.items():
    if k.startswith(("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
                     "tokenizer.ggml.scores", "tokenizer.chat_template")):
        continue
    try:
        if f.types and f.types[0].name == "STRING":
            val = bytes(f.parts[f.data[0]]).decode()
        elif len(f.data) == 1:
            val = f.parts[f.data[0]].tolist()
        else:
            val = [f.parts[i].tolist() for i in f.data]
            val = val if len(val) <= 64 else f"[{len(val)} items] {val[:8]}..."
    except Exception as e:
        val = f"? {e}"
    print(k, val)
c, sz, blk = Counter(), Counter(), Counter()
for t in r.tensors:
    c[t.tensor_type.name] += 1
    sz[t.tensor_type.name] += int(t.n_bytes)
    blk["blk" if t.name.startswith("blk.") else t.name] += int(t.n_bytes)
print("TYPES", {k: (c[k], round(sz[k] / 2**30, 2)) for k in c})
print("NONBLK MiB", {k: round(v / 2**20, 1) for k, v in blk.items() if k != "blk"}, "BLK GiB", round(blk["blk"] / 2**30, 2))
per = Counter()
for t in r.tensors:
    if t.name.startswith("blk."):
        per[int(t.name.split(".")[1])] += int(t.n_bytes)
print("PER-LAYER MiB", {k: round(v / 2**20) for k, v in sorted(per.items())})
for t in r.tensors:
    if t.name.startswith(("blk.0.", "blk.1.", "blk.2.")) or not t.name.startswith("blk."):
        print("T", t.name, t.tensor_type.name, list(map(int, t.shape)), int(t.n_bytes))
role = {}
for t in r.tensors:
    parts = t.name.split(".")
    key = ".".join(parts[2:]) if parts[0] == "blk" else t.name
    role.setdefault(key, Counter())[t.tensor_type.name] += 1
print("ROLES", {k: dict(v) for k, v in sorted(role.items())})
print("LAYERS0-7", [(t.name, t.tensor_type.name) for t in r.tensors if t.name.startswith("blk.") and int(t.name.split(".")[1]) < 8 and "exps" in t.name])
