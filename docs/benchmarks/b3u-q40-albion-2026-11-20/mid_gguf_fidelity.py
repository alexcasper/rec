"""Fidelity check: llama.cpp (Vulkan, F16 mid GGUF, temp 0) vs the torch fp32 reference on
mid-prod, teacher-forced on llama.cpp's own greedy trajectory.
    python mid_gguf_fidelity.py ENGINE_SRC CKPT PORT
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
import torch  # noqa: E402
from rec_mlx.torch_rocm import load_model  # noqa: E402

CKPT, PORT = Path(sys.argv[2]), int(sys.argv[3])
PROMPTS = {"short": [1, 4, 777, 8888, 5],
           "rand64": [1] + [int(t) for t in np.random.default_rng(0).integers(3, 150000, 63)]}
N = 16
model = load_model(CKPT, device="cpu", dtype=torch.float32)
out = {}
for name, prompt in PROMPTS.items():
    req = {"prompt": prompt, "n_predict": N, "temperature": 0, "top_k": 1, "cache_prompt": False,
           "return_tokens": True, "n_probs": 2, "ignore_eos": True}
    data = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion", json.dumps(req).encode(),
        {"Content-Type": "application/json"}), timeout=120).read())
    toks = data["tokens"]
    with torch.inference_mode():
        logits = model.forward(torch.tensor([prompt + toks[:-1]]))[0, len(prompt) - 1:].float().numpy()
    ref = logits.argmax(-1).tolist()
    flips = [{"step": i, "llama": t, "ref": r, "margin": float(logits[i, r] - logits[i, t])}
             for i, (t, r) in enumerate(zip(toks, ref)) if t != r]
    out[name] = {"llama_tokens": toks, "ref_argmax": ref, "agree": sum(t == r for t, r in zip(toks, ref)),
                 "n": len(toks), "distinct": len(set(toks)), "flips": flips}
print(json.dumps(out))
