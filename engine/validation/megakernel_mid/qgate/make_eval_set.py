"""rec-1f4 Step 0: generate the FROZEN quality-gate eval set (run once; the JSON is committed and never
regenerated -- later comparisons are only valid against the committed file, pinned by its sha256).

    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONPATH=<engine/src> \
        python make_eval_set.py --ckpt <mid-prod> --out eval_set_v1.json

Composition (all token ids, the dusk mid-8L checkpoint is synthetic -> no natural-language text exists):
  gen items (greedy decode, n_out = 24, prompt <= 16 so every launch is <= 40 positions, ~0.21 s on dusk):
    legacy/short4, legacy/chat  the rec-pp2 case prompts (continuity with every recorded matrix)
    chat/00..09                 [bos, <|user|>, 2..10 plain tokens, <|assistant|>]
    plain/00..09                [bos, 1..15 uniform plain tokens]
    indist/00..09               [bos, 6..15 tokens SAMPLED from the fp32 reference at T = 1]
  ppl items (teacher-forced NLL): 8 x [bos + 31 tokens sampled from the fp32 reference at T = 1]
Reference = rec_mlx.torch_rocm fp32 compute, fp32 KV, CPU (the spec (a) sign-off pin, f0698a5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

SEED = 20261121
N_OUT = 24
PPL_LEN = 32
N_PPL = 8
MAX_POSITIONS = 40   # dusk display-GPU launch cap: <= 40 positions ~ 0.21 s at ~5.3 ms/position


def canonical_sha(items: dict) -> str:
    blob = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists: the eval set is frozen, never regenerate over it")

    import torch

    from rec_mlx.torch_rocm import load_model
    sys.path.insert(0, str(HERE.parent))
    from mid_mk_check import case_matrix, shard_sha256, vocab_of

    vocab = vocab_of(a.ckpt)
    legacy = case_matrix(vocab)
    model = load_model(a.ckpt, device="cpu", dtype=torch.float32)
    model.kv_store_dtype = None
    gen = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    def plain(n: int) -> list[int]:
        return [int(t) for t in rng.integers(6, vocab, size=n)]

    def sample(prefix: list[int], n: int) -> tuple[list[int], list[float]]:
        seq, ents = list(prefix), []
        with torch.inference_mode():
            for _ in range(n):
                logits = model.forward(torch.tensor([seq]))[0, -1].float()
                p = torch.softmax(logits, -1)
                ents.append(float(-(p * torch.log(p.clamp_min(1e-30))).sum()))
                seq.append(int(torch.multinomial(p, 1, generator=gen)))
        return seq, ents

    items = []
    for k in ("short4", "chat"):
        items.append({"id": f"legacy/{k}", "prompt": legacy[k]["prompt"], "n_out": N_OUT})
    for i in range(10):
        items.append({"id": f"chat/{i:02d}", "prompt": [1, 4] + plain(int(rng.integers(2, 11))) + [5], "n_out": N_OUT})
    for i in range(10):
        items.append({"id": f"plain/{i:02d}", "prompt": [1] + plain(int(rng.integers(1, 16))), "n_out": N_OUT})
    ent_all = []
    for i in range(10):
        seq, ents = sample([1], int(rng.integers(6, 16)))
        ent_all += ents
        items.append({"id": f"indist/{i:02d}", "prompt": seq, "n_out": N_OUT})
    ppl = []
    for i in range(N_PPL):
        seq, ents = sample([1], PPL_LEN - 1)
        ent_all += ents
        ppl.append({"id": f"ppl/{i:02d}", "tokens": seq})
    for it in items:
        assert len(it["prompt"]) + it["n_out"] <= MAX_POSITIONS, it["id"]
        assert all(0 <= t < vocab for t in it["prompt"])
    frozen = {"gen_items": items, "ppl_items": ppl}
    doc = {
        "schema": "rec-qgate-evalset/1",
        "version": 1,
        "ckpt": {"path_hint": str(a.ckpt), "model_id": json.loads((a.ckpt / "manifest.json").read_text())["model_id"]
                 if (a.ckpt / "manifest.json").exists() else "?", "shard_sha256": shard_sha256(a.ckpt),
                 "vocab": vocab},
        "generator": {"script": "make_eval_set.py", "seed": SEED, "reference": "torch fp32 compute, fp32 KV, cpu",
                      "sampling_temperature": 1.0},
        "sampler_entropy_nats": {"mean": float(np.mean(ent_all)), "min": float(np.min(ent_all)),
                                 "max": float(np.max(ent_all)), "uniform": math.log(vocab)},
        "sha256": canonical_sha(frozen),
        **frozen,
    }
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}: {len(items)} gen items, {len(ppl)} ppl items, sha256 {doc['sha256']}")
    print("sampler entropy (nats):", doc["sampler_entropy_nats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
