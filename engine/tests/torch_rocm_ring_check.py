"""rec-1h9 GPU check: ring KV == full forward past the window; KV memory bounded."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "rec-ve/engine/src"))
from rec_mlx.torch_rocm import SlidingKV, load_model  # noqa: E402

ck = Path(sys.argv[1])
total = int(sys.argv[2]) if len(sys.argv) > 2 else 800
with torch.inference_mode():
    for dtype in (torch.float32, torch.bfloat16):
        m = load_model(ck, "cuda", dtype)
        ids = torch.tensor([[int(t) for t in np.random.default_rng(2).integers(6, m.cfg.vocab, size=total)]],
                           device="cuda")
        full = m.forward(ids)
        kv = m.new_cache()
        outs = [m.forward(ids[:, :300], kv)] + [m.forward(ids[:, t:t + 1], kv) for t in range(300, total)]
        d = float((torch.cat(outs, 1) - full).abs().max())
        sliding = [e for e in kv.layers if isinstance(e, SlidingKV)]
        full_layers = [e for e in kv.layers if not isinstance(e, SlidingKV)]
        print(f"{str(dtype):15} {total} tok: ring-incremental vs full max|d| {d:.3e}; "
              f"sliding slots {[e.k.shape[2] for e in sliding]}, full T {[e[0].shape[2] for e in full_layers]}, "
              f"KV {kv.nbytes / 2**20:.2f} MiB")
