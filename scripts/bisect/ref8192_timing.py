"""rec-kcb: time + memory of the torch fp32 CPU reference at depth 8192 (8192-token seeded prompt + 24 teacher-forced
tokens), the scoring path for the gated depth-8192 row. Placeholder tokens: timing only."""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

import mid_mk_check as mk

ckpt = Path(sys.argv[1])
kv = sys.argv[2] if len(sys.argv) > 2 else "fp32"
prompt = np.random.default_rng(8192).integers(3, 150144, size=8192, dtype=np.int32).tolist()
tokens = [1] * 24
t0 = time.time()
ref = mk.Reference(ckpt, "cpu", kv)
rows, gaps = ref.run(prompt, tokens)
print(json.dumps({"kv": kv, "rows": int(len(rows)), "wall_s": round(time.time() - t0, 1),
                  "maxrss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}))
