"""(dusk GPU tool; usage: python tests/torch_rocm_sync_ab.py CKPT ENGINE_DIR)
rec-aip step 2: isolate the MoE host-sync removal (lm_head cache ON in both arms).
ABBA x3 of decode32-after-512 and short-ctx decode32 with moe_sync_free_decode on/off."""
import json
import sys
import time

import torch

sys.path.insert(0, sys.argv[2] + "/src")
import rec_mlx.torch_rocm as T  # noqa: E402

with torch.inference_mode():
    m = T.load_model(sys.argv[1], device="cuda", dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    p512 = torch.randint(3, m.cfg.vocab, (1, 512), generator=g).cuda()
    short = torch.tensor([[1, 5, 6, 7]], device="cuda")

    def window(prompt):
        kv = m.new_cache()
        tok = int(m.forward(prompt, kv)[0, -1].argmax())
        for _ in range(4):
            tok = int(m.forward(torch.tensor([[tok]], device="cuda"), kv)[0, -1].argmax())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(32):
            tok = int(m.forward(torch.tensor([[tok]], device="cuda"), kv)[0, -1].argmax())
        torch.cuda.synchronize()
        return 32 / (time.perf_counter() - t0)

    window(short)
    res = {True: {"ctx512": [], "short": []}, False: {"ctx512": [], "short": []}}
    for arm in [True, False, False, True] * 3:
        m.moe_sync_free_decode = arm
        res[arm]["ctx512"].append(window(p512))
        res[arm]["short"].append(window(short))
    print(json.dumps({("sync_free" if k else "synced"): v for k, v in res.items()}))
    for k in (False, True):
        print(("sync_free" if k else "synced   "), {w: round(sum(v) / len(v), 3) for w, v in res[k].items()})
