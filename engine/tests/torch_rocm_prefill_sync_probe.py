"""rec-aip step 3: what does the prefill MoE host sync cost? (dusk 890M, mid-prod)

    python torch_rocm_prefill_sync_probe.py CKPT ENGINE_DIR OUT.json [--cpu] [--rounds N]

Three prefill512 MoE variants, all required to be bit-identical (torch.equal
logits) to stock:
  A stock    torch_rocm's synced path: per MoE layer one unique().tolist() plus
             one nonzero() per hit expert.
  B oracle   experts and row sets recorded from A, pre-uploaded: zero MoE host
             syncs, identical shapes and ops. A - B is the exact sync cost (the
             lower bound any shape-preserving design can reach).
  C onesync  stable argsort + one bincount().tolist() per layer: identical row
             sets (ascending tokens), 1 sync per layer instead of 1 + #experts.
Every timed forward waits for gpu_busy_percent < 5 on 3 consecutive 0.3 s
samples (rec-ikf aip_gated_probe.py gate). Interleaved ABC/CBA rounds, CUDA
events around each forward plus wall clock. Sync counts via
torch.cuda.set_sync_debug_mode("warn").
"""
from __future__ import annotations

import json
import sys
import time
import types
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, sys.argv[2] + "/src")
import rec_mlx.torch_rocm as T  # noqa: E402

CPU = "--cpu" in sys.argv
DEV = "cpu" if CPU else "cuda"
BUSY = Path("/sys/class/drm/card1/device/gpu_busy_percent")
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 6


def busy() -> int:
    return int(BUSY.read_text().strip()) if BUSY.exists() else 0


def wait_quiet(limit=180.0):
    t0, streak = time.time(), 0
    while time.time() - t0 < limit:
        streak = streak + 1 if busy() < 5 else 0
        if streak >= 3:
            return round(time.time() - t0, 1), True
        time.sleep(0.3)
    return round(time.time() - t0, 1), False


def moe_head(self, x, i):
    """torch_rocm.Model._moe up to the expert loop, verbatim."""
    cfg = self.cfg
    B, L, _ = x.shape
    pre = f"model.layers.{i}.mlp"
    xf = x.float()
    router_w = self.t[f"{pre}.gate.weight"].float()
    bias = self.t[f"{pre}.gate.expert_bias"].float()
    gate = torch.sigmoid(F.linear(xf, router_w))
    idx = torch.topk(gate + bias.view(1, 1, -1), cfg.topk, dim=-1).indices
    w_sel = torch.gather(gate, -1, idx)
    w_sel = w_sel / (w_sel.sum(-1, keepdim=True) + 1e-20)
    flat_x = x.reshape(-1, cfg.hidden)
    flat_idx = idx.reshape(-1, cfg.topk)
    flat_w = w_sel.reshape(-1, cfg.topk)
    out = torch.zeros_like(flat_x, dtype=torch.float32)
    return pre, B, L, flat_x, flat_idx, flat_w, out


def expert_step(self, pre, e, rows, flat_x, flat_idx, flat_w, out):
    """Loop body of the stock path, verbatim (e: host int, rows: device long)."""
    wg = self.q[f"{pre}.switch_mlp.gate_proj"]
    wu = self.q[f"{pre}.switch_mlp.up_proj"]
    wd = self.q[f"{pre}.switch_mlp.down_proj"]
    hit = flat_idx == e
    h = flat_x[rows]
    d = F.linear(F.silu(F.linear(h, wg.expert(e, self.dtype))) * F.linear(h, wu.expert(e, self.dtype)),
                 wd.expert(e, self.dtype))
    wght = (flat_w[rows] * hit[rows]).sum(-1, keepdim=True)
    out.index_add_(0, rows, d.float() * wght)


def tail(self, pre, B, L, out, x):
    return out.to(self.dtype).view(B, L, self.cfg.hidden) + self._mlp(x, f"{pre}.shared_expert")


def make_recorder(model, plan):
    def moe(self, x, i):
        pre, B, L, flat_x, flat_idx, flat_w, out = moe_head(self, x, i)
        steps = []
        for e in torch.unique(flat_idx).tolist():
            rows = (flat_idx == e).any(-1).nonzero(as_tuple=True)[0]
            steps.append((e, rows.clone()))
            expert_step(self, pre, e, rows, flat_x, flat_idx, flat_w, out)
        plan[i] = steps
        return tail(self, pre, B, L, out, x)
    return types.MethodType(moe, model)


def make_oracle(model, plan):
    def moe(self, x, i):
        pre, B, L, flat_x, flat_idx, flat_w, out = moe_head(self, x, i)
        for e, rows in plan[i]:
            expert_step(self, pre, e, rows, flat_x, flat_idx, flat_w, out)
        return tail(self, pre, B, L, out, x)
    return types.MethodType(moe, model)


def make_onesync(model):
    def moe(self, x, i):
        pre, B, L, flat_x, flat_idx, flat_w, out = moe_head(self, x, i)
        k, E = self.cfg.topk, self.cfg.num_experts
        flat = flat_idx.reshape(-1)
        tok = torch.argsort(flat, stable=True) // k  # token ids grouped by expert, ascending within
        counts = torch.bincount(flat, minlength=E).tolist()  # the one host sync
        start = 0
        for e, c in enumerate(counts):
            if c:
                expert_step(self, pre, e, tok[start:start + c], flat_x, flat_idx, flat_w, out)
                start += c
        return tail(self, pre, B, L, out, x)
    return types.MethodType(moe, model)


def main() -> int:
    out_path = Path(sys.argv[3])
    with torch.inference_mode():
        m = T.load_model(sys.argv[1], device=DEV, dtype=torch.bfloat16)
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(3, m.cfg.vocab, (1, 512), generator=g).to(DEV)
        moe_layers = [i for i in range(m.cfg.layers) if i not in m.cfg.dense_layers]

        def sync():
            if not CPU:
                torch.cuda.synchronize()

        def fwd():
            return m.forward(ids, m.new_cache())

        plan: dict[int, list] = {}
        m._moe = make_recorder(m, plan)
        ref = fwd()  # also warms the lm_head cache
        del m._moe
        stock = fwd()
        variants = {"A_stock": None, "B_oracle": make_oracle(m, plan), "C_onesync": make_onesync(m)}

        def use(name):
            if variants[name] is None:
                m.__dict__.pop("_moe", None)
            else:
                m._moe = variants[name]

        identical = {}
        for name in variants:
            use(name)
            identical[name] = bool(torch.equal(fwd(), ref)) and bool(torch.equal(stock, ref))
        n_experts = {i: len(plan[i]) for i in moe_layers}

        syncs = {}
        if not CPU:
            for name in variants:
                use(name)
                sync()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    torch.cuda.set_sync_debug_mode("warn")
                    try:
                        fwd()
                        sync()
                    finally:
                        torch.cuda.set_sync_debug_mode("default")
                syncs[name] = sum("synchroniz" in str(x.message).lower() for x in w)

        for name in variants:  # warmup
            use(name)
            for _ in range(2):
                fwd()
        sync()
        times = {n: {"event_ms": [], "wall_ms": [], "quiet_wait_s": []} for n in variants}
        skipped = 0
        for r in range(ROUNDS):
            for name in (list(variants) if r % 2 == 0 else list(variants)[::-1]):
                use(name)
                waited, ok = wait_quiet()
                if not ok:
                    skipped += 1
                    continue
                sync()
                if CPU:
                    t0 = time.perf_counter()
                    fwd()
                    wall = (time.perf_counter() - t0) * 1e3
                    ev = wall
                else:
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0 = time.perf_counter()
                    a.record()
                    fwd()
                    b.record()
                    torch.cuda.synchronize()
                    wall = (time.perf_counter() - t0) * 1e3
                    ev = a.elapsed_time(b)
                times[name]["event_ms"].append(round(ev, 3))
                times[name]["wall_ms"].append(round(wall, 3))
                times[name]["quiet_wait_s"].append(waited)
        use("A_stock")

    def med(x):
        x = sorted(x)
        return x[len(x) // 2] if len(x) % 2 else (x[len(x) // 2 - 1] + x[len(x) // 2]) / 2

    summary = {n: {"event_ms_median": med(t["event_ms"]), "wall_ms_median": med(t["wall_ms"]),
                   "min": min(t["event_ms"]), "max": max(t["event_ms"]), "n": len(t["event_ms"])}
               for n, t in times.items() if t["event_ms"]}
    a = summary["A_stock"]["event_ms_median"]
    pairs = [x - y for x, y in zip(times["A_stock"]["event_ms"], times["B_oracle"]["event_ms"])]
    pairs_c = [x - y for x, y in zip(times["A_stock"]["event_ms"], times["C_onesync"]["event_ms"])]
    res = {
        "ckpt": sys.argv[1], "device": DEV, "prompt": "randint(3, vocab, 512), seed 0 (torch_rocm_bench)",
        "experts_hit_per_moe_layer": n_experts, "bit_identical_to_stock": identical,
        "host_syncs_per_forward": syncs, "skipped_not_quiet": skipped, "times": times, "summary": summary,
        "sync_share_of_prefill": round((a - summary["B_oracle"]["event_ms_median"]) / a, 4),
        "paired_A_minus_B_ms_median": round(med(pairs), 3), "paired_A_minus_B_positive": sum(p > 0 for p in pairs),
        "paired_A_minus_C_ms_median": round(med(pairs_c), 3), "rounds": ROUNDS,
        "onesync_recovers": round((a - summary["C_onesync"]["event_ms_median"])
                                  / max(a - summary["B_oracle"]["event_ms_median"], 1e-9), 3),
    }
    out_path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("experts_hit_per_moe_layer", "bit_identical_to_stock",
                                           "host_syncs_per_forward", "skipped_not_quiet", "summary",
                                           "sync_share_of_prefill", "paired_A_minus_B_ms_median",
                                           "paired_A_minus_B_positive", "paired_A_minus_C_ms_median",
                                           "onesync_recovers", "rounds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
