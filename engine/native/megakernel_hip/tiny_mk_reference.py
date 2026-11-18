"""Torch reference: greedy decode of the tiny Flint model from the dequantized
bank weights in tiny_mk_io.npz, then compare token ids with the kernel."""
import numpy as np
import torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "tiny_mk_io.npz")

def bits_to_float(u16):
    return torch.from_numpy((u16.copy().astype(np.uint32) << 16)).view(torch.float32)

HID, HEADS, NKV, HDIM = 128, 4, 2, 64
NEXP, TOPK, MOE_I, SH_I, DENSE_I, VOCAB, NL = 8, 2, 64, 64, 256, 512, 2
EPS, SCALE = 1e-6, 0.125
LOG2 = 13.28771237954945

def bf(name): return bits_to_float(d["w_" + name])
def deq(name, n, K): return torch.from_numpy(d["w_" + name].copy()).reshape(n, K)

prompt = d["prompt"].tolist()
n_out = int(d["n_out"])
total = len(prompt) + n_out

def rms(x, wname):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * bf(wname)

def rope(x2, pos, rotary):
    if not rotary: return x2
    p = x2.reshape(-1, HDIM)
    pairs = torch.arange(HDIM // 2, dtype=torch.float32)
    theta = pos * torch.exp2(-pairs / (HDIM // 2) * LOG2)
    c, s = torch.cos(theta), torch.sin(theta)
    a, b = p[:, :HDIM//2].clone(), p[:, HDIM//2:].clone()
    return torch.cat([a * c - b * s, a * s + b * c], dim=-1).reshape(-1, HDIM)

tokens = []
k_cache = [[None] * NKV for _ in range(NL)]
v_cache = [[None] * NKV for _ in range(NL)]
for pos in range(total):
    tok = prompt[pos] if pos < len(prompt) else tokens[pos - len(prompt)]
    x = bf("embed")[tok].clone()
    for layer in range(NL):
        pfx = "l0" if layer == 0 else "l1"
        rotary = layer == 0
        ln = pfx + "_ln"; post = pfx + "_post"
        hvec = rms(x, ln)
        q = (deq(pfx + "_q", HEADS * HDIM, HID) @ hvec).reshape(HEADS, HDIM)
        k = (deq(pfx + "_k", NKV * HDIM, HID) @ hvec).reshape(NKV, HDIM)
        v = (deq(pfx + "_v", NKV * HDIM, HID) @ hvec).reshape(NKV, HDIM)
        q = rope(q * bf(pfx + "_qn").reshape(1, HDIM), pos, rotary)
        k = rope(k * bf(pfx + "_kn").reshape(1, HDIM), pos, rotary)
        sinks = bf(pfx + "_sinks")
        attn = torch.empty(HEADS, HDIM)
        for kvh in range(NKV):
            k_cache[layer][kvh] = k[kvh].unsqueeze(0) if pos == 0 else torch.cat(
                [k_cache[layer][kvh], k[kvh].unsqueeze(0)], 0)
            v_cache[layer][kvh] = v[kvh].unsqueeze(0) if pos == 0 else torch.cat(
                [v_cache[layer][kvh], v[kvh].unsqueeze(0)], 0)
        for kvh in range(NKV):
            for g in range(HEADS // NKV):
                head = kvh * (HEADS // NKV) + g
                sc = torch.cat([k_cache[layer][kvh] @ q[head],
                                sinks[head].unsqueeze(0)])
                pr = torch.softmax(sc, 0)[:pos + 1]
                attn[head] = pr @ v_cache[layer][kvh]
        x = x + deq(pfx + "_o", HID, HEADS * HDIM) @ attn.reshape(-1)
        x = rms(x, post)
        if layer == 0:
            g = deq("l0_gate", DENSE_I, HID) @ x
            u = deq("l0_up", DENSE_I, HID) @ x
            x = x + deq("l0_down", HID, DENSE_I) @ (torch.nn.functional.silu(g) * u)
        else:
            sc = (bf("router_w").reshape(NEXP, HID) @ x) + bf("router_b")
            sig = torch.sigmoid(sc)
            top = torch.topk(sig, TOPK).indices
            wsel = torch.softmax(sig[top], 0)
            moe = torch.zeros(HID)
            for t, e in enumerate(top.tolist()):
                gw = deq("l1_sw_gate", NEXP * MOE_I, HID).reshape(NEXP, MOE_I, HID)[e] @ x
                uw = deq("l1_sw_up", NEXP * MOE_I, HID).reshape(NEXP, MOE_I, HID)[e] @ x
                dn = deq("l1_sw_down", NEXP * HID, MOE_I).reshape(NEXP, HID, MOE_I)[e]
                moe = moe + wsel[t] * (dn @ (torch.nn.functional.silu(gw) * uw))
            sg = deq("l1_sh_gate", SH_I, HID) @ x
            su = deq("l1_sh_up", SH_I, HID) @ x
            moe = moe + deq("l1_sh_down", HID, SH_I) @ (torch.nn.functional.silu(sg) * su)
            x = x + moe
    logits = deq("lm_head", VOCAB, HID) @ rms(x, "fn_norm")
    tokens.append(int(torch.argmax(logits).item()))

kernel_tokens = d["tokens"].tolist()
print("kernel tokens:", kernel_tokens)
print("torch  tokens:", tokens)
# tokens[] has one argmax per position including the prompt positions; the
# kernel only emits the generated ones, so compare against tokens[prompt_len:]
print("MATCH" if kernel_tokens == tokens[len(prompt):] else "MISMATCH")
