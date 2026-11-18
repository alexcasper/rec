"""Compare kernel vs torch first-generated-token logits (pos = prompt_len).

Phase 1 loads flint_tiny.so and runs the megakernel BEFORE importing torch, so
our libhsa binds first (torch's ROCm runtime conflicts otherwise). Phase 2 runs
the torch reference and dumps both logits vectors + diff stats.
"""
import ctypes
import numpy as np

d = np.load("tiny_mk_io.npz")
bank, off = d["bank"], d["offsets"]
prompt, n_out, VOCAB = d["prompt"], 1, 512

kv = np.zeros(2 * 2 * 2 * (len(prompt) + n_out) * 64, dtype=np.uint16)
out_tokens = np.zeros(1, dtype=np.int32)
out_logits = np.zeros(VOCAB, dtype=np.float32)
lib = ctypes.CDLL("./flint_tiny.so")
lib.tiny_launch.restype = None
lib.tiny_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] + [ctypes.c_void_p] + [ctypes.c_int] + [ctypes.c_void_p]
lib.tiny_launch(bank.ctypes.data, off.ctypes.data, kv.ctypes.data,
                np.ascontiguousarray(prompt, np.int32).ctypes.data, ctypes.c_int(len(prompt)),
                out_tokens.ctypes.data, ctypes.c_int(1), out_logits.ctypes.data)
print("kernel first tok", out_tokens[0])
kl = out_logits.copy()
del lib, kv, out_tokens, out_logits

import torch  # noqa: E402  (after CDLL on purpose)


def bits_to_float(u16):
    return torch.from_numpy((u16.copy().astype(np.uint32) << 16)).view(torch.float32)


def bf(name):
    return bits_to_float(d["w_" + name])


def deq(name, n, K):
    return torch.from_numpy(d["w_" + name].copy()).reshape(n, K)


HID, HEADS, NKV, HDIM = 128, 4, 2, 64
NEXP, TOPK, MOE_I, SH_I, DENSE_I, VOCAB2, NL = 8, 2, 64, 64, 256, 512, 2
EPS = 1e-6
LOG2 = 13.28771237954945


def rms(x, wname):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * bf(wname)


def rope(x2, pos, rotary):
    if not rotary:
        return x2
    p = x2.reshape(-1, HDIM)
    pairs = torch.arange(HDIM // 2, dtype=torch.float32)
    theta = pos * torch.exp2(-pairs / (HDIM // 2) * LOG2)
    c, s = torch.cos(theta), torch.sin(theta)
    a, b = p[:, :HDIM // 2].clone(), p[:, HDIM // 2:].clone()
    return torch.cat([a * c - b * s, a * s + b * c], dim=-1).reshape(-1, HDIM)


prompt_l = prompt.tolist()
total = len(prompt_l) + 8
tokens = []
kc = [[None] * NKV for _ in range(NL)]
vc = [[None] * NKV for _ in range(NL)]
tl = None
for pos in range(total):
    tok = prompt_l[pos] if pos < len(prompt_l) else tokens[pos - len(prompt_l)]
    x = bf("embed")[tok].clone()
    for layer in range(NL):
        pfx = "l0" if layer == 0 else "l1"
        rotary = layer == 0
        hvec = rms(x, pfx + "_ln")
        q = (deq(pfx + "_q", HEADS * HDIM, HID) @ hvec).reshape(HEADS, HDIM)
        k = (deq(pfx + "_k", NKV * HDIM, HID) @ hvec).reshape(NKV, HDIM)
        v = (deq(pfx + "_v", NKV * HDIM, HID) @ hvec).reshape(NKV, HDIM)
        q = rope(q * bf(pfx + "_qn").reshape(1, HDIM), pos, rotary)
        k = rope(k * bf(pfx + "_kn").reshape(1, HDIM), pos, rotary)
        sinks = bf(pfx + "_sinks")
        attn = torch.empty(HEADS, HDIM)
        for kvh in range(NKV):
            kc[layer][kvh] = k[kvh].unsqueeze(0) if pos == 0 else torch.cat(
                [kc[layer][kvh], k[kvh].unsqueeze(0)], 0)
            vc[layer][kvh] = v[kvh].unsqueeze(0) if pos == 0 else torch.cat(
                [vc[layer][kvh], v[kvh].unsqueeze(0)], 0)
        for kvh in range(NKV):
            for g in range(HEADS // NKV):
                head = kvh * (HEADS // NKV) + g
                sc = torch.cat([kc[layer][kvh] @ q[head], sinks[head].unsqueeze(0)])
                pr = torch.softmax(sc, 0)[:pos + 1]
                attn[head] = pr @ vc[layer][kvh]
        x = x + deq(pfx + "_o", HID, HEADS * HDIM) @ attn.reshape(-1)
        x = rms(x, pfx + "_post")
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
    logits = deq("lm_head", VOCAB2, HID) @ rms(x, "fn_norm")
    tokens.append(int(torch.argmax(logits).item()))
    if pos == len(prompt_l) - 1:
        tl = logits.numpy()

print("torch first tok", tokens[-1])
print("logits maxabs diff:", float(np.abs(tl - kl).max()))
print("kernel argmax", int(np.argmax(kl)), "torch argmax", int(np.argmax(tl)))
o = np.argsort(tl)[-5:]
print("torch top5:", o.tolist(), [round(float(v), 1) for v in tl[o]])
print("kernel at torch-top5:", [round(float(v), 1) for v in kl[o]])
o2 = np.argsort(kl)[-5:]
print("kernel top5:", o2.tolist(), [round(float(v), 1) for v in kl[o2]])
