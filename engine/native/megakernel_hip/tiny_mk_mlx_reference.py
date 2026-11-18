"""Production-semantics reference for the tiny megakernel.

Loads the exact weight bank the kernel ran on (tiny_mk_io.npz, written by
tiny_mk_driver.py) into rec_mlx.model.Model -- the real Flint model code, not a
hand-written mirror -- and greedy-decodes the same prompt. Compares generated
tokens and the last-step logits with the kernel output.

Why: the original hand-written torch mirror (tiny_mk_reference.py, retired)
shared the kernel's assumptions, so divergences present in both (missing
SiLU, router-bias placement, q4 unpack period) passed silently on a
degenerate bank. This script is the ground truth; tiny_mk_np_reference.py is
the portable fp64 mirror, pinned to this one by tests/test_tiny_mk_reference.py.

Runs anywhere MLX runs (CPU is fine):
    PYTHONPATH=engine/src python tiny_mk_mlx_reference.py [tiny_mk_io.npz]
Exit code 0 = tokens match and logits within tolerance.
"""
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from rec_mlx.model import Model, ModelArgs

HID, HEADS, NKV, HDIM = 128, 4, 2, 64
NEXP, TOPK, MOE_I, SH_I, DENSE_I, VOCAB = 8, 2, 64, 64, 256, 512

CONFIG = {
    "model_type": "flint", "hidden_size": HID, "intermediate_size": DENSE_I,
    "moe_intermediate_size": MOE_I, "shared_expert_intermediate_size": SH_I,
    "num_hidden_layers": 2, "num_attention_heads": HEADS, "num_key_value_heads": NKV,
    "head_dim": HDIM, "num_experts": NEXP, "num_experts_per_tok": TOPK,
    "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "sliding_window": 513, "layer_types": ["sliding_attention", "full_attention"],
    "no_rope_layer_indices": [1], "mlp_only_layers": [0], "tie_word_embeddings": False,
    "kv_key_bits": 8, "kv_value_bits": 4, "kv_group_size": 64, "eos_token_id": [2],
}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_mk_driver import ORDER_BF16 as BF16_ORDER, Q4_ORDER  # noqa: E402  (single source of bank layout)


def bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def unpack_bank(bank, offsets):
    bf, q4 = {}, {}
    for i, (name, elems) in enumerate(BF16_ORDER):
        o = int(offsets[i])
        bf[name] = bf16_bits_to_f32(bank[o:o + 2 * elems].view(np.uint16))
    for j, (name, n, k) in enumerate(Q4_ORDER):
        o = int(offsets[len(BF16_ORDER) + j])
        pb, sb = n * k // 2, n * (k // 64) * 2
        packed = bank[o:o + pb]
        s = bf16_bits_to_f32(bank[o + pb:o + pb + sb].view(np.uint16)).reshape(n, k // 64)
        b = bf16_bits_to_f32(bank[o + pb + sb:o + pb + 2 * sb].view(np.uint16)).reshape(n, k // 64)
        # MLX affine INT4: uint32 words, element j at bits 4*(j%8), low nibble
        # first -- byte = lo | hi<<4 read little-endian is exactly that order.
        q4[name] = (packed.view(np.uint32).reshape(n, k // 8), s, b)
    return bf, q4


def build_model(bf, q4):
    model = Model(ModelArgs.from_dict(CONFIG))
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and p != "model.embed_tokens")
    w = {"model.embed_tokens.weight": bf["embed"].reshape(VOCAB, HID),
         "model.norm.weight": bf["fn_norm"]}
    for l in range(2):
        p = f"model.layers.{l}"
        w[f"{p}.input_layernorm.weight"] = bf[f"l{l}_ln"]
        w[f"{p}.post_attention_layernorm.weight"] = bf[f"l{l}_post"]
        w[f"{p}.self_attn.q_norm.weight"] = bf[f"l{l}_qn"]
        w[f"{p}.self_attn.k_norm.weight"] = bf[f"l{l}_kn"]
        w[f"{p}.self_attn.softmax_offset"] = bf[f"l{l}_sinks"]
    w["model.layers.1.mlp.gate.weight"] = bf["router_w"].reshape(NEXP, HID)
    w["model.layers.1.mlp.gate.expert_bias"] = bf["router_b"]

    def put(prefix, name, experts=None):
        qw, s, b = q4[name]
        if experts:
            qw, s, b = (a.reshape(experts, a.shape[0] // experts, a.shape[1]) for a in (qw, s, b))
        w[prefix + ".weight"], w[prefix + ".scales"], w[prefix + ".biases"] = qw, s, b

    for l in range(2):
        for proj in "qkvo":
            put(f"model.layers.{l}.self_attn.{proj}_proj", f"l{l}_{proj}")
    for proj in ("gate", "up", "down"):
        put(f"model.layers.0.mlp.{proj}_proj", f"l0_{proj}")
        put(f"model.layers.1.mlp.switch_mlp.{proj}_proj", f"l1_sw_{proj}", experts=NEXP)
        put(f"model.layers.1.mlp.shared_expert.{proj}_proj", f"l1_sh_{proj}")
    put("lm_head", "lm_head")
    model.load_weights([(k, mx.array(v)) for k, v in w.items()], strict=True)
    model.eval()
    return model


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "tiny_mk_io.npz"
    d = np.load(path)
    bf, q4 = unpack_bank(d["bank"], d["offsets"])
    model = build_model(bf, q4)
    prompt, n_out = d["prompt"].tolist(), int(d["n_out"])
    kern_tokens = d["tokens"].tolist()
    # Teacher-force along the KERNEL's trajectory (no cache: full causal fp32
    # forward per step). Each kernel token must be the reference argmax, or a
    # near-tie within TIE_TOL of it -- bf16 activations legitimately flip
    # ties, and a free-running compare would cascade one flip into a MISMATCH.
    TIE_TOL = 5e-3
    seq, tokens, ties, bad = list(prompt), [], 0, []
    for step, ktok in enumerate(kern_tokens):
        lg = np.array(model(mx.array([seq]))[0, -1].astype(mx.float32))
        tokens.append(int(lg.argmax()))
        margin = float(lg.max() - lg[ktok])
        if tokens[-1] != ktok:
            (bad.append((step, tokens[-1], ktok, margin)) if margin > TIE_TOL else None)
            ties += margin <= TIE_TOL
        seq.append(ktok)
    # The kernel's last position (total-1) consumes the final generated token
    # and writes the logits that would pick token n_out+1 (discarded).
    ref = np.array(model(mx.array([seq]))[0, -1].astype(mx.float32))
    print("mlx    tokens:", tokens, "(teacher-forced on kernel trajectory)")
    print("kernel tokens:", kern_tokens)
    print(f"mlx logits: absmax {np.abs(ref).max():.3f} std {ref.std():.3f} distinct-tokens {len(set(tokens))}")
    if ties:
        print(f"near-ties accepted: {ties} (argmax margin <= {TIE_TOL})")
    for step, want, got, margin in bad:
        print(f"step {step}: reference {want} kernel {got} margin {margin:.3e} > {TIE_TOL}")
    ok = not bad and len(kern_tokens) == n_out
    if "last_logits" in d.files:
        kl = d["last_logits"]
        diff = float(np.abs(kl - ref).max())
        rel = diff / max(float(np.abs(ref).max()), 1e-6)
        print(f"last-step logits maxabs diff {diff:.4e} (rel {rel:.3e}); "
              f"top5 mlx {np.argsort(-ref)[:5].tolist()} kernel {np.argsort(-kl)[:5].tolist()}")
        ok = ok and rel < 2e-2
    if len(set(tokens)) == 1:
        print("WARNING: reference emits a single repeated token -- test data is degenerate")
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
