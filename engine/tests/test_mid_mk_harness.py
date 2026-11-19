"""rec-pp2: the mid-8L megakernel validation harness has teeth (CPU, tiny prod-format).

The harness (engine/validation/megakernel_mid/mid_mk_check.py) is generic over
checkpoints; this runs it on the tiny preset in production format so it stays
fast. GPU / mid-8L runs are documented in the harness README.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "megakernel_mid"))
import mid_mk_check as H  # noqa: E402


@pytest.fixture(scope="module")
def prod_ckpt(tmp_path_factory):
    from rec_mlx.synthetic import build_checkpoint

    return build_checkpoint(tmp_path_factory.mktemp("prod") / "tiny", "tiny", fmt="production")


def run(ckpt, tmp_path, case, impl="torch-fp32", fault="none", **edit):
    out = H.produce(ckpt, case, tmp_path / f"{case}-{impl}-{fault}.npz", impl, "cpu", fault)
    if edit:
        data = dict(np.load(out))
        data.update(edit)
        np.savez(out, **data)
    return H.check(ckpt, out, "cpu", verbose=False)


def test_production_format_checkpoint(prod_ckpt):
    import json
    import struct

    with open(prod_ckpt / "model-00001.safetensors", "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    assert {v["dtype"] for v in header.values()} == {"U8", "BF16"}
    assert not any(k.endswith("expert_bias") for k in header)


@pytest.mark.parametrize("case", ["short4", "wrap512"])
@pytest.mark.parametrize("impl", ["torch-fp32", "torch-fp32-bf16kv"])
def test_correct_implementations_pass(prod_ckpt, tmp_path, case, impl):
    r = run(prod_ckpt, tmp_path, case, impl)
    assert r["pass"], r
    assert r["tokens_exact"] and r["step_logits_max_rel"] < 1e-4
    assert r["ref_kv"] == ("bf16" if impl.endswith("bf16kv") else "fp32")


@pytest.mark.parametrize("fault", ["no-silu", "softmax-mix"])
def test_semantic_faults_fail(prod_ckpt, tmp_path, fault):
    r = run(prod_ckpt, tmp_path, "short4", "torch-fp32", fault)
    assert not r["pass"]
    assert r["step_logits_max_rel"] > 1e-3  # softmax-mix keeps tokens exact: logits must catch it


def test_bf16_kv_kernel_judged_against_fp32_kv_reference_fails(prod_ckpt, tmp_path):
    """kv_dtype must be declared honestly: a bf16-KV kernel claiming fp32 KV fails."""
    r = run(prod_ckpt, tmp_path, "short4", "torch-fp32-bf16kv", kv_dtype=np.array("fp32"))
    assert not r["pass"] and r["last_logits_rel"] > 1e-3


def test_schema_binding_rejects_wrong_ckpt_and_prompt(prod_ckpt, tmp_path):
    r = run(prod_ckpt, tmp_path, "short4", ckpt_sha256=np.array("0" * 64))
    assert not r["pass"] and any("ckpt_sha256" in p for p in r["problems"])
    r = run(prod_ckpt, tmp_path, "short4", prompt=np.array([1, 7, 8, 9], np.int32))
    assert not r["pass"] and any("prompt" in p for p in r["problems"])


def test_case_prompts_valid_for_any_vocab():
    for vocab in (512, 150144):
        for spec in H.case_matrix(vocab).values():
            assert max(spec["prompt"]) < vocab and spec["n_out"] == H.N_OUT
    assert len(H.case_matrix(512)["wrap512"]["prompt"]) + H.N_OUT > 513  # crosses the sliding window


# ---------------------------------------------------------------- router near-tie rule (README 4a)
def _forced_kernel(ckpt, out, kv, swaps, case="short4", experts=False, report_swaps=None):
    """A 'kernel' that is the exact reference except for the given routing swaps:
    greedy decode with the swaps applied, written as rec-mk-out/1. experts=True adds
    step_experts (the routing actually used, or that of `report_swaps` to model a
    kernel that misreports its routing)."""
    ref = H.Reference(ckpt, "cpu", kv)
    prompt = H.case_matrix(H.vocab_of(ckpt))[case]["prompt"]
    tokens = []
    for _ in range(H.N_OUT):
        tokens.append(int(ref.run(prompt, tokens, swaps)[0][-1].argmax()))
    logits, _ = ref.run(prompt, tokens, swaps)
    extra = {}
    if experts:
        if report_swaps is not None:
            ref.run(prompt, tokens, report_swaps)
        extra["step_experts"] = ref.used.copy()
    np.savez(out, schema=np.array(H.SCHEMA), case=np.array(case), ckpt_sha256=np.array(H.shard_sha256(ckpt)),
             impl=np.array(f"forced-routing {swaps}"), kv_dtype=np.array(kv),
             prompt=np.array(prompt, np.int32), tokens=np.array(tokens, np.int32),
             last_logits=logits[-1], step_logits=logits[:-1], **extra)
    return out


def _swap(g):
    return {"layer": g["layer"], "pos": g["pos"], "drop": g["kth"], "add": g["next"]}


@pytest.fixture(scope="module")
def router_ties(prod_ckpt):
    """Two boundary swaps that each break the bf16-KV tier on their own. Tiny's natural
    gaps are >= ~2e-3, so the tests scale ROUTER_TIE_TOL to the measured gap."""
    ref = H.Reference(prod_ckpt, "cpu", "bf16")
    prompt = H.case_matrix(H.vocab_of(prod_ckpt))["short4"]["prompt"]
    p0 = len(prompt) - 1
    tokens = []
    for _ in range(H.N_OUT):
        tokens.append(int(ref.run(prompt, tokens)[0][-1].argmax()))
    base, gaps = ref.run(prompt, tokens)
    picked = []
    for g in sorted((g for g in gaps if p0 + 2 <= g["pos"] <= p0 + 18), key=lambda g: g["gap"]):
        t = list(tokens[: g["pos"] - p0])  # same trajectory up to the swap
        for _ in range(H.N_OUT - len(t)):
            t.append(int(ref.run(prompt, t, [_swap(g)])[0][-1].argmax()))
        lg = ref.run(prompt, t, [_swap(g)])[0]
        if float((np.abs(lg - base).max(-1) / np.abs(base).max(-1)).max()) > 1e-2:
            picked.append(g)
        if len(picked) == 2:
            return picked
    pytest.skip("no two disruptive router boundaries on this checkpoint")


def test_router_rule_constants():
    assert H.ROUTER_TIE_TOL == 2.5e-4 and H.ROUTER_TIE_TIERS == ("bf16",) and H.MAX_ROUTER_SWAPS == 1


def test_single_router_flip_passes_bf16_tier_and_is_logged(prod_ckpt, tmp_path, router_ties, monkeypatch):
    g = router_ties[0]
    out = _forced_kernel(prod_ckpt, tmp_path / "one.npz", "bf16", [_swap(g)])
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", g["gap"] * 0.99)  # just below: rule cannot apply
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and r["router_near_tie"] is None
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", g["gap"] * 1.01)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert r["pass"], r
    rt = r["router_near_tie"]
    assert (rt["layer"], rt["pos"], rt["drop"], rt["add"]) == (g["layer"], g["pos"], g["kth"], g["next"])
    assert rt in r["near_ties_accepted"] and r["stock_problems"]
    assert r["step_logits_max_rel"] < 1e-6  # substituted reference reproduces the kernel


def test_two_router_flips_fail(prod_ckpt, tmp_path, router_ties, monkeypatch):
    a, _ = router_ties
    # second swap chosen on the trajectory that already carries the first one
    ref = H.Reference(prod_ckpt, "cpu", "bf16")
    prompt = H.case_matrix(H.vocab_of(prod_ckpt))["short4"]["prompt"]
    first = np.load(_forced_kernel(prod_ckpt, tmp_path / "a.npz", "bf16", [_swap(a)]))
    _, gaps = ref.run(prompt, first["tokens"].tolist(), [_swap(a)])
    later = [g for g in gaps if g["pos"] > a["pos"] and g["layer"] == a["layer"]]
    b = min(later, key=lambda g: g["gap"])
    out = _forced_kernel(prod_ckpt, tmp_path / "two.npz", "bf16", [_swap(a), _swap(b)])
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", 1.0)  # any single swap is eligible
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and r["router_near_tie"] is None and r["router_candidates_tried"] >= 1


def test_router_flip_gets_no_allowance_in_fp32_tier(prod_ckpt, tmp_path, router_ties, monkeypatch):
    g = router_ties[0]
    out = _forced_kernel(prod_ckpt, tmp_path / "fp32.npz", "fp32", [_swap(g)])
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", 1.0)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and r["router_near_tie"] is None and r["router_candidates_tried"] == 0


def test_semantic_fault_not_rescued_by_router_rule(prod_ckpt, tmp_path, monkeypatch):
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", 1.0)
    r = run(prod_ckpt, tmp_path, "short4", "torch-fp32-bf16kv", "no-silu")
    assert not r["pass"] and r["router_near_tie"] is None


def test_matrix_requires_both_kv_tiers_per_case(prod_ckpt, tmp_path):
    d = tmp_path / "mx"
    d.mkdir()
    for case in H.case_matrix(H.vocab_of(prod_ckpt)):
        H.produce(prod_ckpt, case, d / f"{case}_fp32.npz", "torch-fp32", "cpu")
    assert H.main(["matrix", "--ckpt", str(prod_ckpt), str(d), "--ref-device", "cpu"]) == 1
    for case in H.case_matrix(H.vocab_of(prod_ckpt)):
        H.produce(prod_ckpt, case, d / f"{case}_bf16kv.npz", "torch-fp32-bf16kv", "cpu")
    assert H.main(["matrix", "--ckpt", str(prod_ckpt), str(d), "--ref-device", "cpu"]) == 0


# ---------------------------------------------------------------- step_experts path (preferred)
def test_step_experts_exact_reference_passes_with_no_deviations(prod_ckpt, tmp_path):
    out = _forced_kernel(prod_ckpt, tmp_path / "exact.npz", "bf16", [], experts=True)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert r["pass"] and r["router_path"] == "step_experts" and r["n_routing_deviations"] == 0


def test_step_experts_single_flip_named_by_kernel(prod_ckpt, tmp_path, router_ties, monkeypatch):
    monkeypatch.setattr(H, "ROUTER_EXPERTS_POLICY", "single")  # superseded policy, still selectable
    g = router_ties[0]
    out = _forced_kernel(prod_ckpt, tmp_path / "one_se.npz", "bf16", [_swap(g)], experts=True)
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", g["gap"] * 1.01)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert r["pass"], r
    rt = r["router_near_tie"]
    assert rt["source"] == "step_experts" and (rt["layer"], rt["pos"]) == (g["layer"], g["pos"])
    assert r["n_routing_deviations"] == 1 and r["routing_deviations"][0]["k_k1_exchange"]
    assert r["router_candidates_tried"] == 0  # nothing searched
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", g["gap"] * 0.99)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and "ROUTER_TIE_TOL" in r["router_note"]


def test_step_experts_two_flips_fail(prod_ckpt, tmp_path, router_ties, monkeypatch):
    monkeypatch.setattr(H, "ROUTER_EXPERTS_POLICY", "single")  # superseded policy, still selectable
    a, _ = router_ties
    ref = H.Reference(prod_ckpt, "cpu", "bf16")
    prompt = H.case_matrix(H.vocab_of(prod_ckpt))["short4"]["prompt"]
    first = np.load(_forced_kernel(prod_ckpt, tmp_path / "a.npz", "bf16", [_swap(a)]))
    _, gaps = ref.run(prompt, first["tokens"].tolist(), [_swap(a)])
    b = min((g for g in gaps if g["pos"] > a["pos"] and g["layer"] == a["layer"]), key=lambda g: g["gap"])
    out = _forced_kernel(prod_ckpt, tmp_path / "two_se.npz", "bf16", [_swap(a), _swap(b)], experts=True)
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", 1.0)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and r["router_near_tie"] is None and r["n_routing_deviations"] >= 2

    # a kernel that reports only the first swap cannot be rescued by it
    out = _forced_kernel(prod_ckpt, tmp_path / "lie.npz", "bf16", [_swap(a), _swap(b)], experts=True,
                         report_swaps=[_swap(a)])
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and r["router_near_tie"] is None


def test_step_experts_fp32_tier_gets_no_allowance(prod_ckpt, tmp_path, router_ties, monkeypatch):
    g = router_ties[0]
    out = _forced_kernel(prod_ckpt, tmp_path / "fp32_se.npz", "fp32", [_swap(g)], experts=True)
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", 1.0)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and r["router_near_tie"] is None and r["n_routing_deviations"] == 1


@pytest.mark.parametrize("bad", ["shape", "dense", "repeat"])
def test_step_experts_malformed_rejected(prod_ckpt, tmp_path, bad):
    out = _forced_kernel(prod_ckpt, tmp_path / f"m_{bad}.npz", "bf16", [], experts=True)
    data = dict(np.load(out))
    se = data["step_experts"].copy()
    ref = H.Reference(prod_ckpt, "cpu", "bf16")
    moe = [i for i in range(ref.model.cfg.layers) if i not in ref.model.cfg.dense_layers]
    if bad == "shape":
        se = se[:-1]
    elif bad == "dense":
        se[0, sorted(ref.model.cfg.dense_layers)[0], 0] = 0
    else:
        se[0, moe[0], 1] = se[0, moe[0], 0]
    data["step_experts"] = se
    np.savez(out, **data)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and any("step_experts" in p for p in r["problems"])


# ---------------------------------------------------------------- all-declared policy (README 4c, ADOPTED rec-tan.4)
def _two_flip_kernel(prod_ckpt, tmp_path, router_ties, **kw):
    a, _ = router_ties
    ref = H.Reference(prod_ckpt, "cpu", "bf16")
    prompt = H.case_matrix(H.vocab_of(prod_ckpt))["short4"]["prompt"]
    first = np.load(_forced_kernel(prod_ckpt, tmp_path / "a2.npz", "bf16", [_swap(a)]))
    _, gaps = ref.run(prompt, first["tokens"].tolist(), [_swap(a)])
    b = min((g for g in gaps if g["pos"] > a["pos"] and g["layer"] == a["layer"]), key=lambda g: g["gap"])
    out = _forced_kernel(prod_ckpt, tmp_path / f"two_{len(kw)}.npz", "bf16", [_swap(a), _swap(b)],
                         experts=True, **{k: v([_swap(a)]) for k, v in kw.items()})
    return out, a, b


def test_policy_default_is_adopted_all_declared():
    assert H.ROUTER_EXPERTS_POLICY == "all-declared"  # rec-tan.4, adopted 2026-11-19


def test_all_declared_passes_two_genuine_near_ties(prod_ckpt, tmp_path, router_ties, monkeypatch):
    out, a, b = _two_flip_kernel(prod_ckpt, tmp_path, router_ties)
    monkeypatch.setattr(H, "ROUTER_EXPERTS_POLICY", "all-declared")
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", 1.0)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert r["pass"], r
    assert r["router_near_tie"]["policy"] == "all-declared" and r["router_near_tie"]["n"] == 2
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", min(a["gap"], b["gap"]) * 0.99)  # one deviation over tol
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and "exceeds the rule" in r["router_note"]


def test_all_declared_misreported_routing_fails(prod_ckpt, tmp_path, router_ties, monkeypatch):
    out, _, _ = _two_flip_kernel(prod_ckpt, tmp_path, router_ties, report_swaps=lambda x: x)
    monkeypatch.setattr(H, "ROUTER_EXPERTS_POLICY", "all-declared")
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", 1.0)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and r["router_near_tie"] is None


def test_all_declared_fp32_tier_rejects_any_deviation(prod_ckpt, tmp_path, router_ties, monkeypatch):
    out = _forced_kernel(prod_ckpt, tmp_path / "fp32_all.npz", "fp32", [_swap(router_ties[0])], experts=True)
    monkeypatch.setattr(H, "ROUTER_EXPERTS_POLICY", "all-declared")
    monkeypatch.setattr(H, "ROUTER_TIE_TOL", 1.0)
    r = H.check(prod_ckpt, out, "cpu", verbose=False)
    assert not r["pass"] and "fp32-KV tier" in r["router_note"]


def test_all_declared_exact_kernel_passes(prod_ckpt, tmp_path, monkeypatch):
    monkeypatch.setattr(H, "ROUTER_EXPERTS_POLICY", "all-declared")
    for kv in ("bf16", "fp32"):
        r = H.check(prod_ckpt, _forced_kernel(prod_ckpt, tmp_path / f"ex_{kv}.npz", kv, [], experts=True),
                    "cpu", verbose=False)
        assert r["pass"] and r["router_near_tie"] is None and r["n_routing_deviations"] == 0
