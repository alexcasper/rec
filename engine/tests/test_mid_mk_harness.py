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
