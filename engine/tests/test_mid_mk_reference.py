"""Pin the mid-8L fp64 numpy mirror (mid_mk_driver.forward_mid_np) to the
production model (rec-aip audit B10; CPU, no GPU).

The mid megakernel's driver self-check (`mid_mk_driver.py --verify`) judges the
kernel against forward_mid_np. Until this test, nothing pinned that hand-written
mirror to rec_mlx.model.Model, unlike tiny (test_tiny_mk_reference.py). A mirror
that shares the kernel's assumptions hides the kernel's bugs (rec-qib).

Checkpoint: the seeded mid preset from the rec-9na protocol
(`python -m rec_mlx.synthetic --preset mid`, seed 0, synthetic format); it is
byte-identical to dusk's `cpu-baseline-mid` bench checkpoint.

The mirror (like the kernel's bank packer) stores every float tensor as bf16, so
ground truth here is the MLX model in fp32 on the same checkpoint with every
float tensor rounded to bf16 (packed weights untouched). Then:
  * mirror vs model:          semantics, fp64 vs fp32 only (measured ~3.4e-6);
  * mirror on the raw F32 ckpt vs model on the bf16-rounded ckpt: only the
    mirror's own F32->bf16 conversion differs (round-half-up instead of
    round-to-nearest-even, ~1e-5 of values; measured 6.2e-4);
  * a semantic fault in the mirror (SiLU dropped) is caught by the same gate.
"""

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

MK_DIR = Path(__file__).resolve().parents[1] / "native" / "megakernel_hip"
sys.path.insert(0, str(MK_DIR))

mx = pytest.importorskip("mlx.core")
D = pytest.importorskip("mid_mk_driver")

PROMPT = [1, 4, 777, 8888, 5]  # bos, <|user|>, two plain tokens, <|assistant|>
N_OUT = 8
SEMANTIC_RTOL = 2e-5    # mirror (fp64) vs model (fp32) on identical bf16 values: measured 3.4e-6
CONVERSION_RTOL = 1e-3  # mirror's own F32->bf16 rounding on the raw ckpt: measured 6.2e-4
CPU_BASELINE_MID_SHA = "20214438574d"  # dusk ~/rec-ve/ckpt/cpu-baseline-mid model-00001


def _bf16_rounded_copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    shard = dst / "model-00001.safetensors"
    w = mx.load(str(shard))
    w = {k: (v.astype(mx.bfloat16).astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v)
         for k, v in w.items()}
    mx.eval(w)  # mx.load is lazy: materialize before overwriting the same file
    mx.save_safetensors(str(shard), w)
    return dst


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    from rec_mlx.synthetic import build_checkpoint

    root = tmp_path_factory.mktemp("mid")
    seeded = build_checkpoint(root / "seeded", "mid")
    return seeded, _bf16_rounded_copy(seeded, root / "bf16r")


@pytest.fixture(scope="module")
def truth(ckpts):
    """MLX greedy free run on the bf16-rounded ckpt; full-sequence fp32 logits."""
    from rec_mlx.quantized import QATModel

    native = QATModel(ckpts[1]).native
    seq = list(PROMPT)
    for _ in range(N_OUT):
        seq.append(int(np.array(native(mx.array([seq])).astype(mx.float32))[0, -1].argmax()))
    logits = np.array(native(mx.array([seq])).astype(mx.float32))[0].astype(np.float64)
    return seq, logits


def _rel_rows(a, b):
    return np.abs(a.astype(np.float64) - b).max(-1) / np.abs(b).max(-1)


def test_seeded_mid_ckpt_is_the_bench_checkpoint(ckpts):
    sha = hashlib.sha256((ckpts[0] / "model-00001.safetensors").read_bytes()).hexdigest()
    assert sha.startswith(CPU_BASELINE_MID_SHA), sha


def test_trajectory_is_non_degenerate(truth):
    seq, _ = truth
    assert len(set(seq[len(PROMPT):])) >= 3, seq


def test_mirror_matches_production_model(ckpts, truth):
    seq, ref = truth
    mir = D.forward_mid_np(D.load_safetensors_raw(ckpts[1]), seq)
    rel = _rel_rows(mir, ref)
    assert (mir.argmax(-1) == ref.argmax(-1)).all()
    assert rel.max() <= SEMANTIC_RTOL, rel


def test_mirror_on_seeded_f32_ckpt_differs_only_by_bf16_conversion(ckpts, truth):
    seq, ref = truth
    mir = D.forward_mid_np(D.load_safetensors_raw(ckpts[0]), seq)
    rel = _rel_rows(mir, ref)
    assert (mir.argmax(-1) == ref.argmax(-1)).all()
    assert rel.max() <= CONVERSION_RTOL, rel


def test_gate_catches_a_semantic_fault_in_the_mirror(ckpts, truth, monkeypatch):
    seq, ref = truth
    monkeypatch.setattr(D, "silu_np", lambda x: x)  # the rec-qib class of bug
    rel = _rel_rows(D.forward_mid_np(D.load_safetensors_raw(ckpts[1]), seq), ref)
    assert rel.max() > 100 * SEMANTIC_RTOL, rel
