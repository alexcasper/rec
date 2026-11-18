"""Pin the tiny-megakernel references to the production model (CPU, no GPU).

The HIP megakernel (engine/native/megakernel_hip) is judged by
tiny_mk_np_reference.py on GPU hosts. These tests keep that numpy reference
honest: it must reproduce rec_mlx.model.Model greedy decode on the exact bank
layout the kernel reads, must reject a wrong trajectory, and the synthetic
bank must be non-degenerate (rec-hjr: an all-positive bank made one token
repeat and hid missing SiLU / router-bias / q4-unpack bugs).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

MK_DIR = Path(__file__).resolve().parents[1] / "native" / "megakernel_hip"
sys.path.insert(0, str(MK_DIR))

mx = pytest.importorskip("mlx.core")
driver = pytest.importorskip("tiny_mk_driver")
np_ref = pytest.importorskip("tiny_mk_np_reference")
mlx_ref = pytest.importorskip("tiny_mk_mlx_reference")

PROMPT = [1, 100, 200, 300]
N_OUT = 10


def _mlx_free_run(bank, off, n_out=N_OUT):
    model = mlx_ref.build_model(*mlx_ref.unpack_bank(bank, off))
    seq, tokens = list(PROMPT), []
    for _ in range(n_out):
        tokens.append(int(mx.argmax(model(mx.array([seq]))[0, -1]).item()))
        seq.append(tokens[-1])
    last = np.array(model(mx.array([seq]))[0, -1].astype(mx.float32))
    return tokens, last


def _record(bank, off, store, tokens, last):
    return {"bank": bank, "offsets": off, "prompt": np.array(PROMPT, np.int32),
            "n_out": np.int32(len(tokens)), "tokens": np.array(tokens, np.int32),
            "last_logits": last.astype(np.float32), **store}


class _Rec(dict):
    @property
    def files(self):
        return list(self.keys())


@pytest.fixture(scope="module")
def signed_bank():
    return driver.make_bank(seed=2, data="signed")  # seed 1 + PROMPT hits a repeat attractor


def test_bank_unpack_matches_dequantized_store(signed_bank):
    bank, off, store = signed_bank
    _, q4 = mlx_ref.unpack_bank(bank, off)
    for name, (w, s, b) in q4.items():
        deq = mx.dequantize(mx.array(w), mx.array(s), mx.array(b), group_size=64, bits=4)
        np.testing.assert_allclose(np.array(deq), store["w_" + name], rtol=0, atol=1e-6, err_msg=name)


def test_numpy_reference_matches_production_model(signed_bank):
    bank, off, store = signed_bank
    tokens, last = _mlx_free_run(bank, off)
    assert len(set(tokens)) >= N_OUT // 2, f"degenerate synthetic bank: {tokens}"
    assert np_ref.check(_Rec(_record(bank, off, store, tokens, last)), verbose=False)


def test_numpy_reference_rejects_wrong_trajectory(signed_bank):
    bank, off, store = signed_bank
    tokens, last = _mlx_free_run(bank, off)
    wrong = tokens[1:] + tokens[:1]
    assert not np_ref.check(_Rec(_record(bank, off, store, wrong, last)), verbose=False)
    assert not np_ref.check(_Rec(_record(bank, off, store, tokens, last + 1.0)), verbose=False)


def test_legacy_bank_is_degenerate():
    """Documents why 'signed' is the default: legacy data cannot discriminate."""
    bank, off, _ = driver.make_bank(seed=0, data="legacy")
    tokens, _ = _mlx_free_run(bank, off, n_out=6)
    assert len(set(tokens[1:])) == 1
