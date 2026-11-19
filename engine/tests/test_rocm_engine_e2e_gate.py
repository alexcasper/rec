"""rec-ft9: the rocm_engine_e2e gates (pure logic, CPU-only)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rocm_engine_e2e as E  # noqa: E402

V, T = 16, 6


def _ref(seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(T, V)).astype(np.float32)
    logits[np.arange(T), np.arange(T)] += 10.0  # argmax at step i is token i, wide margin
    return logits, list(range(T))


def test_clean_run_passes():
    ref, toks = _ref()
    gate = E.evaluate(ref, ref + 1e-6, toks, ref, np.zeros(T), toks)
    assert gate["pass"] and not gate["fp32_flips"] and not gate["bf16_flips"]


def test_fp32_flip_at_near_tie_is_allowed():
    ref, toks = _ref()
    ref[2, 7] = ref[2, 2] + 1e-3  # reference prefers 7 by 1e-3 (< TIE_TOL); engine chose 2
    gate = E.evaluate(ref, ref, toks, ref, np.zeros(T), toks)
    assert gate["fp32_flips"] and not gate["fp32_unexplained_flips"] and gate["pass"]


def test_fp32_flip_with_real_margin_fails():
    ref, toks = _ref()
    ref[2, 7] = ref[2, 2] + 0.25  # the rec-ft9 symptom against a true fp32 reference
    gate = E.evaluate(ref, ref, toks, ref, np.zeros(T), toks)
    assert gate["fp32_unexplained_flips"][0]["step"] == 2 and not gate["pass"]


def test_fp32_device_drift_is_gated_even_without_flips():
    ref, toks = _ref()
    gpu = ref.copy()
    gpu[4] += 2e-4 * np.abs(ref).max()  # no argmax change, but above the fp32 tier
    gate = E.evaluate(ref, gpu, toks, ref, np.zeros(T), toks)
    assert not gate["fp32_flips"] and gate["fp32_device_drift_rel"] > E.FP32_DRIFT_RTOL and not gate["pass"]


def test_bf16_flip_explained_by_perturbation_or_tie():
    ref, toks = _ref()
    t16 = list(toks)
    t16[3] = 9
    ref16 = ref.copy()
    ref16[3, 9] = ref16[3, 3] - 0.3  # reference margin 0.3 against the bf16 choice
    pert = np.zeros(T)
    pert[3] = 0.5
    assert E.evaluate(ref, ref, toks, ref16, pert, t16)["pass"]
    pert[3] = 0.1
    gate = E.evaluate(ref, ref, toks, ref16, pert, t16)
    assert not gate["bf16_flips"][0]["explained"] and not gate["pass"]
