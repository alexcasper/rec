"""rec-aip / rec-9na: the broadcast dequant is bit-identical to the original
repeat_interleave formulation. The torch layer is also the conformance
reference (engine/validation/megakernel_mid), so its numerics must not move."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec_mlx.torch_rocm import GROUP_SIZE, dequant  # noqa: E402


def dequant_reference(packed, scales, biases, bits, dtype):
    """The pre-rec-aip implementation, verbatim."""
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=packed.device, dtype=torch.int32) * bits
    mask = (1 << bits) - 1
    q = (packed.unsqueeze(-1) >> shifts) & mask
    q = q.reshape(*packed.shape[:-1], -1).to(torch.float32)
    s = scales.float().repeat_interleave(GROUP_SIZE, dim=-1)
    b = biases.float().repeat_interleave(GROUP_SIZE, dim=-1)
    return (q * s + b).to(dtype)


@pytest.mark.parametrize("bits", [2, 4, 8])  # 2 = shift-unpack fallback path
@pytest.mark.parametrize("lead", [(96,), (3, 40)])  # dense [out, ..] and stacked experts [E, out, ..]
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sdtype", [torch.bfloat16, torch.float32])  # production BF16 / synthetic F32 scales
def test_dequant_bit_identical(bits, lead, dtype, sdtype):
    g = torch.Generator().manual_seed(bits * 131 + len(lead))
    K = 4 * GROUP_SIZE
    words = K * bits // 32
    packed = torch.randint(-2**31, 2**31 - 1, (*lead, words), generator=g, dtype=torch.int64).to(torch.int32)
    scales = (torch.randn(*lead, K // GROUP_SIZE, generator=g) * 0.05).to(sdtype)
    biases = (torch.randn(*lead, K // GROUP_SIZE, generator=g) * 0.3).to(sdtype)
    new = dequant(packed, scales, biases, bits, dtype)
    old = dequant_reference(packed, scales, biases, bits, dtype)
    assert new.shape == old.shape == (*lead, K) and new.dtype == dtype
    assert torch.equal(new.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
                       old.view(torch.int16 if dtype == torch.bfloat16 else torch.int32))


def test_dequant_rejects_mismatched_groups():
    packed = torch.zeros(4, 8, dtype=torch.int32)  # K = 64 at 4 bits -> 1 group
    with pytest.raises(ValueError):
        dequant(packed, torch.ones(4, 2), torch.zeros(4, 2), 4, torch.float32)
