"""Backend detection and selection for the rec engine.

Three backends are recognised:

- ``metal``: Apple Silicon + MLX Metal, the original build target. Full
  resident-kernel decode, grammar-constrained sampling, prompt prefix cache.
- ``cpu``: MLX CPU backend (Linux/macOS). Runs the portable pure-MLX model
  (``rec_mlx.native_runtime``); no resident megakernel, no grammar-constrained
  decode, no prefix preparation.
- ``rocm``: reserved for the torch-ROCm port (see docs/amd-port-assessment.md).
  Selectable explicitly; not yet implemented behind Engine.

The backend is auto-detected from the MLX runtime and can be forced with the
``REC_BACKEND`` environment variable.
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_BACKENDS = ("metal", "cpu", "rocm")

# The resident Metal megakernel and its KV blocks are sized for Macs with at
# least 48 GiB of unified memory.
METAL_MIN_MEMORY_BYTES = 48 * 1024**3


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class Backend:
    name: str
    system: str
    machine: str
    physical_memory_bytes: int

    @property
    def is_metal(self) -> bool:
        return self.name == "metal"


def physical_memory_bytes() -> int:
    system = platform.system()
    if system == "Darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    if system == "Linux":
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
        raise BackendError("Could not read MemTotal from /proc/meminfo")
    raise BackendError(f"Unsupported platform: {system}")


def detect_backend() -> Backend:
    system, machine = platform.system(), platform.machine()
    memory = physical_memory_bytes()
    override = os.environ.get("REC_BACKEND")
    if override is not None:
        if override not in SUPPORTED_BACKENDS:
            raise BackendError(
                f"REC_BACKEND must be one of {', '.join(SUPPORTED_BACKENDS)}, got {override!r}"
            )
        return Backend(override, system, machine, memory)
    if system == "Darwin" and machine == "arm64":
        return Backend("metal", system, machine, memory)
    import mlx.core as mx

    if mx.default_device().device_type == mx.DeviceType.gpu:
        # Linux MLX GPU builds are CUDA today; a ROCm backend would surface here.
        return Backend("rocm", system, machine, memory)
    return Backend("cpu", system, machine, memory)


def require_backend(name: str) -> Backend:
    backend = detect_backend()
    if backend.name != name:
        raise BackendError(f"This path requires the {name} backend; detected {backend.name}")
    return backend
