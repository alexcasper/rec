"""Backend detection and selection (linux-lift)."""

import unittest

from rec_mlx.backends import (
    SUPPORTED_BACKENDS,
    BackendError,
    detect_backend,
    physical_memory_bytes,
    require_backend,
)


class BackendTests(unittest.TestCase):
    def test_detect_returns_supported_backend(self):
        backend = detect_backend()
        self.assertIn(backend.name, SUPPORTED_BACKENDS)
        self.assertGreater(backend.physical_memory_bytes, 0)

    def test_physical_memory_is_portable(self):
        # Must not depend on the macOS sysctl: works on Linux via /proc/meminfo.
        self.assertGreater(physical_memory_bytes(), 2**30)

    def test_env_override(self, monkeypatch=None):
        import os

        original = os.environ.get("REC_BACKEND")
        os.environ["REC_BACKEND"] = "cpu"
        try:
            self.assertEqual(detect_backend().name, "cpu")
        finally:
            if original is None:
                del os.environ["REC_BACKEND"]
            else:
                os.environ["REC_BACKEND"] = original

    def test_env_override_rejects_unknown(self):
        import os

        original = os.environ.get("REC_BACKEND")
        os.environ["REC_BACKEND"] = "vulkan"
        try:
            with self.assertRaises(BackendError):
                detect_backend()
        finally:
            if original is None:
                del os.environ["REC_BACKEND"]
            else:
                os.environ["REC_BACKEND"] = original

    def test_require_backend_mismatch_raises(self):
        backend = detect_backend()
        other = "metal" if backend.name != "metal" else "cpu"
        with self.assertRaises(BackendError):
            require_backend(other)


if __name__ == "__main__":
    unittest.main()
