"""The run-or-raise device policy: explicit GPU requests never silently degrade.

Policy (docs/development/index.md): an explicit ``device='gpu'`` either runs on
the GPU or raises; ``'auto'`` is the one documented graceful-fallback path.
"""

import pytest


def test_explicit_gpu_requires_an_accelerator(monkeypatch):
    """An installed CPU-only Torch must not satisfy an explicit GPU request."""
    from nltools.algorithms import backends as backends_mod

    def _init_torch_cpu(self):
        self.name = "torch-cpu"
        self.device = "cpu"
        self.xp = object()
        self._torch_device = "cpu"

    monkeypatch.setattr(backends_mod.Backend, "_init_torch", _init_torch_cpu)

    with pytest.raises(RuntimeError, match="no GPU accelerator"):
        backends_mod.resolve_backend("gpu")


def test_auto_may_select_a_cpu_backend(monkeypatch):
    """Automatic selection remains the graceful CPU fallback path."""
    from nltools.algorithms import backends as backends_mod

    def _init_torch_cpu(self):
        self.name = "torch-cpu"
        self.device = "cpu"
        self.xp = object()
        self._torch_device = "cpu"

    monkeypatch.setattr(backends_mod.Backend, "_init_torch", _init_torch_cpu)

    assert backends_mod.resolve_backend("auto").device == "cpu"
