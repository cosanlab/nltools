"""The run-or-raise device policy: explicit GPU requests never silently degrade.

Policy (docs/development/index.md): an explicit ``device='gpu'`` /
``parallel='gpu'`` either runs on the GPU or raises; ``'auto'`` is the one
documented graceful-fallback path. These tests pin the sites that used to
fall back silently.
"""

import numpy as np
import pytest


def _subjects(n=3, voxels=10, samples=24, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((voxels, samples)) for _ in range(n)]


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


class TestSrmRunOrRaise:
    def test_srm_fit_gpu_raises(self):
        from nltools.algorithms.alignment import SRM

        srm = SRM(n_iter=1, features=2)
        with pytest.raises(NotImplementedError, match="gpu"):
            srm.fit(_subjects(), parallel="gpu")

    def test_detsrm_fit_gpu_raises(self):
        from nltools.algorithms.alignment import DetSRM

        srm = DetSRM(n_iter=1, features=2)
        with pytest.raises(NotImplementedError, match="gpu"):
            srm.fit(_subjects(), parallel="gpu")

    def test_srm_transform_gpu_raises(self):
        from nltools.algorithms.alignment import SRM

        srm = SRM(n_iter=1, features=2, rand_seed=0)
        data = _subjects()
        srm.fit(data, parallel=None)
        with pytest.raises(NotImplementedError, match="gpu"):
            srm.transform(data, parallel="gpu")

    def test_detsrm_transform_gpu_raises(self):
        from nltools.algorithms.alignment import DetSRM

        srm = DetSRM(n_iter=1, features=2, rand_seed=0)
        data = _subjects()
        srm.fit(data, parallel=None)
        with pytest.raises(NotImplementedError, match="gpu"):
            srm.transform(data, parallel="gpu")

    def test_dead_max_gpu_memory_gb_kwarg_removed(self):
        from nltools.algorithms.alignment import SRM, DetSRM

        with pytest.raises(TypeError, match="max_gpu_memory_gb"):
            SRM(n_iter=1, features=2).fit(_subjects(), max_gpu_memory_gb=2.0)
        with pytest.raises(TypeError, match="max_gpu_memory_gb"):
            DetSRM(n_iter=1, features=2).fit(_subjects(), max_gpu_memory_gb=2.0)

    def test_cpu_paths_still_work(self):
        from nltools.algorithms.alignment import SRM

        srm = SRM(n_iter=1, features=2, rand_seed=0)
        data = _subjects()
        srm.fit(data, parallel=None)
        shared = srm.transform(data, parallel=None)
        assert len(shared) == len(data)


class TestLocalAlignmentRunOrRaise:
    def test_unknown_parallel_value_rejected(self):
        from nltools.algorithms.alignment import LocalAlignment

        with pytest.raises(ValueError, match="parallel"):
            LocalAlignment(parallel="gup")

    def test_gpu_with_srm_method_raises(self):
        from nltools.algorithms.alignment import LocalAlignment

        with pytest.raises(NotImplementedError, match="gpu"):
            LocalAlignment(method="srm", parallel="gpu")

    def test_gpu_with_hyperalignment_method_raises(self):
        from nltools.algorithms.alignment import LocalAlignment

        with pytest.raises(NotImplementedError, match="gpu"):
            LocalAlignment(method="hyperalignment", parallel="gpu")

    def test_gpu_without_torch_raises(self, monkeypatch):
        """Explicit 'gpu' with no torch installed raises instead of logging."""
        import importlib.util

        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(parallel="gpu")

        if importlib.util.find_spec("torch") is None:
            with pytest.raises(ImportError):
                la._init_backend()
        else:
            # Simulate a missing torch by making Backend('torch') fail
            from nltools.algorithms import backends as backends_mod

            def _no_torch(self):
                raise ImportError("PyTorch not installed (simulated)")

            monkeypatch.setattr(backends_mod.Backend, "_init_torch", _no_torch)
            with pytest.raises(ImportError):
                la._init_backend()
