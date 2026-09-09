import numpy as np
import pytest

from nltools.data import BrainData
from nltools.mask import create_sphere


def _gpu_present():
    """True when PyTorch can reach a CUDA or MPS accelerator."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def _cuda_present():
    """True when PyTorch can reach a CUDA device specifically."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(
    not _gpu_present(), reason="no CUDA or MPS accelerator"
)
requires_cuda = pytest.mark.skipif(not _cuda_present(), reason="no CUDA device")


class TestBrainDataBootstrap:
    def test_bootstrap(self, sim_brain_data):
        """Test bootstrap with mean/std."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # Test basic bootstrap with mean and std (should work)
        # Note: n_samples must be >= 10 for new implementation
        n_samples = 50
        b = masked.bootstrap(stat="mean", n_samples=n_samples)
        # New API returns BrainData directly
        assert isinstance(b, BrainData)
        assert b.shape == (1, masked.shape[1])  # (1, n_voxels)
        b = masked.bootstrap(stat="std", n_samples=n_samples)
        assert isinstance(b, BrainData)
        assert b.shape == (1, masked.shape[1])  # (1, n_voxels)

        # Bootstrap with "predict" requires fitted model (pass X_test to get past that check)
        X_test = np.random.randn(5, 10)  # Dummy test features
        X_train = np.random.randn(len(masked), 10)
        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(
                stat="predict", X=X_train, n_samples=n_samples, X_test=X_test
            )

    def test_bootstrap_tail_pins(self, sim_brain_data):
        """Pin tail=1/tail=2 p-values to the shared bootstrap formula (C-8)."""
        from scipy.stats import norm

        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        two = masked.bootstrap(
            stat="mean", n_samples=50, save_boots=True, random_state=42, tail=2
        )
        one = masked.bootstrap(
            stat="mean", n_samples=50, save_boots=True, random_state=42, tail=1
        )

        z = two["Z"].data
        np.testing.assert_array_equal(z, one["Z"].data)
        np.testing.assert_allclose(two["p"].data, 2 * (1 - norm.cdf(np.abs(z))))
        np.testing.assert_allclose(one["p"].data, 1 - norm.cdf(z))

    def test_bootstrap_invalid_method_error(self, sim_brain_data):
        """Test error raised for unsupported method."""
        # New implementation validates stat names upfront
        with pytest.raises(
            ValueError,
            match="Unsupported stat.*Supported simple stats",
        ):
            sim_brain_data.bootstrap(stat="invalid_method_name", n_samples=10)

    # ==================== Phase 5: New Bootstrap Implementation ====================

    def test_bootstrap_new_stat_param(self, sim_brain_data):
        """Test new bootstrap with stat='mean' parameter."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # New API: stat parameter
        boot = masked.bootstrap(stat="mean", n_samples=100, random_state=42)

        # Should return BrainData with shape (1, n_voxels) for aggregated result
        assert isinstance(boot, BrainData)
        assert boot.shape == (
            1,
            masked.shape[1],
        )  # (1, n_voxels) - aggregated across samples

    def test_bootstrap_new_save_boots_param(self, sim_brain_data):
        """Test new bootstrap with save_boots parameter."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # New API: save_boots=True should return dict
        result = masked.bootstrap(
            stat="mean", n_samples=50, save_boots=True, random_state=42
        )

        # When save_boots=True, should return dict with samples
        assert isinstance(result, dict)
        assert "samples" in result
        assert result["samples"].shape[0] == 50  # n_samples

    def test_bootstrap_new_all_simple_stats(self, sim_brain_data):
        """Test all simple stats work with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        stats = ["mean", "median", "std", "sum", "min", "max"]
        for stat in stats:
            boot = masked.bootstrap(stat=stat, n_samples=50, random_state=42)
            assert isinstance(boot, BrainData)
            assert boot.shape == (1, masked.shape[1])  # (1, n_voxels) - aggregated

    def test_bootstrap_new_ridge_weights_requires_fit(self, sim_brain_data):
        """Test weights bootstrap requires fitted model."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(
                stat="weights", X=np.random.randn(len(masked), 3), n_samples=10
            )

    def test_bootstrap_new_ridge_weights_basic(self, sim_brain_data):
        """Test Ridge weights bootstrap with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # Create design matrix
        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))

        # Fit model
        masked.fit(X=dm, model="ridge", ridge_alpha=1.0)

        # Bootstrap weights
        boot = masked.bootstrap(
            stat="weights", X=dm.to_numpy(), n_samples=100, random_state=42
        )

        # Should return dict with mean, std, Z, p, ci_lower, ci_upper
        assert isinstance(boot, dict)
        assert "mean" in boot
        assert "std" in boot
        assert "Z" in boot
        assert "p" in boot
        assert "ci_lower" in boot
        assert "ci_upper" in boot

        # Mean should be BrainData with shape (n_features, n_voxels)
        assert isinstance(boot["mean"], BrainData)
        assert boot["mean"].shape == (5, masked.shape[1])  # n_features × n_voxels

    def test_bootstrap_new_ridge_predict_requires_fit(self, sim_brain_data):
        """Test predict bootstrap requires fitted model."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        X_test = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(
                stat="predict",
                X=np.random.randn(len(masked), 5),
                X_test=X_test,
                n_samples=10,
            )

    def test_bootstrap_new_ridge_predict_requires_x_test(self, sim_brain_data):
        """Test predict bootstrap requires X_test."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))

        masked.fit(X=dm, model="ridge", ridge_alpha=1.0)

        with pytest.raises(ValueError, match="X_test.*required"):
            masked.bootstrap(stat="predict", X=dm.to_numpy(), n_samples=10)

    def test_bootstrap_new_ridge_predict_basic(self, sim_brain_data):
        """Test Ridge predict bootstrap with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))
        X_test = np.random.randn(10, 5)

        masked.fit(X=dm, model="ridge", ridge_alpha=1.0)

        boot = masked.bootstrap(
            stat="predict",
            X=dm.to_numpy(),
            X_test=X_test,
            n_samples=100,
            random_state=42,
        )

        # Should return dict
        assert isinstance(boot, dict)
        assert "mean" in boot

        # Mean should be BrainData with shape (n_test_samples, n_voxels)
        assert isinstance(boot["mean"], BrainData)
        assert boot["mean"].shape == (10, masked.shape[1])  # n_test × n_voxels


class TestRidgeBootstrapContract:
    """`BrainData.bootstrap` for the two Ridge statistics."""

    @staticmethod
    def _fitted(masked, n_features=4, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((len(masked), n_features))
        masked.fit(model="ridge", X=X, ridge_alpha=1.0)
        return X

    @staticmethod
    def _fitted_banded(masked, seed=1):
        rng = np.random.default_rng(seed)
        spaces = {
            "a": rng.standard_normal((len(masked), 3)),
            "b": rng.standard_normal((len(masked), 2)),
        }
        masked.fit(
            model="ridge",
            X=spaces,
            ridge_alpha=[1.0, 10.0],
            ridge_cv=3,
            ridge_search_iterations=4,
            random_state=0,
        )
        return spaces

    @pytest.fixture()
    def masked(self, sim_brain_data):
        return sim_brain_data.apply_mask(create_sphere(radius=6, coordinates=[0, 0, 0]))

    def test_signature_names_and_defaults(self):
        import inspect

        parameters = inspect.signature(BrainData.bootstrap).parameters
        assert [name for name in parameters if name != "self"] == [
            "stat",
            "X",
            "X_test",
            "n_samples",
            "save_boots",
            "percentiles",
            "device",
            "memory_budget_gb",
            "tail",
            "n_jobs",
            "random_state",
            "progress_bar",
        ]
        assert parameters["X"].default is None
        assert parameters["X_test"].default is None
        assert parameters["memory_budget_gb"].default is None
        assert parameters["device"].default == "cpu"

    def test_max_gpu_memory_gb_is_gone(self, masked):
        X = self._fitted(masked)
        with pytest.raises(TypeError):
            masked.bootstrap(stat="weights", X=X, n_samples=10, max_gpu_memory_gb=1.0)

    def test_memory_budget_gb_is_accepted(self, masked):
        X = self._fitted(masked)
        boot = masked.bootstrap(
            stat="weights", X=X, n_samples=10, memory_budget_gb=1.0, random_state=0
        )
        assert boot["mean"].shape == (X.shape[1], masked.shape[1])

    @pytest.mark.skipif(not _gpu_present(), reason="no CUDA or MPS accelerator")
    def test_memory_budget_gb_reaches_the_gpu_batch_planner(self, masked, monkeypatch):
        """The device-neutral facade name is the budget the GPU planner sees."""
        from nltools.algorithms.inference import bootstrap as engine

        X = self._fitted(masked)
        seen = {}
        original = engine._auto_batch_size_ridge

        def record(*args, max_memory_gb=None, **kwargs):
            seen["budget"] = max_memory_gb
            return original(*args, max_memory_gb=max_memory_gb, **kwargs)

        monkeypatch.setattr(engine, "_auto_batch_size_ridge", record)
        masked.bootstrap(
            stat="weights",
            X=X,
            n_samples=10,
            device="gpu",
            memory_budget_gb=1.5,
            random_state=0,
        )

        assert seen["budget"] == 1.5

    def test_weights_requires_explicit_training_features(self, masked):
        self._fitted(masked)
        with pytest.raises(ValueError, match="requires the training features"):
            masked.bootstrap(stat="weights", n_samples=10)

    def test_predict_requires_explicit_training_features(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="requires the training features"):
            masked.bootstrap(stat="predict", X_test=X[:3], n_samples=10)

    def test_weights_rejects_x_test(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="takes no X_test"):
            masked.bootstrap(stat="weights", X=X, X_test=X[:3], n_samples=10)

    def test_basic_statistics_reject_features(self, masked):
        X = np.random.default_rng(0).standard_normal((len(masked), 3))
        with pytest.raises(ValueError, match="takes no features"):
            masked.bootstrap(stat="mean", X=X, n_samples=10)
        with pytest.raises(ValueError, match="takes no features"):
            masked.bootstrap(stat="mean", X_test=X, n_samples=10)

    def test_wrong_row_count_raises(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="rows"):
            masked.bootstrap(stat="weights", X=X[:-1], n_samples=10)

    def test_a_fitted_glm_is_rejected(self, masked):
        from nltools.data import DesignMatrix

        design = DesignMatrix({"Intercept": np.ones(len(masked))})
        masked.fit(model="glm", X=design)
        with pytest.raises(ValueError, match="only supports a fitted Ridge"):
            masked.bootstrap(stat="weights", X=design.to_numpy(), n_samples=10)

    def test_replicates_hold_the_selected_hyperparameters_fixed(
        self, masked, monkeypatch
    ):
        """Every replicate receives the fitted alpha and simplex weights verbatim."""
        from nltools.models import ridge as ridge_module

        spaces = self._fitted_banded(masked)
        selected_alpha = np.array(masked.model_.alpha_, copy=True)
        selected_weights = np.array(masked.model_.feature_space_weights_, copy=True)

        seen = []
        original = ridge_module._refit_fixed_hyperparameters

        def record(design, y, alpha, feature_space_weights=None, **kwargs):
            seen.append((alpha, feature_space_weights))
            return original(design, y, alpha, feature_space_weights, **kwargs)

        monkeypatch.setattr(ridge_module, "_refit_fixed_hyperparameters", record)
        masked.bootstrap(
            stat="weights", X=spaces, n_samples=10, random_state=0, n_jobs=1
        )

        assert len(seen) == 10
        for alpha, weights in seen:
            np.testing.assert_array_equal(alpha, selected_alpha)
            assert weights is not None, "banded replicate dropped the simplex weights"
            np.testing.assert_array_equal(weights, selected_weights)

    def test_weights_estimate_matches_an_unresampled_refit(self, masked):
        """One replicate drawn over every row in order equals the full-data refit."""
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_design,
            _refit_resample,
        )

        X = self._fitted(masked)
        expected = _refit_resample(
            _bootstrap_design([X], masked.data),
            np.arange(len(masked)),
            masked.model_.alpha_,
        )
        np.testing.assert_allclose(masked.ridge_weights.data, expected, atol=1e-8)

    def test_banded_bootstrap_accepts_a_reordered_mapping(self, masked):
        spaces = self._fitted_banded(masked)
        reordered = {"b": spaces["b"], "a": spaces["a"]}

        boot = masked.bootstrap(
            stat="weights", X=reordered, n_samples=20, random_state=0, n_jobs=1
        )

        assert boot["mean"].shape == (5, masked.shape[1])

    def test_banded_bootstrap_rejects_missing_and_extra_spaces(self, masked):
        spaces = self._fitted_banded(masked)
        with pytest.raises(ValueError, match="exactly the fitted feature spaces"):
            masked.bootstrap(stat="weights", X={"a": spaces["a"]}, n_samples=10)
        with pytest.raises(ValueError, match="exactly the fitted feature spaces"):
            masked.bootstrap(
                stat="weights",
                X={**spaces, "c": spaces["a"]},
                n_samples=10,
            )

    def test_banded_bootstrap_resamples_every_space_with_one_index_draw(self, masked):
        """A banded replicate equals the shared refit on commonly resampled rows."""
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_design,
            _refit_resample,
        )
        from nltools.algorithms.random import generate_bootstrap_indices

        spaces = self._fitted_banded(masked)
        boot = masked.bootstrap(
            stat="weights",
            X=spaces,
            n_samples=3,
            save_boots=True,
            random_state=11,
            n_jobs=1,
        )

        indices = generate_bootstrap_indices(len(masked), 3, random_state=11)
        expected = _refit_resample(
            _bootstrap_design([spaces["a"], spaces["b"]], masked.data),
            indices[0],
            masked.model_.alpha_,
            masked.model_.feature_space_weights_,
        )
        np.testing.assert_allclose(boot["samples"][0], expected, atol=1e-8)

    def test_predict_uses_the_test_row_axis(self, masked):
        X = self._fitted(masked)
        X_test = np.random.default_rng(5).standard_normal((7, X.shape[1]))

        boot = masked.bootstrap(
            stat="predict", X=X, X_test=X_test, n_samples=20, random_state=0
        )

        assert boot["mean"].shape == (7, masked.shape[1])

    def test_unknown_device_raises(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
            masked.bootstrap(stat="weights", X=X, n_samples=10, device="auto")


class TestRidgeBootstrapOnGpu:
    """Explicit `device='gpu'` runs the refits on the accelerator, or raises."""

    @pytest.fixture()
    def masked(self, sim_brain_data):
        return sim_brain_data.apply_mask(create_sphere(radius=6, coordinates=[0, 0, 0]))

    @staticmethod
    def _fitted(masked, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((len(masked), 4))
        masked.fit(model="ridge", X=X, ridge_alpha=1.0)
        return X

    @pytest.mark.skipif(_gpu_present(), reason="an accelerator is available")
    def test_explicit_gpu_raises_without_an_accelerator(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="no CUDA or MPS device"):
            masked.bootstrap(stat="weights", X=X, n_samples=10, device="gpu")

    @requires_gpu
    def test_ordinary_gpu_matches_cpu(self, masked):
        X = self._fitted(masked)
        kwargs = {"stat": "weights", "X": X, "n_samples": 40, "random_state": 7}

        cpu = masked.bootstrap(device="cpu", n_jobs=1, **kwargs)
        gpu = masked.bootstrap(device="gpu", **kwargs)

        np.testing.assert_allclose(
            gpu["mean"].data, cpu["mean"].data, rtol=1e-2, atol=1e-3
        )

    @requires_gpu
    def test_banded_gpu_matches_cpu(self, masked):
        rng = np.random.default_rng(2)
        spaces = {
            "a": rng.standard_normal((len(masked), 3)),
            "b": rng.standard_normal((len(masked), 2)),
        }
        masked.fit(
            model="ridge",
            X=spaces,
            ridge_alpha=[1.0, 10.0],
            ridge_cv=3,
            ridge_search_iterations=4,
            random_state=0,
        )
        kwargs = {"stat": "weights", "X": spaces, "n_samples": 30, "random_state": 3}

        cpu = masked.bootstrap(device="cpu", n_jobs=1, **kwargs)
        gpu = masked.bootstrap(device="gpu", **kwargs)

        np.testing.assert_allclose(
            gpu["mean"].data, cpu["mean"].data, rtol=1e-2, atol=1e-2
        )

    @requires_gpu
    def test_gpu_bootstrap_emits_no_dtype_downcast_warning(self, masked):
        """The shared refit hands the GPU its own working dtype (B-I3)."""
        import warnings

        X = self._fitted(masked)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            masked.bootstrap(
                stat="weights", X=X, n_samples=10, device="gpu", random_state=0
            )

        downcasts = [
            str(record.message)
            for record in caught
            if "single precision" in str(record.message)
            or "cast to float32" in str(record.message)
        ]
        assert not downcasts, downcasts

    @requires_cuda
    def test_cuda_predict_matches_cpu(self, masked):
        X = self._fitted(masked)
        X_test = np.random.default_rng(4).standard_normal((6, X.shape[1]))
        kwargs = {
            "stat": "predict",
            "X": X,
            "X_test": X_test,
            "n_samples": 30,
            "random_state": 5,
        }

        cpu = masked.bootstrap(device="cpu", n_jobs=1, **kwargs)
        gpu = masked.bootstrap(device="gpu", **kwargs)

        np.testing.assert_allclose(
            gpu["mean"].data, cpu["mean"].data, rtol=1e-2, atol=1e-3
        )
