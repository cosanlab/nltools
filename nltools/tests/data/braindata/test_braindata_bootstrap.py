import inspect

import numpy as np
import pytest

from nltools.data import BrainData
from nltools.data.results import BootstrapResult
from nltools.mask import create_sphere

SUMMARY_FIELDS = ("estimate", "standard_error", "ci_lower", "ci_upper")

# Only `bootstrap` emits these two quality advisories, and the bootstrap tests
# below run deliberately small resamples that trip them by design. Filtering
# them here keeps the suite's warning count from tracking the number of
# bootstrap call sites; `pytest.warns` still sees them where one is asserted.
pytestmark = [
    pytest.mark.filterwarnings("ignore:n_samples=:UserWarning"),
    pytest.mark.filterwarnings("ignore:Only .* samples available:UserWarning"),
]


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


class TestBrainDataBootstrapSurface:
    """The public signature and the record it returns."""

    def test_signature_names_and_defaults(self):
        parameters = inspect.signature(BrainData.bootstrap).parameters

        assert [name for name in parameters if name != "self"] == [
            "statistic",
            "X",
            "X_test",
            "n_samples",
            "confidence_level",
            "device",
            "memory_budget_gb",
            "return_samples",
            "n_jobs",
            "random_state",
            "progress_bar",
        ]
        assert parameters["X"].default is None
        assert parameters["X_test"].default is None
        assert parameters["n_samples"].default == 5000
        assert parameters["confidence_level"].default == 0.95
        assert parameters["device"].default == "cpu"
        assert parameters["memory_budget_gb"].default is None
        assert parameters["return_samples"].default is False
        assert parameters["n_jobs"].default == -1
        assert parameters["random_state"].default is None
        assert parameters["progress_bar"].default is False

    def test_everything_after_the_statistic_is_keyword_only(self):
        parameters = inspect.signature(BrainData.bootstrap).parameters

        assert parameters["statistic"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name, parameter in parameters.items():
            if name in ("self", "statistic"):
                continue
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize(
        "removed",
        ["stat", "save_boots", "percentiles", "tail", "max_gpu_memory_gb"],
    )
    def test_removed_keywords_raise_type_error(self, minimal_brain_data, removed):
        with pytest.raises(TypeError):
            minimal_brain_data.bootstrap("mean", n_samples=10, **{removed: 1})

    def test_unknown_statistic_raises(self, minimal_brain_data):
        with pytest.raises(
            ValueError, match="Unsupported statistic.*Supported basic statistics"
        ):
            minimal_brain_data.bootstrap("invalid_name", n_samples=10)


class TestBrainDataBootstrapBasicStatistics:
    """The six reductions, their shapes, and the arguments they refuse."""

    @pytest.fixture()
    def masked(self, sim_brain_data):
        return sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

    @pytest.mark.parametrize(
        "statistic", ["mean", "median", "std", "sum", "min", "max"]
    )
    def test_summaries_use_the_single_map_reduction_shape(self, masked, statistic):
        result = masked.bootstrap(statistic, n_samples=50, random_state=42)

        assert isinstance(result, BootstrapResult)
        for field in SUMMARY_FIELDS:
            payload = getattr(result, field)
            assert isinstance(payload, BrainData)
            assert payload.data.shape == (masked.shape[1],)
        assert result.samples is None

    def test_summary_shape_matches_the_reduction_convention(self, masked):
        result = masked.bootstrap("mean", n_samples=50, random_state=42)

        assert result.estimate.data.shape == masked.mean().data.shape

    def test_estimate_is_the_unresampled_reduction(self, masked):
        """Every result field is float64, so the reference reduction is too."""
        result = masked.bootstrap("mean", n_samples=50, random_state=42)

        assert result.estimate.data.dtype == np.float64
        np.testing.assert_allclose(
            result.estimate.data,
            np.mean(masked.data.astype(np.float64), axis=0),
            rtol=1e-12,
        )

    def test_std_matches_braindata_std(self, masked):
        result = masked.bootstrap("std", n_samples=50, random_state=42)

        np.testing.assert_allclose(result.estimate.data, masked.std().data, rtol=1e-6)

    def test_return_samples_shape_and_interval_agreement(self, masked):
        result = masked.bootstrap(
            "mean", n_samples=50, return_samples=True, random_state=42
        )

        assert result.samples.shape == (50, masked.shape[1])
        np.testing.assert_allclose(
            result.ci_lower.data,
            np.percentile(result.samples, 2.5, axis=0),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result.standard_error.data,
            np.std(result.samples, axis=0, ddof=1),
            rtol=1e-10,
        )

    def test_basic_statistics_reject_features(self, masked):
        X = np.random.default_rng(0).standard_normal((len(masked), 3))

        with pytest.raises(ValueError, match="takes no features"):
            masked.bootstrap("mean", X=X, n_samples=10)
        with pytest.raises(ValueError, match="takes no features"):
            masked.bootstrap("mean", X_test=X, n_samples=10)

    def test_basic_statistics_reject_an_explicit_gpu(self, masked):
        with pytest.raises(ValueError, match="runs on the CPU"):
            masked.bootstrap("mean", n_samples=10, device="gpu")

    def test_basic_statistics_reject_an_unknown_device(self, masked):
        with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
            masked.bootstrap("mean", n_samples=10, device="banana")

    def test_out_of_range_arguments_still_raise_through_the_facade(self, masked):
        """The facade delegates range checking, so the engine's message surfaces."""
        with pytest.raises(ValueError, match="confidence_level"):
            masked.bootstrap("mean", n_samples=10, confidence_level=1.5)

    def test_reproducibility(self, masked):
        kwargs = {"n_samples": 40, "random_state": 7, "n_jobs": 1}

        first = masked.bootstrap("mean", **kwargs)
        second = masked.bootstrap("mean", **kwargs)

        for field in SUMMARY_FIELDS:
            np.testing.assert_array_equal(
                getattr(first, field).data, getattr(second, field).data
            )


class TestBrainDataBootstrapOwnership:
    """Each payload is independently owned and carries the source's spatial state."""

    @pytest.fixture()
    def masked(self, sim_brain_data):
        return sim_brain_data.apply_mask(create_sphere(radius=6, coordinates=[0, 0, 0]))

    def test_mutating_a_payload_touches_nothing_else(self, masked):
        result = masked.bootstrap("mean", n_samples=20, random_state=0)
        source = masked.data.copy()

        result.estimate.data[0] = 12345.0

        np.testing.assert_array_equal(masked.data, source)
        assert result.standard_error.data[0] != 12345.0
        assert result.ci_lower.data[0] != 12345.0

    def test_payloads_preserve_the_source_mask(self, masked):
        result = masked.bootstrap("mean", n_samples=20, random_state=0)

        for field in SUMMARY_FIELDS:
            payload = getattr(result, field)
            np.testing.assert_array_equal(
                payload.mask.get_fdata(), masked.mask.get_fdata()
            )

    def test_payloads_clear_row_metadata_and_fitted_state(self, masked):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((len(masked), 3))
        masked.fit(model="ridge", X=X, ridge_alpha=1.0)

        result = masked.bootstrap("weights", X=X, n_samples=20, random_state=0)

        for field in SUMMARY_FIELDS:
            payload = getattr(result, field)
            assert payload.X.shape == (0, 0)
            assert payload.Y.shape == (0, 0)
            assert not hasattr(payload, "model_")
            assert not hasattr(payload, "ridge_weights")


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

    def test_weights_summaries_keep_the_feature_axis(self, masked):
        X = self._fitted(masked)

        result = masked.bootstrap("weights", X=X, n_samples=20, random_state=0)

        for field in SUMMARY_FIELDS:
            assert getattr(result, field).shape == (X.shape[1], masked.shape[1])

    def test_weights_estimate_is_the_fitted_coefficients(self, masked):
        X = self._fitted(masked)

        result = masked.bootstrap("weights", X=X, n_samples=20, random_state=0)

        np.testing.assert_allclose(result.estimate.data, masked.model_.coef_)
        np.testing.assert_allclose(
            result.estimate.data, masked.ridge_weights.data, atol=1e-10
        )

    def test_weights_retained_samples_shape(self, masked):
        X = self._fitted(masked)

        result = masked.bootstrap(
            "weights", X=X, n_samples=20, return_samples=True, random_state=0
        )

        assert result.samples.shape == (20, X.shape[1], masked.shape[1])

    def test_predict_uses_the_test_row_axis(self, masked):
        X = self._fitted(masked)
        X_test = np.random.default_rng(5).standard_normal((7, X.shape[1]))

        result = masked.bootstrap(
            "predict", X=X, X_test=X_test, n_samples=20, random_state=0
        )

        for field in SUMMARY_FIELDS:
            assert getattr(result, field).shape == (7, masked.shape[1])
        assert result.samples is None

    def test_predict_estimate_is_the_full_data_model_at_the_test_rows(self, masked):
        X = self._fitted(masked)
        X_test = np.random.default_rng(5).standard_normal((3, X.shape[1]))

        result = masked.bootstrap(
            "predict", X=X, X_test=X_test, n_samples=20, random_state=0
        )

        np.testing.assert_allclose(
            result.estimate.data, X_test @ masked.model_.coef_, rtol=1e-10
        )

    def test_a_singleton_test_row_axis_is_not_squeezed(self, masked):
        X = self._fitted(masked)
        X_test = np.random.default_rng(5).standard_normal((1, X.shape[1]))

        result = masked.bootstrap(
            "predict", X=X, X_test=X_test, n_samples=20, random_state=0
        )

        assert result.estimate.data.shape == (1, masked.shape[1])

    def test_memory_budget_gb_is_accepted(self, masked):
        X = self._fitted(masked)

        result = masked.bootstrap(
            "weights", X=X, n_samples=10, memory_budget_gb=1.0, random_state=0
        )

        assert result.estimate.shape == (X.shape[1], masked.shape[1])

    def test_a_tiny_memory_budget_raises_before_resampling(self, masked, monkeypatch):
        from nltools.algorithms.inference import bootstrap as engine

        X = self._fitted(masked)

        def _never(*args, **kwargs):
            raise AssertionError("resampling started despite an over-budget preflight")

        monkeypatch.setattr(engine, "generate_bootstrap_indices", _never)

        with pytest.raises(ValueError, match="memory_budget_gb="):
            masked.bootstrap(
                "weights",
                X=X,
                n_samples=5000,
                return_samples=True,
                memory_budget_gb=1e-6,
                random_state=0,
            )

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
            "weights",
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
            masked.bootstrap("weights", n_samples=10)

    def test_predict_requires_explicit_training_features(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="requires the training features"):
            masked.bootstrap("predict", X_test=X[:3], n_samples=10)

    def test_predict_requires_x_test(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="X_test.*required"):
            masked.bootstrap("predict", X=X, n_samples=10)

    def test_weights_rejects_x_test(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="takes no X_test"):
            masked.bootstrap("weights", X=X, X_test=X[:3], n_samples=10)

    def test_ridge_statistics_require_a_fit(self, masked):
        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap("weights", X=np.random.randn(len(masked), 3), n_samples=10)
        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(
                "predict",
                X=np.random.randn(len(masked), 3),
                X_test=np.random.randn(5, 3),
                n_samples=10,
            )

    def test_wrong_row_count_raises(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="rows"):
            masked.bootstrap("weights", X=X[:-1], n_samples=10)

    def test_a_fitted_glm_is_rejected(self, masked):
        from nltools.data import DesignMatrix

        design = DesignMatrix({"Intercept": np.ones(len(masked))})
        masked.fit(model="glm", X=design)
        with pytest.raises(ValueError, match="only supports a fitted Ridge"):
            masked.bootstrap("weights", X=design.to_numpy(), n_samples=10)

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
        masked.bootstrap("weights", X=spaces, n_samples=10, random_state=0, n_jobs=1)

        assert len(seen) == 10
        for alpha, weights in seen:
            np.testing.assert_array_equal(alpha, selected_alpha)
            assert weights is not None, "banded replicate dropped the simplex weights"
            np.testing.assert_array_equal(weights, selected_weights)

    def test_banded_bootstrap_accepts_a_reordered_mapping(self, masked):
        spaces = self._fitted_banded(masked)
        reordered = {"b": spaces["b"], "a": spaces["a"]}

        result = masked.bootstrap(
            "weights", X=reordered, n_samples=20, random_state=0, n_jobs=1
        )

        assert result.estimate.shape == (5, masked.shape[1])

    def test_banded_bootstrap_rejects_missing_and_extra_spaces(self, masked):
        spaces = self._fitted_banded(masked)
        with pytest.raises(ValueError, match="exactly the fitted feature spaces"):
            masked.bootstrap("weights", X={"a": spaces["a"]}, n_samples=10)
        with pytest.raises(ValueError, match="exactly the fitted feature spaces"):
            masked.bootstrap("weights", X={**spaces, "c": spaces["a"]}, n_samples=10)

    def test_banded_bootstrap_resamples_every_space_with_one_index_draw(self, masked):
        """A banded replicate equals the shared refit on commonly resampled rows."""
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_design,
            _refit_resample,
        )
        from nltools.algorithms.random import generate_bootstrap_indices

        spaces = self._fitted_banded(masked)
        result = masked.bootstrap(
            "weights",
            X=spaces,
            n_samples=3,
            return_samples=True,
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
        np.testing.assert_allclose(result.samples[0], expected, atol=1e-8)

    def test_unknown_device_raises(self, masked):
        X = self._fitted(masked)
        with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
            masked.bootstrap("weights", X=X, n_samples=10, device="auto")

    @pytest.mark.parametrize("n_jobs", [2, 4])
    def test_worker_count_does_not_change_the_result(self, masked, n_jobs):
        X = self._fitted(masked)
        kwargs = {"X": X, "n_samples": 30, "random_state": 9}

        sequential = masked.bootstrap("weights", n_jobs=1, **kwargs)
        parallel = masked.bootstrap("weights", n_jobs=n_jobs, **kwargs)

        for field in SUMMARY_FIELDS:
            np.testing.assert_allclose(
                getattr(parallel, field).data,
                getattr(sequential, field).data,
                rtol=1e-12,
            )


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
            masked.bootstrap("weights", X=X, n_samples=10, device="gpu")

    @requires_gpu
    def test_ordinary_gpu_matches_cpu(self, masked):
        X = self._fitted(masked)
        kwargs = {"X": X, "n_samples": 40, "random_state": 7}

        cpu = masked.bootstrap("weights", device="cpu", n_jobs=1, **kwargs)
        gpu = masked.bootstrap("weights", device="gpu", **kwargs)

        np.testing.assert_allclose(
            gpu.standard_error.data, cpu.standard_error.data, rtol=1e-2, atol=1e-3
        )
        np.testing.assert_allclose(
            gpu.ci_lower.data, cpu.ci_lower.data, rtol=1e-2, atol=1e-3
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
        kwargs = {"X": spaces, "n_samples": 30, "random_state": 3}

        cpu = masked.bootstrap("weights", device="cpu", n_jobs=1, **kwargs)
        gpu = masked.bootstrap("weights", device="gpu", **kwargs)

        np.testing.assert_allclose(
            gpu.standard_error.data, cpu.standard_error.data, rtol=1e-2, atol=1e-2
        )

    @requires_gpu
    def test_forced_small_gpu_batches_do_not_change_the_result(
        self, masked, monkeypatch
    ):
        """Batching is a memory decision, so it must be numerically invisible.

        The batch size is forced through the planner rather than through a tiny
        `memory_budget_gb`: the same budget now also gates the output preflight,
        which would refuse a budget small enough to split this problem's batches.
        """
        from nltools.algorithms.inference import bootstrap as engine

        X = self._fitted(masked)
        kwargs = {"X": X, "n_samples": 60, "random_state": 12, "device": "gpu"}
        batch_counts = []

        one_batch = masked.bootstrap("weights", **kwargs)

        def one_replicate_per_batch(n_bootstrap, *args, **kwargs_inner):
            batch_counts.append(n_bootstrap)
            return 1, n_bootstrap

        monkeypatch.setattr(engine, "_auto_batch_size_ridge", one_replicate_per_batch)
        many_batches = masked.bootstrap("weights", **kwargs)

        assert batch_counts == [60], "the planner was not consulted"
        for field in SUMMARY_FIELDS:
            np.testing.assert_allclose(
                getattr(many_batches, field).data,
                getattr(one_batch, field).data,
                rtol=1e-10,
                atol=1e-12,
            )

    @requires_gpu
    def test_gpu_bootstrap_emits_no_dtype_downcast_warning(self, masked):
        """The shared refit hands the GPU its own working dtype (B-I3)."""
        import warnings

        X = self._fitted(masked)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            masked.bootstrap("weights", X=X, n_samples=10, device="gpu", random_state=0)

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
        kwargs = {"X": X, "X_test": X_test, "n_samples": 30, "random_state": 5}

        cpu = masked.bootstrap("predict", device="cpu", n_jobs=1, **kwargs)
        gpu = masked.bootstrap("predict", device="gpu", **kwargs)

        np.testing.assert_allclose(
            gpu.standard_error.data, cpu.standard_error.data, rtol=1e-2, atol=1e-3
        )
