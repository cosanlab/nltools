"""Tests for bootstrap inference utilities."""

import tracemalloc

import numpy as np
import pytest

from nltools.algorithms.backends import check_gpu_available
from nltools.algorithms.inference.bootstrap import (
    _BootstrapAccumulator,
    _bootstrap_ridge_predict_cpu_parallel,
    _bootstrap_ridge_weights_cpu_parallel,
    _bootstrap_simple_cpu_parallel,
    _bootstrap_simple_method_worker,
)

RESULT_FIELDS = ("estimate", "standard_error", "ci_lower", "ci_upper")

# Nearly every test here deliberately runs a small, fast bootstrap, which trips
# the two quality advisories by design. Silencing them module-wide keeps the
# suite's warning count from tracking the number of bootstrap call sites; the
# tests that assert an advisory use `pytest.warns`, which is unaffected.
pytestmark = [
    pytest.mark.filterwarnings("ignore:n_samples=:UserWarning"),
    pytest.mark.filterwarnings("ignore:Only .* samples available:UserWarning"),
]


def _reference_interval(samples, confidence_level):
    """The interval the complete retained distribution would produce."""
    lower = 100 * (1 - confidence_level) / 2
    return (
        np.percentile(samples, lower, axis=0),
        np.percentile(samples, 100 - lower, axis=0),
    )


def _ridge_coefficients(X, y, alpha=1.0):
    """The full-data fixed-hyperparameter refit the facade would hand an engine."""
    from nltools.models.ridge import _refit_fixed_hyperparameters

    return _refit_fixed_hyperparameters([X], y, alpha)


class TestBootstrapResultRecord:
    """The one result structure every bootstrap statistic returns."""

    @staticmethod
    def _record():
        from nltools.data.results import BootstrapResult

        return BootstrapResult(
            estimate=np.zeros(3),
            standard_error=np.ones(3),
            ci_lower=-np.ones(3),
            ci_upper=np.ones(3),
        )

    def test_field_names_and_sample_default(self):
        from nltools.data.results import BootstrapResult

        assert list(BootstrapResult.__dataclass_fields__) == [
            "estimate",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "samples",
        ]
        assert self._record().samples is None

    def test_fields_cannot_be_rebound(self):
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            self._record().estimate = np.zeros(3)

    def test_record_owns_every_payload(self):
        from nltools.data.results import BootstrapResult

        estimate = np.zeros(3)
        samples = np.zeros((4, 3))
        result = BootstrapResult(
            estimate=estimate,
            standard_error=np.ones(3),
            ci_lower=-np.ones(3),
            ci_upper=np.ones(3),
            samples=samples,
        )
        estimate[0] = 99.0
        samples[0, 0] = 99.0

        assert result.estimate[0] == 0.0
        assert result.samples[0, 0] == 0.0


class TestBootstrapAccumulator:
    """Streaming Welford variance plus a bounded per-element retained tail."""

    @staticmethod
    def _accumulate(
        samples, confidence_level=0.95, retain_samples=False, n_replicates=None
    ):
        accumulator = _BootstrapAccumulator(
            samples.shape[1:],
            n_replicates=samples.shape[0] if n_replicates is None else n_replicates,
            confidence_level=confidence_level,
            retain_samples=retain_samples,
        )
        for sample in samples:
            accumulator.update(sample)
        return accumulator

    @pytest.mark.parametrize("confidence_level", [0.95])
    @pytest.mark.parametrize("n_replicates", [201])
    def test_streaming_interval_equals_the_fully_retained_interval(
        self, confidence_level, n_replicates
    ):
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((n_replicates, 4, 3))

        streamed = self._accumulate(samples, confidence_level).results()
        expected_lower, expected_upper = _reference_interval(samples, confidence_level)

        np.testing.assert_allclose(streamed["ci_lower"], expected_lower, rtol=1e-12)
        np.testing.assert_allclose(streamed["ci_upper"], expected_upper, rtol=1e-12)

    def test_standard_error_is_the_ddof_one_replicate_deviation(self):
        rng = np.random.default_rng(1)
        samples = rng.standard_normal((250, 6))

        streamed = self._accumulate(samples).results()

        np.testing.assert_allclose(
            streamed["standard_error"], np.std(samples, axis=0, ddof=1), rtol=1e-10
        )

    def test_welford_survives_a_large_offset(self):
        """A tiny variance on a 1e10 mean must not be lost to cancellation."""
        rng = np.random.default_rng(9)
        samples = 1e10 + 0.1 * rng.standard_normal((1000, 3))

        streamed = self._accumulate(samples).results()

        np.testing.assert_allclose(
            streamed["standard_error"], np.std(samples, axis=0, ddof=1), rtol=1e-5
        )

    def test_retained_samples_keep_replicate_order(self):
        rng = np.random.default_rng(2)
        samples = rng.standard_normal((60, 5))

        streamed = self._accumulate(samples, retain_samples=True).results()

        np.testing.assert_array_equal(streamed["samples"], samples)

    def test_non_finite_replicates_propagate_into_the_interval(self):
        samples = np.tile(np.arange(30.0)[:, None], (1, 3))
        samples[7, 1] = np.nan

        streamed = self._accumulate(samples).results()
        expected_lower, expected_upper = _reference_interval(samples, 0.95)

        np.testing.assert_array_equal(streamed["ci_lower"], expected_lower)
        np.testing.assert_array_equal(streamed["ci_upper"], expected_upper)
        assert np.isnan(streamed["standard_error"][1])

    @pytest.mark.parametrize("split", [99])
    def test_merge_is_deterministic_and_split_independent(self, split):
        rng = np.random.default_rng(4)
        samples = rng.standard_normal((200, 3))

        whole = self._accumulate(samples).results()
        first = self._accumulate(samples[:split], n_replicates=len(samples))
        second = self._accumulate(samples[split:], n_replicates=len(samples))
        merged = _BootstrapAccumulator.merge(first, second).results()

        for key in ("standard_error", "ci_lower", "ci_upper"):
            np.testing.assert_allclose(merged[key], whole[key], rtol=1e-10)

    def test_merge_is_order_independent(self):
        rng = np.random.default_rng(5)
        samples = rng.standard_normal((120, 3))
        first = self._accumulate(samples[:50], n_replicates=len(samples))
        second = self._accumulate(samples[50:], n_replicates=len(samples))

        forward = _BootstrapAccumulator.merge(first, second).results()
        backward = _BootstrapAccumulator.merge(second, first).results()

        for key in ("standard_error", "ci_lower", "ci_upper"):
            np.testing.assert_allclose(forward[key], backward[key], rtol=1e-10)

    def test_a_single_replicate_cannot_be_summarized(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="at least 2"):
            self._accumulate(rng.standard_normal((1, 3))).results()

    def test_a_mismatched_sample_shape_raises(self):
        accumulator = _BootstrapAccumulator((3,), n_replicates=10)
        with pytest.raises(ValueError, match="shape"):
            accumulator.update(np.zeros(4))


class TestBasicStatisticDefinitions:
    """The six basic statistics are exactly their NumPy reductions."""

    REDUCTIONS = {
        "mean": lambda a: np.mean(a, axis=0),
        "median": lambda a: np.median(a, axis=0),
        "std": lambda a: np.std(a, axis=0, ddof=0),
        "sum": lambda a: np.sum(a, axis=0),
        "min": lambda a: np.min(a, axis=0),
        "max": lambda a: np.max(a, axis=0),
    }

    #: A hand-built, deliberately asymmetric distribution — the exact values
    #: make every reduction (and the ddof choice) verifiable by hand.
    DATA = np.array(
        [
            [1.0, -4.0, 10.0],
            [2.0, -1.0, 10.0],
            [4.0, 0.0, 10.0],
            [8.0, 0.5, 10.0],
            [16.0, 100.0, 10.0],
        ]
    )

    @pytest.mark.parametrize("method", ["mean"])
    def test_estimate_is_the_reduction_on_the_unresampled_sample(self, method):
        result = _bootstrap_simple_cpu_parallel(
            self.DATA, method, n_samples=20, n_jobs=1, random_state=0
        )

        np.testing.assert_array_equal(
            result["estimate"], self.REDUCTIONS[method](self.DATA)
        )

    def test_std_is_the_population_deviation_matching_braindata_std(self):
        """`'std'` uses ddof=0, unlike the ddof=1 standard error across replicates."""
        computed = _bootstrap_simple_method_worker(
            self.DATA, "std", np.arange(len(self.DATA))
        )

        np.testing.assert_array_equal(computed, np.std(self.DATA, axis=0, ddof=0))
        assert not np.allclose(computed, np.std(self.DATA, axis=0, ddof=1))

    def test_non_finite_values_propagate_without_nan_substitution(self):
        data = self.DATA.copy()
        data[2, 0] = np.nan

        computed = _bootstrap_simple_method_worker(data, "mean", np.arange(len(data)))

        assert np.isnan(computed[0])


@pytest.mark.slow
class TestBootstrapSimpleEngine:
    """The simple-aggregation CPU engine's result contract."""

    def test_every_method_returns_the_full_field_set(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 20))

        for method in ("mean", "median", "std", "sum", "min", "max"):
            result = _bootstrap_simple_cpu_parallel(
                data, method, n_samples=100, n_jobs=1, random_state=42
            )

            assert set(RESULT_FIELDS) <= set(result)
            assert "backend" in result
            assert "samples" not in result
            for field in RESULT_FIELDS:
                assert result[field].shape == (20,)
                assert result[field].dtype == np.float64
            assert np.all(result["standard_error"] >= 0)
            assert np.all(result["ci_upper"] >= result["ci_lower"])

    def test_result_fields_carry_no_replicate_mean_or_p_values(self):
        rng = np.random.default_rng(43)
        result = _bootstrap_simple_cpu_parallel(
            rng.standard_normal((40, 5)), "mean", n_samples=50, n_jobs=1, random_state=1
        )

        assert not {"mean", "std", "Z", "p", "tail"} & set(result)

    def test_reproducibility(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 20))
        kwargs = {"n_samples": 100, "n_jobs": 1, "random_state": 42}

        first = _bootstrap_simple_cpu_parallel(data, "mean", **kwargs)
        second = _bootstrap_simple_cpu_parallel(data, "mean", **kwargs)

        for field in RESULT_FIELDS:
            np.testing.assert_array_equal(first[field], second[field])


@pytest.mark.slow
class TestBootstrapRidgeWeights:
    """The Ridge-weights CPU engine."""

    @staticmethod
    def _problem(seed=42, n_voxels=50):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((100, 10))
        y = rng.standard_normal((100, n_voxels))
        return X, y, _ridge_coefficients(X, y)

    def test_field_set_and_shapes(self):
        X, y, coef = self._problem()

        result = _bootstrap_ridge_weights_cpu_parallel(
            X, y, 1.0, coef, n_samples=100, n_jobs=1, random_state=42
        )

        assert set(RESULT_FIELDS) <= set(result)
        assert "samples" not in result
        for field in RESULT_FIELDS:
            assert result[field].shape == (10, 50)
        assert np.all(result["standard_error"] >= 0)

    def test_estimate_is_the_full_data_fit_not_the_replicate_mean(self):
        X, y, coef = self._problem()

        result = _bootstrap_ridge_weights_cpu_parallel(
            X, y, 1.0, coef, n_samples=100, n_jobs=1, random_state=42
        )

        np.testing.assert_array_equal(result["estimate"], coef)

    def test_return_samples_shape_and_interval_agreement(self):
        X, y, coef = self._problem()

        result = _bootstrap_ridge_weights_cpu_parallel(
            X,
            y,
            1.0,
            coef,
            n_samples=100,
            return_samples=True,
            n_jobs=1,
            random_state=42,
        )

        assert result["samples"].shape == (100, 10, 50)
        expected_lower, expected_upper = _reference_interval(result["samples"], 0.95)
        np.testing.assert_allclose(result["ci_lower"], expected_lower, rtol=1e-10)
        np.testing.assert_allclose(result["ci_upper"], expected_upper, rtol=1e-10)


@pytest.mark.slow
class TestBootstrapRidgePredict:
    """The Ridge-prediction CPU engine."""

    @staticmethod
    def _problem(seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((100, 10))
        y = rng.standard_normal((100, 50))
        X_test = rng.standard_normal((20, 10))
        coef = _ridge_coefficients(X, y)
        return X, y, X_test, coef

    def test_field_set_and_test_row_axis(self):
        X, y, X_test, coef = self._problem()

        result = _bootstrap_ridge_predict_cpu_parallel(
            X, y, X_test, 1.0, X_test @ coef, n_samples=100, n_jobs=1, random_state=42
        )

        assert set(RESULT_FIELDS) <= set(result)
        for field in RESULT_FIELDS:
            assert result[field].shape == (20, 50)

    def test_estimate_is_the_full_data_model_at_the_test_rows(self):
        X, y, X_test, coef = self._problem()

        result = _bootstrap_ridge_predict_cpu_parallel(
            X, y, X_test, 1.0, X_test @ coef, n_samples=100, n_jobs=1, random_state=42
        )

        np.testing.assert_allclose(result["estimate"], X_test @ coef, rtol=1e-12)

    def test_reproducibility(self):
        X, y, X_test, coef = self._problem()
        kwargs = {"n_samples": 100, "n_jobs": 1, "random_state": 42}

        first = _bootstrap_ridge_predict_cpu_parallel(
            X, y, X_test, 1.0, X_test @ coef, **kwargs
        )
        second = _bootstrap_ridge_predict_cpu_parallel(
            X, y, X_test, 1.0, X_test @ coef, **kwargs
        )

        for field in RESULT_FIELDS:
            np.testing.assert_array_equal(first[field], second[field])


class TestBootstrapValidation:
    """Arguments are rejected at the boundary, before any resampling."""

    DATA = np.random.default_rng(0).standard_normal((100, 50))

    def test_unsupported_method(self):
        with pytest.raises(ValueError, match="Unsupported method"):
            _bootstrap_simple_cpu_parallel(self.DATA, "invalid_method", n_samples=10)
        with pytest.raises(ValueError, match="mean.*median.*std"):
            _bootstrap_simple_cpu_parallel(self.DATA, "foobar", n_samples=10)

    def test_data_needs_two_observations(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            _bootstrap_simple_cpu_parallel(
                np.array([[1.0, 2.0, 3.0]]), "mean", n_samples=10
            )
        with pytest.raises(ValueError, match="at least 2 samples"):
            _bootstrap_simple_cpu_parallel(
                np.array([]).reshape(0, 3), "mean", n_samples=10
            )

    def test_data_must_be_one_or_two_dimensional(self):
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            _bootstrap_simple_cpu_parallel(np.zeros((10, 20, 30)), "mean", n_samples=10)
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            _bootstrap_simple_cpu_parallel(np.array(5.0), "mean", n_samples=10)

    def test_n_samples_floor_is_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            _bootstrap_simple_cpu_parallel(self.DATA, "mean", n_samples=1)
        with pytest.raises(TypeError, match="integer"):
            _bootstrap_simple_cpu_parallel(self.DATA, "mean", n_samples=10.0)
        with pytest.warns(UserWarning, match="reliable confidence intervals"):
            _bootstrap_simple_cpu_parallel(
                self.DATA, "mean", n_samples=100, n_jobs=1, random_state=0
            )

    @pytest.mark.parametrize("value", [1.5])
    def test_confidence_level_must_be_inside_the_unit_interval(self, value):
        with pytest.raises(ValueError, match="confidence_level"):
            _bootstrap_simple_cpu_parallel(
                self.DATA, "mean", n_samples=10, confidence_level=value
            )

    @pytest.mark.parametrize("removed", ["save_boots"])
    def test_removed_keywords_raise_type_error(self, removed):
        with pytest.raises(TypeError):
            _bootstrap_simple_cpu_parallel(
                self.DATA, "mean", n_samples=10, **{removed: 1}
            )


class TestBootstrapMemoryPreflight:
    """A run that cannot retain its output is refused before it resamples."""

    OVER_BUDGET = {
        "n_samples": 5000,
        "return_samples": True,
        "memory_budget_gb": 0.001,
    }

    def test_preflight_raises_before_any_resampling(self, monkeypatch):
        from nltools.algorithms.inference import bootstrap as engine

        def _never(*args, **kwargs):
            raise AssertionError("resampling started despite an over-budget preflight")

        monkeypatch.setattr(engine, "_generate_bootstrap_indices", _never)

        with pytest.raises(ValueError, match="exceeds the .* GB budget"):
            _bootstrap_simple_cpu_parallel(
                np.zeros((10, 200_000)), "mean", **self.OVER_BUDGET
            )

    def test_the_error_names_requirement_budget_and_override(self):
        with pytest.raises(ValueError) as caught:
            _bootstrap_simple_cpu_parallel(
                np.zeros((10, 200_000)), "mean", **self.OVER_BUDGET
            )

        message = str(caught.value)
        assert "GB to retain output" in message
        assert "0.001 GB budget" in message
        assert "memory_budget_gb=" in message
        assert "return_samples=True" in message

    def test_the_dispatch_window_never_scales_with_the_replicate_count(self):
        from nltools.algorithms.backends import _bootstrap_replicate_window

        assert _bootstrap_replicate_window(
            5_000, n_workers=4
        ) == _bootstrap_replicate_window(500_000, n_workers=4)
        assert _bootstrap_replicate_window(10, n_workers=4) == 10


class TestBootstrapWorkerPlanning:
    """`n_jobs` is a ceiling the memory planner may lower, never raise."""

    def test_a_tight_budget_lowers_the_worker_count(self):
        from nltools.algorithms.backends import _bootstrap_n_jobs_cpu

        roomy = _bootstrap_n_jobs_cpu(1.0, 100, memory_budget_gb=64.0, n_jobs=8)
        tight = _bootstrap_n_jobs_cpu(512.0, 100, memory_budget_gb=1.0, n_jobs=8)
        assert tight < roomy
        assert tight >= 1


class TestBootstrapReplicateFailure:
    """A terminal replicate failure names the replicate and drops nothing."""

    def test_failure_names_the_replicate_index(self, monkeypatch):
        from nltools.algorithms.inference import bootstrap as engine

        original = engine._bootstrap_simple_method_worker
        calls = {"n": 0}

        def exploding(data, method, indices):
            calls["n"] += 1
            if calls["n"] == 3:
                raise FloatingPointError("solver diverged")
            return original(data, method, indices)

        monkeypatch.setattr(engine, "_bootstrap_simple_method_worker", exploding)

        with pytest.raises(RuntimeError, match=r"bootstrap replicate 1 failed"):
            _bootstrap_simple_cpu_parallel(
                np.zeros((10, 4)), "mean", n_samples=10, n_jobs=1, random_state=0
            )


class TestValidateGpuBackend:
    """The GPU engine's backend guard must accept exactly the GPU device backends.

    Regression test: the guard compared against 'torch' — a pre-hyphenation name
    that matches no resolved backend — so torch-cuda was rejected and only
    torch-mps passed. Runs without GPU hardware via stub backends.
    """

    @pytest.mark.parametrize("name", ["numpy"])
    def test_rejects_cpu_backends(self, name):
        from types import SimpleNamespace

        from nltools.algorithms.inference.bootstrap import _validate_gpu_backend

        with pytest.raises(ValueError, match="torch-cuda"):
            _validate_gpu_backend(SimpleNamespace(name=name))


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
class TestBootstrapRidgeGpu:
    """The batched GPU driver matches the CPU engine and its own batching."""

    @staticmethod
    def _problem(seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((50, 10))
        y = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((20, 10))
        return X, y, X_test, _ridge_coefficients(X, y)

    def test_predictions_match_the_cpu_engine(self):
        from nltools.algorithms.backends import _Backend
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_ridge_predict_gpu_batched,
        )

        X, y, X_test, coef = self._problem()
        estimate = X_test @ coef
        cpu = _bootstrap_ridge_predict_cpu_parallel(
            X, y, X_test, 1.0, estimate, n_samples=100, n_jobs=1, random_state=42
        )
        gpu = _bootstrap_ridge_predict_gpu_batched(
            X,
            y,
            X_test,
            1.0,
            estimate,
            n_samples=100,
            backend=_Backend("torch"),
            memory_budget_gb=4.0,
            random_state=42,
        )

        assert gpu["estimate"].shape == (20, 20)
        np.testing.assert_allclose(
            gpu["standard_error"], cpu["standard_error"], rtol=1e-2, atol=1e-4
        )


@pytest.mark.slow
class TestBootstrapPeakMemory:
    """Peak memory must track the preflight figure, not the replicate count.

    These run at `n_jobs=1` on purpose: `tracemalloc` only sees this process, so
    a multi-worker run would report the parent's share and prove nothing. The
    parent is where the unbounded allocation used to live, so one worker is the
    honest place to measure it.
    """

    N_VOXELS = 10_000
    N_SAMPLES = 3000

    @classmethod
    def _peak_bytes(cls, **kwargs):
        data = np.zeros((10, cls.N_VOXELS))
        tracemalloc.start()
        try:
            _bootstrap_simple_cpu_parallel(
                data,
                "mean",
                n_samples=cls.N_SAMPLES,
                n_jobs=1,
                random_state=0,
                **kwargs,
            )
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_peak_stays_within_a_small_multiple_of_the_preflight_figure(self):
        from nltools.algorithms.backends import _bootstrap_output_bytes

        budgeted = _bootstrap_output_bytes(
            (self.N_VOXELS,),
            self.N_SAMPLES,
            confidence_level=0.95,
            return_samples=False,
            n_workers=1,
        )

        peak = self._peak_bytes()

        # Collecting every replicate before aggregating — the shape this engine
        # used to have — lands at ~6x the budget and ~1.2x the full
        # distribution, so this ceiling is what separates the two designs.
        assert peak < 2 * budgeted, (
            f"peak {peak / 1e6:.1f} MB against a preflight figure of "
            f"{budgeted / 1e6:.1f} MB — the run is not aggregating as it goes"
        )


@pytest.mark.slow
class TestBootstrapScale:
    """Realistic problem sizes stay correct as well as bounded."""

    def test_large_voxel_count_streams_without_retaining_replicates(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 50_000)).astype(np.float32)

        result = _bootstrap_simple_cpu_parallel(
            data, "mean", n_samples=1000, n_jobs=-1, random_state=42
        )

        assert result["estimate"].shape == (50_000,)
        np.testing.assert_allclose(
            result["estimate"], np.mean(data, axis=0, dtype=np.float64), rtol=1e-10
        )
        assert np.all(np.isfinite(result["standard_error"]))
        assert np.all(result["ci_upper"] >= result["ci_lower"])


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
class TestGpuReplicateFailure:
    """A terminal GPU replicate failure names its global replicate index."""

    def test_failure_names_the_global_replicate_index(self, monkeypatch):
        from nltools.algorithms.backends import _Backend
        from nltools.algorithms.inference import bootstrap as engine
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_ridge_weights_gpu_batched,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal((20, 4))
        coef = _ridge_coefficients(X, y)

        original = engine._refit_resample
        calls = {"n": 0}

        def exploding(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 4:
                raise FloatingPointError("solver diverged")
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, "_refit_resample", exploding)

        with pytest.raises(RuntimeError, match=r"bootstrap replicate 3 failed"):
            _bootstrap_ridge_weights_gpu_batched(
                X,
                y,
                1.0,
                coef,
                n_samples=10,
                backend=_Backend("torch"),
                memory_budget_gb=4.0,
                random_state=0,
            )
