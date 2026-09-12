"""Tests for bootstrap inference utilities."""

import tracemalloc

import numpy as np
import pytest

from nltools.algorithms.backends import check_gpu_available
from nltools.algorithms.inference.bootstrap import (
    BootstrapAccumulator,
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

    def test_not_re_exported_from_nltools_data(self):
        import nltools.data as data

        assert "BootstrapResult" not in data.__all__
        assert not hasattr(data, "BootstrapResult")


class TestBootstrapAccumulator:
    """Streaming Welford variance plus a bounded per-element retained tail."""

    @staticmethod
    def _accumulate(
        samples, confidence_level=0.95, retain_samples=False, n_replicates=None
    ):
        accumulator = BootstrapAccumulator(
            samples.shape[1:],
            n_replicates=samples.shape[0] if n_replicates is None else n_replicates,
            confidence_level=confidence_level,
            retain_samples=retain_samples,
        )
        for sample in samples:
            accumulator.update(sample)
        return accumulator

    def test_tail_size_follows_the_specified_formula(self):
        from nltools.algorithms.backends import bootstrap_retained_tail_size

        for n_samples, confidence_level in [
            (1000, 0.95),
            (999, 0.95),
            (100, 0.9),
            (50, 0.99),
        ]:
            expected = int(np.ceil((n_samples - 1) * (1 - confidence_level) / 2)) + 1
            assert (
                bootstrap_retained_tail_size(
                    n_samples, confidence_level=confidence_level
                )
                == expected
            )

    def test_tail_storage_is_a_fraction_of_the_replicates(self):
        """95% confidence retains roughly 5% of the distribution per element."""
        accumulator = BootstrapAccumulator((4,), n_replicates=5000)
        assert accumulator.tail_size * 2 / 5000 < 0.06

    @pytest.mark.parametrize("confidence_level", [0.5, 0.9, 0.95, 0.99])
    @pytest.mark.parametrize("n_replicates", [37, 40, 201, 500])
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

    def test_retention_does_not_change_interval_semantics(self):
        rng = np.random.default_rng(3)
        samples = rng.standard_normal((80, 5))

        bounded = self._accumulate(samples).results()
        retained = self._accumulate(samples, retain_samples=True).results()

        np.testing.assert_array_equal(bounded["ci_lower"], retained["ci_lower"])
        np.testing.assert_array_equal(bounded["ci_upper"], retained["ci_upper"])

    def test_non_finite_replicates_propagate_into_the_interval(self):
        samples = np.tile(np.arange(30.0)[:, None], (1, 3))
        samples[7, 1] = np.nan

        streamed = self._accumulate(samples).results()
        expected_lower, expected_upper = _reference_interval(samples, 0.95)

        np.testing.assert_array_equal(streamed["ci_lower"], expected_lower)
        np.testing.assert_array_equal(streamed["ci_upper"], expected_upper)
        assert np.isnan(streamed["standard_error"][1])

    @pytest.mark.parametrize("split", [1, 17, 99, 199])
    def test_merge_is_deterministic_and_split_independent(self, split):
        rng = np.random.default_rng(4)
        samples = rng.standard_normal((200, 3))

        whole = self._accumulate(samples).results()
        first = self._accumulate(samples[:split], n_replicates=len(samples))
        second = self._accumulate(samples[split:], n_replicates=len(samples))
        merged = BootstrapAccumulator.merge(first, second).results()

        for key in ("standard_error", "ci_lower", "ci_upper"):
            np.testing.assert_allclose(merged[key], whole[key], rtol=1e-10)

    def test_merge_is_order_independent(self):
        rng = np.random.default_rng(5)
        samples = rng.standard_normal((120, 3))
        first = self._accumulate(samples[:50], n_replicates=len(samples))
        second = self._accumulate(samples[50:], n_replicates=len(samples))

        forward = BootstrapAccumulator.merge(first, second).results()
        backward = BootstrapAccumulator.merge(second, first).results()

        for key in ("standard_error", "ci_lower", "ci_upper"):
            np.testing.assert_allclose(forward[key], backward[key], rtol=1e-10)

    def test_a_single_replicate_cannot_be_summarized(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="at least 2"):
            self._accumulate(rng.standard_normal((1, 3))).results()

    def test_merge_rejects_blocks_sized_for_different_runs(self):
        rng = np.random.default_rng(7)
        samples = rng.standard_normal((100, 3))
        first = self._accumulate(samples[:40], n_replicates=40)
        second = self._accumulate(samples[40:], n_replicates=100)

        with pytest.raises(ValueError, match="sized for"):
            BootstrapAccumulator.merge(first, second)

    def test_merge_rejects_a_retention_mismatch(self):
        """Merging a retaining block with a non-retaining one would lose draws."""
        rng = np.random.default_rng(8)
        samples = rng.standard_normal((60, 3))
        retaining = self._accumulate(
            samples[:30], retain_samples=True, n_replicates=len(samples)
        )
        bounded = self._accumulate(samples[30:], n_replicates=len(samples))

        with pytest.raises(ValueError, match="retain_samples"):
            BootstrapAccumulator.merge(retaining, bounded)
        with pytest.raises(ValueError, match="retain_samples"):
            BootstrapAccumulator.merge(bounded, retaining)

    def test_merged_retention_keeps_every_replicate_in_order(self):
        rng = np.random.default_rng(10)
        samples = rng.standard_normal((60, 3))
        first = self._accumulate(
            samples[:25], retain_samples=True, n_replicates=len(samples)
        )
        second = self._accumulate(
            samples[25:], retain_samples=True, n_replicates=len(samples)
        )

        merged = BootstrapAccumulator.merge(first, second).results()

        np.testing.assert_array_equal(merged["samples"], samples)

    def test_more_replicates_than_the_accumulator_was_sized_for_raises(self):
        """The tail is sized for `n_replicates`; a further one would misread it."""
        accumulator = BootstrapAccumulator((3,), n_replicates=2)
        accumulator.update(np.zeros(3))
        accumulator.update(np.ones(3))

        with pytest.raises(ValueError, match="sized for 2 replicates"):
            accumulator.update(np.zeros(3))

    def test_a_mismatched_sample_shape_raises(self):
        accumulator = BootstrapAccumulator((3,), n_replicates=10)
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

    @pytest.mark.parametrize("method", sorted(REDUCTIONS))
    def test_worker_matches_the_exact_numpy_reduction(self, method):
        indices = np.array([0, 0, 2, 4, 3])

        computed = _bootstrap_simple_method_worker(self.DATA, method, indices)

        np.testing.assert_array_equal(
            computed, self.REDUCTIONS[method](self.DATA[indices])
        )

    @pytest.mark.parametrize("method", sorted(REDUCTIONS))
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

    def test_return_samples_retains_the_distribution_and_agrees_with_it(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 20))

        result = _bootstrap_simple_cpu_parallel(
            data,
            "mean",
            n_samples=100,
            return_samples=True,
            n_jobs=1,
            random_state=42,
        )

        assert result["samples"].shape == (100, 20)
        expected_lower, expected_upper = _reference_interval(result["samples"], 0.95)
        np.testing.assert_allclose(result["ci_lower"], expected_lower, rtol=1e-10)
        np.testing.assert_allclose(result["ci_upper"], expected_upper, rtol=1e-10)
        np.testing.assert_allclose(
            result["standard_error"],
            np.std(result["samples"], axis=0, ddof=1),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("confidence_level", [0.8, 0.95, 0.99])
    def test_a_wider_confidence_level_gives_a_wider_interval(self, confidence_level):
        rng = np.random.default_rng(45)
        data = rng.standard_normal((50, 20))
        kwargs = {"n_samples": 200, "n_jobs": 1, "random_state": 2}

        narrow = _bootstrap_simple_cpu_parallel(
            data, "mean", confidence_level=0.5, **kwargs
        )
        wide = _bootstrap_simple_cpu_parallel(
            data, "mean", confidence_level=confidence_level, **kwargs
        )

        narrow_width = narrow["ci_upper"] - narrow["ci_lower"]
        wide_width = wide["ci_upper"] - wide["ci_lower"]
        assert np.all(wide_width >= narrow_width)
        assert np.mean(wide_width > narrow_width) > 0.9

    @pytest.mark.parametrize("n_jobs", [2, 3, -1])
    def test_worker_count_does_not_change_the_result(self, n_jobs):
        rng = np.random.default_rng(46)
        data = rng.standard_normal((60, 15))
        kwargs = {"n_samples": 120, "random_state": 3, "confidence_level": 0.9}

        sequential = _bootstrap_simple_cpu_parallel(data, "mean", n_jobs=1, **kwargs)
        parallel = _bootstrap_simple_cpu_parallel(data, "mean", n_jobs=n_jobs, **kwargs)

        for field in RESULT_FIELDS:
            np.testing.assert_allclose(sequential[field], parallel[field], rtol=1e-12)


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

    def test_heavier_regularization_shrinks_the_replicates(self):
        """The replicates must move with alpha — not just the estimate handed in.

        `estimate` is the caller's argument, so asserting on it would pass even
        if the engine ignored `alpha` entirely. `standard_error` and the
        interval come from the refits, so they cannot.
        """
        X, y, _ = self._problem()
        kwargs = {"n_samples": 200, "n_jobs": 1, "random_state": 42}

        light = _bootstrap_ridge_weights_cpu_parallel(
            X, y, 0.1, _ridge_coefficients(X, y, 0.1), **kwargs
        )
        heavy = _bootstrap_ridge_weights_cpu_parallel(
            X, y, 10.0, _ridge_coefficients(X, y, 10.0), **kwargs
        )

        # Shrinkage pulls every refit toward zero, so the replicate spread and
        # the interval both narrow.
        assert np.mean(heavy["standard_error"]) < np.mean(light["standard_error"])
        assert np.mean(heavy["ci_upper"] - heavy["ci_lower"]) < np.mean(
            light["ci_upper"] - light["ci_lower"]
        )
        assert np.mean(np.abs(heavy["ci_upper"])) < np.mean(np.abs(light["ci_upper"]))


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

    def test_replicates_apply_refitted_coefficients_to_the_unchanged_test_rows(self):
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_design,
            _refit_resample,
        )
        from nltools.algorithms.random import generate_bootstrap_indices

        X, y, X_test, coef = self._problem()
        result = _bootstrap_ridge_predict_cpu_parallel(
            X,
            y,
            X_test,
            1.0,
            X_test @ coef,
            n_samples=5,
            return_samples=True,
            n_jobs=1,
            random_state=11,
        )

        indices = generate_bootstrap_indices(len(X), 5, random_state=11)
        expected = X_test @ _refit_resample(_bootstrap_design([X], y), indices[0], 1.0)
        np.testing.assert_allclose(result["samples"][0], expected, atol=1e-8)

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

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 1.5, np.nan, np.inf])
    def test_confidence_level_must_be_inside_the_unit_interval(self, value):
        with pytest.raises(ValueError, match="confidence_level"):
            _bootstrap_simple_cpu_parallel(
                self.DATA, "mean", n_samples=10, confidence_level=value
            )

    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_memory_budget_must_be_finite_and_positive(self, value):
        with pytest.raises(ValueError, match="memory_budget_gb"):
            _bootstrap_simple_cpu_parallel(
                self.DATA, "mean", n_samples=10, memory_budget_gb=value
            )

    @pytest.mark.parametrize("removed", ["save_boots", "percentiles", "tail"])
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

        monkeypatch.setattr(engine, "generate_bootstrap_indices", _never)

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

    def test_bounded_retention_fits_a_budget_full_retention_cannot(self):
        """The refusal is specific to what the run asked to keep."""
        data = np.zeros((10, 50_000))
        kwargs = {"n_samples": 2000, "memory_budget_gb": 0.25, "n_jobs": 1}

        bounded = _bootstrap_simple_cpu_parallel(data, "mean", random_state=0, **kwargs)
        assert bounded["estimate"].shape == (50_000,)
        assert "samples" not in bounded

        with pytest.raises(ValueError, match="return_samples=True"):
            _bootstrap_simple_cpu_parallel(
                data, "mean", return_samples=True, random_state=0, **kwargs
            )

    def test_budget_bytes_charge_eight_per_output_sized_array(self):
        """Every array the run holds at once is charged, not just the tails."""
        from nltools.algorithms.backends import (
            BOOTSTRAP_TAIL_FLUSH_BLOCK as BLOCK,
            bootstrap_output_bytes,
            bootstrap_replicate_window,
            bootstrap_retained_tail_size,
        )

        tail = bootstrap_retained_tail_size(1000, confidence_level=0.95)
        window = bootstrap_replicate_window(1000, n_workers=1)
        # tails + flush buffer + the flush's two temporaries + one dispatch
        # window + two Welford accumulators + four summary payloads
        bounded = 2 * tail + BLOCK + 2 * (tail + BLOCK) + window + 6

        assert (
            bootstrap_output_bytes(
                (7,), 1000, confidence_level=0.95, return_samples=False
            )
            == 7 * bounded * 8
        )
        assert (
            bootstrap_output_bytes(
                (7,), 1000, confidence_level=0.95, return_samples=True
            )
            == 7 * (bounded + 1000) * 8
        )

    def test_the_flush_buffer_and_dispatch_window_are_budgeted(self):
        """The two allocations the engine controls are charged, not ignored."""
        from nltools.algorithms.backends import (
            bootstrap_output_bytes,
            bootstrap_replicate_window,
        )

        one_worker = bootstrap_output_bytes(
            (7,), 5000, confidence_level=0.95, return_samples=False, n_workers=1
        )
        many_workers = bootstrap_output_bytes(
            (7,), 5000, confidence_level=0.95, return_samples=False, n_workers=8
        )
        extra_window = bootstrap_replicate_window(
            5000, n_workers=8
        ) - bootstrap_replicate_window(5000, n_workers=1)

        assert many_workers - one_worker == 7 * extra_window * 8

    def test_the_dispatch_window_never_scales_with_the_replicate_count(self):
        from nltools.algorithms.backends import bootstrap_replicate_window

        assert bootstrap_replicate_window(
            5_000, n_workers=4
        ) == bootstrap_replicate_window(500_000, n_workers=4)
        assert bootstrap_replicate_window(10, n_workers=4) == 10


class TestBootstrapWorkerPlanning:
    """`n_jobs` is a ceiling the memory planner may lower, never raise."""

    def test_worker_count_never_exceeds_the_requested_ceiling(self):
        from nltools.algorithms.backends import bootstrap_n_jobs_cpu

        assert bootstrap_n_jobs_cpu(1.0, 100, n_jobs=2) <= 2
        assert bootstrap_n_jobs_cpu(1.0, 100, n_jobs=1) == 1

    def test_a_tight_budget_lowers_the_worker_count(self):
        from nltools.algorithms.backends import bootstrap_n_jobs_cpu

        roomy = bootstrap_n_jobs_cpu(1.0, 100, memory_budget_gb=64.0, n_jobs=8)
        tight = bootstrap_n_jobs_cpu(512.0, 100, memory_budget_gb=1.0, n_jobs=8)
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

    @pytest.mark.parametrize("name", ["torch-cuda", "torch-mps"])
    def test_accepts_gpu_device_backends(self, name):
        from types import SimpleNamespace

        from nltools.algorithms.inference.bootstrap import _validate_gpu_backend

        _validate_gpu_backend(SimpleNamespace(name=name))  # must not raise

    @pytest.mark.parametrize("name", ["torch-cpu", "numpy"])
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

    def test_weights_match_the_cpu_engine(self):
        from nltools.algorithms.backends import Backend
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_ridge_weights_gpu_batched,
        )

        X, y, _, coef = self._problem()
        cpu = _bootstrap_ridge_weights_cpu_parallel(
            X, y, 1.0, coef, n_samples=100, n_jobs=1, random_state=42
        )
        gpu = _bootstrap_ridge_weights_gpu_batched(
            X,
            y,
            1.0,
            coef,
            n_samples=100,
            backend=Backend("torch"),
            memory_budget_gb=4.0,
            random_state=42,
        )

        assert gpu["estimate"].shape == (10, 20)
        np.testing.assert_allclose(
            gpu["standard_error"], cpu["standard_error"], rtol=1e-2, atol=1e-4
        )
        np.testing.assert_allclose(
            gpu["ci_lower"], cpu["ci_lower"], rtol=1e-2, atol=1e-3
        )

    def test_predictions_match_the_cpu_engine(self):
        from nltools.algorithms.backends import Backend
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
            backend=Backend("torch"),
            memory_budget_gb=4.0,
            random_state=42,
        )

        assert gpu["estimate"].shape == (20, 20)
        np.testing.assert_allclose(
            gpu["standard_error"], cpu["standard_error"], rtol=1e-2, atol=1e-4
        )

    def test_forced_small_batches_do_not_change_the_result(self):
        """Scheduling is a memory decision, not a numerical one."""
        from nltools.algorithms.backends import Backend
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_ridge_weights_gpu_batched,
        )

        X, y, _, coef = self._problem()
        kwargs = {"n_samples": 200, "backend": Backend("torch"), "random_state": 42}

        one_batch = _bootstrap_ridge_weights_gpu_batched(
            X, y, 1.0, coef, memory_budget_gb=4.0, **kwargs
        )
        many_batches = _bootstrap_ridge_weights_gpu_batched(
            X, y, 1.0, coef, memory_budget_gb=0.001, **kwargs
        )

        for field in RESULT_FIELDS:
            np.testing.assert_allclose(
                many_batches[field], one_batch[field], rtol=1e-10, atol=1e-12
            )

    def test_confidence_interval_covers_the_full_data_estimate(self):
        from nltools.algorithms.backends import Backend
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_ridge_weights_gpu_batched,
        )

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        true_weights = rng.standard_normal((10, 20))
        y = X @ true_weights + 0.1 * rng.standard_normal((100, 20))
        coef = _ridge_coefficients(X, y)

        result = _bootstrap_ridge_weights_gpu_batched(
            X,
            y,
            1.0,
            coef,
            n_samples=2000,
            backend=Backend("torch"),
            memory_budget_gb=4.0,
            random_state=42,
        )

        covered = (result["ci_lower"] <= coef) & (coef <= result["ci_upper"])
        assert np.mean(covered) >= 0.90


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
        from nltools.algorithms.backends import bootstrap_output_bytes

        budgeted = bootstrap_output_bytes(
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

    def test_peak_stays_far_below_the_full_replicate_distribution(self):
        full_distribution = self.N_SAMPLES * self.N_VOXELS * 8

        peak = self._peak_bytes()

        assert peak < full_distribution / 3, (
            f"peak {peak / 1e6:.1f} MB against a {full_distribution / 1e6:.1f} MB "
            f"distribution — replicates are being retained"
        )

    def test_full_retention_peak_matches_its_larger_budget(self):
        """`return_samples=True` costs the distribution once, not twice."""
        from nltools.algorithms.backends import bootstrap_output_bytes

        budgeted = bootstrap_output_bytes(
            (self.N_VOXELS,),
            self.N_SAMPLES,
            confidence_level=0.95,
            return_samples=True,
            n_workers=1,
        )

        peak = self._peak_bytes(return_samples=True)

        # Appending to a list and `np.stack`-ing it at the end would hold the
        # distribution twice and land near 1.5x.
        assert peak < 1.25 * budgeted, (
            f"peak {peak / 1e6:.1f} MB against a preflight figure of "
            f"{budgeted / 1e6:.1f} MB — the distribution is being copied"
        )

    def test_doubling_the_replicates_does_not_double_the_peak(self):
        """The tail grows with `(1 - c) * B`; nothing else may grow with `B`."""
        data = np.zeros((10, self.N_VOXELS))

        def peak_for(n_samples):
            tracemalloc.start()
            try:
                _bootstrap_simple_cpu_parallel(
                    data, "mean", n_samples=n_samples, n_jobs=1, random_state=0
                )
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        small = peak_for(self.N_SAMPLES)
        large = peak_for(2 * self.N_SAMPLES)

        # Retaining every replicate would make this ratio ~2.0.
        assert large / small < 1.75


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
        from nltools.algorithms.backends import Backend
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
                backend=Backend("torch"),
                memory_budget_gb=4.0,
                random_state=0,
            )
