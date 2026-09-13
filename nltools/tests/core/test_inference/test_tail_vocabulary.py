"""Canonical tail vocabulary across every p-value-producing entry point (v0.6.0).

The library standard: `tail: int | str = 2`, accepting `2 | 'two'` (two-tailed,
the default everywhere) and `1 | 'one'` (one-tailed in the test's canonical
positive direction — correlation/ISC/similarity > 0, mean > popmean,
group1 > group2). The direction is fixed by the test, never by the data; users
wanting the negative direction negate their data, swap the groups, or flip the
contrast. The old `-1 / 'upper' / 'lower'` public forms are gone (the internal
`_compute_pvalue` keeps them for the forced-tail sites).

Sites with a statistically forced tail (distance correlation, ANOVA's F, isps'
Rayleigh, SRM variance components) and the GLM contrast p-maps (deliberately
nilearn one-sided, documented exception) take no `tail=` knob.

`BrainData.bootstrap` and `Adjacency.bootstrap` are a deliberate carve-out
rather than an omission: they report a percentile confidence interval and no
p-value at all, so there is no tail to choose. A bootstrap hypothesis test is a
separate, not-yet-defined API.
"""

import inspect

import numpy as np
import pytest
from scipy.stats import t as t_dist

from nltools.algorithms import (
    isc,
    isc_group,
    procrustes_distance,
    regress,
)
from nltools.algorithms.similarity import compute_multivariate_similarity
from nltools.algorithms.inference import (
    correlation_permutation_test,
    isc_group_permutation_test,
    isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    timeseries_correlation_permutation_test,
    two_sample_permutation_test,
)
from nltools.data import Adjacency, BrainData
from nltools.data.roc import Roc

TAIL_ENTRY_POINTS = [
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    timeseries_correlation_permutation_test,
    matrix_permutation_test,
    isc_permutation_test,
    isc_group_permutation_test,
    isc,
    isc_group,
    procrustes_distance,
    regress,
    compute_multivariate_similarity,
    BrainData.ttest,
    BrainData.multivariate_similarity,
    Adjacency.ttest,
    Adjacency.regress,
    Adjacency.similarity,
    Roc.calculate,
]


def _qualname(f):
    return getattr(f, "__qualname__", getattr(f, "__name__", str(f)))


@pytest.mark.parametrize("func", TAIL_ENTRY_POINTS, ids=_qualname)
def test_tail_kwarg_present_with_two_tailed_default(func):
    params = inspect.signature(func).parameters
    assert "tail" in params, f"{_qualname(func)} lacks tail="
    assert params["tail"].default == 2, (
        f"{_qualname(func)}: tail must default to 2 (two-tailed library standard)"
    )


@pytest.fixture
def correlated_xy():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(40)
    y = 0.7 * x + 0.3 * rng.standard_normal(40)
    return x, y


class TestVocabulary:
    @pytest.mark.parametrize("tail", [2, "two", 1, "one"])
    def test_accepted_forms(self, correlated_xy, tail):
        x, y = correlated_xy
        out = correlation_permutation_test(
            x, y, n_permute=50, tail=tail, n_jobs=1, random_state=0
        )
        assert 0 < out["p"] <= 1

    @pytest.mark.parametrize("tail", [-1, "upper", "lower", 3, "both"])
    def test_removed_and_invalid_forms_raise(self, correlated_xy, tail):
        x, y = correlated_xy
        with pytest.raises(ValueError, match="tail"):
            correlation_permutation_test(x, y, n_permute=50, tail=tail, n_jobs=1)

    def test_string_and_int_forms_are_identical(self, correlated_xy):
        x, y = correlated_xy
        kwargs = {"n_permute": 50, "n_jobs": 1, "random_state": 0}
        assert (
            correlation_permutation_test(x, y, tail=1, **kwargs)["p"]
            == correlation_permutation_test(x, y, tail="one", **kwargs)["p"]
        )
        assert (
            correlation_permutation_test(x, y, tail=2, **kwargs)["p"]
            == correlation_permutation_test(x, y, tail="two", **kwargs)["p"]
        )

    def test_one_means_positive_direction(self, correlated_xy):
        """tail=1 tests the canonical positive direction (here: correlation > 0)."""
        x, y = correlated_xy
        kwargs = {"n_permute": 200, "n_jobs": 1, "random_state": 0}
        p_one = correlation_permutation_test(x, y, tail=1, **kwargs)["p"]
        p_two = correlation_permutation_test(x, y, tail=2, **kwargs)["p"]
        assert p_one <= p_two


class TestParametricTails:
    def test_regress_tail_relationship(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 3))
        y = X @ np.array([0.8, -0.5, 0.0]) + rng.standard_normal(60)
        _, _, t, p_two, df, _ = regress(X, y, tail=2)
        _, _, _, p_one, _, _ = regress(X, y, tail=1)
        np.testing.assert_allclose(p_one, 1 - t_dist.cdf(t, np.full(t.shape, df)))
        pos = t > 0
        np.testing.assert_allclose(p_one[pos], p_two[pos] / 2)

    def test_multivariate_similarity_tail(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 2))
        y = X @ np.array([1.0, 0.2]) + rng.standard_normal(50)
        p_two = np.asarray(compute_multivariate_similarity(y, X, tail=2)["p"])
        p_one = np.asarray(compute_multivariate_similarity(y, X, tail=1)["p"])
        t = np.asarray(compute_multivariate_similarity(y, X, tail=2)["t"])
        pos = t > 0
        np.testing.assert_allclose(p_one[pos], p_two[pos] / 2)


class TestBootstrapTakesNoTail:
    """The bootstrap facades report an interval, so they expose no `tail=`."""

    @pytest.mark.parametrize("facade", [BrainData.bootstrap, Adjacency.bootstrap])
    def test_no_tail_kwarg(self, facade):
        assert "tail" not in inspect.signature(facade).parameters

    @pytest.mark.filterwarnings("ignore:n_samples=:UserWarning")
    @pytest.mark.filterwarnings("ignore:Only .* samples available:UserWarning")
    def test_the_result_reports_an_interval_and_nothing_tail_dependent(self):
        """The record's whole field set is interval-shaped, so no tail applies."""
        rng = np.random.default_rng(3)
        adj = Adjacency(rng.standard_normal((10, 6)), matrix_type="distance_flat")

        result = adj.bootstrap("mean", n_samples=20, n_jobs=1, random_state=0)

        assert set(type(result).__dataclass_fields__) == {
            "estimate",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "samples",
        }
        # A wider level moves the bounds; nothing here is a directional test.
        wider = adj.bootstrap(
            "mean", n_samples=20, confidence_level=0.99, n_jobs=1, random_state=0
        )
        assert np.all(wider.ci_lower.data <= result.ci_lower.data)
        assert np.all(wider.ci_upper.data >= result.ci_upper.data)


class TestIscFamilyVocabulary:
    @pytest.fixture
    def subjects_data(self):
        rng = np.random.default_rng(0)
        shared = rng.standard_normal((30, 1))
        return shared + 0.5 * rng.standard_normal((30, 6))

    def test_isc_engine_accepts_strings(self, subjects_data):
        out = isc_permutation_test(
            subjects_data, n_permute=30, tail="one", n_jobs=1, random_state=0
        )
        assert 0 < out["p"] <= 1
        with pytest.raises(ValueError, match="tail"):
            isc_permutation_test(subjects_data, n_permute=30, tail="upper", n_jobs=1)

    def test_isc_wrapper_accepts_strings(self, subjects_data):
        out = isc(subjects_data, n_samples=30, tail="one", n_jobs=1, random_state=0)
        assert 0 < out["p"] <= 1
