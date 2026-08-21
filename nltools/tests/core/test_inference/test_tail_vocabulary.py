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
"""

import inspect

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as t_dist

from nltools.algorithms import (
    compute_multivariate_similarity,
    isc,
    isc_group,
    procrustes_distance,
    regress,
)
from nltools.algorithms.inference import (
    correlation_permutation_test,
    isc_group_permutation_test,
    isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    timeseries_correlation_permutation_test,
    two_sample_permutation_test,
)
from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
from nltools.data import Adjacency, BrainCollection, BrainData
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
    OnlineBootstrapStats.get_results,
    BrainData.ttest,
    BrainData.ttest2,
    BrainData.bootstrap,
    BrainData.multivariate_similarity,
    Adjacency.ttest,
    Adjacency.regress,
    Adjacency.similarity,
    Adjacency.bootstrap,
    BrainCollection.ttest,
    BrainCollection.ttest2,
    BrainCollection.permutation_test,
    BrainCollection.permutation_test2,
    BrainCollection.isc_test,
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

    def test_bootstrap_stats_tail(self):
        stats = OnlineBootstrapStats(shape=(5,))
        rng = np.random.default_rng(3)
        for _ in range(200):
            stats.update(1.0 + 0.5 * rng.standard_normal(5))
        two = stats.get_results(tail=2)
        one = stats.get_results(tail=1)
        z = two["Z"]
        np.testing.assert_allclose(one["p"], 1 - norm.cdf(z))
        np.testing.assert_allclose(one["p"][z > 0], two["p"][z > 0] / 2)

    def test_bootstrap_stats_rejects_old_vocab(self):
        stats = OnlineBootstrapStats(shape=(2,))
        for _ in range(3):
            stats.update(np.ones(2))
        with pytest.raises(ValueError, match="tail"):
            stats.get_results(tail="upper")


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
