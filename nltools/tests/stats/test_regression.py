"""Tests for the standalone OLS helper `nltools.stats.regress`.

`regress` is public (re-exported in `nltools.stats.__all__`) and tutorial-facing,
but its numeric output was previously unverified. These tests pin b/se/t/p/df
against an independent reference (`scipy.stats.linregress`) and cover the 1D-vs-2D
Y squeeze, the `stats='betas'/'tstats'` early returns, the near-zero-se t-mask,
and the input-validation branches.
"""

import numpy as np
import pytest
from scipy.stats import linregress

from nltools.stats import regress


@pytest.fixture
def ols_data():
    """A well-conditioned single-predictor design with an intercept column."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 2.5 + 1.3 * x + rng.normal(scale=0.5, size=n)
    X = np.column_stack([np.ones(n), x])
    return X, x, y


def test_regress_matches_scipy_linregress(ols_data):
    """b/se/t/p/df match scipy.stats.linregress for slope and intercept."""
    X, x, y = ols_data
    b, se, t, p, df, res = regress(X, y)

    ref = linregress(x, y)
    n = len(y)

    # Coefficients: column 0 is the intercept, column 1 the slope.
    assert np.isclose(b[0], ref.intercept)
    assert np.isclose(b[1], ref.slope)

    # Standard errors.
    assert np.isclose(se[0], ref.intercept_stderr)
    assert np.isclose(se[1], ref.stderr)

    # Slope t-stat and two-tailed p-value.
    assert np.isclose(t[1], ref.slope / ref.stderr)
    assert np.isclose(p[1], ref.pvalue)

    # Residual degrees of freedom = n - n_regressors.
    assert df == n - 2

    # Residuals reconstruct the fit.
    assert np.allclose(res, y - X @ b)


def test_regress_1d_y_squeezes_to_scalars(ols_data):
    """A 1D Y yields per-regressor 1D arrays and scalar df, not 2D outputs."""
    X, _, y = ols_data
    b, se, t, p, df, res = regress(X, y)

    n_reg = X.shape[1]
    assert b.shape == (n_reg,)
    assert se.shape == (n_reg,)
    assert t.shape == (n_reg,)
    assert p.shape == (n_reg,)
    assert np.ndim(df) == 0  # scalar
    assert res.shape == (X.shape[0],)


def test_regress_2d_y_matches_columnwise(ols_data):
    """Multi-target Y fits each column independently and preserves shape."""
    X, _, y = ols_data
    Y = np.column_stack([y, 3.0 * y - 1.0])  # second target is an affine map
    b, se, t, p, df, res = regress(X, Y)

    n_reg, n_targets = X.shape[1], Y.shape[1]
    assert b.shape == (n_reg, n_targets)
    assert se.shape == (n_reg, n_targets)
    assert t.shape == (n_reg, n_targets)
    assert p.shape == (n_reg, n_targets)
    assert df.shape == (n_targets,)
    assert res.shape == Y.shape

    # Each column must equal an independent single-target fit.
    for j in range(n_targets):
        bj, sej, tj, pj, dfj, resj = regress(X, Y[:, j])
        assert np.allclose(b[:, j], bj)
        assert np.allclose(se[:, j], sej)
        assert np.allclose(t[:, j], tj)
        assert np.allclose(p[:, j], pj)
        assert df[j] == dfj


def test_regress_stats_betas_returns_only_b(ols_data):
    """stats='betas' short-circuits to the coefficient vector alone."""
    X, _, y = ols_data
    b_full = regress(X, y)[0]
    b_only = regress(X, y, stats="betas")

    assert isinstance(b_only, np.ndarray)
    assert b_only.shape == b_full.shape
    assert np.allclose(b_only, b_full)


def test_regress_stats_tstats_returns_b_and_t(ols_data):
    """stats='tstats' returns exactly (b, t) matching the full computation."""
    X, _, y = ols_data
    b_full, _, t_full, *_ = regress(X, y)
    b, t = regress(X, y, stats="tstats")

    assert np.allclose(b, b_full)
    assert np.allclose(t, t_full)


def test_regress_near_zero_se_masks_tstat():
    """When residuals vanish (se -> 0), t is held at 0 instead of dividing by ~0."""
    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    b_true = np.array([1.0, -2.0])
    y = X @ b_true  # perfect fit -> residuals ~ 0 -> se ~ 0

    b, se, t, p, df, res = regress(X, y)

    # Coefficients are still recovered exactly.
    assert np.allclose(b, b_true)
    # se below the 1e-6 mask threshold zeroes the t-stat rather than blowing up.
    assert np.all(se < 1e-6)
    assert np.all(t == 0.0)
    assert np.all(np.isfinite(t))
    # Two-tailed p at t=0 is 1.0.
    assert np.allclose(p, 1.0)


def test_regress_no_intercept_uses_uncentered_rss():
    """Without an intercept column, the residual SE must use uncentered RSS/(n-p).

    `regress` documents "does not add an intercept — include one in X explicitly",
    so an intercept-free fit is a supported usage. There the residuals carry a
    nonzero mean, and `np.std(res, ddof=p)` (which centers) underestimates RSS,
    biasing `se` downward and inflating `t`/shrinking `p`. The sibling
    `stats/correlation.py` already fixed this per GH #287 with sqrt(RSS/(n-p)).
    """
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    X = x[:, None]  # single predictor, NO intercept column
    y = 5.0 + 2.0 * x + rng.normal(scale=1.0, size=n)  # 5.0 offset is unmodeled

    b, se, t, p, df, res = regress(X, y)

    # Correct reference: unbiased residual SE from uncentered RSS.
    p_reg = X.shape[1]
    sigma = np.sqrt(float((res**2).sum()) / (n - p_reg))
    se_expected = np.sqrt(np.diag(np.linalg.pinv(X.T @ X))) * sigma

    assert np.allclose(np.atleast_1d(se), se_expected), (
        f"se {se} != uncentered-RSS reference {se_expected}; regress is "
        "centering residuals (np.std) and biasing se downward."
    )
    # Guard the direction: the (buggy) centered sigma is strictly smaller here.
    assert sigma > np.std(res, ddof=p_reg)


def test_regress_non_ols_method_raises(ols_data):
    """Only 'ols' is supported; legacy methods raise NotImplementedError."""
    X, _, y = ols_data
    with pytest.raises(NotImplementedError):
        regress(X, y, method="robust")


def test_regress_invalid_stats_raises(ols_data):
    """An unknown stats value is rejected with ValueError."""
    X, _, y = ols_data
    with pytest.raises(ValueError):
        regress(X, y, stats="bogus")
