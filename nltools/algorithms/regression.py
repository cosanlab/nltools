"""Ordinary least squares on plain numpy arrays.

`regress` fits `Y ~ X` and returns coefficients, standard errors,
t-statistics, p-values, degrees of freedom, and residuals as arrays. Use it for
quick regressions on tabular or behavioral data; for voxel-wise models on
imaging data use `BrainData.fit(model='glm')`, which adds masking, run
handling, and the modeling helpers.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist


def regress(X, Y, *, stats: str = "full", tail: int | str = 2):
    """Fit an OLS regression of `Y` on `X`.

    Does not add an intercept; include one in `X` explicitly. If `Y` is 2D, a
    separate regression is fit to each column.

    Args:
        X (np.ndarray): Design matrix, shape (n_samples, n_regressors).
        Y (np.ndarray): Response, shape (n_samples,) or (n_samples, n_targets).
        stats (str): 'full' returns the 6-tuple below, 'betas' returns just `b`,
            'tstats' returns `(b, t)`. Defaults to 'full'.
        tail (int | str): 2 or 'two' for two-tailed p-values (default); 1 or
            'one' for a one-tailed test of beta > 0 (negate a regressor for the
            other direction).

    Returns:
        tuple: `(b, se, t, p, df, res)` when `stats='full'`: coefficients,
            standard errors, t-statistics, p-values (per `tail`), residual
            degrees of freedom, and residuals. `stats='betas'` returns just `b`;
            `stats='tstats'` returns `(b, t)`.

    Note:
        Residual degrees of freedom are `n - rank(X)`, so a rank-deficient design
        is scored against the subspace it actually fits. A target whose fit is
        undefined (a NaN in `Y`, say) scores `t` and `p` as NaN; a perfect fit,
        whose standard error is finite but near zero, scores `t = 0`, `p = 1`.
    """
    from .validation import _validate_tail_parameter

    if stats not in ("full", "betas", "tstats"):
        raise ValueError("stats must be one of 'full', 'betas', 'tstats'")
    tail_internal = _validate_tail_parameter(tail)

    # Promote to float before the covariance: an integer design overflows in
    # `X.T @ X` (a column of 1e5 wraps in int32), which leaves `b` correct —
    # `pinv` promotes — while se, t and p come out garbage with no warning.
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    y_was_1d = Y.ndim == 1
    if y_was_1d:
        Y = Y[:, np.newaxis]

    b = np.linalg.pinv(X) @ Y  # (n_regressors, n_targets)
    if stats == "betas":
        return b.squeeze()

    res = Y - X @ b
    # `pinv` fits rank-deficient designs, so the residual projection has rank
    # n - rank(X), not n - n_regressors. Counting columns would let a duplicated
    # regressor change the inference without changing the fit.
    df_scalar = X.shape[0] - int(np.linalg.matrix_rank(X))
    # Unbiased residual SE from *uncentered* RSS: sqrt(RSS / (n - rank(X))).
    # Correct for both intercept and intercept-free models. np.std(res,
    # ddof=rank(X)) would center the residuals, underestimating RSS when X has no
    # intercept (a supported usage — see the docstring). Matches
    # stats/correlation.py; see GH #287.
    sigma = np.sqrt((res**2).sum(axis=0) / df_scalar)  # (n_targets,)
    xtx_inv_diag = np.diag(np.linalg.pinv(X.T @ X))  # (n_regressors,)
    se = np.sqrt(xtx_inv_diag)[:, np.newaxis] * sigma[np.newaxis, :]

    # Two reasons a t-statistic is not b / se. A finite but near-zero se is a
    # perfect fit, which scores t = 0, p = 1 (GH #434). A nonfinite b or se is
    # an undefined fit, which scores NaN — "no effect" would be a confident
    # claim about a regression that never happened.
    t = np.zeros_like(b)
    finite = np.isfinite(b) & np.isfinite(se)
    t[~finite] = np.nan
    mask = finite & (se > 1e-6)
    t[mask] = b[mask] / se[mask]

    if stats == "tstats":
        return b.squeeze(), t.squeeze()

    df = np.full(t.shape[1], df_scalar)
    if tail_internal == "upper":
        p = 1 - t_dist.cdf(t, df)
    else:
        p = 2 * (1 - t_dist.cdf(np.abs(t), df))

    return (
        b.squeeze(),
        se.squeeze(),
        t.squeeze(),
        p.squeeze(),
        df.squeeze(),
        res.squeeze(),
    )
