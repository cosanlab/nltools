"""Standalone OLS regression on numpy arrays.

Pedagogical helper used in tutorials and notebooks where callers want a
``(b, se, t, p, df, res)`` tuple from a design matrix ``X`` and response
``Y`` without constructing a `BrainData` or `Glm`. For
4D neuroimaging data use `BrainData.fit` with ``model='glm'``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist


__all__ = ["regress"]


def regress(X, Y, method: str = "ols", stats: str = "full"):
    """Fit an OLS regression of ``Y`` on ``X``.

    Does not add an intercept — include one in ``X`` explicitly. If ``Y``
    is 2D, a separate regression is fit to each column.

    Args:
        X: Design matrix, shape ``(n_samples, n_regressors)``.
        Y: Response, shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
        method: Only ``'ols'`` is supported in v0.6.0. The legacy
            ``'robust'`` and ``'arma'`` methods were dropped; use
            statsmodels or a dedicated package if you need them.
        stats: ``'full'`` returns the 6-tuple below; ``'betas'`` returns
            just ``b``; ``'tstats'`` returns ``(b, t)``.

    Returns:
        tuple: ``(b, se, t, p, df, res)`` when ``stats='full'``:

        - ``b``: coefficients
        - ``se``: standard errors
        - ``t``: t-statistics
        - ``p``: two-tailed p-values
        - ``df``: residual degrees of freedom
        - ``res``: residuals
    """
    if method != "ols":
        raise NotImplementedError(
            f"regress(method={method!r}) is not supported in v0.6.0. "
            "Only 'ols' is available; use statsmodels for robust/ARMA fits."
        )
    if stats not in ("full", "betas", "tstats"):
        raise ValueError("stats must be one of 'full', 'betas', 'tstats'")

    X = np.asarray(X)
    Y = np.asarray(Y)
    y_was_1d = Y.ndim == 1
    if y_was_1d:
        Y = Y[:, np.newaxis]

    b = np.linalg.pinv(X) @ Y  # (n_regressors, n_targets)
    if stats == "betas":
        return b.squeeze()

    res = Y - X @ b
    df_scalar = X.shape[0] - X.shape[1]
    sigma = np.std(res, axis=0, ddof=X.shape[1])  # (n_targets,)
    xtx_inv_diag = np.diag(np.linalg.pinv(X.T @ X))  # (n_regressors,)
    se = np.sqrt(xtx_inv_diag)[:, np.newaxis] * sigma[np.newaxis, :]

    t = np.zeros_like(b)
    mask = se > 1e-6
    t[mask] = b[mask] / se[mask]

    if stats == "tstats":
        return b.squeeze(), t.squeeze()

    df = np.full(t.shape[1], df_scalar)
    p = 2 * (1 - t_dist.cdf(np.abs(t), df))

    return (
        b.squeeze(),
        se.squeeze(),
        t.squeeze(),
        p.squeeze(),
        df.squeeze(),
        res.squeeze(),
    )
