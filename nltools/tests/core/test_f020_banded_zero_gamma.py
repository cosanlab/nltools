"""Regression test for F020: zero Dirichlet weight poisons X in banded ridge.

The random-search loop used to scale feature blocks in place and restore them
by dividing X back by ``sqrt(gamma)``. When a Dirichlet weight is exactly 0,
that restore divided by zero, writing NaN/Inf into the shared X and corrupting
every subsequent iteration. The fix scales a per-iteration copy instead, so a
zero gamma no longer contaminates later iterations.
"""

import numpy as np

from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv


def test_zero_gamma_does_not_poison_later_iterations():
    """A gamma weight of exactly 0 must not produce NaN in the results."""
    rng = np.random.default_rng(0)
    n_samples = 40
    X1 = rng.standard_normal((n_samples, 8)).astype(np.float32)
    X2 = rng.standard_normal((n_samples, 6)).astype(np.float32)
    Y = rng.standard_normal((n_samples, 3)).astype(np.float32)

    # Inject an exact zero weight in the FIRST iteration; a later iteration with
    # valid weights would come back NaN if X were mutated/divided-by-zero.
    gammas = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

    result = solve_banded_ridge_cv(
        [X1, X2],
        Y,
        n_iter=gammas,
        alphas=[1.0, 10.0],
        cv=3,
        return_weights=True,
        progress_bar=False,
        random_state=0,
    )

    # CV scores for both iterations (including the one after the zero-gamma
    # iteration) must be finite — no NaN/Inf leaking through a corrupted X.
    assert np.isfinite(result["cv_scores"]).all()
    assert np.isfinite(result["coefs"]).all()
