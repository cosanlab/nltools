"""Utility functions for ridge regression.

Contains helper functions for batching, decomposition, and other utilities
following himalaya's implementation patterns.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from collections.abc import Iterator
from sklearn.utils import check_random_state

if TYPE_CHECKING:
    from nltools.algorithms.backends import Backend


def _auto_n_targets_batch(
    max_gpu_memory_gb: float,
    n_samples: int,
    n_alphas_batch: int,
    n_targets: int,
) -> int:
    """Derive a GPU target-batch size from a memory budget.

    Used by the CV solvers when ``n_targets_batch`` is left unset and
    ``parallel='gpu'`` so ``max_gpu_memory_gb`` actually bounds GPU
    allocation instead of processing all targets at once.

    The dominant target-scaling GPU tensor in the CV loop is the
    alpha-batched prediction block (~``n_alphas_batch * n_samples`` float32
    elements per target); a 5x overhead factor keeps peak allocation clear
    of the budget. This mirrors the batch-sizing used by the (now-removed)
    ICC GPU path. The result is floored at 1000 targets and capped at
    ``n_targets``.

    Args:
        max_gpu_memory_gb: GPU memory budget in GB.
        n_samples: Number of training samples.
        n_alphas_batch: Resolved alpha batch size.
        n_targets: Total number of targets (columns of Y).

    Returns:
        int: Target batch size in ``[min(1000, n_targets), n_targets]``.
    """
    bytes_per_element = 4  # float32
    memory_per_target = max(1, n_alphas_batch * n_samples * bytes_per_element * 5)
    n_targets_batch = int(max_gpu_memory_gb * 1e9 / memory_per_target)
    return max(min(1000, n_targets), min(n_targets_batch, n_targets))


def generate_dirichlet_samples(
    n_samples: int,
    n_kernels: int,
    concentration: float | list[float] = [0.1, 1.0],
    random_state: int | None = None,
) -> np.ndarray:
    """Generate samples from a Dirichlet distribution.

    This function generates random samples from a Dirichlet distribution,
    which is used for sampling feature space weights (gamma) in banded ridge
    regression random search.

    Args:
        n_samples: Number of samples to generate.
        n_kernels: Number of dimensions (feature spaces) of the distribution.
        concentration: Concentration parameters of the Dirichlet distribution.
            - A value of 1 corresponds to uniform sampling over the simplex.
            - A value of infinity corresponds to equal weights.
            - If a list, samples cycle through the list.
            Defaults to [0.1, 1.0].
        random_state: Random generator seed. Use an int for deterministic samples.
            Defaults to None.

    Returns:
        np.ndarray: Dirichlet samples of shape (n_samples, n_kernels).
            Each row sums to 1 (lies on simplex).

    Examples:
        >>> # Generate 10 samples for 3 feature spaces
        >>> gammas = generate_dirichlet_samples(10, 3, concentration=[0.1, 1.0])
        >>> gammas.shape
        (10, 3)
        >>> # Each row sums to 1
        >>> np.allclose(gammas.sum(axis=1), 1.0)
        True
    """
    random_generator = check_random_state(random_state)

    concentration_arr = np.atleast_1d(concentration)
    n_concentrations = len(concentration_arr)
    n_samples_per_concentration = int(np.ceil(n_samples / float(n_concentrations)))

    # Generate the gammas
    gammas = []
    for conc in concentration_arr:
        if conc == np.inf:
            # Equal weights for all spaces
            gamma = np.full(n_kernels, fill_value=1.0 / n_kernels)[None]
            gamma = np.tile(gamma, (n_samples_per_concentration, 1))
        else:
            # Sample from Dirichlet distribution
            gamma = random_generator.dirichlet(
                [conc] * n_kernels, n_samples_per_concentration
            )
        gammas.append(gamma)
    gammas = np.vstack(gammas)

    # Reorder the gammas to alternate between concentrations:
    # [a0, a1, a2, a0, a1, a2] instead of [a0, a0, a1, a1, a2, a2]
    gammas = gammas.reshape(n_concentrations, n_samples_per_concentration, n_kernels)
    gammas = np.swapaxes(gammas, 0, 1)
    gammas = gammas.reshape(n_concentrations * n_samples_per_concentration, n_kernels)

    # Remove extra gammas if we generated more than requested
    gammas = gammas[:n_samples]

    return gammas


def _decompose_ridge(
    Xtrain: np.ndarray,
    alphas: np.ndarray,
    n_alphas_batch: int | None = None,
    method: str = "svd",
    backend: Backend | None = None,
) -> Iterator[tuple[np.ndarray, slice]]:
    """Generator that yields resolution matrices for ridge predictions.

    This computes the resolution matrices needed for ridge predictions:
        Ytest_hat = Xtest @ (XtX + alpha * I)^-1 @ Xtrain^T @ Ytrain

    By using SVD decomposition, we can compute:
        matrices = (XtX + alpha * I)^-1 @ Xtrain^T

    for multiple alphas efficiently using a single SVD.

    CRITICAL: This is a GENERATOR (uses yield) for memory efficiency.
    Each alpha batch is yielded, processed, then deleted before the next batch.

    Args:
        Xtrain: Training features of shape (n_samples_train, n_features).
        alphas: Ridge regularization parameters to try, shape (n_alphas,).
        n_alphas_batch: If not None, yields batches of alphas. This saves memory
            when trying many alphas. Defaults to None (process all at once).
        method: Decomposition method. Currently only "svd" is supported.
            Defaults to "svd".
        backend: `Backend` providing `.svd`/`.matmul`/`.expand_dims` used for the
            decomposition. Effectively required: the default `None` is not handled and
            will raise `AttributeError`. Pass a numpy or torch `Backend` instance.

    Yields:
        tuple: (matrices, alpha_batch) where:
            - matrices: Resolution matrices of shape (n_alphas_batch, n_features, n_samples_train)
            - alpha_batch: Slice indicating which alphas this batch corresponds to

    Examples:
        >>> X = np.random.randn(100, 50)
        >>> alphas = np.array([0.1, 1.0, 10.0])
        >>> for matrices, batch in _decompose_ridge(X, alphas, n_alphas_batch=2):
        ...     print(f"Processing alphas {alphas[batch]}")
        ...     # Use matrices for predictions
        ...     # matrices is automatically deleted after this iteration
        Processing alphas [0.1 1.0]
        Processing alphas [10.0]

    Notes:
        - Uses generator pattern for memory efficiency (Principle 2: automatic memory efficiency)
        - Each batch is automatically cleaned up after yielding
    """

    # Default: process all alphas at once
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    # SVD decomposition: X = U @ diag(s) @ Vt
    if method == "svd":
        U, s, Vt = backend.svd(Xtrain, full_matrices=False)
    else:
        raise ValueError(f"Unknown method={method!r}")

    # Yield batches of resolution matrices
    for start in range(0, len(alphas), n_alphas_batch):
        batch = slice(start, start + n_alphas_batch)
        alphas_batch = alphas[batch]

        # Compute eigenvalue weighting for this alpha batch
        # Shape: (n_alphas_batch, n_features)
        if len(alphas_batch.shape) == 0:  # Scalar
            alphas_batch = backend.expand_dims(alphas_batch, 0)

        # Ridge solution: (XtX + alpha*I)^-1 = V @ diag(s/(s^2 + alpha)) @ Vt
        # We compute: s / (alpha + s^2) for each alpha
        ev_weighting = s[None, :] / (alphas_batch[:, None] + s[None, :] ** 2)

        # Resolution matrices: Vt.T @ diag(ev_weighting) @ U.T
        # Shape: (n_alphas_batch, n_features, n_samples_train)
        matrices = backend.matmul(Vt.T, ev_weighting[:, :, None] * U.T[None, :, :])

        yield matrices, batch

        # Delete to free memory (generator cleanup)
        del matrices


def _select_best_alphas(
    scores: np.ndarray,
    alphas: np.ndarray,
    local_alpha: bool,
    backend: Backend | None = None,
    conservative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select best alphas from cross-validation scores.

    Args:
        scores: Cross-validation scores of shape (n_splits, n_alphas, n_targets)
            for each split, alpha, and target.
        alphas: Ridge regularization parameters of shape (n_alphas,).
        local_alpha: If True, select best alpha independently for each target.
            If False, select single best alpha for all targets.
        backend: `Backend` providing `.asarray`/`.xp`/`.full` used for the array
            operations. Effectively required: the default `None` is not handled and will
            raise `AttributeError`. Pass a numpy or torch `Backend` instance.
        conservative: If True, select the largest alpha within 1 std of the best score.
            This provides more regularization when performance is similar.
            Defaults to False.

    Returns:
        tuple: (alphas_argmax, best_scores_mean) where:
            - alphas_argmax: Indices of best alphas for each target, shape (n_targets,)
            - best_scores_mean: Mean scores (averaged over CV splits) for best alphas,
                shape (n_targets,)

    Notes:
        - Follows himalaya's implementation pattern
        - Adds small epsilon bias toward larger alphas (more regularization) when scores are tied
    """

    # Ensure scores and alphas are on the same device
    scores = backend.asarray(scores)
    alphas = backend.asarray(alphas)

    # Average scores over CV splits
    # Shape: (n_alphas, n_targets)
    scores_mean = backend.xp.mean(scores, axis=0)

    # Add tiny epsilon slope to prefer larger alphas when scores are equal
    # This provides a principled tiebreaker toward more regularization
    scores_mean = scores_mean + (backend.xp.log(alphas) * 1e-10)[:, None]

    if local_alpha:
        # Select best alpha independently for each target
        alphas_argmax = backend.xp.argmax(scores_mean, axis=0)  # ty: ignore[unknown-argument]

        if conservative:
            # Conservative: take largest alpha within 1 std of best
            scores_std = backend.xp.std(scores, axis=0)
            best_scores = scores_mean[alphas_argmax, np.arange(len(alphas_argmax))]
            threshold = (
                best_scores - scores_std[alphas_argmax, np.arange(len(alphas_argmax))]
            )

            # Find which alphas beat the threshold
            beats_threshold = scores_mean > threshold[None, :]
            # Bias toward larger alphas
            beats_threshold = (
                beats_threshold.astype(np.float32)
                + backend.xp.log(alphas)[:, None] * 1e-4
            )
            alphas_argmax = backend.xp.argmax(beats_threshold, axis=0)  # ty: ignore[unknown-argument]

    else:
        # Global: select single best alpha for all targets
        if conservative:
            raise NotImplementedError(
                "conservative=True with local_alpha=False not implemented"
            )

        # Mean over targets, then argmax over alphas
        global_scores = backend.xp.mean(scores_mean, axis=1)
        best_alpha_idx = backend.xp.argmax(global_scores)

        # Broadcast to all targets
        alphas_argmax = backend.full(
            scores_mean.shape[1], fill_value=best_alpha_idx, dtype="int64"
        )

    # Get best scores for selected alphas
    best_scores_mean = scores_mean[alphas_argmax, np.arange(scores_mean.shape[1])]

    return alphas_argmax, best_scores_mean  # ty: ignore[invalid-return-type]


def _r2_score(
    y_true: np.ndarray, y_pred: np.ndarray, backend: Backend | None = None
) -> np.ndarray:
    """Compute R² score (coefficient of determination).

    Backend-agnostic implementation that works with NumPy, PyTorch, etc.

    Args:
        y_true: True target values of shape (n_samples, n_targets).
        y_pred: Predicted target values of shape (n_samples, n_targets) or
            (n_alphas, n_samples, n_targets).
        backend: `Backend` providing `.xp` (array module) used for the reductions.
            Effectively required: the default `None` is not handled and will raise
            `AttributeError`. Pass a numpy or torch `Backend` instance.

    Returns:
        np.ndarray: R² scores for each target. Shape (n_targets,) or (n_alphas, n_targets).

    Notes:
        - R² = 1 - SS_res / SS_tot
        - SS_res = sum((y_true - y_pred)^2)  # Residual sum of squares
        - SS_tot = sum((y_true - y_mean)^2)  # Total sum of squares
    """
    # Handle both 2D and 3D predictions (with alpha dimension)
    if len(y_pred.shape) == 3:
        # Shape: (n_alphas, n_samples, n_targets)
        ss_res = backend.xp.sum((y_true[None, :, :] - y_pred) ** 2, axis=1)
        ss_tot = backend.xp.sum(
            (y_true[None, :, :] - backend.xp.mean(y_true, axis=0)[None, None, :]) ** 2,
            axis=1,
        )
    else:
        # Shape: (n_samples, n_targets)
        ss_res = backend.xp.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = backend.xp.sum(
            (y_true - backend.xp.mean(y_true, axis=0)[None, :]) ** 2, axis=0
        )

    # R² = 1 - SS_res / SS_tot
    # Avoid division by zero
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    return r2
