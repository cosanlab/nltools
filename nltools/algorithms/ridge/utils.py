"""Utility functions for ridge regression.

Contains helper functions for batching, decomposition, and other utilities
following himalaya's implementation patterns.
"""

import numpy as np


def _batch_or_skip(array, batch, axis):
    """Apply batch or skip if dimension is 1.

    This elegant pattern from himalaya handles both scalar and per-target
    operations without branching in the hot loop.

    Parameters
    ----------
    array : array
        Array to batch.
    batch : slice
        Batch slice to apply.
    axis : int (0 or 1)
        Axis to batch along.

    Returns
    -------
    array
        Batched array, or original if dimension is 1.

    Examples
    --------
    >>> # Scalar alpha (shape: (1,))
    >>> alphas = np.array([1.0])
    >>> _batch_or_skip(alphas, slice(0, 10), 0)  # Returns full array
    array([1.0])

    >>> # Per-target alphas (shape: (100,))
    >>> alphas = np.random.randn(100)
    >>> _batch_or_skip(alphas, slice(0, 10), 0).shape  # Returns batch
    (10,)
    """
    # If dimension is 1, skip batching (broadcast will handle it)
    if array.shape[axis] == 1:
        return array

    # Otherwise apply batch
    if axis == 0:
        return array[batch]
    elif axis == 1:
        return array[:, batch]
    else:
        raise NotImplementedError(f"Batching not implemented for axis={axis}")


def _decompose_ridge(Xtrain, alphas, n_alphas_batch=None, method="svd"):
    """Generator that yields resolution matrices for ridge predictions.

    This computes the resolution matrices needed for ridge predictions:
        Ytest_hat = Xtest @ (XtX + alpha * I)^-1 @ Xtrain^T @ Ytrain

    By using SVD decomposition, we can compute:
        matrices = (XtX + alpha * I)^-1 @ Xtrain^T

    for multiple alphas efficiently using a single SVD.

    CRITICAL: This is a GENERATOR (uses yield) for memory efficiency.
    Each alpha batch is yielded, processed, then deleted before the next batch.

    Parameters
    ----------
    Xtrain : array of shape (n_samples_train, n_features)
        Training features.
    alphas : array of shape (n_alphas,)
        Ridge regularization parameters to try.
    n_alphas_batch : int or None
        If not None, yields batches of alphas. This saves memory when
        trying many alphas.
    method : str, default="svd"
        Decomposition method. Currently only "svd" is supported.

    Yields
    ------
    matrices : array of shape (n_alphas_batch, n_features, n_samples_train)
        Resolution matrices for this alpha batch.
    alpha_batch : slice
        Slice indicating which alphas this batch corresponds to.

    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> alphas = np.array([0.1, 1.0, 10.0])
    >>> for matrices, batch in _decompose_ridge(X, alphas, n_alphas_batch=2):
    ...     print(f"Processing alphas {alphas[batch]}")
    ...     # Use matrices for predictions
    ...     # matrices is automatically deleted after this iteration
    Processing alphas [0.1 1.0]
    Processing alphas [10.0]
    """
    from .backends import get_backend

    backend = get_backend()

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


def _select_best_alphas(scores, alphas, local_alpha, conservative=False):
    """Select best alphas from cross-validation scores.

    Parameters
    ----------
    scores : array of shape (n_splits, n_alphas, n_targets)
        Cross-validation scores for each split, alpha, and target.
    alphas : array of shape (n_alphas,)
        Ridge regularization parameters.
    local_alpha : bool
        If True, select best alpha independently for each target.
        If False, select single best alpha for all targets.
    conservative : bool, default=False
        If True, select the largest alpha within 1 std of the best score.
        This provides more regularization when performance is similar.

    Returns
    -------
    alphas_argmax : array of shape (n_targets,)
        Indices of best alphas for each target.
    best_scores_mean : array of shape (n_targets,)
        Mean scores (averaged over CV splits) for best alphas.

    Notes
    -----
    Follows himalaya's implementation pattern. Adds small epsilon bias
    toward larger alphas (more regularization) when scores are tied.
    """
    from .backends import get_backend

    backend = get_backend()

    # Average scores over CV splits
    # Shape: (n_alphas, n_targets)
    scores_mean = backend.mean(scores, axis=0)

    # Add tiny epsilon slope to prefer larger alphas when scores are equal
    # This provides a principled tiebreaker toward more regularization
    scores_mean = scores_mean + (backend.log(alphas) * 1e-10)[:, None]

    if local_alpha:
        # Select best alpha independently for each target
        alphas_argmax = backend.argmax(scores_mean, axis=0)

        if conservative:
            # Conservative: take largest alpha within 1 std of best
            scores_std = backend.std(scores, axis=0)
            best_scores = scores_mean[alphas_argmax, np.arange(len(alphas_argmax))]
            threshold = (
                best_scores - scores_std[alphas_argmax, np.arange(len(alphas_argmax))]
            )

            # Find which alphas beat the threshold
            beats_threshold = scores_mean > threshold[None, :]
            # Bias toward larger alphas
            beats_threshold = (
                beats_threshold.astype(np.float32) + backend.log(alphas)[:, None] * 1e-4
            )
            alphas_argmax = backend.argmax(beats_threshold, axis=0)

    else:
        # Global: select single best alpha for all targets
        if conservative:
            raise NotImplementedError(
                "conservative=True with local_alpha=False not implemented"
            )

        # Mean over targets, then argmax over alphas
        global_scores = backend.mean(scores_mean, axis=1)
        best_alpha_idx = backend.argmax(global_scores)

        # Broadcast to all targets
        alphas_argmax = backend.full(
            scores_mean.shape[1], fill_value=best_alpha_idx, dtype="int64"
        )

    # Get best scores for selected alphas
    best_scores_mean = scores_mean[alphas_argmax, np.arange(scores_mean.shape[1])]

    return alphas_argmax, best_scores_mean


def _r2_score(y_true, y_pred):
    """Compute R² score (coefficient of determination).

    Backend-agnostic implementation that works with NumPy, PyTorch, etc.

    Parameters
    ----------
    y_true : array of shape (n_samples, n_targets)
        True target values.
    y_pred : array of shape (n_samples, n_targets) or (n_alphas, n_samples, n_targets)
        Predicted target values.

    Returns
    -------
    r2 : array of shape (n_targets,) or (n_alphas, n_targets)
        R² scores for each target.

    Notes
    -----
    R² = 1 - SS_res / SS_tot
    where:
        SS_res = sum((y_true - y_pred)^2)  # Residual sum of squares
        SS_tot = sum((y_true - y_mean)^2)  # Total sum of squares
    """
    from .backends import get_backend

    backend = get_backend()

    # Handle both 2D and 3D predictions (with alpha dimension)
    if len(y_pred.shape) == 3:
        # Shape: (n_alphas, n_samples, n_targets)
        ss_res = backend.sum((y_true[None, :, :] - y_pred) ** 2, axis=1)
        ss_tot = backend.sum(
            (y_true[None, :, :] - backend.mean(y_true, axis=0)[None, None, :]) ** 2,
            axis=1,
        )
    else:
        # Shape: (n_samples, n_targets)
        ss_res = backend.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = backend.sum(
            (y_true - backend.mean(y_true, axis=0)[None, :]) ** 2, axis=0
        )

    # R² = 1 - SS_res / SS_tot
    # Avoid division by zero
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    return r2
