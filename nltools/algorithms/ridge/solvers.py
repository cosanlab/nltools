"""Ridge regression solvers with cross-validation.

Implements banded ridge regression (multiple feature spaces) and regular ridge
regression (single feature space) with cross-validation for hyperparameter selection.

Follows himalaya's implementation patterns:
- Generator-based batching for memory efficiency
- Y_in_cpu strategy for large target datasets
- Backend abstraction (NumPy, PyTorch, PyTorch+CUDA)
- Per-target or global alpha selection
"""

import numpy as np
from sklearn.model_selection import KFold


def solve_banded_ridge_cv(
    Xs,
    Y,
    alphas=[0.1, 1.0, 10.0],
    cv=5,
    local_alpha=True,
    n_targets_batch=None,
    n_targets_batch_refit=None,
    n_alphas_batch=None,
    Y_in_cpu=True,
    backend="numpy",
    score_func=None,
    fit_intercept=False,
    progress_bar=False,
    conservative=False,
):
    """Solve banded ridge regression with cross-validation.

    This is the GENERAL implementation that handles multiple feature spaces.
    Regular ridge CV is a special case with len(Xs) = 1.

    Banded ridge applies the same regularization across all feature spaces,
    unlike himalaya's group ridge which applies different scalings per space.

    Parameters
    ----------
    Xs : list of arrays, each of shape (n_samples, n_features_i)
        Feature matrices for different feature spaces.
        All must have the same n_samples.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alphas : array-like, default=[0.1, 1.0, 10.0]
        Ridge regularization parameters to try.
    cv : int or sklearn splitter, default=5
        Cross-validation strategy. If int, uses KFold with that many splits.
    local_alpha : bool, default=True
        If True, select best alpha independently for each target.
        If False, select single best alpha for all targets.
    n_targets_batch : int or None, default=None
        Batch size for targets during CV (for memory efficiency).
        If None, processes all targets at once.
    n_targets_batch_refit : int or None, default=None
        Batch size for targets during refit.
        If None, uses n_targets_batch value.
    n_alphas_batch : int or None, default=None
        Batch size for alphas (for memory efficiency).
        If None, processes all alphas at once.
    Y_in_cpu : bool, default=True
        If True, keep Y on CPU and transfer batches to GPU as needed.
        This prevents OOM when Y is large (e.g., 300k voxels).
        DEFAULT is True (recommended for neuroimaging).
    backend : str or module, default="numpy"
        Backend to use: "numpy", "torch", or "torch_cuda".
    score_func : callable or None, default=None
        Scoring function (y_true, y_pred) -> scores.
        If None, uses R² score.
    fit_intercept : bool, default=False
        Whether to fit an intercept. If False, X and Y should be centered.
    progress_bar : bool, default=False
        Whether to display progress bar (requires tqdm).
    conservative : bool, default=False
        If True, select largest alpha within 1 std of best score.

    Returns
    -------
    best_alphas : array of shape (n_targets,)
        Selected best alpha for each target (or same alpha repeated if local_alpha=False).
    coefs : array of shape (n_features_total, n_targets)
        Ridge coefficients refit on entire dataset using best alphas.
        Always returned on CPU (numpy array).
    cv_scores : array of shape (n_targets,)
        Cross-validation scores for best alphas (mean across CV splits).
        Always returned on CPU (numpy array).

    Examples
    --------
    >>> # Single feature space (regular ridge)
    >>> X = np.random.randn(100, 50)
    >>> Y = np.random.randn(100, 10)
    >>> alphas, coefs, scores = solve_banded_ridge_cv([X], Y, alphas=[0.1, 1.0, 10.0])

    >>> # Multiple feature spaces (banded ridge)
    >>> X1 = np.random.randn(100, 30)  # First feature space
    >>> X2 = np.random.randn(100, 20)  # Second feature space
    >>> alphas, coefs, scores = solve_banded_ridge_cv([X1, X2], Y, alphas=[0.1, 1.0, 10.0])

    Notes
    -----
    This follows himalaya's solve_group_ridge_random_search pattern but without
    Dirichlet sampling (we use fixed equal weighting of feature spaces).

    Memory efficiency strategies:
    - Generator pattern for alpha batching (via _decompose_ridge)
    - Target batching (n_targets_batch)
    - Y_in_cpu strategy (transfer only needed batches to GPU)
    - Immediate cleanup with del statements
    """
    from .backends import set_backend, get_backend
    from .utils import _decompose_ridge, _select_best_alphas, _r2_score

    # Set backend
    if isinstance(backend, str):
        set_backend(backend)
    backend = get_backend()

    # Validate inputs
    if not Xs:
        raise ValueError("Xs cannot be empty")

    # Convert alphas to array
    alphas = np.asarray(alphas)
    if alphas.ndim == 0:
        alphas = alphas.reshape(1)

    # Concatenate feature spaces
    X = backend.concatenate([backend.asarray(Xi) for Xi in Xs], axis=1)
    n_samples, n_features = X.shape
    n_samples_y, n_targets = Y.shape

    if n_samples != n_samples_y:
        raise ValueError(f"n_samples mismatch: X has {n_samples}, Y has {n_samples_y}")

    # Validate all Xs have same n_samples
    for i, Xi in enumerate(Xs):
        if Xi.shape[0] != n_samples:
            raise ValueError(f"Xs[{i}] has {Xi.shape[0]} samples, expected {n_samples}")

    # Convert Y to backend array
    dtype = X.dtype
    device = getattr(X, "device", None)
    Y = backend.asarray(Y, dtype=dtype, device="cpu" if Y_in_cpu else device)
    alphas = backend.asarray(alphas, dtype=dtype)

    # Handle intercept
    X_offset, Y_offset = None, None
    if fit_intercept:
        X_offset = backend.mean(X, axis=0)
        Y_offset = backend.mean(Y, axis=0)
        X = X - X_offset[None, :]
        Y = Y - Y_offset[None, :]

    # Set batch sizes
    if n_targets_batch is None:
        n_targets_batch = n_targets
    if n_targets_batch_refit is None:
        n_targets_batch_refit = n_targets_batch
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    # Set score function
    if score_func is None:
        score_func = _r2_score

    # Setup cross-validation
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=False)

    n_splits = cv.get_n_splits()

    # Storage for CV scores
    # Shape: (n_splits, n_alphas, n_targets)
    scores = backend.zeros_like(
        X, shape=(n_splits, len(alphas), n_targets), device="cpu"
    )

    # Cross-validation loop
    for split_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        # Get train/val splits
        train_idx = backend.asarray(train_idx)
        val_idx = backend.asarray(val_idx)

        X_train = X[train_idx]
        X_val = X[val_idx]

        # Handle intercept per fold
        if fit_intercept:
            X_train_mean = backend.mean(X_train, axis=0)
            X_train = X_train - X_train_mean[None, :]
            X_val = X_val - X_train_mean[None, :]

        # Generator: batch over alphas
        for matrices, alpha_batch in _decompose_ridge(
            X_train, alphas, n_alphas_batch=n_alphas_batch
        ):
            # Compute X_val @ matrices for predictions
            # Shape: (n_alphas_batch, n_val_samples, n_train_samples)
            pred_matrix = backend.matmul(X_val, matrices)

            # Batch over targets
            for start in range(0, n_targets, n_targets_batch):
                batch = slice(start, start + n_targets_batch)

                # Get Y batches (transfer to GPU if needed)
                Y_train_batch = Y[:, batch][train_idx]
                Y_val_batch = Y[:, batch][val_idx]

                if Y_in_cpu:
                    Y_train_batch = backend.to_gpu(Y_train_batch, device=device)
                    Y_val_batch = backend.to_gpu(Y_val_batch, device=device)

                # Handle intercept per fold
                if fit_intercept:
                    Y_train_mean = backend.mean(Y_train_batch, axis=0)
                    Y_train_batch = Y_train_batch - Y_train_mean[None, :]
                    Y_val_batch = Y_val_batch - Y_train_mean[None, :]

                # Predictions: pred_matrix @ Y_train_batch
                # Shape: (n_alphas_batch, n_val_samples, n_targets_batch)
                predictions = backend.matmul(pred_matrix, Y_train_batch)

                # Score predictions
                # Shape: (n_alphas_batch, n_targets_batch)
                batch_scores = score_func(Y_val_batch, predictions)
                scores[split_idx, alpha_batch, batch] = backend.to_cpu(batch_scores)

                # Immediate cleanup
                del Y_train_batch, Y_val_batch, predictions

            # Generator cleanup (automatic, but explicit for clarity)
            del matrices, pred_matrix

    # Select best alphas
    alphas_argmax, best_scores = _select_best_alphas(
        scores, alphas, local_alpha, conservative
    )

    # Convert to numpy for indexing
    alphas_np = backend.to_cpu(alphas)
    alphas_argmax_np = backend.to_cpu(alphas_argmax)
    best_alphas = backend.to_cpu(alphas_np[alphas_argmax_np])

    # Refit on full dataset with best alphas
    coefs = _refit_banded_ridge(
        X=X,
        Y=Y,
        best_alphas=best_alphas,
        alphas=alphas,
        n_targets_batch=n_targets_batch_refit,
        n_alphas_batch=n_alphas_batch,
        Y_in_cpu=Y_in_cpu,
        backend=backend,
    )

    # Convert to numpy
    if hasattr(backend, "to_numpy"):
        best_alphas = backend.to_numpy(best_alphas)
        coefs = backend.to_numpy(coefs)
        best_scores = backend.to_numpy(best_scores)
    else:
        # Already numpy (numpy backend)
        best_alphas = np.asarray(best_alphas)
        coefs = np.asarray(coefs)
        best_scores = np.asarray(best_scores)

    return best_alphas, coefs, best_scores


def _refit_banded_ridge(
    X, Y, best_alphas, alphas, n_targets_batch, n_alphas_batch, Y_in_cpu, backend
):
    """Refit ridge regression on full dataset with selected alphas.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Full feature matrix.
    Y : array of shape (n_samples, n_targets)
        Full target matrix.
    best_alphas : array of shape (n_targets,)
        Selected best alpha for each target.
    alphas : array of shape (n_alphas,)
        All candidate alphas.
    n_targets_batch : int
        Batch size for targets.
    n_alphas_batch : int
        Batch size for alphas.
    Y_in_cpu : bool
        Whether Y is stored on CPU.
    backend : module
        Backend module.

    Returns
    -------
    coefs : array of shape (n_features, n_targets)
        Ridge coefficients for each target.
    """
    from .utils import _decompose_ridge

    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    device = getattr(X, "device", None)

    # Get unique alphas to minimize computation
    unique_alphas = backend.unique(best_alphas)
    unique_alphas = backend.asarray(unique_alphas)

    # Storage for coefficients
    coefs = backend.zeros_like(X, shape=(n_features, n_targets), device="cpu")

    # Refit for each unique alpha
    for matrices, alpha_batch in _decompose_ridge(
        X, unique_alphas, n_alphas_batch=min(len(unique_alphas), n_alphas_batch)
    ):
        # Batch over targets
        for start in range(0, n_targets, n_targets_batch):
            batch_slice = slice(start, start + n_targets_batch)

            # Find which targets in this batch use alphas from this alpha_batch
            target_alphas = best_alphas[batch_slice]

            # Get unique alphas in current alpha batch
            batch_alphas = unique_alphas[alpha_batch]

            # Find which targets use these alphas
            mask = backend.isin(target_alphas, batch_alphas)

            if not backend.any(mask):
                continue

            # Get Y for this batch
            Y_batch = Y[:, batch_slice]
            if Y_in_cpu:
                Y_batch = backend.to_gpu(Y_batch, device=device)

            # Compute weights: matrices @ Y_batch
            # Shape: (n_alphas_in_batch, n_features, n_targets_batch)
            weights_all = backend.matmul(matrices, Y_batch)

            # Select correct alpha for each target
            # For each target, find its alpha index in batch_alphas
            for i, (target_alpha, use_target) in enumerate(zip(target_alphas, mask)):
                if not use_target:
                    continue

                # Find index of this target's alpha in batch_alphas
                alpha_idx = backend.searchsorted(batch_alphas, target_alpha)
                alpha_idx = int(backend.to_cpu(alpha_idx))

                # Get weights for this alpha
                target_coefs = weights_all[alpha_idx, :, i]
                coefs[:, start + i] = backend.to_cpu(target_coefs)

            del Y_batch, weights_all

        del matrices

    return coefs


def solve_ridge_cv(
    X,
    Y,
    alphas=[0.1, 1.0, 10.0],
    cv=5,
    local_alpha=True,
    n_targets_batch=None,
    n_targets_batch_refit=None,
    n_alphas_batch=None,
    Y_in_cpu=True,
    backend="numpy",
    score_func=None,
    fit_intercept=False,
    progress_bar=False,
    conservative=False,
):
    """Solve ridge regression with cross-validation.

    This is a WRAPPER around solve_banded_ridge_cv with a single feature space.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Feature matrix.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alphas : array-like, default=[0.1, 1.0, 10.0]
        Ridge regularization parameters to try.
    cv : int or sklearn splitter, default=5
        Cross-validation strategy.
    local_alpha : bool, default=True
        If True, select best alpha independently for each target.
    n_targets_batch : int or None
        Batch size for targets during CV.
    n_targets_batch_refit : int or None
        Batch size for targets during refit.
    n_alphas_batch : int or None
        Batch size for alphas.
    Y_in_cpu : bool, default=True
        Keep Y on CPU and transfer batches to GPU as needed.
    backend : str or module, default="numpy"
        Backend to use.
    score_func : callable or None
        Scoring function. If None, uses R².
    fit_intercept : bool, default=False
        Whether to fit an intercept.
    progress_bar : bool, default=False
        Whether to display progress bar.
    conservative : bool, default=False
        If True, select largest alpha within 1 std of best score.

    Returns
    -------
    best_alphas : array of shape (n_targets,)
        Selected best alpha for each target.
    coefs : array of shape (n_features, n_targets)
        Ridge coefficients refit on entire dataset.
    cv_scores : array of shape (n_targets,)
        Cross-validation scores for best alphas.

    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> Y = np.random.randn(100, 10)
    >>> alphas, coefs, scores = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
    """
    # Just call banded ridge with single feature space
    return solve_banded_ridge_cv(
        Xs=[X],
        Y=Y,
        alphas=alphas,
        cv=cv,
        local_alpha=local_alpha,
        n_targets_batch=n_targets_batch,
        n_targets_batch_refit=n_targets_batch_refit,
        n_alphas_batch=n_alphas_batch,
        Y_in_cpu=Y_in_cpu,
        backend=backend,
        score_func=score_func,
        fit_intercept=fit_intercept,
        progress_bar=progress_bar,
        conservative=conservative,
    )
