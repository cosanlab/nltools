"""Ridge regression solvers with cross-validation.

Implements banded ridge regression (multiple feature spaces) and regular ridge
regression (single feature space) with cross-validation for hyperparameter selection.

Follows himalaya's implementation patterns:
- Generator-based batching for memory efficiency
- Y_in_cpu strategy for large target datasets
- Backend abstraction (NumPy, PyTorch, PyTorch+CUDA)
- Per-target or global alpha selection
- Random search over feature space weights (Dirichlet sampling)
"""

import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state


def solve_banded_ridge_cv(
    Xs,
    Y,
    n_iter=100,
    concentration=[0.1, 1.0],
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
    random_state=None,
    jitter_alphas=False,
    return_weights=True,
    diagonalize_method="svd",
    warn=True,
):
    """Solve banded ridge regression with cross-validation using random search.

    This function implements true banded/group ridge regression (as in Himalaya).
    It searches over feature space weights (gamma) sampled from a Dirichlet
    distribution, combined with alpha grid search.

    Banded ridge (also called group ridge) applies different scaling weights
    per feature space: Z_i = sqrt(gamma_i) * X_i, then solves standard ridge
    regression on the scaled concatenated features. This allows optimizing
    the relative importance of different feature spaces.

    The feature spaces are scaled by sqrt(gamma) for each gamma sample, then
    standard ridge regression is applied with alpha grid search.

    Parameters
    ----------
    Xs : list of arrays, each of shape (n_samples, n_features_i)
        Feature matrices for different feature spaces.
        All must have the same n_samples.
    Y : array of shape (n_samples, n_targets)
        Target data.
    n_iter : int, or array of shape (n_iter, n_spaces)
        Number of feature-space weights combination to search.
        If an array is given, the solver uses it as the list of weights
        to try, instead of sampling from a Dirichlet distribution.
    concentration : float, or list of float
        Concentration parameters of the Dirichlet distribution.
        - A value of 1 corresponds to uniform sampling over the simplex.
        - A value of infinity corresponds to equal weights.
        - If a list, iteratively cycle through the list.
        Not used if n_iter is an array.
    alphas : float or array of shape (n_alphas,)
        Range of ridge regularization parameters to try.
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
    random_state : int, or None, default=None
        Random generator seed. Use an int for deterministic search.
    jitter_alphas : bool, default=False
        If True, alphas range is slightly jittered for each gamma.
    return_weights : bool, default=True
        Whether to refit on the entire dataset and return the weights.
    diagonalize_method : str, default="svd"
        Method used to diagonalize the features. Currently only "svd" is supported.
    warn : bool, default=True
        If True, warn if the number of samples is smaller than the number of
        features.

    Returns
    -------
    deltas : array of shape (n_spaces, n_targets)
        Best log feature-space weights for each target.
        deltas = log(gamma / alpha), where gamma are the feature space weights.
    coefs : array of shape (n_features_total, n_targets), or None
        Ridge coefficients refit on entire dataset using best hyperparameters.
        Only returned if return_weights=True.
        Always returned on CPU (numpy array).
    cv_scores : array of shape (n_iter, n_targets)
        Cross-validation scores per iteration, averaged over splits, for the
        best alpha. Always returned on CPU (numpy array).
    intercept : array of shape (n_targets,), or None
        Intercept. Only returned when fit_intercept=True and return_weights=True.

    Examples
    --------
    >>> # Multiple feature spaces (banded ridge with random search)
    >>> X1 = np.random.randn(100, 30)  # First feature space
    >>> X2 = np.random.randn(100, 20)  # Second feature space
    >>> Y = np.random.randn(100, 10)
    >>> deltas, coefs, scores = solve_banded_ridge_cv(
    ...     [X1, X2], Y, n_iter=50, alphas=[0.1, 1.0, 10.0]
    ... )

    Notes
    -----
    This implements true banded/group ridge regression (as in Himalaya's
    solve_group_ridge_random_search) with:
    - Dirichlet sampling for feature space weights (gamma)
    - Scaling each feature space by sqrt(gamma) for each gamma sample
    - Cross-validation with alpha grid search
    - Per-target selection of best gamma and alpha combination

    This is the correct implementation of banded/group ridge regression, which
    allows different scaling weights per feature space. For single feature space
    ridge regression, use solve_ridge_cv instead.

    Memory efficiency strategies:
    - Generator pattern for alpha batching (via _decompose_ridge)
    - Target batching (n_targets_batch)
    - Y_in_cpu strategy (transfer only needed batches to GPU)
    - Immediate cleanup with del statements
    """
    from .backends import set_backend, get_backend
    from .utils import (
        _decompose_ridge,
        _select_best_alphas,
        _r2_score,
        generate_dirichlet_samples,
    )

    # Set backend
    if isinstance(backend, str):
        set_backend(backend)
    backend = get_backend()

    # Validate inputs
    if not Xs:
        raise ValueError("Xs cannot be empty")

    n_spaces = len(Xs)
    n_samples = Xs[0].shape[0]

    # Generate or use provided gammas
    if isinstance(n_iter, int):
        gammas = generate_dirichlet_samples(
            n_samples=n_iter,
            n_kernels=n_spaces,
            concentration=concentration,
            random_state=random_state,
        )
        # Set first gamma to equal weights (like Himalaya)
        gammas[0] = 1.0 / n_spaces
    elif isinstance(n_iter, np.ndarray) and n_iter.ndim == 2:
        gammas = n_iter
        if gammas.shape[1] != n_spaces:
            raise ValueError(
                f"n_iter shape mismatch: expected (n_iter, {n_spaces}), got {gammas.shape}"
            )
    else:
        raise ValueError(f"Unknown parameter n_iter={n_iter!r}")

    # Convert alphas to array
    alphas = np.asarray(alphas)
    if alphas.ndim == 0:
        alphas = alphas.reshape(1)

    # Concatenate feature spaces
    X = backend.concatenate([backend.asarray(Xi) for Xi in Xs], axis=1)
    n_features_list = [Xi.shape[1] for Xi in Xs]
    n_features = X.shape[1]
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices = [
        slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]

    n_samples_y, n_targets = Y.shape

    if n_samples != n_samples_y:
        raise ValueError(f"n_samples mismatch: X has {n_samples}, Y has {n_samples_y}")

    # Validate all Xs have same n_samples
    for i, Xi in enumerate(Xs):
        if Xi.shape[0] != n_samples:
            raise ValueError(f"Xs[{i}] has {Xi.shape[0]} samples, expected {n_samples}")

    # Warn if n_samples < n_features
    if n_samples < n_features and warn:
        warnings.warn(
            f"Solving banded ridge is slower than solving multiple-kernel ridge "
            f"when n_samples < n_features (here {n_samples} < {n_features}). "
            "Consider using kernel ridge regression instead.",
            UserWarning,
        )

    # Convert to backend arrays
    dtype = X.dtype
    device = getattr(X, "device", None)
    gammas = backend.asarray(gammas, dtype=dtype)
    alphas = backend.asarray(alphas, dtype=dtype)
    Y = backend.asarray(Y, dtype=dtype, device="cpu" if Y_in_cpu else device)

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

    # Setup random generator for alpha jittering
    random_generator, given_alphas = None, None
    if jitter_alphas:
        random_generator = check_random_state(random_state)
        given_alphas = backend.copy(alphas)

    # Initialize storage for best hyperparameters
    best_gammas = backend.full_like(
        X, fill_value=1.0 / n_spaces, shape=(n_spaces, n_targets)
    )
    best_alphas = backend.ones_like(X, shape=n_targets)
    cv_scores = backend.zeros_like(X, shape=(len(gammas), n_targets), device="cpu")
    current_best_scores = backend.full_like(X, fill_value=-np.inf, shape=n_targets)

    # Initialize refit weights
    coefs = None
    if return_weights:
        coefs = backend.zeros_like(X, shape=(n_features, n_targets), device="cpu")

    # Progress bar helper
    try:
        from tqdm import tqdm

        progress_iter = tqdm(
            gammas,
            desc=f"{len(gammas)} random sampling with cv",
            disable=not progress_bar,
        )
    except ImportError:
        progress_iter = gammas

    # Random search loop over gamma samples
    for ii, gamma in enumerate(progress_iter):
        # Scale each feature space by sqrt(gamma)
        for kk in range(n_spaces):
            X[:, slices[kk]] *= backend.sqrt(gamma[kk])

        # Jitter alphas if requested
        if jitter_alphas:
            noise = backend.asarray_like(random_generator.rand(), alphas)
            alphas = backend.asarray(given_alphas * (10 ** (noise - 0.5)), dtype=dtype)

        # Storage for CV scores for this gamma
        scores = backend.zeros_like(X, shape=(n_splits, len(alphas), n_targets))

        # Cross-validation loop
        for jj, (train_idx, val_idx) in enumerate(cv.split(X)):
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
                X_train,
                alphas,
                n_alphas_batch=n_alphas_batch,
                method=diagonalize_method,
            ):
                # Compute X_val @ matrices for predictions
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
                    predictions = backend.matmul(pred_matrix, Y_train_batch)

                    # Score predictions
                    batch_scores = score_func(Y_val_batch, predictions)
                    scores[jj, alpha_batch, batch] = backend.to_cpu(batch_scores)

                    # Immediate cleanup
                    del Y_train_batch, Y_val_batch, predictions

                # Generator cleanup
                del matrices, pred_matrix

            del train_idx, val_idx, X_train, X_val

        # Select best alphas for this gamma
        alphas_argmax, cv_scores_ii = _select_best_alphas(
            scores, alphas, local_alpha, conservative
        )
        cv_scores[ii, :] = backend.to_cpu(cv_scores_ii)

        # Update best_gammas and best_alphas
        # Get dtype safely - handle both numpy and torch dtypes
        try:
            dtype_str = str(dtype)
            if "float32" in dtype_str or dtype == np.float32:
                dtype_for_eps = np.float32
            elif "float64" in dtype_str or dtype == np.float64:
                dtype_for_eps = np.float64
            else:
                dtype_for_eps = np.float32  # Default fallback
        except (TypeError, AttributeError):
            dtype_for_eps = np.float32  # Default fallback
        epsilon = np.finfo(dtype_for_eps).eps
        mask = cv_scores_ii > current_best_scores + epsilon
        current_best_scores[mask] = cv_scores_ii[mask]
        best_gammas[:, mask] = gamma[:, None]
        best_alphas[mask] = alphas[alphas_argmax[mask]]

        # Refit weights on full dataset if requested
        if return_weights:
            update_indices = backend.flatnonzero(mask)
            if Y_in_cpu:
                update_indices = backend.to_cpu(update_indices)
            if len(update_indices) > 0:
                # Refit weights only for alphas used by at least one target
                used_alphas = backend.unique(best_alphas[mask])
                primal_weights = backend.zeros_like(
                    X, shape=(n_features, len(update_indices)), device="cpu"
                )

                for matrix, alpha_batch in _decompose_ridge(
                    Xtrain=X,
                    alphas=used_alphas,
                    n_alphas_batch=min(len(used_alphas), n_alphas_batch),
                    method=diagonalize_method,
                ):
                    for start in range(0, len(update_indices), n_targets_batch_refit):
                        batch = slice(start, start + n_targets_batch_refit)

                        Y_batch = Y[:, update_indices[batch]]
                        if Y_in_cpu:
                            Y_batch = backend.to_gpu(Y_batch, device=device)

                        weights = backend.matmul(matrix, Y_batch)

                        # Select alphas corresponding to best cv_score
                        alphas_indices = backend.searchsorted(
                            used_alphas, best_alphas[mask][batch]
                        )
                        # Mask targets whose selected alphas are outside the alpha batch
                        mask2 = backend.isin(
                            alphas_indices,
                            backend.arange(len(used_alphas))[alpha_batch],
                        )
                        # Get indices in alpha_batch
                        alphas_indices = backend.searchsorted(
                            backend.arange(len(used_alphas))[alpha_batch],
                            alphas_indices[mask2],
                        )
                        # Update corresponding weights
                        mask_target = backend.arange(weights.shape[2])
                        mask_target = backend.to_gpu(mask_target)[mask2]
                        tmp = weights[alphas_indices, :, mask_target]
                        primal_weights[:, batch][:, backend.to_cpu(mask2)] = (
                            backend.to_cpu(tmp).T
                        )
                        del weights, alphas_indices, mask2, mask_target

                    del matrix

                # Unscale weights: multiply by sqrt(gamma) again
                # We want to use the primal weights on the unscaled features Xs,
                # not on the scaled features (sqrt(gamma) * Xs)
                for kk in range(n_spaces):
                    primal_weights[slices[kk]] *= backend.to_cpu(
                        backend.sqrt(gamma[kk])
                    )

                coefs[:, backend.to_cpu(mask)] = primal_weights
                del primal_weights

            del update_indices

        del mask, scores

        # Reset scaling for next iteration
        for kk in range(n_spaces):
            X[:, slices[kk]] /= backend.sqrt(gamma[kk])

    # Compute deltas: log(gamma / alpha)
    deltas = backend.log(best_gammas / best_alphas[None, :])

    # Convert to numpy
    if hasattr(backend, "to_numpy"):
        deltas = backend.to_numpy(deltas)
        cv_scores = backend.to_numpy(cv_scores)
        if coefs is not None:
            coefs = backend.to_numpy(coefs)
    else:
        deltas = np.asarray(deltas)
        cv_scores = np.asarray(cv_scores)
        if coefs is not None:
            coefs = np.asarray(coefs)

    # Compute intercept if requested
    intercept = None
    if fit_intercept and return_weights:
        intercept = backend.to_numpy(Y_offset) - backend.to_numpy(X_offset) @ coefs

    # Return results
    if fit_intercept and return_weights:
        return deltas, coefs, cv_scores, intercept
    else:
        return deltas, coefs, cv_scores


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

    This function solves ridge regression for a single feature space with
    cross-validation for hyperparameter selection.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Feature matrix.
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
    coefs : array of shape (n_features, n_targets)
        Ridge coefficients refit on entire dataset using best alphas.
        Always returned on CPU (numpy array).
    cv_scores : array of shape (n_targets,)
        Cross-validation scores for best alphas (mean across CV splits).
        Always returned on CPU (numpy array).

    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> Y = np.random.randn(100, 10)
    >>> alphas, coefs, scores = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])

    Notes
    -----
    This is the efficient implementation for single feature space ridge regression
    with cross-validation. For multiple feature spaces (banded/group ridge),
    use solve_banded_ridge_cv instead.

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
    X = backend.asarray(X)
    n_samples, n_features = X.shape
    n_samples_y, n_targets = Y.shape

    if n_samples != n_samples_y:
        raise ValueError(f"n_samples mismatch: X has {n_samples}, Y has {n_samples_y}")

    # Convert alphas to array
    alphas = np.asarray(alphas)
    if alphas.ndim == 0:
        alphas = alphas.reshape(1)

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
        scores = backend.to_numpy(scores)
    else:
        # Already numpy (numpy backend)
        best_alphas = np.asarray(best_alphas)
        coefs = np.asarray(coefs)
        scores = np.asarray(scores)

    # Return full CV scores array for backward compatibility
    # Shape: (n_splits, n_alphas, n_targets)
    return best_alphas, coefs, scores
