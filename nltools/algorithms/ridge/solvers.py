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

from __future__ import annotations

import numpy as np
import warnings
from typing import Any
from collections.abc import Callable
from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.utils import check_random_state

from .utils import _auto_n_targets_batch

from ..backends import resolve_backend


def solve_banded_ridge_cv(
    # Required
    Xs: list[np.ndarray],
    Y: np.ndarray,
    *,
    # Optional algorithm parameters
    n_iter: int | np.ndarray = 100,
    concentration: float | list[float] = [0.1, 1.0],
    alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0],
    cv: int | BaseCrossValidator = 5,
    local_alpha: bool = True,
    n_targets_batch: int | None = None,
    n_targets_batch_refit: int | None = None,
    n_alphas_batch: int | None = None,
    Y_in_cpu: bool = True,
    score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    fit_intercept: bool = False,
    progress_bar: bool = False,
    conservative: bool = False,
    jitter_alphas: bool = False,
    return_weights: bool = True,
    diagonalize_method: str = "svd",
    warn: bool = True,
    # Backend parameters (grouped)
    parallel: str | None = "cpu",
    max_gpu_memory_gb: float = 4.0,
    # Random state (last)
    random_state: int | None = None,
) -> dict[str, Any]:
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

    Args:
        Xs: Feature matrices for different feature spaces. Each array has shape
            (n_samples, n_features_i). All must have the same n_samples.
        Y: Target data of shape (n_samples, n_targets).
        n_iter: Number of feature-space weights combination to search, or array
            of shape (n_iter, n_spaces). If an array is given, the solver uses
            it as the list of weights to try, instead of sampling from a Dirichlet
            distribution. Defaults to 100.
        concentration: Concentration parameters of the Dirichlet distribution.
            - A value of 1 corresponds to uniform sampling over the simplex.
            - A value of infinity corresponds to equal weights.
            - If a list, iteratively cycle through the list.
            Not used if n_iter is an array. Defaults to [0.1, 1.0].
        alphas: Range of ridge regularization parameters to try. Can be float
            or array of shape (n_alphas,). Defaults to [0.1, 1.0, 10.0].
        cv: Cross-validation strategy. If int, uses KFold with that many splits.
            Defaults to 5.
        local_alpha: If True, select best alpha independently for each target.
            If False, select single best alpha for all targets. Defaults to True.
        n_targets_batch: Batch size for targets during CV (for memory efficiency).
            If None, processes all targets at once. Defaults to None.
        n_targets_batch_refit: Batch size for targets during refit.
            If None, uses n_targets_batch value. Defaults to None.
        n_alphas_batch: Batch size for alphas (for memory efficiency).
            If None, processes all alphas at once. Defaults to None.
        Y_in_cpu: If True, keep Y on CPU and transfer batches to GPU as needed.
            This prevents OOM when Y is large (e.g., 300k voxels).
            Defaults to True (recommended for neuroimaging).
        score_func: Scoring function (y_true, y_pred) -> scores.
            If None, uses R² score. Defaults to None.
        fit_intercept: Whether to fit an intercept. If False, X and Y should be centered.
            Defaults to False.
        progress_bar: Whether to display progress bar (requires tqdm).
            Defaults to False.
        conservative: If True, select largest alpha within 1 std of best score.
            Defaults to False.
        jitter_alphas: If True, alphas range is slightly jittered for each gamma.
            Defaults to False.
        return_weights: Whether to refit on the entire dataset and return the weights.
            Defaults to True.
        diagonalize_method: Method used to diagonalize the features. Currently only "svd"
            is supported. Defaults to "svd".
        warn: If True, warn if the number of samples is smaller than the number of
            features. Defaults to True.
        parallel: Backend to use: "cpu", "gpu", or None.
            Defaults to "cpu".
        max_gpu_memory_gb: GPU memory budget in GB (only used if parallel="gpu").
            Defaults to 4.0.
        random_state: Random generator seed. Use an int for deterministic search.
            Defaults to None.

    Returns:
        dict: Dictionary with keys:
            - 'deltas': Best log feature-space weights for each target,
                shape (n_spaces, n_targets). deltas = log(gamma / alpha), where
                gamma are the feature space weights.
            - 'cv_scores': Cross-validation scores per iteration, averaged over splits,
                for the best alpha, shape (n_iter, n_targets). Always returned on CPU
                (numpy array).
            - 'coefs': Ridge coefficients refit on entire dataset using best hyperparameters,
                shape (n_features_total, n_targets), or None if return_weights=False.
                Always returned on CPU (numpy array).
            - 'intercept': Intercept of shape (n_targets,), or None if
                fit_intercept=False or return_weights=False.
            - 'backend': Backend used (for transparency).

    Examples:
        >>> # Multiple feature spaces (banded ridge with random search)
        >>> X1 = np.random.randn(100, 30)  # First feature space
        >>> X2 = np.random.randn(100, 20)  # Second feature space
        >>> Y = np.random.randn(100, 10)
        >>> result = solve_banded_ridge_cv(
        ...     [X1, X2], Y, n_iter=50, alphas=[0.1, 1.0, 10.0]
        ... )
        >>> deltas = result['deltas']
        >>> coefs = result['coefs']
        >>> scores = result['cv_scores']

    Notes:
        This implements true banded/group ridge regression (as in Himalaya's
        solve_group_ridge_random_search) with:
        - Dirichlet sampling for feature space weights (gamma)
        - Scaling each feature space by sqrt(gamma) for each gamma sample
        - Cross-validation with alpha grid search
        - Per-target selection of best gamma and alpha combination

        This is the correct implementation of banded/group ridge regression, which
        allows different scaling weights per feature space. For single feature space
        ridge regression, use solve_ridge_cv instead.

        Algorithm details:

        - Random search: Samples gamma weights from Dirichlet distribution
        - Banded ridge: Scales each feature space by sqrt(gamma_i), then solves standard ridge
        - Cross-validation: Evaluates each (gamma, alpha) combination via k-fold CV
        - Best selection: Chooses (gamma, alpha) that maximizes CV score per target

        Memory efficiency strategies (Principle 2: automatic memory efficiency):

        - Generator pattern for alpha batching (via _decompose_ridge): Processes alphas
          in batches to avoid storing all resolution matrices simultaneously
        - Target batching (n_targets_batch): Processes targets in chunks to fit GPU memory
        - Y_in_cpu strategy: Keeps large Y on CPU, transfers only batches needed
          for computation
        - Immediate cleanup with del statements: Explicitly frees memory after each batch

        Performance:

        - Time complexity: O(n_iter × n_splits × (n_alphas_batch × n_features^2 + n_targets_batch × n_samples))
        - Memory complexity: O(n_features × n_targets_batch) per batch
        - GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)

        See ``nltools.algorithms.ridge.utils._decompose_ridge()`` for generator pattern details.
        See ``nltools.algorithms.ridge.DESIGN.md`` for detailed algorithm explanation.
    """
    from .utils import (
        _decompose_ridge,
        _select_best_alphas,
        _r2_score,
        generate_dirichlet_samples,
    )

    backend = resolve_backend(parallel)
    xp = backend.xp

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
        X_offset = xp.mean(X, axis=0)
        Y_offset = xp.mean(Y, axis=0)
        X = X - X_offset[None, :]
        Y = Y - Y_offset[None, :]

    # Set batch sizes. When running on GPU with no explicit target batch,
    # derive one from max_gpu_memory_gb so the budget actually bounds
    # allocation instead of processing all targets at once.
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)
    if n_targets_batch is None:
        n_targets_batch = (
            _auto_n_targets_batch(
                max_gpu_memory_gb, n_alphas_batch * n_samples, n_targets
            )
            if parallel == "gpu"
            else n_targets
        )
    if n_targets_batch_refit is None:
        n_targets_batch_refit = n_targets_batch

    # Set score function
    if score_func is None:
        score_func = lambda y_true, y_pred: _r2_score(y_true, y_pred, backend=backend)

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
        # Scale each feature space by sqrt(gamma) on a fresh copy so the
        # original X is never mutated. Dividing X back by sqrt(gamma) in place
        # would write NaN/Inf whenever a Dirichlet weight underflows to exactly
        # 0 (division by zero), poisoning every subsequent iteration.
        X_scaled = backend.copy(X)
        for kk in range(n_spaces):
            X_scaled[:, slices[kk]] *= xp.sqrt(gamma[kk])

        # Jitter alphas if requested
        if jitter_alphas:
            noise = backend.asarray_like(random_generator.rand(), alphas)
            alphas = backend.asarray(given_alphas * (10 ** (noise - 0.5)), dtype=dtype)

        # Storage for CV scores for this gamma
        scores = backend.zeros_like(X, shape=(n_splits, len(alphas), n_targets))

        # Cross-validation loop
        for jj, (train_idx, val_idx) in enumerate(cv.split(X_scaled)):
            # Keep CPU copies for indexing CPU-resident Y
            train_idx_cpu = train_idx
            val_idx_cpu = val_idx
            train_idx = backend.asarray(train_idx)
            val_idx = backend.asarray(val_idx)
            X_train = X_scaled[train_idx]
            X_val = X_scaled[val_idx]

            # Handle intercept per fold
            if fit_intercept:
                X_train_mean = xp.mean(X_train, axis=0)
                X_train = X_train - X_train_mean[None, :]
                X_val = X_val - X_train_mean[None, :]

            # Generator: batch over alphas
            # _decompose_ridge yields (resolution_matrices, alpha_indices) pairs
            # This avoids storing all resolution matrices simultaneously (memory efficient)
            for matrices, alpha_batch in _decompose_ridge(
                X_train,
                alphas,
                n_alphas_batch=n_alphas_batch,
                method=diagonalize_method,
                backend=backend,
            ):
                # Compute X_val @ matrices for predictions
                # matrices shape: (n_alphas_batch, n_features, n_train_samples)
                # pred_matrix shape: (n_alphas_batch, n_val_samples, n_train_samples)
                pred_matrix = backend.matmul(X_val, matrices)

                # Batch over targets
                for start in range(0, n_targets, n_targets_batch):
                    batch = slice(start, start + n_targets_batch)

                    # Get Y batches — use CPU indices when Y is on CPU
                    y_train = train_idx_cpu if Y_in_cpu else train_idx
                    y_val = val_idx_cpu if Y_in_cpu else val_idx
                    Y_train_batch = Y[:, batch][y_train]
                    Y_val_batch = Y[:, batch][y_val]

                    if Y_in_cpu:
                        Y_train_batch = backend.to_gpu(Y_train_batch, device=device)
                        Y_val_batch = backend.to_gpu(Y_val_batch, device=device)

                    # Handle intercept per fold
                    if fit_intercept:
                        Y_train_mean = xp.mean(Y_train_batch, axis=0)
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
            scores, alphas, local_alpha, backend=backend, conservative=conservative
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
                used_alphas = xp.unique(best_alphas[mask])
                primal_weights = backend.zeros_like(
                    X, shape=(n_features, len(update_indices)), device="cpu"
                )

                for matrix, alpha_batch in _decompose_ridge(
                    Xtrain=X_scaled,
                    alphas=used_alphas,
                    n_alphas_batch=min(len(used_alphas), n_alphas_batch),
                    method=diagonalize_method,
                    backend=backend,
                ):
                    for start in range(0, len(update_indices), n_targets_batch_refit):
                        batch = slice(start, start + n_targets_batch_refit)

                        Y_batch = Y[:, update_indices[batch]]
                        if Y_in_cpu:
                            Y_batch = backend.to_gpu(Y_batch, device=device)

                        weights = backend.matmul(matrix, Y_batch)

                        # Select alphas corresponding to best cv_score
                        alphas_indices = xp.searchsorted(
                            used_alphas, best_alphas[mask][batch]
                        )
                        # Mask targets whose selected alphas are outside the alpha batch
                        mask2 = xp.isin(
                            alphas_indices,
                            xp.arange(len(used_alphas))[alpha_batch],
                        )
                        # Get indices in alpha_batch
                        alphas_indices = xp.searchsorted(
                            xp.arange(len(used_alphas))[alpha_batch],
                            alphas_indices[mask2],
                        )
                        # Update corresponding weights
                        mask_target = xp.arange(weights.shape[2])
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
                    primal_weights[slices[kk]] *= backend.to_cpu(xp.sqrt(gamma[kk]))

                coefs[:, backend.to_cpu(mask)] = primal_weights
                del primal_weights

            del update_indices

        del mask, scores, X_scaled

    # Compute deltas: log(gamma / alpha)
    deltas = xp.log(best_gammas / best_alphas[None, :])

    # Convert to numpy
    deltas = backend.to_numpy(deltas)
    cv_scores = backend.to_numpy(cv_scores)
    if coefs is not None:
        coefs = backend.to_numpy(coefs)

    # Compute intercept if requested
    intercept = None
    if fit_intercept and return_weights:
        intercept = backend.to_numpy(Y_offset) - backend.to_numpy(X_offset) @ coefs

    # Return dict with consistent keys
    result = {
        "deltas": deltas,
        "cv_scores": cv_scores,
        "backend": backend.name,
    }

    if return_weights:
        result["coefs"] = coefs

    if fit_intercept and return_weights:
        result["intercept"] = intercept

    return result


def _refit_banded_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    best_alphas: np.ndarray,
    n_targets_batch: int | None,
    n_alphas_batch: int | None,
    Y_in_cpu: bool,
    backend: Any,
) -> np.ndarray:
    """Refit ridge regression on full dataset with selected alphas.

    Args:
        X: Full feature matrix of shape (n_samples, n_features).
        Y: Full target matrix of shape (n_samples, n_targets).
        best_alphas: Selected best alpha for each target, shape (n_targets,).
        n_targets_batch: Batch size for targets.
        n_alphas_batch: Batch size for alphas.
        Y_in_cpu: Whether Y is stored on CPU.
        backend: Backend module.

    Returns:
        np.ndarray: Ridge coefficients for each target, shape (n_features, n_targets).
    """
    from .utils import _decompose_ridge

    xp = backend.xp
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    device = getattr(X, "device", None)

    # Ensure best_alphas is on the backend device
    best_alphas = backend.asarray(best_alphas)

    # Get unique alphas to minimize computation
    unique_alphas = xp.unique(best_alphas)
    unique_alphas = backend.asarray(unique_alphas)

    # Storage for coefficients
    coefs = backend.zeros_like(X, shape=(n_features, n_targets), device="cpu")

    # Refit for each unique alpha
    for matrices, alpha_batch in _decompose_ridge(
        X,
        unique_alphas,
        n_alphas_batch=min(len(unique_alphas), n_alphas_batch),
        backend=backend,
    ):
        # Batch over targets
        for start in range(0, n_targets, n_targets_batch):
            batch_slice = slice(start, start + n_targets_batch)

            # Find which targets in this batch use alphas from this alpha_batch
            target_alphas = best_alphas[batch_slice]

            # Get unique alphas in current alpha batch
            batch_alphas = unique_alphas[alpha_batch]

            # Find which targets use these alphas
            mask = xp.isin(target_alphas, batch_alphas)

            if not xp.any(mask):
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
                alpha_idx = xp.searchsorted(batch_alphas, target_alpha)
                alpha_idx = int(backend.to_cpu(alpha_idx))

                # Get weights for this alpha
                target_coefs = weights_all[alpha_idx, :, i]
                coefs[:, start + i] = backend.to_cpu(target_coefs)

            del Y_batch, weights_all

        del matrices

    return coefs


def solve_ridge_cv(
    # Required
    X: np.ndarray,
    Y: np.ndarray,
    *,
    # Optional algorithm parameters
    alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0],
    cv: int | BaseCrossValidator = 5,
    local_alpha: bool = True,
    n_targets_batch: int | None = None,
    n_targets_batch_refit: int | None = None,
    n_alphas_batch: int | None = None,
    Y_in_cpu: bool = True,
    score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    fit_intercept: bool = False,
    progress_bar: bool = False,
    conservative: bool = False,
    # Backend parameters (grouped)
    parallel: str | None = "cpu",
    max_gpu_memory_gb: float = 4.0,
    # Random state (last)
    random_state: int | None = None,
) -> dict[str, Any]:
    """Solve ridge regression with cross-validation.

    This function solves ridge regression for a single feature space with
    cross-validation for hyperparameter selection.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        Y: Target data of shape (n_samples, n_targets).
        alphas: Ridge regularization parameters to try.
            Defaults to [0.1, 1.0, 10.0].
        cv: Cross-validation strategy. If int, uses KFold with that many splits.
            Defaults to 5.
        local_alpha: If True, select best alpha independently for each target.
            If False, select single best alpha for all targets. Defaults to True.
        n_targets_batch: Batch size for targets during CV (for memory efficiency).
            If None, processes all targets at once. Defaults to None.
        n_targets_batch_refit: Batch size for targets during refit.
            If None, uses n_targets_batch value. Defaults to None.
        n_alphas_batch: Batch size for alphas (for memory efficiency).
            If None, processes all alphas at once. Defaults to None.
        Y_in_cpu: If True, keep Y on CPU and transfer batches to GPU as needed.
            This prevents OOM when Y is large (e.g., 300k voxels).
            Defaults to True (recommended for neuroimaging).
        score_func: Scoring function (y_true, y_pred) -> scores.
            If None, uses R² score. Defaults to None.
        fit_intercept: Whether to fit an intercept. If False, X and Y should be centered.
            Defaults to False.
        progress_bar: Whether to display progress bar (requires tqdm).
            Defaults to False.
        conservative: If True, select largest alpha within 1 std of best score.
            Defaults to False.
        parallel: Backend to use: "cpu", "gpu", or None.
            Defaults to "cpu".
        max_gpu_memory_gb: GPU memory budget in GB (only used if parallel="gpu").
            Defaults to 4.0.
        random_state: Random generator seed. Use an int for deterministic search.
            Defaults to None.

    Returns:
        dict: Dictionary with keys:
            - 'best_alphas': Selected best alpha for each target (or same alpha repeated
                if local_alpha=False), shape (n_targets,).
            - 'coefs': Ridge coefficients refit on entire dataset using best alphas,
                shape (n_features, n_targets). Always returned on CPU (numpy array).
            - 'cv_scores': Cross-validation scores for best alphas, shape (n_splits, n_alphas, n_targets).
                Always returned on CPU (numpy array).
            - 'intercept': Per-target intercept of shape (n_targets,). Only present
                when ``fit_intercept=True``.
            - 'backend': Backend used (for transparency).

    Examples:
        >>> X = np.random.randn(100, 50)
        >>> Y = np.random.randn(100, 10)
        >>> result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
        >>> alphas = result['best_alphas']
        >>> coefs = result['coefs']
        >>> scores = result['cv_scores']

    Notes:
        This is the efficient implementation for single feature space ridge regression
        with cross-validation. For multiple feature spaces (banded/group ridge),
        use solve_banded_ridge_cv instead.

        Algorithm details:

        - Cross-validation: k-fold CV evaluates each alpha value
        - Alpha selection: Chooses best alpha per target (or globally if local_alpha=False)
        - Refit: Fits final model on full dataset using best alpha(s)

        Memory efficiency strategies (Principle 2: automatic memory efficiency):

        - Generator pattern for alpha batching (via _decompose_ridge): Processes alphas
          in batches to avoid storing all resolution matrices simultaneously
        - Target batching (n_targets_batch): Processes targets in chunks to fit GPU memory
        - Y_in_cpu strategy: Keeps large Y on CPU, transfers only batches needed
          for computation
        - Immediate cleanup with del statements: Explicitly frees memory after each batch

        Performance:

        - Time complexity: O(n_splits × (n_alphas_batch × n_features^2 + n_targets_batch × n_samples))
        - Memory complexity: O(n_features × n_targets_batch) per batch
        - GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)

        See ``nltools.algorithms.ridge.utils._decompose_ridge()`` for generator pattern details.
        See ``nltools.algorithms.ridge.DESIGN.md`` for detailed algorithm explanation.
    """
    from .utils import _decompose_ridge, _select_best_alphas, _r2_score

    backend = resolve_backend(parallel)
    xp = backend.xp

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
        X_offset = xp.mean(X, axis=0)
        Y_offset = xp.mean(Y, axis=0)
        X = X - X_offset[None, :]
        Y = Y - Y_offset[None, :]

    # Set batch sizes. When running on GPU with no explicit target batch,
    # derive one from max_gpu_memory_gb so the budget actually bounds
    # allocation instead of processing all targets at once.
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)
    if n_targets_batch is None:
        n_targets_batch = (
            _auto_n_targets_batch(
                max_gpu_memory_gb, n_alphas_batch * n_samples, n_targets
            )
            if parallel == "gpu"
            else n_targets
        )
    if n_targets_batch_refit is None:
        n_targets_batch_refit = n_targets_batch

    # Set score function
    if score_func is None:
        score_func = lambda y_true, y_pred: _r2_score(y_true, y_pred, backend=backend)

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
        # Get train/val splits — keep CPU copies for indexing CPU-resident Y
        train_idx_cpu = train_idx
        val_idx_cpu = val_idx
        train_idx = backend.asarray(train_idx)
        val_idx = backend.asarray(val_idx)

        X_train = X[train_idx]
        X_val = X[val_idx]

        # Handle intercept per fold
        if fit_intercept:
            X_train_mean = xp.mean(X_train, axis=0)
            X_train = X_train - X_train_mean[None, :]
            X_val = X_val - X_train_mean[None, :]

        # Generator: batch over alphas
        # _decompose_ridge yields (resolution_matrices, alpha_indices) pairs
        # This avoids storing all resolution matrices simultaneously (memory efficient)
        for matrices, alpha_batch in _decompose_ridge(
            X_train,
            alphas,
            n_alphas_batch=n_alphas_batch,
            backend=backend,
        ):
            # Compute X_val @ matrices for predictions
            # matrices shape: (n_alphas_batch, n_features, n_train_samples)
            # pred_matrix shape: (n_alphas_batch, n_val_samples, n_train_samples)
            pred_matrix = backend.matmul(X_val, matrices)

            # Batch over targets
            for start in range(0, n_targets, n_targets_batch):
                batch = slice(start, start + n_targets_batch)

                # Get Y batches — use CPU indices when Y is on CPU
                y_train = train_idx_cpu if Y_in_cpu else train_idx
                y_val = val_idx_cpu if Y_in_cpu else val_idx
                Y_train_batch = Y[:, batch][y_train]
                Y_val_batch = Y[:, batch][y_val]

                if Y_in_cpu:
                    Y_train_batch = backend.to_gpu(Y_train_batch, device=device)
                    Y_val_batch = backend.to_gpu(Y_val_batch, device=device)

                # Handle intercept per fold
                if fit_intercept:
                    Y_train_mean = xp.mean(Y_train_batch, axis=0)
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
        scores, alphas, local_alpha, backend=backend, conservative=conservative
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
        n_targets_batch=n_targets_batch_refit,
        n_alphas_batch=n_alphas_batch,
        Y_in_cpu=Y_in_cpu,
        backend=backend,
    )

    # Convert to numpy
    best_alphas = backend.to_numpy(best_alphas)
    coefs = backend.to_numpy(coefs)
    scores = backend.to_numpy(scores)

    # Return dict with consistent keys
    result = {
        "best_alphas": best_alphas,
        "coefs": coefs,
        "cv_scores": scores,
        "backend": backend.name,
    }
    # Mirror solve_banded_ridge_cv: when intercept was fit, return the
    # per-target intercept derived from the same X/Y means used for
    # centering. Solver owns intercept calculation; callers must not
    # recompute it from the original (un-centered) data.
    if fit_intercept:
        result["intercept"] = (
            backend.to_numpy(Y_offset) - backend.to_numpy(X_offset) @ coefs
        )
    return result


def cross_val_predict_ridge(
    # Required
    X: np.ndarray,
    Y: np.ndarray,
    *,
    alphas: float | np.ndarray,
    cv: int | BaseCrossValidator = 5,
    fit_intercept: bool = False,
    n_targets_batch: int | None = None,
    n_alphas_batch: int | None = None,
    Y_in_cpu: bool = True,
    score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    # Backend parameters (grouped) — same vocabulary as solve_ridge_cv
    parallel: str | None = "cpu",
    max_gpu_memory_gb: float = 4.0,
) -> dict[str, Any]:
    """Held-out ridge predictions per CV fold under a (per-target) alpha.

    For each fold, refits ridge with the supplied alpha (per-target or
    scalar) on the training fold and predicts the held-out fold. Targets
    sharing the same alpha share an SVD of the training fold via
    `_refit_banded_ridge`, so the cost scales with the number of
    *unique* alphas, not the number of targets.

    Designed to be the BrainData CV layer's source of held-out predictions
    when alpha selection has already been done by ``solve_ridge_cv``: pass
    the selected per-voxel alphas back through here to get the fold-by-fold
    predictions and per-fold R² needed for ``cv_results_``.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        Y: Target data of shape (n_samples, n_targets). 1D ``Y`` is
            promoted to (n_samples, 1).
        alphas: Per-target alpha array of shape (n_targets,) or a scalar
            (broadcast to every target).
        cv: Cross-validation strategy. If int, uses KFold with that many
            splits (no shuffling). Generators (e.g. ``KFold(5).split(X)``)
            are rejected — pass the splitter object instead.
        fit_intercept: If True, center X and Y on the *training fold's*
            mean per fold (sklearn convention) and add the intercept back
            so predictions live on the original Y scale.
        n_targets_batch: Batch size for targets during refit (for memory
            efficiency). If None, processes all targets at once.
        n_alphas_batch: Batch size for alphas. If None, processes all
            unique alphas at once.
        Y_in_cpu: If True, keep Y on CPU and transfer batches to backend
            device as needed (recommended for large neuroimaging Y).
        score_func: Per-fold scoring function ``(y_true, y_pred) -> per-target
            scores``. If None, uses R² in NumPy on CPU (cheap at one fold's
            size and decoupled from backend ops to avoid stray transfers).
        parallel: Backend to use: "cpu", "gpu", or None.
        max_gpu_memory_gb: GPU memory budget in GB (only used if
            parallel="gpu").

    Returns:
        dict: Dictionary with keys:
            - 'predictions': (n_samples, n_targets) held-out per-target
              predictions on the original Y scale (CPU numpy).
            - 'folds': (n_samples,) int fold index per row (CPU numpy).
            - 'scores': (n_splits, n_targets) per-fold R² (or
              ``score_func``) at the supplied alpha (CPU numpy).
            - 'backend': Backend used (for transparency).

    Raises:
        TypeError: If ``cv`` is a single-use generator. Pass the splitter
            object instead.
        ValueError: If ``alphas`` does not broadcast to ``(n_targets,)``,
            or ``n_samples`` of X and Y disagree.
    """
    # Reject single-use generators — same reason ridge_cv rejects them:
    # the splitter must be re-iterable across folds.
    is_splitter = hasattr(cv, "split") and hasattr(cv, "get_n_splits")
    if hasattr(cv, "__next__") and not is_splitter:
        raise TypeError(
            "Got a generator for `cv` (e.g. `splitter.split(X, ...)`). "
            "Pass an sklearn CV splitter object instead — KFold(...), "
            "GroupKFold(...), etc. — so this function can iterate it more "
            "than once."
        )

    backend = resolve_backend(parallel)
    xp = backend.xp

    # Coerce X onto backend
    X = backend.asarray(X)
    n_samples, n_features = X.shape
    dtype = X.dtype
    device = getattr(X, "device", None)

    # Normalize Y to 2D
    Y_np = np.asarray(Y)
    if Y_np.ndim == 1:
        Y_np = Y_np[:, None]
    n_samples_y, n_targets = Y_np.shape
    if n_samples != n_samples_y:
        raise ValueError(f"n_samples mismatch: X has {n_samples}, Y has {n_samples_y}")

    # Per-target alpha vector
    alphas_per_target = np.asarray(alphas, dtype=np.float64)
    if alphas_per_target.ndim == 0:
        alphas_per_target = np.full(n_targets, float(alphas_per_target))
    if alphas_per_target.shape != (n_targets,):
        raise ValueError(
            f"alphas must be scalar or shape (n_targets,)={n_targets}; "
            f"got shape {alphas_per_target.shape}"
        )

    # Y onto backend (CPU or device, controlled by Y_in_cpu)
    Y_b = backend.asarray(Y_np, dtype=dtype, device="cpu" if Y_in_cpu else device)

    # Resolve batch sizes. On GPU with no explicit target batch, derive one
    # from max_gpu_memory_gb so the budget actually bounds allocation.
    unique_alphas = np.unique(alphas_per_target)
    if n_alphas_batch is None:
        n_alphas_batch = len(unique_alphas)
    if n_targets_batch is None:
        n_targets_batch = (
            _auto_n_targets_batch(
                max_gpu_memory_gb, n_alphas_batch * n_samples, n_targets
            )
            if parallel == "gpu"
            else n_targets
        )

    # Setup CV
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=False)
    n_splits = cv.get_n_splits()

    # CPU output buffers (callers downstream are numpy-only)
    pred_dtype = np.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
    if pred_dtype.kind != "f":
        pred_dtype = np.float32
    predictions = np.zeros((n_samples, n_targets), dtype=pred_dtype)
    folds = np.zeros(n_samples, dtype=int)
    scores = np.zeros((n_splits, n_targets), dtype=pred_dtype)

    for split_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        train_idx_cpu = train_idx
        val_idx_cpu = val_idx
        train_idx_b = backend.asarray(train_idx)
        val_idx_b = backend.asarray(val_idx)

        X_train = X[train_idx_b]
        X_val = X[val_idx_b]

        # Per-fold centering on the *training* fold (sklearn convention).
        if fit_intercept:
            X_train_mean = xp.mean(X_train, axis=0)
            X_train_c = X_train - X_train_mean[None, :]
            X_val_c = X_val - X_train_mean[None, :]
        else:
            X_train_c = X_train
            X_val_c = X_val

        # Pull this fold's Y_train onto the same device as X — _refit_banded_ridge
        # will see Y_in_cpu=False and operate on backend tensors throughout.
        if Y_in_cpu:
            Y_train_dev = backend.to_gpu(Y_b[train_idx_cpu], device=device)
        else:
            Y_train_dev = Y_b[train_idx_b]

        if fit_intercept:
            Y_train_mean = xp.mean(Y_train_dev, axis=0)
            Y_train_c = Y_train_dev - Y_train_mean[None, :]
        else:
            Y_train_mean = None
            Y_train_c = Y_train_dev

        # Refit per unique alpha, on the chosen backend. _refit_banded_ridge
        # handles target/alpha batching and returns coefs on CPU (n_features,
        # n_targets).
        coefs_cpu = _refit_banded_ridge(
            X=X_train_c,
            Y=Y_train_c,
            best_alphas=alphas_per_target,
            n_targets_batch=n_targets_batch,
            n_alphas_batch=n_alphas_batch,
            Y_in_cpu=False,  # Y_train_c lives on backend already
            backend=backend,
        )

        # Compute predictions on the backend, then bring to CPU.
        coefs_dev = backend.asarray(coefs_cpu, dtype=dtype, device=device)
        pred_dev = backend.matmul(X_val_c, coefs_dev)
        if fit_intercept:
            pred_dev = pred_dev + Y_train_mean[None, :]
        pred_cpu = backend.to_numpy(pred_dev)

        # Score per target on this fold. Doing R² on CPU at one fold's size
        # is negligible and avoids needing a backend-aware mean broadcast.
        Y_val_cpu = (
            np.asarray(Y_b[val_idx_cpu])
            if Y_in_cpu
            else backend.to_numpy(Y_b[val_idx_b])
        )
        if score_func is None:
            ss_res = np.sum((Y_val_cpu - pred_cpu) ** 2, axis=0)
            ss_tot = np.sum((Y_val_cpu - Y_val_cpu.mean(axis=0)) ** 2, axis=0)
            scores[split_idx] = 1.0 - ss_res / (ss_tot + 1e-10)
        else:
            scores[split_idx] = np.asarray(score_func(Y_val_cpu, pred_cpu))

        predictions[val_idx_cpu] = pred_cpu
        folds[val_idx_cpu] = split_idx

        del X_train, X_val, X_train_c, X_val_c, Y_train_dev, Y_train_c
        del coefs_cpu, coefs_dev, pred_dev, pred_cpu, Y_val_cpu

    return {
        "predictions": predictions,
        "folds": folds,
        "scores": scores,
        "backend": backend.name,
    }
