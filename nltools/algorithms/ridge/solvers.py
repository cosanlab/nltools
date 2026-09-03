"""Ridge regression solvers with cross-validation.

`solve_ridge_cv` selects alphas for a single feature space, `solve_banded_ridge_cv`
adds a random search over feature-space weights for several feature spaces, and
`cross_val_predict_ridge` returns held-out predictions once alphas are chosen.
All three batch over alphas and targets to bound memory, can keep a large `Y` on
the CPU and stream batches to the device (`Y_in_cpu`), select alphas per target
or globally, and run on NumPy or PyTorch (CPU, CUDA, MPS) through `parallel=`.
The design follows the himalaya library; `docs/development/ridge-internals.md`
explains the tricks.
"""

from __future__ import annotations

import numbers

import numpy as np
import warnings
from typing import Any
from collections.abc import Callable
from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.utils import check_random_state

from .utils import _auto_n_targets_batch
from ...utils import find_stack_level, maybe_tqdm

from ..backends import resolve_backend


def solve_banded_ridge_cv(
    # Required
    Xs: list[np.ndarray],
    Y: np.ndarray,
    *,
    # Optional algorithm parameters
    n_iter: int | np.integer | np.ndarray = 100,
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
    max_gpu_memory_gb: float | None = None,
    # Random state (last)
    random_state: int | None = None,
) -> dict[str, Any]:
    """Solve banded (group) ridge regression with cross-validated random search.

    Banded ridge gives each feature space its own scale: `Z_i = sqrt(gamma_i) * X_i`,
    then ordinary ridge is solved on the concatenated `Z`, so the relative
    importance of the feature spaces is learned. The weights `gamma` are drawn
    from a Dirichlet distribution (`n_iter` draws; the first is equal weights);
    for each draw every alpha is scored by k-fold cross-validation, and each
    target keeps the `(gamma, alpha)` pair with the best score (himalaya's
    `solve_group_ridge_random_search`). For a single feature space use
    `solve_ridge_cv`.

    **Memory.** Alphas are processed in batches of `n_alphas_batch` from one SVD
    per fold, targets in batches of `n_targets_batch`, and with `Y_in_cpu=True`
    only the current target batch is moved to the device. Time scales as
    `O(n_iter × n_splits × (n_alphas_batch × n_features² + n_targets_batch ×
    n_samples))`, memory as `O(n_features × n_targets_batch)` per batch; a GPU
    is roughly 10-100× faster once `n_features` exceeds ~10K.

    Args:
        Xs (list[np.ndarray]): One feature matrix per feature space, each of
            shape (n_samples, n_features_i) with the same `n_samples`.
        Y (np.ndarray): Targets of shape (n_samples, n_targets).
        n_iter (int | np.ndarray): Number of feature-space weight combinations
            to sample, or an explicit array of weights with shape
            (n_iter, n_spaces) to try instead of sampling. Defaults to 100.
        concentration (float | list[float]): Dirichlet concentration
            parameter(s). A value of 1 samples uniformly over the simplex;
            `np.inf` gives equal weights; a list alternates between its values.
            Ignored when `n_iter` is an array. Defaults to `[0.1, 1.0]`.
        alphas (float | np.ndarray | list[float]): Ridge regularization
            parameters to try. Defaults to `[0.1, 1.0, 10.0]`.
        cv (int | BaseCrossValidator): Number of folds for unshuffled `KFold`, or
            an sklearn cross-validator. Defaults to 5.
        local_alpha (bool): If True, pick the best alpha independently per
            target; if False, one alpha for all targets. Defaults to True.
        n_targets_batch (int | None): Targets per batch during CV. None processes
            all targets at once on the CPU; with `parallel="gpu"` None derives a
            batch size from `max_gpu_memory_gb`. Defaults to None.
        n_targets_batch_refit (int | None): Targets per batch during the refit.
            None reuses `n_targets_batch`. Defaults to None.
        n_alphas_batch (int | None): Alphas per batch. None processes all alphas
            at once. Defaults to None.
        Y_in_cpu (bool): If True, keep `Y` on the CPU and move one target batch
            at a time to the device, which avoids running out of GPU memory on
            large `Y` (e.g. 300k voxels). Defaults to True.
        score_func (Callable | None): Scoring function `(y_true, y_pred) ->
            per-target scores`. None uses R². Defaults to None.
        fit_intercept (bool): If True, center `X` and `Y` per training fold and
            return the intercept. If False, `X` and `Y` should already be
            centered. Defaults to False.
        progress_bar (bool): Show a progress bar over the gamma draws (requires
            tqdm). Defaults to False.
        conservative (bool): If True, pick the largest alpha within one standard
            deviation of the best score (more regularization at similar
            performance). Defaults to False.
        jitter_alphas (bool): If True, multiply the alpha grid by a random factor
            in `[10^-0.5, 10^0.5]` for each gamma draw. Defaults to False.
        return_weights (bool): If True, refit on the full data with the selected
            hyperparameters and return the coefficients. Defaults to True.
        diagonalize_method (str): Feature decomposition; only `"svd"` is
            supported. Defaults to `"svd"`.
        warn (bool): If True, warn when `n_samples < n_features`, where banded
            ridge is slower than kernel ridge. Defaults to True.
        parallel (str | None): Execution backend. `None` or `"cpu"` runs on NumPy;
            `"gpu"` requires a CUDA or MPS accelerator; `"auto"` may use a
            Torch CPU backend when no accelerator is available. Defaults to
            `"cpu"`.
        max_gpu_memory_gb (float | None): GPU memory budget in GB used to derive
            `n_targets_batch` when `parallel="gpu"`. None measures the device.
            Defaults to None.
        random_state (int | None): Random generator seed; use an int for a
            deterministic search. Defaults to None.

    Returns:
        dict: Keys `'deltas'` (np.ndarray, `log(gamma / alpha)` per feature space
            and target, shape (n_spaces, n_targets)), `'cv_scores'` (np.ndarray,
            split-averaged score at the selected alpha for each gamma draw,
            shape (n_iter, n_targets)), `'backend'` (str, backend name), and —
            only when `return_weights=True` — `'coefs'` (np.ndarray, refit
            coefficients on the unscaled features, shape (n_features_total,
            n_targets)) plus, when `fit_intercept=True` as well, `'intercept'`
            (np.ndarray, shape (n_targets,)). Arrays are always NumPy on the CPU.

    Raises:
        ValueError: If `Xs` is empty, the feature spaces or `Y` disagree on
            `n_samples`, or `n_iter` is neither an integer nor a 2D array with
            `n_spaces` columns.

    Examples:
        ```python
        X1 = np.random.randn(100, 30)  # first feature space
        X2 = np.random.randn(100, 20)  # second feature space
        Y = np.random.randn(100, 10)
        result = solve_banded_ridge_cv([X1, X2], Y, n_iter=50, alphas=[0.1, 1.0, 10.0])
        result["deltas"].shape  # → (2, 10)
        result["coefs"].shape  # → (50, 10)
        result["cv_scores"].shape  # → (50, 10)
        ```
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

    # Generate or use provided gammas. n_iter is dual-purpose: an integer count
    # (number of random Dirichlet feature-weight samples to search) OR an
    # explicit (n_iter, n_spaces) gamma array. numbers.Integral accepts numpy
    # integer scalars (np.int64) as well as Python int.
    if isinstance(n_iter, numbers.Integral):
        gammas = generate_dirichlet_samples(
            n_samples=int(n_iter),
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
            stacklevel=find_stack_level(),
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
            if backend.is_gpu
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

    progress_iter = maybe_tqdm(
        gammas,
        progress_bar=progress_bar,
        desc=f"{len(gammas)} random sampling with cv",
    )

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
                        # Indices (within used_alphas) covered by this alpha
                        # batch, on the compute device: torch.arange allocates
                        # on CPU while alphas_indices lives on the device, and
                        # torch.isin/searchsorted require same-device tensors.
                        # to_gpu is a no-op for the numpy backend.
                        batch_alpha_indices = backend.to_gpu(
                            xp.arange(len(used_alphas))[alpha_batch]
                        )
                        # Mask targets whose selected alphas are outside the alpha batch
                        mask2 = xp.isin(alphas_indices, batch_alpha_indices)
                        # Get indices in alpha_batch
                        alphas_indices = xp.searchsorted(
                            batch_alpha_indices,
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
                        del batch_alpha_indices

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
    """Refit ridge on the full data with a (possibly different) alpha per target.

    Targets sharing an alpha share one resolution matrix, so the cost scales
    with the number of unique alphas rather than the number of targets.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Y (np.ndarray): Target matrix of shape (n_samples, n_targets).
        best_alphas (np.ndarray): Selected alpha per target, shape (n_targets,).
        n_targets_batch (int): Targets per batch.
        n_alphas_batch (int): Alphas per batch.
        Y_in_cpu (bool): Whether `Y` lives on the CPU and batches must be moved
            to the device.
        backend (Backend): Resolved backend.

    Returns:
        np.ndarray: Coefficients on the CPU, shape (n_features, n_targets).
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
    max_gpu_memory_gb: float | None = None,
    # Random state (last)
    random_state: int | None = None,
) -> dict[str, Any]:
    """Solve ridge regression for one feature space with cross-validated alphas.

    Every alpha is scored by k-fold cross-validation, the best alpha is chosen
    per target (or once for all targets with `local_alpha=False`), and the model
    is refit on the full data with the chosen alphas. For several feature spaces
    use `solve_banded_ridge_cv`; for held-out predictions at already-selected
    alphas use `cross_val_predict_ridge`.

    **Memory.** Alphas are processed in batches of `n_alphas_batch` from one SVD
    per fold, targets in batches of `n_targets_batch`, and with `Y_in_cpu=True`
    only the current target batch is moved to the device. Time scales as
    `O(n_splits × (n_alphas_batch × n_features² + n_targets_batch × n_samples))`,
    memory as `O(n_features × n_targets_batch)` per batch; a GPU is roughly
    10-100× faster once `n_features` exceeds ~10K.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Y (np.ndarray): Targets of shape (n_samples, n_targets).
        alphas (float | np.ndarray | list[float]): Ridge regularization
            parameters to try. Defaults to `[0.1, 1.0, 10.0]`.
        cv (int | BaseCrossValidator): Number of folds for unshuffled `KFold`, or
            an sklearn cross-validator. Defaults to 5.
        local_alpha (bool): If True, pick the best alpha independently per
            target; if False, one alpha for all targets. Defaults to True.
        n_targets_batch (int | None): Targets per batch during CV. None processes
            all targets at once on the CPU; with `parallel="gpu"` None derives a
            batch size from `max_gpu_memory_gb`. Defaults to None.
        n_targets_batch_refit (int | None): Targets per batch during the refit.
            None reuses `n_targets_batch`. Defaults to None.
        n_alphas_batch (int | None): Alphas per batch. None processes all alphas
            at once. Defaults to None.
        Y_in_cpu (bool): If True, keep `Y` on the CPU and move one target batch
            at a time to the device, which avoids running out of GPU memory on
            large `Y` (e.g. 300k voxels). Defaults to True.
        score_func (Callable | None): Scoring function `(y_true, y_pred) ->
            per-target scores`. None uses R². Defaults to None.
        fit_intercept (bool): If True, center `X` and `Y` per training fold and
            return the intercept. If False, `X` and `Y` should already be
            centered. Defaults to False.
        progress_bar (bool): Accepted for API symmetry with
            `solve_banded_ridge_cv`; this solver shows no progress bar.
            Defaults to False.
        conservative (bool): If True, pick the largest alpha within one standard
            deviation of the best score (more regularization at similar
            performance). Defaults to False.
        parallel (str | None): Execution backend. `None` or `"cpu"` runs on NumPy;
            `"gpu"` requires a CUDA or MPS accelerator; `"auto"` may use a
            Torch CPU backend when no accelerator is available. Defaults to
            `"cpu"`.
        max_gpu_memory_gb (float | None): GPU memory budget in GB used to derive
            `n_targets_batch` when `parallel="gpu"`. None measures the device.
            Defaults to None.
        random_state (int | None): Unused by this solver (the search is
            deterministic); accepted for signature consistency. Defaults to None.

    Returns:
        dict: Keys `'best_alphas'` (np.ndarray, selected alpha per target — the
            same value repeated when `local_alpha=False` — shape (n_targets,)),
            `'coefs'` (np.ndarray, coefficients refit on the full data, shape
            (n_features, n_targets)), `'cv_scores'` (np.ndarray, per-fold score
            of every alpha, shape (n_splits, n_alphas, n_targets)), `'backend'`
            (str, backend name), and — only when `fit_intercept=True` —
            `'intercept'` (np.ndarray, shape (n_targets,)). Arrays are always
            NumPy on the CPU.

    Raises:
        ValueError: If `X` and `Y` disagree on `n_samples`.

    Examples:
        ```python
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)
        result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
        result["best_alphas"].shape  # → (10,)
        result["coefs"].shape  # → (50, 10)
        result["cv_scores"].shape  # → (5, 3, 10)
        ```
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
            if backend.is_gpu
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
    max_gpu_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Held-out ridge predictions per CV fold at a fixed (per-target) alpha.

    For each fold, ridge is refit on the training fold with the supplied alpha
    (scalar or per target) and the held-out fold is predicted. Targets sharing
    an alpha share one SVD of the training fold, so the cost scales with the
    number of *unique* alphas, not the number of targets.

    This is how `BrainData` obtains held-out predictions once `solve_ridge_cv`
    has selected alphas: pass the selected per-voxel alphas back through here to
    get the fold-by-fold predictions and per-fold R² for `cv_results_`.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Y (np.ndarray): Targets of shape (n_samples, n_targets); a 1D `Y` is
            promoted to (n_samples, 1).
        alphas (float | np.ndarray): Per-target alphas of shape (n_targets,), or
            a scalar broadcast to every target.
        cv (int | BaseCrossValidator): Number of folds for unshuffled `KFold`, or
            an sklearn cross-validator. Generators (e.g. `KFold(5).split(X)`)
            are rejected; pass the splitter object. Defaults to 5.
        fit_intercept (bool): If True, center `X` and `Y` on the training fold's
            means (sklearn convention) and add the intercept back so predictions
            are on the original `Y` scale. Defaults to False.
        n_targets_batch (int | None): Targets per batch during the refit. None
            processes all targets at once on the CPU; with `parallel="gpu"` None
            derives a batch size from `max_gpu_memory_gb`. Defaults to None.
        n_alphas_batch (int | None): Alphas per batch. None processes all unique
            alphas at once. Defaults to None.
        Y_in_cpu (bool): If True, keep `Y` on the CPU and move one fold's
            training targets at a time to the device (recommended for large
            neuroimaging `Y`). Defaults to True.
        score_func (Callable | None): Per-fold scoring function `(y_true, y_pred)
            -> per-target scores`, evaluated on NumPy arrays. None uses R²,
            computed in NumPy on the CPU. Defaults to None.
        parallel (str | None): Execution backend. `None` or `"cpu"` runs on NumPy;
            `"gpu"` requires a CUDA or MPS accelerator; `"auto"` may use a
            Torch CPU backend when no accelerator is available. Defaults to
            `"cpu"`.
        max_gpu_memory_gb (float | None): GPU memory budget in GB used to derive
            `n_targets_batch` when `parallel="gpu"`. None measures the device.
            Defaults to None.

    Returns:
        dict: Keys `'predictions'` (np.ndarray, held-out predictions on the
            original `Y` scale, shape (n_samples, n_targets)), `'folds'`
            (np.ndarray, fold index per row, shape (n_samples,)), `'scores'`
            (np.ndarray, per-fold R² or `score_func` output, shape (n_splits,
            n_targets)), and `'backend'` (str, backend name). Arrays are NumPy on
            the CPU.

    Raises:
        TypeError: If `cv` is a single-use generator rather than a splitter.
        ValueError: If `alphas` does not broadcast to `(n_targets,)`, or `X` and
            `Y` disagree on `n_samples`.
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
            if backend.is_gpu
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
