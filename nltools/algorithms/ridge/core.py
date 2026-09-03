"""Ridge regression solvers based on the singular value decomposition.

Ridge regression solves `beta = (X.T @ X + alpha * I)^(-1) @ X.T @ y`. Writing
`X = U @ diag(s) @ V.T` turns this into
`beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y`, which avoids an explicit matrix
inverse, is numerically stable for rank-deficient `X`, and costs
`O(n_samples × n_features × min(n_samples, n_features))` once for any number of
targets.

**Backends.** NumPy (the default) is reliable everywhere; PyTorch on the CPU
performs similarly; PyTorch on a GPU is roughly 10-100× faster for large problems
(`n_features` above ~10K). For cross-validated alpha selection with GPU batching
use `solve_ridge_cv`; the six tricks behind the solvers are described in
`docs/development/ridge-internals.md`.

Inspired by the himalaya library's SVD-based ridge regression (BSD-3-Clause,
https://github.com/gallantlab/himalaya).

References:
    Huth, A. G., et al. (2016). Natural speech reveals the semantic maps that tile
    human cerebral cortex. Nature, 532(7600), 453-458.

    himalaya documentation: https://gallantlab.github.io/himalaya/
"""

from typing import TYPE_CHECKING

import numpy as np

from nltools.algorithms.backends import resolve_backend

if TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator


def ridge_svd(
    # Required
    X: np.ndarray,
    y: np.ndarray,
    *,
    # Optional algorithm parameters
    alpha: float = 1.0,
    # Backend parameters (grouped)
    parallel: str | None = None,
    max_gpu_memory_gb: float | None = None,
    # Random state (last) - not used but kept for consistency
    random_state: int | None = None,
) -> np.ndarray:
    """Solve ridge regression for one alpha using the singular value decomposition.

    With `X = U @ diag(s) @ V.T` the solution is
    `beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y`; the shrinkage factor
    `s / (s**2 + alpha)` damps small singular values without an explicit matrix
    inverse. Time is `O(n_samples × n_features × min(n_samples, n_features))`
    and memory `O(n_samples × n_features)`. As `alpha → 0` this approaches
    ordinary least squares; use `alpha=1e-6` rather than 0 for a stable OLS fit.
    For cross-validated alpha selection use `solve_ridge_cv`.

    Args:
        X (np.ndarray): Training features, shape (n_samples, n_features).
        y (np.ndarray): Targets, shape (n_samples,) for a single target or
            (n_samples, n_targets) for several.
        alpha (float): Regularization strength; must be non-negative. Larger
            values shrink the coefficients harder toward zero. Defaults to 1.0.
        parallel (str | None): Execution backend. `None` or `"cpu"` runs on NumPy;
            `"gpu"` requires a CUDA or MPS accelerator; `"auto"` may use a
            Torch CPU backend when no accelerator is available. Defaults to None.
        max_gpu_memory_gb (float | None): GPU memory budget in GB for batching
            over targets (torch backends only). None measures the device.
            Defaults to None.
        random_state (int | None): Unused; accepted for signature consistency.
            Defaults to None.

    Returns:
        np.ndarray: Coefficients, shape (n_features,) for a single target or
            (n_features, n_targets) for several.

    Raises:
        ValueError: If `alpha` is negative, `X` is not 2D, `y` is not 1D or 2D,
            or the sample counts differ.

    Examples:
        ```python
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        ridge_svd(X, y, alpha=1.0).shape  # → (50,)

        Y = np.random.randn(100, 5)  # multi-target
        ridge_svd(X, Y, alpha=1.0).shape  # → (50, 5)
        ```
    """
    # Input validation
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Convert parallel ('cpu'/'gpu'/None) or pass-through Backend instance.
    # For graceful GPU-to-CPU fallback when torch is missing, use parallel='auto'.
    backend = resolve_backend(parallel)

    # Check dimensions
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim not in [1, 2]:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same n_samples. Got X: {X.shape[0]}, y: {y.shape[0]}"
        )

    # Determine if single or multi-target
    single_target = y.ndim == 1
    if single_target:
        y = y[:, np.newaxis]  # Convert to 2D for uniform processing

    n_samples, n_features = X.shape

    # Transfer to device
    X_device = backend.to_device(X)
    y_device = backend.to_device(y)

    # Compute SVD: X = U @ diag(s) @ Vt
    # Use reduced SVD (full_matrices=False) for efficiency
    # Reduced SVD only computes min(n_samples, n_features) singular values
    U, s, Vt = backend.svd(X_device, full_matrices=False)

    # Compute ridge shrinkage: s / (s**2 + alpha)
    # This is the key step that regularizes the solution
    # Small singular values get heavily shrunk, large ones less so
    # This prevents overfitting in high-dimensional settings
    if backend.name == "numpy":
        shrinkage = s / (s**2 + alpha)
        # Compute: beta = V @ diag(shrinkage) @ U.T @ y
        # V is Vt.T, so: beta = Vt.T @ diag(shrinkage) @ U.T @ y
        Uty = U.T @ y_device  # Shape: (rank, n_targets)
        coef = Vt.T @ (shrinkage[:, np.newaxis] * Uty)  # Broadcasting
    else:
        # PyTorch backend. Batch over target columns so max_gpu_memory_gb
        # bounds the per-batch GPU allocation (the target-scaling tensors are
        # Uty, the shrunk product, and the coef block). Small problems resolve
        # to a single batch and are identical to the unbatched path.
        from .utils import _auto_n_targets_batch

        shrinkage = s / (s**2 + alpha)
        shrink_col = shrinkage[:, None]
        n_targets = y_device.shape[1]
        rank = s.shape[0]
        n_targets_batch = _auto_n_targets_batch(
            max_gpu_memory_gb, rank + n_features + n_samples, n_targets
        )
        if n_targets_batch >= n_targets:
            Uty = backend.matmul(U.T, y_device)
            coef = backend.matmul(Vt.T, shrink_col * Uty)
        else:
            parts = []
            for start in range(0, n_targets, n_targets_batch):
                y_batch = y_device[:, start : start + n_targets_batch]
                Uty_batch = backend.matmul(U.T, y_batch)
                parts.append(backend.matmul(Vt.T, shrink_col * Uty_batch))
            coef = backend.concatenate(parts, axis=1)

    # Transfer back to NumPy
    coef = backend.to_numpy(coef)

    # Return to original shape for single-target
    if single_target:
        coef = coef.squeeze()

    return coef


def ridge_cv(
    # Required
    X: np.ndarray,
    y: np.ndarray,
    *,
    # Optional algorithm parameters
    alphas: np.ndarray | None = None,
    cv: "int | BaseCrossValidator" = 5,  # noqa: F821  (forward ref)
    fit_intercept: bool = False,
    # Backend parameters (grouped)
    parallel: str | None = "cpu",
    max_gpu_memory_gb: float | None = None,
    # Random state (last)
    random_state: int | None = None,
) -> dict:
    """Ridge regression with cross-validated selection of a single global alpha.

    Scores every alpha by out-of-fold R² on each fold, picks the alpha with the
    highest mean R² across folds and targets, then refits on all the data with
    it. For per-target alphas, memory-bounded batching, and GPU-batched folds
    use `solve_ridge_cv`.

    Args:
        X (np.ndarray): Training features, shape (n_samples, n_features).
        y (np.ndarray): Targets, shape (n_samples,) or (n_samples, n_targets).
        alphas (np.ndarray | None): Alpha values to try. None uses
            `np.logspace(-2, 4, 20)` (0.01 to 10000). Defaults to None.
        cv (int | BaseCrossValidator): Number of folds, or an sklearn
            cross-validator (anything with `.split(X)` and `.get_n_splits()`,
            e.g. `KFold(5, shuffle=True)` or `GroupKFold(8)`). The splitter
            drives the actual fold iteration, so leave-one-run-out and shuffled
            K-fold give different results from contiguous K-fold. Defaults to 5.
        fit_intercept (bool): If True, center `X` and `y` on their means before
            fitting and recover the intercept afterwards. The returned `coef` is
            on the centered scale; the intercept is returned under the
            `'intercept'` key. Defaults to False.
        parallel (str | None): Execution backend. `None` or `"cpu"` runs on NumPy;
            `"gpu"` requires a CUDA or MPS accelerator; `"auto"` may use a
            Torch CPU backend when no accelerator is available. Defaults to
            `"cpu"`.
        max_gpu_memory_gb (float | None): GPU memory budget in GB for batching
            over targets (torch backends only). None measures the device.
            Defaults to None.
        random_state (int | None): Unused; accepted for signature consistency.
            Defaults to None.

    Returns:
        dict: Keys `'alpha'` (float, the selected alpha), `'coef'` (np.ndarray,
            coefficients refit on all data with that alpha), `'cv_scores'`
            (np.ndarray, out-of-fold R² with shape (n_folds, n_alphas,
            n_targets)), `'backend'` (str, backend name), and — only when
            `fit_intercept=True` — `'intercept'` (float or np.ndarray).

    Raises:
        TypeError: If `cv` is a generator rather than a re-iterable splitter.

    Examples:
        ```python
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        result = ridge_cv(X, y, cv=3)
        result["alpha"]  # → the selected alpha
        result["coef"].shape  # → (50,)
        ```
    """
    from sklearn.model_selection import check_cv

    # Default alphas: logarithmic range from 0.01 to 10000
    if alphas is None:
        alphas = np.logspace(-2, 4, 20)
    else:
        alphas = np.asarray(alphas)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Convert parallel ('cpu'/'gpu'/None) or pass-through Backend instance.
    backend = resolve_backend(parallel)

    # Determine if single or multi-target
    single_target = y.ndim == 1
    if single_target:
        y = y[:, np.newaxis]

    # sklearn-style centering for fit_intercept. Centering happens *before*
    # the CV split so each training fold sees data shifted by the global
    # mean. solve_ridge_cv uses per-fold centering (more correct in tiny
    # samples) but we keep this simpler form here — the difference is
    # negligible on neuroimaging-sized data.
    if fit_intercept:
        X_offset = X.mean(axis=0)
        y_offset = y.mean(axis=0)
        X = X - X_offset
        y = y - y_offset

    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    n_alphas = len(alphas)

    # Resolve cv to an sklearn splitter. ``check_cv`` accepts int or any
    # sklearn-compatible splitter; everything else (e.g. a generator from
    # ``splitter.split(X)``) blows up here with a clearer error than
    # AttributeError later.
    if hasattr(cv, "__next__") and not hasattr(cv, "split"):
        raise TypeError(
            "ridge_cv received a generator for `cv`. Pass an sklearn CV "
            "splitter object (e.g. KFold(5, shuffle=True), GroupKFold(8)) "
            "rather than the result of `splitter.split(X, ...)` — the "
            "splitter must be re-iterable across alphas."
        )
    cv_splitter = check_cv(cv) if isinstance(cv, int) else cv
    n_splits = cv_splitter.get_n_splits()

    # Initialize CV scores array: (n_splits, n_alphas, n_targets)
    cv_scores = np.zeros((n_splits, n_alphas, n_targets))

    # Perform cross-validation using whatever splitter the caller passed —
    # this is the fix that makes shuffled K-fold and leave-one-run-out
    # actually do what the user asked.
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for i, alpha in enumerate(alphas):
            coef = ridge_svd(
                X_train,
                y_train,
                alpha=alpha,
                parallel=backend,
                max_gpu_memory_gb=max_gpu_memory_gb,
            )
            if coef.ndim == 1:
                coef = coef[:, np.newaxis]
            y_pred = X_test @ coef
            for t in range(n_targets):
                ss_res = np.sum((y_test[:, t] - y_pred[:, t]) ** 2)
                ss_tot = np.sum((y_test[:, t] - y_test[:, t].mean()) ** 2)
                cv_scores[fold, i, t] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Select best alpha: maximize mean R**2 across folds and targets
    mean_scores = cv_scores.mean(axis=(0, 2))
    best_idx = np.argmax(mean_scores)
    best_alpha = alphas[best_idx]

    # Fit final model on full (centered, if applicable) data
    coef_final = ridge_svd(
        X,
        y,
        alpha=best_alpha,
        parallel=backend,
        max_gpu_memory_gb=max_gpu_memory_gb,
    )

    # Recover intercept on the original (uncentered) scale
    intercept = None
    if fit_intercept:
        coef_2d = coef_final if coef_final.ndim == 2 else coef_final[:, np.newaxis]
        intercept = y_offset - X_offset @ coef_2d

    # Return to original shape for single-target
    if single_target:
        coef_final = coef_final.squeeze()
        if intercept is not None:
            intercept = float(np.asarray(intercept).squeeze())

    result = {
        "alpha": float(best_alpha),
        "coef": coef_final,
        "cv_scores": cv_scores,
        "backend": backend.name,
    }
    if intercept is not None:
        result["intercept"] = intercept
    return result
