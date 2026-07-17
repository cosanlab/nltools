"""Ridge regression algorithms using SVD decomposition.

This module implements ridge regression using Singular Value Decomposition (SVD),
which provides numerical stability and efficiency for high-dimensional problems.

Algorithm approach:
    Why SVD vs direct inversion:
        - Direct inversion: beta = (X.T @ X + alpha*I)^(-1) @ X.T @ y
        - SVD approach: X = U @ diag(s) @ V.T, then beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y
        - Benefits: Avoids explicit matrix inversion (numerically stable), efficient for rank-deficient X
        - Performance: O(n_samples × n_features × min(n_samples, n_features)) for SVD

Backend choice trade-offs:
    - NumPy (CPU): Default, reliable, works everywhere
    - PyTorch CPU: Similar performance to NumPy, useful for consistent API
    - PyTorch GPU: ~10-100× speedup for large problems (n_features > 10K), requires GPU

Cross-references:
    - See `nltools.algorithms.ridge.solvers.solve_ridge_cv()` for GPU-accelerated cross-validation
    - See `nltools.algorithms.ridge.utils._decompose_ridge()` for generator-based batching pattern
    - See `nltools.algorithms.ridge.DESIGN.md` for detailed algorithm explanation

Inspired by the himalaya library's efficient SVD-based ridge regression approach.
himalaya is licensed under BSD-3-Clause: https://github.com/gallantlab/himalaya

References:
    - Huth, A. G., et al. (2016). "Natural speech reveals the semantic maps that tile
      human cerebral cortex." Nature, 532(7600), 453-458.
    - himalaya documentation: https://gallantlab.github.io/himalaya/
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
    # Optional algorithm parameters
    alpha: float = 1.0,
    # Backend parameters (grouped)
    parallel: str | None = None,
    max_gpu_memory_gb: float = 4.0,
    # Random state (last) - not used but kept for consistency
    random_state: int | None = None,
) -> np.ndarray:
    """Solve ridge regression using Singular Value Decomposition.

    This function implements ridge regression using SVD, which provides
    numerical stability and efficiency for high-dimensional problems.
    The implementation is inspired by the himalaya library.

    Algorithm:
        The ridge regression solution is:
            beta = (X.T @ X + alpha*I)^(-1) @ X.T @ y

        Using SVD of X = U @ diag(s) @ V.T, this becomes:
            beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y

        This formulation avoids explicit matrix inversion and is numerically stable.
        The shrinkage factor s / (s**2 + alpha) regularizes small singular values.

    Performance:
        - Time complexity: O(n_samples × n_features × min(n_samples, n_features))
        - Space complexity: O(n_samples × n_features)
        - GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)
        - See `solve_ridge_cv()` for cross-validation with GPU support

    Args:
        X (np.ndarray): Training data features with shape (n_samples, n_features)
        y (np.ndarray): Target values with shape (n_samples,) or (n_samples, n_targets).
            Can be 1D for single-target or 2D for multi-target
        alpha (float, optional): Regularization strength. Must be positive. Higher values
            increase regularization (shrink coefficients toward zero). Defaults to 1.0.
        parallel (str, optional): Execution backend.
            - None: Single-threaded NumPy (debugging/small problems)
            - "cpu": CPU-only using NumPy (default)
            - "gpu": GPU acceleration via PyTorch. Requires torch installed
              (raises ImportError otherwise); degrades to torch-CPU only when no
              GPU device is present. Use "auto" for torch-optional CPU fallback.
            Defaults to None.
        max_gpu_memory_gb (float, optional): GPU memory budget in GB (only used if parallel='gpu').
            Defaults to 4.0.
        random_state (int, optional): Random seed (not currently used, kept for consistency).
            Defaults to None.

    Returns:
        np.ndarray: Ridge regression coefficients
            - shape (n_features,) for single-target regression
            - shape (n_features, n_targets) for multi-target regression

    Examples:
        >>> X = np.random.randn(100, 50)
        >>> y = np.random.randn(100)
        >>> beta = ridge_svd(X, y, alpha=1.0)
        >>> beta.shape
        (50,)

        >>> # Multi-target regression
        >>> Y = np.random.randn(100, 5)
        >>> beta = ridge_svd(X, Y, alpha=1.0)
        >>> beta.shape
        (50, 5)

    Notes:
        - Time complexity: O(n_samples * n_features * min(n_samples, n_features))
        - Space complexity: O(n_samples * n_features)
        - For alpha→0, this reduces to ordinary least squares (OLS). Use alpha=1e-6
          for OLS in practice (more numerically stable than alpha=0)
        - Supports both CPU (NumPy) and GPU (PyTorch) backends
        - See `nltools.algorithms.ridge.solvers.solve_ridge_cv()` for cross-validation
        - See `nltools.algorithms.ridge.utils._decompose_ridge()` for generator pattern
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
        # PyTorch backend
        shrinkage = s / (s**2 + alpha)
        Uty = backend.matmul(U.T, y_device)
        coef = backend.matmul(Vt.T, shrinkage[:, None] * Uty)

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
    # Optional algorithm parameters
    alphas: np.ndarray | None = None,
    cv: "int | BaseCrossValidator" = 5,  # noqa: F821  (forward ref)
    fit_intercept: bool = False,
    # Backend parameters (grouped)
    parallel: str | None = "cpu",
    max_gpu_memory_gb: float = 4.0,
    # Random state (last)
    random_state: int | None = None,
) -> dict:
    """Ridge regression with cross-validation for hyperparameter selection.

    Performs k-fold cross-validation to select the best alpha parameter,
    then fits a final model on all data using the selected alpha.

    Args:
        X (np.ndarray): Training data features with shape (n_samples, n_features)
        y (np.ndarray): Target values with shape (n_samples,) or (n_samples, n_targets)
        alphas (np.ndarray, optional): Array of alpha values to try. If None, uses default range:
            np.logspace(-2, 4, 20) = [0.01, 0.015, ..., 10000]
        cv (int or sklearn CV splitter, optional): Number of folds (int) or
            an sklearn cross-validator (anything with ``.split(X)`` and
            ``.get_n_splits()``, e.g. ``KFold(5, shuffle=True)`` or
            ``GroupKFold(8)``). Splitters are honored for the actual fold
            iteration, so leave-one-run-out and shuffled-K-fold give different
            results from contiguous K-fold. Defaults to 5.
        fit_intercept (bool, optional): If True, center X and y on the
            training mean before fitting and recover the intercept after.
            The returned ``coef`` is on the centered scale; the recovered
            intercept is returned under the ``intercept`` key. Defaults to
            False.
        parallel (str, optional): Execution backend.
            - None: Single-threaded NumPy (debugging/small problems)
            - "cpu": CPU-only using NumPy (default)
            - "gpu": GPU acceleration via PyTorch. Requires torch installed
              (raises ImportError otherwise); degrades to torch-CPU only when no
              GPU device is present. Use "auto" for torch-optional CPU fallback.
            Defaults to "cpu".
        max_gpu_memory_gb (float, optional): GPU memory budget in GB (only used if parallel='gpu').
            Defaults to 4.0.
        random_state (int, optional): Random seed (not currently used, kept for consistency).
            Defaults to None.

    Returns:
        dict: Dictionary containing:

            - 'alpha' (float): Best alpha value selected by CV
            - 'coef' (np.ndarray): Coefficients using best alpha on full dataset
            - 'cv_scores' (np.ndarray): Cross-validation R**2 scores for each fold, alpha, and target
                with shape (n_folds, n_alphas, n_targets)
            - 'backend' (str): Backend used for computation

    Examples:
        >>> X = np.random.randn(100, 50)
        >>> y = np.random.randn(100)
        >>> result = ridge_cv(X, y, cv=3)
        >>> result['alpha']  # Best alpha selected
        1.0
        >>> result['coef'].shape
        (50,)

    Notes:
        - Uses R**2 (coefficient of determination) as the scoring metric
        - For multi-target regression, selects alpha that maximizes mean R**2 across targets
        - parallel='gpu' requires torch installed; with torch present but no GPU device it
          runs on torch-CPU. It does not fall back to NumPy when torch is absent — use
          parallel='auto' for that.
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

    # Convert backend to parallel parameter for ridge_svd
    parallel_param = "gpu" if backend.name.startswith("torch") else "cpu"

    # Perform cross-validation using whatever splitter the caller passed —
    # this is the fix that makes shuffled K-fold and leave-one-run-out
    # actually do what the user asked.
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for i, alpha in enumerate(alphas):
            coef = ridge_svd(X_train, y_train, alpha=alpha, parallel=parallel_param)
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
    coef_final = ridge_svd(X, y, alpha=best_alpha, parallel=parallel_param)

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
