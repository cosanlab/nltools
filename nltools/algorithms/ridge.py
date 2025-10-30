"""
Ridge regression algorithms using SVD decomposition.

Inspired by the himalaya library's efficient SVD-based ridge regression approach.
himalaya is licensed under BSD-3-Clause: https://github.com/gallantlab/himalaya

References
----------
- Huth, A. G., et al. (2016). "Natural speech reveals the semantic maps that tile
  human cerebral cortex." Nature, 532(7600), 453-458.
- himalaya documentation: https://gallantlab.github.io/himalaya/
"""

import numpy as np
from typing import Union, Optional
from nltools.backends import Backend, auto_select_backend


def ridge_svd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    backend: Optional[Union[Backend, str]] = None
) -> np.ndarray:
    """
    Solve ridge regression using Singular Value Decomposition.

    This function implements ridge regression using SVD, which provides
    numerical stability and efficiency for high-dimensional problems.
    The implementation is inspired by the himalaya library.

    The ridge regression solution is:
        beta = (X.T @ X + alpha*I)^(-1) @ X.T @ y

    Using SVD of X = U @ diag(s) @ V.T, this becomes:
        beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y

    This formulation avoids explicit matrix inversion and is numerically stable.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training data features
    y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
        Target values. Can be 1D for single-target or 2D for multi-target
    alpha : float, default=1.0
        Regularization strength. Must be positive. Higher values
        increase regularization (shrink coefficients toward zero)
    backend : Backend or str, optional
        Backend for computation ('numpy', 'torch', 'auto', or Backend instance).
        If None, uses NumPy. If 'auto', selects based on problem size.

    Returns
    -------
    coef : np.ndarray
        Ridge regression coefficients
        - shape (n_features,) for single-target regression
        - shape (n_features, n_targets) for multi-target regression

    Examples
    --------
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

    Notes
    -----
    - Time complexity: O(n_samples * n_features * min(n_samples, n_features))
    - Space complexity: O(n_samples * n_features)
    - For alpha→0, this reduces to ordinary least squares (OLS). Use alpha=1e-6
      for OLS in practice (more numerically stable than alpha=0)
    - Supports both CPU (NumPy) and GPU (PyTorch) backends
    """
    # Input validation
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Handle backend
    if backend is None:
        backend = Backend('numpy')
    elif isinstance(backend, str):
        if backend == 'auto':
            n_samples, n_features = X.shape
            backend = auto_select_backend(n_samples, n_features)
        else:
            backend = Backend(backend)

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
    single_target = (y.ndim == 1)
    if single_target:
        y = y[:, np.newaxis]  # Convert to 2D for uniform processing

    n_samples, n_features = X.shape
    n_targets = y.shape[1]

    # Transfer to device
    X_device = backend.to_device(X)
    y_device = backend.to_device(y)

    # Compute SVD: X = U @ diag(s) @ Vt
    # Use reduced SVD (full_matrices=False) for efficiency
    U, s, Vt = backend.svd(X_device, full_matrices=False)

    # Compute ridge shrinkage: s / (s**2 + alpha)
    # This is the key step that regularizes the solution
    if backend.name == 'numpy':
        shrinkage = s / (s**2 + alpha)
        # Compute: beta = V @ diag(shrinkage) @ U.T @ y
        # V is Vt.T, so: beta = Vt.T @ diag(shrinkage) @ U.T @ y
        Uty = U.T @ y_device  # Shape: (rank, n_targets)
        coef = Vt.T @ (shrinkage[:, np.newaxis] * Uty)  # Broadcasting
    else:
        # PyTorch backend
        import torch
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
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    cv: int = 5,
    backend: Union[str, Backend] = 'auto'
) -> dict:
    """
    Ridge regression with cross-validation for hyperparameter selection.

    Performs k-fold cross-validation to select the best alpha parameter,
    then fits a final model on all data using the selected alpha.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training data features
    y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
        Target values
    alphas : np.ndarray, optional
        Array of alpha values to try. If None, uses default range:
        np.logspace(-2, 4, 20) = [0.01, 0.015, ..., 10000]
    cv : int, default=5
        Number of cross-validation folds
    backend : str or Backend, default='auto'
        Backend for computation. 'auto' selects based on problem size.

    Returns
    -------
    result : dict
        Dictionary containing:

        - 'alpha' : float
            Best alpha value selected by CV
        - 'coef' : np.ndarray
            Coefficients using best alpha on full dataset
        - 'cv_scores' : np.ndarray, shape (n_folds, n_alphas, n_targets)
            Cross-validation R**2 scores for each fold, alpha, and target
        - 'backend' : str
            Backend used for computation

    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> result = ridge_cv(X, y, cv=3)
    >>> result['alpha']  # Best alpha selected
    1.0
    >>> result['coef'].shape
    (50,)

    Notes
    -----
    - Uses R**2 (coefficient of determination) as the scoring metric
    - For multi-target regression, selects alpha that maximizes mean R**2 across targets
    - Automatically uses GPU backend for large problems when available
    """
    # Default alphas: logarithmic range from 0.01 to 10000
    if alphas is None:
        alphas = np.logspace(-2, 4, 20)
    else:
        alphas = np.asarray(alphas)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Setup backend
    if isinstance(backend, str):
        if backend == 'auto':
            n_samples, n_features = X.shape
            backend = auto_select_backend(n_samples, n_features, cv=cv)
        else:
            backend = Backend(backend)

    # Determine if single or multi-target
    single_target = (y.ndim == 1)
    if single_target:
        y = y[:, np.newaxis]

    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    n_alphas = len(alphas)

    # Initialize CV scores array: (n_folds, n_alphas, n_targets)
    cv_scores = np.zeros((cv, n_alphas, n_targets))

    # Create fold indices
    fold_size = n_samples // cv
    indices = np.arange(n_samples)

    # Perform cross-validation
    for fold in range(cv):
        # Create train/test split
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < cv - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Try each alpha
        for i, alpha in enumerate(alphas):
            # Fit model
            coef = ridge_svd(X_train, y_train, alpha=alpha, backend=backend)

            # Predict and score
            # Note: y_train is already 2D (n, 1) for single target, so coef is (n_features, 1)
            if coef.ndim == 1:
                coef = coef[:, np.newaxis]

            y_pred = X_test @ coef

            # Calculate R**2 for each target
            for t in range(n_targets):
                ss_res = np.sum((y_test[:, t] - y_pred[:, t]) ** 2)
                ss_tot = np.sum((y_test[:, t] - y_test[:, t].mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                cv_scores[fold, i, t] = r2

    # Select best alpha: maximize mean R**2 across folds and targets
    mean_scores = cv_scores.mean(axis=(0, 2))  # Average over folds and targets
    best_idx = np.argmax(mean_scores)
    best_alpha = alphas[best_idx]

    # Fit final model on full data
    coef_final = ridge_svd(X, y, alpha=best_alpha, backend=backend)

    # Return to original shape for single-target
    if single_target:
        coef_final = coef_final.squeeze()

    return {
        'alpha': float(best_alpha),
        'coef': coef_final,
        'cv_scores': cv_scores,
        'backend': backend.name
    }
