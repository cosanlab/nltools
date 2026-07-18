"""Ridge regression model for neuroimaging data.

Wraps nltools.algorithms.ridge with sklearn-compatible API.
Supports both regular ridge (single feature space) and banded ridge
(multiple feature spaces) with optional random search over feature weights.
"""

from __future__ import annotations

import numpy as np
from .base import BaseModel
from ..algorithms.ridge import ridge_svd
from ..algorithms.ridge.solvers import (
    solve_ridge_cv,
    solve_banded_ridge_cv,
)
from ..algorithms.backends import resolve_backend


class Ridge(BaseModel):
    """Ridge regression with optional GPU acceleration and banded ridge support.

    Wraps nltools SVD-based ridge regression algorithms with
    scikit-learn compatible API. Supports single and multi-target
    regression with optional GPU acceleration via PyTorch.

    Supports both regular ridge (single feature space) and banded ridge
    (multiple feature spaces). The model automatically detects the input type:

    - Array X: Single feature space → uses solve_ridge_cv
    - List X: Multiple feature spaces → uses solve_banded_ridge_cv (true banded/group ridge)

    Args:
        alpha (float or 'auto', default=1.0): Regularization strength. If 'auto',
            uses cross-validation to select optimal alpha from alphas parameter.
        cv (int or None, default=None): Number of cross-validation folds (only used
            if alpha='auto')
        alphas (array-like or None, default=None): Alpha values to try during
            cross-validation. Defaults to [0.1, 1.0, 10.0] if None.
        n_iter (int, default=100): Number of random search iterations.
            Only used when X is a list (multiple feature spaces). Ignored for single
            feature space.
        concentration (float or list, default=[0.1, 1.0]): Concentration parameters
            for Dirichlet sampling. Only used when X is a list (multiple feature spaces).
            - A value of 1 corresponds to uniform sampling over the simplex.
            - A value of infinity corresponds to equal weights.
            - If a list, samples cycle through the list.
        device (str, default='cpu'): Compute device. One of ``'cpu'`` (NumPy),
            ``'gpu'`` (PyTorch on CUDA/MPS when available, else torch-CPU), or
            ``'auto'`` (use a GPU if one is present, otherwise NumPy). Selects
            *where* the SVD/CV math runs; distinct from any CPU-core parallelism.
        local_alpha (bool, default=True): If True, select best alpha independently
            for each target. If False, select single best alpha for all targets.
        fit_intercept (bool, default=False): Whether to fit an intercept.
        conservative (bool, default=False): If True, select largest alpha within
            1 std of best score (more regularization).
        random_state (int or None, default=None): Random seed for reproducibility
            (used for CV splits and random search)
        progress_bar (bool, default=False): Whether to display progress bar during
            banded ridge fitting (when X is a list). Requires tqdm. Not used for
            single feature space ridge regression.

    Attributes:
        coef_ (ndarray of shape (n_features,) or (n_features, n_targets)): Ridge
            coefficients
        alpha_ (float or ndarray): Alpha value(s) used (selected via CV if alpha='auto')
        cv_scores_ (ndarray): Cross-validation scores (only if alpha='auto')
        deltas_ (ndarray or None): Feature space weights (only if X was a list)
            Shape: (n_spaces, n_targets). deltas = log(gamma / alpha)
        backend_ (Backend): Resolved backend instance used for computation
            (its ``.name`` reports the concrete device, e.g. ``'torch-cuda'``).

    Examples:
        >>> from nltools.models import Ridge
        >>> import numpy as np
        >>> X = np.random.randn(100, 50)
        >>> y = np.random.randn(100)
        >>> model = Ridge(alpha=1.0)
        >>> model.fit(X, y)
        Ridge(alpha=1.0, device='cpu')
        >>> y_pred = model.predict(X)
        >>>
        >>> # Banded ridge with multiple feature spaces (automatic detection)
        >>> X1 = np.random.randn(100, 30)
        >>> X2 = np.random.randn(100, 20)
        >>> model = Ridge(alpha='auto', cv=5, n_iter=50)
        >>> model.fit([X1, X2], y)
        >>> print(f"Feature space weights: {model.deltas_}")
    """

    def __init__(
        self,
        *,
        alpha: float | str = 1.0,
        cv: int | None = None,
        alphas: list[float] | np.ndarray | None = None,
        n_iter: int = 100,
        concentration: float | list[float] | None = None,
        device: str = "cpu",
        local_alpha: bool = True,
        fit_intercept: bool = False,
        conservative: bool = False,
        random_state: int | None = None,
        progress_bar: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.cv = cv
        self.alphas = alphas if alphas is not None else [0.1, 1.0, 10.0]
        self.n_iter = n_iter
        self.concentration = [0.1, 1.0] if concentration is None else concentration
        self.device = device
        self.local_alpha = local_alpha
        self.fit_intercept = fit_intercept
        self.conservative = conservative
        self.random_state = random_state
        self.progress_bar = progress_bar

    def fit(self, X: np.ndarray | list[np.ndarray], y: np.ndarray) -> Ridge:
        """Fit ridge regression model.

        Supports both regular ridge (single feature space) and banded ridge
        (multiple feature spaces). If X is a list, banded ridge is used.

        Args:
            X (ndarray of shape (n_samples, n_features) or list of arrays):
                Training data. If list, each element is a feature space for banded ridge.
            y (ndarray of shape (n_samples,) or (n_samples, n_targets)): Target values

        Returns:
            Ridge: Fitted model instance
        """
        # Check if X is a list (banded ridge) or single array (regular ridge)
        is_banded = isinstance(X, list)
        y_was_1d = False

        if is_banded:
            # Validate banded ridge inputs
            if len(X) == 0:
                raise ValueError("X cannot be an empty list")
            n_samples = X[0].shape[0]
            for i, Xi in enumerate(X):
                Xi = np.asarray(Xi)
                if Xi.ndim != 2:
                    raise ValueError(f"X[{i}] must be 2D array")
                if Xi.shape[0] != n_samples:
                    raise ValueError(
                        f"All feature spaces must have same n_samples. "
                        f"X[0] has {n_samples}, X[{i}] has {Xi.shape[0]}"
                    )
            Xs = [np.asarray(Xi) for Xi in X]
        else:
            # Regular ridge: convert to list format for unified handling
            X = self._validate_X(X)
            Xs = [X]

        # Validate y
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError(
                f"y should be 1D or 2D array, got {y.ndim}D array instead."
            )
        if y.ndim == 1:
            y_was_1d = True
            y = y[:, np.newaxis]  # Convert to 2D for uniform processing

        # Resolve the device selector ('cpu'/'gpu'/'auto') to a concrete backend.
        self.backend_ = resolve_backend(self.device)

        # Handle fixed alpha case
        if self.alpha != "auto":
            if is_banded:
                raise ValueError(
                    "Banded ridge requires alpha='auto' with cross-validation. "
                    "For fixed alpha with single feature space, pass X as array, not list."
                )
            # Fixed alpha: use simple ridge_svd. When fit_intercept=True we
            # follow sklearn's convention: center X and y on the training
            # mean, fit ridge on the centered data, then recover the
            # intercept = y_mean - X_mean @ coef. Without this, ridge has
            # no offset to absorb a non-zero target mean and silently
            # produces catastrophically biased predictions.
            self.alpha_ = self.alpha
            X_train = Xs[0]
            if self.fit_intercept:
                X_offset = X_train.mean(axis=0)
                y_offset = y.mean(axis=0)
                self.coef_ = ridge_svd(
                    X_train - X_offset,
                    y - y_offset,
                    alpha=self.alpha_,
                    parallel=self.backend_,
                )
                # coef_ is (n_features, n_targets); intercept is (n_targets,)
                self.intercept_ = y_offset - X_offset @ self.coef_
            else:
                self.coef_ = ridge_svd(
                    X_train, y, alpha=self.alpha_, parallel=self.backend_
                )
                self.intercept_ = np.zeros(y.shape[1], dtype=self.coef_.dtype)
            self.cv_scores_ = None
            self.deltas_ = None

            # Squeeze coef_ / intercept_ if y was originally 1D
            if y_was_1d and self.coef_.ndim == 2 and self.coef_.shape[1] == 1:
                self.coef_ = self.coef_.squeeze(axis=1)
                self.intercept_ = float(np.asarray(self.intercept_).squeeze())

            # Call parent fit to set fitted state
            super().fit(Xs[0], y)
            return self

        # Cross-validation case
        if self.cv is None:
            raise ValueError("cv must be specified when alpha='auto'")

        # Auto-detect: single space vs multiple spaces
        if not is_banded:
            # Single feature space: use solve_ridge_cv
            result = solve_ridge_cv(
                X=Xs[0],
                Y=y,
                alphas=self.alphas,
                cv=self.cv,
                local_alpha=self.local_alpha,
                parallel="gpu" if self.backend_.device in ("cuda", "mps") else "cpu",
                fit_intercept=self.fit_intercept,
                conservative=self.conservative,
            )
            # The solver owns intercept calculation (parity with banded
            # ridge). We just plumb it through; never recompute on the
            # un-centered data here.
            if self.fit_intercept:
                self.intercept_ = result["intercept"]
            else:
                self.intercept_ = np.zeros(y.shape[1], dtype=result["coefs"].dtype)
            self.alpha_ = result["best_alphas"]
            self.deltas_ = None
            coefs = result["coefs"]
            cv_scores = result["cv_scores"]

            # Squeeze alpha_ if single target (backward compatibility)
            if (
                isinstance(self.alpha_, np.ndarray)
                and self.alpha_.ndim == 1
                and self.alpha_.shape[0] == 1
            ):
                self.alpha_ = float(self.alpha_[0])
        else:
            # Multiple feature spaces: use solve_banded_ridge_cv (true banded ridge)
            # Convert backend device to parallel string ('cpu' or 'gpu')
            parallel_str = "gpu" if self.backend_.device in ("cuda", "mps") else "cpu"
            result = solve_banded_ridge_cv(
                Xs=Xs,
                Y=y,
                n_iter=self.n_iter,
                concentration=self.concentration,
                alphas=self.alphas,
                cv=self.cv,
                local_alpha=self.local_alpha,
                parallel=parallel_str,
                fit_intercept=self.fit_intercept,
                conservative=self.conservative,
                random_state=self.random_state,
                return_weights=True,
                progress_bar=self.progress_bar,
            )
            self.deltas_ = result["deltas"]
            self.alpha_ = None  # alphas are embedded in deltas
            coefs = result["coefs"]
            cv_scores = result["cv_scores"]
            # Banded ridge solver returns intercept directly when requested.
            if self.fit_intercept and "intercept" in result:
                self.intercept_ = result["intercept"]
            else:
                self.intercept_ = np.zeros(y.shape[1], dtype=coefs.dtype)

        self.coef_ = coefs
        self.cv_scores_ = cv_scores

        # Squeeze coef_ / intercept_ if y was originally 1D
        if y_was_1d and self.coef_.ndim == 2 and self.coef_.shape[1] == 1:
            self.coef_ = self.coef_.squeeze(axis=1)
            self.intercept_ = float(np.asarray(self.intercept_).squeeze())

        # Call parent fit to set fitted state
        # Use concatenated X for single feature space check
        X_combined = np.concatenate(Xs, axis=1) if is_banded else Xs[0]
        super().fit(X_combined, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the ridge model.

        Args:
            X (ndarray of shape (n_samples, n_features)): Samples to predict

        Returns:
            ndarray of shape (n_samples,) or (n_samples, n_targets): Predicted values
        """
        self._check_is_fitted()
        X = self._validate_X(X, reset=False)

        # Compute predictions
        y_pred = X @ self.coef_
        # Intercept is 0 when fit_intercept=False, so this branch is a
        # cheap no-op then. Stored as scalar / vector matching coef_.
        intercept = getattr(self, "intercept_", 0.0)
        y_pred = y_pred + intercept

        # Squeeze if coef_ is 1D (single target case)
        if self.coef_.ndim == 1:
            y_pred = y_pred.squeeze()

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float | np.ndarray:
        """Return the coefficient of determination R^2 of the prediction.

        For multi-target regression (y is 2D), returns per-target R² scores.
        For single-target regression (y is 1D), returns a scalar R².

        Args:
            X (ndarray of shape (n_samples, n_features)): Test samples
            y (ndarray of shape (n_samples,) or (n_samples, n_targets)): True values
                for X

        Returns:
            float or ndarray:
                - If y is 1D: scalar R²
                - If y is 2D: array of shape (n_targets,) with per-target R² scores
        """
        self._check_is_fitted()
        X, y = self._validate_X_y(X, y)

        y_pred = self.predict(X)

        # Handle single-target case (y is 1D)
        if y.ndim == 1:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            return float(r2)

        # Multi-target case: compute R² per target/voxel
        # y: (n_samples, n_targets)
        # y_pred: (n_samples, n_targets)
        ss_res = np.sum((y - y_pred) ** 2, axis=0)  # (n_targets,)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)  # (n_targets,)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # (n_targets,)

        return r2

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"Ridge(alpha={self.alpha}, device='{self.device}')"
