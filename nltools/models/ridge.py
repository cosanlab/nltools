"""
Ridge regression model for neuroimaging data.

Wraps nltools.algorithms.ridge with sklearn-compatible API.
"""

import numpy as np
from .base import BaseModel
from ..algorithms.ridge import ridge_svd, ridge_cv
from ..backends import Backend


class Ridge(BaseModel):
    """
    Ridge regression with optional GPU acceleration.

    Wraps nltools SVD-based ridge regression algorithms with
    scikit-learn compatible API. Supports single and multi-target
    regression with optional GPU acceleration via PyTorch.

    Parameters
    ----------
    alpha : float or 'auto', default=1.0
        Regularization strength. If 'auto', uses cross-validation
        to select optimal alpha from alphas parameter.
    cv : int or None, default=None
        Number of cross-validation folds (only used if alpha='auto')
    alphas : array-like or None, default=None
        Alpha values to try during cross-validation
    backend : str or Backend, default='numpy'
        Computational backend ('numpy', 'torch', or 'auto')
    random_state : int or None, default=None
        Random seed for reproducibility

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_features, n_targets)
        Ridge coefficients
    alpha_ : float
        Alpha value used (selected via CV if alpha='auto')
    cv_scores_ : ndarray
        Cross-validation scores (only if alpha='auto')
    backend_ : Backend
        Backend instance used for computation

    Examples
    --------
    >>> from nltools.models import Ridge
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = Ridge(alpha=1.0)
    >>> model.fit(X, y)
    Ridge(alpha=1.0, backend='numpy')
    >>> y_pred = model.predict(X)
    """

    def __init__(self, alpha=1.0, cv=None, alphas=None, backend='numpy', random_state=None):
        super().__init__()
        self.alpha = alpha
        self.cv = cv
        self.alphas = alphas
        self.backend = backend
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit ridge regression model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Returns
        -------
        self : Ridge
            Fitted model instance
        """
        # Validate inputs
        X, y = self._validate_X_y(X, y)

        # Set up backend
        if isinstance(self.backend, str):
            self.backend_ = Backend(self.backend)
        else:
            self.backend_ = self.backend

        # Use CV if alpha='auto'
        if self.alpha == 'auto':
            if self.cv is None:
                raise ValueError("cv must be specified when alpha='auto'")

            result = ridge_cv(
                X, y,
                alphas=self.alphas,
                cv=self.cv,
                backend=self.backend_
            )

            self.alpha_ = result['alpha']
            self.coef_ = result['coef']
            self.cv_scores_ = result['cv_scores']
        else:
            # Fixed alpha
            self.alpha_ = self.alpha
            self.coef_ = ridge_svd(X, y, alpha=self.alpha_, backend=self.backend_)

        # Call parent fit to set fitted state and store dimensions
        super().fit(X, y)

        return self

    def predict(self, X):
        """
        Predict using the ridge model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values
        """
        self._check_is_fitted()
        X = self._validate_X(X, reset=False)

        # Compute predictions
        y_pred = X @ self.coef_

        return y_pred

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            True values for X

        Returns
        -------
        score : float
            R^2 of self.predict(X) vs y
        """
        self._check_is_fitted()
        X, y = self._validate_X_y(X, y)

        y_pred = self.predict(X)

        # Compute R^2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return r2

    def __repr__(self):
        """String representation of the model."""
        return f"Ridge(alpha={self.alpha}, backend='{self.backend}')"
