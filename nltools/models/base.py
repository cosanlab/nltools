"""
Base classes for nltools models.

Provides sklearn-compatible API for neuroimaging analysis.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all nltools models.

    Follows scikit-learn API conventions:
    - fit(X, y) trains the model and returns self
    - predict(X) generates predictions
    - score(X, y) evaluates model performance

    Attributes:
        n_features_in_ (int): Number of features seen during fit
        n_samples_ (int): Number of samples seen during fit
        is_fitted_ (bool): Whether the model has been fitted
    """

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to training data.

        Args:
            X (ndarray of shape (n_samples, n_features)): Training data
            y (ndarray of shape (n_samples,) or (n_samples, n_targets)): Target values

        Returns:
            BaseModel: Fitted model instance
        """
        # Store training dimensions
        self.n_samples_, self.n_features_in_ = X.shape
        self.is_fitted_ = True
        return self

    @abstractmethod
    def predict(self, X):
        """
        Generate predictions for new data.

        Args:
            X (ndarray of shape (n_samples, n_features)): Data to predict on

        Returns:
            ndarray: Predicted values
        """
        pass

    @abstractmethod
    def score(self, X, y):
        """
        Evaluate model performance.

        Args:
            X (ndarray of shape (n_samples, n_features)): Test data
            y (ndarray of shape (n_samples,) or (n_samples, n_targets)): True values

        Returns:
            float: Model performance metric
        """
        pass

    def _check_is_fitted(self):
        """
        Check if model has been fitted.

        Raises:
            ValueError: If model has not been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError(
                f"{self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this model."
            )

    def _validate_X(self, X, reset=True):
        """
        Validate input data X.

        Args:
            X (array-like): Input data to validate
            reset (bool, default=True): If True, allows fitting (new n_features_in_).
                If False, validates that n_features matches training.

        Returns:
            ndarray: Validated and converted input array

        Raises:
            ValueError: If X is not 2D or features don't match training
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {X.ndim}D array instead. "
                f"Reshape your data using array.reshape(-1, 1) for single feature."
            )

        if not reset and hasattr(self, "n_features_in_"):
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                    f"was fitted with {self.n_features_in_} features."
                )

        return X

    def _validate_X_y(self, X, y):
        """
        Validate input data X and target y.

        Args:
            X (array-like): Input data
            y (array-like): Target values

        Returns:
            tuple: (X, y) as validated ndarrays

        Raises:
            ValueError: If X/y shapes are invalid or inconsistent
        """
        X = self._validate_X(X)
        y = np.asarray(y)

        if y.ndim > 2:
            raise ValueError(
                f"y should be 1D or 2D array, got {y.ndim}D array instead."
            )

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent number of samples: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )

        return X, y
