"""Transform steps for nltools pipelines.

This module provides reusable transform steps that can be added to pipelines.
Steps implement the TransformStep protocol and can be chained together.

Each step follows the fit/transform pattern:
- `step.fit(data)` returns a FittedX object that holds learned parameters
- `fitted.transform(data)` applies the transformation
- `fitted.inverse_transform(data)` reverses the transformation (if invertible)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


# =============================================================================
# Normalize Step
# =============================================================================


@dataclass
class FittedNormalize:
    """Fitted normalization transform.

    Holds the learned parameters (mean, std or min, range) and applies
    the transformation to new data.

    Attributes
    ----------
    mean : np.ndarray
        For zscore: the mean. For minmax: the min value.
    std : np.ndarray
        For zscore: the standard deviation. For minmax: the range (max - min).
    method : str
        The normalization method ('zscore' or 'minmax').
    """

    mean: np.ndarray
    std: np.ndarray
    method: str

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data.

        Parameters
        ----------
        data : np.ndarray
            Data to normalize.

        Returns
        -------
        np.ndarray
            Normalized data.
        """
        if self.method == "zscore":
            # Avoid division by zero
            safe_std = np.where(self.std == 0, 1, self.std)
            return (data - self.mean) / safe_std
        elif self.method == "minmax":
            # For minmax, mean stores min, std stores range
            safe_range = np.where(self.std == 0, 1, self.std)
            return (data - self.mean) / safe_range
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization.

        Parameters
        ----------
        data : np.ndarray
            Normalized data.

        Returns
        -------
        np.ndarray
            Data in original scale.
        """
        if self.method == "zscore":
            return data * self.std + self.mean
        elif self.method == "minmax":
            return data * self.std + self.mean
        return data


@dataclass
class NormalizeStep:
    """Normalization transform step.

    Computes normalization parameters from training data and applies
    the transformation to new data.

    Parameters
    ----------
    method : str
        Normalization method:
        - 'zscore': subtract mean and divide by standard deviation (default)
        - 'minmax': scale to [0, 1] range
    axis : int
        Axis along which to compute statistics. Default 0 (per-feature
        normalization, treating rows as samples).

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2], [3, 4], [5, 6]])
    >>> step = NormalizeStep(method='zscore')
    >>> fitted = step.fit(data)
    >>> normalized = fitted.transform(data)
    >>> restored = fitted.inverse_transform(normalized)
    >>> np.allclose(data, restored)
    True
    """

    method: str = "zscore"
    axis: int = 0
    invertible: bool = True

    def fit(self, data: np.ndarray) -> FittedNormalize:
        """Compute normalization parameters from data.

        Parameters
        ----------
        data : np.ndarray
            Training data to compute parameters from.

        Returns
        -------
        FittedNormalize
            Fitted transform that can be applied to new data.

        Raises
        ------
        ValueError
            If an unknown normalization method is specified.
        """
        if self.method == "zscore":
            mean = np.mean(data, axis=self.axis, keepdims=True)
            std = np.std(data, axis=self.axis, keepdims=True)
            return FittedNormalize(
                mean=mean.squeeze(), std=std.squeeze(), method=self.method
            )
        elif self.method == "minmax":
            min_val = np.min(data, axis=self.axis, keepdims=True)
            max_val = np.max(data, axis=self.axis, keepdims=True)
            range_val = max_val - min_val
            return FittedNormalize(
                mean=min_val.squeeze(), std=range_val.squeeze(), method=self.method
            )
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


# =============================================================================
# Reduce Step
# =============================================================================


@dataclass
class FittedReduce:
    """Fitted dimensionality reduction transform.

    Holds the fitted sklearn model and applies transformations.

    Attributes
    ----------
    model : Any
        Fitted sklearn decomposition model (PCA, FastICA, etc.).
    method : str
        The reduction method used.
    """

    model: Any  # sklearn PCA, FastICA, etc.
    method: str

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction.

        Parameters
        ----------
        data : np.ndarray
            Data to reduce, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Reduced data, shape (n_samples, n_components).
        """
        return self.model.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse dimensionality reduction (reconstruct original space).

        Parameters
        ----------
        data : np.ndarray
            Reduced data, shape (n_samples, n_components).

        Returns
        -------
        np.ndarray
            Reconstructed data, shape (n_samples, n_features).

        Raises
        ------
        NotImplementedError
            If the reduction method does not support inverse transform.
        """
        if hasattr(self.model, "inverse_transform"):
            return self.model.inverse_transform(data)
        raise NotImplementedError(f"{self.method} does not support inverse_transform")


@dataclass
class ReduceStep:
    """Dimensionality reduction step.

    Fits a dimensionality reduction model to training data and transforms
    new data to the reduced space.

    Parameters
    ----------
    method : str
        Reduction method:
        - 'pca': Principal Component Analysis (default, invertible)
        - 'ica': Independent Component Analysis (not invertible)
    n_components : int, optional
        Number of components to keep. If None, keeps all components
        (or max allowed by the method).
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100, 50)
    >>> step = ReduceStep(method='pca', n_components=10)
    >>> fitted = step.fit(data)
    >>> reduced = fitted.transform(data)
    >>> reduced.shape
    (100, 10)
    """

    method: str = "pca"
    n_components: Optional[int] = None
    random_state: Optional[int] = None

    @property
    def invertible(self) -> bool:
        """Check if the reduction method supports inverse transform.

        Returns
        -------
        bool
            True if method is 'pca', False otherwise.
        """
        return self.method == "pca"

    def fit(self, data: np.ndarray) -> FittedReduce:
        """Fit reduction model to data.

        Parameters
        ----------
        data : np.ndarray
            Training data, shape (n_samples, n_features).

        Returns
        -------
        FittedReduce
            Fitted transform that can be applied to new data.

        Raises
        ------
        ValueError
            If an unknown reduction method is specified.
        """
        if self.method == "pca":
            from sklearn.decomposition import PCA

            model = PCA(n_components=self.n_components, random_state=self.random_state)
        elif self.method == "ica":
            from sklearn.decomposition import FastICA

            model = FastICA(
                n_components=self.n_components, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")

        model.fit(data)
        return FittedReduce(model=model, method=self.method)


# =============================================================================
# Pipe Step (sklearn wrapper)
# =============================================================================


@dataclass
class FittedPipe:
    """Fitted sklearn transformer wrapper.

    Holds a fitted sklearn transformer and delegates transform calls to it.

    Attributes
    ----------
    transformer : Any
        Fitted sklearn-compatible transformer.
    """

    transformer: Any

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the fitted transformer.

        Parameters
        ----------
        data : np.ndarray
            Data to transform.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return self.transformer.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply inverse transform if supported.

        Parameters
        ----------
        data : np.ndarray
            Transformed data.

        Returns
        -------
        np.ndarray
            Data in original space.

        Raises
        ------
        NotImplementedError
            If the transformer does not support inverse_transform.
        """
        if hasattr(self.transformer, "inverse_transform"):
            return self.transformer.inverse_transform(data)
        raise NotImplementedError("Transformer does not support inverse_transform")


@dataclass
class PipeStep:
    """Wrapper for sklearn-compatible transformers.

    Allows any sklearn transformer with a fit/transform interface to be
    used as a pipeline step.

    Parameters
    ----------
    transformer : Any
        An sklearn-compatible transformer instance. Must have fit() and
        transform() methods. The transformer will be cloned before fitting.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> import numpy as np
    >>> data = np.random.randn(100, 10)
    >>> step = PipeStep(transformer=StandardScaler())
    >>> fitted = step.fit(data)
    >>> transformed = fitted.transform(data)
    >>> restored = fitted.inverse_transform(transformed)
    >>> np.allclose(data, restored)
    True
    """

    transformer: Any = None

    @property
    def invertible(self) -> bool:
        """Check if the transformer supports inverse_transform.

        Returns
        -------
        bool
            True if transformer has inverse_transform method.
        """
        return hasattr(self.transformer, "inverse_transform")

    def fit(self, data: np.ndarray) -> FittedPipe:
        """Fit transformer to data.

        The transformer is cloned before fitting to ensure the original
        transformer instance is not modified.

        Parameters
        ----------
        data : np.ndarray
            Training data.

        Returns
        -------
        FittedPipe
            Fitted transform wrapper.
        """
        from sklearn.base import clone

        fitted = clone(self.transformer).fit(data)
        return FittedPipe(transformer=fitted)


__all__ = [
    "FittedNormalize",
    "NormalizeStep",
    "FittedReduce",
    "ReduceStep",
    "FittedPipe",
    "PipeStep",
]
