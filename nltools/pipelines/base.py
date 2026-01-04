"""
Pipeline base infrastructure for nltools.

This module provides the foundational classes and protocols for building
chainable transform pipelines with optional cross-validation support.

The design follows an immutable pattern where each transform method returns
a new Pipeline instance, enabling fluent method chaining without side effects.

Example
-------
>>> pipeline = (
...     Pipeline(data, cv=kfold)
...     .normalize(method='zscore')
...     .reduce(method='pca', n_components=50)
...     .predict(y, algorithm='ridge')
... )
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type variables for generic transform support
T = TypeVar("T")
DataType = TypeVar("DataType")


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class FittedTransform(Protocol):
    """
    Protocol for fitted transform objects.

    A fitted transform holds the learned parameters from fitting on training
    data and can apply the transformation to new data.

    Methods
    -------
    transform(data)
        Apply the learned transformation to data.
    inverse_transform(data)
        Apply the inverse transformation (optional, may raise NotImplementedError).

    Notes
    -----
    Not all transforms are invertible. Check the parent TransformStep's
    `invertible` attribute or use `hasattr` before calling `inverse_transform`.
    """

    def transform(self, data: Any) -> Any:
        """
        Apply the learned transformation to data.

        Parameters
        ----------
        data : Any
            Data to transform (typically ndarray or BrainData).

        Returns
        -------
        Any
            Transformed data.
        """
        ...

    def inverse_transform(self, data: Any) -> Any:
        """
        Apply the inverse transformation to data.

        Parameters
        ----------
        data : Any
            Data to inverse transform.

        Returns
        -------
        Any
            Data in original space.

        Raises
        ------
        NotImplementedError
            If the transform is not invertible.
        """
        ...


@runtime_checkable
class TransformStep(Protocol):
    """
    Protocol for pipeline transform steps.

    A transform step defines a transformation that can be fitted to data.
    Steps are added to a Pipeline and executed sequentially during CV.

    Attributes
    ----------
    invertible : bool
        Whether this transform supports inverse_transform.

    Methods
    -------
    fit(data)
        Fit the transform to data and return a FittedTransform.

    Examples
    --------
    >>> class MyStep:
    ...     invertible = True
    ...     def fit(self, data):
    ...         return MyFittedTransform(learned_params)
    """

    invertible: bool

    def fit(self, data: Any) -> FittedTransform:
        """
        Fit the transform to data.

        Parameters
        ----------
        data : Any
            Training data to fit on.

        Returns
        -------
        FittedTransform
            Fitted transform object that can transform new data.
        """
        ...


@runtime_checkable
class CVScheme(Protocol):
    """
    Protocol for cross-validation schemes.

    Compatible with scikit-learn CV splitters and custom implementations.

    Methods
    -------
    split(data)
        Generate train/test index splits.
    """

    def split(
        self, data: Any
    ) -> Any:  # Returns Iterator[Tuple[ndarray, ndarray]] but protocol is lenient
        """
        Generate train/test index splits.

        Parameters
        ----------
        data : Any
            Data to split (used to determine n_samples).

        Yields
        ------
        train_idx : ndarray
            Training indices for this fold.
        test_idx : ndarray
            Test indices for this fold.
        """
        ...


@runtime_checkable
class Terminal(Protocol):
    """
    Protocol for terminal operations that end a pipeline.

    Terminals perform the final computation (e.g., prediction, similarity)
    and produce results for each CV fold.
    """

    def fit_evaluate(
        self,
        train_data: Any,
        test_data: Any,
        train_idx: NDArray[np.intp],
        test_idx: NDArray[np.intp],
        fitted_stack: FittedStack,
    ) -> Any:
        """
        Fit on training data and evaluate on test data.

        Parameters
        ----------
        train_data : Any
            Transformed training data.
        test_data : Any
            Transformed test data.
        train_idx : ndarray
            Original training indices.
        test_idx : ndarray
            Original test indices.
        fitted_stack : FittedStack
            Stack of fitted transforms for inverse transform support.

        Returns
        -------
        Any
            Fold result (structure depends on terminal type).
        """
        ...


# =============================================================================
# FittedStack - Collection of fitted transforms
# =============================================================================


@dataclass
class FittedStack:
    """
    Collection of fitted transforms for inverse transform support.

    Maintains the sequence of fitted transforms from a pipeline execution,
    enabling inverse transformation back to the original data space.

    Attributes
    ----------
    steps : List[FittedTransform]
        Ordered list of fitted transforms.

    Examples
    --------
    >>> stack = FittedStack()
    >>> stack.append(fitted_pca)
    >>> stack.append(fitted_normalize)
    >>> original_space = stack.inverse_transform(predictions)
    """

    steps: List[FittedTransform] = field(default_factory=list)

    def append(self, fitted_step: FittedTransform) -> None:
        """
        Add a fitted transform to the stack.

        Parameters
        ----------
        fitted_step : FittedTransform
            Fitted transform to append.
        """
        self.steps.append(fitted_step)

    def inverse_transform(self, data: Any) -> Any:
        """
        Apply inverse transforms in reverse order.

        Parameters
        ----------
        data : Any
            Data to inverse transform.

        Returns
        -------
        Any
            Data transformed back toward original space.

        Notes
        -----
        Steps without `inverse_transform` are silently skipped.
        Use `is_fully_invertible` to check if all steps support inversion.
        """
        for step in reversed(self.steps):
            if hasattr(step, "inverse_transform") and callable(step.inverse_transform):
                data = step.inverse_transform(data)
        return data

    @property
    def is_fully_invertible(self) -> bool:
        """
        Check if all steps support inverse transform.

        Returns
        -------
        bool
            True if all steps have callable inverse_transform methods.
        """
        return all(
            hasattr(s, "inverse_transform") and callable(s.inverse_transform)
            for s in self.steps
        )

    def __len__(self) -> int:
        """Return number of fitted steps."""
        return len(self.steps)

    def __iter__(self):
        """Iterate over fitted steps."""
        return iter(self.steps)


# =============================================================================
# Pipeline Base Class
# =============================================================================


@dataclass
class Pipeline:
    """
    Base pipeline for chained transforms with optional cross-validation.

    Pipelines enable fluent, chainable data transformations that are executed
    within a cross-validation framework. Each transform method returns a new
    Pipeline instance (immutable pattern), allowing method chaining without
    side effects.

    Parameters
    ----------
    data : Any
        Input data (typically ndarray or BrainData).
    cv : CVScheme, optional
        Cross-validation scheme. Required for terminal methods like predict().
    steps : List[TransformStep]
        List of transform steps (typically not set directly).

    Attributes
    ----------
    _is_lazy : bool
        Whether pipeline is in lazy evaluation mode (future feature).

    Examples
    --------
    >>> from sklearn.model_selection import KFold
    >>> cv = KFold(n_splits=5)
    >>> result = (
    ...     Pipeline(X, cv=cv)
    ...     .normalize(method='zscore')
    ...     .reduce(method='pca', n_components=50)
    ...     .predict(y, algorithm='ridge')
    ... )

    Notes
    -----
    The pipeline uses an immutable pattern: each method returns a new
    Pipeline instance rather than modifying in place. This enables:
    - Safe method chaining
    - Branching pipelines from intermediate states
    - Functional programming patterns
    """

    data: Any
    cv: Optional[CVScheme] = None
    steps: List[TransformStep] = field(default_factory=list)
    _is_lazy: bool = field(default=False, repr=False)

    def _add_step(self, step: TransformStep) -> Pipeline:
        """
        Add a transform step and return a new pipeline.

        Parameters
        ----------
        step : TransformStep
            Transform step to add.

        Returns
        -------
        Pipeline
            New pipeline instance with the step added.

        Notes
        -----
        This method implements the immutable pattern - the original
        pipeline is not modified.
        """
        new = copy(self)
        new.steps = self.steps + [step]
        return new

    # =========================================================================
    # Chainable Transform Methods (placeholders - Phase 2 implements steps)
    # =========================================================================

    def normalize(self, method: str = "zscore", **kwargs: Any) -> Pipeline:
        """
        Add a normalization step to the pipeline.

        Parameters
        ----------
        method : str, default='zscore'
            Normalization method. Options: 'zscore', 'minmax', 'robust'.
        **kwargs : Any
            Additional arguments passed to the normalizer.

        Returns
        -------
        Pipeline
            New pipeline with normalization step added.

        Examples
        --------
        >>> pipeline.normalize(method='zscore')
        >>> pipeline.normalize(method='minmax', feature_range=(0, 1))
        """
        from .steps import NormalizeStep

        return self._add_step(NormalizeStep(method=method, **kwargs))

    def reduce(
        self, method: str = "pca", n_components: Optional[int] = None, **kwargs: Any
    ) -> Pipeline:
        """
        Add a dimensionality reduction step to the pipeline.

        Parameters
        ----------
        method : str, default='pca'
            Reduction method. Options: 'pca', 'ica', 'nmf', 'srm'.
        n_components : int, optional
            Number of components to keep.
        **kwargs : Any
            Additional arguments passed to the reducer.

        Returns
        -------
        Pipeline
            New pipeline with reduction step added.

        Examples
        --------
        >>> pipeline.reduce(method='pca', n_components=50)
        >>> pipeline.reduce(method='srm', n_components=20, n_iter=100)
        """
        from .steps import ReduceStep

        return self._add_step(
            ReduceStep(method=method, n_components=n_components, **kwargs)
        )

    def pipe(self, transformer: Any) -> Pipeline:
        """
        Add a custom transformer to the pipeline.

        Parameters
        ----------
        transformer : Any
            Custom transformer with fit/transform interface.
            Must be compatible with sklearn transformers or implement
            the TransformStep protocol.

        Returns
        -------
        Pipeline
            New pipeline with custom step added.

        Examples
        --------
        >>> from sklearn.decomposition import FastICA
        >>> pipeline.pipe(FastICA(n_components=20))
        """
        from .steps import PipeStep

        return self._add_step(PipeStep(transformer))

    # =========================================================================
    # Terminal Methods (placeholder - Phase 3 implements)
    # =========================================================================

    def predict(self, y: Any, algorithm: str = "ridge", **kwargs: Any) -> Any:
        """
        Execute pipeline with cross-validation and return prediction results.

        This is a terminal method that triggers pipeline execution.

        Parameters
        ----------
        y : Any
            Target variable to predict.
        algorithm : str, default='ridge'
            Prediction algorithm. Options: 'ridge', 'lasso', 'svr'.
        **kwargs : Any
            Additional arguments passed to the predictor.

        Returns
        -------
        CVResult
            Cross-validation results containing predictions and metrics.

        Raises
        ------
        ValueError
            If no CV scheme is set.

        Examples
        --------
        >>> result = pipeline.predict(y, algorithm='ridge', alpha=1.0)
        >>> print(result.summary())
        """
        if self.cv is None:
            raise ValueError(
                "predict() requires CV context. Use cv= parameter when creating Pipeline."
            )
        from .terminals import PredictTerminal

        return self._execute_cv(PredictTerminal(y, algorithm, **kwargs))

    # =========================================================================
    # CV Execution Engine
    # =========================================================================

    def _execute_cv(self, terminal: Terminal) -> Any:
        """
        Execute pipeline for each CV fold.

        Parameters
        ----------
        terminal : Terminal
            Terminal operation to apply after transforms.

        Returns
        -------
        CVResult
            Aggregated results from all CV folds.

        Raises
        ------
        ValueError
            If no CV scheme is set.
        """
        if self.cv is None:
            raise ValueError("No CV scheme set")

        results = []
        for train_idx, test_idx in self.cv.split(self.data):
            train, test = self._split_data(train_idx, test_idx)
            fitted_stack = FittedStack()

            # Fit and transform through pipeline
            for step in self.steps:
                fitted = step.fit(train)
                fitted_stack.append(fitted)
                train = fitted.transform(train)
                test = fitted.transform(test)

            # Apply terminal
            fold_result = terminal.fit_evaluate(
                train, test, train_idx, test_idx, fitted_stack
            )
            results.append(fold_result)

        from .results import CVResult

        return CVResult(results, self)

    def _split_data(
        self, train_idx: NDArray[np.intp], test_idx: NDArray[np.intp]
    ) -> Tuple[Any, Any]:
        """
        Split data by indices.

        Parameters
        ----------
        train_idx : ndarray
            Indices for training data.
        test_idx : ndarray
            Indices for test data.

        Returns
        -------
        train : Any
            Training data subset.
        test : Any
            Test data subset.

        Raises
        ------
        NotImplementedError
            If data type is not supported. Override in subclasses
            for custom data types like BrainData.
        """
        if isinstance(self.data, np.ndarray):
            return self.data[train_idx], self.data[test_idx]
        # BrainData and other types handled in subclasses
        raise NotImplementedError(f"_split_data not implemented for {type(self.data)}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation of pipeline."""
        step_names = [type(s).__name__ for s in self.steps]
        cv_info = type(self.cv).__name__ if self.cv else "None"
        return f"Pipeline(steps={step_names}, cv={cv_info})"

    @property
    def n_steps(self) -> int:
        """Return number of transform steps."""
        return len(self.steps)

    def copy(self) -> Pipeline:
        """
        Create a shallow copy of the pipeline.

        Returns
        -------
        Pipeline
            New pipeline instance with same configuration.
        """
        return copy(self)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TransformStep",
    "FittedTransform",
    "CVScheme",
    "Terminal",
    "FittedStack",
    "Pipeline",
]
