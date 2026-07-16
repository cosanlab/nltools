"""Low-level pipeline primitives for nltools.

Defines the transform protocols (`TransformStep`, `FittedTransform`) and
`FittedStack`, the container that records fitted transforms so a stack can be
inverted. These primitives back `BrainCollectionPipeline`; the standalone fluent
`Pipeline` orchestrator was removed in v0.6.0 in favor of `BrainCollection`'s
native `.cv().standardize().reduce().predict()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class FittedTransform(Protocol):
    """Protocol for fitted transform objects.

    A fitted transform holds the learned parameters from fitting on training
    data and can apply the transformation to new data.

    Note:
        Not all transforms are invertible. Check the parent TransformStep's
        ``invertible`` attribute or use ``hasattr`` before calling ``inverse_transform``.
    """

    def transform(self, data: Any) -> Any:
        """Apply the learned transformation to data.

        Args:
            data: Data to transform (typically ndarray or BrainData).

        Returns:
            Transformed data.
        """
        ...

    def inverse_transform(self, data: Any) -> Any:
        """Apply the inverse transformation to data.

        Args:
            data: Data to inverse transform.

        Returns:
            Data in original space.

        Raises:
            NotImplementedError: If the transform is not invertible.
        """
        ...


@runtime_checkable
class TransformStep(Protocol):
    """Protocol for pipeline transform steps.

    A transform step defines a transformation that can be fitted to data.
    Steps are added to a Pipeline and executed sequentially during CV.

    Attributes:
        invertible: Whether this transform supports inverse_transform.

    Examples:
    >>> class MyStep:
    ...     invertible = True
    ...     def fit(self, data):
    ...         return MyFittedTransform(learned_params)
    """

    invertible: bool

    def fit(self, data: Any) -> FittedTransform:
        """Fit the transform to data.

        Args:
            data: Training data to fit on.

        Returns:
            Fitted transform object that can transform new data.
        """
        ...


# =============================================================================
# FittedStack - Collection of fitted transforms
# =============================================================================


@dataclass
class FittedStack:
    """Collection of fitted transforms for inverse transform support.

    Maintains the sequence of fitted transforms from a pipeline execution,
    enabling inverse transformation back to the original data space.

    Attributes:
        steps: Ordered list of fitted transforms.

    Examples:
    >>> stack = FittedStack()
    >>> stack.append(fitted_pca)
    >>> stack.append(fitted_normalize)
    >>> original_space = stack.inverse_transform(predictions)
    """

    steps: list[FittedTransform] = field(default_factory=list)

    def append(self, fitted_step: FittedTransform) -> None:
        """Add a fitted transform to the stack.

        Args:
            fitted_step: Fitted transform to append.
        """
        self.steps.append(fitted_step)

    def inverse_transform(self, data: Any) -> Any:
        """Apply inverse transforms in reverse order.

        Args:
            data: Data to inverse transform.

        Returns:
            Data transformed back toward original space.

        Note:
            Steps without ``inverse_transform`` are silently skipped.
            Use ``is_fully_invertible`` to check if all steps support inversion.
        """
        for step in reversed(self.steps):
            if hasattr(step, "inverse_transform") and callable(step.inverse_transform):
                data = step.inverse_transform(data)
        return data

    @property
    def is_fully_invertible(self) -> bool:
        """Check if all steps support inverse transform.

        Returns:
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
