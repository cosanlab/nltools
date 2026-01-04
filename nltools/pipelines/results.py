"""Result containers for nltools pipelines.

This module provides result classes that hold outputs from pipeline execution,
including cross-validation results and per-fold information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class FoldResult:
    """Result from a single CV fold.

    Holds predictions, scores, and fitted transforms for one fold,
    enabling result aggregation and inverse transforms.

    Attributes
    ----------
    score : float
        Model score on test set (e.g., R² or accuracy).
    predictions : np.ndarray
        Model predictions on test set.
    train_idx : np.ndarray
        Indices of training samples.
    test_idx : np.ndarray
        Indices of test samples.
    fitted_stack : FittedStack
        Stack of fitted transforms for inverse transform support.
    """

    score: float
    predictions: NDArray[np.floating]
    train_idx: NDArray[np.intp]
    test_idx: NDArray[np.intp]
    fitted_stack: Any  # FittedStack, avoid circular import

    def __repr__(self) -> str:
        return f"FoldResult(score={self.score:.4f}, n_test={len(self.test_idx)})"


@dataclass
class CVResult:
    """Cross-validation result container.

    Aggregates results from all CV folds, providing access to scores,
    predictions, and inverse transform capability.

    Parameters
    ----------
    fold_results : List[FoldResult]
        Results from each CV fold.
    pipeline : Pipeline
        The pipeline that produced these results.

    Examples
    --------
    >>> result = pipeline.predict(y)
    >>> print(f"Mean score: {result.mean_score:.4f} (+/- {result.std_score:.4f})")
    >>> all_predictions = result.predictions  # In original sample order
    """

    fold_results: List[FoldResult]
    pipeline: Any  # Pipeline type, avoid circular import

    @property
    def scores(self) -> NDArray[np.floating]:
        """Per-fold scores as numpy array."""
        return np.array([f.score for f in self.fold_results])

    @property
    def mean_score(self) -> float:
        """Mean score across all folds."""
        return float(self.scores.mean())

    @property
    def std_score(self) -> float:
        """Standard deviation of scores across folds."""
        return float(self.scores.std())

    @property
    def n_folds(self) -> int:
        """Number of CV folds."""
        return len(self.fold_results)

    @property
    def predictions(self) -> NDArray[np.floating]:
        """All predictions in original sample order.

        Reconstructs predictions array with each sample's prediction
        from the fold where it was in the test set.
        """
        # Determine total samples from indices
        all_test_idx = np.concatenate([f.test_idx for f in self.fold_results])
        n_samples = int(all_test_idx.max()) + 1

        # Determine prediction shape (handle multi-output)
        first_pred = self.fold_results[0].predictions
        if first_pred.ndim == 1:
            preds = np.zeros(n_samples)
        else:
            preds = np.zeros((n_samples, first_pred.shape[1]))

        # Fill in predictions from each fold
        for fold in self.fold_results:
            preds[fold.test_idx] = fold.predictions

        return preds

    def inverse_transform(self, data: Optional[NDArray] = None) -> NDArray:
        """Map predictions back through inverse transforms.

        Uses the fitted transforms from each fold to inverse transform
        predictions back to the original feature space.

        Parameters
        ----------
        data : np.ndarray, optional
            Data to inverse transform. If None, uses self.predictions.

        Returns
        -------
        np.ndarray
            Data in original feature space.

        Notes
        -----
        This applies inverse transforms fold-by-fold, using each fold's
        fitted parameters. Not all pipelines support full inversion.
        """
        if data is None:
            data = self.predictions

        # For now, use the first fold's fitted stack
        # TODO: Could average across folds or return per-fold results
        if self.fold_results and self.fold_results[0].fitted_stack:
            return self.fold_results[0].fitted_stack.inverse_transform(data)
        return data

    @property
    def is_fully_invertible(self) -> bool:
        """Check if all transform steps are invertible."""
        if not self.fold_results:
            return True
        stack = self.fold_results[0].fitted_stack
        if stack is None:
            return True
        return stack.is_fully_invertible

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"CVResult: {self.n_folds} folds",
            f"  Mean score: {self.mean_score:.4f} (+/- {self.std_score:.4f})",
            f"  Scores: {self.scores}",
            f"  Invertible: {self.is_fully_invertible}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CVResult(n_folds={self.n_folds}, mean_score={self.mean_score:.4f})"


__all__ = ["FoldResult", "CVResult"]
