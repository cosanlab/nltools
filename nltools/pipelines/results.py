"""Result containers for nltools pipelines.

This module provides result classes that hold outputs from pipeline execution,
including cross-validation results and per-fold information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class FoldResult:
    """Result from a single CV fold.

    Holds predictions, scores, and fitted transforms for one fold,
    enabling result aggregation and inverse transforms.

    Attributes:
        score: Model score on test set (e.g., R² or accuracy).
        predictions: Model predictions on test set.
        train_idx: Indices of training samples.
        test_idx: Indices of test samples.
        fitted_stack: Stack of fitted transforms for inverse transform support.
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

    Args:
        fold_results: Results from each CV fold.
        pipeline: The pipeline that produced these results.

    Examples
    --------
    >>> result = pipeline.predict(y)
    >>> print(f"Mean score: {result.mean_score:.4f} (+/- {result.std_score:.4f})")
    >>> all_predictions = result.predictions  # In original sample order
    """

    fold_results: list[FoldResult]
    pipeline: Any  # Pipeline type, avoid circular import

    @property
    def scores(self) -> NDArray[np.floating]:
        """Per-fold prediction scores as a numpy array."""
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
        """Number of cross-validation folds."""
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

    def inverse_transform(self, data: NDArray | None = None) -> NDArray:
        """Map predictions back through inverse transforms.

        Uses the fitted transforms from each fold to inverse transform
        predictions back to the original feature space.

        Args:
            data: Data to inverse transform. If None, uses self.predictions.

        Returns:
            Data in original feature space.

        Note:
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


@dataclass
class ISCResult:
    """Result from ISC terminal computation.

    Holds intersubject correlation values, p-values, and confidence intervals
    from the ISC permutation test.

    Attributes:
        isc: ISC values. Scalar for single-feature or (n_voxels,) for voxel-wise ISC.
        p: P-values (Phipson-Smyth corrected).
        ci: Confidence interval (lower, upper).
        method: ISC method used ('pairwise' or 'leave-one-out').
        metric: Summary metric used ('median' or 'mean').
        n_subjects: Number of subjects in the analysis.
    """

    isc: NDArray[np.floating]
    p: NDArray[np.floating]
    ci: tuple
    method: str
    metric: str
    n_subjects: int

    def __repr__(self) -> str:
        if np.ndim(self.isc) == 0 or self.isc.shape == ():
            return f"ISCResult(isc={float(self.isc):.4f}, p={float(self.p):.4f})"
        return (
            f"ISCResult(n_voxels={len(self.isc)}, "
            f"mean_isc={np.mean(self.isc):.4f}, "
            f"sig_voxels={(self.p < 0.05).sum()})"
        )

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"ISCResult: {self.method} method, {self.metric} metric",
            f"  N subjects: {self.n_subjects}",
        ]
        if np.ndim(self.isc) == 0 or self.isc.shape == ():
            lines.append(f"  ISC: {float(self.isc):.4f}")
            lines.append(f"  p-value: {float(self.p):.4f}")
            lines.append(f"  95% CI: [{self.ci[0]:.4f}, {self.ci[1]:.4f}]")
        else:
            lines.append(f"  N voxels: {len(self.isc)}")
            lines.append(f"  Mean ISC: {np.mean(self.isc):.4f}")
            lines.append(f"  Significant voxels (p<0.05): {(self.p < 0.05).sum()}")
        return "\n".join(lines)


@dataclass
class RSAResult:
    """Result from RSA terminal computation.

    Holds representational similarity analysis correlation and p-value.

    Attributes:
        correlation: Correlation between neural RDM and model RDM.
        p_value: P-value from permutation test.
        ci: Confidence interval (lower, upper).
        method: Correlation method used (e.g., 'spearman', 'pearson').
        n_conditions: Number of conditions/stimuli in the RDM.
    """

    correlation: float
    p_value: float
    ci: tuple
    method: str
    n_conditions: int

    def __repr__(self) -> str:
        return f"RSAResult(r={self.correlation:.4f}, p={self.p_value:.4f})"

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"RSAResult: {self.method} correlation",
            f"  N conditions: {self.n_conditions}",
            f"  Correlation: {self.correlation:.4f}",
            f"  p-value: {self.p_value:.4f}",
            f"  95% CI: [{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
        ]
        return "\n".join(lines)


@dataclass
class PermutationResult:
    """Result from permutation testing.

    Contains the observed result from the real data, the null distribution
    of scores from permuted data, and the computed p-value.

    The p-value is calculated as the proportion of permutation scores
    that are greater than or equal to the observed score (for metrics
    where higher is better, like R2 or accuracy).

    Attributes:
        observed: The result from the real (non-permuted) data.
        null_distribution: Array of scores from each permutation.
        p_value: Permutation p-value: proportion of null scores >= observed score.
        n_permutations: Number of permutations performed.

    Examples:
        >>> perm_result = pipeline.permutation_test(y, n_permutations=1000)
        >>> print(f"Observed score: {perm_result.observed.mean_score:.4f}")
        >>> print(f"p-value: {perm_result.p_value:.4f}")

    Note:
        The p-value uses the formula ``p = (n_exceeding + 1) / (n_permutations + 1)``
        to ensure it is never exactly 0 and accounts for the observed value itself.
    """

    observed: CVResult
    null_distribution: NDArray[np.floating]
    p_value: float
    n_permutations: int

    @classmethod
    def from_scores(
        cls,
        observed: CVResult,
        null_scores: NDArray[np.floating],
    ) -> PermutationResult:
        """Create PermutationResult from observed result and null scores.

        Automatically computes the p-value from the null distribution.

        Args:
            observed: The result from the real (non-permuted) data.
            null_scores: Array of scores from each permutation.

        Returns:
            Complete permutation result with computed p-value.
        """
        n_permutations = len(null_scores)
        # Count permutations with score >= observed (plus 1 for observed itself)
        n_exceeding = np.sum(null_scores >= observed.mean_score)
        p_value = (n_exceeding + 1) / (n_permutations + 1)

        return cls(
            observed=observed,
            null_distribution=null_scores,
            p_value=float(p_value),
            n_permutations=n_permutations,
        )

    @property
    def observed_score(self) -> float:
        """Convenience accessor for observed mean score."""
        return self.observed.mean_score

    @property
    def null_mean(self) -> float:
        """Mean of the null distribution."""
        return float(np.mean(self.null_distribution))

    @property
    def null_std(self) -> float:
        """Standard deviation of the null distribution."""
        return float(np.std(self.null_distribution))

    @property
    def z_score(self) -> float:
        """Z-score of observed relative to null distribution."""
        if self.null_std == 0:
            return float("inf") if self.observed_score > self.null_mean else 0.0
        return (self.observed_score - self.null_mean) / self.null_std

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"PermutationResult: {self.n_permutations} permutations",
            f"  Observed score: {self.observed_score:.4f}",
            f"  Null mean: {self.null_mean:.4f} (+/- {self.null_std:.4f})",
            f"  Z-score: {self.z_score:.2f}",
            f"  p-value: {self.p_value:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PermutationResult(observed_score={self.observed_score:.4f}, "
            f"p_value={self.p_value:.4f}, n_permutations={self.n_permutations})"
        )


__all__ = ["CVResult", "FoldResult", "ISCResult", "PermutationResult", "RSAResult"]
