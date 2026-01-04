"""Multi-subject pipeline for cross-subject analyses.

This module extends the base Pipeline to handle multi-subject data,
supporting leave-one-subject-out (LOSO) and run-based CV schemes.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class MultiSubjectPipeline:
    """Pipeline for multi-subject neuroimaging analyses.

    Operates on a list of subject data arrays, supporting:
    - LOSO (leave-one-subject-out): Train on N-1 subjects, test on 1
    - Run-based CV: Split runs within each subject
    - Pooling across subjects for group analyses

    Parameters
    ----------
    data : List[np.ndarray]
        List of subject data arrays, each shape (n_obs, n_voxels).
    cv : CVScheme, optional
        Cross-validation scheme configuration.
    groups : np.ndarray, optional
        Group labels for CV splits (e.g., run labels).
    steps : List
        Transform steps to apply.

    Examples
    --------
    >>> # LOSO CV
    >>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loso'))
    >>> result = pipeline.normalize().predict(y, algorithm='svm')

    >>> # Run-based CV across subjects
    >>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loro'), groups=runs)
    >>> result = pipeline.predict(y)
    """

    data: List[NDArray]  # List of (n_obs, n_voxels) per subject
    cv: Optional[Any] = None  # CVScheme
    groups: Optional[NDArray[np.intp]] = None
    steps: List[Any] = field(default_factory=list)
    _is_lazy: bool = False

    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self.data)

    @property
    def n_steps(self) -> int:
        """Number of transform steps."""
        return len(self.steps)

    def _add_step(self, step) -> "MultiSubjectPipeline":
        """Add step and return new pipeline (immutable)."""
        new = copy(self)
        new.steps = self.steps + [step]
        return new

    # =========================================================================
    # Chainable transforms
    # =========================================================================

    def normalize(self, method: str = "zscore", **kwargs) -> "MultiSubjectPipeline":
        """Add normalization step (per-subject)."""
        from .steps import NormalizeStep

        return self._add_step(NormalizeStep(method=method, **kwargs))

    def reduce(
        self, method: str = "pca", n_components: Optional[int] = None, **kwargs
    ) -> "MultiSubjectPipeline":
        """Add dimensionality reduction step."""
        from .steps import ReduceStep

        return self._add_step(
            ReduceStep(method=method, n_components=n_components, **kwargs)
        )

    def pipe(self, transformer) -> "MultiSubjectPipeline":
        """Add custom sklearn transformer."""
        from .steps import PipeStep

        return self._add_step(PipeStep(transformer=transformer))

    # =========================================================================
    # Terminal methods
    # =========================================================================

    def predict(self, y, algorithm: str = "ridge", **kwargs):
        """Execute pipeline with CV and return prediction results.

        Parameters
        ----------
        y : np.ndarray
            Target variable. For LOSO, should be (n_subjects,).
            For run-based CV, should match pooled observations.
        algorithm : str
            Prediction algorithm.
        **kwargs
            Passed to model constructor.

        Returns
        -------
        CVResult
            Cross-validation results.
        """
        from .terminals import PredictTerminal

        if self.cv is None:
            raise ValueError("predict() requires CV context")

        y = np.asarray(y)
        terminal = PredictTerminal(y=y, algorithm=algorithm, kwargs=kwargs)

        if self.cv.is_loso:
            return self._execute_loso(terminal)
        else:
            return self._execute_run_cv(terminal)

    def _execute_loso(self, terminal):
        """Execute leave-one-subject-out CV.

        Each fold holds out one subject for testing.
        """
        from .base import FittedStack
        from .results import CVResult

        results = []

        for held_out_idx in range(self.n_subjects):
            # Split subjects
            train_subjects = [
                self.data[i] for i in range(self.n_subjects) if i != held_out_idx
            ]
            test_subject = self.data[held_out_idx]

            fitted_stack = FittedStack()

            # Apply transforms to training subjects
            for step in self.steps:
                fitted = step.fit(self._concat_subjects(train_subjects))
                fitted_stack.append(fitted)
                train_subjects = [fitted.transform(s) for s in train_subjects]
                test_subject = fitted.transform(test_subject)

            # Pool training data
            train_pooled = self._concat_subjects(train_subjects)
            test_pooled = (
                test_subject if test_subject.ndim == 2 else test_subject[np.newaxis, :]
            )

            # Get y values
            train_y = (
                np.concatenate(
                    [
                        np.full(self.data[i].shape[0], terminal.y[i])
                        for i in range(self.n_subjects)
                        if i != held_out_idx
                    ]
                )
                if terminal.y.shape[0] == self.n_subjects
                else terminal.y[: -self.data[held_out_idx].shape[0]]
            )

            test_y = (
                np.full(test_pooled.shape[0], terminal.y[held_out_idx])
                if terminal.y.shape[0] == self.n_subjects
                else terminal.y[-self.data[held_out_idx].shape[0] :]
            )

            # Create modified terminal with correct y slices
            from .terminals import PredictTerminal

            fold_terminal = PredictTerminal(
                y=np.concatenate([train_y, test_y]),
                algorithm=terminal.algorithm,
                kwargs=terminal.kwargs,
            )

            train_idx = np.arange(len(train_y))
            test_idx = np.arange(len(train_y), len(train_y) + len(test_y))

            # Fit and evaluate
            fold_result = fold_terminal.fit_evaluate(
                train_pooled, test_pooled, train_idx, test_idx, fitted_stack
            )
            results.append(fold_result)

        return CVResult(fold_results=results, pipeline=self)

    def _execute_run_cv(self, terminal):
        """Execute run-based CV across subjects.

        Splits are done by runs within each subject, then pooled.
        """
        from .base import FittedStack
        from .results import CVResult

        if self.groups is None:
            raise ValueError("Run-based CV requires groups parameter")

        results = []

        # Pool all subjects first to get consistent indexing
        pooled_data = self._concat_subjects(self.data)

        # Create subject-repeated groups
        obs_per_subject = [s.shape[0] for s in self.data]
        expanded_groups = np.concatenate(
            [self.groups[: obs_per_subject[i]] for i in range(self.n_subjects)]
        )

        # cv is guaranteed not None here due to check in predict()
        assert self.cv is not None
        for train_idx, test_idx in self.cv.split(pooled_data, groups=expanded_groups):
            fitted_stack = FittedStack()

            train_data = pooled_data[train_idx]
            test_data = pooled_data[test_idx]

            # Apply transforms
            for step in self.steps:
                fitted = step.fit(train_data)
                fitted_stack.append(fitted)
                train_data = fitted.transform(train_data)
                test_data = fitted.transform(test_data)

            # Fit and evaluate
            fold_result = terminal.fit_evaluate(
                train_data, test_data, train_idx, test_idx, fitted_stack
            )
            results.append(fold_result)

        return CVResult(fold_results=results, pipeline=self)

    def _concat_subjects(self, subjects: List[NDArray]) -> NDArray:
        """Concatenate subject data along observation axis."""
        return np.vstack(subjects)

    def __repr__(self) -> str:
        return f"MultiSubjectPipeline(n_subjects={self.n_subjects}, n_steps={self.n_steps}, cv={self.cv})"


__all__ = ["MultiSubjectPipeline"]
