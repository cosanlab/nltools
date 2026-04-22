"""Multi-subject pipeline for cross-subject analyses.

This module extends the base Pipeline to handle multi-subject data,
supporting leave-one-subject-out (LOSO) and run-based CV schemes.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

    Args:
        data: List of subject data arrays, each shape (n_obs, n_voxels).
        cv: Cross-validation scheme configuration.
        groups: Group labels for CV splits (e.g., run labels).
        steps: Transform steps to apply.

    Examples
    --------
    >>> # LOSO CV
    >>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loso'))
    >>> result = pipeline.normalize().predict(y, algorithm='svm')

    >>> # Run-based CV across subjects
    >>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loro'), groups=runs)
    >>> result = pipeline.predict(y)
    """

    data: list[NDArray]  # List of (n_obs, n_voxels) per subject
    cv: Any | None = None  # CVScheme
    groups: NDArray[np.intp] | None = None
    steps: list[Any] = field(default_factory=list)
    _is_lazy: bool = False

    @property
    def n_subjects(self) -> int:
        """Number of subjects in the multi-subject dataset."""
        return len(self.data)

    @property
    def n_steps(self) -> int:
        """Number of transform steps."""
        return len(self.steps)

    def _add_step(self, step) -> MultiSubjectPipeline:
        """Add step and return new pipeline (immutable)."""
        new = copy(self)
        new.steps = self.steps + [step]
        return new

    # =========================================================================
    # Chainable transforms
    # =========================================================================

    def normalize(self, method: str = "zscore", **kwargs) -> MultiSubjectPipeline:
        """Add normalization step (per-subject)."""
        from .steps import NormalizeStep

        return self._add_step(NormalizeStep(method=method, **kwargs))

    def reduce(
        self, method: str = "pca", n_components: int | None = None, **kwargs
    ) -> MultiSubjectPipeline:
        """Add dimensionality reduction step."""
        from .steps import ReduceStep

        return self._add_step(
            ReduceStep(method=method, n_components=n_components, **kwargs)
        )

    def pipe(self, transformer) -> MultiSubjectPipeline:
        """Add custom sklearn transformer."""
        from .steps import PipeStep

        return self._add_step(PipeStep(transformer=transformer))

    def align(
        self,
        method: str = "srm",
        scheme: str = "global",
        n_features: int | None = 50,
        new_subject: str = "procrustes",
        **kwargs,
    ) -> MultiSubjectPipeline:
        """Add cross-subject alignment step to pipeline.

        Aligns multi-subject data using SRM or HyperAlignment before
        downstream analyses like classification or pooling.

        Args:
            method: Alignment method: 'srm' (Shared Response Model, reduces dimensionality)
                or 'hyperalignment' (Procrustes-based, preserves dimensionality).
                Default is 'srm'.
            scheme: Spatial scheme. Currently only 'global' is supported.
                'searchlight' and 'piecewise' require LocalAlignment (nltools-boll).
            n_features: Number of shared features for SRM. Ignored for hyperalignment.
            new_subject: Method for aligning held-out subjects in LOSO CV.
                Default is 'procrustes'.
            **kwargs: Additional arguments passed to alignment algorithm.

        Returns:
            New pipeline with alignment step added.

        Examples
        --------
        >>> # SRM alignment before classification
        >>> result = (
        ...     MultiSubjectPipeline(data=subjects, cv=CVScheme(scheme='loso'))
        ...     .align(method='srm', n_features=50)
        ...     .predict(y=labels, algorithm='svm')
        ... )

        >>> # Hyperalignment before two-stage GLM
        >>> result = (
        ...     bc.cv(scheme='loso')
        ...     .align(method='hyperalignment')
        ...     .fit(model='glm', X=designs)
        ...     .pool(param='beta')
        ...     .fit(model='ttest', contrast='A-B')
        ... )
        """
        from .steps import AlignStep

        step = AlignStep(
            method=method,
            scheme=scheme,
            n_features=n_features,
            new_subject=new_subject,
            **kwargs,
        )
        return self._add_step(step)

    # =========================================================================
    # Terminal methods
    # =========================================================================

    def isc(
        self,
        method: str = "pairwise",
        metric: str = "median",
        n_permute: int = 5000,
        parallel: str = "cpu",
        **kwargs,
    ):
        """Compute inter-subject correlation across subjects.

        Executes the pipeline and computes ISC using permutation testing.
        Data is transformed through all pipeline steps before ISC computation.

        Args:
            method: ISC computation method: 'pairwise' (average all pairwise
                correlations) or 'leave-one-out' (correlate each subject with mean
                of others). Default is 'pairwise'.
            metric: Summary statistic: 'median' (robust, default) or 'mean'
                (Fisher z-transformed).
            n_permute: Number of bootstrap iterations for p-value computation.
                Default is 5000.
            parallel: Parallelization method: 'cpu', 'gpu', or None. Default is 'cpu'.
            **kwargs: Additional arguments passed to ISCTerminal.

        Returns:
            Result containing ISC values, p-values, and confidence intervals.

        Examples
        --------
        >>> result = (
        ...     MultiSubjectPipeline(data=subjects, cv=None)
        ...     .normalize()
        ...     .isc(method='pairwise', n_permute=1000)
        ... )
        >>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")
        """
        from .terminals import ISCTerminal

        terminal = ISCTerminal(
            method=method,
            metric=metric,
            n_permute=n_permute,
            parallel=parallel,
            kwargs=kwargs,
        )

        # Apply transforms to all subjects
        transformed_data = list(self.data)
        for step in self.steps:
            fitted = step.fit(self._concat_subjects(transformed_data))
            transformed_data = [fitted.transform(s) for s in transformed_data]

        return terminal.fit_evaluate(transformed_data)

    def rsa(
        self,
        model_rdm: NDArray,
        method: str = "spearman",
        n_permute: int = 5000,
        **kwargs,
    ):
        """Compute representational similarity analysis.

        Executes the pipeline and computes RSA correlation between neural
        and model RDMs using permutation testing.

        Args:
            model_rdm: Model RDM to correlate with neural RDMs. Should be symmetric
                matrix or upper triangle (condensed form).
            method: Correlation method: 'spearman' (default), 'pearson', or 'kendall'.
            n_permute: Number of permutations for p-value computation. Default is 5000.
            **kwargs: Additional arguments passed to RSATerminal.

        Returns:
            Result containing correlation coefficient and p-value.

        Examples
        --------
        >>> model = np.corrcoef(conditions)  # Theoretical model
        >>> result = (
        ...     MultiSubjectPipeline(data=subjects, cv=None)
        ...     .normalize()
        ...     .rsa(model_rdm=model, method='spearman')
        ... )
        >>> print(f"r = {result.correlation:.3f}, p = {result.p_value:.3f}")
        """
        from .terminals import RSATerminal

        terminal = RSATerminal(
            model_rdm=model_rdm,
            method=method,
            n_permute=n_permute,
            kwargs=kwargs,
        )

        # Pool all subject data
        pooled = self._concat_subjects(self.data)

        # Apply transforms
        for step in self.steps:
            fitted = step.fit(pooled)
            pooled = fitted.transform(pooled)

        return terminal.fit_evaluate(pooled)

    def predict(self, y, algorithm: str = "ridge", **kwargs):
        """Execute pipeline with CV and return prediction results.

        Args:
            y: Target variable. For LOSO, should be (n_subjects,). For run-based CV,
                should match pooled observations.
            algorithm: Prediction algorithm: 'ridge', 'lasso', 'elastic', 'svr' for
                regression; 'svm', 'logistic', 'rf' for classification.
            **kwargs: Additional arguments passed to sklearn model constructor.
                For classification (svm, logistic), use ``class_weight='balanced'``
                to handle imbalanced classes.

        Returns:
            Cross-validation results.

        Examples
        --------
        Basic regression with LOSO CV::

            result = pipeline.cv('loso').predict(subject_labels, algorithm='ridge')

        Classification with balanced classes::

            result = pipeline.cv('loso').predict(
                group_labels, algorithm='svm', class_weight='balanced'
            )

        Logistic regression with regularization::

            result = pipeline.cv('loso').predict(
                binary_labels, algorithm='logistic', C=0.1, class_weight='balanced'
            )
        """
        from .terminals import PredictTerminal

        if self.cv is None:
            raise ValueError("predict() requires CV context")

        y = np.asarray(y)
        terminal = PredictTerminal(y=y, algorithm=algorithm, kwargs=kwargs)

        if self.cv.is_loso:
            return self._execute_loso(terminal)
        return self._execute_run_cv(terminal)

    def _execute_loso(self, terminal):
        """Execute leave-one-subject-out CV.

        Each fold holds out one subject for testing.
        """
        from .base import FittedStack
        from .results import CVResult
        from .steps import AlignStep, FittedAlign

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
                if isinstance(step, AlignStep):
                    # Alignment expects list of subjects, not concatenated data
                    fitted = step.fit(train_subjects)
                else:
                    fitted = step.fit(self._concat_subjects(train_subjects))
                fitted_stack.append(fitted)

                # Alignment uses different transform methods for train vs test
                if isinstance(fitted, FittedAlign):
                    # Transform training subjects together (returns list)
                    train_subjects = list(fitted.transform(train_subjects))
                    # Transform held-out subject using new_subject method
                    test_subject = fitted.transform_new_subject(test_subject)
                else:
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

    def _concat_subjects(self, subjects: list[NDArray]) -> NDArray:
        """Concatenate subject data along observation axis."""
        return np.vstack(subjects)

    def __repr__(self) -> str:
        return f"MultiSubjectPipeline(n_subjects={self.n_subjects}, n_steps={self.n_steps}, cv={self.cv})"


__all__ = ["MultiSubjectPipeline"]
