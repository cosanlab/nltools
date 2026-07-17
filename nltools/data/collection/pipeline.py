"""Pipeline classes for BrainCollection.

Provides BrainCollectionPipeline for fluent pipeline API with cross-validation,
and FittedBrainCollection for chaining pool() after fit(). CV-aware
``predict()`` returns a ``BrainData`` with CV attributes attached
(``cv_scores``, ``cv_predictions``, ``mean_score``, ``std_score``,
``fold_results``, ``cv_pipeline``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nltools.data.collection import BrainCollection


class BrainCollectionPipeline:
    """Pipeline for BrainCollection with multi-subject CV support.

    Wraps BrainCollection to provide fluent pipeline API with LOSO
    and run-based cross-validation.

    This class enables method chaining for preprocessing and prediction
    with proper cross-validation semantics for multi-subject neuroimaging
    analyses.

    Attributes:
        n_subjects: Number of subjects/images in the collection.
        cv: The cross-validation scheme configuration.
        n_steps: Number of transform steps in the pipeline.

    Examples:
        >>> # Leave-one-subject-out with preprocessing
        >>> result = (bc
        ...     .cv(scheme='loso')
        ...     .standardize()
        ...     .reduce(n_components=50)
        ...     .predict(labels, method='svm'))
        >>> print(f"Mean accuracy: {result.mean_score:.2%}")
    """

    def __init__(
        self,
        brain_collection: BrainCollection,
        cv=None,
        groups: np.ndarray | None = None,
    ):
        """Initialize pipeline wrapper.

        Args:
            brain_collection: BrainCollection to wrap.
            cv: CVScheme configuration.
            groups: Group labels for CV splits.
        """
        self._bc = brain_collection
        self._cv = cv
        self._groups = groups
        self._steps = []

    @property
    def n_subjects(self) -> int:
        """Number of subjects/images."""
        return self._bc.n_subjects

    @property
    def cv(self):
        """Cross-validation scheme."""
        return self._cv

    @property
    def n_steps(self) -> int:
        """Number of transform steps."""
        return len(self._steps)

    def _add_step(self, step) -> BrainCollectionPipeline:
        """Add step and return new pipeline (immutable).

        Args:
            step: Transform step to add.

        Returns:
            New pipeline with step added.
        """
        from copy import copy

        new = copy(self)
        new._steps = self._steps + [step]
        return new

    def standardize(  # nosemgrep: kwargs-internal-forwarding  # forwards to the sklearn transformer via NormalizeStep
        self, method: str = "zscore", **kwargs
    ) -> BrainCollectionPipeline:
        """Add standardization step.

        Args:
            method: Standardization method ('zscore', 'minmax').
            **kwargs: Additional arguments for NormalizeStep.

        Returns:
            New pipeline with standardization step added.
        """
        from nltools.pipelines.steps import NormalizeStep

        return self._add_step(NormalizeStep(method=method, **kwargs))

    def reduce(  # nosemgrep: kwargs-internal-forwarding  # forwards to the sklearn transformer via ReduceStep
        self, method: str = "pca", n_components: int | None = None, **kwargs
    ) -> BrainCollectionPipeline:
        """Add dimensionality reduction step.

        Args:
            method: Reduction method ('pca', 'ica').
            n_components: Number of components to keep.
            **kwargs: Additional arguments for ReduceStep.

        Returns:
            New pipeline with reduction step added.
        """
        from nltools.pipelines.steps import ReduceStep

        return self._add_step(
            ReduceStep(method=method, n_components=n_components, **kwargs)
        )

    def pipe(self, transformer) -> BrainCollectionPipeline:
        """Add custom sklearn transformer.

        Args:
            transformer: sklearn-compatible transformer with fit/transform interface.

        Returns:
            New pipeline with custom step added.
        """
        from nltools.pipelines.steps import PipeStep

        return self._add_step(PipeStep(transformer=transformer))

    def predict(  # nosemgrep: kwargs-internal-forwarding  # **kwargs forwards to the sklearn estimator constructor
        self, y, method: str = "ridge", **kwargs
    ):
        """Execute pipeline with CV and return prediction results.

        Args:
            y: Target variable. For LOSO, shape should be (n_subjects,).
            method: Prediction algorithm ('ridge', 'svm', 'logistic', etc.)
            **kwargs: Passed to model constructor.

        Returns:
            ``BrainData`` carrying out-of-fold predictions plus CV attributes
            (``cv_scores``, ``cv_predictions``, ``mean_score``, ``std_score``,
            ``fold_results``, ``cv_pipeline``).

        Raises:
            ValueError: If no CV context is set or if non-LOSO CV is used without groups.
        """

        if self._cv is None:
            raise ValueError("predict() requires CV context")

        y = np.asarray(y)

        # Get data as list of numpy arrays
        subject_data = [bd.data for bd in self._bc]

        if self._cv.is_loso:
            return self._execute_loso(subject_data, y, method, kwargs)
        return self._execute_pooled_cv(subject_data, y, method, kwargs)

    def _execute_loso(
        self, subject_data: list, y: np.ndarray, algorithm: str, model_kwargs: dict
    ):
        """Execute leave-one-subject-out CV.

        Args:
            subject_data: List of arrays, one per subject.
            y: Target labels.
            algorithm: Prediction algorithm name.
            model_kwargs: Kwargs passed to model constructor.

        Returns:
            ``BrainData`` with CV attributes attached.
        """
        from nltools.pipelines.base import FittedStack

        results = []
        n_subjects = len(subject_data)

        for held_out_idx in range(n_subjects):
            # Split subjects
            train_subjects = [
                subject_data[i] for i in range(n_subjects) if i != held_out_idx
            ]
            test_subject = subject_data[held_out_idx]

            fitted_stack = FittedStack()

            # Pool training data
            train_pooled = np.vstack(train_subjects)
            test_data = (
                test_subject if test_subject.ndim == 2 else test_subject[np.newaxis, :]
            )

            # Apply transforms
            for step in self._steps:
                fitted = step.fit(train_pooled)
                fitted_stack.append(fitted)
                train_pooled = fitted.transform(train_pooled)
                test_data = fitted.transform(test_data)

            # Handle y based on shape
            if y.shape[0] == n_subjects:
                # One label per subject
                train_y = np.concatenate(
                    [
                        np.full(subject_data[i].shape[0], y[i])
                        for i in range(n_subjects)
                        if i != held_out_idx
                    ]
                )
                test_y = np.full(test_data.shape[0], y[held_out_idx])
            else:
                # Labels match observations - need proper indexing
                obs_per_subj = [s.shape[0] for s in subject_data]
                cumsum = np.cumsum([0] + obs_per_subj)
                train_mask = np.ones(sum(obs_per_subj), dtype=bool)
                train_mask[cumsum[held_out_idx] : cumsum[held_out_idx + 1]] = False
                train_y = y[train_mask]
                test_y = y[cumsum[held_out_idx] : cumsum[held_out_idx + 1]]

            # Get model
            model = self._get_model(algorithm, model_kwargs)
            model.fit(train_pooled, train_y)

            predictions = model.predict(test_data)
            score = model.score(test_data, test_y)

            train_idx = np.arange(len(train_y))
            test_idx = np.arange(len(train_y), len(train_y) + len(test_y))

            results.append(
                {
                    "score": score,
                    "predictions": predictions,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "fitted_stack": fitted_stack,
                    "held_out_subject": held_out_idx,
                }
            )

        # Stitch held-out predictions into (n_total_obs, n_voxels) in
        # original subject order. Each fold's predictions correspond to a
        # contiguous slice of the pooled data.
        obs_per_subj = [s.shape[0] for s in subject_data]
        cumsum = np.cumsum([0] + obs_per_subj)
        n_total_obs = cumsum[-1]
        n_voxels = subject_data[0].shape[1] if subject_data[0].ndim == 2 else 1
        cv_predictions = np.zeros((n_total_obs, n_voxels), dtype=np.float64)
        for r in results:
            held = r["held_out_subject"]
            preds = np.asarray(r["predictions"]).reshape(obs_per_subj[held], -1)
            cv_predictions[cumsum[held] : cumsum[held + 1]] = preds

        return self._make_cv_braindata(results, cv_predictions)

    def _execute_pooled_cv(
        self, subject_data: list, y: np.ndarray, algorithm: str, model_kwargs: dict
    ):
        """Execute CV on pooled data.

        Args:
            subject_data: List of arrays, one per subject.
            y: Target labels.
            algorithm: Prediction algorithm name.
            model_kwargs: Kwargs passed to model constructor.

        Returns:
            Cross-validation results.

        Raises:
            ValueError: If groups parameter is not set.
        """
        from nltools.pipelines.base import FittedStack

        # Pool all data
        pooled_data = np.vstack(subject_data)

        if self._groups is None:
            raise ValueError("Non-LOSO CV requires groups parameter")

        results = []

        for train_idx, test_idx in self._cv.split(pooled_data, groups=self._groups):
            fitted_stack = FittedStack()

            train_data = pooled_data[train_idx]
            test_data = pooled_data[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]

            # Apply transforms
            for step in self._steps:
                fitted = step.fit(train_data)
                fitted_stack.append(fitted)
                train_data = fitted.transform(train_data)
                test_data = fitted.transform(test_data)

            # Fit and evaluate
            model = self._get_model(algorithm, model_kwargs)
            model.fit(train_data, train_y)

            predictions = model.predict(test_data)
            score = model.score(test_data, test_y)

            results.append(
                {
                    "score": score,
                    "predictions": predictions,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "fitted_stack": fitted_stack,
                }
            )

        # Stitch held-out predictions back into original pooled order.
        n_total_obs = pooled_data.shape[0]
        n_voxels = pooled_data.shape[1] if pooled_data.ndim == 2 else 1
        cv_predictions = np.zeros((n_total_obs, n_voxels), dtype=np.float64)
        for r in results:
            preds = np.asarray(r["predictions"]).reshape(len(r["test_idx"]), -1)
            cv_predictions[r["test_idx"]] = preds

        return self._make_cv_braindata(results, cv_predictions)

    def _make_cv_braindata(self, results: list, cv_predictions: np.ndarray):
        """Wrap CV fold results as a ``BrainData`` with CV attrs attached.

        The CV attributes mirror what ``BrainData.predict(...)`` exposes so
        the same downstream consumers work for both single-subject and
        BrainCollection CV paths.
        """
        from nltools.data.braindata import BrainData

        cv_scores = np.array([r["score"] for r in results], dtype=np.float64)
        bd = BrainData(cv_predictions.astype(np.float32), mask=self._bc._mask)
        bd.cv_scores = cv_scores
        bd.cv_predictions = cv_predictions
        bd.mean_score = float(cv_scores.mean()) if cv_scores.size else 0.0
        bd.std_score = float(cv_scores.std()) if cv_scores.size else 0.0
        bd.fold_results = results
        bd.cv_pipeline = self
        return bd

    def _get_model(self, algorithm: str, kwargs: dict):
        """Get sklearn model instance.

        Args:
            algorithm: Algorithm name.
            kwargs: Model constructor arguments.

        Returns:
            Sklearn estimator instance.

        Raises:
            ValueError: If algorithm is unknown.
        """
        if algorithm == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(**kwargs)
        if algorithm == "lasso":
            from sklearn.linear_model import Lasso

            return Lasso(**kwargs)
        if algorithm == "svm":
            from sklearn.svm import SVC

            return SVC(**kwargs)
        if algorithm == "svr":
            from sklearn.svm import SVR

            return SVR(**kwargs)
        if algorithm == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**kwargs)
        raise ValueError(f"Unknown algorithm: {algorithm}")

    def __repr__(self):
        """Return string representation."""
        return f"BrainCollectionPipeline(n_subjects={self.n_subjects}, n_steps={self.n_steps})"


class FittedBrainCollection:
    """Wrapper for fitted BrainCollection enabling pool() chaining.

    This class wraps the results of bc.fit() and provides the .pool()
    method for aggregating across subjects.

    The execution model:
    - fit() executes immediately (eager)
    - pool() aggregates the fitted parameters
    - pool() returns PooledData for second-level analysis

    Args:
        brain_collection: The original collection that was fitted.
        fitted_results: The fitted results. Can be a BrainCollection (betas or
            scores) or a dict mapping stat names to BrainCollections
            (e.g., {'betas': ..., 't': ...}).
        model: The model type that was fitted ('glm' or 'ridge').
        condition_names: Names of conditions/regressors from the design matrix.

    Examples
    --------
    >>> fitted = bc.fit(model='glm', X=dm)
    >>> pool = fitted.pool(param='beta')
    >>> result = pool.fit(model='ttest', contrast='A-B')
    """

    def __init__(
        self,
        brain_collection: BrainCollection,
        fitted_results: BrainCollection | dict[str, BrainCollection],
        model: str,
        condition_names: list[str] | None = None,
    ):
        self._bc = brain_collection
        self._fitted = fitted_results
        self._model = model
        self._condition_names = condition_names

    @property
    def n_subjects(self) -> int:
        """Number of subjects in the fitted collection."""
        if isinstance(self._fitted, dict):
            # Get from first value in dict
            first = next(iter(self._fitted.values()))
            return len(first)
        return len(self._fitted)

    @property
    def results(self) -> BrainCollection | dict[str, BrainCollection]:
        """Access the fitted results directly.

        Returns the underlying BrainCollection or dict of BrainCollections.
        Use this for backward compatibility or when pool() is not needed.
        """
        return self._fitted

    @property
    def betas(self) -> BrainCollection:
        """Convenience accessor for beta coefficients from a GLM fit.

        Returns:
            Beta coefficients from GLM fit.

        Raises:
            ValueError: If model is not GLM or betas not available.
        """
        from nltools.data.collection import BrainCollection as _BC

        if isinstance(self._fitted, dict):
            if "betas" in self._fitted:
                betas = self._fitted["betas"]
                if not isinstance(betas, _BC):
                    raise ValueError(
                        f"Expected BrainCollection betas, got {type(betas).__name__}"
                    )
                return betas
            raise ValueError("No 'betas' key in fitted results dict")
        if self._model == "glm":
            if not isinstance(self._fitted, _BC):
                raise ValueError(
                    f"Expected BrainCollection betas, got {type(self._fitted).__name__}"
                )
            return self._fitted
        raise ValueError(f"'betas' not available for model='{self._model}'")

    def pool(
        self,
        param: str = "beta",
        contrast: str | None = None,
        save: str | None = None,
        save_fitted: bool = False,
    ):
        """Pool fitted parameters across subjects.

        Aggregates per-subject fitted results for group-level analysis.
        Returns a PooledData object that can be passed to second-level
        statistical tests.

        Args:
            param: Parameter to pool. GLM options: 'beta', 't', 'r2', 'p', 'se',
                'residual'. Ridge options: 'scores', 'weights'. Default is 'beta'.
            contrast: Apply contrast before pooling. Format: 'A-B' or 'A+B'.
                Requires condition_names to be available.
            save: Path template to save per-subject results before pooling.
                Supports {subject}, {idx} placeholders.
            save_fitted: If True, save full fitted state for later repool().

        Returns:
            Pooled data ready for second-level analysis.

        Examples
        --------
        >>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
        >>> result = pool.fit(model='ttest', contrast='face-house')

        >>> # Pool t-statistics instead of betas
        >>> pool = bc.fit(model='glm', X=dm, return_stats=['t']).pool(param='t')
        """
        from nltools.pipelines.pool import PooledData

        # Determine what data to pool
        if isinstance(self._fitted, dict):
            # Results include multiple stats
            param_key = param if param != "beta" else "betas"
            if param_key not in self._fitted:
                available = list(self._fitted.keys())
                raise ValueError(
                    f"Parameter '{param}' not found. Available: {available}"
                )
            data_to_pool = self._fitted[param_key]
        else:
            # Single BrainCollection result
            if param not in ("beta", "betas", "scores", "weights"):
                raise ValueError(
                    f"Parameter '{param}' not available. For GLM stats, "
                    "use return_stats=['t', 'p', ...] in fit()."
                )
            data_to_pool = self._fitted

        # Extract data as array: (n_subjects, n_conditions, n_voxels)
        # Each item in data_to_pool is a BrainData with shape (n_conditions, n_voxels)
        pooled_list = []
        for i in range(len(data_to_pool)):
            bd = data_to_pool[i]
            pooled_list.append(bd.data)

        pooled_array = np.stack(pooled_list)

        # Apply contrast if specified
        if contrast is not None:
            pooled_array = self._apply_contrast(pooled_array, contrast)

        # Save per-subject if requested
        if save:
            self._save_per_subject(save)

        # Get subject IDs from metadata if available
        subject_ids = None
        if self._bc.metadata is not None and "subject" in self._bc.metadata.columns:
            subject_ids = list(self._bc.metadata["subject"])

        # Get condition names from fitted results if available
        condition_names = self._condition_names
        if condition_names is None and hasattr(data_to_pool, "_design_columns"):
            condition_names = data_to_pool._design_columns

        return PooledData(
            data=pooled_array,
            param=param,
            condition_names=condition_names,
            subject_ids=subject_ids,
            mask=self._bc.mask,
            fitted_state=self._fitted if save_fitted else None,
            save_path=save,
        )

    def _apply_contrast(self, data: np.ndarray, contrast: str) -> np.ndarray:
        """Apply contrast weights to pooled data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_subjects, n_conditions, n_voxels).
        contrast : str
            Contrast specification like 'A-B' or 'A+B'.

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_voxels) after contrast.
        """
        if self._condition_names is None:
            raise ValueError(
                "Cannot apply contrast: condition_names not available. "
                "Ensure fit() received a design matrix with column names."
            )

        # Parse contrast string
        contrast_weights = self._parse_contrast(contrast)

        # Apply contrast: weighted sum across conditions
        result = np.zeros((data.shape[0], data.shape[2]))  # (n_subjects, n_voxels)
        for i, (cond, weight) in enumerate(contrast_weights.items()):
            if cond not in self._condition_names:
                raise ValueError(
                    f"Condition '{cond}' not in conditions: {self._condition_names}"
                )
            idx = self._condition_names.index(cond)
            result += weight * data[:, idx, :]

        return result

    def _parse_contrast(self, contrast: str) -> dict[str, float]:
        """Parse contrast string into weights dict.

        Examples:
        - 'A-B' -> {'A': 1.0, 'B': -1.0}
        - 'A+B' -> {'A': 1.0, 'B': 1.0}
        - '2*A-B' -> {'A': 2.0, 'B': -1.0}
        """
        import re

        weights = {}
        # Split by + or - keeping the delimiter
        parts = re.split(r"(\+|-)", contrast)

        sign = 1.0
        for part in parts:
            part = part.strip()
            if part == "+":
                sign = 1.0
            elif part == "-":
                sign = -1.0
            elif part:
                # Check for coefficient
                match = re.match(r"(\d*\.?\d*)\*?(.+)", part)
                if match:
                    coef_str, name = match.groups()
                    coef = float(coef_str) if coef_str else 1.0
                    weights[name.strip()] = sign * coef
                else:
                    weights[part] = sign

        return weights

    def _save_per_subject(self, save_dir: str) -> None:
        """Save each subject's fitted results."""
        from pathlib import Path

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        data_to_save = (
            self._fitted
            if not isinstance(self._fitted, dict)
            else self._fitted.get("betas", next(iter(self._fitted.values())))
        )

        for i in range(len(data_to_save)):
            bd = data_to_save[i]
            subj_path = save_path / f"subj{i + 1:02d}.nii.gz"
            bd.write(str(subj_path))

    def __repr__(self) -> str:
        if isinstance(self._fitted, dict):
            keys = list(self._fitted.keys())
            return (
                f"FittedBrainCollection(n_subjects={self.n_subjects}, "
                f"model='{self._model}', outputs={keys})"
            )
        return (
            f"FittedBrainCollection(n_subjects={self.n_subjects}, "
            f"model='{self._model}')"
        )

    # Delegate common operations to underlying results for backward compatibility
    def __len__(self) -> int:
        return self.n_subjects

    def __getitem__(self, key):
        """Allow indexing into fitted results."""
        if isinstance(self._fitted, dict):
            if isinstance(key, str):
                return self._fitted[key]
            # Numeric index - apply to all values
            return {k: v[key] for k, v in self._fitted.items()}
        return self._fitted[key]

    def __iter__(self):
        """Iterate over fitted results."""
        if isinstance(self._fitted, dict):
            return iter(self._fitted)
        return iter(self._fitted)
