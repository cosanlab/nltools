"""BrainData pipeline and cross-validation result classes."""

import numpy as np


class BrainDataPipeline:
    """Pipeline specialized for BrainData with CV support.

    Wraps the base Pipeline to handle BrainData-specific operations
    like splitting by samples and accessing the underlying data array.
    """

    def __init__(self, brain_data, cv=None, groups=None):
        from nltools.pipelines.base import FittedStack  # noqa: F401

        self._brain_data = brain_data
        self._cv = cv
        self._groups = groups
        self._steps = []

    @property
    def data(self):
        """Get underlying data array."""
        return self._brain_data.data

    @property
    def cv(self):
        """Cross-validation splitter for this pipeline.

        Returns:
            sklearn cross-validator or None: The cross-validation strategy
            set for this pipeline, or None if not configured.
        """
        return self._cv

    @property
    def n_steps(self) -> int:
        """Number of processing steps in this pipeline.

        Returns:
            int: The count of steps added to this pipeline.
        """
        return len(self._steps)

    def _add_step(self, step) -> "BrainDataPipeline":
        """Add step and return new pipeline (immutable)."""
        from copy import copy

        new = copy(self)
        new._steps = self._steps + [step]
        return new

    def normalize(self, method: str = "zscore", **kwargs) -> "BrainDataPipeline":
        """Add normalization step."""
        from nltools.pipelines.steps import NormalizeStep

        return self._add_step(NormalizeStep(method=method, **kwargs))

    def reduce(
        self, method: str = "pca", n_components: int | None = None, **kwargs
    ) -> "BrainDataPipeline":
        """Add dimensionality reduction step."""
        from nltools.pipelines.steps import ReduceStep

        return self._add_step(
            ReduceStep(method=method, n_components=n_components, **kwargs)
        )

    def pipe(self, transformer) -> "BrainDataPipeline":
        """Add custom sklearn transformer."""
        from nltools.pipelines.steps import PipeStep

        return self._add_step(PipeStep(transformer=transformer))

    def predict(self, y, algorithm: str = "ridge", **kwargs):
        """Execute pipeline with CV and return prediction results.

        This is a terminal method that executes the full pipeline.

        Args:
            y: Target variable (labels or continuous values).
            algorithm: Prediction algorithm. Options:
                - 'ridge': Ridge regression (continuous targets)
                - 'lasso': Lasso regression (continuous targets)
                - 'svr': Support Vector Regression (continuous targets)
                - 'svm': Support Vector Classification (categorical targets)
            **kwargs: Additional arguments passed to sklearn model constructor.
                For classification (svm), use class_weight='balanced' to handle
                imbalanced classes. See sklearn documentation for all options.

        Returns:
            BrainDataCVResult with scores, predictions, and fold information.

        Examples:
            Basic regression::

                result = brain.cv(5).predict(continuous_y, algorithm='ridge', alpha=1.0)

            Classification with balanced classes::

                result = brain.cv(5).predict(labels, algorithm='svm', class_weight='balanced')
        """
        from nltools.pipelines.base import FittedStack
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.svm import SVC, SVR

        if self._cv is None:
            raise ValueError("predict() requires CV context")

        data = self.data
        y = np.asarray(y)

        results = []
        for train_idx, test_idx in self._cv.split(data, groups=self._groups):
            train_data = data[train_idx]
            test_data = data[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]

            fitted_stack = FittedStack()

            # Apply transform steps
            for step in self._steps:
                fitted = step.fit(train_data)
                fitted_stack.append(fitted)
                train_data = fitted.transform(train_data)
                test_data = fitted.transform(test_data)

            # Fit predictor and evaluate
            if algorithm == "ridge":
                model = Ridge(**kwargs)
            elif algorithm == "lasso":
                model = Lasso(**kwargs)
            elif algorithm == "svr":
                model = SVR(**kwargs)
            elif algorithm == "svm":
                model = SVC(**kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

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

        return BrainDataCVResult(results, self)


class BrainDataCVResult:
    """Cross-validation results for BrainData pipelines."""

    def __init__(self, fold_results: list, pipeline):
        self.fold_results = fold_results
        self.pipeline = pipeline

    @property
    def scores(self) -> np.ndarray:
        """Per-fold prediction scores as a numpy array."""
        return np.array([f["score"] for f in self.fold_results])

    @property
    def mean_score(self) -> float:
        """Mean score across folds."""
        return self.scores.mean()

    @property
    def std_score(self) -> float:
        """Standard deviation of scores."""
        return self.scores.std()

    @property
    def predictions(self) -> np.ndarray:
        """All predictions in original sample order."""
        # Reconstruct in original order
        n_samples = sum(len(f["test_idx"]) for f in self.fold_results)
        preds = np.zeros(n_samples)
        for f in self.fold_results:
            preds[f["test_idx"]] = f["predictions"]
        return preds

    def __repr__(self):
        return f"BrainDataCVResult(n_folds={len(self.fold_results)}, mean_score={self.mean_score:.4f})"
