"""Terminal operations for nltools pipelines.

Terminals are the final step in a pipeline that produce results.
They execute prediction, classification, or other evaluation tasks
within cross-validation folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .results import FoldResult, ISCResult, RSAResult


@dataclass
class PredictTerminal:
    """Prediction/classification terminal for CV pipelines.

    Fits a prediction model on training data and evaluates on test data
    within each CV fold.

    Args:
        y: Target variable to predict (labels or continuous values).
        algorithm: Prediction algorithm. Regression options: 'ridge' (default, L2),
            'lasso' (L1), 'elastic' (L1+L2), 'svr' (kernel-based), 'rf' (random forest,
            auto-detected). Classification options: 'svm' (kernel-based), 'logistic'
            (linear), 'rf' (auto-detected for discrete y).
        kwargs: Additional arguments passed to the sklearn model constructor.
            Common kwargs: ``class_weight='balanced'`` for imbalanced classification,
            ``C`` for regularization strength (svm, logistic), ``alpha`` for
            regularization strength (ridge, lasso, elastic).

    Examples
    --------
    Basic classification::

        >>> terminal = PredictTerminal(y=labels, algorithm='svm', kwargs={'C': 1.0})

    Balanced classification for imbalanced data::

        >>> terminal = PredictTerminal(
        ...     y=imbalanced_labels,
        ...     algorithm='svm',
        ...     kwargs={'class_weight': 'balanced'}
        ... )

    Logistic regression with balanced classes::

        >>> terminal = PredictTerminal(
        ...     y=binary_labels,
        ...     algorithm='logistic',
        ...     kwargs={'class_weight': 'balanced', 'C': 0.1}
        ... )
    """

    y: NDArray
    algorithm: str = "ridge"
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert y to numpy array if needed."""
        self.y = np.asarray(self.y)

    def _get_model(self):
        """Get sklearn model instance based on algorithm.

        Returns:
            Scikit-learn estimator instance configured with kwargs.

        Raises:
            ValueError: If algorithm is not recognized.
        """
        if self.algorithm == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(**self.kwargs)
        if self.algorithm == "lasso":
            from sklearn.linear_model import Lasso

            return Lasso(**self.kwargs)
        if self.algorithm == "elastic":
            from sklearn.linear_model import ElasticNet

            return ElasticNet(**self.kwargs)
        if self.algorithm == "svr":
            from sklearn.svm import SVR

            return SVR(**self.kwargs)
        if self.algorithm == "svm":
            from sklearn.svm import SVC

            return SVC(**self.kwargs)
        if self.algorithm == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**self.kwargs)
        if self.algorithm == "rf":
            # Auto-detect classification vs regression
            unique_y = np.unique(self.y)
            if len(unique_y) <= 20 and np.all(unique_y == unique_y.astype(int)):
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(**self.kwargs)
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**self.kwargs)
        raise ValueError(
            f"Unknown algorithm: {self.algorithm}. "
            f"Options: ridge, lasso, elastic, svr, svm, logistic, rf"
        )

    def fit_evaluate(
        self,
        train_data: NDArray,
        test_data: NDArray,
        train_idx: NDArray[np.intp],
        test_idx: NDArray[np.intp],
        fitted_stack: Any,
    ) -> FoldResult:
        """Fit model on training data and evaluate on test data.

        Args:
            train_data: Transformed training features, shape (n_train, n_features).
            test_data: Transformed test features, shape (n_test, n_features).
            train_idx: Original indices of training samples.
            test_idx: Original indices of test samples.
            fitted_stack: Stack of fitted transforms for this fold.

        Returns:
            Result containing score, predictions, indices, and fitted stack.
        """
        from .results import FoldResult

        # Get target values for train/test
        train_y = self.y[train_idx]
        test_y = self.y[test_idx]

        # Fit model
        model = self._get_model()
        model.fit(train_data, train_y)

        # Predict and score
        predictions = model.predict(test_data)
        score = model.score(test_data, test_y)

        return FoldResult(
            score=score,
            predictions=predictions,
            train_idx=train_idx,
            test_idx=test_idx,
            fitted_stack=fitted_stack,
        )

    def with_y(self, new_y: NDArray) -> PredictTerminal:
        """Create copy with different target variable.

        Useful for permutation testing.

        Args:
            new_y: New target variable.

        Returns:
            New terminal with updated y.
        """
        return PredictTerminal(y=new_y, algorithm=self.algorithm, kwargs=self.kwargs)


@dataclass
class ISCTerminal:
    """ISC terminal for multi-subject pipelines.

    Computes inter-subject correlation across subjects in the pipeline.
    Uses the ISC permutation test from nltools.algorithms.inference.isc.

    Args:
        method: ISC computation method: 'pairwise' (default) or 'leave-one-out'.
        metric: Summary statistic: 'median' (default, robust) or 'mean'
            (Fisher z-transformed).
        n_permute: Number of bootstrap iterations for p-value computation.
            Default is 5000.
        parallel: Parallelization method: 'cpu' (default), 'gpu', or None.
        kwargs: Additional arguments passed to isc_permutation_test.

    Examples
    --------
    >>> terminal = ISCTerminal(method='pairwise', n_permute=1000)
    >>> result = terminal.fit_evaluate(data_list)
    >>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")
    """

    method: str = "pairwise"
    metric: str = "median"
    n_permute: int = 5000
    parallel: str = "cpu"
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate parameters."""
        if self.method not in ("pairwise", "leave-one-out"):
            raise ValueError(
                f"method must be 'pairwise' or 'leave-one-out', got {self.method!r}"
            )
        if self.metric not in ("median", "mean"):
            raise ValueError(f"metric must be 'median' or 'mean', got {self.metric!r}")
        if self.parallel not in (None, "cpu", "gpu"):
            raise ValueError(
                f"parallel must be None, 'cpu', or 'gpu', got {self.parallel!r}"
            )

    def fit_evaluate(
        self,
        data: list,
        **kwargs,
    ) -> ISCResult:
        """Compute ISC across subjects.

        Args:
            data: List of subject data arrays. Each array should have shape
                (n_observations, n_features) where n_observations is the same
                across subjects (e.g., timepoints in fMRI).

        Returns:
            Result containing ISC values, p-values, and confidence intervals.
        """
        from ..algorithms.inference.isc import isc_permutation_test
        from .results import ISCResult

        # Validate input
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError("data must be a list of at least 2 subject arrays")

        # Convert list to 3D array: (n_observations, n_subjects, n_features)
        # Each subject array is (n_observations, n_features)
        n_subjects = len(data)
        data_arrays = [np.asarray(d) for d in data]

        # Validate shapes
        n_obs = data_arrays[0].shape[0]
        for i, d in enumerate(data_arrays):
            if d.shape[0] != n_obs:
                raise ValueError(
                    f"All subjects must have same number of observations. "
                    f"Subject 0 has {n_obs}, subject {i} has {d.shape[0]}"
                )

        # Stack into (n_observations, n_subjects, n_features)
        if data_arrays[0].ndim == 1:
            # Single feature per observation: (n_obs,) -> (n_obs, n_subjects)
            stacked = np.stack(data_arrays, axis=1)
        else:
            # Multiple features: (n_obs, n_features) -> (n_obs, n_subjects, n_features)
            stacked = np.stack(data_arrays, axis=1)

        # Map method name to summary_statistic parameter
        summary_statistic = self.method

        # Call ISC permutation test
        merged_kwargs = {**self.kwargs, **kwargs}
        result = isc_permutation_test(
            data=stacked,
            n_permute=self.n_permute,
            metric=self.metric,
            summary_statistic=summary_statistic,
            parallel=self.parallel,
            **merged_kwargs,
        )

        return ISCResult(
            isc=result["isc"],
            p=result["p"],
            ci=result["ci"],
            method=self.method,
            metric=self.metric,
            n_subjects=n_subjects,
        )


@dataclass
class RSATerminal:
    """RSA terminal for multi-subject pipelines.

    Computes representational similarity analysis by correlating neural RDMs
    with a model RDM.

    Args:
        model_rdm: Model RDM to correlate with neural RDMs. Should be a symmetric
            matrix or upper triangle (condensed form).
        method: Correlation method: 'spearman' (default), 'pearson', or 'kendall'.
        n_permute: Number of permutations for p-value computation. Default is 5000.
        kwargs: Additional arguments passed to correlation computation.

    Examples
    --------
    >>> model = np.random.rand(10, 10)  # 10 conditions
    >>> model = (model + model.T) / 2  # Make symmetric
    >>> terminal = RSATerminal(model_rdm=model, method='spearman')
    >>> result = terminal.fit_evaluate(neural_rdm)
    """

    model_rdm: NDArray
    method: str = "spearman"
    n_permute: int = 5000
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and prepare model RDM."""
        self.model_rdm = np.asarray(self.model_rdm)

        if self.method not in ("spearman", "pearson", "kendall"):
            raise ValueError(
                f"method must be 'spearman', 'pearson', or 'kendall', "
                f"got {self.method!r}"
            )

        # Convert to condensed form if square matrix
        if self.model_rdm.ndim == 2:
            if self.model_rdm.shape[0] != self.model_rdm.shape[1]:
                raise ValueError("model_rdm must be square if 2D")
            from scipy.spatial.distance import squareform

            self.model_rdm = squareform(self.model_rdm, checks=False)
        elif self.model_rdm.ndim != 1:
            raise ValueError("model_rdm must be 1D (condensed) or 2D (square)")

    def fit_evaluate(
        self,
        data: NDArray,
        **kwargs,
    ) -> RSAResult:
        """Compute RSA correlation between neural and model RDMs.

        Args:
            data: Neural data to compute RDM from, or pre-computed RDM.
                If 2D square, treated as RDM (upper triangle extracted). If 1D,
                treated as condensed RDM. If 2D non-square (n_conditions, n_features),
                RDM is computed using correlation distance.

        Returns:
            Result containing correlation coefficient and p-value.
        """
        from scipy.spatial.distance import squareform
        from scipy.stats import spearmanr, pearsonr, kendalltau
        from sklearn.utils import check_random_state
        from .results import RSAResult

        data = np.asarray(data)

        # Handle input data format
        if data.ndim == 2:
            if data.shape[0] == data.shape[1]:
                # Square matrix - assume it's an RDM
                neural_rdm = squareform(data, checks=False)
            else:
                # (n_conditions, n_features) - compute RDM
                from sklearn.metrics import pairwise_distances

                dist_matrix = pairwise_distances(data, metric="correlation")
                neural_rdm = squareform(dist_matrix, checks=False)
        elif data.ndim == 1:
            # Already condensed
            neural_rdm = data
        else:
            raise ValueError("data must be 1D (condensed RDM) or 2D")

        # Validate size match
        if len(neural_rdm) != len(self.model_rdm):
            # Compute expected n_conditions from length
            # n_pairs = n*(n-1)/2, so n = (1 + sqrt(1 + 8*n_pairs)) / 2
            n_model = int((1 + np.sqrt(1 + 8 * len(self.model_rdm))) / 2)
            n_neural = int((1 + np.sqrt(1 + 8 * len(neural_rdm))) / 2)
            raise ValueError(
                f"RDM size mismatch: model has {n_model} conditions, "
                f"neural has {n_neural} conditions"
            )

        # Compute correlation
        if self.method == "spearman":
            corr_func = lambda x, y: spearmanr(x, y)
        elif self.method == "pearson":
            corr_func = lambda x, y: pearsonr(x, y)
        else:  # kendall
            corr_func = lambda x, y: kendalltau(x, y)

        observed_corr, _ = corr_func(neural_rdm, self.model_rdm)

        # Permutation test
        merged_kwargs = {**self.kwargs, **kwargs}
        random_state = merged_kwargs.get("random_state", None)
        rng = check_random_state(random_state)

        n_conditions = int((1 + np.sqrt(1 + 8 * len(self.model_rdm))) / 2)
        null_dist = np.zeros(self.n_permute)

        for i in range(self.n_permute):
            # Permute row/column order of the model RDM
            # This is equivalent to permuting condition labels
            perm_idx = rng.permutation(n_conditions)

            # Reconstruct square matrix, permute, and re-extract upper triangle
            model_square = squareform(self.model_rdm)
            model_perm = model_square[np.ix_(perm_idx, perm_idx)]
            model_rdm_perm = squareform(model_perm, checks=False)

            null_dist[i], _ = corr_func(neural_rdm, model_rdm_perm)

        # Compute p-value (two-tailed)
        # Phipson-Smyth correction: (b + 1) / (m + 1)
        n_extreme = np.sum(np.abs(null_dist) >= np.abs(observed_corr))
        p_value = (n_extreme + 1) / (self.n_permute + 1)

        # Compute confidence interval from null distribution
        ci_lower = np.percentile(null_dist, 2.5)
        ci_upper = np.percentile(null_dist, 97.5)

        return RSAResult(
            correlation=float(observed_corr),
            p_value=float(p_value),
            ci=(float(ci_lower), float(ci_upper)),
            method=self.method,
            n_conditions=n_conditions,
        )


__all__ = ["ISCTerminal", "PredictTerminal", "RSATerminal"]
