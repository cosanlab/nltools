"""Terminal operations for nltools pipelines.

Terminals are the final step in a pipeline that produce results.
They execute prediction, classification, or other evaluation tasks
within cross-validation folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .results import FoldResult


@dataclass
class PredictTerminal:
    """Prediction/classification terminal for CV pipelines.

    Fits a prediction model on training data and evaluates on test data
    within each CV fold.

    Parameters
    ----------
    y : np.ndarray
        Target variable to predict (labels or continuous values).
    algorithm : str
        Prediction algorithm. Options:
        - 'ridge': Ridge regression (default)
        - 'lasso': Lasso regression
        - 'elastic': ElasticNet regression
        - 'svr': Support Vector Regression
        - 'svm': Support Vector Classification
        - 'logistic': Logistic Regression
        - 'rf': Random Forest (auto-detects classification vs regression)
    kwargs : dict
        Additional arguments passed to the model constructor.

    Examples
    --------
    >>> terminal = PredictTerminal(y=labels, algorithm='svm', C=1.0)
    >>> fold_result = terminal.fit_evaluate(train_X, test_X, train_idx, test_idx, stack)
    """

    y: NDArray
    algorithm: str = "ridge"
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert y to numpy array if needed."""
        self.y = np.asarray(self.y)

    def _get_model(self):
        """Get sklearn model instance based on algorithm.

        Returns
        -------
        model
            Scikit-learn estimator instance configured with kwargs.

        Raises
        ------
        ValueError
            If algorithm is not recognized.
        """
        if self.algorithm == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(**self.kwargs)
        elif self.algorithm == "lasso":
            from sklearn.linear_model import Lasso

            return Lasso(**self.kwargs)
        elif self.algorithm == "elastic":
            from sklearn.linear_model import ElasticNet

            return ElasticNet(**self.kwargs)
        elif self.algorithm == "svr":
            from sklearn.svm import SVR

            return SVR(**self.kwargs)
        elif self.algorithm == "svm":
            from sklearn.svm import SVC

            return SVC(**self.kwargs)
        elif self.algorithm == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**self.kwargs)
        elif self.algorithm == "rf":
            # Auto-detect classification vs regression
            unique_y = np.unique(self.y)
            if len(unique_y) <= 20 and np.all(unique_y == unique_y.astype(int)):
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(**self.kwargs)
            else:
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(**self.kwargs)
        else:
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
    ) -> "FoldResult":
        """Fit model on training data and evaluate on test data.

        Parameters
        ----------
        train_data : np.ndarray
            Transformed training features, shape (n_train, n_features).
        test_data : np.ndarray
            Transformed test features, shape (n_test, n_features).
        train_idx : np.ndarray
            Original indices of training samples.
        test_idx : np.ndarray
            Original indices of test samples.
        fitted_stack : FittedStack
            Stack of fitted transforms for this fold.

        Returns
        -------
        FoldResult
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

    def with_y(self, new_y: NDArray) -> "PredictTerminal":
        """Create copy with different target variable.

        Useful for permutation testing.

        Parameters
        ----------
        new_y : np.ndarray
            New target variable.

        Returns
        -------
        PredictTerminal
            New terminal with updated y.
        """
        return PredictTerminal(y=new_y, algorithm=self.algorithm, kwargs=self.kwargs)


__all__ = ["PredictTerminal"]
