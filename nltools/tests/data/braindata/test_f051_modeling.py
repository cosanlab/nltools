"""Regression test for non-mutating ridge fits.

F051 (``_assemble_ridge_cv_results`` mapping the selected alpha back onto an
unsorted alpha grid with ``np.searchsorted``) no longer has a subject: the
facade runs no second cross-validation pass, and ``Ridge`` reports the
selection itself as ``alpha_`` and ``cv_scores_``.
"""

import numpy as np


class TestFitInplaceFalseContract:
    """A non-inplace fit leaves every part of the original untouched."""

    def test_inplace_false_leaves_data_and_result_attrs_unmutated(
        self, minimal_brain_data
    ):
        brain = minimal_brain_data.copy()
        for attr in [
            "ridge_weights",
            "ridge_fitted_values",
            "ridge_r2",
            "model_",
        ]:
            if hasattr(brain, attr):
                delattr(brain, attr)

        X = np.random.randn(len(brain), 10)
        original = brain.data.copy()

        fitted = brain.fit(model="ridge", ridge_alpha=1.0, X=X, inplace=False)

        np.testing.assert_array_equal(brain.data, original)
        assert not hasattr(brain, "ridge_weights")
        assert not hasattr(brain, "model_")
        assert not hasattr(brain, "X_")
        preds = fitted.predict(X=np.random.randn(4, 10))
        assert preds.shape == (4, brain.shape[1])
