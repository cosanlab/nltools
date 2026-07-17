"""Regression tests for F050 / F051 in braindata/modeling.py.

- F051: ``_assemble_ridge_cv_results`` used ``np.searchsorted`` to map the
  per-voxel selected alpha back to a column of the ``(n_splits, n_alphas,
  n_voxels)`` score cube. searchsorted assumes an ascending grid; with
  user-supplied unsorted ``alphas`` it returns the wrong index, so the
  reported per-fold CV scores correspond to the wrong alpha.
- F050: ``fit(inplace=False)`` documentation/behavior contract (see the
  docstring test below).
"""

import numpy as np


class TestRidgeCvUnsortedAlphas:
    def test_scores_match_selected_alpha_with_unsorted_alphas(
        self, small_brain_data_for_cv
    ):
        """Per-fold scores must correspond to each voxel's SELECTED alpha,
        even when the alpha grid is not sorted ascending."""
        brain, X = small_brain_data_for_cv

        brain.fit(
            model="ridge",
            alpha="auto",
            alphas=[10.0, 0.1, 1.0],  # deliberately unsorted
            local_alpha=True,
            cv=3,
            X=X,
        )

        cv = brain.cv_results_
        alpha_scores = cv["alpha_scores"]  # (n_splits, n_alphas, n_voxels)
        best_alpha = np.atleast_1d(np.asarray(cv["best_alpha"], dtype=float))
        alpha_grid = np.asarray(brain.model_.alphas, dtype=float)

        n_splits, n_alphas, n_voxels = alpha_scores.shape
        if best_alpha.shape[0] == 1 and n_voxels > 1:
            best_alpha = np.full(n_voxels, best_alpha[0])

        # Independently recover the per-voxel column by VALUE (nearest alpha),
        # which is order-independent and correct.
        expected_idx = np.argmin(
            np.abs(alpha_grid[:, None] - best_alpha[None, :]), axis=0
        )
        expected_scores = np.stack(
            [alpha_scores[:, expected_idx[v], v] for v in range(n_voxels)],
            axis=1,
        )

        np.testing.assert_allclose(cv["scores"], expected_scores)


class TestFitInplaceFalseDocstringContract:
    """F050: with inplace=False, `.data` and the ridge_*/glm_* result
    attributes stay unmutated, but model_/X_ ARE set on self so predict()
    works. This codifies the (now accurately documented) behavior."""

    def test_inplace_false_leaves_data_and_result_attrs_unmutated(
        self, minimal_brain_data
    ):
        brain = minimal_brain_data.copy()
        for attr in [
            "ridge_weights",
            "ridge_fitted_values",
            "ridge_scores",
            "model_",
            "X_",
        ]:
            if hasattr(brain, attr):
                delattr(brain, attr)

        X = np.random.randn(len(brain), 10)
        original = brain.data.copy()

        brain.fit(model="ridge", alpha=1.0, X=X, inplace=False)

        # Data and result attributes are NOT mutated...
        np.testing.assert_array_equal(brain.data, original)
        assert not hasattr(brain, "ridge_weights")
        # ...but model_/X_ ARE set so predict() can run on self.
        assert hasattr(brain, "model_")
        assert hasattr(brain, "X_")
        preds = brain.predict(X=np.random.randn(4, 10))
        assert preds.shape == (4, brain.shape[1])
