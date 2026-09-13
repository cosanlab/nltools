import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from nltools.data import BrainData, DesignMatrix


def _brain_data_from_array(values):
    """Build a BrainData whose masked data holds `values`, shape (n_obs, n_voxels).

    Images are passed as a list so a one-voxel stack keeps its voxel axis (the
    constructor squeezes singleton axes out of array and 4-D inputs).
    """
    import nibabel as nib

    values = np.asarray(values, dtype=float)
    n_samples, n_voxels = values.shape
    spatial_shape = (n_voxels, 1, 1)
    affine = np.eye(4)
    images = [
        nib.Nifti1Image(values[t].reshape(spatial_shape), affine)
        for t in range(n_samples)
    ]
    return BrainData(
        images,
        mask=nib.Nifti1Image(np.ones(spatial_shape, dtype=np.float32), affine),
    )


class TestBrainDataModeling:
    # ==================== Unified fit/predict API ====================

    def test_fit_predict_ridge_workflow(self, minimal_brain_data):
        """Test complete Ridge fit/predict workflow."""
        from nltools.models import _Ridge

        X_train = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", ridge_alpha=1.0, X=X_train)

        # Check model stored
        assert hasattr(minimal_brain_data, "model_")
        assert isinstance(minimal_brain_data.model_, _Ridge)
        assert minimal_brain_data.model_.is_fitted_

        # Check attributes set
        assert hasattr(minimal_brain_data, "ridge_weights")
        assert hasattr(minimal_brain_data, "ridge_fitted_values")
        assert hasattr(minimal_brain_data, "ridge_r2")

        # Predict on new data
        X_test = np.random.randn(20, 10)
        predictions = minimal_brain_data.predict(X=X_test)
        assert isinstance(predictions, BrainData)
        assert predictions.shape == (20, minimal_brain_data.shape[1])

        # Predict on training data (X=None) uses self.data as target
        train_predictions = minimal_brain_data.predict()
        assert train_predictions.shape == minimal_brain_data.shape

    @pytest.mark.slow
    def test_fit_predict_glm_workflow(self, minimal_brain_data):
        """Test complete GLM fit/predict workflow."""
        from nltools.data import DesignMatrix
        from nltools.models import _Glm

        design_matrix = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", glm_noise_model="ols", X=design_matrix)

        assert hasattr(minimal_brain_data, "model_")
        assert isinstance(minimal_brain_data.model_, _Glm)
        assert hasattr(minimal_brain_data, "glm_betas")

        predictions = minimal_brain_data.predict()
        assert predictions.shape == minimal_brain_data.shape

    @pytest.mark.slow
    def test_fit_forwards_model_options_to_the_estimator(self, minimal_brain_data):
        """Prefixed and ridge options reach their estimator's constructor."""
        from nltools.data import DesignMatrix

        X = np.random.randn(len(minimal_brain_data), 10)

        minimal_brain_data.fit(model="ridge", ridge_alpha=1.0, ridge_device="cpu", X=X)
        assert minimal_brain_data.model_.device == "cpu"

        design_matrix = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", glm_noise_model="ar1", X=design_matrix)
        assert minimal_brain_data.model_.noise_model == "ar1"

    def test_predict_requires_fitted_model(self, minimal_brain_data):
        """Test predict() raises error if fit() not called first."""
        bd = minimal_brain_data.copy()
        if hasattr(bd, "model_"):
            del bd.model_

        with pytest.raises(ValueError, match="Must call fit"):
            bd.predict()

    def test_predict_validates_X_dimensions(self, minimal_brain_data):
        """Test predict() validates X has correct n_features."""
        X_train = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", ridge_alpha=1.0, X=X_train)

        X_wrong = np.random.randn(15, 5)
        with pytest.raises(ValueError, match="features"):
            minimal_brain_data.predict(X=X_wrong)

    # ==================== Fit inplace parameter tests ====================

    def test_fit_inplace_default_true(self, minimal_brain_data):
        """Test inplace=True (default) preserves backward compatibility."""
        X_train = np.random.randn(len(minimal_brain_data), 10)

        # Default (inplace=True)
        result = minimal_brain_data.fit(model="ridge", ridge_alpha=1.0, X=X_train)
        assert result is minimal_brain_data
        assert hasattr(minimal_brain_data, "ridge_weights")
        assert hasattr(minimal_brain_data, "ridge_fitted_values")
        assert hasattr(minimal_brain_data, "ridge_r2")
        assert hasattr(minimal_brain_data, "model_")
        assert not hasattr(minimal_brain_data, "X_")
        assert minimal_brain_data.model_.progress_bar is False

    def test_fit_inplace_false_returns_independent_fitted_brain_data(
        self, minimal_brain_data
    ):
        """A non-inplace ridge fit owns its complete fitted state."""
        brain = minimal_brain_data.copy()
        for attr in [
            "ridge_weights",
            "ridge_fitted_values",
            "ridge_r2",
            "glm_betas",
            "glm_residual",
            "glm_predicted",
            "glm_r2",
            "model_",
        ]:
            if hasattr(brain, attr):
                delattr(brain, attr)

        X_train = np.random.randn(len(brain), 10)
        original_data = brain.data.copy()

        fitted = brain.fit(model="ridge", ridge_alpha=1.0, X=X_train, inplace=False)

        assert isinstance(fitted, BrainData)
        assert fitted is not brain
        assert fitted.ridge_fitted_values.shape == brain.shape
        assert fitted.ridge_weights.shape == (10, brain.shape[1])
        assert fitted.ridge_r2.shape == (1, brain.shape[1])
        assert not hasattr(brain, "ridge_weights")
        assert not hasattr(brain, "model_")
        assert not hasattr(brain, "X_")
        np.testing.assert_array_equal(brain.data, original_data)

        fitted.data[0, 0] = 123.0
        fitted.ridge_weights.data[0, 0] = 789.0
        assert brain.data[0, 0] != 123.0
        assert fitted.model_.coef_[0, 0] != 789.0
        assert not hasattr(fitted.ridge_weights, "model_")

    def test_copy_owns_nested_cv_state(self, minimal_brain_data):
        X = np.random.randn(len(minimal_brain_data), 4)
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=[0.1, 1.0], ridge_cv=3)

        copied = minimal_brain_data.copy()
        copied.model_.cv_scores_[0] = 91.0
        copied.model_.alpha_[0] = 92.0

        assert minimal_brain_data.model_.cv_scores_[0] != 91.0
        assert minimal_brain_data.model_.alpha_[0] != 92.0

    @pytest.mark.slow
    def test_refit_replaces_prior_model_state(self, minimal_brain_data):
        """Refitting across model types cannot leave mixed result state."""
        ridge_X = np.random.randn(len(minimal_brain_data), 3)
        minimal_brain_data.fit(model="ridge", X=ridge_X, ridge_alpha=1.0)

        glm_X = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condition": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", X=glm_X, glm_noise_model="ols")

        assert hasattr(minimal_brain_data, "glm_betas")
        assert not hasattr(minimal_brain_data, "ridge_weights")
        assert not hasattr(minimal_brain_data, "ridge_fitted_values")
        assert not hasattr(minimal_brain_data, "ridge_r2")

        minimal_brain_data.fit(model="ridge", X=ridge_X, ridge_alpha=1.0)

        assert hasattr(minimal_brain_data, "ridge_weights")
        assert not hasattr(minimal_brain_data, "glm_betas")
        assert not hasattr(minimal_brain_data, "glm_predicted")

    @pytest.mark.slow
    def test_glm_fit_numerical_correctness(self, minimal_brain_data):
        """Test fit(model='glm') produces numerically correct results."""
        design_matrix = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", glm_noise_model="ols", X=design_matrix)

        assert not np.isnan(minimal_brain_data.glm_betas.data).any()
        assert not np.allclose(minimal_brain_data.glm_betas.data, 0)
        assert not np.isnan(minimal_brain_data.glm_r2.data).any()

    def test_fit_validates_model_name(self, minimal_brain_data):
        """Test fit() raises error for unknown model names."""
        X = np.random.randn(len(minimal_brain_data), 10)
        with pytest.raises(TypeError, match="supported models are"):
            minimal_brain_data.fit(model="unknown_model", X=X)

    def test_fit_validates_X_shape(self, minimal_brain_data):
        """Test fit() validates X has correct n_samples."""
        X_wrong = np.random.randn(len(minimal_brain_data) + 5, 10)
        with pytest.raises(ValueError, match="number of samples"):
            minimal_brain_data.fit(model="ridge", ridge_alpha=1.0, X=X_wrong)

    def test_predict_with_no_X_uses_training_data(self, minimal_brain_data):
        """Test predict() with no X returns predictions on training data."""
        X_train = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", ridge_alpha=1.0, X=X_train)

        predictions_explicit = minimal_brain_data.predict(X=X_train)
        predictions_implicit = minimal_brain_data.predict()

        np.testing.assert_allclose(predictions_explicit.data, predictions_implicit.data)
        assert predictions_implicit.shape == minimal_brain_data.shape

    # ==================== Ridge CV Tests ====================

    def test_fit_ridge_cv_selects_alphas(self, small_brain_data_for_cv):
        """A sequence `ridge_alpha` with `ridge_cv` selects per-voxel alphas."""
        brain_data, X = small_brain_data_for_cv
        alphas = [0.1, 1.0, 10.0]

        brain_data.fit(model="ridge", ridge_alpha=alphas, ridge_cv=3, X=X)

        model = brain_data.model_
        assert model.alpha_.shape == (5,)
        assert np.all(np.isin(model.alpha_, alphas))
        assert model.cv_scores_.shape == (5,)
        assert hasattr(brain_data, "ridge_weights")
        assert not hasattr(brain_data, "cv_results_")

    def test_fit_ridge_cv_accepts_a_splitter(self, small_brain_data_for_cv):
        brain_data, X = small_brain_data_for_cv
        splitter = KFold(n_splits=3, shuffle=True, random_state=42)

        brain_data.fit(
            model="ridge", ridge_alpha=[0.1, 1.0, 10.0], ridge_cv=splitter, X=X
        )

        assert brain_data.model_.alpha_.shape == (5,)

    def test_fit_ridge_shared_alpha_is_scalar(self, small_brain_data_for_cv):
        """`ridge_per_target_alpha=False` collapses the selection to one alpha."""
        brain_data, X = small_brain_data_for_cv

        brain_data.fit(
            model="ridge",
            ridge_alpha=[0.1, 1.0, 10.0],
            ridge_cv=3,
            ridge_per_target_alpha=False,
            X=X,
        )

        assert np.ndim(brain_data.model_.alpha_) == 0

    # ============ design estimated as given + rank diagnostics (GLM) ============

    @pytest.mark.slow
    def test_fit_estimates_every_column_given(self, minimal_brain_data):
        """fit() does not silently drop correlated regressors.

        Two regressors correlated at r=0.99 are collinear in the colloquial
        sense but the design is still full rank and estimable, so every column
        must appear in the betas. Dropping is the caller's decision, made
        explicitly via `DesignMatrix.clean()`.
        """
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        b = a + 0.1 * rng.standard_normal(n)
        design_matrix = DesignMatrix({"Intercept": np.ones(n), "condA": a, "condB": b})
        r = abs(np.corrcoef(a, b)[0, 1])
        assert r > 0.95, f"setup invariant violated: |r|={r}"
        assert np.linalg.matrix_rank(design_matrix.to_numpy()) == 3

        minimal_brain_data.fit(model="glm", X=design_matrix)
        assert minimal_brain_data.glm_betas.shape[0] == 3

    @pytest.mark.slow
    def test_rank_deficient_design_warns(self, minimal_brain_data):
        """A singular design produces non-unique betas, so warn loudly."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        design_matrix = DesignMatrix(
            {"Intercept": np.ones(n), "condA": a, "condA_dup": a}
        )
        assert np.linalg.matrix_rank(design_matrix.to_numpy()) == 2

        with pytest.warns(UserWarning, match="rank deficient"):
            minimal_brain_data.fit(model="glm", X=design_matrix)

        # Still fits (pseudo-inverse), and keeps every column.
        assert minimal_brain_data.glm_betas.shape[0] == 3

    @pytest.mark.slow
    def test_rank_deficient_warning_names_the_columns(self, minimal_brain_data):
        """The warning must be actionable: report rank, size, and next step."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        design_matrix = DesignMatrix(
            {"Intercept": np.ones(n), "condA": a, "condA_dup": a}
        )
        with pytest.warns(UserWarning) as record:
            minimal_brain_data.fit(model="glm", X=design_matrix)
        msg = "\n".join(str(w.message) for w in record)
        assert "2" in msg and "3" in msg  # rank 2 of 3
        # Regularization is the recommended fix: ridge has a unique solution
        # even when X'X is singular, and is invariant to column order.
        assert "ridge" in msg
        assert "vif" in msg.lower()
        assert "clean" in msg


class TestWarnRankDeficient:
    """Unit tests for the rank-deficiency check `fit(model='glm')` runs.

    The helper is pure (array in, warning out), so these run without brain
    data. Integration coverage through `fit()` lives in TestBrainDataModeling
    (slow-marked).
    """

    @staticmethod
    def _check(X):
        from nltools.data.braindata.modeling import _warn_if_rank_deficient

        _warn_if_rank_deficient(np.asarray(X, dtype=float))

    def test_warns_with_named_category(self):
        from nltools.utils import DesignMatrixWarning

        rng = np.random.default_rng(0)
        a = rng.standard_normal(30)
        with pytest.warns(DesignMatrixWarning, match="rank deficient"):
            self._check(np.column_stack([np.ones(30), a, a]))

    def test_message_counts_the_dependent_columns(self):
        """The diagnosis counts the dependent columns rather than naming them.

        Which member of a dependent set QR would blame is arbitrary, so a name
        was a guess dressed as a finding; the count is what is actually known.
        """
        rng = np.random.default_rng(0)
        a = rng.standard_normal(30)
        with pytest.warns(UserWarning) as record:
            self._check(np.column_stack([np.ones(30), a, a]))
        msg = str(record[0].message)
        assert "rank 2 of 3 columns" in msg
        assert "1 column(s) are linear combinations of the others" in msg
        # No column is singled out as the culprit.
        assert "likely involved" not in msg


class TestBrainDataTTest:
    def test_ttest_one_sample(self, minimal_brain_data):
        """One-sample t-test returns mean + t + z + p (all as BrainData)."""
        from scipy.stats import ttest_1samp

        result = minimal_brain_data.ttest()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"mean", "t", "z", "p"}
        for key in ("mean", "t", "z", "p"):
            assert isinstance(result[key], BrainData)

        expected_t, expected_p = ttest_1samp(minimal_brain_data.data, 0.0, axis=0)
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(result["p"].data, expected_p)
        np.testing.assert_allclose(
            result["mean"].data, minimal_brain_data.data.mean(axis=0)
        )
        assert result["t"].data.shape == (minimal_brain_data.data.shape[1],)

    def test_ttest2_removed(self):
        """ttest2 was a 0.6.0-dev convenience method; it must not exist."""
        assert not hasattr(BrainData, "ttest2")

    def test_ttest_parametric_honors_tail(self, minimal_brain_data):
        """tail=1 must reach the parametric path (was silently ignored pre-0.6.0)."""
        from scipy.stats import ttest_1samp

        result = minimal_brain_data.ttest(tail=1)
        _, expected_p = ttest_1samp(
            minimal_brain_data.data, 0.0, axis=0, alternative="greater"
        )
        np.testing.assert_allclose(result["p"].data, expected_p)
        with pytest.raises(ValueError, match="tail"):
            minimal_brain_data.ttest(tail="upper")

    def test_ttest_popmean(self, minimal_brain_data):
        """popmean kwarg shifts the null and the reported mean."""
        from scipy.stats import ttest_1samp

        result = minimal_brain_data.ttest(popmean=0.5)
        expected_t, _ = ttest_1samp(minimal_brain_data.data, 0.5, axis=0)
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(
            result["mean"].data, minimal_brain_data.data.mean(axis=0) - 0.5
        )

    def test_ttest_single_image_raises(self, minimal_brain_data):
        """t-test on a single image should raise."""
        single = minimal_brain_data[0]
        with pytest.raises(ValueError, match="multiple images"):
            single.ttest()

    def test_ttest_permutation(self, minimal_brain_data):
        """permutation=True reports empirical p but still returns mean/t/z/p."""
        result = minimal_brain_data.ttest(
            permutation=True, n_permute=50, random_state=0
        )
        assert set(result.keys()) == {"mean", "t", "z", "p"}
        for key in ("mean", "t", "z", "p"):
            assert isinstance(result[key], BrainData)
        np.testing.assert_allclose(
            result["mean"].data, minimal_brain_data.data.mean(axis=0)
        )

    def test_ttest_permutation_popmean_tests_shifted_hypothesis(
        self, minimal_brain_data
    ):
        """permutation=True must test mean != popmean, not mean != 0.

        Data centered at popmean is null under the requested hypothesis, so
        the sign-flip p-values must look uniform — a zero-referenced test
        (the pre-fix bug) would return the minimum p at every voxel.
        """
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 5.0  # ~N(5, 1) per voxel
        res = bd.ttest(popmean=5.0, permutation=True, n_permute=200, random_state=0)
        # Under the correct H0 (mean == 5) nothing should be at the floor
        # p = 1/(n_permute+1) ≈ 0.005; the zero-referenced bug puts every
        # voxel there.
        assert np.all(np.asarray(res["p"].data) > 0.02)
        # Sanity: the same data against popmean=0 is overwhelmingly significant.
        res0 = bd.ttest(popmean=0.0, permutation=True, n_permute=200, random_state=0)
        assert np.all(np.asarray(res0["p"].data) < 0.01)

    # ── Shared one-sample contract (development/specs/ttest.md) ────────

    @pytest.mark.parametrize("popmean", [0.0])
    @pytest.mark.parametrize("tail,alternative", [(2, "two-sided")])
    def test_ttest_parametric_matches_scipy(
        self, minimal_brain_data, popmean, tail, alternative
    ):
        """Parametric mean/t/p match direct NumPy/SciPy for both tails."""
        from scipy.stats import ttest_1samp

        data = minimal_brain_data.data
        result = minimal_brain_data.ttest(popmean=popmean, tail=tail)
        assert set(result) == {"mean", "t", "z", "p"}
        expected_t, expected_p = ttest_1samp(
            data, popmean, axis=0, alternative=alternative
        )
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(result["p"].data, expected_p)
        np.testing.assert_allclose(result["mean"].data, data.mean(axis=0) - popmean)

    def test_ttest_null_only_when_permuting_and_requested(self, minimal_brain_data):
        """`null_dist` appears only with permutation=True and return_null=True."""
        assert "null_dist" not in minimal_brain_data.ttest(return_null=True)
        assert "null_dist" not in minimal_brain_data.ttest(
            permutation=True, n_permute=16, random_state=0
        )
        with_null = minimal_brain_data.ttest(
            permutation=True, n_permute=16, return_null=True, random_state=0
        )
        without_null = minimal_brain_data.ttest(
            permutation=True, n_permute=16, return_null=False, random_state=0
        )
        assert set(with_null) == {"mean", "t", "z", "p", "null_dist"}
        for key in ("mean", "t", "z", "p"):
            np.testing.assert_array_equal(with_null[key].data, without_null[key].data)

    def test_ttest_permutation_reports_the_t_statistic(self, minimal_brain_data):
        """`t` is the observed SciPy statistic on the permutation path too."""
        from scipy.stats import ttest_1samp

        result = minimal_brain_data.ttest(
            permutation=True, n_permute=32, random_state=0
        )
        expected_t, _ = ttest_1samp(minimal_brain_data.data, 0.0, axis=0)
        np.testing.assert_allclose(result["t"].data, expected_t)

    def test_ttest_z_endpoints_stay_finite(self):
        """z is finite at both the p floor and p == 1.0."""
        rng = np.random.default_rng(5)
        bd = _brain_data_from_array(rng.standard_normal((30, 3)) * 1e-8 + 50.0)
        huge = bd.ttest()
        assert np.all(np.isfinite(huge["z"].data)) and np.all(huge["z"].data > 0)
        saturated = bd.ttest(popmean=100.0, tail=1)
        assert np.all(np.asarray(saturated["p"].data) == 1.0)
        assert np.all(np.isfinite(saturated["z"].data))
        assert np.all(np.asarray(saturated["z"].data) < 0)

    def test_ttest_results_are_owned_and_rows_cleared(self, minimal_brain_data):
        """Maps alias neither the input nor each other, and X/Y are cleared."""
        original = minimal_brain_data.data.copy()
        result = minimal_brain_data.ttest(
            permutation=True, n_permute=16, return_null=True, random_state=0
        )
        maps = [result[key] for key in ("mean", "t", "z", "p")]
        arrays = [m.data for m in maps] + [result["null_dist"]]
        for i, first in enumerate(arrays):
            assert not np.shares_memory(first, minimal_brain_data.data)
            for second in arrays[i + 1 :]:
                assert not np.shares_memory(first, second)
        for image in maps:
            assert image.X.is_empty()
            assert image.Y.is_empty()
        for array in arrays:
            array[...] = -999.0
        np.testing.assert_array_equal(minimal_brain_data.data, original)
        for key, image in zip(("mean", "t", "z", "p"), maps):
            assert np.all(np.asarray(image.data) == -999.0)


class TestBrainDataRidgePerTargetAlpha:
    """`ridge_per_target_alpha=True` (the default) selects one alpha per voxel."""

    @staticmethod
    def _fixture(n=80, p=12, n_voxels=6, snr_high=5.0, snr_low=0.2, seed=0):
        """Half the voxels are noisy, so SNR — not chance — drives selection."""
        import nibabel as nib

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p)).astype(np.float32)
        coef = rng.standard_normal((p, n_voxels)).astype(np.float32)
        noise = rng.standard_normal((n, n_voxels)).astype(np.float32)
        scales = np.array(
            [snr_low if j < n_voxels // 2 else snr_high for j in range(n_voxels)],
            dtype=np.float32,
        )
        Y = X @ coef + noise * (1.0 / scales[None, :])

        spatial_shape = (n_voxels, 1, 1)
        affine = np.eye(4)
        volume = np.zeros(spatial_shape + (n,), dtype=np.float32)
        for t in range(n):
            volume[..., t].flat[:n_voxels] = Y[t]
        mask = np.zeros(spatial_shape, dtype=np.float32)
        mask.flat[:n_voxels] = 1.0
        bd = BrainData(
            nib.Nifti1Image(volume, affine), mask=nib.Nifti1Image(mask, affine)
        )
        return bd, X

    def test_selected_alpha_is_one_value_per_voxel(self):
        bd, X = self._fixture()
        alphas = np.logspace(-2, 4, 12)

        bd.fit(model="ridge", X=X, ridge_alpha=alphas, ridge_cv=5)

        assert bd.model_.alpha_.shape == (bd.shape[1],)
        assert np.all(np.isin(bd.model_.alpha_, alphas))

    def test_full_data_weights_match_a_fixed_refit_at_those_alphas(self):
        from nltools.models.ridge import _refit_fixed_hyperparameters

        bd, X = self._fixture()
        bd.fit(model="ridge", X=X, ridge_alpha=np.logspace(-2, 4, 12), ridge_cv=5)

        expected = _refit_fixed_hyperparameters([X], bd.data, bd.model_.alpha_)
        np.testing.assert_allclose(
            bd.ridge_weights.data, expected, rtol=1e-4, atol=1e-4
        )


class TestGlmFacadeContract:
    """The `BrainData` GLM boundary defined in specs/glm.md and specs/braindata.md."""

    @staticmethod
    def _design(brain, seed=0):
        from nltools.data import DesignMatrix

        rng = np.random.default_rng(seed)
        return DesignMatrix(
            {
                "intercept": np.ones(len(brain)),
                "cond_a": rng.normal(size=len(brain)),
                "cond_b": rng.normal(size=len(brain)),
            }
        )

    # ---------------------------------------------------------------- fit

    @pytest.mark.parametrize("removed", ["standardize"])
    def test_fit_rejects_removed_preprocessing_keywords(
        self, minimal_brain_data, removed
    ):
        design = self._design(minimal_brain_data)
        with pytest.raises(TypeError):
            minimal_brain_data.fit(model="glm", X=design, **{removed: None})

    @pytest.mark.parametrize(
        "option",
        [
            {"ridge_cv": 3},
        ],
    )
    def test_fit_glm_rejects_non_default_ridge_options(
        self, minimal_brain_data, option
    ):
        design = self._design(minimal_brain_data)
        with pytest.raises(ValueError, match="unselected estimator|does not accept"):
            minimal_brain_data.fit(model="glm", X=design, **option)

    def test_fit_glm_rejects_unknown_keywords(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        with pytest.raises(TypeError):
            minimal_brain_data.fit(model="glm", X=design, t_r=2.0)

    def test_fit_glm_requires_a_design_matrix(self, minimal_brain_data):
        frame = pd.DataFrame({"intercept": np.ones(len(minimal_brain_data))})
        with pytest.raises(TypeError, match="DesignMatrix"):
            minimal_brain_data.fit(model="glm", X=frame)

    def test_fit_attaches_only_the_documented_state(self, minimal_brain_data):
        from nltools.models import _Glm

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        assert isinstance(minimal_brain_data.model_, _Glm)
        assert minimal_brain_data.glm_betas.shape == (3, minimal_brain_data.shape[1])
        assert minimal_brain_data.glm_residual.shape == minimal_brain_data.shape
        assert minimal_brain_data.glm_predicted.shape == minimal_brain_data.shape
        assert minimal_brain_data.glm_r2.shape[-1] == minimal_brain_data.shape[1]
        for removed in ("glm_t", "glm_p", "glm_se", "X_", "design_matrix"):
            assert not hasattr(minimal_brain_data, removed)

    def test_fit_state_enumeration_is_exhaustive(self, minimal_brain_data):
        from nltools.data.braindata.utils import _FIT_STATE_ATTRIBUTES

        before = set(vars(minimal_brain_data))
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)
        attached = set(vars(minimal_brain_data)) - before

        assert attached
        assert attached <= set(_FIT_STATE_ATTRIBUTES)
        assert "design_matrix" not in _FIT_STATE_ATTRIBUTES

    def test_fit_betas_match_least_squares(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        expected = np.linalg.lstsq(
            design.to_numpy(), minimal_brain_data.data, rcond=None
        )[0]
        np.testing.assert_allclose(
            minimal_brain_data.glm_betas.data, expected, atol=1e-8
        )

    # ---------------------------------------------------- compute_contrasts

    def test_compute_contrasts_before_fit_raises_runtime_error(
        self, minimal_brain_data
    ):
        with pytest.raises(RuntimeError):
            minimal_brain_data.compute_contrasts("cond_a - cond_b")

    def test_compute_contrasts_on_a_fitted_ridge_raises_value_error(
        self, minimal_brain_data
    ):
        X = np.random.default_rng(0).normal(size=(len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)
        with pytest.raises(ValueError, match="Ridge"):
            minimal_brain_data.compute_contrasts([1, -1, 0])

    @pytest.mark.parametrize("contrast", ["cond_a - cond_b"], ids=["string"])
    def test_effect_contrast_equals_beta_arithmetic(self, minimal_brain_data, contrast):
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        effect = minimal_brain_data.compute_contrasts(contrast)

        assert isinstance(effect, BrainData)
        assert effect.X.is_empty() and effect.Y.is_empty()
        np.testing.assert_allclose(
            effect.data,
            np.array([0.0, 1.0, -1.0]) @ minimal_brain_data.glm_betas.data,
            atol=1e-10,
        )

    def test_inference_returns_owned_brain_data_payloads(self, minimal_brain_data):
        from nltools.models import ContrastResult

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        result = minimal_brain_data.compute_contrasts("cond_a - cond_b", inference=True)

        assert isinstance(result, ContrastResult)
        maps = [
            result.effect,
            result.variance,
            result.standard_error,
            result.statistic,
            result.z_score,
            result.p_value,
        ]
        assert all(isinstance(payload, BrainData) for payload in maps)
        assert all(payload.X.is_empty() and payload.Y.is_empty() for payload in maps)
        for left, right in itertools.combinations(maps, 2):
            assert not np.shares_memory(left.data, right.data)
        assert isinstance(result.degrees_of_freedom, float)

        statistic_before = result.statistic.data.copy()
        result.effect.data[0] = 1234.0
        assert minimal_brain_data.glm_betas.data[0, 0] != 1234.0
        np.testing.assert_array_equal(result.statistic.data, statistic_before)

    def test_inference_mapping_returns_keyed_results(self, minimal_brain_data):
        from nltools.models import ContrastResult

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        results = minimal_brain_data.compute_contrasts(
            {"a_vs_b": "cond_a - cond_b"}, inference=True
        )

        assert set(results) == {"a_vs_b"}
        assert isinstance(results["a_vs_b"], ContrastResult)
        assert isinstance(results["a_vs_b"].statistic, BrainData)

    def test_facade_does_not_parse_contrasts(self):
        from nltools.data.braindata import modeling

        assert not hasattr(modeling, "parse_contrast_string")
        assert not hasattr(modeling, "_functional_contrast")

    # ------------------------------------------------------------ predict

    def test_predict_with_a_new_design_delegates_to_glm(self, minimal_brain_data):
        from nltools.data import DesignMatrix

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        rng = np.random.default_rng(3)
        new = DesignMatrix(
            {
                "cond_b": rng.normal(size=6),
                "intercept": np.ones(6),
                "cond_a": rng.normal(size=6),
            }
        )
        predicted = minimal_brain_data.predict(X=new)

        assert isinstance(predicted, BrainData)
        assert predicted.shape == (6, minimal_brain_data.shape[1])
        np.testing.assert_allclose(
            predicted.data,
            new[["intercept", "cond_a", "cond_b"]].to_numpy()
            @ minimal_brain_data.model_.coef_,
        )
        assert predicted.X.is_empty() and predicted.Y.is_empty()

    # ------------------------------------------------------------- report

    # ------------------------------------------------- second-level design


class TestRidgeFacadeContract:
    """`BrainData.fit(model='ridge', ...)` — the prefixed keyword contract."""

    @staticmethod
    def _features(bd, n_features=4, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((len(bd), n_features))

    def test_signature_names_and_defaults(self):
        import inspect

        signature = inspect.signature(BrainData.fit)
        parameters = signature.parameters
        assert parameters["model"].default == "glm"
        expected = {
            "X": None,
            "ridge_alpha": 1.0,
            "ridge_cv": None,
            "ridge_search_iterations": 100,
            "ridge_dirichlet_concentration": (0.1, 1.0),
            "ridge_device": "cpu",
            "ridge_memory_budget_gb": None,
            "ridge_per_target_alpha": True,
            "ridge_prefer_conservative_alpha": False,
            "ridge_progress_bar": False,
            "glm_noise_model": "ols",
            "glm_bins": 100,
            "glm_n_jobs": 1,
            "inplace": True,
            "random_state": None,
        }
        assert [name for name in parameters if name != "self"] == [
            "model",
            *expected,
        ]
        for name, default in expected.items():
            assert parameters[name].default == default, name
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    @pytest.mark.parametrize(
        "name,value",
        [
            ("alpha", 1.0),
        ],
    )
    def test_removed_keyword_raises_type_error(self, minimal_brain_data, name, value):
        X = self._features(minimal_brain_data)
        with pytest.raises(TypeError):
            minimal_brain_data.fit(model="ridge", X=X, **{name: value})

    def test_banded_keywords_reach_the_estimator(self, minimal_brain_data):
        rng = np.random.default_rng(1)
        spaces = {
            "a": rng.standard_normal((len(minimal_brain_data), 3)),
            "b": rng.standard_normal((len(minimal_brain_data), 2)),
        }
        fitted = minimal_brain_data.fit(
            model="ridge",
            X=spaces,
            ridge_alpha=[1.0, 10.0],
            ridge_cv=3,
            ridge_search_iterations=4,
            ridge_dirichlet_concentration=1.0,
            random_state=0,
        )
        assert fitted.model_.search_iterations == 4
        assert fitted.model_.dirichlet_concentration == 1.0
        assert fitted.model_.feature_space_names_ == ("a", "b")

    def test_device_auto_is_rejected(self, minimal_brain_data):
        X = self._features(minimal_brain_data)
        with pytest.raises(ValueError, match="device"):
            minimal_brain_data.fit(model="ridge", X=X, ridge_device="auto")

    def test_fit_attaches_only_the_specified_state(self, minimal_brain_data):
        X = self._features(minimal_brain_data)
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)

        for name in ("model_", "ridge_weights", "ridge_fitted_values", "ridge_r2"):
            assert hasattr(minimal_brain_data, name), name
        for name in ("X_", "cv_results_", "ridge_scores"):
            assert not hasattr(minimal_brain_data, name), name

    def test_fit_state_enumeration_is_exhaustive(self, minimal_brain_data):
        from nltools.data.braindata.utils import _FIT_STATE_ATTRIBUTES

        before = set(vars(minimal_brain_data))
        X = self._features(minimal_brain_data)
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)
        attached = set(vars(minimal_brain_data)) - before

        assert attached == {
            "model_",
            "ridge_weights",
            "ridge_fitted_values",
            "ridge_r2",
        }
        assert attached <= set(_FIT_STATE_ATTRIBUTES)
        for removed in ("X_", "cv_results_", "ridge_scores"):
            assert removed not in _FIT_STATE_ATTRIBUTES

    def test_predict_before_fit_raises_value_error(self, minimal_brain_data):
        with pytest.raises(ValueError):
            minimal_brain_data.predict(X=self._features(minimal_brain_data))
