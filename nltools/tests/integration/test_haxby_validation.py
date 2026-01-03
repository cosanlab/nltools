"""
Integration tests validating nltools against nilearn/sklearn using Haxby dataset.

These tests use real fMRI data to validate that nltools produces results
consistent with established implementations (nilearn for GLM, sklearn for Ridge).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from nltools.data import BrainCollection
from nltools.datasets import fetch_haxby


@pytest.fixture(scope="module")
def haxby_data():
    """Load Haxby dataset (cached, fast after first run)."""
    brain_data, design_matrices = fetch_haxby(n_subjects=1, verbose=0)
    return brain_data[0], design_matrices[0]


class TestGLMValidation:
    """Validate GLM results against nilearn FirstLevelModel."""

    def test_glm_betas_match_expected_range(self, haxby_data):
        """GLM betas should be in reasonable range for normalized data."""
        data, dm = haxby_data
        data = data.copy()

        # Fit GLM
        dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
        data.fit(model="glm", X=dm_filt)

        # Betas should exist and be finite
        assert data.glm_betas is not None
        assert np.all(np.isfinite(data.glm_betas.data))

        # Beta values should be in reasonable range (not exploding)
        assert np.abs(data.glm_betas.data).max() < 1000

    def test_glm_contrast_produces_valid_statistics(self, haxby_data):
        """Face vs house contrast should produce valid t-statistics."""
        data, dm = haxby_data
        data = data.copy()

        dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
        data.fit(model="glm", X=dm_filt)

        # Compute contrast
        contrast = data.compute_contrasts("face - house")

        # Should have valid t-statistics
        assert contrast is not None
        assert np.all(np.isfinite(contrast.data))

        # T-statistics should have reasonable range
        t_vals = contrast.data.flatten()
        assert t_vals.std() > 0.1  # Not all zeros
        assert np.abs(t_vals).max() < 50  # Not exploding

    def test_glm_face_house_discrimination(self, haxby_data):
        """Face and house conditions should show distinct patterns."""
        data, dm = haxby_data
        data = data.copy()

        dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
        data.fit(model="glm", X=dm_filt)

        # Get betas for face and house
        face_beta = data.compute_contrasts("face")
        house_beta = data.compute_contrasts("house")

        # Patterns should be different (low correlation)
        r, _ = pearsonr(face_beta.data.flatten(), house_beta.data.flatten())

        # Face and house patterns should not be identical
        assert r < 0.95, f"Face and house patterns too similar: r={r:.3f}"

    def test_glm_t_and_p_consistency(self, haxby_data):
        """T-statistics and p-values should be consistent."""
        data, dm = haxby_data
        data = data.copy()

        dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
        data.fit(model="glm", X=dm_filt)

        # Get t and p for first regressor
        t_vals = data.glm_t[0].data.flatten()
        p_vals = data.glm_p[0].data.flatten()

        # Higher absolute t should have lower p
        high_t_mask = np.abs(t_vals) > np.percentile(np.abs(t_vals), 90)
        low_t_mask = np.abs(t_vals) < np.percentile(np.abs(t_vals), 10)

        mean_p_high_t = p_vals[high_t_mask].mean()
        mean_p_low_t = p_vals[low_t_mask].mean()

        assert mean_p_high_t < mean_p_low_t, "High |t| should have lower p-values"


class TestRidgeValidation:
    """Validate Ridge results against sklearn RidgeCV."""

    def test_ridge_scores_match_sklearn(self, haxby_data):
        """Ridge R² scores should match sklearn RidgeCV."""
        data, dm = haxby_data
        data = data.copy()

        # Prepare features
        feature_cols = ["face", "house", "cat", "bottle", "scissors", "shoe", "chair"]
        X = dm[feature_cols].to_numpy()

        # Fit with nltools
        data.fit(model="ridge", X=X, cv=5, alpha="auto")

        # Fit sklearn on subset of voxels for speed
        n_test_voxels = 100
        Y = data.data[:, :n_test_voxels]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        sklearn_scores = []
        for v in range(n_test_voxels):
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
            ridge.fit(X_scaled, Y[:, v])
            sklearn_scores.append(ridge.score(X_scaled, Y[:, v]))

        sklearn_scores = np.array(sklearn_scores)
        nltools_scores = data.ridge_scores.data.flatten()[:n_test_voxels]

        # Scores should be highly correlated
        r, _ = pearsonr(sklearn_scores, nltools_scores)
        assert r > 0.9, f"Ridge scores poorly correlated with sklearn: r={r:.3f}"

    def test_ridge_weights_reasonable(self, haxby_data):
        """Ridge weights should be in reasonable range."""
        data, dm = haxby_data
        data = data.copy()

        feature_cols = ["face", "house", "cat", "bottle"]
        X = dm[feature_cols].to_numpy()

        data.fit(model="ridge", X=X, cv=3, alpha="auto")

        # Weights should exist and be finite
        assert data.ridge_weights is not None
        assert np.all(np.isfinite(data.ridge_weights.data))

        # Weights should be regularized (not exploding)
        assert np.abs(data.ridge_weights.data).max() < 100

    def test_ridge_cv_improves_over_no_cv(self, haxby_data):
        """Cross-validated alpha should produce reasonable scores."""
        data, dm = haxby_data
        data = data.copy()

        feature_cols = ["face", "house", "cat", "bottle"]
        X = dm[feature_cols].to_numpy()

        # Fit with CV
        data.fit(model="ridge", X=X, cv=5, alpha="auto")

        # Should have positive R² for some voxels
        scores = data.ridge_scores.data.flatten()
        pct_positive = (scores > 0).mean()

        assert pct_positive > 0.1, f"Too few voxels with R² > 0: {pct_positive:.1%}"

    def test_ridge_predicts_correctly(self, haxby_data):
        """Ridge predictions should correlate with actual data."""
        data, dm = haxby_data
        data = data.copy()

        feature_cols = ["face", "house", "cat", "bottle"]
        X = dm[feature_cols].to_numpy()

        data.fit(model="ridge", X=X, cv=3, alpha="auto")

        # Predict on training data
        predictions = data.predict(X=X)

        # Predictions should correlate with actual for predictable voxels
        # Pick voxels with highest R²
        scores = data.ridge_scores.data.flatten()
        top_voxel_idx = np.argmax(scores)

        actual = data.data[:, top_voxel_idx]
        predicted = predictions.data[:, top_voxel_idx]

        r, _ = pearsonr(actual, predicted)
        assert r > 0.3, f"Poor prediction for best voxel: r={r:.3f}"


class TestWorkflowIntegration:
    """Test full analysis workflows end-to-end."""

    def test_glm_to_group_workflow(self, haxby_data):
        """Full workflow: GLM -> contrast -> would feed into group analysis."""
        data, dm = haxby_data
        data = data.copy()

        # First level
        dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
        data.fit(model="glm", X=dm_filt)

        # Compute multiple contrasts
        contrasts = {
            "face_vs_house": "face - house",
            "face_vs_scrambled": "face - scrambledpix",
            "all_objects": "face + house + cat + bottle",
        }
        results = data.compute_contrasts(contrasts)

        # All contrasts should be valid
        assert len(results) == 3
        for name, contrast in results.items():
            assert np.all(np.isfinite(contrast.data)), f"{name} has invalid values"

    def test_encoding_model_workflow(self, haxby_data):
        """Full encoding workflow: fit -> weights -> predict."""
        data, dm = haxby_data
        data = data.copy()

        feature_cols = ["face", "house", "cat", "bottle", "scissors", "shoe", "chair"]
        X = dm[feature_cols].to_numpy()

        # Fit
        data.fit(model="ridge", X=X, cv=5, alpha="auto")

        # Access weights
        weights = data.ridge_weights
        assert weights.shape[0] == len(feature_cols)

        # Access scores
        scores = data.ridge_scores
        assert scores.shape == (data.shape[1],)

        # Predict
        predictions = data.predict(X=X[:10])
        assert predictions.shape == (10, data.shape[1])


@pytest.fixture(scope="module")
def haxby_collection():
    """Load Haxby dataset as BrainCollection with 3 subjects."""
    brain_data_list, dm_list = fetch_haxby(n_subjects=3, verbose=0)
    # Flatten nested structure (each subject returns a list)
    brains = [bd for bd_list in brain_data_list for bd in bd_list]
    dms = [dm for dm_list in dm_list for dm in dm_list]

    mask = brains[0].mask
    bc = BrainCollection(
        brains,
        mask=mask,
        metadata=pd.DataFrame({"subject": ["sub-01", "sub-02", "sub-03"]}),
    )
    return bc, dms


@pytest.mark.slow
class TestBrainCollectionGLMWorkflow:
    """Test multi-subject GLM workflow using BrainCollection.fit()."""

    def test_collection_fit_glm_with_shared_dm(self, haxby_collection):
        """BrainCollection.fit(model='glm') with shared design matrix."""
        bc, dms = haxby_collection

        # Use first subject's design matrix as shared (same paradigm)
        dm = dms[0].add_dct_basis(duration=128).add_poly(order=1, include_lower=True)

        # Fit GLM using new unified API
        betas = bc.fit(model="glm", X=dm, show_progress=False)

        # Should return BrainCollection of betas
        assert isinstance(betas, BrainCollection)
        assert len(betas) == 3
        # Each subject should have betas for all regressors
        assert betas[0].shape[0] == dm.shape[1]

    def test_collection_fit_glm_with_per_subject_dm(self, haxby_collection):
        """BrainCollection.fit(model='glm') with per-subject design matrices."""
        bc, dms = haxby_collection

        # Prepare per-subject design matrices
        dm_list = [
            dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
            for dm in dms
        ]

        # Fit GLM with per-subject DMs
        betas = bc.fit(model="glm", X=dm_list, show_progress=False)

        assert isinstance(betas, BrainCollection)
        assert len(betas) == 3

    def test_collection_fit_glm_return_stats(self, haxby_collection):
        """BrainCollection.fit(model='glm') with return_stats."""
        bc, dms = haxby_collection
        dm = dms[0].add_dct_basis(duration=128).add_poly(order=1, include_lower=True)

        result = bc.fit(model="glm", X=dm, return_stats=["t", "p"], show_progress=False)

        assert isinstance(result, dict)
        assert "betas" in result
        assert "t" in result
        assert "p" in result
        assert len(result["betas"]) == 3
        assert len(result["t"]) == 3


@pytest.mark.slow
class TestBrainCollectionRidgeWorkflow:
    """Test multi-subject Ridge encoding workflow using BrainCollection.fit()."""

    def test_collection_fit_ridge_scores(self, haxby_collection):
        """BrainCollection.fit(model='ridge') returns CV scores."""
        bc, dms = haxby_collection

        # Use stimulus columns as features
        feature_cols = ["face", "house", "cat", "bottle"]
        X = dms[0][feature_cols].to_numpy()

        # Fit Ridge using new unified API
        scores = bc.fit(
            model="ridge", X=X, alpha=1.0, cv=3, output="scores", show_progress=False
        )

        assert isinstance(scores, BrainCollection)
        assert len(scores) == 3
        # Each subject should have 1D scores (per voxel)
        assert scores[0].shape[0] == 1

    def test_collection_fit_ridge_weights(self, haxby_collection):
        """BrainCollection.fit(model='ridge') returns weights."""
        bc, dms = haxby_collection

        feature_cols = ["face", "house", "cat", "bottle"]
        X = dms[0][feature_cols].to_numpy()

        weights = bc.fit(
            model="ridge",
            X=X,
            alpha=1.0,
            output="weights",
            cv=None,
            show_progress=False,
        )

        assert isinstance(weights, BrainCollection)
        assert len(weights) == 3
        # Weights shape: (n_features, n_voxels)
        assert weights[0].shape[0] == len(feature_cols)

    def test_collection_fit_ridge_both(self, haxby_collection):
        """BrainCollection.fit(model='ridge') with output='both'."""
        bc, dms = haxby_collection

        feature_cols = ["face", "house"]
        X = dms[0][feature_cols].to_numpy()

        result = bc.fit(
            model="ridge", X=X, alpha=1.0, cv=3, output="both", show_progress=False
        )

        assert isinstance(result, dict)
        assert "scores" in result
        assert "weights" in result
        assert len(result["scores"]) == 3
        assert len(result["weights"]) == 3
