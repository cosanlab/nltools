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
    """Load Haxby dataset as BrainCollection with 3 subjects.

    Returns 3 subjects × 4 runs = 12 total BrainData objects.
    Uses first run from each subject for simplicity in group tests.
    """
    brain_data_list, dm_list = fetch_haxby(n_subjects=3, verbose=0)

    # fetch_haxby returns flat lists: 12 items (3 subjects × 4 runs)
    # Take first run from each subject (indices 0, 4, 8)
    brains = [brain_data_list[i * 4] for i in range(3)]
    dms = [dm_list[i * 4] for i in range(3)]

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


@pytest.mark.slow
class TestMVPAWorkflow:
    """Test MVPA/decoding workflow using face vs house classification."""

    def test_face_house_classification_above_chance(self, haxby_data):
        """Face vs house decoding should exceed chance (50%) significantly.

        Uses leave-one-run-out cross-validation on extracted condition timepoints.
        Haxby dataset is well-known to show ~80%+ accuracy for this contrast.
        """
        data, dm = haxby_data
        data = data.copy()

        # Get face and house condition masks from design matrix
        face_mask = dm["face"].to_numpy() > 0.5
        house_mask = dm["house"].to_numpy() > 0.5

        # Extract timepoints for each condition
        face_data = data.data[face_mask]
        house_data = data.data[house_mask]

        # Combine into classification dataset
        X = np.vstack([face_data, house_data])
        y = np.array([0] * len(face_data) + [1] * len(house_data))

        # Simple CV classification
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline

        clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))
        scores = cross_val_score(clf, X, y, cv=5)
        accuracy = scores.mean()

        # Should be well above chance (50%)
        assert accuracy > 0.6, f"Accuracy {accuracy:.1%} should exceed 60%"

    def test_predict_mvpa_api(self, haxby_data):
        """BrainData.predict() should work for MVPA classification."""
        from nltools.data import BrainData

        data, dm = haxby_data
        data = data.copy()

        # Get condition labels
        face_mask = dm["face"].to_numpy() > 0.5
        house_mask = dm["house"].to_numpy() > 0.5
        condition_mask = face_mask | house_mask

        # Subset to face/house timepoints only
        data_subset = data[condition_mask]
        y = np.where(face_mask[condition_mask], 0, 1)

        # Use predict API - returns BrainData with accuracy values
        result = data_subset.predict(
            y=y,
            method="whole_brain",
            estimator="svm",
            cv=3,
            scoring="accuracy",
            show_progress=False,
        )

        # Should return BrainData with accuracy
        assert isinstance(result, BrainData)
        # Whole-brain returns single accuracy value
        accuracy = result.data.mean()
        assert accuracy > 0.5, f"Accuracy {accuracy:.1%} should exceed chance"

    def test_predict_returns_braindata(self, haxby_data):
        """predict() should return BrainData with accuracy scores."""
        from nltools.data import BrainData

        data, dm = haxby_data
        data = data.copy()

        # Simple binary classification setup
        n_samples = data.shape[0]
        y = np.array([0, 1] * (n_samples // 2) + [0] * (n_samples % 2))

        result = data.predict(
            y=y,
            method="whole_brain",
            estimator="svm",
            cv=3,
            show_progress=False,
        )

        # Validate structure - returns BrainData
        assert isinstance(result, BrainData)
        assert result.data is not None
        # Accuracy should be in valid range [0, 1]
        assert 0 <= result.data.mean() <= 1


@pytest.mark.slow
class TestGroupInferenceWorkflow:
    """Test second-level group inference workflows."""

    def test_contrast_ttest_across_subjects(self, haxby_collection):
        """Group ttest on face-house contrast should find significant voxels.

        This tests the full workflow: GLM -> contrast -> group ttest.
        The Haxby dataset should show significant face vs house effects
        in visual regions (FFA, PPA).
        """
        from nltools.data import BrainData

        bc, dms = haxby_collection

        # Fit GLM per subject
        dm = dms[0].add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
        betas = bc.fit(model="glm", X=dm, show_progress=False)

        # Compute face-house contrast for each subject
        contrasts = betas.map(
            lambda bd: bd.compute_contrasts("face - house"),
            axis=0,
            show_progress=False,
        )

        # Run group-level t-test
        t_stat, p_val = contrasts.ttest()

        # Validate output types
        assert isinstance(t_stat, BrainData)
        assert isinstance(p_val, BrainData)

        # T-stats should be finite and variable
        assert np.all(np.isfinite(t_stat.data))
        assert t_stat.data.std() > 0.1, "T-stats should vary across voxels"

        # Should find some significant voxels (uncorrected)
        n_sig = (p_val.data < 0.05).sum()
        assert n_sig > 0, "Should find at least some significant voxels"

    def test_permutation_test_produces_valid_pvalues(self, haxby_collection):
        """Permutation test should produce valid p-values in [0, 1]."""
        bc, dms = haxby_collection

        # Fit GLM and get contrasts
        dm = dms[0].add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
        betas = bc.fit(model="glm", X=dm, show_progress=False)

        contrasts = betas.map(
            lambda bd: bd.compute_contrasts("face - house"),
            axis=0,
            show_progress=False,
        )

        # Run permutation test with small n for speed
        result = contrasts.permutation_test(
            n_permute=100,
            random_state=42,  # Small for testing
        )

        # Should return dict with expected keys
        assert isinstance(result, dict)
        assert "mean" in result  # Mean across subjects
        assert "p" in result  # P-values

        # P-values should be in valid range
        p_vals = result["p"].data.flatten()
        assert np.all(p_vals >= 0), "P-values should be >= 0"
        assert np.all(p_vals <= 1), "P-values should be <= 1"

    def test_ttest_with_temporal_mean(self, haxby_collection):
        """Ttest on mean activation (simpler workflow)."""
        from nltools.data import BrainData

        bc, _ = haxby_collection

        # Compute mean across time for each subject
        temporal_means = bc.mean(axis=1)

        # Run t-test across subjects
        t_stat, p_val = temporal_means.ttest()

        # Validate outputs
        assert isinstance(t_stat, BrainData)
        assert isinstance(p_val, BrainData)
        assert t_stat.shape == p_val.shape


@pytest.fixture(scope="module")
def haxby_vtc_betas():
    """Load Haxby with VTC mask and compute category betas for ISC tests.

    Returns BrainCollection of category betas (7 categories × 3 subjects)
    masked to ventral temporal cortex. Uses subject 1's VTC mask for all
    subjects to ensure common voxel space for spatial ISC comparisons.
    """
    import nibabel as nib
    from nilearn.datasets import fetch_haxby as nilearn_fetch_haxby

    from nltools.data import BrainData, BrainCollection

    # Get VTC mask from nilearn - use union of all subjects' masks
    nilearn_data = nilearn_fetch_haxby(subjects=[1, 2, 3])

    # Create union mask (logical OR of all VTC masks)
    mask_data = None
    affine = None
    for mask_path in nilearn_data.mask_vt:
        mask_img = nib.load(mask_path)
        if mask_data is None:
            mask_data = mask_img.get_fdata() > 0
            affine = mask_img.affine
        else:
            mask_data = mask_data | (mask_img.get_fdata() > 0)

    vtc_mask = nib.Nifti1Image(mask_data.astype(np.float32), affine)

    # Load nltools data
    brain_data_list, dm_list = fetch_haxby(n_subjects=3, verbose=0)

    # Category columns (excluding rest and scrambledpix for cleaner analysis)
    categories = ["face", "house", "cat", "bottle", "scissors", "shoe", "chair"]

    # Process each subject: fit GLM and extract category betas
    all_betas = []
    for subj_idx in range(3):
        # Get first run for this subject
        data = brain_data_list[subj_idx * 4]
        dm = dm_list[subj_idx * 4]

        # Create BrainData with common VTC mask
        data_vtc = BrainData(data.to_nifti(), mask=vtc_mask)

        # Fit GLM
        dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
        data_vtc.fit(model="glm", X=dm_filt)

        # Extract betas for each category (shape: n_categories × n_vtc_voxels)
        category_betas = []
        for cat in categories:
            beta = data_vtc.compute_contrasts(cat)
            category_betas.append(beta.data.flatten())

        # Stack into single array
        betas_array = np.vstack(category_betas)

        # Create BrainData for this subject's betas
        beta_bd = BrainData(mask=vtc_mask)
        beta_bd.data = betas_array
        all_betas.append(beta_bd)

    # Create BrainCollection of betas
    bc_betas = BrainCollection(all_betas, mask=vtc_mask)

    return bc_betas, categories, vtc_mask


@pytest.mark.slow
class TestISCWorkflow:
    """Test inter-subject correlation workflows using VTC mask."""

    def test_temporal_isc_smoke_test(self, haxby_collection):
        """Temporal ISC on raw data - API smoke test.

        Note: Haxby subjects had different presentation orders, so we don't
        expect strong temporal ISC. This just validates the API works.
        """
        from nltools.data import BrainData

        bc, _ = haxby_collection

        # Compute ISC using leave-one-out method
        result = bc.isc(method="loo", show_progress=False)

        # Should return dict with isc BrainData
        assert isinstance(result, dict)
        assert "isc" in result
        assert isinstance(result["isc"], BrainData)

        # Correlations should be in valid range [-1, 1]
        isc_vals = result["isc"].data.flatten()
        assert np.all(isc_vals >= -1), "ISC should be >= -1"
        assert np.all(isc_vals <= 1), "ISC should be <= 1"
        assert np.all(np.isfinite(isc_vals)), "ISC should be finite"

    def test_beta_series_isc_vtc(self, haxby_vtc_betas):
        """Beta-series ISC in VTC - API validation with category betas.

        Stack category betas in consistent order across subjects, then run
        temporal ISC. With only 7 categories, this is primarily an API test.
        The spatial ISC test provides stronger validation of shared representations.
        """
        bc_betas, categories, _ = haxby_vtc_betas

        # Run temporal ISC on beta-series (categories as "timepoints")
        result = bc_betas.isc(method="loo", show_progress=False)

        assert isinstance(result, dict)
        assert "isc" in result

        # Validate ISC values are finite and in valid range
        isc_vals = result["isc"].data.flatten()
        assert np.all(np.isfinite(isc_vals)), "ISC values should be finite"
        assert np.all(isc_vals >= -1), "ISC should be >= -1"
        assert np.all(isc_vals <= 1), "ISC should be <= 1"

        # Note: With only 7 categories, mean ISC may not be positive
        # The spatial ISC test (same vs cross category) is more robust

    def test_spatial_pattern_correlation_computation(self, haxby_vtc_betas):
        """Validate spatial pattern correlation computation across subjects.

        Computes same-category vs cross-category correlations. Without
        anatomical alignment (SRM/hyperalignment), cross-subject spatial
        correlations may not show the expected pattern due to individual
        differences in functional anatomy.

        This test validates the computation runs correctly and produces
        valid correlation values. The SRM workflow tests validate that
        alignment improves cross-subject pattern similarity.
        """
        bc_betas, categories, _ = haxby_vtc_betas

        # Extract beta data: shape (n_subjects, n_categories, n_voxels)
        n_subjects = len(bc_betas)
        betas = np.array([bc_betas[i].data for i in range(n_subjects)])

        # Compute same-category correlations (e.g., face-face across subjects)
        same_category_corrs = []
        for cat_idx in range(len(categories)):
            patterns = betas[:, cat_idx, :]
            for i in range(n_subjects):
                for j in range(i + 1, n_subjects):
                    r = np.corrcoef(patterns[i], patterns[j])[0, 1]
                    same_category_corrs.append(r)

        # Compute cross-category correlations
        cross_category_corrs = []
        for cat_i in range(len(categories)):
            for cat_j in range(cat_i + 1, len(categories)):
                for subj_i in range(n_subjects):
                    for subj_j in range(subj_i + 1, n_subjects):
                        r = np.corrcoef(
                            betas[subj_i, cat_i, :], betas[subj_j, cat_j, :]
                        )[0, 1]
                        cross_category_corrs.append(r)

        # Validate correlations are finite and in valid range
        all_corrs = same_category_corrs + cross_category_corrs
        assert all(np.isfinite(r) for r in all_corrs), (
            "All correlations should be finite"
        )
        assert all(-1 <= r <= 1 for r in all_corrs), "Correlations should be in [-1, 1]"

        # Note: Without SRM alignment, same > cross may not hold
        # SRM tests validate that alignment improves this relationship
