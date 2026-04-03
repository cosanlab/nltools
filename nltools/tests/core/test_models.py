"""
Test model classes for neuroimaging analysis.

Part of functional core - tests sklearn-compatible model APIs.
Following model-spec.md Sprint 2 implementation.
"""

import numpy as np
import pytest
from nltools.models import BaseModel, Ridge

pytestmark = pytest.mark.slow


# ============================================================================
# Helper Functions
# ============================================================================


def _torch_available():
    """Check if PyTorch is installed"""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


# ============================================================================
# Module-Scoped Fixtures - Reduce Redundant Computation
# ============================================================================


@pytest.fixture(scope="module")
def ridge_single_target_data():
    """Standard single-target Ridge test data.

    Module-scoped: deterministic data shared across tests.
    """
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    X_test = np.random.randn(20, 50).astype(np.float32)
    return {"X": X, "y": y, "X_test": X_test}


@pytest.fixture(scope="module")
def ridge_multi_target_data():
    """Standard multi-target Ridge test data."""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)
    X_test = np.random.randn(20, 50).astype(np.float32)
    return {"X": X, "Y": Y, "X_test": X_test}


@pytest.fixture(scope="module")
def fitted_ridge_single(ridge_single_target_data):
    """Pre-fitted Ridge model for property tests.

    Module-scoped: expensive fit() runs once.
    """
    model = Ridge(alpha=1.0)
    model.fit(ridge_single_target_data["X"], ridge_single_target_data["y"])
    return model, ridge_single_target_data


@pytest.fixture(scope="module")
def fitted_ridge_cv(ridge_single_target_data):
    """Pre-fitted Ridge with CV for property tests."""
    model = Ridge(alpha="auto", cv=3)
    model.fit(ridge_single_target_data["X"], ridge_single_target_data["y"])
    return model, ridge_single_target_data


# ============================================================================
# BaseModel Abstract Interface
# ============================================================================


def test_basemodel_is_abstract():
    """BaseModel cannot be instantiated directly"""
    with pytest.raises(TypeError, match="abstract"):
        BaseModel()


def test_basemodel_defines_fit():
    """BaseModel requires fit() implementation"""

    # Create minimal concrete subclass missing fit()
    class Incomplete(BaseModel):
        def predict(self, X):
            pass

        def score(self, X, y):
            pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_basemodel_defines_predict():
    """BaseModel requires predict() implementation"""

    class Incomplete(BaseModel):
        def fit(self, X, y):
            pass

        def score(self, X, y):
            pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_basemodel_defines_score():
    """BaseModel requires score() implementation"""

    class Incomplete(BaseModel):
        def fit(self, X, y):
            pass

        def predict(self, X):
            pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_basemodel_concrete_subclass():
    """Concrete subclass with all methods should instantiate"""

    class Concrete(BaseModel):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

        def score(self, X, y):
            return 0.0

    model = Concrete()
    assert isinstance(model, BaseModel)


# ============================================================================
# BaseModel Shared Functionality
# ============================================================================


def test_basemodel_fit_returns_self():
    """fit() should return self for method chaining"""

    class Concrete(BaseModel):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

        def score(self, X, y):
            return 0.0

    model = Concrete()
    X = np.random.randn(100, 50)
    y = np.random.randn(100)

    result = model.fit(X, y)
    assert result is model


def test_basemodel_tracks_fitted_state():
    """BaseModel should track whether fit() has been called"""

    class Concrete(BaseModel):
        def fit(self, X, y):
            super().fit(X, y)  # Calls BaseModel.fit() to set state
            return self

        def predict(self, X):
            self._check_is_fitted()
            return np.zeros(X.shape[0])

        def score(self, X, y):
            return 0.0

    model = Concrete()
    X = np.random.randn(100, 50)
    y = np.random.randn(100)

    # Before fit
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)

    # After fit
    model.fit(X, y)
    result = model.predict(X)  # Should not raise
    assert result.shape == (100,)


def test_basemodel_stores_training_shape():
    """BaseModel should store X and y shapes from training"""

    class Concrete(BaseModel):
        def fit(self, X, y):
            super().fit(X, y)
            return self

        def predict(self, X):
            self._check_is_fitted()
            return np.zeros(X.shape[0])

        def score(self, X, y):
            return 0.0

    model = Concrete()
    X = np.random.randn(100, 50)
    y = np.random.randn(100)

    model.fit(X, y)

    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == 50
    assert hasattr(model, "n_samples_")
    assert model.n_samples_ == 100


# ============================================================================
# BaseModel Input Validation
# ============================================================================


def test_basemodel_validates_X_shape():
    """BaseModel should validate X is 2D array"""

    class Concrete(BaseModel):
        def fit(self, X, y):
            X = self._validate_X(X)
            super().fit(X, y)
            return self

        def predict(self, X):
            X = self._validate_X(X)
            return np.zeros(X.shape[0])

        def score(self, X, y):
            return 0.0

    model = Concrete()

    # 1D array should fail
    X_1d = np.random.randn(100)
    y = np.random.randn(100)
    with pytest.raises(ValueError, match="2D array"):
        model.fit(X_1d, y)

    # 3D array should fail
    X_3d = np.random.randn(10, 20, 30)
    with pytest.raises(ValueError, match="2D array"):
        model.fit(X_3d, y)

    # 2D array should work
    X_2d = np.random.randn(100, 50)
    model.fit(X_2d, y)  # Should not raise


def test_basemodel_validates_y_shape():
    """BaseModel should validate y shape matches X"""

    class Concrete(BaseModel):
        def fit(self, X, y):
            X, y = self._validate_X_y(X, y)
            super().fit(X, y)
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

        def score(self, X, y):
            return 0.0

    model = Concrete()
    X = np.random.randn(100, 50)

    # Mismatched samples
    y_wrong = np.random.randn(90)
    with pytest.raises(ValueError, match="samples"):
        model.fit(X, y_wrong)

    # Correct 1D y
    y_1d = np.random.randn(100)
    model.fit(X, y_1d)  # Should not raise

    # Correct 2D y (multi-target)
    y_2d = np.random.randn(100, 5)
    model.fit(X, y_2d)  # Should not raise


def test_basemodel_validates_predict_features():
    """predict() should validate feature count matches training"""

    class Concrete(BaseModel):
        def fit(self, X, y):
            X, y = self._validate_X_y(X, y)
            super().fit(X, y)
            return self

        def predict(self, X):
            self._check_is_fitted()
            X = self._validate_X(X, reset=False)  # Check features match
            return np.zeros(X.shape[0])

        def score(self, X, y):
            return 0.0

    model = Concrete()
    X_train = np.random.randn(100, 50)
    y_train = np.random.randn(100)
    model.fit(X_train, y_train)

    # Correct features
    X_test = np.random.randn(20, 50)
    model.predict(X_test)  # Should not raise

    # Wrong features
    X_wrong = np.random.randn(20, 40)
    with pytest.raises(ValueError, match="features"):
        model.predict(X_wrong)


# ============================================================================
# Ridge Model - Basic Fit/Predict
# ============================================================================


def test_ridge_instantiation():
    """Ridge should instantiate with alpha parameter"""
    model = Ridge(alpha=1.0)
    assert model.alpha == 1.0
    assert not model.is_fitted_


def test_ridge_default_alpha():
    """Ridge should use default alpha if not specified"""
    model = Ridge()
    assert model.alpha == 1.0  # Default value


def test_ridge_auto_detects_single_space():
    """Ridge should auto-detect single feature space from array input."""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha="auto", cv=3)
    model.fit(X, y)

    # Should have selected an alpha (single space uses solve_ridge_cv)
    assert hasattr(model, "alpha_")
    assert isinstance(model.alpha_, float)
    assert model.deltas_ is None  # No feature space weights for single space


def test_ridge_auto_detects_multiple_spaces():
    """Ridge should auto-detect multiple feature spaces from list input."""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha="auto", cv=3, n_iter=5, random_state=42)
    model.fit([X1, X2], y)

    # Should have feature space weights (multiple spaces uses solve_banded_ridge_cv)
    assert hasattr(model, "deltas_")
    assert model.deltas_ is not None
    assert model.deltas_.shape == (2, 1)  # 2 spaces, 1 target
    # alpha_ should be None (alphas are embedded in deltas)
    assert model.alpha_ is None
    # Verify progress_bar parameter exists and defaults to False
    assert hasattr(model, "progress_bar")
    assert model.progress_bar is False


def test_ridge_multiple_spaces_requires_cv():
    """Multiple feature spaces require alpha='auto' with CV."""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha=1.0)  # Fixed alpha
    with pytest.raises(ValueError, match="Banded ridge requires"):
        model.fit([X1, X2], y)

    # Test that progress_bar=True works with banded ridge (where progress bar is shown)
    model_with_pb = Ridge(alpha="auto", cv=3, n_iter=3, progress_bar=True)
    assert model_with_pb.progress_bar is True
    model_with_pb.fit([X1, X2], y)
    assert model_with_pb.is_fitted_


def test_ridge_single_target_properties(fitted_ridge_single):
    """Test all single-target Ridge fit/predict properties.

    Consolidates: fit_single_target, predict_single_target tests.
    Single fit(), multiple assertions.
    """
    model, data = fitted_ridge_single

    # 1. Model should be fitted
    assert model.is_fitted_, "Model should be fitted"

    # 2. Should store coefficients with correct shape
    assert hasattr(model, "coef_"), "Missing coef_ attribute"
    assert model.coef_.shape == (50,), f"coef_ shape {model.coef_.shape} != (50,)"

    # 3. progress_bar parameter should exist
    assert hasattr(model, "progress_bar")
    assert model.progress_bar is False

    # 4. Predictions should work and be valid
    y_pred = model.predict(data["X_test"])
    assert y_pred.shape == (20,), f"Prediction shape {y_pred.shape} != (20,)"
    assert not np.isnan(y_pred).any(), "Predictions contain NaN"
    assert not np.allclose(y_pred, 0), "Predictions are all zeros"


def test_ridge_multi_target_properties(ridge_multi_target_data):
    """Test all multi-target Ridge fit/predict properties.

    Single fit(), multiple assertions.
    """
    model = Ridge(alpha=1.0)
    model.fit(ridge_multi_target_data["X"], ridge_multi_target_data["Y"])

    # 1. Coefficients should be 2D with correct shape
    assert model.coef_.shape == (50, 5), f"coef_ shape {model.coef_.shape} != (50, 5)"

    # 2. Predictions should have correct shape
    Y_pred = model.predict(ridge_multi_target_data["X_test"])
    assert Y_pred.shape == (20, 5), f"Prediction shape {Y_pred.shape} != (20, 5)"


def test_ridge_predict_without_fit():
    """Ridge should raise error if predict called before fit"""
    model = Ridge(alpha=1.0)
    X_test = np.random.randn(20, 50)

    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X_test)


def test_ridge_vs_sklearn():
    """Ridge should match sklearn Ridge results"""
    from sklearn.linear_model import Ridge as SklearnRidge

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # Our implementation
    model_ours = Ridge(alpha=alpha)
    model_ours.fit(X, y)
    pred_ours = model_ours.predict(X)

    # sklearn
    model_sklearn = SklearnRidge(alpha=alpha, fit_intercept=False, solver="svd")
    model_sklearn.fit(X, y)
    pred_sklearn = model_sklearn.predict(X)

    # Should match
    np.testing.assert_allclose(pred_ours, pred_sklearn, rtol=1e-4)
    np.testing.assert_allclose(model_ours.coef_, model_sklearn.coef_, rtol=1e-4)


# ============================================================================
# Ridge Model - Cross-Validation
# ============================================================================


def test_ridge_cv_instantiation():
    """Ridge with cv should instantiate properly"""
    model = Ridge(alpha="auto", cv=5)
    assert model.alpha == "auto"
    assert model.cv == 5


def test_ridge_cv_properties(fitted_ridge_cv):
    """Test all Ridge CV fit properties.

    Consolidates: cv_fits_and_selects_alpha, cv_reproducibility tests.
    Uses pre-fitted fixture.
    """
    model, data = fitted_ridge_cv

    # 1. Should have selected an alpha
    assert hasattr(model, "alpha_"), "Missing alpha_ attribute"
    assert isinstance(model.alpha_, float), "alpha_ should be float"
    assert model.alpha_ > 0, "alpha_ should be positive"

    # 2. Should have CV scores
    assert hasattr(model, "cv_scores_"), "Missing cv_scores_ attribute"
    assert model.cv_scores_.shape[0] == 3, (
        f"cv_scores_ n_folds {model.cv_scores_.shape[0]} != 3"
    )

    # 3. Reproducibility: same data should give same results
    model2 = Ridge(alpha="auto", cv=3)
    model2.fit(data["X"], data["y"])
    assert model.alpha_ == model2.alpha_, "CV should be reproducible"
    np.testing.assert_allclose(model.coef_, model2.coef_, rtol=1e-5)


def test_ridge_cv_alphas_parameter(ridge_single_target_data):
    """Ridge should accept custom alpha range for CV"""
    alphas = [0.1, 1.0, 10.0]
    model = Ridge(alpha="auto", cv=3, alphas=alphas)
    model.fit(ridge_single_target_data["X"], ridge_single_target_data["y"])

    # Selected alpha should be from our list
    assert model.alpha_ in alphas


def test_ridge_cv_multi_target(ridge_multi_target_data):
    """Ridge CV should work with multiple targets"""
    model = Ridge(alpha="auto", cv=3)
    model.fit(ridge_multi_target_data["X"], ridge_multi_target_data["Y"])

    # Should fit all targets
    assert model.coef_.shape == (50, 5)

    # CV scores should include all targets
    assert model.cv_scores_.shape[2] == 5  # n_targets


# ============================================================================
# Ridge Model - Backend Integration
# ============================================================================


def test_ridge_numpy_backend():
    """Ridge should work with NumPy backend"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha=1.0, backend="numpy")
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == (100,)
    assert model.backend_.name == "numpy"


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_torch_backend():
    """Ridge should work with PyTorch backend"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha=1.0, backend="torch")
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == (100,)
    assert model.backend_.name.startswith("torch")


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_cpu_gpu_equivalence():
    """Ridge should give same results on CPU and GPU"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # CPU
    model_cpu = Ridge(alpha=1.0, backend="numpy")
    model_cpu.fit(X, y)
    pred_cpu = model_cpu.predict(X)

    # GPU
    model_gpu = Ridge(alpha=1.0, backend="torch")
    model_gpu.fit(X, y)
    pred_gpu = model_gpu.predict(X)

    # Allow small numerical differences between backends
    np.testing.assert_allclose(pred_gpu, pred_cpu, rtol=1e-3, atol=1e-6)


def test_ridge_auto_backend():
    """Ridge should select backend automatically"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha=1.0, backend="auto")
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == (100,)
    assert hasattr(model, "backend_")
    assert model.backend_.name in ["numpy", "torch-cpu", "torch-cuda", "torch-mps"]


# ============================================================================
# GLMModel - Basic Interface
# ============================================================================


def test_glm_instantiation():
    """Glm should instantiate with optional parameters"""
    from nltools.models import Glm

    # Default instantiation
    model = Glm()
    assert not model.is_fitted_

    # With parameters
    model = Glm(t_r=2.0, noise_model="ar1", smoothing_fwhm=5.0)
    assert model.t_r == 2.0
    assert model.noise_model == "ar1"
    assert model.smoothing_fwhm == 5.0


def test_glm_requires_nilearn():
    """Glm should fail gracefully if nilearn is not available"""
    # Note: In practice, nilearn is always available in our tests
    # This test documents the expected behavior
    from nltools.models import Glm

    model = Glm()
    assert hasattr(model, "fit")


def test_glm_fit_with_design_matrix():
    """Glm should fit with images and design matrices"""
    from nltools.models import Glm
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    # Create synthetic fMRI data (small)
    np.random.seed(42)
    n_scans = 20
    img_shape = (10, 10, 10)  # Small for speed
    fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
    affine = np.eye(4)
    img = Nifti1Image(fmri_data.T, affine)  # Note: nibabel uses (x,y,z,t) order

    # Create a simple mask (all ones)
    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    # Create design matrix
    frame_times = np.arange(n_scans) * 2.0  # TR = 2s
    events = pd.DataFrame(
        {
            "onset": [0, 10, 20, 30],
            "duration": [1, 1, 1, 1],
            "trial_type": ["task", "task", "rest", "rest"],
        }
    )
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm"
    )

    # Fit GLM with explicit mask
    model = Glm(t_r=2.0, mask=mask_img)
    result = model.fit(img, design_matrices=design_matrix)

    # Should return self
    assert result is model
    assert model.is_fitted_


def test_glm_fit_tracks_state():
    """Glm should track fitted state like BaseModel"""
    from nltools.models import Glm

    model = Glm()
    assert not model.is_fitted_

    # Fitting should be tested with actual data in other tests
    # This just checks the attribute exists


def test_glm_suppresses_drift_model_warning():
    """Glm should suppress drift_model warning when design matrices are supplied"""
    import warnings
    from nltools.models import Glm
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    # Create synthetic fMRI data (small)
    np.random.seed(42)
    n_scans = 20
    img_shape = (10, 10, 10)  # Small for speed
    fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
    affine = np.eye(4)
    img = Nifti1Image(fmri_data.T, affine)  # Note: nibabel uses (x,y,z,t) order

    # Create a simple mask (all ones)
    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    # Create design matrix
    frame_times = np.arange(n_scans) * 2.0  # TR = 2s
    events = pd.DataFrame(
        {
            "onset": [0, 10, 20, 30],
            "duration": [1, 1, 1, 1],
            "trial_type": ["task", "task", "rest", "rest"],
        }
    )
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm"
    )

    # Create Glm with drift_model set (this would trigger warning without suppression)
    model = Glm(t_r=2.0, mask=mask_img, drift_model="cosine")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Capture all warnings
        model.fit(img, design_matrices=design_matrix)

        # Check that drift_model warning is NOT present
        drift_warnings = [
            warn
            for warn in w
            if "drift_model" in str(warn.message).lower()
            and "will be ignored" in str(warn.message).lower()
        ]
        assert len(drift_warnings) == 0, (
            f"Expected no drift_model warnings, but got {len(drift_warnings)}: "
            f"{[str(w.message) for w in drift_warnings]}"
        )

    # Verify model was fitted successfully
    assert model.is_fitted_
    # Verify progress_bar parameter exists and defaults to False
    assert hasattr(model, "progress_bar")
    assert model.progress_bar is False

    # Test with progress_bar=True - should work without errors
    model_with_pb = Glm(t_r=2.0, mask=mask_img, progress_bar=True)
    assert model_with_pb.progress_bar is True
    model_with_pb.fit(img, design_matrices=design_matrix)
    assert model_with_pb.is_fitted_


def test_glm_compute_contrast_after_fit():
    """Glm should compute contrasts after fitting"""
    from nltools.models import Glm
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    # Create synthetic data
    np.random.seed(42)
    n_scans = 20
    img_shape = (10, 10, 10)
    fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
    affine = np.eye(4)
    img = Nifti1Image(fmri_data.T, affine)

    # Create mask
    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    # Create design matrix
    frame_times = np.arange(n_scans) * 2.0
    events = pd.DataFrame(
        {"onset": [0, 10], "duration": [1, 1], "trial_type": ["task", "task"]}
    )
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm"
    )

    # Fit and compute contrast
    model = Glm(t_r=2.0, mask=mask_img)
    model.fit(img, design_matrices=design_matrix)

    # Compute simple contrast (effect of task)
    contrast_map = model.compute_contrast("task")

    # Should return a Nifti image
    assert hasattr(contrast_map, "get_fdata")
    assert contrast_map.shape == img_shape


def test_glm_compute_contrast_before_fit_raises():
    """Glm should raise error if compute_contrast called before fit"""
    from nltools.models import Glm

    model = Glm()

    with pytest.raises(ValueError, match="not fitted"):
        model.compute_contrast("task")


def test_glm_multiple_runs():
    """Glm should handle multiple runs"""
    from nltools.models import Glm
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    # Create synthetic data for 2 runs
    np.random.seed(42)
    n_scans = 20
    img_shape = (10, 10, 10)

    # Create mask
    affine = np.eye(4)
    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    images = []
    design_matrices = []

    for run in range(2):
        fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
        img = Nifti1Image(fmri_data.T, affine)
        images.append(img)

        frame_times = np.arange(n_scans) * 2.0
        events = pd.DataFrame(
            {"onset": [0, 10], "duration": [1, 1], "trial_type": ["task", "task"]}
        )
        design_matrix = make_first_level_design_matrix(
            frame_times, events=events, hrf_model="spm"
        )
        design_matrices.append(design_matrix)

    # Fit GLM with multiple runs
    model = Glm(t_r=2.0, mask=mask_img)
    model.fit(images, design_matrices=design_matrices)

    assert model.is_fitted_
    # Verify progress_bar parameter exists and defaults to False
    assert hasattr(model, "progress_bar")
    assert model.progress_bar is False

    # Compute contrast across runs
    contrast_map = model.compute_contrast("task")
    assert hasattr(contrast_map, "get_fdata")


# ============================================================================
# Glm - Property Access (Advanced Features)
# ============================================================================


def test_glm_residuals_property():
    """Glm should expose residuals via property"""
    from nltools.models import Glm
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    # Create synthetic data
    np.random.seed(42)
    n_scans = 20
    img_shape = (10, 10, 10)
    fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
    affine = np.eye(4)
    img = Nifti1Image(fmri_data.T, affine)
    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    # Create design matrix
    frame_times = np.arange(n_scans) * 2.0
    events = pd.DataFrame(
        {"onset": [0, 10], "duration": [1, 1], "trial_type": ["task", "task"]}
    )
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm"
    )

    # Fit and access residuals
    model = Glm(t_r=2.0, mask=mask_img)
    model.fit(img, design_matrices=design_matrix)

    residuals = model.residuals
    assert isinstance(residuals, list)
    assert len(residuals) == 1  # One run
    assert hasattr(residuals[0], "get_fdata")
    # Verify progress_bar parameter exists and defaults to False
    assert hasattr(model, "progress_bar")
    assert model.progress_bar is False


def test_glm_design_matrices_property():
    """Glm should expose design_matrices_ via property"""
    from nltools.models import Glm
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    # Create synthetic data
    np.random.seed(42)
    n_scans = 20
    img_shape = (10, 10, 10)
    fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
    affine = np.eye(4)
    img = Nifti1Image(fmri_data.T, affine)
    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    # Create design matrix
    frame_times = np.arange(n_scans) * 2.0
    events = pd.DataFrame(
        {"onset": [0, 10], "duration": [1, 1], "trial_type": ["task", "task"]}
    )
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm"
    )

    # Fit and access design matrices
    model = Glm(t_r=2.0, mask=mask_img)
    model.fit(img, design_matrices=design_matrix)

    design_mats = model.design_matrices_
    assert isinstance(design_mats, list)
    assert len(design_mats) == 1  # One run
    assert "task" in design_mats[0].columns


def test_glm_glm_property_advanced_access():
    """Glm should expose internal FirstLevelModel for advanced use"""
    from nltools.models import Glm
    from nilearn.glm.first_level import FirstLevelModel

    model = Glm(t_r=2.0)

    # glm_ property should return FirstLevelModel instance
    assert isinstance(model.glm_, FirstLevelModel)
    assert model.glm_ is model._glm  # Same object


def test_glm_score_returns_valid_r_squared():
    """Glm.score() should return mean R² across voxels and runs"""
    from nltools.models import Glm
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    # Create synthetic fMRI data with signal
    np.random.seed(42)
    n_scans = 30
    img_shape = (8, 8, 8)  # Small for speed

    # Create data with actual signal for meaningful R²
    # Generate a signal that correlates with task timing
    task_signal = np.zeros(n_scans)
    task_signal[5:10] = 1.0  # Task block 1
    task_signal[20:25] = 1.0  # Task block 2

    # Create fMRI data: signal + noise
    fmri_data = np.zeros((n_scans, *img_shape), dtype=np.float32)
    for t in range(n_scans):
        # Add task signal to some voxels (not all, for realistic R²)
        signal_voxels = fmri_data[t, :4, :4, :4]  # Subset of voxels
        signal_voxels += task_signal[t] * 2.0  # Signal
        # Add noise everywhere
        fmri_data[t] += np.random.randn(*img_shape).astype(np.float32) * 0.5

    affine = np.eye(4)
    img = Nifti1Image(fmri_data.T, affine)  # nibabel uses (x,y,z,t) order

    # Create mask
    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    # Create design matrix with task regressor
    frame_times = np.arange(n_scans) * 2.0  # TR = 2s
    events = pd.DataFrame(
        {
            "onset": [10, 40],  # Match task blocks
            "duration": [10, 10],
            "trial_type": ["task", "task"],
        }
    )
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm"
    )

    # Fit GLM
    model = Glm(t_r=2.0, mask=mask_img)
    model.fit(img, design_matrices=design_matrix)

    # Test score() method
    r2 = model.score()

    # Assertions
    assert isinstance(r2, float), "score() should return float"
    assert 0.0 <= r2 <= 1.0, f"R² should be in [0, 1], got {r2}"
    assert r2 > 0.0, "R² should be positive with signal present"
    assert r2 < 1.0, "R² should not be perfect (noise present)"

    # With actual signal, R² should be meaningful (>0.01 at minimum)
    assert r2 > 0.01, f"R² should be >0.01 with clear signal, got {r2}"


def test_glm_score_before_fit_raises():
    """Glm.score() should raise error if called before fitting"""
    from nltools.models import Glm

    model = Glm(t_r=2.0)

    with pytest.raises(ValueError, match="not fitted yet"):
        model.score()


def test_glm_score_sklearn_api_compatibility():
    """Glm.score() should accept X and y for sklearn compatibility"""
    from nltools.models import Glm
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    # Create minimal fMRI data
    np.random.seed(42)
    n_scans = 20
    img_shape = (6, 6, 6)
    fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
    affine = np.eye(4)
    img = Nifti1Image(fmri_data.T, affine)

    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    # Create design matrix
    frame_times = np.arange(n_scans) * 2.0
    events = pd.DataFrame(
        {"onset": [0, 20], "duration": [5, 5], "trial_type": ["task", "task"]}
    )
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm"
    )

    # Fit and score
    model = Glm(t_r=2.0, mask=mask_img)
    model.fit(img, design_matrices=design_matrix)

    # score() should work with no arguments
    r2_no_args = model.score()

    # score() should also accept X and y (sklearn API) but ignore them
    r2_with_args = model.score(X=None, y=None)

    # Both should return same value
    assert r2_no_args == r2_with_args
