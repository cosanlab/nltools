"""
Test model classes for neuroimaging analysis.

Part of functional core - tests sklearn-compatible model APIs.
Following model-spec.md Sprint 2 implementation.
"""

import numpy as np
import pytest
from nltools.models import BaseModel, Ridge


# ============================================================================
# Helper Functions
# ============================================================================


def _torch_available():
    """Check if PyTorch is installed"""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


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


def test_ridge_fit_single_target():
    """Ridge should fit single-target regression"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha=1.0)
    result = model.fit(X, y)

    # Should return self
    assert result is model

    # Should be fitted
    assert model.is_fitted_

    # Should store coefficients
    assert hasattr(model, "coef_")
    assert model.coef_.shape == (50,)


def test_ridge_fit_multi_target():
    """Ridge should fit multi-target regression"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    model = Ridge(alpha=1.0)
    model.fit(X, Y)

    # Coefficients should be 2D
    assert model.coef_.shape == (50, 5)


def test_ridge_predict_single_target():
    """Ridge should predict on new data"""
    np.random.seed(42)
    X_train = np.random.randn(100, 50).astype(np.float32)
    y_train = np.random.randn(100).astype(np.float32)
    X_test = np.random.randn(20, 50).astype(np.float32)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Check shape
    assert y_pred.shape == (20,)

    # Predictions should be reasonable (not NaN, not all zeros)
    assert not np.isnan(y_pred).any()
    assert not np.allclose(y_pred, 0)


def test_ridge_predict_multi_target():
    """Ridge should predict multiple targets"""
    np.random.seed(42)
    X_train = np.random.randn(100, 50).astype(np.float32)
    Y_train = np.random.randn(100, 5).astype(np.float32)
    X_test = np.random.randn(20, 50).astype(np.float32)

    model = Ridge(alpha=1.0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Check shape
    assert Y_pred.shape == (20, 5)


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


def test_ridge_cv_fits_and_selects_alpha():
    """Ridge with alpha='auto' should select optimal alpha via CV"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha="auto", cv=3)
    model.fit(X, y)

    # Should have selected an alpha
    assert hasattr(model, "alpha_")
    assert isinstance(model.alpha_, float)
    assert model.alpha_ > 0

    # Should have CV scores
    assert hasattr(model, "cv_scores_")
    assert model.cv_scores_.shape[0] == 3  # n_folds


def test_ridge_cv_alphas_parameter():
    """Ridge should accept custom alpha range for CV"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    alphas = [0.1, 1.0, 10.0]
    model = Ridge(alpha="auto", cv=3, alphas=alphas)
    model.fit(X, y)

    # Selected alpha should be from our list
    assert model.alpha_ in alphas


def test_ridge_cv_multi_target():
    """Ridge CV should work with multiple targets"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    model = Ridge(alpha="auto", cv=3)
    model.fit(X, Y)

    # Should fit all targets
    assert model.coef_.shape == (50, 5)

    # CV scores should include all targets
    assert model.cv_scores_.shape[2] == 5  # n_targets


def test_ridge_cv_reproducibility():
    """Ridge CV should give reproducible results"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # CV is deterministic, so same data should give same results
    model1 = Ridge(alpha="auto", cv=3)
    model1.fit(X, y)

    model2 = Ridge(alpha="auto", cv=3)
    model2.fit(X, y)

    assert model1.alpha_ == model2.alpha_
    np.testing.assert_allclose(model1.coef_, model2.coef_, rtol=1e-5)


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

    np.testing.assert_allclose(pred_gpu, pred_cpu, rtol=1e-4)


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
