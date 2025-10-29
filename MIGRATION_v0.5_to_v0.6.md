# Migration Guide: nltools v0.5.1 to v0.6.0

**Last Updated:** 2025-10-29

## Overview
Version 0.6.0 is a **breaking release** that refactors nltools to better leverage nilearn functionality and establish a cleaner architecture. This guide helps you migrate your code.

---

## Breaking Changes

### 1. Removed Classes
The following classes have been removed from this release and will be reimplemented in a future Priority 3 release:
- `Brain_Collection` - Will be part of new Model framework
- `Model` - Future class for advanced ML workflows

### 2. Removed Brain_Data Methods
The following methods have been deprecated and moved to the future Model class:

#### `.predict()` - Use scikit-learn directly
**Old:**
```python
results = brain_data.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5})
```

**New:** Use scikit-learn directly:
```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

X = brain_data.data
y = your_labels  # Manage labels separately

clf = SVC(kernel='linear')
cv_results = cross_validate(clf, X, y, cv=5)
```

#### `.ttest()` - Use scipy.stats
**Old:**
```python
t_map = brain_data.ttest(threshold_dict={'unc': 0.05})
```

**New:** Use scipy:
```python
from scipy import stats
t_stat, p_values = stats.ttest_1samp(brain_data.data, 0, axis=0)
```

#### `.randomise()` and `.predict_multi()`
These methods will be available in the future Model class. Consider using nilearn's permutation testing or SearchLight for now.

### 3. Deprecated Attributes
- `.X` - Deprecated, still works for backward compatibility but will be removed in v0.7.0
- `.Y` - Deprecated, still works for backward compatibility but will be removed in v0.7.0

Labels and design matrices should now be managed separately or passed as arguments.

---

## New Feature: Brain_Data.fit() and .predict()

Brain_Data now supports sklearn-style `fit()` and `predict()` methods for Ridge and GLM models. This is the **recommended API** going forward.

### Ridge Regression Workflow

```python
from nltools import Brain_Data
import numpy as np

# Load brain data
brain = Brain_Data('task_fmri.nii.gz')  # 100 samples × 50000 voxels

# Create feature matrix
features = np.random.randn(100, 10)  # 100 samples × 10 features

# Fit Ridge model with automatic alpha selection
brain.fit(model='ridge', alpha='auto', cv=5, X=features)

# Access results as attributes
print(brain.ridge_weights.shape)  # (10, 50000) - feature weights per voxel
print(brain.ridge_scores.shape)   # (1, 50000) - R² per voxel

# Predict on new data
new_features = np.random.randn(20, 10)
predictions = brain.predict(X=new_features)  # Returns Brain_Data (20, 50000)

# Or predict on training data (X=None)
fitted_values = brain.predict()  # Returns Brain_Data (100, 50000)
```

### GLM Regression Workflow

```python
import pandas as pd

# Create design matrix
design_matrix = pd.DataFrame({
    'Intercept': np.ones(100),
    'ConditionA': condition_a,
    'ConditionB': condition_b
})

# Fit GLM model
brain.fit(model='glm', noise_model='ar1', X=design_matrix)

# Access results as attributes
print(brain.glm_betas.shape)  # (3, 50000) - beta per regressor × voxel
print(brain.glm_t.shape)      # (3, 50000) - t-stat per regressor × voxel

# Predict fitted values
fitted = brain.predict()  # Returns Brain_Data with fitted values
```

### Available Models
- `'ridge'`: Ridge regression with optional cross-validation and GPU acceleration
- `'glm'`: General Linear Model wrapping nilearn's FirstLevelModel

### Attributes Set by fit()

**For Ridge models** (`model='ridge'`):
- `model_`: Fitted Ridge model instance
- `X_`: Training data (for predict() default)
- `ridge_weights`: Brain_Data of coefficients (n_features, n_voxels)
- `ridge_fitted_values`: Brain_Data of fitted values
- `ridge_scores`: Brain_Data of R² scores

**For GLM models** (`model='glm'`):
- `model_`: Fitted Glm model instance
- `X_`: Training design matrix
- `glm_betas`: Brain_Data of beta coefficients
- `glm_t`: Brain_Data of t-statistics
- `glm_p`: Brain_Data of p-values
- `glm_se`: Brain_Data of standard errors
- `glm_residual`: Brain_Data of residuals
- `glm_predicted`: Brain_Data of fitted values
- `glm_r2`: Brain_Data of R² values

---

## New Feature: Cross-Validation Support

**NEW in v0.6.0:** The `fit()` method supports cross-validation for Ridge regression via the `cv` parameter.

### Basic CV for Performance Reporting

```python
# Fit Ridge with 5-fold CV
brain.fit(model='ridge', alpha=1.0, cv=5, X=features)

# Access CV results
print(f"Mean CV R²: {brain.cv_results_['mean_score'].mean():.3f}")
print(f"CV scores shape: {brain.cv_results_['scores'].shape}")  # (5, n_voxels)

# CV predictions are out-of-fold predictions
cv_predictions = brain.cv_results_['predictions']  # Brain_Data object
fold_ids = brain.cv_results_['folds']  # (n_samples,)
```

### Automatic Alpha Selection

```python
# cv='auto' triggers alpha selection (default: 5 folds)
brain.fit(model='ridge', cv='auto', alphas=[0.1, 1.0, 10.0, 100.0], X=features)

# Access best alpha
best_alpha = brain.cv_results_['best_alpha']
print(f"Selected alpha: {best_alpha}")

# Alpha selection scores: (n_folds, n_alphas, n_voxels)
alpha_scores = brain.cv_results_['alpha_scores']
```

### Combining Alpha Selection + CV Scoring

```python
# Alpha selection + CV scoring with explicit fold count
brain.fit(model='ridge', alpha='auto', cv=3, alphas=[0.1, 1.0, 10.0], X=features)

# Results contain both alpha selection and final CV scores
print(f"Best alpha: {brain.cv_results_['best_alpha']}")
print(f"CV R² with best alpha: {brain.cv_results_['mean_score'].mean():.3f}")
```

### Custom CV Strategies

```python
from sklearn.model_selection import KFold

# Custom CV strategy
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
brain.fit(model='ridge', alpha=1.0, cv=cv_splitter, X=features)
```

### CV Results Dictionary

When `cv` is provided, a `cv_results_` dict is created with:
- `'scores'`: (n_folds, n_voxels) R² per fold and voxel
- `'mean_score'`: (n_voxels,) mean R² across folds
- `'predictions'`: Brain_Data with out-of-fold predictions
- `'folds'`: (n_samples,) fold index for each sample
- `'best_alpha'`: Selected alpha (if alpha selection performed)
- `'alpha_scores'`: (n_folds, n_alphas, n_voxels) R² grid (if alpha selection)

**Important notes:**
- The full model is always fitted on *all* training data after CV
- `predict()` returns predictions from the full model, not CV predictions
- CV predictions are out-of-fold and available in `cv_results_['predictions']`
- CV is currently only supported for Ridge models (not GLM)

---

## Updated Methods

### .regress() - DEPRECATED

**⚠️ IMPORTANT:** `.regress()` is deprecated and will raise an error in v0.7.0. Use `fit(model='glm', X=design_matrix)` instead.

**Old (deprecated, emits FutureWarning):**
```python
brain_data.X = design_matrix
results = brain_data.regress()  # FutureWarning
# Returns dict with 'beta', 't', 'p', 'residual' for backward compatibility
```

**New (recommended):**
```python
# Use fit/predict API
brain_data.fit(model='glm', noise_model='ar1', X=design_matrix)

# Results stored as attributes:
# brain_data.glm_betas, glm_t, glm_p, glm_se,
# glm_residual, glm_predicted, glm_r2

# Predict fitted values or new timepoints
fitted = brain_data.predict()  # X=None uses training data
new_pred = brain_data.predict(X=new_design_matrix)
```

**Migration note:**
- `.regress()` now calls `fit(model='glm', ...)` internally
- Returns dict for backward compatibility (deprecated)
- Sets `glm_model` attribute as alias for `model_`
- `mode='robust'` parameter is silently ignored

### .extract_roi()
Now uses nilearn's NiftiLabelsMasker for better performance with labeled atlases. Interface remains the same.

**Note:** Invalid metrics now raise `NotImplementedError` instead of `ValueError` for consistency with other deprecated methods.

### .smooth()
Now returns a copy instead of modifying the object in-place.

**Old behavior (modified in-place):**
```python
brain_data.smooth(5.0)  # Modified brain_data directly
```

**New behavior (returns copy):**
```python
smoothed = brain_data.smooth(5.0)  # Returns new Brain_Data object
# Original brain_data is unchanged
```

### .compute_contrasts()
Compute contrasts from GLM results using string specifications or numeric vectors.

```python
# First fit GLM
brain_data.fit(model='glm', X=design_matrix)

# Compute contrasts
contrast1 = brain_data.compute_contrasts("conditionA - conditionB")

# Multiple contrasts
contrasts = brain_data.compute_contrasts({
    "main_effect": "conditionA - conditionB",
    "interaction": [1, -1, -1, 1]
})
```

---

## New Feature: HyperAlignment Class

A new `HyperAlignment` class has been extracted from the `align()` function, providing direct access to Procrustes-based hyperalignment with a clean sklearn-style API.

### Basic Usage

```python
from nltools.algorithms import HyperAlignment
import numpy as np

# Create sample multi-subject data (list of [features x samples] matrices)
data = [subject1_data, subject2_data, subject3_data]

# Fit hyperalignment model
hyper = HyperAlignment(n_iter=2, auto_pad=True)
hyper.fit(data)

# Transform data to common space
aligned_data = hyper.transform(data)

# Access common template
common_template = hyper.s_  # or hyper.common_model_

# Access transformation matrices
transformations = hyper.w_

# Align a new subject to the common space
new_subject_data = ...  # [features x samples]
transformed, R, disparity, scale = hyper.transform_subject(new_subject_data)
```

### Parameters
- `n_iter` (int, default=2): Number of template refinement iterations
- `auto_pad` (bool, default=True): Automatically zero-pad matrices to handle different feature counts

### Attributes
- `w_`: List of transformation matrices (one per subject)
- `s_`: Common template in aligned space
- `common_model_`: Alias for `s_` (backward compatibility)
- `disparity_`: Alignment quality metrics (sum of squared differences)
- `scale_`: Scale factors for each subject

### Why Use HyperAlignment Directly?
- More control over alignment parameters (`n_iter`, `auto_pad`)
- Access to intermediate outputs (transformation matrices, quality metrics)
- Reusable model for aligning new subjects
- Clean sklearn-compatible API

**Backward compatibility:** The `align(method='procrustes')` function continues to work identically, now using `HyperAlignment` internally.

---

## HDF5 File Compatibility
- Files saved with v0.5.1 can still be loaded
- `.X` and `.Y` fields in old files are loaded as attributes for backward compatibility
- Legacy HDF5 format (pre-0.4.8) is still supported with automatic detection
- New saves will store X and Y for backward compatibility if they exist

---

## Code Examples

### Before (v0.5.1)
```python
from nltools import Brain_Data

# Load data
brain = Brain_Data('data.nii.gz')
brain.Y = pd.DataFrame({'label': [0, 1, 0, 1]})

# Run prediction
results = brain.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5})

# Run GLM
brain.X = design_matrix
glm_results = brain.regress()
betas = glm_results['beta']

# Extract ROIs
roi_data = brain.extract_roi(atlas)
```

### After (v0.6.0)
```python
from nltools import Brain_Data
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

# Load data
brain = Brain_Data('data.nii.gz')
labels = pd.DataFrame({'label': [0, 1, 0, 1]})  # Manage separately

# For prediction - use scikit-learn directly
X = brain.data
y = labels['label']
clf = SVC(kernel='linear')
cv_results = cross_validate(clf, X, y, cv=5)

# Run GLM - new interface
brain.fit(model='glm', X=design_matrix)
betas = brain.glm_betas  # Access as attributes

# Compute contrasts - new feature
contrast = brain.compute_contrasts("conditionA - conditionB")

# Extract ROIs - same interface, better performance
roi_data = brain.extract_roi(atlas)
```

---

## Testing Your Code

After upgrading, test for deprecated methods:
```python
try:
    brain.predict()
except NotImplementedError as e:
    print(f"Method deprecated: {e}")
    # Update your code to use alternatives
```

---

## Advanced Usage: Direct Model Access

For advanced users who need direct access to the underlying Ridge and Glm model classes (not typical usage):

```python
from nltools.models import Ridge, Glm

# Ridge model (standalone)
model = Ridge(alpha=1.0, backend='auto')
model.fit(X, y)
predictions = model.predict(X_test)

# Glm model (standalone)
model = Glm(t_r=2.0, noise_model='ar1')
model.fit(fmri_img, design_matrices=design_matrix)
contrast_map = model.compute_contrast('task')
```

**Note:** Most users should use `Brain_Data.fit(model='ridge'|'glm')` instead, which provides a more convenient interface with automatic result storage as Brain_Data attributes.

---

## Documentation Status

### Tutorial Require Updating
All tutorials are consistent with previous version not new changes

---

*Note: The Model class and Brain_Collection will be reimplemented in a future release (Priority 3) with enhanced functionality.*

*Last updated: 2025-10-29*
*Lines: ~450 (condensed from 724)*
