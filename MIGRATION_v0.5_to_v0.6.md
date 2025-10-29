# Migration Guide: nltools v0.5.1 to v0.6.0

## Overview
Version 0.6.0 is a **breaking release** that refactors nltools to better leverage nilearn functionality and establish a cleaner architecture. This guide helps you migrate your code.

## Breaking Changes

### 1. Removed Classes
The following classes have been removed from this release and will be reimplemented in a future Priority 3 release:
- `Brain_Collection` - Will be part of new Model framework
- `Model` - Future class for advanced ML workflows

### 2. Removed Brain_Data Methods
The following methods have been deprecated and moved to the future Model class:

#### `.predict()`
**Old:**
```python
results = brain_data.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5})
```

**New:** Will be available in Model class (Priority 3). For now, use scikit-learn directly:
```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

# Extract data
X = brain_data.data
y = your_labels  # Manage labels separately

# Use scikit-learn directly
clf = SVC(kernel='linear')
cv_results = cross_validate(clf, X, y, cv=5)
```

#### `.ttest()`
**Old:**
```python
t_map = brain_data.ttest(threshold_dict={'unc': 0.05})
```

**New:** Will be available in Model class. For now, use scipy:
```python
from scipy import stats
t_stat, p_values = stats.ttest_1samp(brain_data.data, 0, axis=0)
```

#### `.randomise()`
**Old:**
```python
results = brain_data.randomise(n_permute=5000)
```

**New:** Will be available in Model class. Consider using nilearn's permutation testing.

#### `.predict_multi()`
**Old:**
```python
results = brain_data.predict_multi(method='searchlight')
```

**New:** Will be available in Model class. Use nilearn's SearchLight directly for now.

### 3. Deprecated Attributes
- `.X` - Deprecated, still works for backward compatibility but will be removed in v0.7.0
- `.Y` - Deprecated, still works for backward compatibility but will be removed in v0.7.0

Labels and design matrices should now be managed separately or passed as arguments. The `.X` attribute is still used internally by `.regress()` for backward compatibility when no design_matrix is provided.

## Updated Methods

### `.regress()`
Now stores results as attributes and prefers passing design_matrix directly. Old API still works with deprecation warnings.

**Old (still works with deprecation warning):**
```python
brain_data.X = design_matrix
results = brain_data.regress()  # DeprecationWarning: Use regress(design_matrix)
# Returns dict with 'beta', 't', 'p', 'residual' for backward compatibility
```

**New (recommended):**
```python
brain_data.regress(design_matrix)
# Results stored as attributes:
# brain_data.glm_betas
# brain_data.glm_t
# brain_data.glm_p
# brain_data.glm_se
# brain_data.glm_residual
# brain_data.glm_predicted
# brain_data.glm_r2

# Note: Currently still returns dict for backward compatibility (deprecated)
```

**Note on robust mode:**
The `mode='robust'` parameter is deprecated and ignored. Robust regression will be reimplemented in the future Model class.

### `.extract_roi()`
Now uses nilearn's NiftiLabelsMasker for better performance with labeled atlases.

**Old & New:** Interface remains the same
```python
roi_values = brain_data.extract_roi(atlas_mask)
```

**Note:** Invalid metrics now raise `NotImplementedError` instead of `ValueError` for consistency with other deprecated methods.

### `.smooth()`
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

## New Methods

### `.compute_contrasts()`
Compute contrasts from GLM results using string specifications or numeric vectors.

```python
# First run regression
brain_data.regress(design_matrix)

# Compute contrasts
contrast1 = brain_data.compute_contrasts("conditionA - conditionB")

# Multiple contrasts
contrasts = brain_data.compute_contrasts({
    "main_effect": "conditionA - conditionB",
    "interaction": [1, -1, -1, 1]
})
```

## HDF5 File Compatibility
- Files saved with v0.5.1 can still be loaded
- `.X` and `.Y` fields in old files are loaded as attributes for backward compatibility
- Legacy HDF5 format (pre-0.4.8) is still supported with automatic detection
- New saves will store X and Y for backward compatibility if they exist

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
brain.regress(design_matrix)
betas = brain.glm_betas  # Access as attributes

# Compute contrasts - new feature
contrast = brain.compute_contrasts("conditionA - conditionB")

# Extract ROIs - same interface, better performance
roi_data = brain.extract_roi(atlas)
```

## Testing Your Code

After upgrading, test for deprecated methods:
```python
try:
    brain.predict()
except NotImplementedError as e:
    print(f"Method deprecated: {e}")
    # Update your code to use alternatives
```

## Documentation Status

### Tutorial Updates
All tutorials have been updated for v0.6.0 with the following approach:
- Tutorials remain pedagogically complete with explanatory text
- Code using deprecated methods (`.predict()`, `.ttest()`, etc.) has been commented with TODO markers
- Documentation builds successfully (`jupyter-book build docs/`) for internal tracking
- See `docs/TODO_TRACKER.md` for a complete list of tutorials awaiting Priority 3 features

### Working Tutorials
The following tutorials work fully with v0.6.0:
- All Brain_Data, Design_Matrix, and Adjacency basic tutorials
- All data operation tutorials
- Decomposition, similarity, and hyperalignment analysis tutorials

### Commented Tutorials
Tutorials with commented code blocks waiting for Model class implementation:
- Multivariate classification and prediction tutorials (awaiting `.predict()`)
- Univariate regression and statistical testing tutorials (awaiting `.ttest()`)
- Brain_Collection tutorial (awaiting class implementation)

These will be uncommented and updated as Priority 3 features are implemented.

## New Feature: Ridge Model Class (v0.6.0)

### Overview
nltools v0.6.0 introduces sklearn-compatible model classes, starting with Ridge regression:

```python
from nltools.models import Ridge
import numpy as np

# Basic usage
X = np.random.randn(100, 50)  # samples × features (e.g., voxels)
y = np.random.randn(100)       # target values

model = Ridge(alpha=1.0)
model.fit(X, y)
y_pred = model.predict(X)
r2_score = model.score(X, y)
```

### Ridge Regression Features

**Fixed alpha:**
```python
# Single target
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Multi-target (e.g., predict multiple ROIs)
Y_train = np.random.randn(100, 5)  # 5 targets
model = Ridge(alpha=1.0)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)  # shape: (n_samples, 5)
```

**Automatic alpha selection via cross-validation:**
```python
model = Ridge(alpha='auto', cv=5)
model.fit(X_train, y_train)
print(f"Selected alpha: {model.alpha_}")
print(f"CV scores shape: {model.cv_scores_.shape}")  # (n_folds, n_alphas, n_targets)
```

**Custom alpha range:**
```python
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
model = Ridge(alpha='auto', cv=5, alphas=alphas)
model.fit(X_train, y_train)
```

**GPU acceleration:**
```python
# Automatic GPU detection
model = Ridge(alpha=1.0, backend='auto')
model.fit(X_train, y_train)

# Force GPU (if available)
model = Ridge(alpha=1.0, backend='torch')

# Force CPU
model = Ridge(alpha=1.0, backend='numpy')
```

### Migration from Deprecated Methods

**Before (v0.5.1):**
```python
from nltools import Brain_Data

brain = Brain_Data('data.nii.gz')
brain.X = design_matrix
brain.Y = target_values

# This is now deprecated
results = brain.predict(algorithm='ridge', cv_dict={'type': 'kfolds', 'n_folds': 5})
```

**After (v0.6.0):**
```python
from nltools import Brain_Data
from nltools.models import Ridge

brain = Brain_Data('data.nii.gz')

# Use Ridge model directly
model = Ridge(alpha='auto', cv=5)
model.fit(brain.data, target_values)  # brain.data = samples × voxels
predictions = model.predict(brain.data)

# Or with GPU acceleration
model = Ridge(alpha='auto', cv=5, backend='auto')
model.fit(brain.data, target_values)
```

### Practical Example: Encoding Model

```python
from nltools import Brain_Data
from nltools.models import Ridge
import numpy as np

# Load brain data
brain = Brain_Data('task_fmri.nii.gz')  # 200 samples × 50000 voxels

# Create feature matrix (e.g., stimulus attributes)
features = np.random.randn(200, 10)  # 200 timepoints × 10 features

# Fit encoding model with automatic alpha selection
model = Ridge(alpha='auto', cv=5, backend='auto')
model.fit(features, brain.data)  # Predict all 50k voxels

# Coefficients show feature weights for each voxel
print(f"Coefficients shape: {model.coef_.shape}")  # (10, 50000)
print(f"Selected alpha: {model.alpha_}")

# Predict brain activity from new features
new_features = np.random.randn(50, 10)
predicted_activity = model.predict(new_features)  # shape: (50, 50000)
```

### Performance Notes

- **Small datasets** (< 10M elements): NumPy backend is faster
- **Large datasets** (> 30M elements): GPU backend provides speedup
- **Cross-validation**: GPU often faster even for medium datasets
- On Apple Silicon (MPS): 1.4-2.2x speedup (SVD falls back to CPU)
- On NVIDIA CUDA: Expected 10-30x speedup

See `docs/performance.md` for detailed benchmarks.

## Questions or Issues?
Please report any migration issues at: https://github.com/cosanlab/nltools/issues

---
*Note: The Model class and Brain_Collection will be reimplemented in a future release (Priority 3) with enhanced functionality.*