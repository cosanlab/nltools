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

## Questions or Issues?
Please report any migration issues at: https://github.com/cosanlab/nltools/issues

---
*Note: The Model class and Brain_Collection will be reimplemented in a future release (Priority 3) with enhanced functionality.*