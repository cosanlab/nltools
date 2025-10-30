# Cross-Validation

Cross-validation utilities for neuroimaging analyses.

## Overview

The `nltools.cross_validation` module provides cross-validation tools designed for neuroimaging data. It includes stratified K-fold cross-validation and helpers for setting up CV schemes compatible with scikit-learn and nltools models.

## Key Functions

**KFoldStratified** - Stratified K-fold cross-validation
- Ensures balanced class representation across folds
- Compatible with Brain_Data and sklearn estimators
- Supports both classification and regression

**set_cv** - Configure cross-validation scheme
- Converts various CV specifications to sklearn splitters
- Handles integers, sklearn CV objects, and custom splitters
- Validates CV compatibility with data

## Quick Start

```python
from nltools.cross_validation import KFoldStratified, set_cv
from nltools.data import Brain_Data

# Create stratified K-fold CV
cv = KFoldStratified(n_splits=5)

# Use with Brain_Data.fit()
data = Brain_Data('images.nii.gz')
data.fit(model='ridge', X=design_matrix, cv=5)

# Or configure CV scheme
cv_splitter = set_cv(cv=5, y=labels)
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.cross_validation
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - Brain_Data.fit() with cv parameter
- {doc}`models` - Ridge and other models
- sklearn.model_selection - scikit-learn CV tools