# `nltools.cross_validation`

Cross-validation utilities for neuroimaging analyses.

## Overview

The `nltools.cross_validation` module provides `KFoldStratified`, a scikit-learn compatible cross-validation splitter that stratifies continuous regression targets. Unlike sklearn's `StratifiedKFold` which only works with discrete class labels, `KFoldStratified` works with continuous targets by sorting values and distributing them evenly across folds.

## Key Functions

**KFoldStratified** - Stratified K-fold cross-validation for continuous targets
- Stratifies continuous regression targets by sorting and distributing evenly
- Compatible with sklearn BaseCrossValidator API
- Works with sklearn's cross_val_score and other utilities
- Unlike sklearn's StratifiedKFold, supports continuous targets (not just discrete classes)

## Quick Start

```python
from nltools.cross_validation import KFoldStratified
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Create stratified K-fold CV for continuous targets
cv = KFoldStratified(n_splits=5)

# Use with sklearn utilities
X = np.random.randn(100, 10)
y = np.random.randn(100)  # Continuous target
model = LinearRegression()

scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

# Use with BrainData.fit()
from nltools.data import BrainData
data = BrainData('images.nii.gz')
data.fit(model='ridge', X=design_matrix, cv=cv)  # Pass CV object directly
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.cross_validation
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - BrainData.fit() with cv parameter
- {doc}`models` - Ridge and other models
- sklearn.model_selection - scikit-learn CV tools