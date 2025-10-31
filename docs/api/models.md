# `nltools.models`

**Scikit-learn Compatible Model Classes for Neuroimaging**

Provides sklearn-compatible model classes for common neuroimaging analyses.
Starting with Ridge regression, with GPU acceleration and cross-validation support.

```{eval-rst}
.. automodule:: nltools.models
    :members:
    :undoc-members:
    :show-inheritance:
```

## Quick Start

```python
from nltools.models import Ridge
import numpy as np

# Load your data
X = np.random.randn(100, 50000)  # samples × voxels
y = np.random.randn(100)          # target values

# Fit ridge regression with automatic alpha selection
model = Ridge(alpha='auto', cv=5, backend='auto')
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
r2_score = model.score(X, y)

print(f"Selected alpha: {model.alpha_}")
print(f"R² score: {r2_score:.3f}")
```

## Ridge Regression

The `Ridge` class provides ridge regression with:
- Automatic or manual alpha (regularization) selection
- Cross-validation for hyperparameter tuning
- GPU acceleration via PyTorch backends
- Single and multi-target regression
- sklearn-compatible API

### Basic Usage

**Fixed alpha:**
```python
from nltools.models import Ridge

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

**Automatic alpha selection:**
```python
# Let cross-validation select optimal alpha
model = Ridge(alpha='auto', cv=5)
model.fit(X_train, y_train)

print(f"Selected alpha: {model.alpha_}")
print(f"CV scores: {model.cv_scores_.mean()}")
```

**Custom alpha range:**
```python
# Specify alpha values to test
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
model = Ridge(alpha='auto', cv=5, alphas=alphas)
model.fit(X_train, y_train)
```

### GPU Acceleration

Ridge regression supports GPU acceleration for large-scale problems:

```python
# Automatic GPU detection
model = Ridge(alpha=1.0, backend='auto')
model.fit(X_train, y_train)

# Force GPU (if available)
model = Ridge(alpha=1.0, backend='torch')

# Force CPU
model = Ridge(alpha=1.0, backend='numpy')
```

**When to use GPU:**
- Large datasets (> 30M elements, e.g., 300 samples × 100k voxels)
- Cross-validation (multiplies effective problem size)
- Multi-target regression (many voxels/ROIs)

See [Performance Guide](../performance.md) for detailed benchmarks.

### Brain Data Integration

Use Ridge with nltools BrainData:

```python
from nltools import BrainData
from nltools.models import Ridge

# Load brain data
brain = BrainData('task_fmri.nii.gz')

# Create feature matrix (e.g., stimulus features)
features = np.random.randn(brain.shape()[0], 10)

# Fit encoding model
model = Ridge(alpha='auto', cv=5, backend='auto')
model.fit(features, brain.data)  # brain.data = samples × voxels

# Coefficients show feature weights for each voxel
print(f"Coefficients shape: {model.coef_.shape}")
# (10, n_voxels) - each feature weight across all voxels
```

### Encoding Models

Ridge is commonly used for encoding models in neuroimaging:

```python
from nltools import BrainData
from nltools.models import Ridge
import numpy as np

# Load fMRI data
brain = BrainData('movie_fmri.nii.gz')  # 1000 TRs × 50k voxels

# Create semantic features from movie annotations
# (e.g., valence, arousal, social features)
semantic_features = np.random.randn(1000, 20)

# Fit encoding model with GPU acceleration
model = Ridge(alpha='auto', cv=5, backend='auto')
model.fit(semantic_features, brain.data)

# Interpret: which features predict each brain region?
# model.coef_ shape: (20 features, 50k voxels)

# Predict brain activity from new features
new_clip_features = np.random.randn(100, 20)
predicted_brain = model.predict(new_clip_features)
# Shape: (100, 50k voxels)
```

### Decoding Models

Ridge can also be used for decoding (predict features from brain):

```python
from nltools.models import Ridge

# Flip X and y to decode features from brain activity
# X = brain activity (samples × voxels)
# y = behavioral/stimulus features to decode

model = Ridge(alpha='auto', cv=5)
model.fit(brain.data, behavior_scores)  # Decode behavior from brain

# Predict behavior from new brain scans
predicted_behavior = model.predict(new_brain_data)
```

## Comparison with scikit-learn

Ridge is fully compatible with scikit-learn's API:

```python
from nltools.models import Ridge as NLToolsRidge
from sklearn.linear_model import Ridge as SklearnRidge

# Both have the same interface
model_nl = NLToolsRidge(alpha=1.0)
model_sk = SklearnRidge(alpha=1.0, fit_intercept=False, solver='svd')

# Fit and predict work the same way
model_nl.fit(X, y)
model_sk.fit(X, y)

predictions_nl = model_nl.predict(X_test)
predictions_sk = model_sk.predict(X_test)

# Results match (within numerical precision)
np.testing.assert_allclose(predictions_nl, predictions_sk, rtol=1e-4)
```

**Advantages of nltools Ridge:**
- GPU acceleration for large neuroimaging datasets
- Automatic alpha selection via cross-validation
- Multi-target vectorization optimized for voxel-wise analysis
- Integrated with nltools BrainData workflow

## BaseModel

All model classes inherit from `BaseModel`, which provides:
- Consistent sklearn-compatible interface
- Input validation for neuroimaging data
- Fitted state tracking
- Shape checking and error messages

```python
from nltools.models import BaseModel

# All models follow this interface
class CustomModel(BaseModel):
    def fit(self, X, y):
        # Your fitting logic
        super().fit(X, y)  # Sets fitted state
        return self

    def predict(self, X):
        self._check_is_fitted()  # Validates model is fitted
        X = self._validate_X(X, reset=False)  # Validates features match
        # Your prediction logic
        return predictions

    def score(self, X, y):
        # Your scoring logic (e.g., R²)
        return score
```

## Migration from v0.5.1

If you used deprecated `BrainData.predict()`:

**Before (v0.5.1):**
```python
from nltools import BrainData

brain = BrainData('data.nii.gz')
brain.X = design_matrix
brain.Y = target_values

# Deprecated method
results = brain.predict(
    algorithm='ridge',
    cv_dict={'type': 'kfolds', 'n_folds': 5}
)
```

**After (v0.6.0):**
```python
from nltools import BrainData
from nltools.models import Ridge

brain = BrainData('data.nii.gz')

# Use Ridge model directly
model = Ridge(alpha='auto', cv=5, backend='auto')
model.fit(brain.data, target_values)
predictions = model.predict(brain.data)
```

## Performance Tips

### Memory Optimization

Ridge automatically uses float32 for memory efficiency:

```python
# Input data is automatically converted to float32
X_float64 = np.random.randn(100, 50000)  # float64 (8 bytes/element)
model = Ridge(alpha=1.0)
model.fit(X_float64, y)  # Internally uses float32 (4 bytes/element)
```

### Backend Selection

For optimal performance:

```python
from nltools.models import Ridge

# Small datasets (< 10M elements): NumPy is faster
model_small = Ridge(alpha=1.0, backend='numpy')

# Large datasets (> 30M elements): GPU is faster
model_large = Ridge(alpha=1.0, backend='torch')

# Let nltools decide based on problem size
model_auto = Ridge(alpha=1.0, backend='auto')
```

### Cross-Validation Performance

Cross-validation multiplies computational cost. Use GPU for CV when possible:

```python
# Without CV: 100 samples × 50k features = 5M elements (CPU okay)
model_no_cv = Ridge(alpha=1.0, backend='numpy')

# With 5-fold CV: 5× 100 samples × 50k features = 25M elements (GPU better)
model_cv = Ridge(alpha='auto', cv=5, backend='auto')
```

## API Reference

See the full API documentation above for details on:
- `BaseModel` - Abstract base class
- `Ridge` - Ridge regression with GPU support

### Key Classes

**`BaseModel`**
- Abstract base class for all models
- Provides sklearn-compatible interface
- Input validation and state management

**`Ridge(alpha=1.0, cv=None, alphas=None, backend='numpy', random_state=None)`**
- Ridge regression with optional GPU acceleration
- Parameters:
  - `alpha`: Regularization strength or 'auto' for CV selection
  - `cv`: Number of cross-validation folds (required if alpha='auto')
  - `alphas`: Custom alpha values to test during CV
  - `backend`: 'numpy', 'torch', or 'auto'
  - `random_state`: Random seed (for future compatibility)

## See Also

- [Performance Guide](../performance.md) - Benchmarks and optimization tips
- [Backends API](backends.md) - GPU acceleration details
- [Algorithms API](algorithms.md) - Lower-level ridge regression functions
- [Migration Guide](../migration-guide.md) - Upgrading from v0.5.1
