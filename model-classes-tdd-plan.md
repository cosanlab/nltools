# Test-Driven Development Plan: BaseModel + Ridge Model Classes
**Sprint 2 from model-spec.md - Focused Scope**

**Created:** 2025-10-28
**Status:** Planning → Implementation

---

## Executive Summary

**What we're building:**
- **BaseModel**: Abstract base class defining sklearn-compatible interface
- **Ridge**: Concrete ridge regression model wrapping our `ridge.py` algorithms

**Why narrow scope:**
- Focus on establishing solid model architecture foundation
- Get sklearn API patterns right before expanding
- GLMModel and InferenceModel can follow this pattern later
- Incremental delivery: working Ridge model enables Brain_Data integration next

**Integration with existing code:**
- Uses **`nltools/algorithms/ridge.py`** (already complete from Sprint 1)
- Uses **`nltools/backends.py`** (already complete from Sprint 1)
- New tests in **`nltools/tests/core/test_models.py`** (function-based)
- New implementation in **`nltools/models/`** directory

**Timeline:** 1-2 days for both classes
**Branch:** Continue on `uv-cleanup`
**Current baseline:** 91 tests passing

---

## Architecture Overview

### Class Hierarchy

```
BaseModel (abstract)
└── Ridge (concrete ridge regression)
```

**Design principles:**
1. **sklearn-compatible API**: `.fit()`, `.predict()`, `.score()`
2. **Neuroimaging-aware**: Handles 2D arrays (samples × voxels/features)
3. **GPU-optional**: Leverages existing Backend abstraction
4. **Stateful but predictable**: Clear fit/predict separation
5. **Multi-target support**: Vectorized operations across voxels

### File Structure

```
nltools/
├── models/
│   ├── __init__.py           # Exports BaseModel, Ridge
│   ├── base.py               # BaseModel abstract class
│   └── ridge.py              # Ridge concrete class
├── algorithms/
│   └── ridge.py              # ✅ Already exists (ridge_svd, ridge_cv)
├── backends.py               # ✅ Already exists
└── tests/
    └── core/
        ├── test_backends.py  # ✅ Already exists
        ├── test_ridge.py     # ✅ Already exists (algorithms)
        └── test_models.py    # 🆕 New file for model classes
```

---

## Pre-Implementation: Environment Setup

### Step 0: Verify Current State & Create Structure

**Actions:**
1. Verify baseline tests pass (91 tests)
2. Create models/ directory structure
3. Create test file

**Commands:**
```bash
# 1. Verify clean baseline
uv run pytest nltools/tests/ -x

# 2. Create models module
mkdir -p nltools/models
touch nltools/models/__init__.py
touch nltools/models/base.py
touch nltools/models/ridge.py

# 3. Create test file
touch nltools/tests/core/test_models.py

# 4. Verify structure
tree nltools/models/
ls -la nltools/tests/core/
```

**Deliverables:**
- [ ] Confirm 91 tests passing
- [ ] Empty module files created
- [ ] Empty test file created

---

## Phase 1: BaseModel Abstract Class (TDD Cycles)

**Pattern:** All tests go in `nltools/tests/core/test_models.py` as **functions**

### Cycle 1.1: BaseModel Interface Definition

**Write Tests First:**

```python
# nltools/tests/core/test_models.py

"""
Test model classes for neuroimaging analysis.

Part of functional core - tests sklearn-compatible model APIs.
Following model-spec.md Sprint 2 implementation.
"""

import numpy as np
import pytest
from nltools.models import BaseModel


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

    assert hasattr(model, 'n_features_in_')
    assert model.n_features_in_ == 50
    assert hasattr(model, 'n_samples_')
    assert model.n_samples_ == 100
```

**TDD Workflow:**
```bash
# Run tests (expect failures)
uv run pytest nltools/tests/core/test_models.py -k "basemodel" -xvs 2>&1 | tee test_basemodel.log

# Implement BaseModel in nltools/models/base.py

# Verify tests pass
uv run pytest nltools/tests/core/test_models.py -k "basemodel" -xvs

# Check for regressions
uv run pytest nltools/tests/ -x
```

**Implementation Guide:**

```python
# nltools/models/base.py

"""
Base classes for nltools models.

Provides sklearn-compatible API for neuroimaging analysis.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all nltools models.

    Follows scikit-learn API conventions:
    - fit(X, y) trains the model and returns self
    - predict(X) generates predictions
    - score(X, y) evaluates model performance

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit
    n_samples_ : int
        Number of samples seen during fit
    is_fitted_ : bool
        Whether the model has been fitted
    """

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Returns
        -------
        self : BaseModel
            Fitted model instance
        """
        # Store training dimensions
        self.n_samples_, self.n_features_in_ = X.shape
        self.is_fitted_ = True
        return self

    @abstractmethod
    def predict(self, X):
        """
        Generate predictions for new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict on

        Returns
        -------
        y_pred : ndarray
            Predicted values
        """
        pass

    @abstractmethod
    def score(self, X, y):
        """
        Evaluate model performance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            True values

        Returns
        -------
        score : float
            Model performance metric
        """
        pass

    def _check_is_fitted(self):
        """
        Check if model has been fitted.

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError(
                f"{self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this model."
            )
```

---

### Cycle 1.2: BaseModel Input Validation

**Write Tests:**

```python
# Add to nltools/tests/core/test_models.py

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
```

**TDD Workflow:**
```bash
# Run validation tests
uv run pytest nltools/tests/core/test_models.py -k "validates" -xvs 2>&1 | tee test_validation.log

# Implement validation methods in BaseModel

# Verify
uv run pytest nltools/tests/core/test_models.py -k "basemodel" -xvs
```

**Implementation:** Add validation methods to `BaseModel`:

```python
# Add to nltools/models/base.py

    def _validate_X(self, X, reset=True):
        """Validate input data X."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {X.ndim}D array instead. "
                f"Reshape your data using array.reshape(-1, 1) for single feature."
            )

        if not reset and hasattr(self, 'n_features_in_'):
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                    f"was fitted with {self.n_features_in_} features."
                )

        return X

    def _validate_X_y(self, X, y):
        """Validate input data X and target y."""
        X = self._validate_X(X)
        y = np.asarray(y)

        if y.ndim > 2:
            raise ValueError(
                f"y should be 1D or 2D array, got {y.ndim}D array instead."
            )

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent number of samples: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )

        return X, y
```

---

### Phase 1 Completion Checklist

```bash
# Run all BaseModel tests
uv run pytest nltools/tests/core/test_models.py -k "basemodel" -xvs

# Verify no regressions
uv run pytest nltools/tests/ -x

# Count new tests
uv run pytest nltools/tests/core/test_models.py -k "basemodel" --collect-only
```

**Expected outcome:**
- [ ] ~10 BaseModel tests passing
- [ ] No regressions (91 existing tests still pass)
- [ ] BaseModel abstract class complete (~100 lines)
- [ ] Ready for Phase 2 (Ridge implementation)

---

## Phase 2: Ridge Model Class (TDD Cycles)

**Pattern:** Build on BaseModel, wrap existing `ridge.py` algorithms

### Cycle 2.1: Ridge Basic Fit/Predict

**Write Tests:**

```python
# Add to nltools/tests/core/test_models.py

from nltools.models import Ridge

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
    assert hasattr(model, 'coef_')
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
    model_sklearn = SklearnRidge(alpha=alpha, fit_intercept=False, solver='svd')
    model_sklearn.fit(X, y)
    pred_sklearn = model_sklearn.predict(X)

    # Should match
    np.testing.assert_allclose(pred_ours, pred_sklearn, rtol=1e-4)
    np.testing.assert_allclose(model_ours.coef_, model_sklearn.coef_, rtol=1e-4)
```

**TDD Workflow:**
```bash
# Run tests (expect failures)
uv run pytest nltools/tests/core/test_models.py -k "ridge" -xvs 2>&1 | tee test_ridge_model.log

# Implement Ridge class in nltools/models/ridge.py

# Verify
uv run pytest nltools/tests/core/test_models.py -k "ridge" -xvs
```

**Implementation Guide:**

```python
# nltools/models/ridge.py

"""
Ridge regression model for neuroimaging data.

Wraps nltools.algorithms.ridge with sklearn-compatible API.
"""

import numpy as np
from .base import BaseModel
from ..algorithms.ridge import ridge_svd
from ..backends import Backend


class Ridge(BaseModel):
    """
    Ridge regression with optional GPU acceleration.

    Wraps nltools SVD-based ridge regression algorithms with
    scikit-learn compatible API. Supports single and multi-target
    regression with optional GPU acceleration via PyTorch.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Must be positive.
    backend : str or Backend, default='numpy'
        Computational backend ('numpy', 'torch', or 'auto')

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_features, n_targets)
        Ridge coefficients
    backend_ : Backend
        Backend instance used for computation

    Examples
    --------
    >>> from nltools.models import Ridge
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = Ridge(alpha=1.0)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    """

    def __init__(self, alpha=1.0, backend='numpy'):
        super().__init__()
        self.alpha = alpha
        self.backend = backend

    def fit(self, X, y):
        """
        Fit ridge regression model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Returns
        -------
        self : Ridge
            Fitted model instance
        """
        # Validate inputs
        X, y = self._validate_X_y(X, y)

        # Set up backend
        if isinstance(self.backend, str):
            self.backend_ = Backend(self.backend)
        else:
            self.backend_ = self.backend

        # Fit using ridge_svd algorithm
        self.coef_ = ridge_svd(X, y, alpha=self.alpha, backend=self.backend_)

        # Call parent fit to set fitted state and store dimensions
        super().fit(X, y)

        return self

    def predict(self, X):
        """
        Predict using the ridge model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values
        """
        self._check_is_fitted()
        X = self._validate_X(X, reset=False)

        # Compute predictions
        y_pred = X @ self.coef_

        return y_pred

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            True values for X

        Returns
        -------
        score : float
            R^2 of self.predict(X) vs y
        """
        self._check_is_fitted()
        X, y = self._validate_X_y(X, y)

        y_pred = self.predict(X)

        # Compute R^2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return r2
```

---

### Cycle 2.2: Ridge Cross-Validation Support

**Write Tests:**

```python
# Add to nltools/tests/core/test_models.py

# ============================================================================
# Ridge Model - Cross-Validation
# ============================================================================

def test_ridge_cv_instantiation():
    """Ridge with cv should instantiate properly"""
    model = Ridge(alpha='auto', cv=5)
    assert model.alpha == 'auto'
    assert model.cv == 5


def test_ridge_cv_fits_and_selects_alpha():
    """Ridge with alpha='auto' should select optimal alpha via CV"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha='auto', cv=3)
    model.fit(X, y)

    # Should have selected an alpha
    assert hasattr(model, 'alpha_')
    assert isinstance(model.alpha_, float)
    assert model.alpha_ > 0

    # Should have CV scores
    assert hasattr(model, 'cv_scores_')
    assert model.cv_scores_.shape[0] == 3  # n_folds


def test_ridge_cv_alphas_parameter():
    """Ridge should accept custom alpha range for CV"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    alphas = [0.1, 1.0, 10.0]
    model = Ridge(alpha='auto', cv=3, alphas=alphas)
    model.fit(X, y)

    # Selected alpha should be from our list
    assert model.alpha_ in alphas


def test_ridge_cv_multi_target():
    """Ridge CV should work with multiple targets"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    model = Ridge(alpha='auto', cv=3)
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

    model1 = Ridge(alpha='auto', cv=3, random_state=42)
    model1.fit(X, y)

    model2 = Ridge(alpha='auto', cv=3, random_state=42)
    model2.fit(X, y)

    assert model1.alpha_ == model2.alpha_
    np.testing.assert_allclose(model1.coef_, model2.coef_, rtol=1e-5)
```

**TDD Workflow:**
```bash
# Run CV tests
uv run pytest nltools/tests/core/test_models.py -k "ridge_cv" -xvs 2>&1 | tee test_ridge_cv.log

# Extend Ridge class to support CV

# Verify
uv run pytest nltools/tests/core/test_models.py -k "ridge" -xvs
```

**Implementation:** Extend Ridge to support CV:

```python
# Update nltools/models/ridge.py

from ..algorithms.ridge import ridge_svd, ridge_cv

class Ridge(BaseModel):
    """
    Ridge regression with optional GPU acceleration.

    Parameters
    ----------
    alpha : float or 'auto', default=1.0
        Regularization strength. If 'auto', uses cross-validation
        to select optimal alpha from alphas parameter.
    cv : int or None, default=None
        Number of cross-validation folds (only used if alpha='auto')
    alphas : array-like or None, default=None
        Alpha values to try during cross-validation
    backend : str or Backend, default='numpy'
        Computational backend
    random_state : int or None, default=None
        Random seed for reproducibility

    Attributes
    ----------
    coef_ : ndarray
        Ridge coefficients
    alpha_ : float
        Alpha value used (selected via CV if alpha='auto')
    cv_scores_ : ndarray
        Cross-validation scores (only if alpha='auto')
    """

    def __init__(self, alpha=1.0, cv=None, alphas=None, backend='numpy', random_state=None):
        super().__init__()
        self.alpha = alpha
        self.cv = cv
        self.alphas = alphas
        self.backend = backend
        self.random_state = random_state

    def fit(self, X, y):
        """Fit ridge regression model."""
        X, y = self._validate_X_y(X, y)

        # Set up backend
        if isinstance(self.backend, str):
            self.backend_ = Backend(self.backend)
        else:
            self.backend_ = self.backend

        # Use CV if alpha='auto'
        if self.alpha == 'auto':
            if self.cv is None:
                raise ValueError("cv must be specified when alpha='auto'")

            result = ridge_cv(
                X, y,
                alphas=self.alphas,
                cv=self.cv,
                backend=self.backend_,
                random_state=self.random_state
            )

            self.alpha_ = result['alpha']
            self.coef_ = result['coef']
            self.cv_scores_ = result['cv_scores']
        else:
            # Fixed alpha
            self.alpha_ = self.alpha
            self.coef_ = ridge_svd(X, y, alpha=self.alpha_, backend=self.backend_)

        super().fit(X, y)
        return self
```

---

### Cycle 2.3: Ridge Backend Integration & GPU Support

**Write Tests:**

```python
# Add to nltools/tests/core/test_models.py

# ============================================================================
# Ridge Model - Backend Integration
# ============================================================================

def test_ridge_numpy_backend():
    """Ridge should work with NumPy backend"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha=1.0, backend='numpy')
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == (100,)
    assert model.backend_.name == 'numpy'


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_torch_backend():
    """Ridge should work with PyTorch backend"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha=1.0, backend='torch')
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == (100,)
    assert model.backend_.name.startswith('torch')


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_cpu_gpu_equivalence():
    """Ridge should give same results on CPU and GPU"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # CPU
    model_cpu = Ridge(alpha=1.0, backend='numpy')
    model_cpu.fit(X, y)
    pred_cpu = model_cpu.predict(X)

    # GPU
    model_gpu = Ridge(alpha=1.0, backend='torch')
    model_gpu.fit(X, y)
    pred_gpu = model_gpu.predict(X)

    np.testing.assert_allclose(pred_gpu, pred_cpu, rtol=1e-4)


def test_ridge_auto_backend():
    """Ridge should select backend automatically"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = Ridge(alpha=1.0, backend='auto')
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == (100,)
    assert hasattr(model, 'backend_')
    assert model.backend_.name in ['numpy', 'torch-cpu', 'torch-cuda', 'torch-mps']


# ============================================================================
# Helper Functions
# ============================================================================

def _torch_available():
    """Check if PyTorch is installed"""
    try:
        import torch
        return True
    except ImportError:
        return False
```

**TDD Workflow:**
```bash
# Run backend tests
uv run pytest nltools/tests/core/test_models.py -k "backend" -xvs

# Verify all Ridge tests pass
uv run pytest nltools/tests/core/test_models.py -k "ridge" -xvs --tb=long 2>&1 | tee test_ridge_full.log
```

---

### Phase 2 Completion Checklist

```bash
# Run all Ridge tests
uv run pytest nltools/tests/core/test_models.py -k "ridge" -xvs

# Verify no regressions
uv run pytest nltools/tests/ -x

# Count new tests
uv run pytest nltools/tests/core/test_models.py --collect-only
```

**Expected outcome:**
- [ ] ~20 Ridge tests passing
- [ ] No regressions (91 + ~10 BaseModel tests still pass)
- [ ] Ridge model complete (~150 lines)
- [ ] Full sklearn API compatibility
- [ ] GPU support via backend abstraction

---

## Integration & Documentation

### Step 1: Update models module exports

```python
# nltools/models/__init__.py
"""
Model classes for neuroimaging analysis.

Provides sklearn-compatible APIs for common neuroimaging analyses.
"""

from .base import BaseModel
from .ridge import Ridge

__all__ = ['BaseModel', 'Ridge']
```

### Step 2: Update documentation

**Files to update:**
1. **`model-spec.md`** - Mark Sprint 2 progress (BaseModel + Ridge complete)
2. **`REFACTORING_PLAN.md`** - Update progress tracker
3. **`MIGRATION_v0.5_to_v0.6.md`** - Document new Model classes

**Add to MIGRATION guide:**
```markdown
## New Feature: Model Classes

nltools v0.6.0 introduces sklearn-compatible model classes:

### Ridge Regression

**Basic usage:**
```python
from nltools.models import Ridge
import numpy as np

X = np.random.randn(100, 50)
y = np.random.randn(100)

# Fixed alpha
model = Ridge(alpha=1.0)
model.fit(X, y)
y_pred = model.predict(X)
score = model.score(X, y)

# Automatic alpha selection via CV
model = Ridge(alpha='auto', cv=5)
model.fit(X, y)
print(f"Selected alpha: {model.alpha_}")

# GPU acceleration
model = Ridge(alpha=1.0, backend='torch')
model.fit(X, y)
```

**Migration from deprecated methods:**
```python
# OLD (v0.5.1)
results = brain_data.predict(algorithm='ridge', cv_dict={'type': 'kfolds', 'n_folds': 5})

# NEW (v0.6.0)
from nltools.models import Ridge
model = Ridge(alpha='auto', cv=5)
model.fit(brain_data.data, target_values)
predictions = model.predict(brain_data.data)
```
```

### Step 3: Final verification

```bash
# Run full test suite
uv run pytest nltools/tests/ -x --tb=short 2>&1 | tee test_final_sprint2.log

# Check test count (should be ~121 tests)
# 91 existing + 10 BaseModel + 20 Ridge = 121
uv run pytest --collect-only | grep "test session"

# Verify imports work
uv run python -c "from nltools.models import BaseModel, Ridge; print('✓ Imports successful')"

# Check no regressions on existing tests
uv run pytest nltools/tests/shell/test_brain_data.py -xvs
```

---

## Summary Checklist

### Phase 1: BaseModel ✅
- [ ] test_models.py created in core/
- [ ] Abstract interface tests (5 tests)
- [ ] Shared functionality tests (3 tests)
- [ ] Input validation tests (3 tests)
- [ ] base.py implementation (~100 lines)
- [ ] All tests passing
- [ ] No regressions

### Phase 2: Ridge Model ✅
- [ ] Basic fit/predict tests (9 tests)
- [ ] Cross-validation tests (5 tests)
- [ ] Backend integration tests (4 tests)
- [ ] ridge.py implementation (~150 lines)
- [ ] All tests passing
- [ ] No regressions

### Integration ✅
- [ ] models/__init__.py exports
- [ ] model-spec.md updated
- [ ] REFACTORING_PLAN.md updated
- [ ] MIGRATION_v0.5_to_v0.6.md updated
- [ ] All imports work
- [ ] ~121 total tests passing

### Final Actions
- [ ] Stage all changes: `git add .`
- [ ] Review: `git status` + `git diff --staged`
- [ ] Say: "Sprint 2 (BaseModel + Ridge) complete - changes staged and ready for review"
- [ ] **WAIT FOR APPROVAL** before committing

---

## Estimated Timeline

- **Phase 1 (BaseModel)**: 3-4 hours
  - Cycle 1.1: 1.5 hours (abstract interface)
  - Cycle 1.2: 1.5 hours (validation)

- **Phase 2 (Ridge)**: 4-6 hours
  - Cycle 2.1: 2 hours (basic fit/predict)
  - Cycle 2.2: 2 hours (cross-validation)
  - Cycle 2.3: 1 hour (backend integration)

- **Integration**: 1-2 hours
  - Documentation updates
  - Final verification

**Total: 1-2 days**

---

## Token Efficiency Tips

✅ **Always capture pytest output to log files FIRST**
✅ **Use Read/Grep tools on logs instead of re-running tests**
✅ **Run targeted tests during development** (`-k pattern`)
✅ **Only run full suite at checkpoints**

**Token savings:**
- Each pytest run: 1,000-5,000 tokens
- Each Grep search: ~50 tokens
- Searching 5 patterns: 25,000 tokens (re-running) vs 5,250 tokens (using logs) = **80% savings**

---

## Key Success Factors

1. **Leverage existing algorithms**: Ridge wraps `ridge.py` (already tested)
2. **sklearn API compliance**: Follow fit/predict/score pattern exactly
3. **Clear separation**: BaseModel = interface, Ridge = implementation
4. **Test organization**: Function-based tests in core/
5. **Backend abstraction**: Use existing Backend class (no reimplementation)
6. **No regressions**: 91 existing tests must continue passing
7. **Documentation**: Update migration guide with examples

---

## Future Expansion Path

**After this sprint is complete**, the pattern is established for:
- `GLMModel(BaseModel)` - Wraps nilearn FirstLevelModel/SecondLevelModel
- `InferenceModel(BaseModel)` - T-tests, permutation testing
- `SearchlightModel(BaseModel)` - Searchlight/multi-ROI analysis

Each follows the same TDD pattern:
1. Write tests in `test_models.py`
2. Implement in `nltools/models/{name}.py`
3. Export from `nltools/models/__init__.py`
4. Update documentation

---

## Progress Log

### 2025-10-28: Sprint 2 Plan Created

**What was planned:**
- ✅ Comprehensive TDD plan for BaseModel + Ridge
- ✅ Aligned with existing test suite organization
- ✅ Following model-spec-log.md pattern
- ✅ Focused scope (BaseModel + Ridge only)

**Rationale for focused scope:**
- Establish solid architecture foundation first
- Get sklearn API patterns right
- GLMModel and InferenceModel can follow this proven pattern
- Enables incremental delivery and earlier testing

**Next steps:**
1. Review plan with Eshin
2. Begin implementation with Cycle 1.1
3. Follow TDD workflow strictly
4. Commit only after approval
