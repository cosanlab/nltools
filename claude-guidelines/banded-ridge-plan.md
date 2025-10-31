# TDD Plan: Banded Ridge Regression for nltools

*Created: 2025-10-29*
*Status: Ready for implementation*

---

## Quick Summary

Implement **banded ridge regression** - an extension of ridge regression that learns **different regularization strengths per feature group**. This is critical for neuroimaging where different feature spaces (e.g., visual, semantic, motion) need different regularization levels. Implementation follows our existing Ridge pattern: functional core + imperative shell.

---

## Architecture Approach

### Pattern: Same as Ridge (proven successful)

**Functional Core** → `nltools/algorithms/banded_ridge.py`:
```python
banded_ridge_svd(Xs, y, deltas, alpha, backend)  # Core solver
banded_ridge_cv(Xs, y, n_iter, concentration, alphas, cv, backend)  # CV wrapper
_generate_dirichlet_samples(...)  # Helper for random search
```

**Imperative Shell** → `nltools/models/banded_ridge.py`:
```python
class BandedRidge(BaseModel):
    def __init__(self, deltas='auto', n_iter=100, concentration=[0.1, 1.0], ...)
    def fit(self, Xs, y)  # Xs is LIST of feature matrices
    def predict(self, Xs)
    def score(self, Xs, y)
```

**Test Suite** → Following test reorganization:
- `nltools/tests/core/test_banded_ridge.py`: Functional core tests (algorithm correctness)
- `nltools/tests/shell/test_models.py`: Add BandedRidge class tests

### Key Mathematical Concept

**Standard ridge**: `min ||X@b - Y||² + α||b||²` (single α)

**Banded ridge**: `min ||Z@b - Y||² + ||b||²` where `Z_i = exp(δ_i/2) * X_i`

The innovation: **reformulate grouped regularization as feature scaling** + standard ridge!
- Each feature group gets scaling factor `exp(δ_i/2)`
- The `δ` parameters encode relative regularization strengths
- When all δ_i are equal → standard ridge regression

### Hyperparameter Search Strategy

Instead of k-dimensional grid search over (α₁, α₂, ..., αₖ):
1. Sample **γ = (γ₁, ..., γₖ)** from Dirichlet distribution (ensures Σγᵢ = 1)
2. For candidate α values, compute **δᵢ = log(γᵢ/α)**
3. Cross-validate and select best (γ, α) combination
4. This explores the **simplex** efficiently (published NeuroImage 2022)

### Reference Implementation

Inspired by himalaya library (BSD-3-Clause):
- **Repository**: https://github.com/gallantlab/himalaya
- **Paper**: Dupré La Tour et al. (2022). "Feature-space selection with banded ridge regression." NeuroImage.
- **Key files reviewed**:
  - `/tmp/himalaya/himalaya/ridge/_random_search.py`: `solve_group_ridge_random_search()`
  - `/tmp/himalaya/himalaya/ridge/_sklearn_api.py`: `GroupRidgeCV` class

---

## Confirmed Design Decisions

### 1. API Design: List of arrays ✅

**Confirmed**: `fit(Xs, y)` where `Xs` is list of feature matrices

```python
# Feature groups as separate matrices
visual_features = X[:, :100]
semantic_features = X[:, 100:200]
model = BandedRidge()
model.fit([visual_features, semantic_features], y)
```

**Why this works**:
- More explicit about feature grouping
- Matches himalaya API (easier to validate against)
- Natural for neuroimaging (different feature extractors → different matrices)
- Simpler implementation (no slicing logic)

### 2. Backend Support: NumPy + PyTorch ✅

**Confirmed**: Implement both backends from the start, just like Ridge

- Use `Backend` class for abstraction
- Support `backend='numpy'`, `backend='torch'`, `backend='auto'`
- Follow exact same pattern as `ridge_svd()` and `ridge_cv()`

### 3. Dependency: scipy for Dirichlet sampling ✅

**Confirmed**: Use `scipy.stats.dirichlet`
- Already in dependencies via nilearn
- Battle-tested implementation
- Can add custom GPU sampler later if needed

### 4. Naming: BandedRidge ✅

**Confirmed**: Class name is `BandedRidge` (no GroupRidge alias)

```python
from nltools.models import BandedRidge
```

Documentation will mention "group ridge" terminology for searchability.

---

## TDD Implementation Plan

### Phase 1: Functional Core (`algorithms/banded_ridge.py`)

#### Test Suite Structure (`tests/core/test_banded_ridge.py`):

**Group 1: Dirichlet Sampling Helpers** (3 tests)

```python
def test_dirichlet_samples_shape():
    """Should generate samples on simplex with correct shape"""
    # Generate 100 samples for 3 feature spaces
    samples = _generate_dirichlet_samples(n_samples=100, n_spaces=3)
    assert samples.shape == (100, 3)
    # All rows should sum to 1 (on simplex)
    np.testing.assert_allclose(samples.sum(axis=1), 1.0, rtol=1e-5)

def test_dirichlet_concentration_effect():
    """Higher concentration should concentrate mass more"""
    # Sample with concentration=0.1 vs 10.0
    samples_low = _generate_dirichlet_samples(100, 3, concentration=0.1)
    samples_high = _generate_dirichlet_samples(100, 3, concentration=10.0)
    # Lower concentration → higher variance
    assert samples_low.var() > samples_high.var()

def test_dirichlet_uniform_initialization():
    """First sample should be uniform (1/n_spaces)"""
    # Per himalaya: gammas[0] = 1/n_spaces
    # Ensures uniform weighting is always tested
    samples = _generate_dirichlet_samples(100, 3)
    expected = np.array([1/3, 1/3, 1/3])
    np.testing.assert_allclose(samples[0], expected, rtol=1e-5)
```

**Group 2: Banded Ridge SVD Solver** (5 tests)

```python
def test_banded_ridge_single_group_equals_ridge():
    """Single group should match standard ridge"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # Banded ridge with single group
    Xs = [X]
    deltas = np.array([0.0])
    coef_banded = banded_ridge_svd(Xs, y, deltas, alpha, backend='numpy')

    # Standard ridge
    from nltools.algorithms.ridge import ridge_svd
    coef_ridge = ridge_svd(X, y, alpha, backend='numpy')

    np.testing.assert_allclose(coef_banded, coef_ridge, rtol=1e-5)

def test_banded_ridge_uniform_deltas_equals_ridge():
    """Uniform deltas should match standard ridge on concatenated features"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # Banded ridge with uniform deltas
    Xs = [X1, X2]
    deltas = np.array([0.0, 0.0])  # exp(0) = 1, so no scaling
    coef_banded = banded_ridge_svd(Xs, y, deltas, alpha, backend='numpy')

    # Standard ridge on concatenated features
    from nltools.algorithms.ridge import ridge_svd
    X_concat = np.concatenate([X1, X2], axis=1)
    coef_ridge = ridge_svd(X_concat, y, alpha, backend='numpy')

    np.testing.assert_allclose(coef_banded, coef_ridge, rtol=1e-5)

def test_banded_ridge_multi_group():
    """Should handle multiple feature groups correctly"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    X3 = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    Xs = [X1, X2, X3]
    deltas = np.array([0.5, -0.3, 0.2])
    alpha = 1.0

    coef = banded_ridge_svd(Xs, y, deltas, alpha, backend='numpy')

    # Check coefficient shape
    assert coef.shape == (60,)  # 30 + 20 + 10

    # Check that solution is not all zeros or NaN
    assert not np.any(np.isnan(coef))
    assert np.any(coef != 0)

def test_banded_ridge_delta_scaling_effect():
    """Larger delta should increase feature space weight"""
    np.random.seed(42)
    n_samples = 100

    # Create two groups with known signal
    X1 = np.random.randn(n_samples, 20).astype(np.float32)
    X2 = np.random.randn(n_samples, 20).astype(np.float32)

    # Generate y from X1 only
    beta_true = np.random.randn(20).astype(np.float32)
    y = X1 @ beta_true + 0.1 * np.random.randn(n_samples).astype(np.float32)

    # Give X1 higher weight (larger delta)
    deltas_favoring_X1 = np.array([1.0, -1.0])
    coef = banded_ridge_svd([X1, X2], y, deltas_favoring_X1, alpha=1.0)

    # Coefficients for X1 should have larger magnitude
    norm_X1 = np.linalg.norm(coef[:20])
    norm_X2 = np.linalg.norm(coef[20:])
    assert norm_X1 > norm_X2

def test_banded_ridge_input_validation():
    """Should validate inputs properly"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # Xs must be list
    with pytest.raises((TypeError, ValueError)):
        banded_ridge_svd(X, y, deltas=[0.0], alpha=1.0)

    # All Xs must have same n_samples
    X1 = np.random.randn(100, 20).astype(np.float32)
    X2 = np.random.randn(90, 30).astype(np.float32)  # Wrong size
    with pytest.raises(ValueError):
        banded_ridge_svd([X1, X2], y, deltas=[0.0, 0.0], alpha=1.0)

    # deltas shape must match len(Xs)
    X1 = np.random.randn(100, 20).astype(np.float32)
    X2 = np.random.randn(100, 30).astype(np.float32)
    with pytest.raises(ValueError):
        banded_ridge_svd([X1, X2], y, deltas=[0.0], alpha=1.0)  # Only 1 delta

    # alpha must be positive
    with pytest.raises(ValueError):
        banded_ridge_svd([X1, X2], y, deltas=[0.0, 0.0], alpha=-1.0)
```

**Group 3: Cross-Validation** (6 tests)

```python
def test_banded_ridge_cv_basic():
    """Should select deltas and return results"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = banded_ridge_cv(
        [X1, X2], y,
        n_iter=10,
        alphas=[0.1, 1.0, 10.0],
        cv=3,
        backend='numpy'
    )

    # Verify result keys
    assert 'deltas' in result
    assert 'coef' in result
    assert 'cv_scores' in result
    assert 'alpha' in result
    assert 'backend' in result

    # Check shapes
    assert result['deltas'].shape == (2,)  # 2 groups, single target
    assert result['coef'].shape == (50,)  # 30 + 20 features
    assert result['cv_scores'].shape == (10,)  # n_iter

    # Check selected alpha
    assert result['alpha'] in [0.1, 1.0, 10.0]

def test_banded_ridge_cv_multi_target():
    """Should handle multiple targets"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    result = banded_ridge_cv([X1, X2], Y, n_iter=10, cv=3, backend='numpy')

    # deltas should be per-target
    assert result['deltas'].shape == (2, 5)  # (n_groups, n_targets)

    # coef should be multi-target
    assert result['coef'].shape == (50, 5)  # (n_features, n_targets)

    # cv_scores per target
    assert result['cv_scores'].shape == (10, 5)  # (n_iter, n_targets)

def test_banded_ridge_cv_default_parameters():
    """Should work with default concentration and alphas"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # Don't specify concentration or alphas
    result = banded_ridge_cv([X1, X2], y, n_iter=10, cv=3, backend='numpy')

    # Should complete successfully
    assert result['alpha'] > 0
    assert result['coef'].shape == (50,)

def test_banded_ridge_cv_reproducibility():
    """Same random_state should give same results"""
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result1 = banded_ridge_cv(
        [X1, X2], y, n_iter=10, cv=3,
        random_state=42, backend='numpy'
    )

    result2 = banded_ridge_cv(
        [X1, X2], y, n_iter=10, cv=3,
        random_state=42, backend='numpy'
    )

    assert result1['alpha'] == result2['alpha']
    np.testing.assert_allclose(result1['deltas'], result2['deltas'], rtol=1e-5)
    np.testing.assert_allclose(result1['coef'], result2['coef'], rtol=1e-5)

def test_banded_ridge_cv_single_alpha():
    """Should work with single alpha value"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # alphas=1.0 (not a list)
    result = banded_ridge_cv(
        [X1, X2], y, n_iter=10, alphas=1.0, cv=3, backend='numpy'
    )

    # Should still search over gammas
    assert result['alpha'] == 1.0
    assert result['coef'].shape == (50,)

def test_banded_ridge_cv_score_shapes():
    """CV scores should have correct shape"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    Y = np.random.randn(100, 3).astype(np.float32)

    n_iter = 20
    result = banded_ridge_cv([X1, X2], Y, n_iter=n_iter, cv=5, backend='numpy')

    # cv_scores: (n_iter, n_targets)
    assert result['cv_scores'].shape == (n_iter, 3)
```

**Group 4: Backend Tests** (3 tests)

```python
def test_banded_ridge_numpy_backend():
    """Should work with explicit numpy backend"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    from nltools.backends import Backend
    backend = Backend('numpy')

    deltas = np.array([0.5, -0.5])
    coef = banded_ridge_svd([X1, X2], y, deltas, alpha=1.0, backend=backend)

    assert coef.shape == (50,)
    assert isinstance(coef, np.ndarray)

@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_banded_ridge_torch_backend():
    """Should work with PyTorch backend"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    from nltools.backends import Backend
    backend = Backend('torch')

    deltas = np.array([0.5, -0.5])
    coef = banded_ridge_svd([X1, X2], y, deltas, alpha=1.0, backend=backend)

    assert coef.shape == (50,)
    assert isinstance(coef, np.ndarray)  # Should return numpy array

@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_banded_ridge_cpu_gpu_equivalence():
    """CPU and GPU backends should give same results"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    deltas = np.array([0.5, -0.5])
    alpha = 1.0

    # CPU
    from nltools.backends import Backend
    backend_cpu = Backend('numpy')
    coef_cpu = banded_ridge_svd([X1, X2], y, deltas, alpha, backend=backend_cpu)

    # GPU
    backend_gpu = Backend('torch')
    coef_gpu = banded_ridge_svd([X1, X2], y, deltas, alpha, backend=backend_gpu)

    np.testing.assert_allclose(coef_gpu, coef_cpu, rtol=1e-4)
```

**Group 5: Comparison Tests** (3 tests)

```python
@pytest.mark.skipif(not _himalaya_available(), reason="himalaya not installed")
def test_vs_himalaya_basic():
    """Should match himalaya on simple problem"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # Our implementation
    result_ours = banded_ridge_cv(
        [X1, X2], y, n_iter=20, cv=3,
        random_state=42, backend='numpy'
    )

    # Himalaya
    from himalaya.ridge import solve_group_ridge_random_search
    result_himalaya = solve_group_ridge_random_search(
        [X1, X2], y[:, np.newaxis], n_iter=20, cv=3,
        random_state=42, return_weights=True
    )

    # Should be very close (different CV folds might cause small differences)
    np.testing.assert_allclose(
        result_ours['coef'],
        result_himalaya[1].squeeze(),
        rtol=1e-3
    )

def test_vs_sklearn_single_group():
    """Single group should match sklearn Ridge"""
    from sklearn.linear_model import Ridge

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # Our implementation
    coef_ours = banded_ridge_svd([X], y, deltas=[0.0], alpha=alpha)

    # sklearn
    ridge_sklearn = Ridge(alpha=alpha, fit_intercept=False, solver='svd')
    ridge_sklearn.fit(X, y)
    coef_sklearn = ridge_sklearn.coef_

    np.testing.assert_allclose(coef_ours, coef_sklearn, rtol=1e-4)

def test_perfect_vs_noisy_data():
    """Should adapt regularization to noise level"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    beta_true = np.random.randn(50).astype(np.float32)
    X = np.concatenate([X1, X2], axis=1)

    # Perfect fit
    y_perfect = X @ beta_true
    result_perfect = banded_ridge_cv(
        [X1, X2], y_perfect, n_iter=20, cv=3, backend='numpy'
    )

    # Noisy fit
    y_noisy = X @ beta_true + 0.5 * np.random.randn(100).astype(np.float32)
    result_noisy = banded_ridge_cv(
        [X1, X2], y_noisy, n_iter=20, cv=3, backend='numpy'
    )

    # Perfect fit should prefer smaller alpha (less regularization)
    assert result_perfect['alpha'] <= result_noisy['alpha']
```

**Group 6: Edge Cases** (3 tests)

```python
def test_many_groups_handling():
    """Should handle many feature groups (10+)"""
    np.random.seed(42)
    n_groups = 10
    Xs = [np.random.randn(100, 10).astype(np.float32) for _ in range(n_groups)]
    y = np.random.randn(100).astype(np.float32)

    result = banded_ridge_cv(Xs, y, n_iter=20, cv=3, backend='numpy')

    # Should complete without error
    assert result['deltas'].shape == (n_groups,)
    assert result['coef'].shape == (100,)  # 10 groups * 10 features

def test_unbalanced_groups():
    """Should handle groups with very different sizes"""
    np.random.seed(42)
    X1 = np.random.randn(100, 10).astype(np.float32)
    X2 = np.random.randn(100, 1000).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = banded_ridge_cv([X1, X2], y, n_iter=10, cv=3, backend='numpy')

    # Should not crash or produce NaNs
    assert not np.any(np.isnan(result['coef']))
    assert not np.any(np.isnan(result['deltas']))

def test_minimal_samples():
    """Should handle n_samples < n_features"""
    np.random.seed(42)
    X1 = np.random.randn(50, 30).astype(np.float32)
    X2 = np.random.randn(50, 40).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)

    # Should work (SVD handles underdetermined systems)
    result = banded_ridge_cv([X1, X2], y, n_iter=10, cv=3, backend='numpy')

    assert result['coef'].shape == (70,)
```

**Total Phase 1 Tests**: ~23 tests

---

### Phase 2: Imperative Shell (`models/banded_ridge.py`)

#### Test Suite Structure (`tests/shell/test_models.py` - add to existing):

**Group 7: BandedRidge Class API** (5 tests)

```python
def test_banded_ridge_class_basic():
    """BandedRidge should fit and predict"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = BandedRidge(n_iter=10, cv=3)
    model.fit([X1, X2], y)

    # Should be fitted
    assert model.is_fitted_

    # Should have coefficients
    assert hasattr(model, 'coef_')
    assert model.coef_.shape == (50,)

    # Should predict
    y_pred = model.predict([X1, X2])
    assert y_pred.shape == (100,)

def test_banded_ridge_auto_cv():
    """deltas='auto' should trigger CV"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = BandedRidge(deltas='auto', n_iter=10, cv=3)
    model.fit([X1, X2], y)

    # Should store selected deltas
    assert hasattr(model, 'deltas_')
    assert model.deltas_.shape == (2,)

    # Should store CV scores
    assert hasattr(model, 'cv_scores_')

def test_banded_ridge_fixed_deltas():
    """Should accept fixed deltas"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    fixed_deltas = np.array([0.5, -0.5])
    model = BandedRidge(deltas=fixed_deltas, alpha=1.0)
    model.fit([X1, X2], y)

    # Should use provided deltas
    assert hasattr(model, 'deltas_')
    np.testing.assert_array_equal(model.deltas_, fixed_deltas)

    # Should not have CV scores
    assert not hasattr(model, 'cv_scores_')

def test_banded_ridge_score():
    """score() should compute R²"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    beta_true = np.random.randn(50).astype(np.float32)
    X = np.concatenate([X1, X2], axis=1)
    y = X @ beta_true + 0.1 * np.random.randn(100).astype(np.float32)

    model = BandedRidge(n_iter=20, cv=3)
    model.fit([X1, X2], y)

    score = model.score([X1, X2], y)

    # Should have reasonable R² (close to 1 for low noise)
    assert 0.7 < score < 1.0

def test_banded_ridge_repr():
    """Should have informative string representation"""
    model = BandedRidge(deltas='auto', n_iter=50, backend='numpy')

    repr_str = repr(model)
    assert 'BandedRidge' in repr_str
    assert 'deltas=' in repr_str or 'auto' in repr_str
```

**Group 8: Multi-target Support** (2 tests)

```python
def test_banded_ridge_multi_target_fit():
    """Should fit multi-target regression"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    model = BandedRidge(n_iter=10, cv=3)
    model.fit([X1, X2], Y)

    # Coefficients should be (n_features, n_targets)
    assert model.coef_.shape == (50, 5)

    # Deltas should be per-target
    assert model.deltas_.shape == (2, 5)

def test_banded_ridge_multi_target_predict():
    """Should predict multi-target regression"""
    np.random.seed(42)
    X1 = np.random.randn(100, 30).astype(np.float32)
    X2 = np.random.randn(100, 20).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    model = BandedRidge(n_iter=10, cv=3)
    model.fit([X1, X2], Y)

    Y_pred = model.predict([X1, X2])
    assert Y_pred.shape == (100, 5)
```

**Total Phase 2 Tests**: ~7 tests

**Grand Total**: ~30 tests

---

### Phase 3: Documentation & Integration

**Files to Create/Modify**:

1. **New Files**:
   - `nltools/algorithms/banded_ridge.py` (~300 lines)
   - `nltools/tests/core/test_banded_ridge.py` (~500 lines)
   - `nltools/models/banded_ridge.py` (~200 lines)

2. **Modified Files**:
   - `nltools/models/__init__.py`: Add `from .banded_ridge import BandedRidge`
   - `nltools/algorithms/__init__.py`: Add banded ridge exports
   - `docs/migration-guide.md`: Document new feature

3. **Documentation**:
   - Add comprehensive docstring example to `BandedRidge` class
   - Create usage example in docstrings
   - Add reference to himalaya paper

---

## Implementation Order (TDD Workflow)

### Step 1: Setup & Helpers
```bash
# Create files
touch nltools/algorithms/banded_ridge.py
touch nltools/tests/core/test_banded_ridge.py

# Write Group 1 tests (Dirichlet sampling)
# Implement _generate_dirichlet_samples()
uv run pytest nltools/tests/core/test_banded_ridge.py::test_dirichlet* -xvs
```

### Step 2: Core Solver
```bash
# Write Group 2 tests (banded_ridge_svd)
# Implement banded_ridge_svd() - build on ridge_svd pattern
uv run pytest nltools/tests/core/test_banded_ridge.py::test_banded_ridge_single* -xvs
# Fix until passing, then continue with other Group 2 tests
```

### Step 3: Cross-Validation
```bash
# Write Group 3 tests (banded_ridge_cv)
# Implement banded_ridge_cv() - build on ridge_cv pattern
uv run pytest nltools/tests/core/test_banded_ridge.py::test_banded_ridge_cv* -xvs
```

### Step 4: Backend Support
```bash
# Write Group 4 tests (backend integration)
# Ensure torch/numpy backends work correctly
uv run pytest nltools/tests/core/test_banded_ridge.py::test_banded_ridge_*backend* -xvs
```

### Step 5: Validation
```bash
# Write Group 5 tests (comparisons)
# Fix any bugs found
# May need to adjust algorithm based on himalaya comparison
uv run pytest nltools/tests/core/test_banded_ridge.py::test_vs_* -xvs
```

### Step 6: Edge Cases
```bash
# Write Group 6 tests
# Handle edge cases gracefully
uv run pytest nltools/tests/core/test_banded_ridge.py::test_*groups* -xvs
uv run pytest nltools/tests/core/test_banded_ridge.py::test_minimal* -xvs
```

### Step 7: Imperative Shell
```bash
# Create nltools/models/banded_ridge.py
# Write Group 7 & 8 tests
# Implement BandedRidge class following Ridge pattern
uv run pytest nltools/tests/shell/test_models.py::*BandedRidge* -xvs
```

### Step 8: Full Suite
```bash
# Run all banded ridge tests
uv run pytest nltools/tests/core/test_banded_ridge.py -xvs

# Run full test suite to check for regressions
uv run pytest nltools/tests/ -x
```

---

## Success Criteria

**Functional Correctness**:
- ✅ Single group case matches standard ridge
- ✅ Uniform deltas match standard ridge on concatenated features
- ✅ Matches himalaya results (if available for comparison)
- ✅ Matches sklearn Ridge when appropriate
- ✅ Both NumPy and PyTorch backends produce same results

**API Quality**:
- ✅ Sklearn-compatible API (fit/predict/score)
- ✅ Inherits from BaseModel properly
- ✅ Clear docstrings with examples
- ✅ Type hints on all public functions

**Performance**:
- ✅ Completes on neuroimaging-scale data (300 samples, 50k features, 3 groups)
- ✅ Backend abstraction works (numpy/torch)
- ✅ Auto backend selection functional

**Testing**:
- ✅ ~30 total tests covering core + shell
- ✅ All tests pass
- ✅ No regressions in existing test suite

---

## Implementation Notes

### Core Algorithm Structure

Following `nltools/algorithms/ridge.py` pattern:

```python
def banded_ridge_svd(
    Xs: list[np.ndarray],
    y: np.ndarray,
    deltas: np.ndarray,
    alpha: float = 1.0,
    backend: Optional[Union[Backend, str]] = None
) -> np.ndarray:
    """
    Solve banded ridge regression using SVD.

    Solves: min ||Z @ b - y||² + ||b||²
    where Z_i = exp(deltas[i] / 2) * X_i
    """
    # 1. Concatenate feature spaces
    # 2. Apply per-group scaling: Z_i = exp(delta_i/2) * X_i
    # 3. Use ridge_svd on scaled features
    # 4. Return coefficients
```

### Random Search Pattern

Following himalaya's approach:

```python
def banded_ridge_cv(
    Xs: list[np.ndarray],
    y: np.ndarray,
    n_iter: int = 100,
    concentration: list[float] = [0.1, 1.0],
    alphas: Optional[np.ndarray] = None,
    cv: int = 5,
    backend: Union[str, Backend] = 'auto',
    random_state: Optional[int] = None
) -> dict:
    """
    Banded ridge with cross-validation via random search.

    Samples gamma from Dirichlet, computes deltas = log(gamma/alpha),
    evaluates via CV, selects best (gamma, alpha) combination.
    """
    # 1. Generate gamma samples from Dirichlet
    # 2. For each alpha candidate:
    #    - Compute deltas = log(gamma/alpha)
    #    - Cross-validate
    # 3. Select best (gamma, alpha)
    # 4. Refit on full data
```

---

## Estimated Complexity

**Implementation Time**: ~6-8 hours
- Phase 1 (functional core): ~4 hours
- Phase 2 (imperative shell): ~2 hours
- Phase 3 (docs/integration): ~1 hour
- Testing/debugging: ~1 hour

**Complexity Factors**:
- ✅ **Low**: Algorithm well-documented in himalaya
- ✅ **Low**: Architecture pattern proven with Ridge
- ⚠️ **Medium**: Random search more complex than grid search
- ⚠️ **Medium**: Feature concatenation/scaling logic
- ✅ **Low**: Backend integration (same pattern as Ridge)

**Biggest Risks**:
1. Getting Dirichlet sampling parameters right (concentration values)
2. Properly handling feature group boundaries and scaling
3. CV score aggregation for hyperparameter selection
4. Ensuring numerical stability with extreme delta values

**Mitigation Strategy**:
- Test against himalaya library outputs directly
- Start with simple cases (2 groups, balanced sizes)
- Add extensive input validation

---

## References

**Primary Source**:
- Dupré La Tour, T., Eickenberg, M., Nunez-Elizalde, A.O., & Gallant, J. L. (2022). "Feature-space selection with banded ridge regression." *NeuroImage*, 257, 119334.
- DOI: https://doi.org/10.1016/j.neuroimage.2022.119334

**Implementation Reference**:
- himalaya library: https://github.com/gallantlab/himalaya
- BSD-3-Clause License
- Files reviewed: `ridge/_random_search.py`, `ridge/_sklearn_api.py`

**Key Concepts**:
- Dirichlet distribution for simplex sampling
- Feature space scaling: `Z_i = exp(δ_i/2) * X_i`
- Random search vs grid search efficiency
- Group regularization vs global regularization

---

## Future Enhancements (Post-v0.6.0)

**Potential additions**:
1. **Gradient-based optimization**: himalaya also implements gradient descent for hyperparameter search (faster than random search)
2. **Kernel banded ridge**: Kernel version for n_samples < n_features case
3. **Sparse group lasso**: Extension to L1 + group penalties
4. **Integration with BrainData**: Add `.banded_ridge()` method
5. **GPU optimization**: Custom Dirichlet sampler for torch backend

**Lower priority**:
- Warm start for hyperparameter search
- Parallel CV fold evaluation
- Early stopping criteria

---

*End of implementation plan*
