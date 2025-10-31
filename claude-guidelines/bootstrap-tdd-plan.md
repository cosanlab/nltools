# Bootstrap Refactor - TDD Implementation Plan

**Created**: 2025-10-31
**Status**: Active Implementation
**Branch**: `uv-cleanup`
**Version Target**: v0.6.0

---

## Executive Summary

Implement efficient, GPU-capable bootstrap utilities in `nltools/algorithms/inference/bootstrap.py` following inference module patterns. Replace naive implementations in `BrainData.bootstrap()` and `Adjacency.bootstrap()` with memory-efficient, high-performance versions.

**Key Improvements**:
- **Memory**: 1,673× reduction using online statistics (efficient mode)
- **Speed**: 10-100× faster for Ridge models (bypass BrainData overhead)
- **Features**: Confidence intervals, fitted model support (`weights`, `predict`)
- **Patterns**: Match inference module (pure functions, progress bars, Backend abstraction)

---

## Architecture Overview

### New Module Structure

```
nltools/algorithms/inference/
├── bootstrap.py (NEW)
│   ├── OnlineBootstrapStats          # Memory-efficient aggregator
│   ├── _bootstrap_cpu_parallel()     # CPU parallelization (default)
│   ├── _bootstrap_gpu_batched()      # Optional GPU support (future)
│   └── bootstrap()                   # Public API
└── utils.py (EXTEND)
    └── _generate_bootstrap_indices() # Bootstrap index generation
```

### Integration Points

**BrainData.bootstrap()** → delegates to `nltools.algorithms.inference.bootstrap()`
**Adjacency.bootstrap()** → delegates to `nltools.algorithms.inference.bootstrap()`

### Key Design Patterns

**From inference module**:
- ✅ Pure functions working with numpy arrays
- ✅ CPU parallelization via joblib (default)
- ✅ Optional GPU support via Backend abstraction
- ✅ Automatic batching for memory management
- ✅ Progress bars for user feedback
- ✅ Deterministic RNG (pre-generate seeds)

**From bootstrap refactor plan**:
- ✅ OnlineBootstrapStats class for memory efficiency
- ✅ Dual modes: Efficient (default) vs. full (save_weights=True)
- ✅ Farm out to pure functions (ridge_svd) for performance
- ✅ Support fitted model methods: `weights`, `predict`

---

## TDD Implementation Phases

### Phase 1: Infrastructure (5 tests)

**Goal**: Build `OnlineBootstrapStats` class for memory-efficient aggregation using Welford's algorithm

**File**: `nltools/algorithms/inference/bootstrap.py`

**Tests to implement** (`nltools/tests/core/test_bootstrap.py`):

1. **test_online_stats_1d_array**
   - Input: 1D array shape (n_samples,)
   - Output: mean, std, Z, p values
   - Validates: Basic aggregation works

2. **test_online_stats_2d_array**
   - Input: 2D array shape (n_samples, n_features)
   - Output: mean, std, Z, p per feature
   - Validates: Multi-dimensional aggregation

3. **test_online_stats_confidence_intervals**
   - Input: 1D/2D arrays
   - Output: ci_lower, ci_upper using normal approximation
   - Validates: CIs computed correctly (mean ± 1.96*std)

4. **test_online_stats_sample_storage**
   - Input: 1D/2D arrays with save_samples=True
   - Output: All bootstrap samples stored
   - Validates: Exact percentile CIs possible

5. **test_online_stats_numerical_stability**
   - Input: Large values (1e10), small variances
   - Output: Accurate mean/std (no catastrophic cancellation)
   - Validates: Welford's algorithm stability

**Implementation checklist**:
- [ ] Create `OnlineBootstrapStats` class
- [ ] Implement Welford's online mean/variance algorithm
- [ ] Implement `update(sample)` method
- [ ] Implement `get_results(percentiles=(2.5, 97.5))` method
- [ ] Support both efficient and full modes
- [ ] Write 5 tests
- [ ] Verify tests pass

**Expected outcome**: Reusable statistics aggregator for all bootstrap methods

---

### Phase 2: Simple Methods Backward Compatibility (4 tests)

**Goal**: Maintain existing API for simple aggregation methods

**File**: `nltools/algorithms/inference/bootstrap.py`

**Tests to implement** (`nltools/tests/core/test_bootstrap.py`):

6. **test_bootstrap_simple_methods_all**
   - Methods: mean, std, median, sum, min, max
   - Validates: All return correct dict structure
   - Validates: Results have expected shapes

7. **test_bootstrap_simple_reproducibility**
   - Run bootstrap with same random_state twice
   - Validates: Identical results both times

8. **test_bootstrap_simple_save_weights**
   - Run with save_weights=True
   - Validates: 'samples' key contains all bootstrap samples
   - Validates: Exact percentile CIs match samples

9. **test_bootstrap_simple_progress**
   - Run bootstrap with tqdm enabled
   - Validates: Progress bar appears (check output)

**Also update** (`nltools/tests/shell/test_brain_data.py`):
- Unskip existing `test_bootstrap()` (line 1370)
- Verify backward compatibility

**Implementation checklist**:
- [ ] Implement `_bootstrap_simple_method_worker(data, function, indices)`
- [ ] Implement `_bootstrap_cpu_parallel(data, function, n_samples, ...)`
- [ ] Integrate with `OnlineBootstrapStats`
- [ ] Add progress bars (tqdm)
- [ ] Write 4 new tests
- [ ] Unskip and fix existing test
- [ ] Verify all tests pass

**Expected outcome**: Simple methods work exactly as before, but more efficient

---

### Phase 3: Ridge Model Weights Bootstrap (6 tests)

**Goal**: Bootstrap fitted Ridge model weights efficiently (10-100× speedup)

**File**: `nltools/algorithms/inference/bootstrap.py`

**Tests to implement** (`nltools/tests/core/test_bootstrap.py`):

10. **test_bootstrap_weights_requires_fit**
    - Call bootstrap('weights') without fitting
    - Validates: Raises ValueError with helpful message

11. **test_bootstrap_ridge_weights_efficient**
    - Fit Ridge model, bootstrap weights (efficient mode)
    - Validates: Returns mean, std, Z, p, ci_lower, ci_upper
    - Validates: Shape matches (n_features, n_voxels)

12. **test_bootstrap_ridge_weights_full**
    - Fit Ridge model, bootstrap weights (save_weights=True)
    - Validates: 'samples' contains all weights
    - Validates: Percentile CIs match samples

13. **test_bootstrap_ridge_weights_variance**
    - Bootstrap weights from fitted model
    - Validates: std > 0 (not all samples identical)
    - Validates: Reasonable variance across samples

14. **test_bootstrap_ridge_weights_preserves_alpha**
    - Fit with alpha=10.0, bootstrap
    - Validates: All bootstrap fits use same alpha

15. **test_bootstrap_ridge_weights_memory**
    - Track memory usage: efficient vs. full mode
    - Validates: Efficient mode uses <10% of full mode

**Implementation checklist**:
- [ ] Extract Ridge hyperparameters from `_fit_params_`
- [ ] Implement `_bootstrap_ridge_weights_worker(X, y, indices, alpha, ...)`
- [ ] Farm out to `ridge_svd()` for performance
- [ ] Bypass BrainData overhead (work with numpy arrays)
- [ ] Integrate with `OnlineBootstrapStats`
- [ ] Write 6 tests
- [ ] Verify tests pass

**Expected outcome**: Ridge weights bootstrap is fast and memory-efficient

---

### Phase 4: Ridge Predict Bootstrap (5 tests)

**Goal**: Bootstrap predictions from fitted Ridge models

**File**: `nltools/algorithms/inference/bootstrap.py`

**Tests to implement** (`nltools/tests/core/test_bootstrap.py`):

16. **test_bootstrap_predict_requires_fit**
    - Call bootstrap('predict') without fitting
    - Validates: Raises ValueError with helpful message

17. **test_bootstrap_ridge_predict_efficient**
    - Fit Ridge model, bootstrap predict (efficient mode)
    - Validates: Returns mean, std, Z, p, ci_lower, ci_upper
    - Validates: Shape matches (n_samples, n_voxels)

18. **test_bootstrap_ridge_predict_full**
    - Fit Ridge model, bootstrap predict (save_weights=True)
    - Validates: 'samples' contains all predictions
    - Validates: Percentile CIs match samples

19. **test_bootstrap_ridge_predict_variance**
    - Bootstrap predictions
    - Validates: std > 0 across samples

20. **test_bootstrap_ridge_predict_correctness**
    - Compare bootstrap mean to single predict call
    - Validates: Should be very similar (within tolerance)

**Implementation checklist**:
- [ ] Implement `_bootstrap_ridge_predict_worker(X, y, X_pred, indices, alpha, ...)`
- [ ] Farm out to `ridge_svd()` + matrix multiply
- [ ] Integrate with `OnlineBootstrapStats`
- [ ] Write 5 tests
- [ ] Verify tests pass

**Expected outcome**: Ridge predict bootstrap works efficiently

---

### Phase 5: GLM Model Methods (4 tests)

**Goal**: Bootstrap GLM models (must use BrainData path, slower but necessary)

**File**: `nltools/algorithms/inference/bootstrap.py`

**Tests to implement** (`nltools/tests/core/test_bootstrap.py`):

21. **test_bootstrap_glm_weights_efficient**
    - Fit GLM model, bootstrap weights (efficient mode)
    - Validates: Works despite needing BrainData path

22. **test_bootstrap_glm_weights_full**
    - Fit GLM model, bootstrap weights (save_weights=True)
    - Validates: Percentile CIs work

23. **test_bootstrap_glm_predict_efficient**
    - Fit GLM model, bootstrap predict (efficient mode)
    - Validates: Returns correct structure

24. **test_bootstrap_glm_predict_full**
    - Fit GLM model, bootstrap predict (save_weights=True)
    - Validates: All samples stored

**Implementation checklist**:
- [ ] Detect GLM model type
- [ ] Implement `_bootstrap_glm_worker(brain_data, function, indices, fit_params)`
- [ ] Use BrainData path (nilearn requires nifti)
- [ ] Integrate with `OnlineBootstrapStats`
- [ ] Write 4 tests
- [ ] Verify tests pass

**Expected outcome**: GLM bootstrap works (slower than Ridge, but unavoidable)

---

### Phase 6: Error Handling (6 tests)

**Goal**: Robust validation and helpful error messages

**File**: Multiple (bootstrap.py, test_bootstrap.py, test_brain_data.py)

**Tests to implement**:

25. **test_bootstrap_invalid_method**
    - Call with unsupported method name
    - Validates: ValueError lists supported methods

26. **test_bootstrap_single_image_error**
    - Call bootstrap on single image
    - Validates: Informative error message

27. **test_bootstrap_custom_percentiles**
    - Call with percentiles=(5, 95) for 90% CI
    - Validates: Custom CIs computed correctly

28. **test_bootstrap_too_few_samples**
    - Call with n_samples=10 (very low)
    - Validates: Warning issued (but still works)

29. **test_bootstrap_parallel_deterministic**
    - Run CPU parallel vs. sequential with same seed
    - Validates: Results match exactly

30. **test_bootstrap_adjacency_compatibility**
    - Call Adjacency.bootstrap('mean')
    - Validates: Still works after refactor

**Implementation checklist**:
- [ ] Add input validation to bootstrap()
- [ ] Check for fitted model when needed
- [ ] Check single image case
- [ ] Add warning for low n_samples
- [ ] Verify deterministic RNG
- [ ] Test Adjacency integration
- [ ] Write 6 tests
- [ ] Verify all tests pass

**Expected outcome**: Clear errors help users debug issues

---

### Phase 7: Performance Validation (3 tests)

**Goal**: Verify performance improvements are real

**File**: `nltools/tests/core/test_bootstrap.py` (mark as tier2)

**Tests to implement**:

31. **test_bootstrap_ridge_speedup**
    - Time Ridge bootstrap vs. naive implementation
    - Validates: New implementation ≥10× faster

32. **test_bootstrap_memory_efficient**
    - Measure memory: efficient vs. full mode
    - Validates: Efficient mode uses ≤10% of full mode

33. **test_bootstrap_large_scale**
    - Bootstrap 100K voxels × 5000 samples
    - Validates: Completes without OOM

**Implementation checklist**:
- [ ] Write performance benchmark tests
- [ ] Mark as tier2 (slow tests)
- [ ] Verify speedups are achieved
- [ ] Document performance characteristics

**Expected outcome**: Performance claims validated

---

## Integration & Refactoring

### Step 8: Refactor BrainData.bootstrap()

**File**: `nltools/data/brain_data.py`

**Changes**:
```python
def bootstrap(self, function, n_samples=5000, save_weights=False,
              n_jobs=-1, random_state=None, percentiles=(2.5, 97.5), **kwargs):
    """Bootstrap a BrainData method.

    Delegates to nltools.algorithms.inference.bootstrap() for efficient
    implementation with memory optimization.
    """
    from nltools.algorithms.inference import bootstrap as bootstrap_inference

    # Call new implementation
    result = bootstrap_inference(
        data=self,
        function=function,
        n_samples=n_samples,
        save_weights=save_weights,
        n_jobs=n_jobs,
        random_state=random_state,
        percentiles=percentiles,
        **kwargs
    )

    # Wrap numpy arrays back into BrainData
    return _wrap_bootstrap_results(result, mask=self.mask)
```

**Checklist**:
- [ ] Update BrainData.bootstrap() to delegate
- [ ] Remove old implementation (keep for reference)
- [ ] Run BrainData bootstrap tests
- [ ] Verify backward compatibility

---

### Step 9: Refactor Adjacency.bootstrap()

**File**: `nltools/data/adjacency.py`

**Changes**: Similar to BrainData, wrap results in Adjacency

**Checklist**:
- [ ] Update Adjacency.bootstrap() to delegate
- [ ] Run Adjacency bootstrap tests
- [ ] Verify backward compatibility

---

### Step 10: Update Documentation

**Files**:
- `docs/migration-guide.md`
- `nltools/algorithms/inference/bootstrap.py` (docstrings)

**Add to migration guide**:
```markdown
### Bootstrap Improvements (v0.6.0)

**New Features**:
- Confidence intervals: `result['ci_lower']`, `result['ci_upper']`
- Fitted model support: `.bootstrap('weights')`, `.bootstrap('predict')`
- Custom percentiles: `percentiles=(5, 95)` for 90% CI
- Memory efficiency: 10-100× reduction (efficient mode)
- Performance: 10-100× faster for Ridge models

**Backward Compatible**:
- Existing code using `.bootstrap('mean')` works unchanged
- All existing dict keys preserved: 'Z', 'p', 'mean', 'samples'

**Example**:
```python
# Fit a model
brain.fit(model='ridge', alpha=1.0, X=features, y=outcome)

# Bootstrap the weights
result = brain.bootstrap('weights', n_samples=5000)
# Returns: mean, std, Z, p, ci_lower, ci_upper

# Bootstrap predictions
result = brain.bootstrap('predict', n_samples=5000)
```
```

**Checklist**:
- [ ] Update migration guide
- [ ] Write comprehensive docstrings
- [ ] Add usage examples

---

## Testing Strategy

### Test Organization

**File structure**:
```
nltools/tests/
├── core/
│   └── test_bootstrap.py (NEW)  # Pure function tests (30 tests)
├── shell/
│   ├── test_brain_data.py       # Integration tests (update existing)
│   └── test_adjacency.py        # Integration tests (update existing)
```

**Test tiers**:
- **Tier 1**: Tests 1-30 (fast, <2 min total with -n auto)
- **Tier 2**: Tests 31-33 (performance, ~1 min)

**Running tests**:
```bash
# During development (targeted)
uv run pytest nltools/tests/core/test_bootstrap.py::test_name -xvs

# Phase verification (parallel)
uv run pytest nltools/tests/core/test_bootstrap.py -n auto -x

# Full regression check
uv run pytest -m tier1 -n auto

# Performance validation (ask permission first!)
uv run pytest -m tier2 -xvs --tb=long
```

---

## Success Criteria

- ✅ All 33 tests pass (tier1: 30 tests <2min, tier2: 3 perf tests ~1min)
- ✅ Existing code using `.bootstrap('mean')` works unchanged
- ✅ Memory reduction verified (>100× for large problems)
- ✅ Ridge speedup verified (>10× for typical problems)
- ✅ Backward compatible API (existing dict keys preserved)
- ✅ Migration guide updated with new features
- ✅ No regressions in existing bootstrap tests

---

## Timeline Estimate

- **Phase 1**: ~2 hours (infrastructure + 5 tests)
- **Phase 2**: ~1.5 hours (simple methods + 4 tests)
- **Phase 3**: ~2 hours (Ridge weights + 6 tests)
- **Phase 4**: ~1.5 hours (Ridge predict + 5 tests)
- **Phase 5**: ~1 hour (GLM + 4 tests)
- **Phase 6**: ~1.5 hours (error handling + 6 tests)
- **Phase 7**: ~1 hour (performance + 3 tests)
- **Integration**: ~1 hour (refactor BrainData/Adjacency)
- **Documentation**: ~0.5 hours

**Total**: ~12 hours of focused implementation

---

## Implementation Notes

### Deterministic RNG Pattern

Follow inference module pattern for reproducibility:

```python
from sklearn.utils import check_random_state

def _generate_bootstrap_indices(n_samples, n_bootstrap, random_state=None):
    """Generate bootstrap indices deterministically."""
    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_bootstrap)

    # Each bootstrap gets independent RandomState
    indices = np.array([
        np.random.RandomState(seeds[i]).choice(n_samples, n_samples, replace=True)
        for i in range(n_bootstrap)
    ])

    return indices  # Shape: (n_bootstrap, n_samples)
```

### Welford's Algorithm for Online Statistics

Numerically stable online mean and variance:

```python
class OnlineBootstrapStats:
    def __init__(self, shape, save_samples=False, percentiles=(2.5, 97.5)):
        self.n = 0
        self.mean = np.zeros(shape)
        self.M2 = np.zeros(shape)  # Sum of squared differences
        self.samples = [] if save_samples else None
        self.percentiles = percentiles

    def update(self, sample):
        """Update statistics with new sample (Welford's algorithm)."""
        self.n += 1
        delta = sample - self.mean
        self.mean += delta / self.n
        delta2 = sample - self.mean
        self.M2 += delta * delta2

        if self.samples is not None:
            self.samples.append(sample)

    def get_results(self):
        """Compute final statistics."""
        std = np.sqrt(self.M2 / (self.n - 1))
        z = self.mean / std
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(np.abs(z)))

        # Confidence intervals
        if self.samples is not None:
            # Exact percentile CIs
            samples_array = np.array(self.samples)
            ci_lower = np.percentile(samples_array, self.percentiles[0], axis=0)
            ci_upper = np.percentile(samples_array, self.percentiles[1], axis=0)
        else:
            # Normal approximation CIs
            from scipy.stats import norm
            z_score = norm.ppf(1 - self.percentiles[0] / 100)
            ci_lower = self.mean - z_score * std
            ci_upper = self.mean + z_score * std

        return {
            'mean': self.mean,
            'std': std,
            'Z': z,
            'p': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'samples': np.array(self.samples) if self.samples else None
        }
```

### Ridge Optimization Path

Bypass BrainData overhead for maximum performance:

```python
def _bootstrap_ridge_weights_worker(X, y, indices, alpha, **ridge_params):
    """Bootstrap worker for Ridge weights (pure numpy)."""
    # Resample data
    X_boot = X[indices]
    y_boot = y[indices]

    # Call optimized ridge_svd directly (no BrainData overhead)
    from nltools.stats import ridge_svd
    weights = ridge_svd(X_boot, y_boot, alpha=alpha, **ridge_params)

    return weights  # Shape: (n_features, n_voxels)
```

---

## References

- **Research doc**: See research report from agent (comprehensive analysis)
- **Bootstrap refactor plan**: `claude-guidelines/bootstrap-refactor.md`
- **Inference module**: `nltools/algorithms/inference/one_sample.py` (reference implementation)
- **Knowledge base**: `claude-guidelines/knowledge-base.md`
- **Design philosophy**: `claude-guidelines/design-philosophy.md`

---

**Last Updated**: 2025-10-31
**Status**: Ready for Phase 1 implementation
