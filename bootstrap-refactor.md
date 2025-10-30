# Bootstrap Refactoring Plan - nltools v0.6.0

**Status**: Planning
**Priority**: 2.8 (Pre-Release Blocker)
**Estimated Effort**: 14-18 hours
**Last Updated**: 2025-10-29

---

## Executive Summary

**Goal**: Refactor `.bootstrap()` to be memory-efficient, support fitted model methods (`weights`, `predict`), and optimize performance by farming out computations to underlying algorithms.

**Key Achievements**:
- **Memory efficiency**: 1,673× reduction (9.2TB → 5.5GB for worst-case scenario)
- **Performance**: 10-100× faster for Ridge models (bypass Brain_Data overhead)
- **Generality**: Support `weights` and `predict` for all fitted models (Ridge, GLM)
- **Dual modes**: Efficient (default) vs. full sample storage (`save_weights=True`)

**Design Principles**:
1. **Memory efficiency**: Use online statistics (Welford's algorithm) to avoid O(n_bootstrap × data_size) arrays
2. **Performance**: Farm out to `ridge_svd()` and pure functions instead of Brain_Data methods
3. **Generality**: Support `.bootstrap('weights')` for all models, `.bootstrap('predict')`
4. **Backward compatibility**: Keep existing simple methods working unchanged

---

## Memory Efficiency Analysis

### The Problem

**Worst-case scenario**:
- Dataset: 1000 timepoints × 230k voxels × 5000 features
- Bootstrap: 5000 iterations
- Traditional approach: Store (5000, 1000, 230000) = **9.2 TB** for predict bootstrap

### Our Solution

**Efficient mode** (`save_weights=False`, default):
- **Memory**: 3 arrays × (1000, 230000) = **5.5 GB** (1,673× reduction!)
- **Method**: Online statistics (Welford's algorithm)
- **Returns**: mean, std, z, p, CI (normal approximation)

**Full mode** (`save_weights=True`):
- **Memory**: 9.2 TB (same as traditional)
- **Returns**: Above + exact percentile CIs + all samples
- **Use case**: When exact percentiles needed or downstream aggregation required

### Memory Comparison Table

| Mode | Memory Usage | CI Method | Use Case |
|------|-------------|-----------|----------|
| Efficient (default) | O(output_shape) | Normal approx | Most analyses, production use |
| Full (save_weights) | O(n_bootstrap × output_shape) | Exact percentiles | Research, distribution visualization |

**Example for 1000×230k×5000 dataset, 5000 bootstraps**:
- Simple methods: ~1.8 GB (regardless of mode)
- Weights (efficient): ~5.5 GB
- Weights (full): ~27.6 TB
- Predict (efficient): ~5.5 GB
- Predict (full): ~9.2 TB

---

## Architecture: Three-Tier Performance Strategy

### Tier 1: Simple Methods (Current, Keep As-Is) ✅
**Methods**: `mean`, `median`, `std`, `sum`, `min`, `max`

**Implementation**:
- Use existing `_bootstrap_apply_func()` helper
- Work with Brain_Data objects (overhead acceptable for simple ops)
- Minimal changes needed

**Performance**: Already efficient for these operations

---

### Tier 2: Ridge Model Methods (NEW, Farm Out to ridge_svd) 🚀
**Methods**: `weights`, `predict`

**Strategy**: **Bypass Brain_Data entirely**
- Work directly with numpy arrays (`data.data`, `data.X_`)
- Call `ridge_svd()` function directly (pure numpy/torch)
- Only wrap final results in Brain_Data for return

**Performance gains**:
- No mask/nifti_masker overhead
- No X/Y/design_matrix DataFrame copying
- No _shallow_copy_with_data overhead
- Estimated: **10-100× faster** than Brain_Data-based approach

**Example workflow**:
```python
# OLD (slow): Create Brain_Data, all overhead
new_dat = data[indices]  # Copies mask, masker, X, Y, etc.
new_dat.fit(model='ridge', X=new_dat.X_)  # More copying
weights = new_dat.ridge_weights  # Extract from Brain_Data

# NEW (fast): Pure numpy, direct ridge_svd
X_boot = data.X_[indices]  # Just index numpy array
y_boot = data.data[indices]  # Just index numpy array
weights = ridge_svd(X_boot, y_boot, alpha=alpha)  # Pure function call
```

---

### Tier 3: GLM Model Methods (NEW, Use Brain_Data) 🔧
**Methods**: `weights`, `predict`

**Strategy**: Must use Brain_Data (nilearn requirement)
- nilearn.glm.FirstLevelModel expects 4D nifti images
- Cannot bypass Brain_Data without reimplementing nilearn
- Still use efficient online statistics for aggregation

**Performance**: Slower than Ridge, but unavoidable

---

## Detailed Design

### 1. Updated `.fit()` Method

**Store fit parameters for bootstrap replication**:

```python
def fit(self, model=None, X=None, cv=None, **kwargs):
    # ... existing validation ...

    # NEW: Store fit parameters for bootstrap
    self._fit_params_ = {
        'model_type': model,  # 'ridge' or 'glm'
        'kwargs': kwargs.copy(),  # alpha, backend, noise_model, etc.
    }

    # ... rest of existing code ...
```

**Changes required**:
- **File**: `nltools/data/brain_data.py` lines 557-671
- **Lines added**: 4 lines
- **Tests**: Verify `_fit_params_` exists after fit()

---

### 2. OnlineBootstrapStats Helper Class

**Purpose**: Accumulate bootstrap statistics without storing all samples

**File**: `nltools/stats.py` (new class, ~100 lines)

**Key features**:
- Welford's online algorithm for numerical stability
- Handles multi-dimensional arrays (weights, predictions)
- Optional sample storage for exact percentile CIs
- Memory: O(shape) vs. O(n_samples × shape)

**Class interface**:
```python
class OnlineBootstrapStats:
    """
    Accumulate bootstrap statistics without storing all samples.

    Uses Welford's online algorithm for numerical stability.

    Parameters
    ----------
    shape : tuple
        Shape of each bootstrap sample (e.g., (n_features, n_voxels))
    save_samples : bool
        If True, store all samples for percentile CIs (memory intensive)

    Examples
    --------
    >>> stats = OnlineBootstrapStats(shape=(100, 50000), save_samples=False)
    >>> for i in range(5000):
    ...     sample = get_bootstrap_sample()  # (100, 50000)
    ...     stats.update(sample)
    >>> result = stats.finalize(percentiles=(2.5, 97.5))
    >>> result['mean'].shape  # (100, 50000)
    >>> 'samples' in result  # False (efficient mode)
    """

    def __init__(self, shape, save_samples=False):
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)  # For variance
        self.save_samples = save_samples
        self.samples = [] if save_samples else None

    def update(self, sample):
        """Update statistics with new bootstrap sample (Welford's algorithm)."""
        sample = np.asarray(sample, dtype=np.float64)
        self.count += 1
        delta = sample - self.mean
        self.mean += delta / self.count
        delta2 = sample - self.mean
        self.M2 += delta * delta2

        if self.save_samples:
            self.samples.append(sample)

    def finalize(self, percentiles=(2.5, 97.5)):
        """
        Compute final statistics.

        Returns
        -------
        dict : Statistics (numpy arrays, not Brain_Data yet)
            - 'mean': Mean across bootstraps
            - 'std': Standard deviation
            - 'variance': Variance
            - 'Z': Z-scores (mean/std)
            - 'p': Two-tailed p-values
            - 'ci_lower': Lower confidence bound
            - 'ci_upper': Upper confidence bound
            - 'samples': All samples (only if save_samples=True)
        """
        if self.count < 2:
            raise ValueError("Need at least 2 bootstrap samples")

        # Variance and std
        variance = self.M2 / self.count
        std = np.sqrt(variance)

        # Z-scores and p-values
        with np.errstate(invalid='ignore', divide='ignore'):
            z = self.mean / std

        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(np.abs(z)))

        result = {
            'mean': self.mean,
            'std': std,
            'variance': variance,
            'Z': z,
            'p': p,
        }

        # Confidence intervals
        if self.save_samples and self.samples:
            # Exact percentile CIs
            samples_array = np.array(self.samples)  # (n_bootstrap, ...)
            result['ci_lower'] = np.percentile(samples_array, percentiles[0], axis=0)
            result['ci_upper'] = np.percentile(samples_array, percentiles[1], axis=0)
            result['samples'] = samples_array
        else:
            # Normal approximation CIs
            z_crit = norm.ppf(1 - (100 - (percentiles[1] - percentiles[0])) / 200)
            result['ci_lower'] = self.mean - z_crit * std
            result['ci_upper'] = self.mean + z_crit * std

        return result
```

**Mathematical background**:
- Welford's algorithm: Single-pass, numerically stable variance computation
- Mean: μ_n = μ_{n-1} + (x_n - μ_{n-1})/n
- Variance: M2_n = M2_{n-1} + (x_n - μ_{n-1})(x_n - μ_n)
- Final variance: σ² = M2_n / n

---

### 3. Efficient Bootstrap Helpers

**File**: `nltools/utils.py` (new functions, ~85 lines total)

#### Function 1: `_bootstrap_fitted_weights()`

**Purpose**: Bootstrap model weights efficiently

```python
def _bootstrap_fitted_weights(data, fit_params, indices):
    """
    Bootstrap model weights efficiently by farming out to underlying algorithms.

    Avoids Brain_Data overhead for Ridge models.

    Parameters
    ----------
    data : Brain_Data
        Original dataset
    fit_params : dict
        Fit parameters from data._fit_params_
    indices : ndarray
        Bootstrap sample indices

    Returns
    -------
    weights : ndarray of shape (n_features, n_voxels) or (n_regressors, n_voxels)
    """
    model_type = fit_params['model_type']

    if model_type == 'ridge':
        # EFFICIENT PATH: Farm out to ridge_svd (pure numpy function)
        X_boot = data.X_[indices]
        y_boot = data.data[indices]

        from nltools.algorithms.ridge import ridge_svd
        weights = ridge_svd(
            X_boot, y_boot,
            alpha=fit_params['kwargs'].get('alpha', 1.0),
            backend=fit_params['kwargs'].get('backend', 'auto')
        )
        # Returns: (n_features, n_voxels)

    elif model_type == 'glm':
        # GLM PATH: Need Brain_Data because nilearn expects nifti
        from nltools.data import Brain_Data

        new_dat = data[indices]  # Creates Brain_Data (overhead unavoidable)
        new_dat.fit(
            model=model_type,
            X=new_dat.X_,  # Design matrix from bootstrap sample
            **fit_params['kwargs']
        )
        weights = new_dat.glm_betas.data  # Extract numpy array
        # Returns: (n_regressors, n_voxels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return weights
```

#### Function 2: `_bootstrap_fitted_predict()`

**Purpose**: Bootstrap predictions efficiently

```python
def _bootstrap_fitted_predict(data, fit_params, indices):
    """
    Bootstrap predictions efficiently.

    Parameters
    ----------
    data : Brain_Data
        Original dataset
    fit_params : dict
        Fit parameters from data._fit_params_
    indices : ndarray
        Bootstrap sample indices

    Returns
    -------
    predictions : ndarray of shape (n_samples_bootstrap, n_voxels)
    """
    model_type = fit_params['model_type']

    if model_type == 'ridge':
        # EFFICIENT PATH: ridge_svd + matrix multiply
        X_boot = data.X_[indices]
        y_boot = data.data[indices]

        from nltools.algorithms.ridge import ridge_svd
        weights = ridge_svd(
            X_boot, y_boot,
            alpha=fit_params['kwargs'].get('alpha', 1.0),
            backend=fit_params['kwargs'].get('backend', 'auto')
        )
        predictions = X_boot @ weights  # (n_samples_boot, n_voxels)

    elif model_type == 'glm':
        # GLM PATH: Need Brain_Data
        from nltools.data import Brain_Data

        new_dat = data[indices]
        new_dat.fit(model=model_type, X=new_dat.X_, **fit_params['kwargs'])
        predictions = new_dat.predict().data  # (n_samples_boot, n_voxels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return predictions
```

#### Function 3: `_bootstrap_iteration()`

**Purpose**: Single bootstrap iteration (for parallel execution)

```python
def _bootstrap_iteration(data, bootstrap_func, seed):
    """
    Execute single bootstrap iteration.

    Parameters
    ----------
    data : Brain_Data
        Original dataset
    bootstrap_func : callable
        Function to apply (_bootstrap_fitted_weights or _bootstrap_fitted_predict)
    seed : int
        Random seed for this iteration

    Returns
    -------
    result : ndarray
        Bootstrap result (weights or predictions)
    """
    from sklearn.utils import check_random_state

    random_state = check_random_state(seed)
    data_row_id = range(data.shape[0])
    indices = random_state.choice(data_row_id, size=len(data_row_id), replace=True)

    result = bootstrap_func(data, data._fit_params_, indices)
    return result
```

---

### 4. Refactored `.bootstrap()` Method

**File**: `nltools/data/brain_data.py` (lines 2035-2078, complete rewrite, ~200 lines total)

**Main method** (~80 lines):

```python
def bootstrap(
    self,
    function,
    n_samples=5000,
    save_weights=False,
    n_jobs=-1,
    random_state=None,
    percentiles=(2.5, 97.5),
    **kwargs
):
    """
    Bootstrap Brain_Data methods with memory-efficient online statistics.

    Supports simple aggregations and fitted model methods. For fitted models,
    uses efficient algorithms that bypass Brain_Data overhead when possible.

    Parameters
    ----------
    function : str
        Method to bootstrap:
        - Simple: 'mean', 'median', 'std', 'sum', 'min', 'max'
        - Fitted models: 'weights', 'predict' (requires fit() first)
    n_samples : int, default=5000
        Number of bootstrap iterations
    save_weights : bool, default=False
        If False: Use memory-efficient online statistics (normal approx CIs)
        If True: Store all samples for exact percentile CIs (memory intensive!)
    n_jobs : int, default=-1
        Number of parallel workers (-1 = all CPUs)
    random_state : int, optional
        Random seed for reproducibility
    percentiles : tuple of (lower, upper), default=(2.5, 97.5)
        Percentiles for confidence intervals (e.g., (2.5, 97.5) for 95% CI)
    **kwargs : dict
        Additional arguments (currently unused, for future extensions)

    Returns
    -------
    dict : Bootstrap statistics
        Keys depend on save_weights parameter:

        Always included:
        - 'mean': Mean across bootstraps (Brain_Data)
        - 'std': Standard deviation (Brain_Data)
        - 'Z': Z-scores (mean/std) (Brain_Data)
        - 'p': Two-tailed p-values (Brain_Data)
        - 'ci_lower': Lower confidence bound (Brain_Data)
        - 'ci_upper': Upper confidence bound (Brain_Data)

        If save_weights=True:
        - 'samples': All bootstrap samples (Brain_Data)
        - 'ci_lower'/'ci_upper': Exact percentile CIs (not normal approx)

    Raises
    ------
    ValueError
        If function not supported, or if fitted method called without fit()

    Examples
    --------
    >>> # Simple aggregation (always efficient)
    >>> result = brain.bootstrap('mean', n_samples=1000)
    >>> result['Z']  # Z-score map
    >>>
    >>> # Ridge weights (efficient by default)
    >>> brain.fit(model='ridge', alpha=1.0, X=features)
    >>> result = brain.bootstrap('weights', n_samples=1000)
    >>> result['mean']  # Mean weight map (n_features, n_voxels)
    >>>
    >>> # With exact percentile CIs (memory intensive!)
    >>> result = brain.bootstrap('weights', n_samples=1000, save_weights=True)
    >>> result['ci_lower']  # Exact 2.5th percentile
    >>> result['samples'].shape  # (1000, n_features, n_voxels)
    >>>
    >>> # Bootstrap predictions
    >>> result = brain.bootstrap('predict', n_samples=1000)
    >>> result['mean'].shape  # (n_samples, n_voxels)

    Notes
    -----
    Memory usage:
    - Simple methods: ~O(n_voxels) regardless of n_samples
    - Fitted methods (save_weights=False): ~O(output_shape)
    - Fitted methods (save_weights=True): O(n_samples * output_shape) - can be huge!

    Performance optimization:
    - Ridge models: Bypasses Brain_Data overhead, uses ridge_svd() directly
    - GLM models: Uses Brain_Data (required by nilearn)
    """
    from nltools.utils import (
        _bootstrap_apply_func,
        _bootstrap_fitted_weights,
        _bootstrap_fitted_predict,
    )
    from nltools.stats import OnlineBootstrapStats
    from sklearn.utils import check_random_state

    # 1. Validate inputs
    SIMPLE_METHODS = ['mean', 'median', 'std', 'sum', 'min', 'max']
    FITTED_METHODS = ['weights', 'predict']
    ALL_METHODS = SIMPLE_METHODS + FITTED_METHODS

    if function not in ALL_METHODS:
        raise ValueError(
            f"Unsupported bootstrap method '{function}'. "
            f"Supported methods:\n"
            f"  Simple: {SIMPLE_METHODS}\n"
            f"  Fitted models: {FITTED_METHODS} (requires fit() first)"
        )

    if check_brain_data_is_single(self):
        raise ValueError(
            "Cannot bootstrap a single image. Bootstrap requires multiple samples."
        )

    # 2. Check prerequisites for fitted methods
    if function in FITTED_METHODS:
        if not hasattr(self, 'model_') or not hasattr(self, '_fit_params_'):
            raise ValueError(
                f"Must call fit() before bootstrapping '{function}'.\n"
                f"Example: brain.fit(model='ridge', alpha=1.0, X=features)"
            )

    # 3. Setup random state
    random_state = check_random_state(random_state)
    seeds = random_state.randint(MAX_INT, size=n_samples)

    # 4. Choose execution path
    if function in SIMPLE_METHODS:
        # SIMPLE METHODS: Use existing pattern (backward compatible)
        bootstrapped = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_apply_func)(
                self, function, random_state=seeds[i], **kwargs
            )
            for i in range(n_samples)
        )
        bootstrapped = Brain_Data(bootstrapped, mask=self.mask)
        return summarize_bootstrap(bootstrapped, save_weights=save_weights)

    else:
        # FITTED METHODS: Choose efficient or traditional path
        if function == 'weights':
            bootstrap_func = _bootstrap_fitted_weights
        elif function == 'predict':
            bootstrap_func = _bootstrap_fitted_predict

        if save_weights:
            return self._bootstrap_traditional(
                bootstrap_func, n_samples, n_jobs, seeds, save_weights, percentiles
            )
        else:
            return self._bootstrap_online(
                bootstrap_func, n_samples, n_jobs, seeds, percentiles
            )
```

**Helper: `_bootstrap_online()`** (~50 lines):

```python
def _bootstrap_online(self, bootstrap_func, n_samples, n_jobs, seeds, percentiles):
    """Memory-efficient bootstrap using online statistics (Welford's algorithm)."""
    from nltools.stats import OnlineBootstrapStats
    from nltools.data import Brain_Data
    from nltools.utils import _bootstrap_iteration
    from joblib import Parallel, delayed

    # Get one sample to determine output shape
    data_row_id = range(self.shape[0])
    test_indices = np.random.choice(data_row_id, size=len(data_row_id), replace=True)
    test_result = bootstrap_func(self, self._fit_params_, test_indices)
    result_shape = test_result.shape

    # Initialize online statistics accumulator
    stats = OnlineBootstrapStats(shape=result_shape, save_samples=False)

    # Parallel bootstrap with batching (manage memory)
    batch_size = min(100, n_samples)
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_seeds = seeds[batch_start:batch_end]

        # Run batch in parallel
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_iteration)(
                self, bootstrap_func, batch_seeds[i]
            )
            for i in range(len(batch_seeds))
        )

        # Update online statistics
        for result in batch_results:
            stats.update(result)

    # Finalize statistics (returns dict of numpy arrays)
    result_dict = stats.finalize(percentiles=percentiles)

    # Wrap in Brain_Data objects
    wrapped_results = {}
    for key, value in result_dict.items():
        wrapped_results[key] = Brain_Data(value, mask=self.mask)

    return wrapped_results
```

**Helper: `_bootstrap_traditional()`** (~40 lines):

```python
def _bootstrap_traditional(self, bootstrap_func, n_samples, n_jobs, seeds,
                          save_weights, percentiles):
    """Traditional bootstrap: collect all samples (memory intensive)."""
    from nltools.stats import OnlineBootstrapStats
    from nltools.data import Brain_Data
    from nltools.utils import _bootstrap_iteration
    from joblib import Parallel, delayed

    # Run all bootstraps in parallel
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_iteration)(self, bootstrap_func, seeds[i])
        for i in range(n_samples)
    )

    # Stack results
    all_results = np.array(all_results)  # (n_samples, ...)

    # Compute statistics with samples stored
    stats = OnlineBootstrapStats(shape=all_results.shape[1:], save_samples=True)
    for result in all_results:
        stats.update(result)

    result_dict = stats.finalize(percentiles=percentiles)

    # Wrap in Brain_Data
    wrapped_results = {}
    for key, value in result_dict.items():
        wrapped_results[key] = Brain_Data(value, mask=self.mask)

    return wrapped_results
```

---

## Test-Driven Development Plan

### Phase 1: Infrastructure (Online Statistics)

#### Test 1.1: OnlineBootstrapStats - 1D arrays
```python
def test_online_bootstrap_stats_single_dimension():
    """Test online stats with 1D arrays."""
    from nltools.stats import OnlineBootstrapStats

    # Create known samples (all identical)
    samples = [np.array([1.0, 2.0, 3.0]) for _ in range(100)]

    stats = OnlineBootstrapStats(shape=(3,), save_samples=False)
    for sample in samples:
        stats.update(sample)

    result = stats.finalize()

    # All samples identical → std should be 0
    assert np.allclose(result['std'], 0.0)
    assert np.allclose(result['mean'], [1.0, 2.0, 3.0])
    assert 'samples' not in result  # Efficient mode
```

#### Test 1.2: OnlineBootstrapStats - 2D arrays (weights shape)
```python
def test_online_bootstrap_stats_2d():
    """Test online stats with weight matrix shape."""
    stats = OnlineBootstrapStats(shape=(10, 1000), save_samples=False)

    # Add random samples
    np.random.seed(42)
    for _ in range(50):
        sample = np.random.randn(10, 1000)
        stats.update(sample)

    result = stats.finalize()

    assert result['mean'].shape == (10, 1000)
    assert result['std'].shape == (10, 1000)
    assert result['Z'].shape == (10, 1000)
    assert result['p'].shape == (10, 1000)
    assert 'samples' not in result  # save_samples=False
```

#### Test 1.3: OnlineBootstrapStats - sample storage mode
```python
def test_online_bootstrap_stats_save_samples():
    """Test exact percentile CIs when samples stored."""
    stats = OnlineBootstrapStats(shape=(5,), save_samples=True)

    samples_list = []
    np.random.seed(42)
    for _ in range(100):
        sample = np.random.randn(5)
        stats.update(sample)
        samples_list.append(sample)

    result = stats.finalize(percentiles=(2.5, 97.5))

    # Check exact percentiles
    samples_array = np.array(samples_list)
    expected_lower = np.percentile(samples_array, 2.5, axis=0)

    assert 'samples' in result
    assert result['samples'].shape == (100, 5)
    assert np.allclose(result['ci_lower'], expected_lower, rtol=1e-10)
```

#### Test 1.4: OnlineBootstrapStats - numerical stability
```python
def test_online_bootstrap_stats_numerical_stability():
    """Test Welford's algorithm is numerically stable."""
    # Create samples with large mean, small variance
    # (tests numerical precision)
    stats = OnlineBootstrapStats(shape=(3,), save_samples=False)

    np.random.seed(42)
    mean_val = 1e10
    for _ in range(1000):
        sample = np.array([mean_val, mean_val, mean_val]) + np.random.randn(3) * 0.1
        stats.update(sample)

    result = stats.finalize()

    # Should accurately capture small variance despite large mean
    assert np.allclose(result['mean'], mean_val, rtol=1e-8)
    assert np.all(result['std'] < 1.0)  # Small variance preserved
```

---

### Phase 2: Simple Methods (Backward Compatibility)

#### Test 2.1: Unskip and update existing test
```python
# In nltools/tests/shell/test_brain_data.py, line 1475
# Remove: @pytest.mark.skip(reason="method needs refactoring")

def test_bootstrap_mean_and_std(self, sim_brain_data):
    """Test bootstrap with mean and std (simple methods)."""
    masked = sim_brain_data.apply_mask(
        create_sphere(radius=10, coordinates=[0, 0, 0])
    )

    n_samples = 10
    result = masked.bootstrap("mean", n_samples=n_samples)

    # Should return dict with expected keys
    assert isinstance(result, dict)
    assert "Z" in result
    assert "p" in result
    assert "mean" in result
    assert isinstance(result["Z"], Brain_Data)

    # Test std as well
    result = masked.bootstrap("std", n_samples=n_samples)
    assert isinstance(result["Z"], Brain_Data)
```

#### Test 2.2: All simple methods
```python
@pytest.mark.parametrize("method", ["mean", "std", "median", "sum", "min", "max"])
def test_bootstrap_simple_methods(self, sim_brain_data, method):
    """Test all simple aggregation methods work."""
    result = sim_brain_data.bootstrap(method, n_samples=5)

    assert isinstance(result, dict)
    assert isinstance(result["Z"], Brain_Data)
    assert "mean" in result
    assert "p" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
```

#### Test 2.3: Simple methods with save_weights
```python
def test_bootstrap_simple_methods_save_weights(self, sim_brain_data):
    """Test simple methods can save samples."""
    n_samples = 10
    result = sim_brain_data.bootstrap("mean", n_samples=n_samples, save_weights=True)

    assert "samples" in result
    assert isinstance(result["samples"], Brain_Data)
    assert result["samples"].shape[0] == n_samples
```

---

### Phase 3: Weights Bootstrap (New Functionality)

#### Test 3.1: Weights requires fitted model
```python
def test_bootstrap_weights_requires_fit(self, sim_brain_data):
    """Test weights bootstrap requires fit() first."""
    with pytest.raises(ValueError, match="Must call fit.*before bootstrap"):
        sim_brain_data.bootstrap("weights", n_samples=5)
```

#### Test 3.2: Ridge weights - efficient mode
```python
def test_bootstrap_ridge_weights_efficient(self, sim_brain_data):
    """Test memory-efficient weights bootstrap (default mode)."""
    X = np.random.randn(len(sim_brain_data), 10)
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    # Efficient mode (save_weights=False, default)
    result = sim_brain_data.bootstrap("weights", n_samples=50, save_weights=False)

    # Should return stats without samples
    assert "mean" in result
    assert "std" in result
    assert "Z" in result
    assert "p" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "samples" not in result  # Memory efficient!

    # Shape should be (n_features, n_voxels)
    n_features = 10
    n_voxels = sim_brain_data.shape[1]
    assert result["mean"].shape == (n_features, n_voxels)
    assert isinstance(result["mean"], Brain_Data)
```

#### Test 3.3: Ridge weights - full mode
```python
def test_bootstrap_ridge_weights_full_mode(self, sim_brain_data):
    """Test full weights bootstrap with sample storage."""
    X = np.random.randn(len(sim_brain_data), 10)
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    n_bootstrap = 20
    result = sim_brain_data.bootstrap("weights", n_samples=n_bootstrap, save_weights=True)

    # Should include samples and exact percentile CIs
    assert "samples" in result
    expected_shape = (n_bootstrap, 10, sim_brain_data.shape[1])
    assert result["samples"].shape == expected_shape

    # CIs should exist and have correct shape
    assert result["ci_lower"].shape == (10, sim_brain_data.shape[1])
    assert result["ci_upper"].shape == (10, sim_brain_data.shape[1])
```

#### Test 3.4: GLM weights bootstrap
```python
def test_bootstrap_glm_weights(self, sim_brain_data):
    """Test weights bootstrap works with GLM."""
    design = pd.DataFrame({
        "Intercept": np.ones(len(sim_brain_data)),
        "X1": np.random.randn(len(sim_brain_data)),
        "X2": np.random.randn(len(sim_brain_data))
    })
    sim_brain_data.fit(model="glm", noise_model="ols", X=design)

    result = sim_brain_data.bootstrap("weights", n_samples=10, save_weights=False)

    # Should have shape (n_regressors, n_voxels)
    assert result["mean"].shape == (3, sim_brain_data.shape[1])
```

#### Test 3.5: Weights bootstrap preserves hyperparameters
```python
def test_bootstrap_weights_preserves_alpha(self, sim_brain_data):
    """Test bootstrap refits with same alpha."""
    X = np.random.randn(len(sim_brain_data), 5)
    alpha_high = 100.0

    sim_brain_data.fit(model="ridge", alpha=alpha_high, X=X)
    result = sim_brain_data.bootstrap("weights", n_samples=10, save_weights=False)

    # High alpha → weights should be small (heavily regularized)
    # This is an indirect test but verifies alpha is being used
    assert np.all(np.abs(result["mean"].data) < 10.0)  # Shrunk toward zero
```

#### Test 3.6: Weights bootstrap has variance
```python
def test_bootstrap_weights_has_variance(self, sim_brain_data):
    """Test bootstrap weights vary across samples."""
    X = np.random.randn(len(sim_brain_data), 5)
    sim_brain_data.fit(model="ridge", alpha=0.1, X=X)

    result = sim_brain_data.bootstrap("weights", n_samples=50, save_weights=True)

    # Different bootstrap samples → different weights
    assert np.all(result["std"].data > 0)  # Non-zero variance
```

---

### Phase 4: Predict Bootstrap (New Functionality)

#### Test 4.1: Predict requires fitted model
```python
def test_bootstrap_predict_requires_fit(self, sim_brain_data):
    """Test predict bootstrap requires fit() first."""
    with pytest.raises(ValueError, match="Must call fit.*before bootstrap"):
        sim_brain_data.bootstrap("predict", n_samples=5)
```

#### Test 4.2: Ridge predict - efficient mode
```python
def test_bootstrap_ridge_predict_efficient(self, sim_brain_data):
    """Test memory-efficient predict bootstrap."""
    X = np.random.randn(len(sim_brain_data), 10)
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    result = sim_brain_data.bootstrap("predict", n_samples=30, save_weights=False)

    # Should return stats for each (sample, voxel) position
    # Shape: (n_samples, n_voxels)
    expected_shape = (len(sim_brain_data), sim_brain_data.shape[1])
    assert result["mean"].shape == expected_shape
    assert "samples" not in result  # Memory efficient
```

#### Test 4.3: Ridge predict - full mode
```python
def test_bootstrap_predict_full_mode(self, sim_brain_data):
    """Test full predict bootstrap (memory intensive)."""
    X = np.random.randn(len(sim_brain_data), 10)
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    n_bootstrap = 15
    result = sim_brain_data.bootstrap("predict", n_samples=n_bootstrap, save_weights=True)

    # Should store all predictions
    # Shape: (n_bootstrap, n_samples, n_voxels)
    expected_shape = (n_bootstrap, len(sim_brain_data), sim_brain_data.shape[1])
    assert result["samples"].shape == expected_shape
```

#### Test 4.4: Predict bootstrap variability
```python
def test_bootstrap_predict_has_variance(self, sim_brain_data):
    """Test predictions vary across bootstrap samples."""
    X = np.random.randn(len(sim_brain_data), 10)
    sim_brain_data.fit(model="ridge", alpha=0.1, X=X)

    result = sim_brain_data.bootstrap("predict", n_samples=50, save_weights=True)

    # Bootstrap samples should have variance
    # (different samples → different fits → different predictions)
    assert np.all(result["std"].data > 0)  # Non-zero variance
```

#### Test 4.5: GLM predict bootstrap
```python
def test_bootstrap_glm_predict(self, sim_brain_data):
    """Test predict bootstrap works with GLM."""
    design = pd.DataFrame({
        "Intercept": np.ones(len(sim_brain_data)),
        "X1": np.random.randn(len(sim_brain_data))
    })
    sim_brain_data.fit(model="glm", noise_model="ols", X=design)

    result = sim_brain_data.bootstrap("predict", n_samples=10, save_weights=False)

    # Should have shape (n_samples, n_voxels)
    assert result["mean"].shape == (len(sim_brain_data), sim_brain_data.shape[1])
```

---

### Phase 5: Error Handling & Edge Cases

#### Test 5.1: Invalid method raises clear error
```python
def test_bootstrap_invalid_method(self, sim_brain_data):
    """Test clear error for unsupported methods."""
    with pytest.raises(ValueError, match="Unsupported bootstrap method.*weights.*predict"):
        sim_brain_data.bootstrap("invalid_method", n_samples=5)
```

#### Test 5.2: Single image raises error
```python
def test_bootstrap_single_image_raises_error(self):
    """Test bootstrap with single image fails."""
    single = Brain_Data(nib.load(MNI_Template.MNI152_T1_1mm))

    with pytest.raises(ValueError, match="Cannot bootstrap.*single image"):
        single.bootstrap("mean", n_samples=10)
```

#### Test 5.3: Custom percentiles
```python
def test_bootstrap_custom_percentiles(self, sim_brain_data):
    """Test custom confidence interval percentiles."""
    X = np.random.randn(len(sim_brain_data), 5)
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    # 90% CI (5th to 95th percentile)
    result = sim_brain_data.bootstrap("weights", n_samples=20,
                                      percentiles=(5, 95), save_weights=False)

    assert "ci_lower" in result
    assert "ci_upper" in result
    # CIs should be narrower than 95% CI (can't test directly without storing samples)
```

#### Test 5.4: Reproducibility with random_state
```python
def test_bootstrap_reproducible_with_seed(self, sim_brain_data):
    """Test bootstrap is reproducible with random_state."""
    X = np.random.randn(len(sim_brain_data), 5)
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    result1 = sim_brain_data.bootstrap("weights", n_samples=20, random_state=42)
    result2 = sim_brain_data.bootstrap("weights", n_samples=20, random_state=42)

    # Should get identical results
    np.testing.assert_allclose(result1["mean"].data, result2["mean"].data)
```

#### Test 5.5: Too few bootstrap samples
```python
def test_bootstrap_too_few_samples(self, sim_brain_data):
    """Test error with n_samples < 2."""
    X = np.random.randn(len(sim_brain_data), 5)
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    # Need at least 2 samples for variance
    with pytest.raises(ValueError, match="at least 2.*samples"):
        sim_brain_data.bootstrap("weights", n_samples=1)
```

---

### Phase 6: Performance & Memory Tests

#### Test 6.1: Memory efficiency verification
```python
def test_bootstrap_memory_efficient(self, sim_brain_data):
    """Test efficient mode doesn't store all samples."""
    X = np.random.randn(len(sim_brain_data), 100)  # Many features
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    # Run efficient bootstrap
    result = sim_brain_data.bootstrap("weights", n_samples=100, save_weights=False)

    # Should not have samples stored
    assert "samples" not in result

    # Memory usage should be much smaller than full mode
    # (Can't easily test actual memory, but verify structure)
    assert all(key in result for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"])
```

#### Test 6.2: Ridge performance (farms out to ridge_svd)
```python
def test_ridge_bootstrap_performance(self, sim_brain_data):
    """Test Ridge bootstrap completes quickly (efficient path)."""
    X = np.random.randn(len(sim_brain_data), 200)  # Many features
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    import time
    start = time.time()
    result = sim_brain_data.bootstrap("weights", n_samples=10, save_weights=False)
    elapsed = time.time() - start

    # Should complete reasonably quickly
    # (Actual threshold depends on hardware, but <10s is reasonable)
    assert elapsed < 10.0
    assert result["mean"].shape == (200, sim_brain_data.shape[1])
```

#### Test 6.3: Batching works correctly
```python
def test_bootstrap_batching(self, sim_brain_data):
    """Test batched processing gives same results as unbatched."""
    X = np.random.randn(len(sim_brain_data), 5)
    sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

    # Run with same seed
    result = sim_brain_data.bootstrap("weights", n_samples=250, random_state=42)

    # Should work correctly despite batching (batch_size=100 internally)
    assert result["mean"].shape == (5, sim_brain_data.shape[1])
    assert np.all(np.isfinite(result["mean"].data))
```

---

## Implementation Checklist

### Step 1: Add OnlineBootstrapStats class ✓
**File**: `nltools/stats.py`
- Location: Add at end of file
- Lines: ~100 lines
- Tests: Phase 1 (4 tests)
- **Estimated time**: 2-3 hours

### Step 2: Add efficient bootstrap helpers ✓
**File**: `nltools/utils.py`
- Add 3 functions:
  - `_bootstrap_fitted_weights()` (~40 lines)
  - `_bootstrap_fitted_predict()` (~35 lines)
  - `_bootstrap_iteration()` (~10 lines)
- Total: ~85 lines
- Tests: Implicitly tested via Phase 3-4
- **Estimated time**: 2-3 hours

### Step 3: Update `.fit()` to store parameters ✓
**File**: `nltools/data/brain_data.py`
- Location: Lines 557-671
- Change: Add 4 lines to store `_fit_params_`
- Tests: Quick verification test
- **Estimated time**: 15 minutes

### Step 4: Refactor `.bootstrap()` method ✓
**File**: `nltools/data/brain_data.py`
- Location: Lines 2035-2078 (complete rewrite)
- New code: ~200 lines total
  - Main `bootstrap()` method (~80 lines)
  - `_bootstrap_online()` helper (~50 lines)
  - `_bootstrap_traditional()` helper (~40 lines)
  - Add imports (~30 lines distributed)
- Tests: All phases (2-6)
- **Estimated time**: 4-5 hours

### Step 5: Update tests ✓
**File**: `nltools/tests/shell/test_brain_data.py`
- Remove skip decorator (line 1475)
- Update existing test (~10 lines)
- Add new test class: `TestBootstrapFittedModels` (~400 lines)
  - Phase 1 tests: Infrastructure (4 tests, ~80 lines)
  - Phase 2 tests: Simple methods (3 tests, ~50 lines)
  - Phase 3 tests: Weights (6 tests, ~120 lines)
  - Phase 4 tests: Predict (5 tests, ~100 lines)
  - Phase 5 tests: Errors (5 tests, ~100 lines)
  - Phase 6 tests: Performance (3 tests, ~60 lines)
- Total: 26 new/updated tests
- **Estimated time**: 4-5 hours

### Step 6: Update documentation ✓
**Files**:
- `REFACTORING_PLAN.md`: Mark task 2.8 complete, update test counts
- `MIGRATION_v0.5_to_v0.6.md`: Document new bootstrap functionality
- **Estimated time**: 1 hour

---

## Migration Impact

### Breaking Changes
**None** - Fully backward compatible

### Deprecations
**None**

### New Features
1. ✅ `.bootstrap('weights')` - Bootstrap model weights (Ridge, GLM)
2. ✅ `.bootstrap('predict')` - Bootstrap predictions
3. ✅ Memory-efficient mode (default): O(output_shape) instead of O(n_bootstrap × output_shape)
4. ✅ Exact percentile CIs with `save_weights=True`
5. ✅ Custom percentiles via `percentiles` parameter
6. ✅ Reproducible bootstraps via `random_state` parameter

### Performance Improvements
- **Ridge bootstrap**: ~10-100× faster (bypass Brain_Data overhead)
- **Memory**: ~1000-10000× reduction in efficient mode
- **Parallelization**: Efficient batching for large n_samples

---

## API Examples

### Example 1: Simple aggregation (unchanged)
```python
# Classic bootstrap (backward compatible)
result = brain.bootstrap('mean', n_samples=1000)
print(result['Z'])  # Z-score map
print(result['p'])  # P-value map
```

### Example 2: Ridge weights (memory efficient)
```python
# Fit model
brain.fit(model='ridge', alpha=1.0, X=features)

# Bootstrap weights (efficient mode)
result = brain.bootstrap('weights', n_samples=1000)
print(result['mean'].shape)  # (n_features, n_voxels)
print(result['ci_lower'])    # Lower 95% CI (normal approx)
```

### Example 3: Exact percentile CIs (memory intensive)
```python
# Fit model
brain.fit(model='ridge', alpha=10.0, X=features)

# Bootstrap with exact CIs
result = brain.bootstrap('weights', n_samples=5000, save_weights=True)
print(result['samples'].shape)  # (5000, n_features, n_voxels)
print(result['ci_lower'])       # Exact 2.5th percentile
```

### Example 4: Bootstrap predictions
```python
# Fit model
brain.fit(model='ridge', alpha=1.0, X=train_features)

# Bootstrap predictions to get uncertainty
result = brain.bootstrap('predict', n_samples=1000)
print(result['mean'].shape)  # (n_samples, n_voxels)
print(result['std'])         # Prediction uncertainty
```

### Example 5: Custom CIs and reproducibility
```python
# 90% confidence intervals, reproducible
brain.fit(model='glm', noise_model='ar1', X=design)
result = brain.bootstrap('weights',
                        n_samples=1000,
                        percentiles=(5, 95),  # 90% CI
                        random_state=42)      # Reproducible
```

---

## Open Questions

**None** - All design decisions finalized:
1. ✅ CI strategy: Dual mode (normal approx efficient, exact when save_weights=True)
2. ✅ Weights scope: Bootstrap entire (n_features, n_voxels) matrix together
3. ✅ Predict output: Return all predictions (n_samples, n_voxels) per iteration
4. ✅ Performance: Farm out to ridge_svd() for Ridge models

---

## Success Criteria

### Must Have ✅
- All existing bootstrap tests pass (simple methods)
- New weights bootstrap works for Ridge and GLM
- New predict bootstrap works for Ridge and GLM
- Memory-efficient mode uses O(output_shape) memory
- Full mode provides exact percentile CIs
- Clear error messages for unsupported methods
- Documentation updated

### Nice to Have ⭐
- Performance benchmarks showing Ridge speedup
- Example notebooks demonstrating new functionality
- Comparison with other bootstrap implementations

---

**Ready for implementation**: This plan provides complete specifications for memory-efficient, high-performance bootstrap functionality with comprehensive test coverage.

---

*Last updated: 2025-10-29*
*Estimated total effort: 14-18 hours*
*Test coverage: 26 new tests across 6 phases*
