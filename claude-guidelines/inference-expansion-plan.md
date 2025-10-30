# GPU-Accelerated Inference Module Expansion

**Date**: 2025-10-30
**Goal**: Extend `nltools/algorithms/inference/` with correlation, timeseries, and matrix permutation tests

---

## Current State

**Completed** (working modules):
- `one_sample.py` - Sign-flipping permutation test
- `two_sample.py` - Group permutation test
- `utils.py` - Shared utilities (_compute_pvalue, _auto_batch_size, _generate_sign_flips)

**Architecture pattern** (proven successful):
1. CPU-parallel implementation (`_*_cpu_parallel()`) using joblib
2. GPU-batched implementation (`_*_gpu_batched()`) with automatic batching
3. Main public function routes to appropriate backend
4. Progress bars for both CPU and GPU modes
5. Consistent API: (data, n_permute, tail, return_null, backend, n_jobs, max_gpu_memory_gb, random_state)

---

## New Modules to Build

### 1. `correlation.py` - Correlation Permutation Tests

**Purpose**: Test correlation between two variables by permuting one and computing null distribution

**What to replace**: `stats.correlation_permutation()` (lines 656-735)

**Key function**: `correlation_permutation_test(data1, data2, ...)`

**Permutation method**: Shuffle indices of data1, correlate with data2

**Metrics** (start with Pearson):
- Pearson (vectorized: easy on GPU)
- Spearman (rank-based: add later)
- Kendall (rank-based: add later)

**Implementation pattern**:
```python
def _correlation_permutation_cpu_parallel(...):
    # Worker function: permute data1, compute correlation
    # Parallel execution with joblib

def _correlation_permutation_gpu_batched(...):
    # Batch generate permutations
    # Vectorized correlation computation on GPU

def correlation_permutation_test(...):
    # Route to CPU or GPU
```

**Pearson correlation** (GPU-friendly):
```python
# Vectorized for multiple permutations
# data1_perm: (n_permute, n_samples)
# data2: (n_samples,)

data1_centered = data1_perm - data1_perm.mean(axis=1, keepdims=True)
data2_centered = data2 - data2.mean()

numerator = (data1_centered @ data2_centered) / n_samples
denominator = data1_centered.std(axis=1) * data2_centered.std()
correlations = numerator / denominator  # (n_permute,)
```

---

### 2. `timeseries.py` - Time-Series Permutation Methods

**Purpose**: Permutation methods that preserve temporal structure (autocorrelation)

**What to replace**: `stats.circle_shift()`, `stats.phase_randomize()`, and time-series options in `correlation_permutation()`

**Key functions**:

**A. `circle_shift(data, shift_amount=None, random_state=None)`**
- Circularly shift time series
- Preserves autocorrelation structure
- Use: `np.roll()` or `np.concatenate([data[-shift:], data[:-shift]])`

**B. `phase_randomize(data, backend=None, random_state=None)`**
- FFT-based phase randomization
- Preserves power spectrum, destroys phase relationships
- Algorithm:
  1. FFT: `fft_data = np.fft.rfft(data)`
  2. Generate random phases: `φ ~ Uniform(0, 2π)`
  3. Apply phase shifts: `fft_data[1:] *= exp(i*φ)`
  4. Inverse FFT: `np.fft.irfft(fft_data, n=len(data))`

**GPU acceleration**: Excellent for FFT operations
- Use `torch.fft.rfft()` and `torch.fft.irfft()`
- Can batch multiple phase randomizations

**C. `timeseries_correlation_permutation_test(...)`**
- Similar to correlation_permutation_test
- Uses circle_shift or phase_randomize instead of simple permutation
- Parameter: `method='circle_shift'` or `method='phase_randomize'`

**Critical test**: Verify phase randomization preserves power spectrum
```python
power_orig = np.abs(np.fft.rfft(data))**2
power_rand = np.abs(np.fft.rfft(phase_randomize(data)))**2
np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)
```

---

### 3. `matrix.py` - Matrix Permutation (Mantel Test)

**Purpose**: Test correlation between two matrices (e.g., distance matrices)

**What to replace**: `stats.matrix_permutation()` (lines 737+)

**Key function**: `matrix_permutation_test(data1, data2, ...)`

**Mantel test**:
- Input: Two square matrices (n×n)
- Extract elements: Upper/lower triangle or full matrix
- Permutation: Symmetrically permute rows AND columns together
- Statistic: Correlation between corresponding matrix elements

**Parameters**:
- `how='upper'|'lower'|'full'`: Which matrix elements to use
- `include_diag=bool`: Include diagonal (for `how='full'`)

**Key operations**:

**Extract elements**:
```python
# Upper triangle (excluding diagonal)
indices = np.triu_indices(n, k=1)
elements = matrix[indices]
```

**Symmetric permutation** (CRITICAL):
```python
# Permute both rows and columns together
permuted_matrix = matrix[perm][:, perm]
# Equivalent to: matrix[perm, :][:, perm]
```

**GPU challenge**: Matrix indexing less efficient on GPU
- May need to keep permutation generation on CPU
- Advanced indexing: `matrix[perm][:, perm]` can be slow
- CPU-parallel version is primary; GPU is bonus

---

## Implementation Order

**Phase 1: correlation.py** (most straightforward)
1. Study one_sample.py pattern thoroughly
2. Write tests first (TDD)
3. Implement CPU-parallel version
4. Implement GPU-batched version
5. Test backend consistency

**Phase 2: timeseries.py** (FFT complexity)
1. Implement circle_shift (simple)
2. Implement phase_randomize (complex, GPU-friendly)
3. Test power spectrum preservation
4. Implement timeseries_correlation_permutation_test
5. Test against stats.py

**Phase 3: matrix.py** (symmetric permutation)
1. Implement helper functions (_extract_matrix_elements, _permute_matrix)
2. Test symmetric permutation thoroughly
3. Implement CPU-parallel version (primary)
4. Attempt GPU-batched version (if feasible)
5. Test against stats.py

---

## Testing Strategy

**Test organization** (in `test_inference.py`):

```python
# Correlation tests
class TestCorrelationPermutation:
    def test_basic_functionality_single_feature(self):
    def test_basic_functionality_multi_feature(self):
    def test_deterministic_with_seed(self):
    def test_significant_correlation(self):
    def test_cpu_parallel_correctness(self):
    def test_gpu_batching_correctness(self):
    def test_matches_stats_py(self):

# Time-series tests
class TestCircleShift:
    def test_preserves_shape(self):
    def test_deterministic(self):
    def test_preserves_values(self):

class TestPhaseRandomize:
    def test_preserves_power_spectrum(self):  # CRITICAL
    def test_backend_consistency(self):

class TestTimeseriesCorrelation:
    def test_circle_shift_method(self):
    def test_phase_randomize_method(self):
    def test_matches_stats_py(self):

# Matrix tests
class TestMatrixHelpers:
    def test_extract_upper_triangle(self):
    def test_permute_matrix_symmetric(self):

class TestMatrixPermutation:
    def test_basic_functionality(self):
    def test_identical_matrices(self):
    def test_different_how_parameters(self):
    def test_matches_stats_py(self):
```

**Backward compatibility**: Each new function must match stats.py results
- Use existing stats.py as reference
- Test with: `result_new['p'] ≈ result_old['p']` (within tolerance ~15%)

---

## TDD Workflow (per module)

```bash
# 1. Write first test (will fail)
uv run pytest nltools/tests/core/test_inference.py::TestCorrelation::test_basic -xvs

# 2. Implement minimal code to pass

# 3. Run ONLY that test
uv run pytest --lf -x

# 4. Add next test, repeat

# 5. Run module tests
uv run pytest nltools/tests/core/test_inference.py::TestCorrelation -n auto -x

# 6. Run tier1 regression
uv run pytest -m tier1 -n auto

# 7. Create log for analysis if needed
uv run pytest -m tier1 -n auto 2>&1 | tee pytest_tier1.log
```

---

## Shared Utilities (already exist in utils.py)

**Reuse**:
- `_compute_pvalue(obs_stat, null_dist, tail)` - P-value computation
- `_auto_batch_size(n_permute, n_samples, n_features, max_memory_gb)` - GPU batching

**May need to add**:
- Helper for computing correlations (Pearson, Spearman, Kendall)
- Helper for FFT operations (if needed)

---

## API Consistency

**All functions follow pattern**:
```python
def *_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,  # (or single array for one-sample)
    n_permute: int = 5000,
    tail: int = 2,
    return_null: bool = False,
    backend: Optional[Union[Backend, str]] = None,
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    random_state: Optional[int] = None,
) -> dict:
    """Returns: {'statistic': ..., 'p': ..., 'backend': ..., 'null_dist': ...}"""
```

**Return format**:
```python
{
    'correlation': float or array,  # Observed statistic
    'p': float or array,            # P-value(s)
    'backend': str,                 # Backend used
    'null_dist': array,             # Null distribution (if return_null=True)
}
```

---

## Exit Criteria (per module)

**Correlation**:
- ✅ CPU-parallel and GPU-batched implementations
- ✅ Matches stats.correlation_permutation() results
- ✅ Pearson correlation working (Spearman/Kendall later)
- ✅ All tests passing

**Timeseries**:
- ✅ circle_shift() preserves autocorrelation
- ✅ phase_randomize() preserves power spectrum
- ✅ GPU acceleration for FFT
- ✅ Matches stats.py results
- ✅ All tests passing

**Matrix**:
- ✅ Symmetric permutation correct
- ✅ Different 'how' parameters work
- ✅ CPU-parallel implementation (GPU bonus)
- ✅ Matches stats.matrix_permutation() results
- ✅ All tests passing

**Integration**:
- ✅ All tier1 tests pass
- ✅ Exports added to __init__.py
- ✅ No regressions in existing code

---

## Success Metrics

**Correctness**:
- Match stats.py results (tolerance ~15% for stochastic)
- Backend consistency (NumPy ≈ PyTorch, tolerance 1e-6)

**Performance**:
- CPU-parallel: 4-8× speedup over sequential
- GPU-batched: 10-100× speedup for large problems

**Quality**:
- Comprehensive test coverage
- Type hints on all functions
- NumPy-style docstrings with examples

---

## References

**Phase randomization**:
- Theiler et al. (1991). Testing for nonlinearity in time series: the method of surrogate data
- Lancaster et al. (2018). Surrogate data for hypothesis testing of physical systems

**Mantel test**:
- Mantel (1967). The detection of disease clustering
- Chen et al. (2016). Untangling the relatedness among correlations

**GPU permutation testing**:
- Eklund et al. (2014). BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs

---

*Status*: Implementation plan
*Next*: Begin Phase 1 (correlation.py)
