# Test Suite Refactoring Plan

---

## ✅ COMPLETED - 2025-10-28

### Status: Implementation Complete

**Final Test Status**: 91/91 passing (100%) ✅

**Implementation completed in 4 commits:**
1. `refactor(tests): Reorganize test_brain_data.py into TestBrainData class` - 38 tests
2. `refactor(tests): Reorganize test_adjacency.py into TestAdjacency class` - 30 tests
3. `refactor(tests): Reorganize test_design_matrix.py into TestDesignMatrix class` - 10 tests
4. `refactor(tests): Organize test_stats.py with section headers and docstrings` - 13 tests

**Test organization follows "imperative shell, functional core" pattern:**
- **Imperative Shell (Class-based)**: test_brain_data.py, test_adjacency.py, test_design_matrix.py
- **Functional Core (Function-based)**: test_stats.py

**See CLAUDE.md for usage examples and test running patterns.**

---

## 🔧 ARCHIVED SESSION HISTORY - Session Summary (2025-10-28)

### Status: Final Test Fix Needed (Historical)

**Historical Test Status**: 131/132 passing (99.2%), 1 failure remaining

### ✅ Completed Fixes

1. **Fixed Adjacency Property Conversions**
   - Converted `.shape()` → `@property` ✅
   - Converted `.isempty()` → `@property` ✅
   - Updated `__repr__()` to use `.shape` not `.shape()` ✅
   - Updated all 7 arithmetic method calls to use `.shape` not `.shape()` ✅
   - **Result**: `test_append` now passes ✅

2. **Partially Fixed align() Function (stats.py)**
   - Added `np.atleast_1d()` wrapper for `a.mean(axis=1)` to handle scalar case ✅
   - **Result**: `test_hyperalignment` now passes ✅
   - **Still Failing**: `test_align` - complex ISC dimension issue 🔄

3. **Removed Plotting Tests (2025-10-28)**
   - Deleted `test_pref_and_plotting()` from test_prefs.py ✅
   - Deleted `test_plot()` from test_adjacency.py ✅
   - Deleted `test_plot_mds()` from test_adjacency.py ✅
   - Replaced `plot_grid_simulation()` calls in test_simulator.py with direct method calls:
     - `simulation.fit()` + `simulation.threshold_simulation()` + `simulation.run_multiple_simulations()` ✅
   - **Result**: 3 plotting tests removed, 131/132 tests passing ✅

**Previously failing tests now resolved** ✅:
- All 6 Adjacency failures fixed (property conversions) ✅
- `test_pref_and_plotting` removed (plotting deprecated) ✅

### Files Modified

1. `nltools/data/adjacency.py`
   - Line 504: `isempty()` → `@property`
   - Line 671: `shape()` → `@property`
   - Line 282: `__repr__()` uses `.shape` not `.shape()`
   - Lines 316, 330, 344, 358, 372, 386, 400: arithmetic ops use `.shape` not `.shape()`

2. `nltools/stats.py`
   - Line 1437: Added `np.atleast_1d()` wrapper

3. `nltools/tests/test_prefs.py` (2025-10-28)
   - Deleted `test_pref_and_plotting()` function (lines 121-133)

4. `nltools/tests/test_adjacency.py` (2025-10-28)
   - Deleted `test_plot()` function (lines 279-281)
   - Deleted `test_plot_mds()` function (lines 284-286)

5. `nltools/tests/test_simulator.py` (2025-10-28)
   - Lines 24-30: Replaced `plot_grid_simulation()` with `fit()` + `threshold_simulation()` + `run_multiple_simulations()`
   - Lines 52-63: Same replacement for FDR test

### How to Resume

```bash
# Check current test status (should show 131/132 passing)
uv run pytest nltools/tests/ -v --tb=no | grep "FAILED\|PASSED"

# After applying the one-line fix (stats.py line 1444: shape[0] → shape[1]):
# Verify all tests pass
uv run pytest nltools/tests/ -v

# Expected result: 132/132 passing (100%) ✅
```

### Key Insights from Session

1. **Property Conversion Cascade**: Converting `.shape()` to `@property` in Brain_Data (already done in v0.6.0) required the same change in Adjacency. This cascaded to fix multiple test failures. Always check for similar patterns across related classes.

2. **Plotting Removal Strategy**: When removing deprecated functionality, check for both direct test functions AND indirect calls within other tests. The `plot_grid_simulation()` method had side effects (calling `fit()` and `threshold_simulation()`), so we couldn't just delete the calls - we had to replace them with the underlying method calls.

---

## Core Principle
- **Imperative Shell**: 1 test class per custom class → tests how methods are USED
- **Functional Core**: Simple tests for pure functions → tests the actual COMPUTATIONS

---

## Reorganized Test Structure

```
nltools/tests/
├── conftest.py                    # Shared fixtures
├── pytest.ini (in pyproject.toml) # Logging config
│
# IMPERATIVE SHELL - Test class usage, not internals
├── test_brain_data.py             # NEW: Single TestBrainData class
├── test_adjacency.py              # REFACTOR: Single TestAdjacency class
├── test_design_matrix.py          # REFACTOR: Single TestDesignMatrix class
│
# FUNCTIONAL CORE - Test pure functions/algorithms
├── test_stats.py                  # Keep simple (already mostly good)
├── test_utils.py                  # Keep simple
├── test_algorithms.py             # If needed
├── test_plotting.py               # If we add plotting function tests
│
# INTEGRATION & SUPPORT
├── test_datasets.py               # Already well-organized with classes
├── test_simulator.py              # Keep as-is or single TestSimulator class
├── test_file_reader.py            # Keep simple
├── test_mask.py                   # Keep simple
├── test_prefs.py                  # Keep simple or TestMNITemplate class
└── test_efficient_copy.py         # Keep (performance tests)
```

---

## Imperative Shell Pattern

### Example: test_brain_data.py

```python
class TestBrainData:
    """Test Brain_Data class - focus on method usage, not implementation"""

    # === Initialization & I/O ===
    def test_init_from_nifti(self, tmpdir):
        """Test loading Brain_Data from nifti file"""
        ...

    def test_init_from_list(self, tmpdir):
        """Test loading Brain_Data from list of files"""
        ...

    def test_write_and_load_h5(self, tmpdir):
        """Test round-trip HDF5 save/load"""
        ...

    def test_load_legacy_h5_backward_compatible(self, old_h5_brain):
        """Test loading old HDF5 format"""
        ...

    # === Arithmetic Operations ===
    def test_add_returns_new_instance(self, sim_brain_data):
        """Test that addition returns new Brain_Data"""
        ...

    def test_subtract_preserves_mask(self, sim_brain_data):
        """Test subtraction doesn't modify original mask"""
        ...

    def test_multiply_by_scalar(self, sim_brain_data):
        """Test scalar multiplication"""
        ...

    def test_divide_by_brain_data(self, sim_brain_data):
        """Test element-wise division"""
        ...

    # === Transform Methods ===
    def test_smooth_calls_nilearn(self, sim_brain_data):
        """Test smooth delegates to nilearn.image.smooth_img"""
        ...

    def test_standardize_preserves_shape(self, sim_brain_data):
        """Test standardization maintains dimensions"""
        ...

    def test_threshold_returns_copy(self):
        """Test threshold returns new instance"""
        ...

    def test_detrend_removes_linear_trend(self, sim_brain_data):
        """Test detrending functionality"""
        ...

    # === GLM & Regression ===
    def test_regress_accepts_design_matrix(self, sim_brain_data):
        """Test regress() works with Design_Matrix argument"""
        ...

    def test_regress_stores_attributes(self, sim_brain_data):
        """Test regress() stores results as attributes (glm_betas, glm_t, etc.)"""
        ...

    def test_regress_backward_compatible_with_self_X(self, sim_brain_data):
        """Test regress() still works with deprecated self.X pattern"""
        ...

    def test_compute_contrasts_requires_regress_first(self, sim_brain_data):
        """Test compute_contrasts() fails if regress() not called"""
        ...

    # === ROI Extraction ===
    def test_extract_roi_with_atlas(self, sim_brain_data):
        """Test ROI extraction using labeled atlas"""
        ...

    def test_extract_roi_invalid_metric_raises_error(self, sim_brain_data):
        """Test invalid metric parameter raises ValueError"""
        ...

    # === Deprecated Methods ===
    def test_predict_raises_not_implemented(self, sim_brain_data):
        """Test .predict() raises NotImplementedError with helpful message"""
        ...

    def test_ttest_raises_not_implemented(self, sim_brain_data):
        """Test .ttest() raises NotImplementedError"""
        ...
```

**Key Points:**
- Test **usage patterns**, not implementation details
- Test **interface contracts**: what goes in, what comes out, what gets stored
- Test **error handling**: invalid inputs, missing prerequisites
- Test **backward compatibility**: deprecated patterns still work with warnings

### Example: test_adjacency.py

```python
class TestAdjacency:
    """Test Adjacency class - focus on matrix type handling and operations"""

    # === Initialization & Type Inference ===
    def test_init_infers_symmetric(self):
        """Test symmetric matrix type detection"""
        ...

    def test_init_directed_matrix(self):
        """Test directed matrix initialization"""
        ...

    def test_init_from_csv(self, tmpdir):
        """Test loading from CSV file"""
        ...

    # === Matrix Operations ===
    def test_squareform_roundtrip(self, sim_adjacency_single):
        """Test vector → matrix → vector preserves data"""
        ...

    def test_append_preserves_labels(self, sim_adjacency_single):
        """Test appending adjacency objects maintains labels"""
        ...

    def test_mean_aggregates_correctly(self, sim_adjacency_multiple):
        """Test mean() across multiple adjacency matrices"""
        ...

    def test_sum_handles_matrix_types(self):
        """Test sum() respects symmetric vs directed matrices"""
        ...

    # === Statistical Methods ===
    def test_ttest_stores_results(self, sim_adjacency_multiple):
        """Test ttest() stores p-values and statistics"""
        ...

    def test_bootstrap_returns_ci(self, sim_adjacency_multiple):
        """Test bootstrap() returns confidence intervals"""
        ...

    # === I/O ===
    def test_write_and_load_h5(self, sim_adjacency_multiple, tmpdir):
        """Test HDF5 round-trip"""
        ...

    def test_load_legacy_h5_backward_compatible(self, old_h5_adj_single):
        """Test loading old HDF5 format"""
        ...
```

### Example: test_design_matrix.py

```python
class TestDesignMatrix:
    """Test Design_Matrix class - focus on transform operations"""

    # === Transform Operations ===
    def test_add_poly_increases_columns(self, sim_design_matrix):
        """Test polynomial basis expansion"""
        ...

    def test_convolve_with_hrf(self, sim_design_matrix):
        """Test HRF convolution"""
        ...

    def test_zscore_specific_columns(self, sim_design_matrix):
        """Test selective z-scoring"""
        ...

    # === Resampling ===
    def test_upsample_increases_rows(self, sim_design_matrix):
        """Test upsampling to higher frequency"""
        ...

    def test_downsample_decreases_rows(self, sim_design_matrix):
        """Test downsampling to lower frequency"""
        ...

    # === Utilities ===
    def test_vif_detects_multicollinearity(self, sim_design_matrix):
        """Test VIF calculation"""
        ...

    def test_clean_removes_high_vif(self, sim_design_matrix):
        """Test automatic collinearity removal"""
        ...
```

---

## Functional Core Pattern

### test_stats.py - Keep as functions, add organization

```python
"""Tests for nltools.stats module - pure function tests

These test the actual computational correctness of statistical algorithms,
not how they're called from Brain_Data/Adjacency methods.
"""

# === Permutation Functions ===

def test_permutation():
    """Test basic permutation generation"""
    ...

def test_matrix_permutation():
    """Test matrix-aware permutation"""
    ...

# === Alignment & ISC Functions ===

def test_align_deterministic_srm():
    """Test deterministic SRM alignment algorithm"""
    ...

def test_align_pairwise():
    """Test pairwise alignment"""
    ...

def test_isc_computation():
    """Test intersubject correlation calculation"""
    ...

def test_isfc_computation():
    """Test intersubject functional correlation"""
    ...

def test_isps_computation():
    """Test intersubject pattern similarity"""
    ...

# === Transform Functions ===

def test_downsample():
    """Test downsampling algorithm"""
    ...

def test_upsample():
    """Test upsampling algorithm"""
    ...

def test_winsorize():
    """Test winsorizing outlier handling"""
    ...

# === Statistical Transforms ===

def test_fisher_r_to_z():
    """Test Fisher r-to-z transformation"""
    ...

def test_transform_pairwise():
    """Test pairwise distance transformations"""
    ...
```

**Key Points:**
- Test **algorithmic correctness**: given input X, expect output Y
- Test **edge cases**: empty arrays, single values, NaN handling
- Test **mathematical properties**: inverse operations, symmetry, etc.
- Keep simple - just functions testing functions

---

## Logging Infrastructure

### pyproject.toml Configuration

```toml
[tool.pytest.ini_options]
# Logging configuration for better debugging
log_file = "tests/pytest_debug.log"
log_file_level = "DEBUG"
log_file_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
log_file_date_format = "%Y-%m-%d %H:%M:%S"

# CLI logging for test runs
log_cli = true
log_cli_level = "INFO"

# Test discovery
testpaths = ["nltools/tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### Usage Examples

```bash
# Capture debug logs to file while running tests
pytest nltools/tests/ -v

# Later, search logs instead of re-running tests
grep "FAILED\|ERROR" tests/pytest_debug.log
grep -A10 "Brain_Data.regress" tests/pytest_debug.log

# Use pytest cache for iteration
pytest --lf  # Run last failed tests
pytest --ff  # Run failures first, then rest

# Selective class-based running
pytest nltools/tests/test_brain_data.py::TestBrainData -v
pytest nltools/tests/test_brain_data.py::TestBrainData::test_regress_accepts_design_matrix -xvs

# Pattern-based selection (now more powerful with classes)
pytest -k "TestBrainData and regress"
pytest -k "TestAdjacency and (append or sum)"
```

---

## Implementation Phases

### Phase 1: Refactor to Test Classes (~2-3 days)

**One file at a time, validate after each:**

#### Step 1: test_brain_data_old.py → test_brain_data.py
1. Create new `test_brain_data.py` with `TestBrainData` class
2. Group existing tests into logical sections:
   - Initialization & I/O
   - Arithmetic operations
   - Transform methods
   - GLM & regression
   - ROI extraction
   - Deprecated methods
3. Keep `test_brain_data_old.py` until validated
4. Run: `pytest nltools/tests/test_brain_data.py -v`
5. Verify: `pytest nltools/tests/test_brain_data.py::TestBrainData -v`

**Estimated**: 1 day

#### Step 2: test_adjacency.py (refactor in place)
1. Wrap all existing tests in `TestAdjacency` class
2. Organize into sections:
   - Initialization & type inference
   - Matrix operations
   - Statistical methods
   - Graph operations
   - I/O
3. Run: `pytest nltools/tests/test_adjacency.py -v`
4. Verify selective running: `pytest -k "TestAdjacency and sum"`

**Estimated**: 0.5 days

#### Step 3: test_design_matrix.py (refactor in place)
1. Wrap all existing tests in `TestDesignMatrix` class
2. Organize into sections:
   - Transform operations
   - Resampling
   - Utilities
3. Run: `pytest nltools/tests/test_design_matrix.py -v`

**Estimated**: 0.5 days

#### Step 4: test_stats.py (add organization only)
1. Keep as functions (no class needed for functional core)
2. Add section comment headers
3. Add docstrings if missing
4. Run: `pytest nltools/tests/test_stats.py -v`

**Estimated**: 0.5 days

### Phase 2: Add Logging & Documentation (~1 day)

1. **Add logging config to pyproject.toml**
   - Configure file logging to `tests/pytest_debug.log`
   - Configure CLI logging level

2. **Update .gitignore**
   ```
   tests/pytest_debug.log
   tests/*.log
   .pytest_cache/
   ```

3. **Document in CLAUDE.md**
   - Add section on test organization
   - Add examples of selective test running
   - Add logging best practices

4. **Update REFACTORING_PLAN.md**
   - Mark test refactoring as complete
   - Document new test patterns

### Phase 3: Cleanup (~0.5 days)

1. Remove `test_brain_data_old.py` once validated
2. Remove any other temporary/old test files
3. Run full test suite one final time
4. Verify coverage hasn't regressed

---

## Validation Checklist

After each phase, verify:

```bash
# ✅ All tests pass
pytest nltools/tests/ -v

# ✅ Class-based selection works
pytest nltools/tests/test_brain_data.py::TestBrainData -v

# ✅ Method-level selection works
pytest nltools/tests/test_brain_data.py::TestBrainData::test_regress_accepts_design_matrix -v

# ✅ Pattern selection works
pytest -k "BrainData and regress" -v
pytest -k "Adjacency and (sum or mean)" -v

# ✅ Cache works
pytest --lf  # Should only run previously failed tests
pytest --ff  # Should run failures first

# ✅ Logs are captured
ls tests/pytest_debug.log  # File should exist
grep "test_" tests/pytest_debug.log  # Should contain test info

# ✅ Coverage hasn't regressed
pytest --cov=nltools --cov-report=term
```

---

## Benefits of This Refactor

### 1. **Clearer Test Organization**
- One class per library class = obvious mapping
- Logical grouping within each test class
- Easy to find tests for specific functionality

### 2. **Faster Test Iteration**
```bash
# Before: Run all 38 Brain_Data tests
pytest nltools/tests/test_brain_data_old.py

# After: Run just GLM tests
pytest nltools/tests/test_brain_data.py::TestBrainData -k "regress or contrast"
```

### 3. **Better Debugging**
- Logs persist to file for analysis
- Can grep logs instead of re-running tests
- Class/method organization makes failures easier to locate

### 4. **Follows pytest Best Practices**
- Uses recommended class-based organization
- Leverages built-in features (fixtures, markers, cache)
- Aligns with pytest documentation patterns

### 5. **Aligns with Architecture**
- Imperative shell tests focus on **usage**
- Functional core tests focus on **correctness**
- Clear separation matches codebase design

---

## Future Enhancements (Optional)

### Add Integration Tests
Once basic refactor is complete, could add:

```python
class TestGLMWorkflow:
    """End-to-end GLM analysis workflow"""

    def test_complete_glm_pipeline(self):
        """Test: load → preprocess → regress → contrast → threshold"""
        ...

class TestROIAnalysis:
    """End-to-end ROI-based analysis"""

    def test_roi_extraction_and_statistics(self):
        """Test: load atlas → extract ROIs → compute stats"""
        ...
```

### Add Performance Tests
```python
class TestBrainDataPerformance:
    """Performance benchmarks for Brain_Data operations"""

    def test_method_chaining_no_deep_copy_overhead(self):
        """Verify efficient copying implementation"""
        ...
```

### Add Property-Based Tests
Using hypothesis for edge case discovery:
```python
from hypothesis import given, strategies as st

class TestBrainDataProperties:
    @given(st.integers(min_value=1, max_value=1000))
    def test_arithmetic_commutative(self, n):
        """Test a + b == b + a for all valid inputs"""
        ...
```

---

## Summary

**Total Effort**: ~4 days
- Phase 1: Refactor to classes (~2-3 days)
- Phase 2: Logging & docs (~1 day)
- Phase 3: Cleanup (~0.5 days)

**Key Principles**:
- ✅ Simple: 1 class per library class
- ✅ Pragmatic: Keep functional core as functions
- ✅ Organized: Logical grouping, clear structure
- ✅ Maintainable: Easy to find and run tests
- ❌ No over-engineering: Flat structure, no complex markers/directories
