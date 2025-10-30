# Polars Migration Plan v3 - TDD Approach
**Date**: 2025-10-29
**Status**: Ready for implementation
**Approach**: Test-driven, behavior-focused, idiomatic Polars

---

## Executive Summary

**Strategy**: Build new implementation via TDD, test BEHAVIOR not implementation details
**Scope**: Design_Matrix only (remove Brain_Data.X/Y, defer Adjacency)
**Polars usage**: Idiomatic expressions in eager mode
**Timeline**: ~20 hours focused work

---

## Part 1: Core Principles

### 1. Test Behavior, Not Implementation
```python
# ✅ Good test - specifies WHAT we want
def test_convolve_shifts_response():
    """HRF convolution should delay peak response by ~5-6 seconds"""
    dm = DesignMatrix({'stim': [0, 1, 1, 1, 0, 0, 0, 0]}, sampling_freq=1.0)
    dm_conv = dm.convolve()

    # Original peak at index 1-3, convolved peak should shift later
    assert dm['stim'].arg_max() < dm_conv['stim'].arg_max()

# ❌ Bad test - couples to implementation
def test_convolve_uses_glover_hrf():
    """Don't test that we use specific HRF internally"""
    # This is implementation detail, not behavioral contract
```

### 2. Idiomatic Polars (No pandas Patterns)
```python
# ✅ Idiomatic Polars
dm._df = dm._df.with_columns(
    pl.col('stim').rolling_mean(window_size=3).alias('smoothed')
)

# ❌ Pandas-style (don't do this)
dm.loc[10:15, 'stim'] = 1  # We're dropping .loc[] entirely
```

### 3. Clean Composition Pattern
```python
class DesignMatrix:
    def __init__(self, data, *, sampling_freq=None, ...):
        self._df = pl.DataFrame(data)  # Polars DataFrame
        self.sampling_freq = sampling_freq  # Metadata

    def _copy_with(self, new_df, **metadata_updates):
        """Helper: create new instance with updated data/metadata"""
        # All transformations use this pattern
```

### 4. Build Separately, Integrate at End
- Temporarily call it `DesignMatrix` (new class name)
- Keep `Design_Matrix` (old) until ready to swap
- Integration testing only after DesignMatrix fully working

---

## Part 2: Behavior Specification (What Design_Matrix Does)

### Core Functionality Contracts

**1. Construction**
- From numpy array (with optional column names)
- From dict
- From Polars/pandas DataFrame
- Empty initialization
- Preserve metadata: `sampling_freq`, `convolved`, `polys`, `multi`

**2. Data Access**
- `dm['col']` → Polars Series
- `dm[['col1', 'col2']]` → DesignMatrix subset
- `dm['col'] = value` → Set/create column
- Properties: `shape`, `columns`, `empty`

**3. Transformations** (all return new DesignMatrix)
- `convolve(conv_func='hrf', columns=None)` - Convolve with HRF kernel
- `add_poly(order, include_lower=True)` - Add Legendre polynomials
- `add_dct_basis(duration=180, drop=0)` - Add DCT basis functions
- `downsample(target)` - Temporal downsampling
- `upsample(target)` - Temporal upsampling
- `zscore(columns=[])` - Z-score standardization

**4. Concatenation** (most complex!)
- `append(dm, axis=0, keep_separate=True, unique_cols=None, fill_na=0)`
  - **Vertical (axis=0)**: Stack runs, auto-separate polynomials
  - **Horizontal (axis=1)**: Add columns
  - **Wildcard support**: `unique_cols=['house*']` matches prefixes
  - **Auto-numbering**: `0_poly_0`, `1_poly_0` for multi-run

**5. Diagnostics**
- `vif(exclude_polys=True)` - Variance inflation factor
- `clean(fill_na=0, exclude_polys=False, thresh=0.95)` - Drop correlated columns
- `corr()` - Correlation matrix

**6. Utilities**
- `drop(columns=[])` - Drop columns
- `fillna(value)` - Fill NaNs
- `details()` - Print metadata summary
- `heatmap()` - SPM-style visualization
- `replace_data(data, column_names=None)` - Replace data, keep metadata

---

## Part 3: TDD Implementation Plan

### Phase 1: Test Design (2 hours)

Write comprehensive tests BEFORE implementing. Tests define the behavioral contract.

**File structure:**
```
nltools/data/design_matrix_new.py     # New implementation
nltools/tests/shell/test_design_matrix_new.py  # New tests
```

**Test categories:**

#### 1.1 Construction Tests
```python
class TestDesignMatrixConstruction:
    """Test all ways to create a DesignMatrix"""

    def test_from_numpy_with_columns(self):
        data = np.random.randn(100, 3)
        dm = DesignMatrix(data, sampling_freq=2, columns=['a', 'b', 'c'])
        assert dm.shape == (100, 3)
        assert dm.columns == ['a', 'b', 'c']
        assert dm.sampling_freq == 2

    def test_from_numpy_auto_columns(self):
        data = np.random.randn(50, 2)
        dm = DesignMatrix(data, sampling_freq=1)
        assert dm.columns == ['0', '1']  # Auto-generated

    def test_from_dict(self):
        dm = DesignMatrix({'a': [1, 2], 'b': [3, 4]}, sampling_freq=2)
        assert dm.shape == (2, 2)
        assert 'a' in dm.columns

    def test_from_polars_dataframe(self):
        df = pl.DataFrame({'x': [1, 2, 3]})
        dm = DesignMatrix(df, sampling_freq=1)
        assert dm.shape == (3, 1)

    def test_from_pandas_dataframe(self):
        """Backward compatibility"""
        pdf = pd.DataFrame({'x': [1, 2, 3]})
        dm = DesignMatrix(pdf, sampling_freq=1)
        assert dm.shape == (3, 1)
        # Internally should be Polars
        assert isinstance(dm._df, pl.DataFrame)

    def test_empty_initialization(self):
        dm = DesignMatrix(sampling_freq=2)
        assert dm.shape == (0, 0)
        assert dm.empty
        assert dm.sampling_freq == 2
```

#### 1.2 Data Access Tests
```python
class TestDesignMatrixAccess:
    """Test column access and manipulation"""

    def test_getitem_single_column_returns_series(self):
        dm = DesignMatrix({'a': [1, 2, 3]}, sampling_freq=1)
        col = dm['a']
        assert isinstance(col, pl.Series)
        assert col.to_list() == [1, 2, 3]

    def test_getitem_multiple_columns_returns_designmatrix(self):
        dm = DesignMatrix({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}, sampling_freq=1)
        subset = dm[['a', 'c']]
        assert isinstance(subset, DesignMatrix)
        assert subset.columns == ['a', 'c']
        assert subset.sampling_freq == 1  # Metadata preserved

    def test_setitem_scalar_broadcasts(self):
        dm = DesignMatrix({'a': [1, 2, 3]}, sampling_freq=1)
        dm['b'] = 0
        assert dm['b'].to_list() == [0, 0, 0]

    def test_setitem_array_matches_length(self):
        dm = DesignMatrix({'a': [1, 2, 3]}, sampling_freq=1)
        dm['b'] = [10, 20, 30]
        assert dm['b'].to_list() == [10, 20, 30]

    def test_setitem_replaces_existing_column(self):
        dm = DesignMatrix({'a': [1, 2, 3]}, sampling_freq=1)
        dm['a'] = [4, 5, 6]
        assert dm['a'].to_list() == [4, 5, 6]
```

#### 1.3 Transformation Tests
```python
class TestDesignMatrixTransformations:
    """Test data transformations"""

    def test_fillna_replaces_missing_values(self):
        dm = DesignMatrix({'a': [1.0, None, 3.0]}, sampling_freq=1)
        dm_filled = dm.fillna(0)
        assert dm_filled['a'].to_list() == [1.0, 0.0, 3.0]
        # Original unchanged
        assert dm['a'][1] is None

    def test_drop_columns(self):
        dm = DesignMatrix({'a': [1], 'b': [2], 'c': [3]}, sampling_freq=1)
        dm_dropped = dm.drop(columns=['b'])
        assert dm_dropped.columns == ['a', 'c']
        # Original unchanged
        assert dm.columns == ['a', 'b', 'c']

    def test_zscore_standardizes_columns(self):
        dm = DesignMatrix({'a': [1, 2, 3, 4, 5]}, sampling_freq=1)
        dm_z = dm.zscore(columns=['a'])
        assert dm_z['a'].mean() == pytest.approx(0.0, abs=1e-10)
        assert dm_z['a'].std() == pytest.approx(1.0, abs=1e-10)

    def test_zscore_all_columns_by_default(self):
        dm = DesignMatrix({'a': [1, 2, 3], 'b': [10, 20, 30]}, sampling_freq=1)
        dm_z = dm.zscore()
        assert dm_z['a'].mean() == pytest.approx(0.0, abs=1e-10)
        assert dm_z['b'].mean() == pytest.approx(0.0, abs=1e-10)
```

#### 1.4 Convolution Tests
```python
class TestDesignMatrixConvolution:
    """Test HRF convolution"""

    def test_convolve_with_default_hrf(self):
        """Convolve with Glover HRF should delay response"""
        # Box-car stimulus at TRs 2-4
        dm = DesignMatrix(
            {'stim': [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]},
            sampling_freq=0.5  # 2s TR
        )
        dm_conv = dm.convolve()

        # Peak should shift later (HRF peaks ~5-6s = 2-3 TRs)
        original_peak = dm['stim'].arg_max()
        convolved_peak = dm_conv['stim'].arg_max()
        assert convolved_peak > original_peak

    def test_convolve_custom_kernel(self):
        """Can convolve with custom kernel"""
        dm = DesignMatrix({'stim': [1, 0, 0, 0]}, sampling_freq=1)

        # Custom kernel: simple 3-point average
        kernel = np.array([0.33, 0.33, 0.33])
        dm_conv = dm.convolve(conv_func=kernel)

        # Should smooth the signal
        assert dm_conv['stim'].to_list()[0] == pytest.approx(0.33, abs=0.01)

    def test_convolve_ignores_polynomial_columns(self):
        """Convolution should skip polynomial columns"""
        dm = DesignMatrix({'stim': [1, 0, 0, 0]}, sampling_freq=1)
        dm = dm.add_poly(order=0)  # Add intercept

        dm_conv = dm.convolve()

        # Polynomial columns should be unchanged
        assert 'poly_0' in dm_conv.columns
        assert dm_conv['poly_0'].to_list() == dm['poly_0'].to_list()

    def test_convolve_specific_columns_only(self):
        """Can convolve only specified columns"""
        dm = DesignMatrix(
            {'stim_A': [1, 0, 0, 0], 'stim_B': [0, 1, 0, 0]},
            sampling_freq=1
        )

        dm_conv = dm.convolve(columns=['stim_A'])

        # Only stim_A should be convolved
        assert dm_conv['stim_A'].to_list() != dm['stim_A'].to_list()
        assert dm_conv['stim_B'].to_list() == dm['stim_B'].to_list()
```

#### 1.5 Polynomial Tests
```python
class TestDesignMatrixPolynomials:
    """Test polynomial and DCT basis addition"""

    def test_add_poly_creates_legendre_polynomials(self):
        dm = DesignMatrix({'stim': [1, 0, 0, 0] * 10}, sampling_freq=1)
        dm_poly = dm.add_poly(order=2, include_lower=True)

        # Should add poly_0, poly_1, poly_2
        assert dm_poly.shape[1] == 4  # stim + 3 polys
        assert 'poly_0' in dm_poly.columns
        assert 'poly_1' in dm_poly.columns
        assert 'poly_2' in dm_poly.columns
        assert dm_poly.polys == ['poly_0', 'poly_1', 'poly_2']

    def test_add_poly_intercept_is_constant(self):
        dm = DesignMatrix({'stim': [1, 2, 3, 4]}, sampling_freq=1)
        dm_poly = dm.add_poly(order=0)

        # poly_0 (intercept) should be constant ~1.0
        assert dm_poly['poly_0'].mean() == pytest.approx(1.0, abs=1e-10)
        assert dm_poly['poly_0'].std() < 1e-10  # Near-zero variance

    def test_add_poly_without_lower_terms(self):
        dm = DesignMatrix({'stim': [1, 2, 3, 4]}, sampling_freq=1)
        dm_poly = dm.add_poly(order=2, include_lower=False)

        # Should only add poly_2
        assert dm_poly.shape[1] == 2  # stim + poly_2
        assert 'poly_2' in dm_poly.columns
        assert 'poly_0' not in dm_poly.columns
        assert 'poly_1' not in dm_poly.columns

    def test_add_poly_idempotent(self):
        """Adding same polynomial twice should skip"""
        dm = DesignMatrix({'stim': [1, 2, 3, 4]}, sampling_freq=1)
        dm1 = dm.add_poly(order=1)
        dm2 = dm1.add_poly(order=1)

        # Should be same (not duplicate columns)
        assert dm1.shape == dm2.shape

    def test_add_dct_basis_creates_cosine_filters(self):
        dm = DesignMatrix(
            np.zeros((100, 1)),
            sampling_freq=0.5,  # 2s TR
            columns=['stim']
        )

        dm_dct = dm.add_dct_basis(duration=60)  # 60s filter

        # Should add cosine basis functions
        assert 'cosine_1' in dm_dct.columns
        assert len([c for c in dm_dct.columns if 'cosine' in c]) > 1
        assert 'cosine_1' in dm_dct.polys
```

#### 1.6 Append Tests (Most Critical!)
```python
class TestDesignMatrixAppend:
    """Test concatenation operations"""

    def test_horizontal_append_adds_columns(self):
        dm1 = DesignMatrix({'a': [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({'b': [3, 4]}, sampling_freq=1)

        dm_combined = dm1.append(dm2, axis=1)

        assert dm_combined.shape == (2, 2)
        assert dm_combined.columns == ['a', 'b']

    def test_vertical_append_stacks_rows(self):
        dm1 = DesignMatrix({'a': [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({'a': [3, 4]}, sampling_freq=1)

        dm_combined = dm1.append(dm2, axis=0, keep_separate=False)

        assert dm_combined.shape == (4, 1)
        assert dm_combined['a'].to_list() == [1, 2, 3, 4]

    def test_vertical_append_separates_polynomials_automatically(self):
        """Core multi-run feature: polynomials auto-separated"""
        # Run 1
        dm1 = DesignMatrix({'stim': [1, 0, 0, 0]}, sampling_freq=1)
        dm1 = dm1.add_poly(order=0)  # Intercept

        # Run 2
        dm2 = DesignMatrix({'stim': [0, 1, 0, 0]}, sampling_freq=1)
        dm2 = dm2.add_poly(order=0)

        # Append with keep_separate=True
        dm_runs = dm1.append(dm2, axis=0, keep_separate=True)

        # Should have: stim (shared), 0_poly_0, 1_poly_0 (separated)
        assert dm_runs.shape == (8, 3)
        assert 'stim' in dm_runs.columns
        assert '0_poly_0' in dm_runs.columns
        assert '1_poly_0' in dm_runs.columns
        assert dm_runs.multi == True

        # Check separation: run1 intercept active in first 4 rows
        assert dm_runs['0_poly_0'][:4].sum() > 0
        assert dm_runs['0_poly_0'][4:].sum() == 0
        assert dm_runs['1_poly_0'][:4].sum() == 0
        assert dm_runs['1_poly_0'][4:].sum() > 0

    def test_vertical_append_unique_cols_exact_match(self):
        """Keep specific columns separated across runs"""
        dm1 = DesignMatrix({
            'motion_x': [1, 2],
            'motion_y': [3, 4],
            'stim': [1, 0]
        }, sampling_freq=1)

        dm2 = DesignMatrix({
            'motion_x': [5, 6],
            'motion_y': [7, 8],
            'stim': [0, 1]
        }, sampling_freq=1)

        dm_runs = dm1.append(dm2, axis=0, unique_cols=['motion_x', 'motion_y'])

        # Motion columns separated, stim shared
        assert 'stim' in dm_runs.columns
        assert '0_motion_x' in dm_runs.columns
        assert '0_motion_y' in dm_runs.columns
        assert '1_motion_x' in dm_runs.columns
        assert '1_motion_y' in dm_runs.columns

    def test_vertical_append_unique_cols_wildcard_prefix(self):
        """Wildcard matching: 'house*' matches house_A, house_B"""
        dm1 = DesignMatrix({
            'house_A': [1, 0],
            'house_B': [0, 1],
            'face_A': [1, 1]
        }, sampling_freq=1)

        dm2 = DesignMatrix({
            'house_A': [0, 1],
            'house_B': [1, 0],
            'face_A': [1, 1]
        }, sampling_freq=1)

        dm_runs = dm1.append(dm2, axis=0, unique_cols=['house*'])

        # House columns separated by wildcard
        assert '0_house_A' in dm_runs.columns
        assert '0_house_B' in dm_runs.columns
        assert '1_house_A' in dm_runs.columns
        assert '1_house_B' in dm_runs.columns
        # Face shared
        assert 'face_A' in dm_runs.columns
        assert '0_face_A' not in dm_runs.columns

    def test_vertical_append_unique_cols_wildcard_suffix(self):
        """Wildcard matching: '*_motion' matches x_motion, y_motion"""
        dm1 = DesignMatrix({
            'x_motion': [1, 2],
            'y_motion': [3, 4]
        }, sampling_freq=1)

        dm2 = DesignMatrix({
            'x_motion': [5, 6],
            'y_motion': [7, 8]
        }, sampling_freq=1)

        dm_runs = dm1.append(dm2, axis=0, unique_cols=['*_motion'])

        assert '0_x_motion' in dm_runs.columns
        assert '1_y_motion' in dm_runs.columns

    def test_vertical_append_multiple_runs_increments_numbering(self):
        """Appending 3+ runs should increment numbering correctly"""
        dm1 = DesignMatrix({'s': [1]}, sampling_freq=1).add_poly(0)
        dm2 = DesignMatrix({'s': [2]}, sampling_freq=1).add_poly(0)
        dm3 = DesignMatrix({'s': [3]}, sampling_freq=1).add_poly(0)

        dm_runs = dm1.append(dm2, axis=0).append(dm3, axis=0)

        assert '0_poly_0' in dm_runs.columns
        assert '1_poly_0' in dm_runs.columns
        assert '2_poly_0' in dm_runs.columns

    def test_vertical_append_fill_na(self):
        """Mismatched columns should fill with specified value"""
        dm1 = DesignMatrix({'a': [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({'b': [3, 4]}, sampling_freq=1)

        dm_combined = dm1.append(dm2, axis=0, fill_na=0)

        assert dm_combined.shape == (4, 2)
        # First 2 rows: a=1,2 b=0
        # Last 2 rows: a=0 b=3,4
        assert dm_combined['a'].to_list() == [1, 2, 0, 0]
        assert dm_combined['b'].to_list() == [0, 0, 3, 4]
```

#### 1.7 Diagnostic Tests
```python
class TestDesignMatrixDiagnostics:
    """Test diagnostic utilities"""

    def test_vif_computes_variance_inflation_factor(self):
        """VIF should detect collinearity"""
        # Create correlated columns
        dm = DesignMatrix({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],  # Perfectly correlated with a
            'c': [1, 1, 2, 2, 3]    # Moderately correlated
        }, sampling_freq=1)

        vifs = dm.vif(exclude_polys=True)

        # VIF > 10 indicates problematic collinearity
        # b should have very high VIF (corr with a = 1.0)
        assert vifs[1] > 10  # Index 1 = column 'b'

    def test_clean_removes_highly_correlated_columns(self):
        """Clean should drop columns correlated >= threshold"""
        dm = DesignMatrix({
            'a': [1, 2, 3, 4],
            'b': [1.01, 2.01, 3.01, 4.01],  # r ≈ 1.0 with a
            'c': [4, 3, 2, 1]  # Uncorrelated
        }, sampling_freq=1)

        dm_clean = dm.clean(thresh=0.95)

        # Should drop 'b' (correlated with 'a')
        assert 'a' in dm_clean.columns
        assert 'b' not in dm_clean.columns
        assert 'c' in dm_clean.columns

    def test_clean_excludes_polys_from_check(self):
        """Polynomials can be excluded from collinearity check"""
        dm = DesignMatrix({'a': [1, 2, 3, 4]}, sampling_freq=1)
        dm = dm.add_poly(order=2)  # Polys might be correlated

        dm_clean = dm.clean(exclude_polys=True)

        # Polynomial columns should be retained
        assert 'poly_0' in dm_clean.columns
        assert 'poly_1' in dm_clean.columns
        assert 'poly_2' in dm_clean.columns
```

#### 1.8 Utility Tests
```python
class TestDesignMatrixUtilities:
    """Test misc utilities"""

    def test_details_shows_metadata(self):
        dm = DesignMatrix({'a': [1, 2, 3]}, sampling_freq=2)
        dm = dm.add_poly(0)
        dm = dm.convolve()

        details = dm.details()

        assert 'sampling_freq=2' in details
        assert 'shape=(3, 2)' in details
        assert 'poly_0' in details
        assert 'convolved' in details

    def test_replace_data_keeps_metadata_and_polys(self):
        """Replace data but preserve polynomial columns and metadata"""
        dm = DesignMatrix({'a': [1, 2, 3], 'b': [4, 5, 6]}, sampling_freq=2)
        dm = dm.add_poly(order=0)

        new_data = np.array([[10, 20], [30, 40], [50, 60]])
        dm_replaced = dm.replace_data(new_data, column_names=['x', 'y'])

        # New data columns
        assert dm_replaced.columns == ['x', 'y', 'poly_0']
        assert dm_replaced['x'].to_list() == [10, 30, 50]
        # Metadata preserved
        assert dm_replaced.sampling_freq == 2
        # Polynomials preserved
        assert dm_replaced['poly_0'].to_list() == dm['poly_0'].to_list()
```

**Estimated Phase 1**: 2 hours to write all tests

---

### Phase 2: TDD Implementation (12 hours)

For each test category, follow TDD cycle:
1. Run tests (should fail)
2. Implement minimal code to pass
3. Refactor using idiomatic Polars
4. Move to next test

**Implementation order:**

#### 2.1 Construction (1 hour)
- Implement `__init__()` handling all input types
- Ensure column names are strings
- Metadata initialization

**Idiomatic Polars patterns:**
```python
# Creating from numpy
if isinstance(data, np.ndarray):
    self._df = pl.DataFrame(data, schema=columns or [str(i) for i in range(ncols)])

# Creating from dict
elif isinstance(data, dict):
    self._df = pl.DataFrame(data)

# Ensure string column names
self._df = self._df.rename({col: str(col) for col in self._df.columns})
```

#### 2.2 Data Access (1 hour)
- `__getitem__()` and `__setitem__()`
- Properties: `shape`, `columns`, `empty`
- `_copy_with()` helper

**Idiomatic Polars patterns:**
```python
def __setitem__(self, key, value):
    # Broadcast scalar
    if isinstance(value, (int, float)):
        self._df = self._df.with_columns(pl.lit(value).alias(key))
    # Array-like
    elif isinstance(value, (list, np.ndarray)):
        self._df = self._df.with_columns(pl.Series(key, value))
```

#### 2.3 Simple Transforms (2 hours)
- `fillna()`, `drop()`, `corr()`
- `_copy_with()` pattern throughout

**Idiomatic Polars patterns:**
```python
def fillna(self, value):
    return self._copy_with(self._df.fill_nan(value))

def drop(self, columns):
    return self._copy_with(self._df.drop(columns))
```

#### 2.4 Stats Utilities (2 hours)
- Port `zscore()`, `downsample()`, `upsample()` from stats.py
- Use Polars expressions

**Idiomatic Polars patterns:**
```python
def zscore(self, columns=None):
    cols = columns or self.columns
    cols = [c for c in cols if c not in self.polys]

    z_exprs = [
        ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
        for c in cols
    ]
    other_cols = [c for c in self.columns if c not in cols]

    new_df = self._df.select(z_exprs + [pl.col(c) for c in other_cols])
    return self._copy_with(new_df)
```

#### 2.5 Domain Methods (3 hours)
- `convolve()` - HRF convolution
- `add_poly()` - Legendre polynomials
- `add_dct_basis()` - DCT basis

**Idiomatic Polars patterns:**
```python
def convolve(self, conv_func='hrf', columns=None):
    # Get columns to convolve (exclude polys)
    if columns is None:
        columns = [c for c in self.columns if c not in self.polys]

    # Get or create kernel
    if conv_func == 'hrf':
        kernel = glover_hrf(1.0 / self.sampling_freq, oversampling=1.0)
    else:
        kernel = conv_func

    # Convolve each column using Polars apply
    conv_exprs = [
        pl.col(c).map_elements(
            lambda x: np.convolve(x, kernel)[:len(self._df)],
            return_dtype=pl.Float64
        ).alias(c)
        for c in columns
    ]

    # Keep non-convolved columns unchanged
    other_cols = [c for c in self.columns if c not in columns]

    new_df = self._df.select(conv_exprs + [pl.col(c) for c in other_cols])
    return self._copy_with(new_df, convolved=columns)
```

#### 2.6 Append Logic (2 hours) **MOST COMPLEX**

**Strategy**: Test-driven approach for complex logic
1. Start with horizontal append (simpler)
2. Then vertical without separation
3. Then vertical with polynomial separation
4. Finally wildcard support

**Idiomatic Polars patterns:**
```python
def append(self, dm, axis=0, keep_separate=True, unique_cols=None, fill_na=0):
    if axis == 1:
        return self._horizontal_concat(dm, fill_na)
    else:
        return self._vertical_concat(dm, keep_separate, unique_cols, fill_na)

def _horizontal_concat(self, dm, fill_na):
    """Simple horizontal concatenation"""
    new_df = pl.concat([self._df, dm._df], how='horizontal')
    if fill_na is not None:
        new_df = new_df.fill_nan(fill_na)

    return self._copy_with(
        new_df,
        polys=self.polys + dm.polys
    )

def _vertical_concat(self, dm, keep_separate, unique_cols, fill_na):
    """Complex vertical concatenation with run separation"""
    # This is where the complex logic goes
    # Use Polars rename() for column numbering
    # Use pl.concat() for stacking
    # Pattern: build list of renamed DataFrames, concat once

    # Determine columns to separate
    cols_to_sep = set()
    if keep_separate:
        cols_to_sep.update(self.polys)  # Always separate polynomials

    if unique_cols:
        # Expand wildcards
        for pattern in unique_cols:
            if pattern.startswith('*'):
                suffix = pattern[1:]
                cols_to_sep.update([c for c in self.columns if c.endswith(suffix)])
            elif pattern.endswith('*'):
                prefix = pattern[:-1]
                cols_to_sep.update([c for c in self.columns if c.startswith(prefix)])
            else:
                cols_to_sep.add(pattern)

    # Rename columns in both DataFrames
    df1 = self._rename_for_multi_run(self._df, cols_to_sep, run_idx=0)
    df2 = dm._rename_for_multi_run(dm._df, cols_to_sep, run_idx=1)

    # Concatenate
    new_df = pl.concat([df1, df2], how='diagonal')  # Fills missing with null
    if fill_na is not None:
        new_df = new_df.fill_null(fill_na)

    return self._copy_with(new_df, multi=True)
```

#### 2.7 Diagnostics (1 hour)
- `vif()` - Use correlation matrix
- `clean()` - Drop correlated columns

**Pragmatic approach**: Use pandas for correlation
```python
def corr(self):
    """Compute correlation matrix (uses pandas internally)"""
    return self._df.to_pandas().corr()

def vif(self, exclude_polys=True):
    """Variance inflation factor"""
    cols = [c for c in self.columns if c not in self.polys] if exclude_polys else self.columns

    if len(cols) <= 1:
        raise ValueError("Need at least 2 columns for VIF")

    # Use correlation matrix approach (like R/Matlab)
    corr_matrix = self._df.select(cols).to_pandas().corr()
    vifs = np.diag(np.linalg.inv(corr_matrix))
    return vifs
```

**Estimated Phase 2**: 12 hours

---

### Phase 3: Integration (4 hours)

Once DesignMatrix passes all tests in isolation:

#### 3.1 Remove Brain_Data.X and Brain_Data.Y (1 hour)
```python
# nltools/data/brain_data.py

class Brain_Data:
    def __init__(self, ...):
        # REMOVE these lines:
        # self.X = None
        # self.Y = None

        # Update docstring to remove X/Y references
```

**Search for all X/Y usage:**
```bash
uv run rg "\.X\b|\.Y\b" nltools/data/brain_data.py
```

**Replace with:** Direct `design_matrix` parameter in `.fit()`

#### 3.2 Update file_reader.py (1 hour)
```python
# nltools/file_reader.py

def onsets_to_dm(...):
    # Change return type from Design_Matrix to DesignMatrix
    return DesignMatrix(...)  # Use new implementation
```

#### 3.3 Rename and Replace (1 hour)
```bash
# Backup old implementation
mv nltools/data/design_matrix.py nltools/data/design_matrix_old.py

# Rename new implementation
mv nltools/data/design_matrix_new.py nltools/data/design_matrix.py

# Update class name in new file
# DesignMatrix → Design_Matrix (preserve public API name)
```

#### 3.4 Integration Tests (1 hour)
```python
def test_brain_data_fit_with_design_matrix():
    """Test that new Design_Matrix works with Brain_Data.fit()"""
    brain = Brain_Data(...)
    dm = Design_Matrix({'stim': [...]}, sampling_freq=2)
    dm = dm.convolve().add_poly(order=1)

    brain.fit(design_matrix=dm, model='ridge')
    # Should work without errors
```

**Estimated Phase 3**: 4 hours

---

### Phase 4: Cleanup (2 hours)

#### 4.1 Update Dependencies
```toml
# pyproject.toml
[project.dependencies]
polars = ">=0.20.0"
# pandas still needed by nilearn (transitive)
```

#### 4.2 Update Migration Guide
Document behavior changes:
- `.loc[]` removed - use Polars syntax
- X/Y attributes removed - use `design_matrix` parameter

#### 4.3 Mark Adjacency for Future Work
```python
# nltools/tests/shell/test_adjacency.py
@pytest.mark.skip(reason="Adjacency Polars migration deferred")
def test_adjacency_with_polars():
    pass
```

**Estimated Phase 4**: 2 hours

---

## Part 4: Key Implementation Patterns

### Pattern 1: Immutable Transformations
```python
def any_transformation(self, ...):
    """All methods return NEW DesignMatrix, never modify self"""
    new_df = self._df.with_columns(...)  # Polars operation
    return self._copy_with(new_df, **metadata_updates)
```

### Pattern 2: Wildcard Matching
```python
def _match_wildcard(pattern, columns):
    """Expand wildcard pattern to matching columns"""
    if pattern.startswith('*'):
        suffix = pattern[1:]
        return [c for c in columns if c.endswith(suffix)]
    elif pattern.endswith('*'):
        prefix = pattern[:-1]
        return [c for c in columns if c.startswith(prefix)]
    else:
        return [pattern] if pattern in columns else []
```

### Pattern 3: Metadata Copying
```python
def _copy_with(self, new_df, **metadata_updates):
    """Create new instance with updated data/metadata"""
    new = DesignMatrix.__new__(DesignMatrix)
    new._df = new_df
    new.sampling_freq = metadata_updates.get('sampling_freq', self.sampling_freq)
    new.convolved = metadata_updates.get('convolved', self.convolved.copy())
    new.polys = metadata_updates.get('polys', self.polys.copy())
    new.multi = metadata_updates.get('multi', self.multi)
    return new
```

---

## Part 5: Success Criteria

- ✅ All new tests pass (50+ tests)
- ✅ No pandas in design_matrix.py (except for .corr() helper)
- ✅ Brain_Data.X and Brain_Data.Y removed
- ✅ Tutorials can be updated to idiomatic Polars (manual task)
- ✅ No regressions in existing Brain_Data functionality

---

## Part 6: Timeline Summary

| Phase | Hours | Description |
|-------|-------|-------------|
| 1 | 2 | Write comprehensive tests (behavior specs) |
| 2.1-2.3 | 4 | Construction, access, simple transforms |
| 2.4-2.5 | 5 | Stats utils, domain methods |
| 2.6-2.7 | 3 | Append logic, diagnostics |
| 3 | 4 | Integration (remove X/Y, swap implementations) |
| 4 | 2 | Cleanup, docs, dependencies |
| **Total** | **20** | Focused TDD implementation |

---

## Part 7: What We're NOT Doing (Deferred)

- ❌ Lazy evaluation (save for later optimization)
- ❌ Adjacency Polars migration (separate effort)
- ❌ Tutorial rewrites (manual task after implementation)
- ❌ Polars GPU support (v0.7.0)
- ❌ `.loc[]` compatibility layer (dropped per decision)

---

## Next Steps

1. **Create test file**: `nltools/tests/shell/test_design_matrix_new.py`
2. **Write all tests** (Phase 1)
3. **Watch them fail** (satisfying red state)
4. **Implement to green** (TDD cycle)
5. **Integration & cleanup**

**Ready to start?** Let me know and I'll begin with Phase 1 (test design).

---

**Last Updated**: 2025-10-29
**Status**: Ready for implementation
**Approach**: TDD, behavior-focused, idiomatic Polars
