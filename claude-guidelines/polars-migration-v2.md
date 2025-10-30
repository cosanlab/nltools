# Pandas → Polars Migration Plan (Revised v2)
**Date**: 2025-10-29
**Status**: Ready for implementation
**Approach**: Clean break with compatibility layer for pandas-like syntax

---

## Executive Summary

**Goal**: Complete pandas removal while preserving user-facing API
**Strategy**: Composition pattern with compatibility layer for pandas idioms
**Scope**: DesignMatrix (primary), Brain_Data.X/Y, Adjacency.Y, stats.py utilities
**Timeline**: ~20-24 hours of focused work

---

## Part 1: User-Facing API Requirements (from tutorials)

### Critical Patterns That MUST Work

**1. DataFrame-like construction:**
```python
# From numpy array
dm = DesignMatrix(np.zeros((100, 1)), sampling_freq=2, columns=['a'])

# From pandas DataFrame (backward compat)
dm = DesignMatrix(pd.read_csv('file.csv'), sampling_freq=2)

# Empty initialization
dm = DesignMatrix(sampling_freq=2)
```

**2. Column access and assignment:**
```python
dm['ConditionA'] = 0              # Add/set column
dm['ConditionA']                  # Get column (Series-like)
dm[['col1', 'col2']]              # Get multiple columns
```

**3. Pandas-style indexing (CRITICAL):**
```python
dm.loc[10:15, 'ConditionA'] = 1   # Row-column assignment
dm.loc[10:15]                     # Row slicing
```

**4. DataFrame operations:**
```python
dm.drop(columns=[0])              # Drop columns
dm.fillna(0)                      # Fill NaNs
dm.max().max()                    # Chained aggregations
dm.columns                        # Column names
dm.shape                          # Shape tuple
```

**5. Custom DesignMatrix methods:**
```python
dm.convolve()                     # HRF convolution
dm.add_poly(order=2)              # Add polynomial regressors
dm.add_dct_basis(duration=180)    # Add DCT basis functions
dm.append(dm2, axis=0)            # Vertical concatenation
dm.append(dm2, axis=1)            # Horizontal concatenation
dm.downsample(target=0.5)         # Temporal downsampling
dm.upsample(target=2)             # Temporal upsampling
dm.zscore(columns=[])             # Z-score normalization
dm.vif()                          # Variance inflation factor
dm.clean()                        # Remove correlated columns
dm.heatmap()                      # Visualization
dm.details()                      # Metadata display
```

**6. Matplotlib integration:**
```python
ax.plot(dm['ConditionA'])         # Plot column directly
```

**7. Metadata preservation:**
```python
dm.sampling_freq                  # Sampling frequency in Hz
dm.convolved                      # List of convolved columns
dm.polys                          # List of polynomial columns
dm.multi                          # Multi-run flag
```

---

## Part 2: Technical Architecture

### Composition Pattern with Compatibility Layer

```python
import polars as pl
import numpy as np
from typing import Union, List, Optional

class _LocIndexer:
    """Pandas-style .loc[] indexer for Polars-backed DesignMatrix"""

    def __init__(self, dm):
        self._dm = dm

    def __getitem__(self, key):
        """
        Support patterns:
        - dm.loc[10:15, 'col']  # Row slice, single column
        - dm.loc[10:15]         # Row slice, all columns
        - dm.loc[:, 'col']      # All rows, single column
        """
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        # Handle row indexing
        if isinstance(row_key, slice):
            start = row_key.start or 0
            stop = row_key.stop or len(self._dm._df)
            df_subset = self._dm._df.slice(start, stop - start)
        elif isinstance(row_key, int):
            df_subset = self._dm._df.slice(row_key, 1)
        else:
            raise NotImplementedError(f"Row indexing type {type(row_key)} not supported")

        # Handle column indexing
        if col_key == slice(None):
            result_df = df_subset
        elif isinstance(col_key, str):
            result_df = df_subset.select(col_key)
        elif isinstance(col_key, list):
            result_df = df_subset.select(col_key)
        else:
            raise NotImplementedError(f"Column indexing type {type(col_key)} not supported")

        # Return DesignMatrix if multiple columns, Series-like if single column
        if result_df.width == 1:
            return result_df.to_series()
        else:
            return self._dm._from_polars(result_df, self._dm._get_metadata())

    def __setitem__(self, key, value):
        """
        Support pattern: dm.loc[10:15, 'col'] = 1
        This is the CRITICAL pattern from tutorials
        """
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            raise ValueError("Use dm.loc[rows, col] = value syntax")

        if not isinstance(col_key, str):
            raise NotImplementedError("Only single column assignment supported")

        # Handle row indexing
        if isinstance(row_key, slice):
            start = row_key.start or 0
            stop = row_key.stop or len(self._dm._df)
        else:
            raise NotImplementedError(f"Row indexing type {type(row_key)} not supported")

        # Create a column with values set at specified rows
        # This is tricky with Polars - need to use when-then-otherwise
        col_expr = pl.when(
            (pl.int_range(0, pl.len()) >= start) &
            (pl.int_range(0, pl.len()) < stop)
        ).then(pl.lit(value)).otherwise(pl.col(col_key))

        # Update the column (or create if doesn't exist)
        if col_key in self._dm._df.columns:
            self._dm._df = self._dm._df.with_columns(col_expr.alias(col_key))
        else:
            # Create new column initialized to 0, then set values
            self._dm._df = self._dm._df.with_columns(pl.lit(0).alias(col_key))
            self._dm._df = self._dm._df.with_columns(col_expr.alias(col_key))


class DesignMatrix:
    """
    Polars-based design matrix with neuroimaging-specific methods.

    Wraps pl.DataFrame internally while providing pandas-like API for compatibility.

    Parameters
    ----------
    data : DataFrame, ndarray, dict, or None
        Input data
    sampling_freq : float, optional
        Sampling frequency in hertz (1/TR for fMRI)
    columns : list of str, optional
        Column names
    convolved : list of str, optional
        Names of convolved columns
    polys : list of str, optional
        Names of polynomial columns
    """

    _metadata = ["sampling_freq", "convolved", "polys", "multi"]

    def __init__(self, data=None, sampling_freq=None, columns=None,
                 convolved=None, polys=None, **kwargs):
        # Metadata
        self.sampling_freq = sampling_freq
        self.convolved = convolved if convolved is not None else []
        self.polys = polys if polys is not None else []
        self.multi = False

        # Create internal Polars DataFrame
        if data is None:
            # Empty initialization: DesignMatrix(sampling_freq=2)
            self._df = pl.DataFrame()
        elif isinstance(data, pl.DataFrame):
            self._df = data
        elif isinstance(data, pd.DataFrame):
            # Backward compatibility with pandas
            self._df = pl.from_pandas(data)
        elif isinstance(data, np.ndarray):
            if columns is not None:
                self._df = pl.DataFrame(data, schema=columns)
            else:
                # Auto-generate column names: 0, 1, 2, ...
                n_cols = data.shape[1] if len(data.shape) > 1 else 1
                self._df = pl.DataFrame(data, schema=[str(i) for i in range(n_cols)])
        elif isinstance(data, dict):
            self._df = pl.DataFrame(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Ensure all column names are strings (critical for consistency)
        if not self._df.is_empty():
            self._df = self._df.rename({col: str(col) for col in self._df.columns})

    # ==================== Compatibility Layer ====================

    @property
    def loc(self):
        """Pandas-style .loc[] indexer"""
        return _LocIndexer(self)

    def __getitem__(self, key):
        """
        Support patterns:
        - dm['col']           # Single column (returns Series-like)
        - dm[['col1', 'col2']] # Multiple columns (returns DesignMatrix)
        """
        if isinstance(key, str):
            # Single column - return Polars Series
            return self._df[key]
        elif isinstance(key, list):
            # Multiple columns - return DesignMatrix
            return self._from_polars(self._df.select(key), self._get_metadata())
        else:
            raise KeyError(f"Unsupported key type: {type(key)}")

    def __setitem__(self, key, value):
        """
        Support pattern: dm['col'] = value
        """
        if isinstance(key, str):
            # Single column assignment
            if isinstance(value, (int, float)):
                # Scalar: broadcast to all rows
                self._df = self._df.with_columns(pl.lit(value).alias(key))
            elif isinstance(value, (list, np.ndarray)):
                # Array-like: must match length
                self._df = self._df.with_columns(pl.Series(key, value))
            elif isinstance(value, pl.Series):
                self._df = self._df.with_columns(value.alias(key))
            else:
                raise TypeError(f"Unsupported value type: {type(value)}")
        else:
            raise KeyError(f"Unsupported key type: {type(key)}")

    # ==================== DataFrame-like Properties ====================

    @property
    def shape(self):
        """Return (n_rows, n_cols) tuple"""
        return self._df.shape

    @property
    def columns(self):
        """Return column names"""
        return self._df.columns

    @columns.setter
    def columns(self, new_names):
        """Set column names"""
        rename_dict = {old: new for old, new in zip(self._df.columns, new_names)}
        self._df = self._df.rename(rename_dict)

    @property
    def empty(self):
        """Check if DesignMatrix is empty"""
        return self._df.is_empty()

    def __len__(self):
        """Return number of rows"""
        return len(self._df)

    # ==================== DataFrame-like Methods ====================

    def drop(self, columns=None, **kwargs):
        """Drop columns (pandas-compatible API)"""
        if columns is None:
            raise ValueError("Must specify columns to drop")
        new_df = self._df.drop(columns)
        return self._from_polars(new_df, self._get_metadata())

    def fillna(self, value):
        """Fill NaN values"""
        new_df = self._df.fill_nan(value)
        return self._from_polars(new_df, self._get_metadata())

    def max(self):
        """Return max of each column (returns Series-like for .max().max() pattern)"""
        # Return a special object that supports .max() chaining
        class MaxResult:
            def __init__(self, df):
                self._max_values = df.max()

            def max(self):
                # Get max across all columns
                return max([self._max_values[col][0] for col in self._max_values.columns])

        return MaxResult(self._df)

    def corr(self):
        """Compute correlation matrix (for .vif() method)"""
        # Polars doesn't have .corr() method, need to compute manually
        # Convert to pandas temporarily for correlation
        return self._df.to_pandas().corr()

    # ==================== Internal Helpers ====================

    def _get_metadata(self):
        """Extract metadata as dict"""
        return {
            'sampling_freq': self.sampling_freq,
            'convolved': self.convolved.copy(),
            'polys': self.polys.copy(),
            'multi': self.multi
        }

    @classmethod
    def _from_polars(cls, df, metadata=None):
        """Create DesignMatrix from Polars DataFrame, preserving metadata"""
        dm = cls.__new__(cls)
        dm._df = df
        if metadata:
            for k, v in metadata.items():
                setattr(dm, k, v)
        else:
            dm.sampling_freq = None
            dm.convolved = []
            dm.polys = []
            dm.multi = False
        return dm

    def _inherit_attributes(self, dm_out, atts=None):
        """Copy attributes from self to dm_out"""
        if atts is None:
            atts = self._metadata
        for item in atts:
            setattr(dm_out, item, getattr(self, item))
        return dm_out

    # ==================== Custom Methods (preserve existing logic) ====================

    def details(self):
        """Return string representation of metadata"""
        return (f"{self.__class__.__module__}.{self.__class__.__name__}"
                f"(sampling_freq={self.sampling_freq} (hz), "
                f"shape={self.shape}, multi={self.multi}, "
                f"convolved={self.convolved}, polynomials={self.polys})")

    # NOTE: All other methods (convolve, add_poly, append, etc.)
    # will be migrated from existing implementation with minimal changes
    # The key is they operate on self._df (Polars) instead of self (pandas)
```

### Key Design Decisions

1. **`.loc[]` compatibility**: Critical pattern from tutorials. Implemented via `_LocIndexer` class
2. **Column assignment**: `dm['col'] = value` works via `__setitem__`
3. **Metadata preservation**: All methods return new DesignMatrix with copied metadata
4. **Polars expressions**: Use `pl.when().then().otherwise()` for conditional updates
5. **Backward compat**: Accept pandas DataFrame in constructor, convert to Polars internally

---

## Part 3: Implementation Plan

### Phase 1: Core DesignMatrix Implementation (~8 hours)

**1.1 Create new file structure:**
```bash
nltools/data/design_matrix_polars.py  # New implementation
nltools/data/design_matrix_old.py     # Rename current for reference
nltools/data/design_matrix.py         # Will be replaced
```

**1.2 Implement base class:**
- [ ] `__init__()` with all input types
- [ ] `.loc` property and `_LocIndexer` class
- [ ] `__getitem__()` and `__setitem__()`
- [ ] Properties: `.shape`, `.columns`, `.empty`
- [ ] Methods: `.drop()`, `.fillna()`, `.max()`, `.corr()`
- [ ] Metadata helpers: `_get_metadata()`, `_from_polars()`, `_inherit_attributes()`

**1.3 Port existing methods (preserve logic, swap pandas→polars):**
- [ ] `.convolve()` - HRF convolution
- [ ] `.add_poly()` - Legendre polynomials
- [ ] `.add_dct_basis()` - DCT basis functions
- [ ] `.append()` - Smart concatenation (complex logic!)
- [ ] `.vif()` - Variance inflation factor
- [ ] `.heatmap()` - Visualization
- [ ] `.downsample()` - Temporal downsampling
- [ ] `.upsample()` - Temporal upsampling
- [ ] `.zscore()` - Z-score normalization
- [ ] `.clean()` - Remove correlated columns
- [ ] `.replace_data()` - Replace data
- [ ] `._sort_cols()` - Internal sorting helper
- [ ] `._horzcat()` - Horizontal concatenation helper
- [ ] `._vertcat()` - Vertical concatenation helper (very complex!)

**Estimated**: 8 hours (`.append()` is complex, needs careful porting)

---

### Phase 2: Stats Utilities Migration (~3 hours)

**File**: `nltools/stats.py`

Migrate functions to accept/return Polars DataFrames:

**2.1 Functions to migrate:**
- [ ] `zscore()` - Use Polars expressions `(pl.col() - mean) / std`
- [ ] `downsample()` - Use `.group_by().agg()`
- [ ] `upsample()` - Use `.upsample()` method

**2.2 Pattern:**
```python
# Before (pandas)
def zscore(df):
    return (df - df.mean()) / df.std()

# After (polars)
def zscore(df):
    if isinstance(df, pl.DataFrame):
        return df.select([
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
            for col in df.columns
        ])
    elif isinstance(df, pl.Series):
        return (df - df.mean()) / df.std()
    else:
        # Fallback for numpy
        return (df - np.mean(df)) / np.std(df)
```

**Estimated**: 3 hours

---

### Phase 3: Brain_Data and Adjacency Integration (~4 hours)

**3.1 Brain_Data.X and Brain_Data.Y:**

Current usage (deprecated in v0.6.0 anyway):
- `Brain_Data.X` stores pandas DataFrame
- Used in old `.regress()` method (now deprecated)

**Action**: Remove X/Y attributes entirely OR keep as compatibility layer

**Recommendation**: Remove. They're deprecated and new `.fit()` method uses `design_matrix` parameter.

**3.2 Adjacency.Y:**

Current usage:
- `Adjacency.Y` stores pandas DataFrame with edge labels
- Used in statistical methods

**Migration**:
```python
# In adjacency.py
class Adjacency:
    def __init__(self, ...):
        self._Y = None  # Store as Polars internally

    @property
    def Y(self):
        """Return Polars DataFrame"""
        return self._Y

    @Y.setter
    def Y(self, value):
        if isinstance(value, pd.DataFrame):
            self._Y = pl.from_pandas(value)
        elif isinstance(value, pl.DataFrame):
            self._Y = value
        else:
            raise TypeError("Y must be a DataFrame")
```

**Update Adjacency methods**:
- [ ] `distance_stats()` - Use Polars `.filter()` instead of `.loc[]`
- [ ] `ttest()` - Use Polars `.group_by()`
- [ ] `plot_silhouette()` - Convert to pandas only for plotting

**Estimated**: 4 hours

---

### Phase 4: File I/O and Utilities (~2 hours)

**4.1 File reader:**
- [ ] `onsets_to_dm()` - Return DesignMatrix with Polars
- [ ] CSV reading - Use `pl.read_csv()` instead of `pd.read_csv()`

**4.2 Validation:**
- [ ] `nltools/data/_validation.py` - Update DataFrame checks

**Estimated**: 2 hours

---

### Phase 5: Test Suite Updates (~5 hours)

**5.1 Update test fixtures:**
```python
# nltools/tests/conftest.py
@pytest.fixture
def design_matrix():
    import polars as pl
    df = pl.DataFrame({
        'stim_A': [0, 1, 1, 0],
        'stim_B': [1, 0, 0, 1]
    })
    return DesignMatrix(df, sampling_freq=2.0)
```

**5.2 Update DesignMatrix tests:**
- [ ] `test_design_matrix.py` - All 10+ tests
- [ ] Update to use Polars patterns
- [ ] Add new tests for `.loc[]` compatibility

**5.3 Update integration tests:**
- [ ] Brain_Data tests that use DesignMatrix
- [ ] Adjacency tests
- [ ] Stats tests

**Estimated**: 5 hours

---

### Phase 6: Dependencies and Cleanup (~2 hours)

**6.1 Update pyproject.toml:**
```toml
[project.dependencies]
polars = ">=0.20.0"
# Remove pandas as direct dependency (still transitive via nilearn)
```

**6.2 Clean up imports:**
- [ ] Remove `import pandas as pd` from all files
- [ ] Add `import polars as pl` where needed

**6.3 Update migration guide:**
- [ ] Document API changes
- [ ] Provide migration examples
- [ ] Note performance improvements

**Estimated**: 2 hours

---

## Part 4: Critical Implementation Details

### Handling `.append()` Complexity

The `.append()` method is VERY complex (lines 110-456 in current design_matrix.py). It handles:
- Vertical and horizontal concatenation
- Automatic polynomial separation across runs
- Custom column separation with wildcards
- Column renaming for multi-run tracking

**Strategy**:
1. Port logic line-by-line, preserving semantics
2. Replace pandas operations with Polars equivalents:
   - `pd.concat()` → `pl.concat()`
   - `.loc[]` → `.filter()` + `.select()`
   - `.rename()` → `.rename()`
3. Test extensively with multi-run workflows

### Handling Correlation Matrix

Polars doesn't have built-in `.corr()` method. Options:
1. **Use pandas temporarily**: Convert to pandas for `.corr()`, then back
2. **Implement manually**: Use Polars expressions
3. **Use numpy**: Convert to numpy, compute corr, create Polars DataFrame

**Recommendation**: Option 1 for simplicity (`.vif()` and `.clean()` are not performance-critical)

### Matplotlib Integration

Need to ensure `ax.plot(dm['ConditionA'])` works.

**Solution**: Return Polars Series from `dm['col']`, which matplotlib can handle (it uses `__array__()` protocol).

Test:
```python
import matplotlib.pyplot as plt
import polars as pl

df = pl.DataFrame({'a': [1, 2, 3, 4]})
plt.plot(df['a'])  # Should work
```

---

## Part 5: Testing Strategy

### Validation Criteria

1. **All existing tests pass**: 10+ DesignMatrix tests
2. **Tutorial code works unchanged**: Both tutorial notebooks run without modification
3. **Performance improvement**: Measure on `downsample()`, `append()`, `zscore()`
4. **Memory efficiency**: Check with large design matrices (10k rows, 100 cols)

### New Tests Needed

```python
def test_loc_indexer():
    """Test pandas-style .loc[] syntax"""
    dm = DesignMatrix(np.zeros((100, 2)), sampling_freq=2, columns=['a', 'b'])

    # Assignment
    dm.loc[10:15, 'a'] = 1
    assert dm._df['a'][10] == 1
    assert dm._df['a'][15] == 1
    assert dm._df['a'][16] == 0

    # Retrieval
    subset = dm.loc[10:20]
    assert subset.shape == (10, 2)

def test_polars_pandas_roundtrip():
    """Test backward compatibility with pandas"""
    pd_df = pd.DataFrame({'a': [1, 2, 3]})
    dm = DesignMatrix(pd_df, sampling_freq=2)
    assert isinstance(dm._df, pl.DataFrame)

    # Can export to pandas if needed
    pd_out = dm._df.to_pandas()
    pd.testing.assert_frame_equal(pd_df, pd_out)

def test_matplotlib_integration():
    """Test plotting works"""
    dm = DesignMatrix(np.array([[1], [2], [3]]), sampling_freq=1, columns=['a'])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(dm['a'])  # Should work without error
    plt.close(fig)
```

---

## Part 6: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `.loc[]` compatibility incomplete | Medium | High | Extensive testing, iterate on edge cases |
| `.append()` logic breaks | Medium | High | Port carefully, test multi-run workflows |
| Matplotlib incompatibility | Low | Medium | Test early, fallback to `.to_numpy()` if needed |
| Performance regression | Low | Low | Benchmark, Polars is typically faster |
| Missing Polars features | Medium | Medium | Use pandas fallback for corner cases |

---

## Part 7: Timeline

**Total**: ~24 hours (aggressive), ~30 hours (comfortable)

- **Phase 1**: Core DesignMatrix - 8 hours
- **Phase 2**: Stats utilities - 3 hours
- **Phase 3**: Integration - 4 hours
- **Phase 4**: File I/O - 2 hours
- **Phase 5**: Tests - 5 hours
- **Phase 6**: Cleanup - 2 hours

**Parallelization opportunities**: None significant (DesignMatrix is central)

---

## Part 8: Success Criteria

- ✅ All 317 existing tests pass
- ✅ Both tutorial notebooks run without changes
- ✅ No pandas imports in `nltools/data/design_matrix.py`
- ✅ `.loc[]` syntax works for all tutorial patterns
- ✅ Performance is same or better (benchmark on realistic workflows)
- ✅ Memory usage is same or lower
- ✅ Migration guide updated with examples

---

## Part 9: Post-Migration Benefits

1. **Performance**: Polars is 5-10x faster on large datasets
2. **Memory**: Polars uses less memory (Arrow format)
3. **Consistency**: Aligns with Eshin's preference (from background.md)
4. **Future-proof**: Polars GPU support in v0.7.0
5. **Cleaner code**: No pandas index management

---

**Last Updated**: 2025-10-29
**Status**: Ready for implementation
**Next Step**: Begin Phase 1 (Core DesignMatrix implementation)
