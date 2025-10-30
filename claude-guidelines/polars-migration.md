# Pandas → Polars Migration Plan for nltools v0.6.0

**Status**: Planning
**Last Updated**: 2025-10-28
**Target Release**: v0.6.0
**Coordinator**: Parallel with model-spec.md implementation

---

## Executive Summary

**Scope**: Complete pandas removal (CPU-only, defer GPU to v0.7.0)
**Target**: All 4 classes (Brain_Data, DesignMatrix, Adjacency, stats modules)
**Approach**: Parallel with Model class work, coordinated at integration points
**Timeline**: ~5-7 days of focused work

---

## 🎯 Coordination with model-spec.md

### Synergies (Mutual Benefits)

1. **Backend abstraction** (model-spec Phase 1)
   - Model spec implements `backends.py` for numpy/torch switching
   - **Polars benefit**: We can use the same pattern for Polars/numpy conversions
   - **Action**: Design our conversion layer to align with their Backend interface

2. **HDF5 deprecation** (model-spec mentions moving away from HDFStore)
   - Both plans move away from pandas HDFStore
   - **Shared solution**: Standardize on Parquet for DataFrames, h5py for arrays
   - **Action**: Coordinate file format migration (can reuse utility functions)

3. **Testing infrastructure**
   - Model spec adds `test_backends.py`, performance benchmarks
   - **Polars benefit**: Add similar benchmark suite for Polars operations
   - **Action**: Share testing patterns for CPU/Polars equivalence tests

### Potential Conflicts (Coordination Needed)

1. **Stats module dependencies**
   - Model spec's `ridge.py` may import from `stats.py`
   - **Risk**: If we change stats.py signatures before Model work
   - **Mitigation**: Migrate stats.py LAST, or coordinate API changes

2. **Brain_Data method signatures**
   - Model spec adds `.predict()` back to Brain_Data
   - **Risk**: If we change internal data structures before Model integration
   - **Mitigation**: Keep `brain_data.data` as numpy array (both plans need this)

3. **DesignMatrix in regression**
   - Model spec may use DesignMatrix in ridge regression
   - **Risk**: Breaking DesignMatrix API during Model implementation
   - **Mitigation**: Prioritize DesignMatrix migration early, freeze API

### Division of Labor

| Component | Polars Plan | Model Plan | Owner |
|-----------|-------------|------------|-------|
| `backends.py` | - | ✅ Implements | Model team |
| `stats.py` (resampling) | ✅ Migrates to Polars | - | Polars team |
| `stats/ridge.py` | - | ✅ New file | Model team |
| Brain_Data.X, Y | ✅ Polars conversion layer | Uses if present | Polars team |
| Brain_Data.predict() | - | ✅ Implements | Model team |
| DesignMatrix | ✅ Full rewrite | May consume | Polars team |
| HDF5 utilities | ✅ Migrate | May use | Shared |

---

## 📋 Research Summary: Pandas Usage in nltools

### Complete Inventory

**Files with pandas usage**: 15+ files
**Core classes using pandas**: 4 (Brain_Data, DesignMatrix, Adjacency, Validation)
**DataFrame attributes**:
- Brain_Data: `X`, `Y` (deprecated in v0.6.0)
- Adjacency: `Y`
- DesignMatrix: IS a DataFrame (subclasses pd.DataFrame)

### Critical Patterns Found

1. **Deprecated `.append()` calls** (3 instances in adjacency.py)
   - Lines 715, 987, 1038
   - Must replace with `pl.concat()`

2. **DesignMatrix subclassing**
   - Inherits ALL pandas methods
   - Has `_constructor` properties for slicing behavior
   - Most complex migration challenge

3. **Boolean indexing** (adjacency.py statistical methods)
   - Heavy use of `.loc[condition, column]`
   - Polars equivalent: `.filter().select()`

4. **GroupBy operations** (stats.py, adjacency.py)
   - Downsampling, aggregations
   - Polars: More explicit, better optimized

### Complexity Categories

**Simple** (1-2 hours each):
- CSV I/O: `pd.read_csv()` → `pl.read_csv()`
- Basic operations: `.shape`, `.empty`, `.columns`
- Type checking: `isinstance(x, pd.DataFrame)`

**Medium** (3-5 hours each):
- Indexing: `.iloc[]`, `.loc[]` → `.slice()`, `.filter()`
- Concatenation: `pd.concat()` → `pl.concat()`
- GroupBy: `.groupby().agg()` (similar but better syntax)

**Complex** (8-10 hours):
- DesignMatrix: Full class restructure needed
- Adjacency statistics: Optimize with lazy evaluation
- HDF5 I/O: Migrate from pandas HDFStore

---

## 📋 Tight Implementation Plan

### Phase 1: Foundations (Day 1, ~4 hours)
**Goal**: Set up conversion infrastructure, no breaking changes yet

**1.1 Polars/Pandas Bridge Module** (~2 hours)
- Create `nltools/utils/dataframe_bridge.py`
- Functions: `ensure_polars()`, `ensure_pandas()`, `to_polars()`, `to_pandas()`
- Coordinate with model-spec's `backends.py` pattern
- Add deprecation warnings for pandas inputs

```python
# nltools/utils/dataframe_bridge.py
import warnings
import pandas as pd
import polars as pl
from typing import Union

def ensure_polars(df: Union[pd.DataFrame, pl.DataFrame, None]) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars with deprecation warning."""
    if df is None:
        return pl.DataFrame()
    if isinstance(df, pd.DataFrame):
        warnings.warn(
            "pandas DataFrames are deprecated in nltools v0.6.0. "
            "Please use polars DataFrames. "
            "This will be an error in v0.7.0.",
            DeprecationWarning,
            stacklevel=2
        )
        return pl.from_pandas(df)
    return df

def ensure_pandas(df: Union[pd.DataFrame, pl.DataFrame, None]) -> pd.DataFrame:
    """Convert Polars DataFrame to pandas for legacy support."""
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df
```

**1.2 Fix Deprecated Patterns** (~2 hours)
- Adjacency: Replace `.append()` with `pl.concat()` (3 instances)
- Test that changes work with pandas AND polars

**Deliverable**: Bridge module, deprecated patterns fixed, tests green

---

### Phase 2: Internal Utilities (Day 2, ~6 hours)
**Goal**: Migrate internal functions users don't directly call

**2.1 Stats Module** (~3 hours)
- `zscore()`: Use Polars expressions `(pl.col() - mean) / std`
- `downsample()`: Use `.group_by().agg()` instead of pandas groupby
- `upsample()`: Leverage lazy evaluation for efficiency
- **Coordination**: Ensure ridge.py (model-spec) doesn't break

```python
# nltools/stats.py - zscore example
def zscore(df):
    """Z-score normalization using Polars."""
    if isinstance(df, pl.DataFrame):
        return df.select([
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std())
            .alias(col)
            for col in df.columns
        ])
    elif isinstance(df, pl.Series):
        return (df - df.mean()) / df.std()
    else:
        # Legacy numpy support
        return (df - np.mean(df)) / np.std(df)
```

**2.2 Validation Module** (~1 hour)
- `validate_frame()`: `pd.read_csv()` → `pl.read_csv()`
- Simple 1:1 replacements

**2.3 File I/O Utilities** (~2 hours)
- Deprecate pandas HDFStore for DataFrames
- **Coordinate with model-spec**: Use shared Parquet format
- Keep h5py for numpy arrays (both plans need this)

**Deliverable**: stats.py, validation.py, utils.py migrated, 100% tests passing

---

### Phase 3: Brain_Data (Day 3, ~5 hours)
**Goal**: Compatibility layer for deprecated X/Y attributes

**3.1 Internal Storage Migration** (~3 hours)
```python
class Brain_Data:
    def __init__(self, ...):
        self._X_internal = None  # Stores as Polars
        self._Y_internal = None

    @property
    def X(self):
        """DEPRECATED: Returns pandas for v0.6.0 compatibility"""
        warnings.warn(
            "Brain_Data.X is deprecated. Use design_matrix parameter in .regress() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._X_internal.to_pandas() if self._X_internal else None

    @X.setter
    def X(self, value):
        self._X_internal = ensure_polars(value)

    @property
    def Y(self):
        """DEPRECATED: Returns pandas for v0.6.0 compatibility"""
        warnings.warn(
            "Brain_Data.Y is deprecated.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._Y_internal.to_pandas() if self._Y_internal else None

    @Y.setter
    def Y(self, value):
        self._Y_internal = ensure_polars(value)
```

**3.2 Method Updates** (~2 hours)
- `__getitem__()`: Use Polars `.slice()` instead of `.iloc[]`
- `__eq__()`: Use Polars `.equals()`
- `.regress()`: Accept both pandas/Polars, convert internally
- **Coordination**: Ensure Model spec's `.predict()` works with this

**Deliverable**: Brain_Data with Polars internally, backward compatible properties

---

### Phase 4: DesignMatrix (Days 4-5, ~10 hours) **MOST COMPLEX**
**Goal**: Composition pattern to replace pandas subclassing

**4.1 Research Finding: Polars Discourages Subclassing**
- Polars operations return base DataFrame, not subclass
- Recommended: Namespace registration OR composition
- **Decision**: Composition pattern (DesignMatrix needs to be a DataFrame-like object)

**Why Polars doesn't support subclassing**:
> "Methods that produce new objects will 'lose' the sub-class, meaning any DataFrame operations return base DataFrame objects, not your subclass." - Polars Issue #2846

**4.2 New DesignMatrix Structure** (~5 hours)
```python
class DesignMatrix:
    """
    Polars-based design matrix with neuroimaging-specific methods.

    Uses composition pattern: wraps pl.DataFrame with metadata.
    Delegates DataFrame methods via __getattr__ for seamless UX.

    Parameters
    ----------
    data : DataFrame, ndarray, or dict
        Input data
    sampling_freq : float, optional
        Sampling frequency in Hz
    convolved : list, optional
        List of convolved column names
    polys : list, optional
        List of polynomial column names
    """

    def __init__(self, data, sampling_freq=None, convolved=None, polys=None, **kwargs):
        # Store metadata
        self.sampling_freq = sampling_freq
        self.convolved = convolved or []
        self.polys = polys or []
        self.multi = None

        # Store Polars DataFrame
        if isinstance(data, pl.DataFrame):
            self._df = data
        elif isinstance(data, pd.DataFrame):
            warnings.warn("pandas input deprecated", DeprecationWarning)
            self._df = pl.from_pandas(data)
        elif isinstance(data, dict):
            self._df = pl.DataFrame(data)
        elif isinstance(data, np.ndarray):
            self._df = pl.DataFrame(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Ensure column names are strings
        self._df = self._df.rename({col: str(col) for col in self._df.columns})

    def __getattr__(self, name):
        """Delegate DataFrame methods to internal Polars DataFrame"""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        attr = getattr(self._df, name)
        if callable(attr):
            # Wrap methods to return DesignMatrix instead of DataFrame
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pl.DataFrame):
                    return self._from_polars(result, self._get_metadata())
                return result
            return wrapper
        return attr

    def _get_metadata(self):
        """Extract metadata as dict"""
        return {
            'sampling_freq': self.sampling_freq,
            'convolved': self.convolved,
            'polys': self.polys,
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
            dm.multi = None
        return dm

    # Implement key DataFrame-like properties
    @property
    def shape(self):
        return self._df.shape

    @property
    def columns(self):
        return self._df.columns

    def __len__(self):
        return len(self._df)

    def __getitem__(self, key):
        """Support indexing like DataFrame"""
        result = self._df[key]
        if isinstance(result, pl.DataFrame):
            return self._from_polars(result, self._get_metadata())
        return result

    # Custom methods
    def append(self, other, axis=1):
        """
        Append columns or rows (Polars-style concatenation).

        Parameters
        ----------
        other : DesignMatrix or DataFrame
            Data to append
        axis : int, default=1
            0 for rows (vertical), 1 for columns (horizontal)
        """
        if isinstance(other, DesignMatrix):
            other_df = other._df
        else:
            other_df = ensure_polars(other)

        if axis == 1:
            new_df = pl.concat([self._df, other_df], how="horizontal")
        else:
            new_df = pl.concat([self._df, other_df], how="vertical")

        return self._from_polars(new_df, self._get_metadata())

    def to_pandas(self):
        """Convert to pandas DataFrame (for backward compatibility)"""
        return self._df.to_pandas()

    def to_polars(self):
        """Return underlying Polars DataFrame"""
        return self._df
```

**4.3 Migration Strategy** (~3 hours)
- Test each DesignMatrix method with Polars
- Critical operations to validate:
  - `.convolve()` - Creates convolved regressors
  - `.add_poly()` - Polynomial detrending
  - `.vif()` - Variance inflation (uses correlation matrix)
  - `.zscore()` - Standardization
  - `.downsample()` - Temporal downsampling

**4.4 Backward Compatibility** (~2 hours)
- Provide `.to_pandas()` escape hatch
- Accept pandas inputs with deprecation warnings
- Ensure nilearn compatibility (may need pandas conversion)

**Deliverable**: DesignMatrix fully Polars-based, all methods working, tests green

---

### Phase 5: Adjacency (Day 6, ~6 hours)
**Goal**: Complex statistical methods with efficient Polars operations

**5.1 Core Refactoring** (~4 hours)

**Problem**: Sequential `.append()` antipattern
```python
# OLD (slow, deprecated)
for i in groups:
    tmp = pd.DataFrame(...)
    out = out.append(tmp)  # Sequential, creates copies
```

**Solution**: Lazy evaluation with single concat
```python
# NEW (fast, Polars-idiomatic)
group_dfs = [
    pl.DataFrame({
        "Distance": extract_distances(i),
        "Group": i,
        "Type": "Within"
    }).lazy()
    for i in groups
]
out = pl.concat(group_dfs).collect()  # Single optimized execution
```

**5.2 Boolean Indexing Translation** (~2 hours)
```python
# pandas
tmp = out.loc[(out["Group"] == i) & (out["Type"] == "Within"), "Distance"]

# Polars (cleaner expression API)
tmp = out.filter(
    (pl.col("Group") == i) & (pl.col("Type") == "Within")
).select("Distance")
```

**Key Adjacency Methods to Migrate**:
1. **distance_stats()** (lines 973-987)
   - Replace sequential `.append()` with lazy concat
   - Use `.filter()` instead of `.loc[]`

2. **ttest()** (lines 1028-1043)
   - Boolean indexing → `.filter().select()`
   - GroupBy operations for statistics

3. **plot_silhouette()** (various)
   - Keep Polars internally, convert to pandas only for plotting

**5.3 Y Attribute** (~1 hour)
- Convert `self.Y` to Polars internally
- Provide pandas compatibility layer like Brain_Data

**Deliverable**: Adjacency with Polars, 5-10x speedup on statistics, tests green

---

### Phase 6: Integration & Testing (Day 7, ~5 hours)

**6.1 Test Suite Update** (~2 hours)
- Update fixtures: `pd.read_csv()` → `pl.read_csv()`
- Add conversion tests (pandas ↔ polars round-trip)
- **Coordinate**: Ensure model-spec tests still pass

```python
# Test example
def test_pandas_polars_conversion():
    """Test backward compatibility with pandas inputs"""
    # Create pandas DataFrame
    pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Should accept pandas and convert internally
    dm = DesignMatrix(pd_df, sampling_freq=2.0)

    # Internal storage is Polars
    assert isinstance(dm._df, pl.DataFrame)

    # Can convert back to pandas
    pd_result = dm.to_pandas()
    assert isinstance(pd_result, pd.DataFrame)
    pd.testing.assert_frame_equal(pd_df, pd_result)
```

**6.2 Compatibility Testing** (~2 hours)
- Run all 38 existing tests
- Test with real data workflows
- Verify nilearn integration (may need pandas conversion layer)

**6.3 Documentation Updates** (~1 hour)
- Update docs/migration-guide.md
- Add Polars migration notes
- Document `.to_pandas()` escape hatches

**Deliverable**: 100% tests passing, documentation updated

---

## 🔧 Technical Decisions

### DesignMatrix: Why Composition Over Subclassing?

**Polars position**: "Subclassing is not supported. Methods that produce new objects will lose the sub-class."

**Comparison**:

| Pattern | Pros | Cons | Verdict |
|---------|------|------|---------|
| Subclass Polars | Automatic method inheritance | Breaks on operations, not supported | ❌ Don't use |
| Namespace registration | Official Polars pattern | Can't be a DataFrame-like object | ❌ Wrong use case |
| **Composition + delegation** | **Full control, metadata preserved** | **Need to wrap methods** | ✅ **Best fit** |

**Why composition works for DesignMatrix**:
1. Wrap Polars DataFrame internally (`self._df`)
2. Delegate methods via `__getattr__` (automatic forwarding)
3. Return new DesignMatrix instances (not raw DataFrames)
4. Preserve metadata (sampling_freq, convolved, polys)
5. Similar UX to pandas subclassing, but robust

**Reference**: Polars Issue #2846, Stack Overflow discussions on extending Polars

---

## 🚀 Key Migration Patterns

### Pattern 1: DataFrame Creation
```python
# pandas
df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

# Polars
df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
```

### Pattern 2: CSV I/O
```python
# pandas
df = pd.read_csv("file.csv", header=None, index_col=None)
df.to_csv("out.csv", index=None)

# Polars
df = pl.read_csv("file.csv", has_header=False)
df.write_csv("out.csv")
```

### Pattern 3: Indexing
```python
# pandas (iloc)
subset = df.iloc[5:10]

# Polars (slice)
subset = df.slice(5, 5)  # offset, length
```

### Pattern 4: Boolean Filtering
```python
# pandas
filtered = df.loc[df["age"] > 30, ["name", "age"]]

# Polars (expression API)
filtered = df.filter(pl.col("age") > 30).select(["name", "age"])
```

### Pattern 5: GroupBy Aggregation
```python
# pandas
result = df.groupby("group").agg({"value": "mean"}).reset_index(drop=True)

# Polars (cleaner)
result = df.group_by("group").agg(pl.col("value").mean())
```

### Pattern 6: Concatenation
```python
# pandas
combined = pd.concat([df1, df2], axis=0)  # vertical
combined = pd.concat([df1, df2], axis=1)  # horizontal

# Polars (more explicit)
combined = pl.concat([df1, df2], how="vertical")
combined = pl.concat([df1, df2], how="horizontal")
```

### Pattern 7: Apply Functions
```python
# pandas (slow)
df["new"] = df["old"].apply(lambda x: x * 2)

# Polars (vectorized, fast)
df = df.with_columns((pl.col("old") * 2).alias("new"))
```

---

## 🚫 What We're NOT Doing (Deferred to v0.7.0)

1. **Polars GPU Engine** - Keep CPU-only, avoid complexity
2. **PyTorch integration** - That's model-spec's domain
3. **Lazy evaluation by default** - Use eager mode, add lazy optimization later
4. **Complete API redesign** - Maintain v0.5.1 compatibility where possible
5. **Remove pandas dependency entirely** - Still downstream via nilearn

---

## ⚠️ Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| DesignMatrix breaks user code | HIGH | Extensive testing, `.to_pandas()` escape hatch |
| Nilearn expects pandas | MEDIUM | Conversion layer at boundaries |
| Model spec conflicts | MEDIUM | Coordinate via shared Brain_Data.data (numpy) |
| Performance regression | LOW | Polars is faster, but benchmark to verify |
| Test failures | MEDIUM | Update tests incrementally per phase |
| Polars API changes | LOW | Pin to stable version (>=0.20.0) |

---

## 📊 Success Criteria

- ✅ Pandas removed from direct dependencies
- ✅ All 38 tests passing with Polars
- ✅ DesignMatrix works identically to v0.5.1 (user perspective)
- ✅ 2-5x speedup on Adjacency statistics (lazy evaluation)
- ✅ Zero import errors (polars vs pandas)
- ✅ Model spec's work unaffected (both tests pass together)
- ✅ Backward compatibility: Accept pandas with deprecation warnings

---

## 🔄 Coordination Checkpoints with Model Spec

**Before Phase 2**: Confirm stats.py signatures stable
**Before Phase 3**: Verify Brain_Data.data stays numpy (both need it)
**After Phase 4**: Test DesignMatrix with model-spec ridge regression
**After Phase 6**: Run both test suites together (Polars + Models)

---

## 📦 Dependencies

**Add** (required):
```toml
[project.dependencies]
polars = ">=0.20.0"  # Modern API, stable
```

**Keep** (for now, deprecate path):
```toml
pandas = ">=1.5.0"  # Still downstream dependency via nilearn
```

**Future** (v0.7.0):
```toml
polars = { version = ">=0.20.0", extras = ["gpu"] }  # GPU engine
```

---

## 🎯 Timeline Summary

- **Day 1**: Bridge module, fix deprecated patterns (~4 hrs)
- **Day 2**: Stats, validation, file I/O (~6 hrs)
- **Day 3**: Brain_Data compatibility layer (~5 hrs)
- **Days 4-5**: DesignMatrix composition pattern (~10 hrs)
- **Day 6**: Adjacency lazy evaluation (~6 hrs)
- **Day 7**: Integration testing (~5 hrs)

**Total: 5-7 days** (can be parallelized with Model spec work)

---

## 💡 Key Insights

1. **Polars is NOT pandas** - Don't do naive 1:1 replacement, use expression API
2. **Lazy evaluation is powerful** - Especially for Adjacency statistics (5-10x speedup)
3. **Composition beats subclassing** - For DesignMatrix metadata needs
4. **Bridge layers enable migration** - Accept both, convert internally, deprecate pandas
5. **Coordinate at interfaces** - Brain_Data.data (numpy) is shared contract with Model spec
6. **GPU deferral is wise** - CPU Polars already gives 2-5x, GPU in v0.7.0 gives another 10x
7. **No index = cleaner code** - Polars' integer positions eliminate `.reset_index()` calls

---

## 📚 References

### Polars Documentation
- **Migration Guide**: https://docs.pola.rs/user-guide/migration/pandas/
- **Expression API**: https://docs.pola.rs/user-guide/expressions/
- **Lazy API**: https://docs.pola.rs/user-guide/concepts/lazy-api/

### Research Sources
- Polars Issue #2846: "Enabling the sub-classing of core data types"
- Polars GPU Engine: https://pola.rs/posts/polars-on-gpu/
- Stack Overflow: Best practices for extending Polars DataFrames

### Internal Documents
- `model-spec.md` - Model class implementation plan (parallel work)
- `refactor-plan.md` - Overall v0.6.0 strategic vision
- `refactor-todos.md` - Task tracking
- `docs/migration-guide.md` - User-facing migration guide
- `CLAUDE.md` - Development knowledge base

---

**Last Updated**: 2025-10-28
**Status**: Planning → Ready for Implementation
**Coordinator**: Execute in parallel with model-spec.md, sync at checkpoints
