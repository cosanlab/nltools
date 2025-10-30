# DesignMatrix Polars Implementation - Code Review (Updated)

**Date**: 2025-10-29 (Updated after Phase 1 refactoring)
**Reviewer**: Claude (following Eshin's Polars teaching philosophy)
**File**: `nltools/data/design_matrix_new.py`
**Status**: ✅ All 68 tests passing | ✅ High-priority refactoring COMPLETE

---

## Executive Summary

### ✅ COMPLETED (Phase 1 Refactoring - 45 min)

High-priority refactoring is **DONE**:
1. ✅ **Extracted helper methods** - `_get_data_columns()`, `_to_pandas()`
2. ✅ **Simplified zscore()** - 50% code reduction, uses `.with_columns()`
3. ✅ **Simplified convolve()** - 60% code reduction, idiomatic Polars
4. ✅ **Removed dead code** - Deleted unused `_from_polars()` method
5. ✅ **All tests passing** - 68/68 tests in 0.32 seconds

**See `polars-refactoring-summary.md` for details**

### 🎯 REMAINING OPPORTUNITIES

**Remaining effort**: ~1.5 hours for optional polish
**Risk**: Minimal (all changes are refinements, not bug fixes)

---

## Core Teaching Principles Applied

From Eshin's tutorial at https://stat-intuitions.com/labs/3/01_polars-solutions.html:

1. ✅ **Readability over cleverness** - "Python's big advantage is that it's very easy to read"
2. ✅ **Declarative expressions** - Use contexts + expressions, not imperative loops
3. ✅ **Avoid over-engineering** - "Don't get carried away"
4. ⚠️ **Use `.with_columns()` to preserve data** - We could leverage this more
5. ⚠️ **Chain operations for clarity** - Some methods can be more concise
6. ⚠️ **Use selectors for bulk operations** - Opportunity for `polars.selectors`

---

## ✅ COMPLETED: High-Priority Refactoring

All high-priority items have been completed. See `polars-refactoring-summary.md` for details.

**What was done**:
1. ✅ Extracted `_get_data_columns()` helper (replaced 7+ instances)
2. ✅ Extracted `_to_pandas()` helper (replaced 3 instances)
3. ✅ Simplified `zscore()` - 50% code reduction, uses `.with_columns()`
4. ✅ Simplified `convolve()` single-kernel - 60% code reduction
5. ✅ Updated `vif()`, `clean()` to use helpers
6. ✅ Removed dead code (`_from_polars()` method)
7. ✅ All 68 tests passing

---

## 🟡 MEDIUM PRIORITY: Performance & Idiomaticity

### 4. **Polars Selectors for Bulk Operations**

**Opportunity**: Several methods iterate over column lists when selectors could be more concise.

**Example - vif() method (lines 913-924)**:

**Current**:
```python
if exclude_polys and self.polys:
    cols_to_use = [c for c in self.columns if c not in self.polys]
else:
    cols_to_use = [c for c in self.columns if "poly_0" not in c]

subset_df = self._df.select(cols_to_use)
data_array = subset_df.to_numpy()
```

**Refactored with selectors**:
```python
from polars import selectors as cs

# More declarative: "select all columns except these patterns"
if exclude_polys and self.polys:
    subset_df = self._df.select(cs.exclude(self.polys))
else:
    # Select columns that don't contain "poly_0"
    subset_df = self._df.select([c for c in self.columns if "poly_0" not in c])

data_array = subset_df.to_numpy()
```

**Benefit**:
- More declarative ("exclude these" vs "keep everything except these")
- Polars can optimize selector-based operations
- Aligns with tutorial's selector pattern teaching

**From Eshin's tutorial**:
> "Use `polars.selectors` for dynamic column selection... improves discoverability"

---

### 5. **Unnecessary `.to_numpy()` Conversions in Convolution**

**Current code** (lines 435-451):
```python
# Single kernel case
convolved_data = {}
for col in columns_to_convolve:
    col_data = self._df[col].to_numpy()  # Convert to numpy
    convolved = np.convolve(col_data, conv_func)[:n_rows]
    convolved_data[col] = convolved

# Rebuild DataFrame from dict
all_cols_data = {}
for col in self.columns:
    if col in columns_to_convolve:
        all_cols_data[col] = convolved_data[col]
    else:
        all_cols_data[col] = self._df[col].to_numpy()  # Another conversion!

new_df = pl.DataFrame(all_cols_data)
```

**Problem**:
- Converts ALL columns to numpy (both convolved and non-convolved)
- Rebuilds entire DataFrame from scratch (inefficient)

**Refactored solution**:
```python
# Single kernel case
convolved_exprs = []
for col in columns_to_convolve:
    col_data = self._df[col].to_numpy()  # Only converted columns
    convolved = np.convolve(col_data, conv_func)[:n_rows]
    convolved_exprs.append(pl.Series(col, convolved))

# Use .with_columns() to replace only convolved columns
new_df = self._df.with_columns(convolved_exprs)
```

**Benefit**:
- **Fewer conversions** (only convolved columns → numpy, not all columns)
- **Preserves original columns** (`.with_columns()` keeps non-convolved as-is)
- **Better GPU path** (fewer numpy conversions = easier GPU migration later)

---

### 6. **Simplify `convolve()` Multi-Kernel Case** (lines 456-476)

**Current**:
```python
# Multiple kernels: shape is (samples, n_kernels)
n_kernels = conv_func.shape[1]
all_convolved_data = {}

for col in columns_to_convolve:
    col_data = self._df[col].to_numpy()
    for k_idx in range(n_kernels):
        kernel = conv_func[:, k_idx]
        convolved = np.convolve(col_data, kernel)[:n_rows]
        all_convolved_data[f"{col}_c{k_idx}"] = convolved

# Create new DataFrame with all convolved columns + non-convolved
convolved_df = pl.DataFrame(all_convolved_data)
non_convolved_df = self._df.select(non_convolved_cols) if non_convolved_cols else pl.DataFrame()

# Concatenate horizontally
if non_convolved_cols:
    new_df = pl.concat([convolved_df, non_convolved_df], how="horizontal")
else:
    new_df = convolved_df
```

**Refactored**:
```python
# Multiple kernels
n_kernels = conv_func.shape[1]
convolved_series = []

for col in columns_to_convolve:
    col_data = self._df[col].to_numpy()
    for k_idx in range(n_kernels):
        kernel = conv_func[:, k_idx]
        convolved = np.convolve(col_data, kernel)[:n_rows]
        convolved_series.append(pl.Series(f"{col}_c{k_idx}", convolved))

# Drop original columns, add convolved variants
new_df = self._df.drop(columns_to_convolve).with_columns(convolved_series)
```

**Benefit**:
- Simpler logic (no manual horizontal concat)
- Clearer intent (drop old, add new)
- Same result, fewer intermediate DataFrames

---

## 🟢 LOW PRIORITY: Code Quality & Future-Proofing

### 7. **Remove Unused `_from_polars()` Method** (lines 1228-1236)

**Current**:
```python
@classmethod
def _from_polars(cls, df: pl.DataFrame, metadata: Optional[dict] = None) -> "DesignMatrix":
    """
    Create DesignMatrix from Polars DataFrame with optional metadata.

    Used internally for constructing results of operations.
    """
    # TODO: Implement
    raise NotImplementedError("_from_polars not yet implemented")
```

**Problem**:
- Dead code (never called, all operations use `_copy_with()` instead)
- Confusing (suggests it should be used, but isn't)

**Decision needed**:
- **Option A**: Delete it (simplify)
- **Option B**: Implement it properly (for future public API)
- **Option C**: Keep as private helper for specific use cases

**Recommendation**: **Delete it**. We have `_copy_with()` which serves the same purpose more explicitly.

---

### 8. **Clean Up TODOs** (3 instances)

**Lines 291, 337**: Polars-native resampling
```python
# TODO: Implement Polars-native downsampling in future optimization
```

**Action**: Either:
1. Defer to v0.7.0 (change TODO to "# Future v0.7.0: Polars-native resampling")
2. Implement now (use `.group_by_dynamic()` from tutorial examples)

**Line 1235**: `_from_polars()` not implemented (see #7 above)

**Recommendation**:
- Update downsample/upsample TODOs to reference v0.7.0 milestone
- Delete `_from_polars()` TODO

---

### 9. **Improve Error Messages** (12 error locations)

**Current error messages are good, but could be more helpful**:

**Example - line 286**:
```python
raise ValueError(
    f"Target ({target} Hz) must be less than current sampling_freq ({self.sampling_freq} Hz)"
)
```

**Enhanced**:
```python
raise ValueError(
    f"Downsampling target ({target} Hz) must be less than current sampling_freq "
    f"({self.sampling_freq} Hz). For upsampling, use .upsample() instead."
)
```

**Benefit**: Suggests fix, not just states problem

---

### 10. **Type Hints - Complete Coverage**

**Current**: Most methods have type hints ✅
**Missing**: Some internal helpers lack return type hints

**Example - line 723** (`_match_column_pattern`):
```python
def _match_column_pattern(self, columns: List[str], pattern: str) -> List[str]:  # ✅ Has hint
```

**Example - line 1219** (`_get_metadata`):
```python
def _get_metadata(self) -> dict:  # ✅ Has hint
```

**Status**: Type hints are complete! No action needed.

---

## 🚀 GPU-Readiness Assessment (for v0.7.0)

### Current Patterns That Help GPU Migration:

✅ **Good patterns**:
1. Expression-based operations (`.select()`, `.with_columns()`)
2. Vectorized operations (no row-by-row iteration)
3. Minimal numpy conversions (only when necessary: convolution, stats)

⚠️ **Patterns to watch**:
1. `.to_numpy()` calls (5 instances) - GPU would need CuPy equivalent
2. Numpy-specific operations (`np.convolve`, `np.corrcoef`) - need GPU implementations
3. Scipy dependencies (`scipy.special.legendre`) - need GPU fallback

### GPU Migration Path (v0.7.0):

```python
# 1. Add polars GPU engine dependency
# polars = { version = ">=0.20.0", extras = ["gpu"] }

# 2. Add GPU backend selection
def _get_array_backend(self):
    """Select numpy or cupy based on GPU availability."""
    if self._use_gpu and pl.gpu_available():
        import cupy as cp
        return cp
    return np

# 3. Wrap numpy operations
xp = self._get_array_backend()
col_data = self._df[col].to_numpy()  # Or .to_cupy() on GPU
convolved = xp.convolve(col_data, conv_func)

# 4. Test: GPU operations should return same results as CPU
```

---

## 📊 Summary of Recommendations

### MUST DO (Before Cutover):

| # | Issue | Lines | Effort | Impact |
|---|-------|-------|--------|--------|
| 1 | Fix zscore() duplication | 217-262 | 15 min | High - 50% code reduction |
| 2 | Add `_get_data_columns()` helper | New | 10 min | High - DRY principle |
| 3 | Add `_to_pandas()` helper | New | 5 min | Medium - future-proofing |
| 4 | Simplify convolve() single kernel | 429-454 | 20 min | Medium - clarity |

**Total effort**: ~1 hour
**Risk**: Low (behavior preserved, tests verify)

### SHOULD DO (Before v0.6.0 Release):

| # | Issue | Lines | Effort | Impact |
|---|-------|-------|--------|--------|
| 5 | Use selectors in vif() | 913-924 | 10 min | Low - idiomaticity |
| 6 | Simplify convolve() multi-kernel | 456-476 | 15 min | Low - clarity |
| 7 | Remove `_from_polars()` | 1228-1236 | 5 min | Low - cleanup |
| 8 | Update TODO comments | 291, 337, 1235 | 5 min | Low - clarity |

**Total effort**: ~35 min
**Risk**: Minimal

### NICE TO HAVE (Optional):

| # | Issue | Effort | Impact |
|---|-------|--------|--------|
| 9 | Enhance error messages | 30 min | Low - UX |
| 10 | Add GPU readiness notes | 15 min | Low - documentation |

---

## 🎯 Refactoring Strategy

### Phase 1: Extract Helpers (30 min)
1. Add `_get_data_columns()` helper
2. Add `_to_pandas()` helper
3. Run tests: `uv run pytest nltools/tests/test_design_matrix_new.py -x`

### Phase 2: Simplify zscore() (20 min)
1. Replace `.select()` with `.with_columns()`
2. Remove unused variables
3. Run tests: `uv run pytest nltools/tests/test_design_matrix_new.py::test_zscore -xvs`

### Phase 3: Simplify convolve() (30 min)
1. Refactor single kernel case
2. Refactor multi-kernel case
3. Run tests: `uv run pytest nltools/tests/test_design_matrix_new.py::test_convolve -xvs`

### Phase 4: Cleanup (20 min)
1. Remove `_from_polars()`
2. Update TODOs
3. Use selectors where appropriate
4. Full test suite: `uv run pytest nltools/tests/test_design_matrix_new.py -x`

**Total**: ~2 hours (conservative estimate)

---

## 📝 Code Quality Metrics

### Current State:
- **Lines of code**: 1,237
- **Methods**: 30
- **Helper methods**: 6 (good separation of concerns)
- **Code duplication**: Medium (7+ repeated patterns)
- **Polars idioms**: Good (expression-based, mostly uses contexts correctly)
- **Readability**: Good (clear comments, logical structure)
- **Test coverage**: Excellent (68/68 tests passing)

### After Refactoring:
- **Lines of code**: ~1,150 (-87 lines, -7%)
- **Code duplication**: Low (helpers extract repeated patterns)
- **Polars idioms**: Excellent (uses `.with_columns()`, selectors)
- **Readability**: Excellent (less duplication, clearer intent)
- **Maintainability**: High (single source of truth for patterns)

---

## ✅ What's Already Great

Don't change these patterns:

1. ✅ **Composition pattern** - Wrapping `pl.DataFrame` works perfectly
2. ✅ **Metadata preservation** - `_copy_with()` is elegant and consistent
3. ✅ **Immutable transformations** - All methods return new instances
4. ✅ **Dict-based pandas conversion** - Smart choice to avoid pyarrow dependency
5. ✅ **Comprehensive docstrings** - Most methods well-documented
6. ✅ **Error handling** - Good validation with helpful messages
7. ✅ **Test coverage** - 68/68 tests is excellent

---

## 🤔 Questions for Eshin

1. **`_from_polars()` method**: Delete or implement? (I recommend delete)

2. **Polars-native resampling**: Implement now or defer to v0.7.0?
   - **Now**: Use `.group_by_dynamic()` (from tutorial)
   - **Later**: Keep pandas bridge for v0.6.0

3. **Selectors usage**: How aggressive should we be?
   - **Conservative**: Only where clearly better (vif, clean)
   - **Aggressive**: Import `polars.selectors as cs` at module level, use throughout

4. **GPU notes**: Include GPU readiness comments in code, or just in docs?

5. **Error message enhancement**: Worth the time for v0.6.0, or defer to UX polish phase?

---

**Next Steps**:
1. Get your feedback on these recommendations
2. Implement Phase 1-4 refactoring
3. Run full test suite
4. Update refactor-progress.md with learnings

**Ready to proceed when you are!** 🚀
