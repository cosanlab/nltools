# DesignMatrix Polars Refactoring Summary

**Date**: 2025-10-29
**Status**: ✅ Complete - All 68 tests passing
**Time**: ~45 minutes actual implementation
**Lines of code**: 1,256 (added well-documented helpers, removed duplication)

---

## What Was Done

### Phase 1: Helper Methods (10 min)

Added two helper methods to reduce code duplication:

**1. `_get_data_columns(exclude_polys=True)`** (~30 lines with docstring)
- Replaced 7+ instances of `[col for col in self.columns if col not in self.polys]`
- Single source of truth for column filtering logic
- Clear, documented behavior

**2. `_to_pandas()`** (~30 lines with docstring)
- Replaced 3 instances of `pd.DataFrame(self._df.to_dict(as_series=False))`
- Centralizes pandas conversion pattern
- Documented future pyarrow migration path
- Used in: downsample(), upsample(), heatmap()

**3. Removed `_from_polars()`** (dead code)
- Unused method with NotImplementedError
- Replaced everywhere by `_copy_with()` pattern

---

### Phase 2: zscore() Simplification (15 min)

**Before** (45 lines with duplication):
```python
# Build expressions
zscore_exprs = [...]  # Created but NEVER used!
unchanged_cols = [...]  # Created but NEVER used!

# Rebuild expressions AGAIN (duplication!)
all_exprs = []
for col in self.columns:
    if col in columns_to_zscore:
        all_exprs.append(
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        )
    else:
        all_exprs.append(pl.col(col))

# Apply with .select() (manual column ordering)
zscored_df = self._df.select(all_exprs)
```

**After** (30 lines, no duplication):
```python
# Get columns using helper
columns_to_zscore = self._get_data_columns(exclude_polys=True)

# Build expressions ONCE
zscore_exprs = [
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
    for col in columns_to_zscore
]

# Use .with_columns() - idiomatic Polars pattern!
zscored_df = self._df.with_columns(zscore_exprs)
```

**Benefits**:
- **50% less code** in the critical section (15 lines → 8 lines)
- **No duplication** (expression defined once)
- **Idiomatic Polars** (`.with_columns()` from Eshin's tutorial)
- **Clearer intent** ("zscore these, keep rest as-is")

**Key insight from Eshin's tutorial**:
> "Use `.with_columns()` when you want to preserve original data"

This is exactly that pattern!

---

### Phase 3: convolve() Simplification (15 min)

**Before** (single kernel case - 22 lines, complex):
```python
# Build dict
convolved_data = {}
for col in columns_to_convolve:
    col_data = self._df[col].to_numpy()
    convolved = np.convolve(col_data, conv_func)[:n_rows]
    convolved_data[col] = convolved

# Create intermediate DataFrames (unused!)
convolved_df = pl.DataFrame(convolved_data)
non_convolved_df = self._df.select(non_convolved_cols) if non_convolved_cols else pl.DataFrame()

# Rebuild ENTIRE DataFrame from scratch (inefficient!)
all_cols_data = {}
for col in self.columns:
    if col in columns_to_convolve:
        all_cols_data[col] = convolved_data[col]
    else:
        all_cols_data[col] = self._df[col].to_numpy()  # More conversions!

new_df = pl.DataFrame(all_cols_data)
```

**After** (9 lines, idiomatic):
```python
# Build Series list
convolved_series = []
for col in columns_to_convolve:
    col_data = self._df[col].to_numpy()
    convolved = np.convolve(col_data, conv_func)[:n_rows]
    convolved_series.append(pl.Series(col, convolved))

# Use .with_columns() to replace only convolved columns
new_df = self._df.with_columns(convolved_series)
```

**Benefits**:
- **60% less code** (22 lines → 9 lines)
- **Fewer conversions** (only convolved columns → numpy, not ALL columns)
- **No intermediate DataFrames** (cleaner, less memory)
- **Better GPU path** (fewer numpy conversions)

**Also updated**:
- `convolve()` now uses `_get_data_columns()` helper
- `vif()` uses `_get_data_columns()` helper
- `clean()` uses `_get_data_columns()` helper

---

## Metrics

### Code Quality Improvements

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total lines | 1,237 | 1,256 | +19 (added helpers) |
| Helper methods | 4 | 6 | +2 |
| Code duplication | High (7+ instances) | Low (DRY) | ✅ |
| Polars idioms | Good | Excellent | ✅ |
| `.with_columns()` usage | 3 | 5 | +2 |
| `.to_numpy()` conversions | 5 | 5 | Same (minimal) |
| Pandas conversions | 3 duplicated | 3 via helper | ✅ |

### Test Results

```
============================== 68 passed in 0.32s ==============================
```

✅ **100% tests passing** (68/68)
✅ **Fast test execution** (0.32 seconds)
✅ **No regressions**

---

## Alignment with Teaching Philosophy

From Eshin's tutorial at https://stat-intuitions.com/labs/3/01_polars-solutions.html:

### ✅ Applied Principles

1. **"Use `.with_columns()` to preserve original data"**
   - zscore(): Now uses `.with_columns()` instead of `.select()`
   - convolve(): Now uses `.with_columns()` instead of rebuilding DataFrame

2. **"Readability over cleverness"**
   - Extracted helpers with clear names (`_get_data_columns`, `_to_pandas`)
   - Removed unused intermediate variables (zscore_exprs, keep_exprs)
   - Simplified convolve() logic

3. **"Don't get carried away"**
   - Didn't over-engineer - kept changes focused on high-priority items
   - Preserved existing patterns that work well
   - Added helpful comments explaining design choices

4. **"Python's advantage is it's easy to read"**
   - Helper methods have comprehensive docstrings
   - Removed confusing duplication
   - Clearer variable names and intent

### ✅ Patterns Demonstrated

- **Expression building**: `[expr for col in columns]` pattern
- **Context usage**: `.with_columns()`, `.select()`, `.filter()`
- **Helper functions**: Reusable column selection logic
- **Method chaining**: `pl.col().mean().std().alias()`

---

## What Was NOT Changed

**Good patterns preserved** (don't fix what isn't broken):

1. ✅ **Composition pattern** - Wrapping `pl.DataFrame` works perfectly
2. ✅ **`_copy_with()` pattern** - Elegant and consistent
3. ✅ **Immutable transformations** - All methods return new instances
4. ✅ **Dict-based pandas conversion** - Smart to avoid pyarrow dependency
5. ✅ **Comprehensive docstrings** - Most methods well-documented
6. ✅ **Error handling** - Good validation with helpful messages
7. ✅ **Test coverage** - 68/68 tests is excellent
8. ✅ **Multi-kernel convolve()** - Logic is sound, left as-is

---

## Future Opportunities (Deferred)

These were identified but not implemented (out of scope for high-priority):

### SHOULD DO (Before v0.6.0 Release):
- Use `polars.selectors` in vif() and clean() for more declarative code
- Simplify convolve() multi-kernel case (similar to single-kernel refactor)
- Enhance error messages with suggested fixes

### NICE TO HAVE (Optional):
- Add GPU readiness comments for v0.7.0
- Implement Polars-native resampling (`.group_by_dynamic()`)
- Add pyarrow optimization path

**Total deferred effort**: ~1 hour
**These can be done later without risk**

---

## Files Modified

1. **`nltools/data/design_matrix_new.py`**
   - Added: `_get_data_columns()` helper
   - Added: `_to_pandas()` helper
   - Removed: `_from_polars()` dead code
   - Refactored: `zscore()` to use `.with_columns()`
   - Refactored: `convolve()` single-kernel case
   - Updated: `downsample()`, `upsample()`, `heatmap()` to use helper
   - Updated: `vif()`, `clean()` to use helper
   - Updated: TODOs with v0.7.0 milestones

---

## Next Steps

### Immediate (Ready for Cutover):

1. **Switch design_matrix.py to use design_matrix_new.py**
   - Replace shim file with Polars implementation
   - All refactoring complete, tests green ✅

2. **Test with real workflows**
   - Run actual analysis scripts (not just unit tests)
   - Verify nilearn integration works
   - Profile performance improvements

3. **Update migration guide**
   - Document DesignMatrix Polars API
   - Add Polars idioms examples
   - Note `.to_pandas()` escape hatch

### Later (v0.6.0 Polish):
- Consider implementing deferred optimizations
- Add performance benchmarks
- Consider pyarrow dependency decision

---

## Key Learnings

### What Worked Well:
1. **TDD approach** - Having 68 tests made refactoring safe and fast
2. **Phased implementation** - Small, testable changes
3. **Eshin's tutorial** - Perfect reference for idiomatic patterns
4. **`.with_columns()` pattern** - Natural fit for many transformations

### Insights:
1. **`.with_columns()` is underutilized** - We could use it more in v0.7.0
2. **Helper extraction is powerful** - DRY principle paid off immediately
3. **Polars is concise** - Idiomatic code is shorter AND clearer
4. **Good tests enable refactoring** - 68 tests caught any issues instantly

### For Future Migrations:
1. Look for column filtering patterns first
2. Look for pandas conversion patterns early
3. Prefer `.with_columns()` over `.select()` when preserving data
4. Extract helpers before refactoring methods

---

**Status**: ✅ Ready for integration testing and cutover
**Risk**: Low (all tests pass, behavior preserved)
**Estimated performance**: Same or slightly better (fewer conversions)

**Next action**: Get Eshin's approval to proceed with design_matrix.py cutover
