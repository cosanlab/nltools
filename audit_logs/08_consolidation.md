# Consolidation Audit Report for nltools v0.6.0

**Date:** 2026-01-24
**Scope:** All source files in `nltools/`
**Focus:** Deduplication and consolidation opportunities

---

## Executive Summary

This audit identifies significant consolidation opportunities across the three main data classes:
- `BrainData` (4,785 lines)
- `Adjacency` (1,968 lines)
- `DesignMatrix` (1,410 lines)
- `BrainCollection` (5,054 lines)

**Key Findings:**
1. **Arithmetic operations** in Adjacency are heavily duplicated (8 nearly-identical methods)
2. **Statistical methods** (`mean`, `std`, `sum`, `median`) share common patterns across classes
3. **IO methods** (`write`, `copy`) have similar structures but no shared base
4. **Validation utilities** are well-consolidated in `nltools/data/_validation.py` and `nltools/algorithms/_validation.py`
5. **Bootstrap methods** are duplicated between BrainData and Adjacency
6. **Fisher transforms** (`r_to_z`, `z_to_r`) are duplicated between BrainData and Adjacency

---

## 1. Duplicate Method Implementations

### 1.1 Arithmetic Operations (Adjacency)

**Location:** `nltools/data/adjacency/__init__.py:298-394`

The Adjacency class has **8 nearly-identical arithmetic methods** that differ only in:
- The numpy operation used (add, subtract, multiply, divide)
- The operation name in error messages

```python
# Pattern repeated 8 times (lines 298-394):
def __add__(self, y):
    new = deepcopy(self)
    if isinstance(y, (int, np.integer, float, np.floating)):
        new.data = new.data + y  # Only this line differs
    elif isinstance(y, Adjacency):
        if self.shape != y.shape:
            raise ValueError("Both Adjacency() instances need to be the same shape.")
        new.data = new.data + y.data  # And this line
    else:
        raise ValueError("Can only add int, float, or Adjacency")  # And error message
    return new
```

**Methods affected:**
- `__add__` (line 298)
- `__radd__` (line 312)
- `__sub__` (line 326)
- `__rsub__` (line 340)
- `__mul__` (line 354)
- `__rmul__` (line 368)
- `__truediv__` (line 382)

**Consolidation Recommendation:**
Create a `_perform_arithmetic` method similar to BrainData's implementation:

```python
def _perform_arithmetic(self, other, operation, operation_name):
    """Unified arithmetic operation handler."""
    new = deepcopy(self)
    if isinstance(other, (int, np.integer, float, np.floating)):
        new.data = operation(new.data, other)
    elif isinstance(other, Adjacency):
        if self.shape != other.shape:
            raise ValueError(f"Both Adjacency instances need same shape for {operation_name}")
        new.data = operation(new.data, other.data)
    else:
        raise ValueError(f"Can only {operation_name} int, float, or Adjacency")
    return new
```

**Estimated Effort:** 2 hours
**Lines Saved:** ~70 lines

---

### 1.2 Statistical Methods (BrainData, Adjacency, BrainCollection)

**Locations:**
- `nltools/data/brain_data.py:1187-1237` (`mean`, `std`, `sum`, `median`)
- `nltools/data/adjacency/__init__.py:563-657` (`mean`, `std`, `sum`, `median`)
- `nltools/data/collection.py:1221-1280` (`mean`, `std`, `sum`)

**Pattern in BrainData (well-refactored):**
```python
def mean(self, axis=0):
    return self._apply_func(np.mean, axis)

def std(self, axis=0):
    return self._apply_func(np.std, axis)
```

**Pattern in Adjacency (duplicated):**
```python
def mean(self, axis=0):
    if self.is_single_matrix:
        return np.nanmean(self.data)
    else:
        if axis == 0:
            return Adjacency(data=np.nanmean(self.data, axis=axis), ...)
        elif axis == 1:
            return np.nanmean(self.data, axis=axis)

def std(self, axis=0):
    # Identical structure to mean()
    ...
```

**Consolidation Recommendation:**
Adjacency should adopt BrainData's `_apply_func` pattern:

```python
def _apply_func(self, stat_func, axis=0):
    if self.is_single_matrix:
        return stat_func(self.data)
    if axis == 0:
        return Adjacency(data=stat_func(self.data, axis=axis), matrix_type=self.matrix_type + "_flat")
    elif axis == 1:
        return stat_func(self.data, axis=axis)
```

**Estimated Effort:** 1 hour
**Lines Saved:** ~50 lines

---

### 1.3 Fisher Transform Methods (r_to_z, z_to_r)

**Locations:**
- `nltools/data/brain_data.py:2834-2850`
- `nltools/data/adjacency/__init__.py:986-999`

Both classes have identical implementations:

```python
# BrainData
def r_to_z(self):
    out = self._shallow_copy_with_data()
    out.data = fisher_r_to_z(self.data)
    return out

# Adjacency (line 986)
def r_to_z(self):
    out = self.copy()
    out.data = fisher_r_to_z(out.data)
    return out
```

**Consolidation Recommendation:**
Create a shared mixin class:

```python
class FisherTransformMixin:
    """Mixin for Fisher r-to-z and z-to-r transformations."""

    def r_to_z(self):
        """Apply Fisher's r to z transformation."""
        out = self.copy()
        out.data = fisher_r_to_z(out.data)
        return out

    def z_to_r(self):
        """Convert z score back to r value."""
        out = self.copy()
        out.data = fisher_z_to_r(out.data)
        return out
```

**Estimated Effort:** 30 minutes
**Lines Saved:** ~15 lines

---

### 1.4 Bootstrap Methods

**Locations:**
- `nltools/data/brain_data.py:3093-3330` (~240 lines)
- `nltools/data/adjacency/__init__.py:1210-1313` (~100 lines)

Both classes have bootstrap methods with similar structure but different return type handling:

**Common Elements:**
- Same parameter signature (`stat`, `n_samples`, `save_boots`, `n_jobs`, `random_state`, `percentiles`)
- Same stat validation (`SIMPLE_STATS = ["mean", "median", "std", "sum", "min", "max"]`)
- Same call to `_bootstrap_simple_cpu_parallel`
- Similar result conversion methods

**Key Differences:**
- BrainData supports fitted model bootstrap (`weights`, `predict`) with GPU acceleration
- Adjacency only supports simple stats
- Return type conversion differs (`_convert_bootstrap_results_to_brain_data` vs `_convert_bootstrap_results_to_adjacency`)

**Consolidation Recommendation:**
The bootstrap logic is already well-factored into `nltools/algorithms/inference/bootstrap.py`. The class methods are thin wrappers that handle type conversion. Keep as-is but ensure the conversion methods have consistent signatures.

**Estimated Effort:** N/A (already well-factored)
**Status:** Good - functional core properly extracted

---

### 1.5 Copy and Deepcopy Methods

**Locations:**
- `nltools/data/brain_data.py:1161-1174` (`copy()`)
- `nltools/data/brain_data.py:1118-1159` (`_shallow_copy_with_data()`)
- `nltools/data/adjacency/__init__.py:736-738` (`copy()`)
- `nltools/data/design_matrix.py:208-226` (`copy()`)

**Patterns:**

BrainData has sophisticated copy machinery with `__deepcopy__` and `_shallow_copy_with_data()` for performance.

Adjacency uses simple `deepcopy(self)`:
```python
def copy(self):
    return deepcopy(self)
```

DesignMatrix uses Polars clone:
```python
def copy(self):
    cloned_df = self._df.clone()
    return self._copy_with(cloned_df)
```

**Consolidation Recommendation:**
The copy semantics are appropriately different for each class:
- BrainData needs shallow copy for performance (large arrays, fitted models)
- Adjacency can use deepcopy (smaller data)
- DesignMatrix uses Polars-native clone

**Status:** Appropriate differentiation - no consolidation needed

---

### 1.6 Write/IO Methods

**Locations:**
- `nltools/data/brain_data.py:1395-1446` (`write()`)
- `nltools/data/adjacency/__init__.py:768-797` (`write()`)

Both support HDF5 and CSV/Nifti output with similar structure:

```python
# Pattern in both classes:
def write(self, file_name, ...):
    if isinstance(file_name, Path):
        file_name = str(file_name)

    if (".h5" in file_name) or (".hdf5" in file_name):
        to_h5(self, file_name, obj_type="...")
    else:
        # Class-specific output (nifti vs csv)
        ...
```

**Consolidation Recommendation:**
The HDF5 path detection is duplicated. Extract to utility:

```python
# In nltools/utils.py
def is_h5_path(file_name: str | Path) -> bool:
    """Check if file path is HDF5."""
    file_str = str(file_name)
    return ".h5" in file_str or ".hdf5" in file_str
```

**Estimated Effort:** 30 minutes
**Lines Saved:** ~10 lines

---

## 2. Common Utility Code to Extract

### 2.1 Validation Patterns (ALREADY CONSOLIDATED)

**Good Practice Found:**

The codebase already has well-organized validation modules:
- `nltools/data/_validation.py` - BrainData-specific validation (280 lines)
- `nltools/algorithms/_validation.py` - Algorithm-specific validation (380 lines)

These modules contain:
- `validate_frame()` - DataFrame validation
- `validate_brain_data_shapes()` - Shape compatibility
- `validate_arithmetic_operand()` - Arithmetic type checking
- `validate_data_type()` - Input type detection
- `validate_list_data()` - List homogeneity
- `validate_index_operations()` - Indexing bounds
- `validate_append_shapes()` - Append compatibility

**Status:** Well-consolidated. No further action needed.

---

### 2.2 isinstance() Checks

**Pattern Found (75+ occurrences):**
```python
isinstance(y, (int, np.integer, float, np.floating))
```

**Locations:** Throughout `brain_data.py`, `adjacency/__init__.py`, `collection.py`

**Consolidation Recommendation:**
Create type check utilities:

```python
# In nltools/utils.py or nltools/data/_validation.py
def is_scalar_numeric(value) -> bool:
    """Check if value is a scalar numeric (int or float)."""
    return isinstance(value, (int, np.integer, float, np.floating))

def is_array_like(value) -> bool:
    """Check if value is array-like (list or ndarray)."""
    return isinstance(value, (list, np.ndarray))
```

**Estimated Effort:** 1 hour
**Lines Saved:** ~20 lines (more importantly: consistency)

---

### 2.3 H5 Path Detection

As noted in 1.6, the H5 path detection is duplicated across write methods.

---

## 3. Repeated Docstring Patterns

### 3.1 Axis Parameter Documentation

**Pattern repeated across classes:**
```python
Args:
    axis: (int) calculate [operation] over features (0) or data (1).
          For data it will be on upper triangle.
```

**Recommendation:** Not critical - docstrings should be specific to each method's context.

---

## 4. Similar Class Structures - Mixin Opportunities

### 4.1 Potential Mixins

| Mixin Name | Methods | Classes | Effort |
|------------|---------|---------|--------|
| `FisherTransformMixin` | `r_to_z`, `z_to_r` | BrainData, Adjacency | 30 min |
| `StatisticalMixin` | `_apply_func`, `mean`, `std`, `sum`, `median` | BrainData, Adjacency, BrainCollection | 2 hours |
| `ThresholdMixin` | `threshold` | BrainData, Adjacency, BrainCollection | 1 hour |

### 4.2 Not Recommended for Mixins

| Method Group | Reason |
|--------------|--------|
| `copy` | Different copy semantics per class |
| `write` | Different output formats (nifti vs csv vs polars) |
| `bootstrap` | Already uses functional core; class methods are thin wrappers |
| `arithmetic` | BrainData uses `_perform_arithmetic`, Adjacency should adopt same pattern |

---

## 5. Specific Consolidation Recommendations

### Priority 1: High Impact, Low Effort

| Recommendation | Location | Effort | Lines Saved |
|----------------|----------|--------|-------------|
| Add `_perform_arithmetic` to Adjacency | `adjacency/__init__.py` | 2 hours | 70 |
| Add `_apply_func` to Adjacency | `adjacency/__init__.py` | 1 hour | 50 |
| Create `FisherTransformMixin` | new file or `utils.py` | 30 min | 15 |
| Add `is_h5_path()` utility | `utils.py` | 30 min | 10 |

**Total Priority 1 Effort:** ~4 hours
**Total Lines Saved:** ~145 lines

### Priority 2: Medium Impact

| Recommendation | Location | Effort | Lines Saved |
|----------------|----------|--------|-------------|
| Add `is_scalar_numeric()` utility | `utils.py` or `_validation.py` | 1 hour | 20 |
| Standardize copy method signatures | All classes | 1 hour | 0 (clarity) |

### Priority 3: Documentation

| Recommendation | Location | Effort |
|----------------|----------|--------|
| Document existing validation utilities | `_validation.py` files | 1 hour |
| Add type hints to validation functions | `_validation.py` files | 2 hours |

---

## 6. Architecture Recommendations

### 6.1 Current State

```
nltools/data/
    __init__.py
    _validation.py          # BrainData validation (well-organized)
    brain_data.py           # Main class (4,785 lines)
    adjacency/__init__.py   # Adjacency class (1,968 lines)
    design_matrix.py        # DesignMatrix class (1,410 lines)
    collection.py           # BrainCollection (5,054 lines)
    fit_results.py          # Fit results container

nltools/algorithms/
    _validation.py          # Algorithm validation (well-organized)
    ...
```

### 6.2 Recommended Structure

```
nltools/data/
    __init__.py
    _validation.py          # Keep as-is (well-organized)
    _mixins.py              # NEW: FisherTransformMixin, StatisticalMixin
    _utils.py               # NEW: is_h5_path, is_scalar_numeric
    brain_data.py           # Inherits from mixins
    adjacency/__init__.py   # Inherits from mixins, uses _perform_arithmetic
    design_matrix.py        # Keep as-is (Polars-specific)
    collection.py           # Inherits from mixins
    fit_results.py          # Keep as-is
```

---

## 7. Summary

### What's Already Good

1. **Validation utilities** are well-consolidated in dedicated modules
2. **Bootstrap infrastructure** uses functional core pattern correctly
3. **BrainData** has good internal refactoring (`_perform_arithmetic`, `_apply_func`, `_shallow_copy_with_data`)
4. **DesignMatrix** uses Polars-native patterns appropriately
5. **Algorithms module** has consistent validation patterns

### What Needs Work

1. **Adjacency arithmetic** - 8 nearly-identical methods need refactoring
2. **Adjacency statistical methods** - Should adopt `_apply_func` pattern
3. **Fisher transforms** - Duplicated across classes
4. **Type checking utilities** - `isinstance()` patterns could be consolidated

### Total Estimated Effort

| Priority | Effort | Impact |
|----------|--------|--------|
| Priority 1 | 4 hours | ~145 lines saved, cleaner architecture |
| Priority 2 | 2 hours | Improved consistency |
| Priority 3 | 3 hours | Better documentation |
| **Total** | **9 hours** | Cleaner, more maintainable codebase |

---

## Appendix: File Line Counts

| File | Lines | Notes |
|------|-------|-------|
| `brain_data.py` | 4,785 | Main class, well-refactored internally |
| `collection.py` | 5,054 | BrainCollection, newer code |
| `adjacency/__init__.py` | 1,968 | Needs arithmetic/stats refactoring |
| `design_matrix.py` | 1,410 | Polars-based, clean design |
| `data/_validation.py` | 280 | Well-organized validation |
| `algorithms/_validation.py` | 380 | Well-organized validation |
