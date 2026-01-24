# API Shell Layer Audit Report

**Date:** 2026-01-24
**Scope:** `nltools/data/` - BrainData, Adjacency, DesignMatrix, BrainCollection
**Version Target:** v0.6.0

---

## Executive Summary

This audit identifies API inconsistencies across the imperative shell layer. The four main classes (`BrainData`, `Adjacency`, `DesignMatrix`, `BrainCollection`) have evolved independently, resulting in inconsistent naming conventions, return types, and parameter patterns that may confuse users.

**Key Findings:**
- 6 naming inconsistencies (empty/isempty, zscore/standardize, etc.)
- 4 return type inconsistencies
- 3 parameter signature inconsistencies
- 2 property vs method inconsistencies

---

## 1. Empty/IsEmpty Check Inconsistency

| Class | Implementation | File:Line | Type |
|-------|---------------|-----------|------|
| BrainData | `empty()` method (creates empty copy) | brain_data.py:2315 | method |
| BrainData | `isempty` property (checks if empty) | brain_data.py:2324 | property |
| Adjacency | `isempty` property (checks if empty) | adjacency/__init__.py:492 | property |
| DesignMatrix | `empty` property (checks if empty) | design_matrix.py:142 | property |
| BrainCollection | N/A | - | - |

**Issue:**
1. `BrainData.empty()` is a METHOD that creates a copy with empty data
2. `BrainData.isempty` is a PROPERTY that checks if empty
3. `Adjacency.isempty` is a PROPERTY that checks if empty
4. `DesignMatrix.empty` is a PROPERTY that checks if empty (different name!)

**Recommendation:**
- Standardize on `is_empty` property for checking (PEP8 snake_case)
- Rename `BrainData.empty()` method to `create_empty()` or `with_empty_data()`
- Add deprecation warnings for old names

---

## 2. Normalization Method Naming

| Class | Method Name | File:Line | Parameters |
|-------|-------------|-----------|------------|
| BrainData | `standardize(axis=0, method="center")` | brain_data.py:2899 | method=['center', 'zscore'] |
| DesignMatrix | `zscore(columns=None)` | design_matrix.py:230 | columns only |
| BrainCollection | `standardize(axis=0, method="center", ...)` | collection.py:2301 | delegates to BrainData |
| Adjacency | N/A | - | - |

**Issue:**
- `BrainData.standardize()` uses `method="zscore"` for z-scoring
- `DesignMatrix.zscore()` uses a different name entirely
- Inconsistent mental model for users

**Recommendation:**
- Add `DesignMatrix.standardize(method='zscore')` as preferred API
- Deprecate `DesignMatrix.zscore()` with redirect to `standardize()`
- Or: Add `BrainData.zscore()` as alias for `standardize(method='zscore')`

---

## 3. Mean/Aggregate Return Type Inconsistency

| Class | Method | Returns (single) | Returns (stacked, axis=0) | Returns (axis=1) |
|-------|--------|------------------|---------------------------|------------------|
| BrainData | `mean(axis=0)` | float | BrainData | np.array |
| Adjacency | `mean(axis=0)` | float | Adjacency | np.array |
| BrainCollection | `mean(axis=0)` | BrainData | BrainData | BrainCollection |
| DesignMatrix | N/A | - | - | - |

**BrainData.mean()** (brain_data.py:1187):
```python
def mean(self, axis=0):
    return self._apply_func(np.mean, axis)
    # axis=0 returns BrainData, axis=1 returns np.array
```

**Adjacency.mean()** (adjacency/__init__.py:563):
```python
def mean(self, axis=0):
    if self.is_single_matrix:
        return np.nanmean(self.data)  # Returns float!
    else:
        if axis == 0:
            return Adjacency(...)  # Returns Adjacency
        elif axis == 1:
            return np.nanmean(self.data, axis=axis)  # Returns np.array
```

**Issue:**
- Return types are consistent between BrainData and Adjacency (good)
- But DesignMatrix lacks `mean()` method entirely

**Recommendation:**
- Consider adding basic statistics to DesignMatrix if needed
- Document return type patterns clearly in docstrings

---

## 4. Verbose Parameter Type Inconsistency

| Class | Location | Type | Default |
|-------|----------|------|---------|
| BrainData | `__init__()` | bool | `False` |
| BrainData | `NiftiMasker` kwarg | int (0/1) | 0 |
| Adjacency | `__init__()` kwargs | bool | `False` |
| DesignMatrix | `append()` | bool | `False` |
| DesignMatrix | `clean()` | bool | `True` |

**File References:**
- brain_data.py:150: `self.verbose = kwargs.pop("verbose", False)`
- brain_data.py:255: `verbose=kwargs.get("verbose", 0)`
- adjacency/__init__.py:75: `verbose = kwargs.pop("verbose", False)`
- design_matrix.py:632: `verbose: bool = False`
- design_matrix.py:957: `verbose: bool = True`

**Issue:**
- NiftiMasker expects int (0/1) but most methods use bool
- Default varies (False in most, True in `clean()`)

**Recommendation:**
- Standardize all `verbose` parameters to `bool` type
- Use `int(verbose)` when passing to nilearn
- Consider using a `verbosity: int` parameter for granular control (0/1/2)

---

## 5. Inplace Parameter Inconsistency

| Class | Method | Has inplace? | Default |
|-------|--------|--------------|---------|
| BrainData | `fit()` | Yes | `True` |
| BrainData | `__iadd__`, `__isub__`, etc. | Yes (implicit) | `True` |
| Adjacency | `append()` | No | N/A (returns new) |
| DesignMatrix | All methods | No | N/A (returns new) |
| BrainCollection | All methods | No | N/A (returns new) |

**File References:**
- brain_data.py:1554: `inplace=True`
- brain_data.py:1024: `return self._perform_arithmetic(y, np.add, "add", inplace=True)`

**Issue:**
- Only BrainData uses `inplace` parameter
- Adjacency and DesignMatrix always return new objects
- This is inconsistent but may be by design (immutability preference)

**Recommendation:**
- Document the immutability pattern for Adjacency/DesignMatrix
- Consider deprecating `inplace=True` in BrainData.fit() for v0.7.0
- Align with pandas-style `inplace=False` as default

---

## 6. Append Method Signature Inconsistency

| Class | Signature | File:Line |
|-------|-----------|-----------|
| BrainData | `append(data, ignore_attrs=False, **kwargs)` | brain_data.py:2253 |
| Adjacency | `append(data)` | adjacency/__init__.py:740 |
| DesignMatrix | `append(dm, axis=0, keep_separate=True, unique_cols=None, fill_na=0, verbose=False)` | design_matrix.py:625 |

**Issue:**
- Very different signatures
- DesignMatrix supports `axis` parameter (horizontal/vertical)
- BrainData has `ignore_attrs` for X/Y handling
- Adjacency is minimal

**Recommendation:**
- This is likely intentional due to different data structures
- Document the differences clearly in each class docstring
- Consider adding `axis` parameter to BrainData.append() for consistency

---

## 7. Copy Method Consistency

| Class | Method | Implementation | File:Line |
|-------|--------|----------------|-----------|
| BrainData | `copy()` | `deepcopy(self)` | brain_data.py:1161 |
| Adjacency | `copy()` | `deepcopy(self)` | adjacency/__init__.py:736 |
| DesignMatrix | `copy()` | `_copy_with(cloned_df)` | design_matrix.py:208 |

**Status:** Consistent (all provide deep copies) - **No action needed**

---

## 8. Write/Save Method Naming

| Class | Method | File:Line |
|-------|--------|-----------|
| BrainData | `write(file_name)` | brain_data.py:1395 |
| Adjacency | `write(file_name, method="long")` | adjacency/__init__.py:768 |
| DesignMatrix | N/A | - |
| BrainCollection | N/A | - |

**Issue:**
- DesignMatrix lacks `write()` method
- BrainCollection lacks `write()` method

**Recommendation:**
- Add `DesignMatrix.write()` for TSV/CSV export
- Add `BrainCollection.write()` for batch export
- Consider alias `save()` for discoverability

---

## 9. Conversion Method Naming

| Class | Method | File:Line |
|-------|--------|-----------|
| BrainData | `to_nifti()` | brain_data.py:1254 |
| DesignMatrix | `to_numpy()` | design_matrix.py:1266 |
| DesignMatrix | `_to_pandas()` (private) | design_matrix.py:1238 |
| Adjacency | `squareform()` | adjacency/__init__.py:496 |

**Issue:**
- No `to_numpy()` on BrainData (data attribute is already numpy)
- `_to_pandas()` is private on DesignMatrix
- Adjacency uses `squareform()` instead of `to_square()`

**Recommendation:**
- Consider making `DesignMatrix._to_pandas()` public as `to_pandas()`
- Add `Adjacency.to_square()` alias for `squareform()` (clearer naming)

---

## 10. Axis Parameter Semantics

| Class | axis=0 means | axis=1 means | axis=2 means |
|-------|--------------|--------------|--------------|
| BrainData | across images | within images | N/A |
| Adjacency | across matrices | across upper triangle | N/A |
| BrainCollection | across images | across timepoints | across voxels |
| DesignMatrix | across rows | across columns | N/A |

**Note:** This is consistent with numpy conventions - **No action needed**

---

## Summary Table of Recommended Changes

| Priority | Issue | Recommendation | Breaking? |
|----------|-------|----------------|-----------|
| HIGH | `empty` vs `isempty` vs `is_empty` | Standardize to `is_empty` property | Yes (deprecate old) |
| HIGH | `zscore()` vs `standardize()` | Add `standardize()` to DesignMatrix | No |
| MEDIUM | `verbose` type (bool vs int) | Standardize to bool | No |
| MEDIUM | `inplace` default | Document pattern, deprecate for v0.7 | Future |
| LOW | Missing `write()` methods | Add to DesignMatrix, BrainCollection | No |
| LOW | `squareform()` naming | Add `to_square()` alias | No |
| LOW | `_to_pandas()` visibility | Make public | No |

---

## Files Audited

1. `/Users/esh/Documents/pypackages/nltools/nltools/data/brain_data.py`
2. `/Users/esh/Documents/pypackages/nltools/nltools/data/adjacency/__init__.py`
3. `/Users/esh/Documents/pypackages/nltools/nltools/data/design_matrix.py`
4. `/Users/esh/Documents/pypackages/nltools/nltools/data/collection.py`

---

## Next Steps

1. Create GitHub issues for each HIGH priority item
2. Add deprecation warnings in v0.6.0
3. Document API changes in CHANGELOG
4. Update docstrings to reflect standardized patterns
