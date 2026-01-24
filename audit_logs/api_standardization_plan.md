# API Standardization Plan for nltools v0.6.0

**Date:** 2026-01-24
**Goal:** Establish consistent API patterns across BrainData, Adjacency, DesignMatrix, and Collection

---

## 1. Property Naming Standardization

### 1.1 Empty Check (`is_empty`)

**Current State:**
| Class | Current | Type |
|-------|---------|------|
| BrainData | `empty()` + `isempty` | method + property |
| Adjacency | `isempty` | property |
| DesignMatrix | `empty` | property |
| Collection | N/A | - |

**Target State:**
- All classes: `is_empty` property (PEP8 snake_case)
- `BrainData.empty()` renamed to `create_empty()`

**Implementation:**
```python
# In all classes
@property
def is_empty(self) -> bool:
    """Check if object contains no data."""
    return self.data is None or len(self.data) == 0

# Add deprecation to old names
@property
def isempty(self) -> bool:
    warnings.warn(
        "isempty is deprecated. Use is_empty instead.",
        DeprecationWarning, stacklevel=2
    )
    return self.is_empty
```

---

## 2. Method Naming Standardization

### 2.1 Normalization Methods

**Current State:**
| Class | Method | Parameters |
|-------|--------|------------|
| BrainData | `standardize()` | `axis`, `method=['center', 'zscore']` |
| DesignMatrix | `zscore()` | `columns` |
| Collection | `standardize()` | `axis`, `method` |
| Adjacency | N/A | - |

**Target State:**
- All classes with normalization: `standardize(method='zscore')`
- Keep `zscore()` as alias for backward compatibility

**Implementation for DesignMatrix:**
```python
def standardize(self, method: str = "zscore", columns: list | None = None):
    """Standardize columns using specified method.

    Args:
        method: 'zscore' (default) or 'center'
        columns: Specific columns to standardize
    """
    if method == "zscore":
        return self.zscore(columns)
    elif method == "center":
        # Implement centering
        pass
```

---

### 2.2 Conversion Methods (`to_*`)

**Current State:**
| Class | Method | Returns |
|-------|--------|---------|
| BrainData | `to_nifti()` | NiftiImage |
| DesignMatrix | `to_numpy()` | np.ndarray |
| DesignMatrix | `_to_pandas()` | pd.DataFrame (private) |
| Adjacency | `squareform()` | np.ndarray |

**Target State:**
- Make `DesignMatrix._to_pandas()` public as `to_pandas()`
- Add `Adjacency.to_square()` as alias for `squareform()`

**Implementation:**
```python
# Adjacency
def to_square(self):
    """Convert to square matrix format. Alias for squareform()."""
    return self.squareform()

# DesignMatrix - rename _to_pandas to to_pandas
def to_pandas(self):
    """Convert to pandas DataFrame."""
    return pd.DataFrame(self._df.to_numpy(), columns=self.columns)
```

---

## 3. Parameter Standardization

### 3.1 `verbose` Parameter

**Current State:**
| Location | Type | Default |
|----------|------|---------|
| BrainData.__init__ | bool | False |
| NiftiMasker kwargs | int (0/1) | 0 |
| DesignMatrix.clean | bool | True |
| DesignMatrix.append | bool | False |

**Target State:**
- All `verbose` parameters: `bool` type
- Default: `False` (except where output is expected, like `clean()`)
- Convert to int when passing to nilearn internally

**Implementation:**
```python
def __init__(self, ..., verbose: bool = False):
    # Convert to int for nilearn
    nilearn_verbose = int(verbose)
    self.masker = NiftiMasker(..., verbose=nilearn_verbose)
```

---

### 3.2 `axis` Parameter Semantics

**Current State (Consistent - No Changes Needed):**
| Class | axis=0 | axis=1 |
|-------|--------|--------|
| BrainData | across images | within images |
| Adjacency | across matrices | across upper triangle |
| Collection | across images | across timepoints |
| DesignMatrix | across rows | across columns |

This follows numpy conventions - **no changes needed**.

---

### 3.3 `inplace` Parameter

**Current State:**
| Class | Has `inplace`? | Default |
|-------|----------------|---------|
| BrainData | Yes (in `fit()`) | True |
| Adjacency | No | - |
| DesignMatrix | No | - |
| Collection | No | - |

**Target State (v0.7.0):**
- Deprecate `inplace=True` in BrainData
- Align with pandas-style `inplace=False` as default
- All classes return new objects (immutability pattern)

**For v0.6.0:**
- Document the immutability pattern for Adjacency/DesignMatrix
- Add warning about future deprecation of `inplace=True`

---

## 4. Return Type Standardization

### 4.1 Statistical Methods

**Current State:**
| Method | BrainData | Adjacency | Collection |
|--------|-----------|-----------|------------|
| `mean(axis=0)` | BrainData | Adjacency | BrainData |
| `mean(axis=1)` | np.array | np.array | BrainCollection |
| Single item mean | float | float | BrainData |

**Target State:** Current behavior is intentional and consistent - **no changes needed**.

---

## 5. Missing Method Additions

### 5.1 `write()` Methods

**Current State:**
| Class | Has `write()`? |
|-------|----------------|
| BrainData | Yes |
| Adjacency | Yes |
| DesignMatrix | No |
| Collection | No |

**Target State:**
- Add `DesignMatrix.write()` for TSV/CSV export
- Add `Collection.write()` for batch export

**Implementation:**
```python
# DesignMatrix
def write(self, file_name: str, format: str = "tsv"):
    """Write design matrix to file.

    Args:
        file_name: Output file path
        format: 'tsv', 'csv', or 'h5'
    """
    if is_h5_path(file_name):
        to_h5(self, file_name, obj_type="design_matrix")
    elif format == "csv":
        self._df.write_csv(file_name)
    else:
        self._df.write_csv(file_name, separator="\t")
```

---

## 6. Signature Alignment

### 6.1 `append()` Method

**Current Signatures:**
```python
# BrainData
def append(self, data, ignore_attrs=False, **kwargs)

# Adjacency
def append(self, data)

# DesignMatrix
def append(self, dm, axis=0, keep_separate=True, unique_cols=None, fill_na=0, verbose=False)
```

**Target State:**
- Keep different signatures (intentional due to different data structures)
- Document the differences clearly in each class docstring
- Consider adding `axis` parameter to BrainData.append() for horizontal stacking

---

## 7. Implementation Priority

### Phase 1: v0.6.0 (Breaking)
1. Add `is_empty` property to all classes
2. Add deprecation warnings to `isempty`/`empty`
3. Add `standardize()` to DesignMatrix
4. Add `to_pandas()` to DesignMatrix
5. Add `to_square()` alias to Adjacency

### Phase 2: v0.6.x (Non-Breaking)
6. Add `DesignMatrix.write()`
7. Add `Collection.write()`
8. Standardize `verbose` to bool type

### Phase 3: v0.7.0 (Breaking)
9. Remove deprecated `isempty`/`empty` properties
10. Deprecate `inplace=True` in BrainData.fit()
11. Remove `zscore()` from DesignMatrix (use `standardize()`)

---

## 8. Verification Tests

Add tests to verify API consistency:

```python
def test_is_empty_property():
    """All data classes should have is_empty property."""
    from nltools.data import BrainData, Adjacency, DesignMatrix

    for cls in [BrainData, Adjacency, DesignMatrix]:
        assert hasattr(cls, 'is_empty')
        assert isinstance(getattr(cls, 'is_empty'), property)

def test_standardize_method():
    """BrainData, DesignMatrix, Collection should have standardize()."""
    from nltools.data import BrainData, DesignMatrix
    from nltools.data.collection import BrainCollection

    for cls in [BrainData, DesignMatrix, BrainCollection]:
        assert hasattr(cls, 'standardize')
        assert callable(getattr(cls, 'standardize'))
```

---

*Generated from API consistency audit on 2026-01-24*
