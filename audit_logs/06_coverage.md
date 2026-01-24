# Test Coverage Audit Report

**Date:** 2026-01-24
**Version Target:** v0.6.0
**Methodology:** Static analysis mapping public methods to test file usage patterns

---

## Executive Summary

| Class | Methods | Tested | Coverage |
|-------|---------|--------|----------|
| BrainData | 54 | 41 | **75.9%** |
| Adjacency | 31 | 25 | **80.6%** |
| DesignMatrix | 21 | 18 | **85.7%** |
| Collection | 57 | 46 | **80.7%** |
| **Overall** | **163** | **130** | **79.8%** |

---

## Detailed Analysis by Class

### 1. BrainData (75.9% Coverage)

**Source:** `nltools/data/brain_data.py`
**Test Files:**
- `nltools/tests/shell/test_brain_data.py` (main tests)
- `nltools/tests/shell/test_brain_data_icc.py` (ICC-specific tests)
- `nltools/tests/shell/test_brain_data_plot.py` (visualization tests)

#### Tested Methods (41)

| Method | Status |
|--------|--------|
| align | Tested |
| append | Tested |
| apply_mask | Tested |
| astype | Tested |
| bootstrap | Tested |
| compute_contrasts | Tested |
| copy | Tested |
| cv | Tested |
| data | Tested |
| decompose | Tested |
| detrend | Tested |
| distance | Tested |
| empty | Tested |
| extract_roi | Tested |
| filter | Tested |
| fit | Tested |
| icc | Tested |
| mean | Tested |
| median | Tested |
| plot | Tested |
| predict | Tested |
| predictions | Tested |
| r_to_z | Tested |
| randomise | Tested |
| regions | Tested |
| regress | Tested |
| resample_to | Tested |
| scale | Tested |
| scores | Tested |
| shape | Tested |
| similarity | Tested |
| smooth | Tested |
| standardize | Tested |
| std | Tested |
| sum | Tested |
| temporal_resample | Tested |
| threshold | Tested |
| to_nifti | Tested |
| ttest | Tested |
| write | Tested |
| z_to_r | Tested |

#### Untested Methods (13)

| Method | Priority | Reason |
|--------|----------|--------|
| `multivariate_similarity` | **HIGH** | Core analysis method |
| `pipe` | **HIGH** | Key composition pattern |
| `normalize` | **MEDIUM** | Common preprocessing |
| `reduce` | **MEDIUM** | Aggregation utility |
| `transform_pairwise` | **MEDIUM** | Similarity analysis |
| `plot_flatmap` | **LOW** | Visualization (harder to test) |
| `find_spikes` | **LOW** | Specialized QC utility |
| `dtype` | **LOW** | Property accessor |
| `isempty` | **LOW** | Property accessor |
| `mean_score` | **LOW** | Derived from scores |
| `n_steps` | **LOW** | Property accessor |
| `std_score` | **LOW** | Derived from scores |
| `upload_neurovault` | **LOW** | External service integration |

---

### 2. Adjacency (80.6% Coverage)

**Source:** `nltools/data/adjacency/__init__.py`
**Test File:** `nltools/tests/shell/test_adjacency.py`

#### Tested Methods (25)

| Method | Status |
|--------|--------|
| append | Tested |
| bootstrap | Tested |
| cluster_summary | Tested |
| copy | Tested |
| distance | Tested |
| distance_to_similarity | Tested |
| generate_permutations | Tested |
| mean | Tested |
| median | Tested |
| n_nodes | Tested |
| r_to_z | Tested |
| regress | Tested |
| shape | Tested |
| similarity | Tested |
| social_relations_model | Tested |
| square_shape | Tested |
| squareform | Tested |
| stats_label_distance | Tested |
| std | Tested |
| sum | Tested |
| threshold | Tested |
| to_graph | Tested |
| ttest | Tested |
| write | Tested |
| z_to_r | Tested |

#### Untested Methods (6)

| Method | Priority | Reason |
|--------|----------|--------|
| `plot` | **MEDIUM** | Visualization method |
| `plot_label_distance` | **MEDIUM** | Visualization method |
| `plot_mds` | **MEDIUM** | Visualization method |
| `plot_silhouette` | **MEDIUM** | Visualization method |
| `isempty` | **LOW** | Property accessor |
| `vector_shape` | **LOW** | Property accessor |

---

### 3. DesignMatrix (85.7% Coverage)

**Source:** `nltools/data/design_matrix.py`
**Test File:** `nltools/tests/shell/test_design_matrix.py`

#### Tested Methods (18)

| Method | Status |
|--------|--------|
| add_dct_basis | Tested |
| add_poly | Tested |
| append | Tested |
| clean | Tested |
| columns | Tested |
| convolve | Tested |
| details | Tested |
| downsample | Tested |
| drop | Tested |
| empty | Tested |
| fillna | Tested |
| heatmap | Tested |
| replace_data | Tested |
| shape | Tested |
| to_numpy | Tested |
| upsample | Tested |
| vif | Tested |
| zscore | Tested |

#### Untested Methods (3)

| Method | Priority | Reason |
|--------|----------|--------|
| `copy` | **MEDIUM** | Core utility (may be implicitly tested) |
| `reset_index` | **LOW** | Utility method |
| `sum` | **LOW** | Aggregation (may be implicitly tested) |

---

### 4. Collection (80.7% Coverage)

**Source:** `nltools/data/collection.py`
**Test File:** `nltools/tests/shell/test_collection.py`

#### Tested Methods (46)

| Method | Status |
|--------|--------|
| align | Tested |
| anova | Tested |
| betas | Tested |
| compute_contrasts | Tested |
| cv | Tested |
| detrend | Tested |
| filter | Tested |
| fit | Tested |
| fit_from_events | Tested |
| fit_glm | Tested |
| fit_ridge | Tested |
| from_bids | Tested |
| from_glob | Tested |
| from_stacked | Tested |
| is_loaded | Tested |
| isc | Tested |
| isc_test | Tested |
| iter_batches | Tested |
| load | Tested |
| map | Tested |
| mask | Tested |
| max | Tested |
| mean | Tested |
| median | Tested |
| memory_estimate | Tested |
| metadata | Tested |
| min | Tested |
| n_images | Tested |
| n_voxels | Tested |
| permutation_test | Tested |
| permutation_test2 | Tested |
| predict | Tested |
| results | Tested |
| scores | Tested |
| select_feature | Tested |
| shape | Tested |
| standardize | Tested |
| std | Tested |
| sum | Tested |
| threshold | Tested |
| to_list | Tested |
| to_stacked | Tested |
| to_tensor | Tested |
| ttest | Tested |
| ttest2 | Tested |
| var | Tested |

#### Untested Methods (11)

| Method | Priority | Reason |
|--------|----------|--------|
| `pipe` | **HIGH** | Key composition pattern |
| `pool` | **HIGH** | Core pooling operation |
| `normalize` | **MEDIUM** | Common preprocessing |
| `reduce` | **MEDIUM** | Aggregation utility |
| `smooth` | **MEDIUM** | Common preprocessing |
| `unload` | **MEDIUM** | Memory management |
| `mean_score` | **LOW** | Derived property |
| `n_folds` | **LOW** | Property accessor |
| `n_steps` | **LOW** | Property accessor |
| `n_subjects` | **LOW** | Property accessor |
| `std_score` | **LOW** | Derived property |

---

## Priority Recommendations

### High Priority (Add tests immediately)

1. **`BrainData.multivariate_similarity`** - Core analysis method for multi-voxel pattern analysis
2. **`BrainData.pipe` / `Collection.pipe`** - Critical for composition pattern, used in chaining operations
3. **`Collection.pool`** - Core method for pooling data across subjects

### Medium Priority (Add tests before release)

4. **`BrainData.normalize`** / **`Collection.normalize`** - Common preprocessing step
5. **`BrainData.reduce`** / **`Collection.reduce`** - Aggregation operations
6. **`Collection.smooth`** - Common preprocessing
7. **`Collection.unload`** - Memory management for large datasets
8. **`BrainData.transform_pairwise`** - Used in similarity analyses
9. **Adjacency plotting methods** (`plot`, `plot_label_distance`, `plot_mds`, `plot_silhouette`)
10. **`DesignMatrix.copy`** - Core utility method

### Low Priority (Nice to have)

11. Property accessors (`dtype`, `isempty`, `n_steps`, `n_folds`, etc.) - Often implicitly tested
12. Derived score properties (`mean_score`, `std_score`) - Low complexity
13. **`BrainData.upload_neurovault`** - External service, difficult to test in CI
14. **`BrainData.plot_flatmap`** - Visualization, harder to test
15. **`BrainData.find_spikes`** - Specialized QC utility

---

## Test Infrastructure Notes

### Test File Organization

```
nltools/tests/
├── shell/                          # Imperative shell tests
│   ├── test_brain_data.py          # Main BrainData tests
│   ├── test_brain_data_icc.py      # ICC-specific tests
│   ├── test_brain_data_plot.py     # Visualization tests
│   ├── test_adjacency.py           # Adjacency tests
│   ├── test_design_matrix.py       # DesignMatrix tests
│   └── test_collection.py          # Collection tests
├── core/                           # Functional core tests
│   ├── test_hyperalignment.py
│   ├── test_srm.py
│   ├── test_ridge.py
│   └── ... (algorithm tests)
└── support/                        # Integration tests
    └── ...
```

### Recommendations for Test Additions

1. **Create `test_brain_data_pipe.py`** - Dedicated tests for pipe/composition patterns
2. **Add parametrized tests** for normalize/smooth/reduce across both BrainData and Collection
3. **Consider visual regression testing** for Adjacency plot methods using pytest-mpl or similar

---

## Methodology

This analysis was performed using:

1. **Method Extraction:** `rg '^\s{4}def\s+[a-z][a-z0-9_]*\('` to extract public method definitions (excluding `_` prefixed private methods)

2. **Test Coverage Detection:** Two-pass search:
   - Search for `.method_name(` patterns in test files
   - Search for `test.*method_name` patterns in test function names

3. **Classification:** Methods were classified by:
   - Whether they appear in test files (tested/untested)
   - Importance for v0.6.0 release (high/medium/low priority)

### Limitations

- This is static analysis, not runtime coverage measurement
- Methods may be indirectly tested through other method calls
- Some methods may be tested in integration tests not scanned
- Visualization methods are harder to test automatically

---

## Next Steps

1. Run actual coverage report: `uv run pytest --cov=nltools --cov-report=html`
2. Create issues in beads for high-priority gaps
3. Add targeted tests for `pipe`, `pool`, and `multivariate_similarity`
4. Consider adding snapshot/visual testing for plot methods
