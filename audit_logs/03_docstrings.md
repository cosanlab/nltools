# Docstring Audit Report for nltools v0.6.0

**Generated:** 2026-01-24
**Scope:** All Python source files in `nltools/` (excluding tests)

---

## Executive Summary

The nltools codebase has **excellent docstring presence** but **incomplete docstring content** in many areas. While 94-99% of public items have docstrings, many are missing required Google-style sections (Args, Returns, Raises) and examples.

| Metric | Status | Notes |
|--------|--------|-------|
| Docstring Presence | **GOOD** | 99.4% of public methods have docstrings |
| Args/Returns Sections | **NEEDS WORK** | 248 public items missing required sections |
| Examples | **NEEDS WORK** | Key API classes missing usage examples |
| Deprecated Docs | **GOOD** | Well-documented deprecation notices |

---

## Overall Coverage Statistics

| Category | Total | With Docstring | Coverage |
|----------|-------|----------------|----------|
| Classes | 56 | 53 | 94.6% |
| Functions (all) | 273 | 269 | 98.5% |
| Functions (public) | 179 | 177 | 98.9% |
| Methods (all) | 531 | 479 | 90.2% |
| Methods (public) | 313 | 311 | 99.4% |

---

## Per-File Coverage

| File | Classes | Functions | Methods | Public Coverage |
|------|---------|-----------|---------|-----------------|
| nltools/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/algorithms/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/algorithms/_random.py | 0/0 | 4/4 | 0/0 | 100% |
| nltools/algorithms/_shape_utils.py | 0/0 | 4/4 | 0/0 | 100% |
| nltools/algorithms/_validation.py | 0/0 | 17/17 | 0/0 | 100% |
| nltools/algorithms/alignment/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/algorithms/alignment/_local.py | 2/2 | 4/4 | 8/9 | 100% |
| nltools/algorithms/hrf.py | 0/0 | 6/6 | 0/0 | 100% |
| nltools/algorithms/hyperalignment.py | 1/1 | 1/1 | 5/5 | 100% |
| nltools/algorithms/inference/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/algorithms/inference/bootstrap.py | 1/1 | 13/13 | 2/3 | 100% |
| nltools/algorithms/inference/correlation.py | 0/0 | 7/7 | 0/0 | 100% |
| nltools/algorithms/inference/icc.py | 0/0 | 6/6 | 0/0 | 100% |
| nltools/algorithms/inference/isc.py | 0/0 | 19/19 | 0/0 | 100% |
| nltools/algorithms/inference/matrix.py | 0/0 | 8/8 | 0/0 | 100% |
| nltools/algorithms/inference/one_sample.py | 0/0 | 3/3 | 0/0 | 100% |
| nltools/algorithms/inference/timeseries.py | 0/0 | 8/8 | 0/0 | 100% |
| nltools/algorithms/inference/two_sample.py | 0/0 | 3/3 | 0/0 | 100% |
| nltools/algorithms/inference/utils.py | 0/0 | 5/5 | 0/0 | N/A |
| nltools/algorithms/ridge/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/algorithms/ridge/_core.py | 0/0 | 2/2 | 0/0 | 100% |
| nltools/algorithms/ridge/backends/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/algorithms/ridge/backends/_utils.py | 0/0 | 4/4 | 0/0 | 100% |
| nltools/algorithms/ridge/backends/numpy.py | 0/0 | 11/11 | 0/0 | 100% |
| nltools/algorithms/ridge/backends/torch.py | 0/0 | 22/22 | 0/0 | 100% |
| nltools/algorithms/ridge/backends/torch_cuda.py | 0/0 | 7/7 | 0/0 | 100% |
| nltools/algorithms/ridge/solvers.py | 0/0 | 3/3 | 0/0 | 100% |
| nltools/algorithms/ridge/utils.py | 0/0 | 5/5 | 0/0 | 100% |
| nltools/algorithms/srm.py | 2/2 | 1/1 | 14/16 | 100% |
| nltools/analysis.py | 1/1 | 0/0 | 3/4 | 100% |
| nltools/backends.py | 1/1 | 3/3 | 7/8 | 100% |
| nltools/cache.py | 1/1 | 3/3 | 7/8 | 100% |
| nltools/cross_validation.py | 1/1 | 0/0 | 1/4 | 100% |
| nltools/data/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/data/_validation.py | 0/0 | 7/7 | 0/0 | 100% |
| nltools/data/adjacency/__init__.py | 1/1 | 0/0 | 34/46 | 100% |
| nltools/data/brain_data.py | 3/3 | 1/1 | 100/110 | 96% |
| nltools/data/collection.py | 4/4 | 3/3 | 106/109 | 100% |
| nltools/data/design_matrix.py | 1/1 | 0/0 | 38/38 | 100% |
| nltools/data/fit_results.py | 1/1 | 0/0 | 2/2 | 100% |
| nltools/datasets.py | 0/0 | 7/7 | 0/0 | 100% |
| nltools/file_reader.py | 0/0 | 1/1 | 0/0 | 100% |
| nltools/mask.py | 0/0 | 4/4 | 0/0 | 100% |
| nltools/models/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/models/base.py | 1/1 | 0/0 | 6/7 | 100% |
| nltools/models/glm.py | 1/1 | 0/0 | 8/9 | 100% |
| nltools/models/ridge.py | 1/1 | 0/0 | 4/5 | 100% |
| nltools/neighborhoods.py | 1/1 | 1/1 | 6/7 | 100% |
| nltools/pipelines/__init__.py | 0/0 | 0/0 | 0/0 | N/A |
| nltools/pipelines/base.py | 6/6 | 0/0 | 20/20 | 100% |
| nltools/pipelines/cv.py | 2/2 | 0/0 | 14/14 | 100% |
| nltools/pipelines/multi_subject.py | 1/1 | 0/0 | 13/14 | 100% |
| nltools/pipelines/pool.py | 3/3 | 0/0 | 16/19 | 100% |
| nltools/pipelines/results.py | 5/5 | 0/0 | 16/21 | 100% |
| nltools/pipelines/steps.py | 8/8 | 0/0 | 17/18 | 100% |
| nltools/pipelines/terminals.py | 3/3 | 0/0 | 8/8 | 100% |
| nltools/plotting.py | 0/0 | 17/18 | 0/0 | 100% |
| nltools/prefs.py | 1/1 | 1/1 | 3/4 | 100% |
| nltools/simulator.py | 0/2 | 0/0 | 21/23 | 100% |
| nltools/stats.py | 0/0 | 39/40 | 0/0 | 100% |
| nltools/utils.py | 0/1 | 19/21 | 0/0 | 88% |
| nltools/version.py | 0/0 | 0/0 | 0/0 | N/A |

---

## Issues by Type Summary

| Issue Type | Total | Public | Priority |
|------------|-------|--------|----------|
| Missing Docstring (class) | 3 | 3 | HIGH |
| Missing Docstring (function) | 4 | 2 | HIGH |
| Missing Docstring (method) | 52 | 2 | HIGH/LOW |
| Missing Args Section | 149 | 77 | MEDIUM |
| Missing Returns Section | 262 | 171 | MEDIUM |

---

## Priority 1: Public Items Missing Docstrings (CRITICAL)

These must be documented before v0.6.0 release.

| File | Line | Name | Type |
|------|------|------|------|
| `nltools/data/brain_data.py` | 4632 | `BrainDataPipeline.cv` | property |
| `nltools/data/brain_data.py` | 4636 | `BrainDataPipeline.n_steps` | property |
| `nltools/simulator.py` | 27 | `Simulator` | class |
| `nltools/simulator.py` | 484 | `SimulateGrid` | class |
| `nltools/utils.py` | 684 | `attempt_to_import` | function |
| `nltools/utils.py` | 695 | `all_same` | function |
| `nltools/utils.py` | 790 | `AmbiguityError` | class |

**Total: 7 public items missing docstrings**

---

## Priority 2: Key Public API Missing Args/Returns Sections

These are important public API methods that have docstrings but are missing required sections.

### Core Data Classes

#### BrainData (`nltools/data/brain_data.py`)

| Line | Method | Issue |
|------|--------|-------|
| 1177 | `BrainData.shape` | Missing Returns |
| 1183 | `BrainData.dtype` | Missing Returns |
| 1254 | `BrainData.to_nifti` | Missing Returns |
| 2070 | `BrainData.regress` | Missing Args (3 params) |
| 2315 | `BrainData.empty` | Missing Returns |
| 2324 | `BrainData.isempty` | Missing Returns |
| 2834 | `BrainData.r_to_z` | Missing Returns |
| 2844 | `BrainData.z_to_r` | Missing Returns |
| 4627 | `BrainDataPipeline.data` | Missing Returns |
| 4647 | `BrainDataPipeline.normalize` | Missing Args, Returns |
| 4653 | `BrainDataPipeline.reduce` | Missing Args, Returns |
| 4663 | `BrainDataPipeline.pipe` | Missing Args, Returns |
| 4760 | `BrainDataCVResult.scores` | Missing Returns |
| 4765 | `BrainDataCVResult.mean_score` | Missing Returns |
| 4770 | `BrainDataCVResult.std_score` | Missing Returns |
| 4775 | `BrainDataCVResult.predictions` | Missing Returns |

#### Adjacency (`nltools/data/adjacency/__init__.py`)

| Line | Method | Issue |
|------|--------|-------|
| 492 | `Adjacency.isempty` | Missing Returns |
| 496 | `Adjacency.squareform` | Missing Returns |
| 715 | `Adjacency.square_shape` | Missing Returns |
| 736 | `Adjacency.copy` | Missing Returns |
| 798 | `Adjacency.similarity` | Missing Returns |
| 986 | `Adjacency.r_to_z` | Missing Returns |
| 994 | `Adjacency.z_to_r` | Missing Returns |
| 1039 | `Adjacency.to_graph` | Missing Returns |
| 1189 | `Adjacency.plot_silhouette` | Missing Args (4 params), Returns |

#### DesignMatrix (`nltools/data/design_matrix.py`)

| Line | Method | Issue |
|------|--------|-------|
| 125 | `DesignMatrix.shape` | Missing Returns |
| 130 | `DesignMatrix.columns` | Missing Returns |
| 135 | `DesignMatrix.columns` (setter) | Missing Args |
| 142 | `DesignMatrix.empty` | Missing Returns |
| 196 | `DesignMatrix.fillna` | Missing Args, Returns |
| 203 | `DesignMatrix.drop` | Missing Args, Returns |
| 498 | `DesignMatrix.add_poly` | Missing Returns |
| 555 | `DesignMatrix.add_dct_basis` | Missing Returns |
| 625 | `DesignMatrix.append` | Missing Returns |

### Algorithms

#### HyperAlignment (`nltools/algorithms/hyperalignment.py`)

| Line | Method | Issue |
|------|--------|-------|
| 384 | `HyperAlignment.common_model_` | Missing Returns |

#### LocalAlignment (`nltools/algorithms/alignment/_local.py`)

| Line | Method | Issue |
|------|--------|-------|
| 94 | `PiecewiseNeighborhoods.iter_neighborhoods` | Missing Args |
| 503 | `LocalAlignment.fit` | Missing Args (2 params), Returns |
| 635 | `LocalAlignment.transform` | Missing Args, Returns |
| 735 | `LocalAlignment.fit_transform` | Missing Args (2 params), Returns |

### Stats & Utils

#### stats.py

| Line | Function | Issue |
|------|----------|-------|
| 87 | `pearson` | Missing Args (2 params), Returns |
| 638 | `fisher_z_to_r` | Missing Args, Returns |

#### utils.py

| Line | Function | Issue |
|------|----------|-------|
| 261 | `get_resource_path` | Missing Returns |
| 266 | `get_anatomical` | Missing Returns |
| 675 | `isiterable` | Missing Args, Returns |
| 699 | `concatenate` | Missing Args, Returns |
| 729 | `check_square_numpy_matrix` | Missing Args, Returns |
| 751 | `check_brain_data` | Missing Args (2 params), Returns |

### Validation Functions (`nltools/algorithms/_validation.py`)

All 17 validation functions are missing Returns sections:

- `validate_parallel_parameter` (line 20)
- `validate_parallel_parameter_matrix` (line 33)
- `validate_array_shape` (line 83)
- `validate_array_shape_range` (line 104)
- `validate_same_shape` (line 128)
- `validate_same_first_dimension` (line 152)
- `validate_metric_parameter` (line 176)
- `validate_how_parameter` (line 196)
- `validate_square_matrix` (line 209)
- `validate_n_samples` (line 223)
- `validate_percentiles` (line 240)
- `validate_alpha` (line 266)
- `validate_shape_compatibility` (line 280)
- `validate_bootstrap_method` (line 304)
- `validate_bootstrap_data` (line 327)
- `validate_isc_parameters` (line 353)

---

## Priority 3: Missing Examples on Key Public API

The following key classes and methods lack usage examples:

### Classes Without Examples (Should Have)

| File | Class | Status |
|------|-------|--------|
| `nltools/data/brain_data.py` | `BrainData` | **NO EXAMPLE** |
| `nltools/data/brain_data.py` | `BrainDataPipeline` | **NO EXAMPLE** |
| `nltools/data/brain_data.py` | `BrainDataCVResult` | **NO EXAMPLE** |
| `nltools/data/adjacency/__init__.py` | `Adjacency` | **NO EXAMPLE** |
| `nltools/simulator.py` | `Simulator` | **NO EXAMPLE** |
| `nltools/simulator.py` | `SimulateGrid` | **NO EXAMPLE** |
| `nltools/utils.py` | `AmbiguityError` | NO EXAMPLE (ok) |

### Classes With Examples (Good)

| File | Class |
|------|-------|
| `nltools/data/design_matrix.py` | `DesignMatrix` |
| `nltools/algorithms/hyperalignment.py` | `HyperAlignment` |
| `nltools/algorithms/srm.py` | `SRM` |
| `nltools/algorithms/srm.py` | `DetSRM` |
| `nltools/models/ridge.py` | `Ridge` |
| `nltools/models/glm.py` | `Glm` |

### Key Methods Missing Examples

#### BrainData Methods (High Priority)

- `mean` (line 1187)
- `std` (line 1213)
- `to_nifti` (line 1254)
- `resample_to` (line 1259)
- `regress` (line 2070)
- `similarity` (line 2360)
- `multivariate_similarity` (line 2406)
- `apply_mask` (line 2436)
- `threshold` (line 2927)
- `transform_pairwise` (line 3078)
- `decompose` (line 3417)
- `smooth` (line 3548)
- `temporal_resample` (line 3589)

#### Adjacency Methods

- `plot` (line 516)
- `mean` (line 563)
- `std` (line 611)
- `similarity` (line 798)
- `threshold` (line 1001)
- `plot_label_distance` (line 1087)
- `plot_silhouette` (line 1189)
- `plot_mds` (line 1314)
- `distance_to_similarity` (line 1413)
- `regress` (line 1494)

#### Algorithm fit/transform Methods

- `HyperAlignment.fit` (line 218)
- `HyperAlignment.transform` (line 388)
- `HyperAlignment.transform_subject` (line 485)
- `SRM.fit` (line 208)
- `SRM.transform` (line 276)
- `SRM.transform_subject` (line 485)
- `DetSRM.fit` (line 760)
- `DetSRM.transform` (line 825)
- `DetSRM.transform_subject` (line 974)

#### Model Methods

- `Ridge.fit` (line 114)
- `Ridge.predict` (line 264)
- `Glm.fit` (line 131)
- `Glm.predict` (line 270)

---

## Priority 4: Private Items (Lower Priority)

**Total: 52 private items missing docstrings**

These are `__init__`, `__repr__`, and other dunder methods that typically don't require extensive documentation.

<details>
<summary>Click to expand private items list</summary>

- `nltools/algorithms/alignment/_local.py:112` - `PiecewiseNeighborhoods.__repr__`
- `nltools/algorithms/inference/bootstrap.py:121` - `OnlineBootstrapStats.__init__`
- `nltools/algorithms/srm.py:200` - `SRM.__init__`
- `nltools/algorithms/srm.py:753` - `DetSRM.__init__`
- `nltools/analysis.py:36` - `Roc.__init__`
- `nltools/backends.py:38` - `Backend.__init__`
- `nltools/cache.py:107` - `CacheManager.__init__`
- `nltools/cross_validation.py:39` - `KFoldStratified.__init__`
- `nltools/cross_validation.py:44` - `KFoldStratified._make_test_folds`
- `nltools/cross_validation.py:53` - `KFoldStratified._iter_test_masks`
- `nltools/data/adjacency/__init__.py:60` - `Adjacency.__init__`
- `nltools/data/adjacency/__init__.py:266` - `Adjacency.__repr__`
- `nltools/data/adjacency/__init__.py:276` - `Adjacency.__getitem__`
- `nltools/data/adjacency/__init__.py:288` - `Adjacency.__len__`
- `nltools/data/adjacency/__init__.py:294` - `Adjacency.__iter__`
- `nltools/data/adjacency/__init__.py:298` - `Adjacency.__add__`
- `nltools/data/adjacency/__init__.py:312` - `Adjacency.__radd__`
- `nltools/data/adjacency/__init__.py:326` - `Adjacency.__sub__`
- `nltools/data/adjacency/__init__.py:340` - `Adjacency.__rsub__`
- `nltools/data/adjacency/__init__.py:354` - `Adjacency.__mul__`
- `nltools/data/adjacency/__init__.py:368` - `Adjacency.__rmul__`
- `nltools/data/adjacency/__init__.py:382` - `Adjacency.__truediv__`
- `nltools/data/brain_data.py:916` - `BrainData.__repr__`
- `nltools/data/brain_data.py:943` - `BrainData.__getitem__`
- `nltools/data/brain_data.py:963` - `BrainData.__setitem__`
- `nltools/data/brain_data.py:980` - `BrainData.__len__`
- `nltools/data/brain_data.py:1039` - `BrainData.__iter__`
- `nltools/data/brain_data.py:4618` - `BrainDataPipeline.__init__`
- `nltools/data/brain_data.py:4755` - `BrainDataCVResult.__init__`
- `nltools/data/brain_data.py:4784` - `BrainDataCVResult.__repr__`
- `nltools/data/collection.py:4789` - `FittedBrainCollection.__init__`
- `nltools/data/collection.py:5025` - `FittedBrainCollection.__repr__`
- `nltools/data/collection.py:5038` - `FittedBrainCollection.__len__`
- `nltools/models/base.py:26` - `BaseModel.__init__`
- `nltools/models/glm.py:91` - `Glm.__init__`
- `nltools/models/ridge.py:87` - `Ridge.__init__`
- `nltools/neighborhoods.py:129` - `SphereNeighborhoods.__repr__`
- `nltools/pipelines/multi_subject.py:480` - `MultiSubjectPipeline.__repr__`
- `nltools/pipelines/pool.py:414` - `PooledData.__repr__`
- `nltools/pipelines/pool.py:498` - `StatResult.__repr__`
- `nltools/pipelines/pool.py:530` - `ResultDict.__repr__`
- `nltools/pipelines/results.py:46` - `FoldResult.__repr__`
- `nltools/pipelines/results.py:168` - `CVResult.__repr__`
- `nltools/pipelines/results.py:204` - `ISCResult.__repr__`
- `nltools/pipelines/results.py:256` - `RSAResult.__repr__`
- `nltools/pipelines/results.py:382` - `PermutationResult.__repr__`
- `nltools/pipelines/steps.py:566` - `AlignStep.__init__`
- `nltools/plotting.py:130` - `_viewer`
- `nltools/prefs.py:72` - `MNI_Template_Factory.__repr__`
- `nltools/simulator.py:28` - `Simulator.__init__`
- `nltools/simulator.py:485` - `SimulateGrid.__init__`
- `nltools/stats.py:650` - `_permute_group`

</details>

---

## Deprecated Functions Documentation Status

The codebase has **well-documented deprecation notices**. All deprecated functions include:

1. Clear deprecation warnings in docstrings
2. Runtime `DeprecationWarning` or `NotImplementedError`
3. Migration guidance to new API

### Deprecated Items

| File | Line | Item | Status |
|------|------|------|--------|
| `nltools/data/brain_data.py` | 2071 | `BrainData.regress` | Deprecated, points to `fit(model='glm')` |
| `nltools/data/brain_data.py` | 4598 | `BrainData.randomise` | DEPRECATED, raises NotImplementedError |
| `nltools/data/brain_data.py` | 4605 | `BrainData.ttest` | DEPRECATED, raises NotImplementedError |
| `nltools/data/adjacency/__init__.py` | 718 | `Adjacency.square_shape` | Deprecated, use `.shape` |
| `nltools/stats.py` | 91 | `pearson` | Deprecated, will be removed |
| `nltools/utils.py` | 268 | `get_anatomical` | Deprecated, use `MNI_Template.plot` |
| `nltools/datasets.py` | 446 | `get_collection_image_metadata` | Deprecated |
| `nltools/datasets.py` | 464 | `download_collection` | Deprecated |

---

## Recommendations for v0.6.0

### Immediate Action Required (Before Release)

1. **Add docstrings to 7 public items** (Priority 1)
   - Focus on `Simulator`, `SimulateGrid` classes
   - Add docstrings to utility functions `attempt_to_import`, `all_same`

2. **Add class-level examples** to:
   - `BrainData` (most important user-facing class)
   - `Adjacency`
   - `Simulator`

### High Priority (Should Address)

3. **Add Returns sections** to property docstrings
   - Many properties like `shape`, `dtype`, `isempty` are missing Returns
   - Quick fix: one-line Returns section

4. **Add Args/Returns to key methods**:
   - `BrainData.regress` - 3 params, no Args
   - `LocalAlignment.fit/transform/fit_transform`
   - Pipeline methods (`normalize`, `reduce`, `pipe`)

### Medium Priority (Nice to Have)

5. **Add examples to key methods**:
   - Model `fit`/`predict` methods
   - `HyperAlignment`/`SRM` transform methods
   - `BrainData` analysis methods

6. **Complete validation function Returns sections**:
   - All 17 validation functions need Returns sections
   - Can be batch-edited since they follow the same pattern

### Lower Priority (Post-Release)

7. **Pipeline module documentation**:
   - Many pipeline methods missing Args/Returns
   - Lower priority as this is newer API

---

## Statistics Summary

| Category | Count |
|----------|-------|
| Files analyzed | 63 |
| Total classes | 56 |
| Total functions | 273 |
| Total methods | 531 |
| Public items with docstrings | 99%+ |
| Public items missing Args section | 77 |
| Public items missing Returns section | 171 |
| Key classes missing examples | 6 |
| Deprecated items properly documented | 8/8 (100%) |

---

*Report generated by docstring audit script*
