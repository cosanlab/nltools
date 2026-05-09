# Changelog

All notable changes to nltools are documented here.

## [0.6.0] - Unreleased

**This is a major release** with breaking API changes, new modules, GPU acceleration, and modernized internals. See the [Migration Guide](migration-guide.md) for upgrade instructions.

### Breaking Changes

**Removed Methods**
- `.regress()` removed → Use `.fit(model='glm', X=design_matrix)` 
- `.ttest()` removed → Use `scipy.stats.ttest_1samp()` directly
- `stats.regress_permutation()` removed → Use `inference.one_sample_permutation_test()`
- `stats.correlation()` removed → Use `inference.correlation_permutation_test()`
- `stats.pearson()` removed → Use `scipy.stats.pearsonr` or the inference module
- Old `Brain_Collection` class removed → Use the new `BrainCollection` class

**Changed Signatures**
- `BrainData.predict(algorithm=..., cv_dict=...)` → `predict(spatial_scale=..., cv=...)` (new canonical kwarg for ROI/searchlight/whole-brain dispatch — `method=` is reserved for algorithm choice across the facade)
- `BrainData.distance(spatial_scale=...)` and `Adjacency.spatial_scale` / `to_brain()` / `similarity(project=True)` — new RSA workflow: per-ROI/searchlight RDMs back-project to voxel-space `BrainData`
- `.shape()` is now a property: `.shape` (on both `BrainData` and `Adjacency`)
- `.isempty()` deprecated → Use `.is_empty` property
- `.smooth()` now returns a copy instead of mutating in-place

**Dependency Requirements**
- Python >= 3.11 (dropped 3.10)
- nilearn >= 0.12 (3D images now transform to 1D arrays; nltools handles this internally)
- polars >= 1.35 (DesignMatrix backend migrated from pandas)
- h5py >= 3.15

### New Modules & Classes

**`BrainCollection`** — Multi-subject data container
- 3-axis indexing: `(n_images, n_observations, n_voxels)`
- Group statistics: `.ttest()`, `.mean()`, `.std()`, filtering
- Group inference: permutation testing, ANOVA
- Encoding models: `.fit_ridge()`, `.fit_glm()`, `.predict()` with CV
- ISC analysis: `.isc()`, `.isc_test()` with permutation testing
- Construction: `from_glob()`, `from_bids()`, `from_stacked()`

**`nltools.pipelines`** — Composable analysis pipelines
- `Pipeline` class for fluent transform chaining with built-in CV
- Steps: `NormalizeStep`, `ReduceStep`, `AlignStep`, `PipeStep`
- Terminals: `PredictTerminal`, `ISCTerminal`, `RSATerminal`
- `CVScheme` / `NestedCVScheme` for flexible cross-validation
- `MultiSubjectPipeline` for coordinated group-level analyses
- Result dataclasses: `CVResult`, `FoldResult`, `ISCResult`, `RSAResult`, `PermutationResult`

**`nltools.algorithms.inference`** — GPU-accelerated permutation testing
- `one_sample_permutation_test()`, `two_sample_permutation_test()`
- `correlation_permutation_test()` (Pearson/Spearman/Kendall)
- `isc_permutation_test()`, `isc_group_permutation_test()`
- `matrix_permutation_test()` (Mantel test)
- `icc_permutation_test()`
- `OnlineBootstrapStats` for memory-efficient resampling
- CPU and PyTorch/CUDA backends with deterministic RNG

**`nltools.algorithms.SRM` / `DetSRM`** — Shared Response Model
- Probabilistic and deterministic variants
- sklearn-compatible `fit()` / `transform()` API
- `.transform_subject()` for aligning new subjects
- Support for unequal sample counts across subjects ([#410](https://github.com/cosanlab/nltools/issues/410))

**`nltools.algorithms.LocalAlignment`** — Searchlight-based local alignment
- Searchlight alignment for fine-grained shared responses
- CPU parallelization and GPU backend support
- Generator-based batching for memory efficiency
- Piecewise scheme support

**`nltools.algorithms.HyperAlignment`** — Refactored hyperalignment
- Extracted to standalone sklearn-compatible class
- Stores transformations (`w_`, `s_`) and template
- `.transform_subject()` for new-subject alignment
- Legacy `align()` function still available

**`nltools.models`** — Model abstractions
- `Glm` — First-level GLM wrapping nilearn
- `Ridge` — Ridge regression with GPU support
- `Fit` dataclass — Immutable results container with serialization

**`nltools.neighborhoods`** — Searchlight caching infrastructure

### New Features (Existing Classes)

**`BrainData`**
- `.fit(model='glm', X=dm)` — Unified GLM fitting via nilearn
- `.fit(model='ridge', X=features, cv='auto')` — Ridge regression with auto alpha selection
- `.fit(inplace=False)` returns `Fit` object without mutating state
- `.predict()` — Unified API for timeseries and MVPA decoding
- `.pipe()` — Method chaining support
- Efficient copying for method chaining (~80% performance improvement)

**`Adjacency`**
- `.shape` now returns `(n_nodes, n_nodes)` for API consistency

**Cross-Validation**
- Built-in `cv=` parameter for all model fitting (k-fold, stratified, nested, LOSO)
- Auto alpha selection: `cv='auto'` with grid search for ridge regression
- Out-of-fold predictions and fold indices tracked in results

**P-value options**
- Explicit `tail=` parameter for one-tailed/two-tailed tests ([#315](https://github.com/cosanlab/nltools/issues/315))

### Bug Fixes

- Atlas/label data now correctly uses nearest-neighbor interpolation ([#446](https://github.com/cosanlab/nltools/issues/446))
- `Adjacency.similarity()` NaN handling with `perm_type='2d'` ([#432](https://github.com/cosanlab/nltools/issues/432))
- `Brain_Data.threshold()` now works when `upper=0` or `lower=0` ([#370](https://github.com/cosanlab/nltools/issues/370))
- Unbiased sigma estimator in regression ([#287](https://github.com/cosanlab/nltools/issues/287))
- `MultiSubjectPipeline.align()` works with LOSO CV
- Unequal sample counts in SRM and LocalAlignment ([#410](https://github.com/cosanlab/nltools/issues/410))

### Documentation

- Comprehensive [Migration Guide](migration-guide.md) with before/after examples
- New tutorials: encoding models, BrainCollection, pipeline workflows
- Standardized all docstrings to Google style
- API reference pages for pipelines, simulator, neighborhoods, and cache modules
- Resolved 236 documentation build warnings

---

## [0.5.1] - 2023-03-28

- Fix deprecated `scipy.stats.binom_test` usage
- CI and conda workflow updates

## [0.5.0] - 2022-12-01

- Replace deepdish with h5py for data serialization
- Support Path objects in `BrainData` and `Adjacency` write methods
- MNI preferences tutorial
- Various bug fixes ([#392](https://github.com/cosanlab/nltools/issues/392), [#409](https://github.com/cosanlab/nltools/issues/409))
- pandas 2.0 compatibility

## [0.4.6] - 2022-08-15

- Maintenance release (see [GitHub releases](https://github.com/cosanlab/nltools/releases) for earlier history)
