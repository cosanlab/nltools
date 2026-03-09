# Changelog

All notable changes to nltools are documented here.

## [Unreleased]

### Bug Fixes

- Atlas/label data now correctly uses nearest-neighbor interpolation (#446)
- Adjacency.similarity() NaN handling with perm_type='2d' (#432)
- Brain_Data.threshold() now works when upper=0 or lower=0 (#370)
- Adjacency.shape returns (n_nodes, n_nodes) for API consistency
- Remove incorrect @pytest.mark.slow markers from fast tests
- MultiSubjectPipeline.align() now works with LOSO CV (nltools-7j3g)
- Skip CI for beads sync commits, skip surface tests when files missing
- CI test failures - FittedBrainCollection, thresholds, tolerance
- More robust CI tests - tolerance, nan handling, constant input checks
- Use unbiased sigma estimator in regression (GH #287)
- Correct type annotations for ty type checker

### Documentation

- Add GitHub issues audit for v0.6.0 planning
- Add update notices to tutorials using deprecated localizer dataset
- Document API issues in migration guide
- Add BrainCollection tutorial
- Update migration guide and API docs for v0.6.0
- Add encoding models tutorial (08_encoding_models.py)
- Consolidate group_analysis + thresholding tutorials
- Heavy prune 01_glm.py tutorial
- Prune tutorials removing pedagogy, keeping practical code
- Add CHANGELOG.md with pipeline infrastructure release notes
- Add Pipeline workflow tutorials
- Document predict algorithms and class_weight='balanced' (GH #182, #177)
- Add comprehensive v0.6.0 codebase audit
- Fix all 236 documentation build warnings

### Features

- Add three user-requested enhancements for v0.6.0
- Add BrainCollection class for multi-subject data
- *(BrainCollection)* Add group inference and transformation methods
- Add searchlight neighborhood caching infrastructure
- *(BrainCollection)* Add ISC computation methods
- *(BrainCollection)* Add isc_test() for permutation testing
- *(collection)* Add GLM/Ridge workflow helper functions
- *(plotting)* Add cortical flatmap visualization
- *(collection)* Add BrainCollection.fit_glm() for group-level GLM
- *(collection)* Add BrainCollection.fit_ridge() for group encoding models
- *(collection)* Add compute_contrasts() and select_feature() methods
- *(collection)* Change fit_ridge() default output to CV scores
- Unified predict() API for timeseries and MVPA decoding
- Add tests for map(axis=1), isc_test, and ISC ROI extraction
- Add unified BrainCollection.fit() API matching BrainData
- Add Phase 2 tests and docs for BrainCollection.fit() API
- Add workflow integration tests (MVPA, Group Inference, ISC)
- Add MNI-aligned VTC masks and update ISC tests
- Add RSA workflow tests
- Add SRM workflow validation tests
- Add cross-subject pooled decoding test for SRM workflow
- Add pipeline infrastructure for fluent CV and Pool API (Phases 1-6)
- Add alignment pipeline step for SRM/HyperAlignment (Phase 7)
- Add Phase 8 terminals and advanced CV (ISC, RSA, Permutation, Nested CV)
- Refactor BrainData.predict() to use Pipeline infrastructure (Phase 9)
- Add LocalAlignment stub for Phase 1 (nltools-oqil)
- Implement LocalAlignment Phase 1 with searchlight alignment (nltools-oqil)
- Add piecewise scheme support to LocalAlignment (nltools-oqil.4)
- Add generator-based batching to LocalAlignment (nltools-pc2i)
- Add CPU parallelization to LocalAlignment (Phase 3)
- Add GPU/Backend integration to LocalAlignment (Phase 4)
- Add BrainCollection.align() for functional alignment (Phase 5)
- V0.6.0 prep - LocalAlignment complete, GH issue reconciliation
- Add explicit tail options for MCP-compatible p-values (GH #315)
- Complete P1 API improvements for v0.6.0
- Complete P2 API improvements for v0.6.0
- Add type annotations and pipe() tests for v0.6.0 P2 items
- Support unequal sample counts in SRM and LocalAlignment (GH #410)

### Miscellaneous

- Fix test warnings and polars migration consistency
- Suppress sklearn false-positive warnings in standardize tests
- Add GitHub Actions workflows and mark slow tests
- Sync beads (close nltools-dkoy)
- Sync beads state
- Migrate project tracking docs to Linear

### Refactoring

- Complete P0 deprecations for v0.6.0 release

### Testing

- Add from_bids tests with minimal BIDS fixture
- Add failing tests for align+LOSO bug (nltools-7j3g)

## [0.6.0-isc-fix] - 2026-01-02

### Bug Fixes

- *(inference)* Achieve perfect backward compatibility via deterministic RNG pattern
- *(inference)* Achieve perfect cross-backend determinism for all permutation tests
- Convert Backend object to parallel string in Ridge CV calls
- Handle NaN values in Adjacency.similarity() (#432)
- Resolve API mismatches in BrainData and test expectations
- Prevent fitted model state from propagating to copies
- Resolve pandas deprecation and logic bug in Adjacency
- Replace legacy tier1/tier2 markers with slow/gpu
- Adjacency.regress() now correctly sets is_single_matrix for single-regressor DesignMatrix case
- ISC calculation in align() now correctly handles all axis/data_type combos

### Documentation

- Complete documentation update for GPU-accelerated inference module

### Features

- Gpu acceleration one-sample test
- GPU-accelerated inference module with clean architecture
- *(inference)* Add GPU-accelerated correlation permutation test module
- *(inference)* Add Spearman and Kendall correlation metrics to correlation module
- *(inference)* Add matrix permutation test (Mantel test) module
- *(inference)* Add GPU-accelerated Intersubject Correlation (ISC) module

### Miscellaneous

- Remove 4 dead placeholder tests for moved functions

### Refactoring

- *(polars)* Complete Polars optimization with native resampling
- Move Adjacency to subdirectory structure

### Styling

- Format prefs.py (ruff)

### Testing

- Add missing Adjacency tests for median, generate_permutations, stats_label_distance, and distance_to_similarity euclidean

## [0.6.0-polars-complete] - 2025-10-30

### Documentation

- Refactor documentation into focused, purpose-built files
- Suppress sphinx build warnings with exclude patterns
- Eliminate all Sphinx build warnings (45→0)
- *(tests)* Document parallel testing safety with pytest-xdist
- Convert all docstrings from NumPy to Google style
- *(testing)* Enforce parallel-first and permission-gated tier2 testing
- *(polars)* Update refactor docs and archive completed research

### Features

- Polars migration TDD scaffolding for Design_Matrix
- *(polars)* Implement DesignMatrix Phase 1 - Construction and basic operations
- *(polars)* Implement DesignMatrix Phase 2 - Statistical operations
- *(polars)* Implement DesignMatrix Phase 3 - HRF convolution
- *(polars)* Implement DesignMatrix Phase 4 & Phase 5a/5b - Polynomials and basic append
- *(polars)* Implement DesignMatrix Phase 5c - Polynomial separation (multi-run support)
- *(polars)* Implement DesignMatrix Phase 6 - Diagnostics (VIF and clean)
- *(polars)* Complete DesignMatrix Polars migration - Phase 7 (Utilities)
- *(tests)* Implement 2-tier testing strategy with 16× speedup
- *(polars)* Complete file_reader integration with DesignMatrix methods
- *(polars)* Complete Polars DesignMatrix integration - fix Adjacency.regress()

### Miscellaneous

- *(deps)* Update dependencies and fix nilearn 0.12 compatibility
- Remove unused docstring conversion script

### Refactoring

- *(polars)* Optimize DesignMatrix with idiomatic Polars patterns
- *(polars)* Optimize DesignMatrix with selectors and enhanced errors
- *(polars)* Consolidate design_matrix files and remove old implementation
- *(polars)* Remove dead Design_Matrix_Series code
- *(polars)* Standardize on DesignMatrix naming throughout codebase

### Checkpoint

- Before polars migration finalization

## [docs-update-v1] - 2025-10-30

### Bug Fixes

- Complete Round 1 audit fixes - 4 critical bugs resolved

### Documentation

- Add remaining v0.6.0 tasks to refactoring plan
- Improve API documentation infrastructure and organization

### Features

- Add cross-validation support to Brain_Data.fit()

### Testing

- Add filter() and compute_contrasts() tests + minimal_brain_data fixture
- Add comprehensive SRM/DetSRM tests (34 tests)

## [0.6.0-test-align-split] - 2025-10-29

### Features

- Add sklearn-style fit/predict API and deprecate regress()

### Testing

- Split test_align to document working alignment functionality

## [0.6.0-brain-data-copy-fix] - 2025-10-29

### Bug Fixes

- Handle model attributes in Brain_Data.copy() to prevent pickle errors

### Documentation

- Update documentation to reflect completed test suite refactoring
- Update documentation for R², effect variance, and filter method
- Update nilearn-log.md with Phase 1 & 2 completion status
- Update nilearn-log.md with Phase 3 completion status
- Update REFACTORING_PLAN.md with Priority 2.5 completion
- Add systematic benchmarking framework and update project specs

### Features

- Add backend abstraction for CPU/GPU operations
- Add ridge regression algorithms with SVD decomposition
- Add cluster thresholding to Brain_Data.threshold() method
- Complete ridge regression test suite (Cycles 2.2 & 2.3)
- Migrate apply_mask to nilearn for better performance
- Implement BaseModel and Ridge model classes with GPU support
- Extract HyperAlignment class from align() function
- Add Glm model class wrapping nilearn FirstLevelModel
- Add CLI-based benchmarking with dry-run and progress tracking

### Refactoring

- *(tests)* Reorganize Brain_Data tests into class-based structure
- *(tests)* Reorganize Adjacency tests into class-based structure
- *(tests)* Reorganize Design_Matrix tests into class-based structure
- *(tests)* Organize test_stats.py with section headers and docstrings
- *(tests)* Organize test suite into subdirectories following architectural patterns
- Integrate Glm model into Brain_Data.regress()

## [0.6.0-align-deferred] - 2025-10-29

### Testing

- Skip test_align and defer ISC fix to future release

## [0.6.0-adjacency-properties] - 2025-10-29

### Refactoring

- *(adjacency)* Convert shape() and isempty() to properties

## [0.6.0-plotting-removal] - 2025-10-29

### Documentation

- Streamline CLAUDE.md and add token-efficient pytest guidance

### Features

- *(brain_data)* Implement efficient copying for method chaining (~80% performance improvement)

### Refactoring

- *(api)* Convert .shape(), .isempty(), .dtype() to properties + quick fixes

### Testing

- Remove deprecated plotting tests

## [0.4.6] - 2022-08-15


