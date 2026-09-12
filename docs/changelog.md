# Changelog

All notable changes to nltools are documented here.



## Unreleased

### Features
- <span class="badge badge-feature">Feature</span> Implement efficient copying for method chaining (~80% performance improvement)
- <span class="badge badge-feature">Feature</span> Add backend abstraction for CPU/GPU operations
- <span class="badge badge-feature">Feature</span> Add ridge regression algorithms with SVD decomposition
- <span class="badge badge-feature">Feature</span> Add cluster thresholding to Brain_Data.threshold() method
- <span class="badge badge-feature">Feature</span> Complete ridge regression test suite (Cycles 2.2 & 2.3)
- <span class="badge badge-feature">Feature</span> Migrate apply_mask to nilearn for better performance
- <span class="badge badge-feature">Feature</span> Implement BaseModel and Ridge model classes with GPU support
- <span class="badge badge-feature">Feature</span> Extract HyperAlignment class from align() function
- <span class="badge badge-feature">Feature</span> Add Glm model class wrapping nilearn FirstLevelModel
- <span class="badge badge-feature">Feature</span> Add CLI-based benchmarking with dry-run and progress tracking
- <span class="badge badge-feature">Feature</span> Add sklearn-style fit/predict API and deprecate regress()
- <span class="badge badge-feature">Feature</span> Add cross-validation support to Brain_Data.fit()
- <span class="badge badge-feature">Feature</span> Polars migration TDD scaffolding for Design_Matrix
- <span class="badge badge-feature">Feature</span> Implement DesignMatrix Phase 1 - Construction and basic operations
- <span class="badge badge-feature">Feature</span> Implement DesignMatrix Phase 2 - Statistical operations
- <span class="badge badge-feature">Feature</span> Implement DesignMatrix Phase 3 - HRF convolution
- <span class="badge badge-feature">Feature</span> Implement DesignMatrix Phase 4 & Phase 5a/5b - Polynomials and basic append
- <span class="badge badge-feature">Feature</span> Implement DesignMatrix Phase 5c - Polynomial separation (multi-run support)
- <span class="badge badge-feature">Feature</span> Implement DesignMatrix Phase 6 - Diagnostics (VIF and clean)
- <span class="badge badge-feature">Feature</span> Complete DesignMatrix Polars migration - Phase 7 (Utilities)
- <span class="badge badge-feature">Feature</span> feat(polars): Complete DesignMatrix Polars migration with GLM integration
- <span class="badge badge-feature">Feature</span> Implement 2-tier testing strategy with 16× speedup
- <span class="badge badge-feature">Feature</span> Complete file_reader integration with DesignMatrix methods
- <span class="badge badge-feature">Feature</span> Complete Polars DesignMatrix integration - fix Adjacency.regress()
- <span class="badge badge-feature">Feature</span> gpu acceleration one-sample test
- <span class="badge badge-feature">Feature</span> GPU-accelerated inference module with clean architecture
- <span class="badge badge-feature">Feature</span> Add GPU-accelerated correlation permutation test module
- <span class="badge badge-feature">Feature</span> Add Spearman and Kendall correlation metrics to correlation module
- <span class="badge badge-feature">Feature</span> Add matrix permutation test (Mantel test) module
- <span class="badge badge-feature">Feature</span> Add GPU-accelerated Intersubject Correlation (ISC) module
- <span class="badge badge-feature">Feature</span> add three user-requested enhancements for v0.6.0
- <span class="badge badge-feature">Feature</span> Add BrainCollection class for multi-subject data
- <span class="badge badge-feature">Feature</span> Add group inference and transformation methods
- <span class="badge badge-feature">Feature</span> Add searchlight neighborhood caching infrastructure
- <span class="badge badge-feature">Feature</span> Add ISC computation methods
- <span class="badge badge-feature">Feature</span> Add isc_test() for permutation testing
- <span class="badge badge-feature">Feature</span> Add GLM/Ridge workflow helper functions
- <span class="badge badge-feature">Feature</span> Add cortical flatmap visualization
- <span class="badge badge-feature">Feature</span> Add BrainCollection.fit_glm() for group-level GLM
- <span class="badge badge-feature">Feature</span> Add BrainCollection.fit_ridge() for group encoding models
- <span class="badge badge-feature">Feature</span> Add compute_contrasts() and select_feature() methods
- <span class="badge badge-feature">Feature</span> Change fit_ridge() default output to CV scores
- <span class="badge badge-feature">Feature</span> Unified predict() API for timeseries and MVPA decoding
- <span class="badge badge-feature">Feature</span> Add tests for map(axis=1), isc_test, and ISC ROI extraction
- <span class="badge badge-feature">Feature</span> Add unified BrainCollection.fit() API matching BrainData
- <span class="badge badge-feature">Feature</span> Add Phase 2 tests and docs for BrainCollection.fit() API
- <span class="badge badge-feature">Feature</span> Add workflow integration tests (MVPA, Group Inference, ISC)
- <span class="badge badge-feature">Feature</span> Add MNI-aligned VTC masks and update ISC tests
- <span class="badge badge-feature">Feature</span> Add RSA workflow tests
- <span class="badge badge-feature">Feature</span> Add SRM workflow validation tests
- <span class="badge badge-feature">Feature</span> Add cross-subject pooled decoding test for SRM workflow
- <span class="badge badge-feature">Feature</span> Add pipeline infrastructure for fluent CV and Pool API (Phases 1-6)
- <span class="badge badge-feature">Feature</span> Add alignment pipeline step for SRM/HyperAlignment (Phase 7)
- <span class="badge badge-feature">Feature</span> Add Phase 8 terminals and advanced CV (ISC, RSA, Permutation, Nested CV)
- <span class="badge badge-feature">Feature</span> Refactor BrainData.predict() to use Pipeline infrastructure (Phase 9)
- <span class="badge badge-feature">Feature</span> Add LocalAlignment stub for Phase 1 (nltools-oqil)
- <span class="badge badge-feature">Feature</span> Implement LocalAlignment Phase 1 with searchlight alignment (nltools-oqil)
- <span class="badge badge-feature">Feature</span> Add piecewise scheme support to LocalAlignment (nltools-oqil.4)
- <span class="badge badge-feature">Feature</span> Add generator-based batching to LocalAlignment (nltools-pc2i)
- <span class="badge badge-feature">Feature</span> Add CPU parallelization to LocalAlignment (Phase 3)
- <span class="badge badge-feature">Feature</span> Add GPU/Backend integration to LocalAlignment (Phase 4)
- <span class="badge badge-feature">Feature</span> Add BrainCollection.align() for functional alignment (Phase 5)
- <span class="badge badge-feature">Feature</span> v0.6.0 prep - LocalAlignment complete, GH issue reconciliation
- <span class="badge badge-feature">Feature</span> Add explicit tail options for MCP-compatible p-values (GH #315)
- <span class="badge badge-feature">Feature</span> Complete P1 API improvements for v0.6.0
- <span class="badge badge-feature">Feature</span> Complete P2 API improvements for v0.6.0
- <span class="badge badge-feature">Feature</span> Add type annotations and pipe() tests for v0.6.0 P2 items
- <span class="badge badge-feature">Feature</span> Support unequal sample counts in SRM and LocalAlignment (GH #410)
- <span class="badge badge-feature">Feature</span> add verbose parameter to standardize() to suppress sklearn warnings
- <span class="badge badge-feature">Feature</span> rewrite GLM tutorial and auto-detect duplicate intercept in add_poly
- <span class="badge badge-feature">Feature</span> restore BrainData.ttest / add ttest2
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> replace parallel kwarg with device on user-facing methods

    - **Breaking:** BrainCollection.{permutation_test,permutation_test2,isc,isc_test,align} now take device='cpu'|'gpu' and n_jobs, instead of the old parallel='cpu'|'gpu'|None. Use n_jobs=1 for single-threaded execution. Returned dict key 'parallel' renamed to 'device'.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> standardize progress flag to progress_bar; default False

    - **Breaking:** All show_progress kwargs across BrainData, BrainCollection, and their submodules renamed to progress_bar. Defaults flipped from True to False to match scikit-learn convention. verbose retained where it controls log-level output (sklearn warning suppression in standardize, info prints in DesignMatrix.clean/append).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> standardize algorithm-choice kwarg to method

    - **Breaking:** rename algorithm/icc_type/extract_type/scheme/kind/noise_model/mode to `method` on user-facing BrainData, Adjacency, BrainCollection methods. Adjacency.similarity perm_type renamed to permutation_method to avoid collision with metric. Algorithm-layer APIs (CVScheme.scheme, Glm.noise_model, compute_icc_voxelwise.icc_type, LocalAlignment.scheme) retain their names; the class facades translate at the boundary. predict.estimator unchanged (would collide with existing method= on that signature).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> rename plot thr_upper/thr_lower to upper/lower

    - **Breaking:** BrainData.plot now accepts upper/lower (matching the canonical threshold method signature used across BrainData/Adjacency/BrainCollection) instead of thr_upper/thr_lower. The convenience scalar `threshold` kwarg is unchanged.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> standardize permutation count kwarg to n_permute

    - **Breaking:** Adjacency.generate_permutations now takes `n_permute` instead of `n_perm`, matching the convention already used in Adjacency.similarity, plot_silhouette, stats_label_distance, and BrainCollection.isc_test/permutation_test[2]. BrainCollection.align's `n_iter` stays (optimizer iterations, not permutations).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> flip similarity diagonal flag to include_diag

    - **Breaking:** Adjacency.similarity now takes `include_diag=False` instead of `ignore_diagonal=False`. The polarity is flipped to match Adjacency.distance's kwarg, AND the default is changed: self-similarity on the diagonal is trivially 1.0 and uninformative, so directed matrices now exclude it by default. Symmetric matrices never store the diagonal, so this is a no-op for them.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> rename user-facing radius kwargs to radius_mm

    - **Breaking:** All user-facing `radius` kwargs taking a sphere/searchlight radius in millimeters are renamed to `radius_mm` for unit clarity.
- <span class="badge badge-feature">Feature</span> restore legacy h5 read support for nltools <= 0.5.1 files
- <span class="badge badge-feature">Feature</span> add load_haxby_example offline demo dataset
- <span class="badge badge-feature">Feature</span> append(axis=1) accepts pandas/polars DataFrames
- <span class="badge badge-feature">Feature</span> construct from a numpy array + explicit mask
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> ttest returns {mean, t, z, p} uniformly

    - **Breaking:** the returned dict keys have changed. Parametric callers that read result["t"]/result["p"] continue to work unchanged. Permutation callers previously got {"mean", "p"} — "p" still works, and "mean" still works; no keys were removed.
- <span class="badge badge-feature">Feature</span> compute_contrasts(contrast_type="all") + glm_ docs
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> load from BIDS file paths + rename polys→confounds

    - **Breaking:** - `nltools.io.onsets_to_dm` and `nltools.io.file_reader` are removed.   Use `DesignMatrix(path, run_length=N, TR=t)` (or `events_to_dm`   for in-memory frames) + explicit `.convolve()`. - `DesignMatrix.polys` → `DesignMatrix.confounds` (attribute + init   kwarg). `DesignMatrix.vif(exclude_polys=)` → `exclude_confounds=`.   `DesignMatrix.clean(exclude_polys=)` → `exclude_confounds=`. - `DesignMatrix.write()` to HDF5 stores `confounds` metadata attr   (was `polys`). No DM HDF5 reader exists, so no legacy files to   migrate.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> lazy-fetch niftis from HF dataset

    - **Breaking:** huggingface-hub is a new runtime dep. First template access requires network. Wheel no longer ships niftis. Missing-file exception changed from FileNotFoundError to huggingface_hub.errors.EntryNotFoundError (LocalEntryNotFoundError when offline-and-uncached). Pyodide async path is a NotImplementedError stub for now.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> default model='glm' in fit(); tighten validation

    - **Breaking:** Error type for unknown model changed from ValueError to TypeError. Calls passing model=None previously raised TypeError("model must be provided"); now they succeed using glm.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> add design_clean kwargs to fit() for GLM validation

    - **Breaking:** GLM designs containing regressors with abs(r) >= 0.95 are now auto-cleaned by default. Pass design_clean=False to preserve the previous behavior of keeping all regressors regardless of collinearity.
- <span class="badge badge-feature">Feature</span> pyodide async seed + sync fetch
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> add `limit` kwarg + change slices default view to "z"

    - **Breaking:** `BrainData.plot()` returns `list[Figure]` for multi-image glass/slices plots instead of a single Figure. The default slices `view` changed from `"xyz"` to `"z"`; pass `view="xyz"` to restore the prior three-figure layout.
- <span class="badge badge-feature">Feature</span> IDBFS-backed persistent cache for pyodide
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> gate MNI-only plot paths on standard-space data

    - **Breaking:** bd.plot(method='glass') and bd.plot_flatmap()/plot_surf() now raise ValueError on non-standard-space data instead of silently rendering against MNI scaffolding. bd.plot(method='slices') without bg_img also raises (previously warned + fell back to the MNI 2mm template). Pass bg_img=<your subject anatomical> for native-space data, or call bd.resample() to bring data into standard space first. The one
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> per-voxel α + held-out predictions through BrainData CV path

    - **Breaking:** - cv_results_["best_alpha"] is now a (n_voxels,) array (was scalar)   when local_alpha=True. Pass local_alpha=False for the legacy   scalar-α-shared-across-voxels behavior. - BrainData.fit's signature is now keyword-only after `model` (the *   marker after the primary positional arg).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> copy-constructor + read-only convolved/confounds

    - **Breaking:** ``dm.convolved = […]`` and ``dm.confounds = […]`` now raise ``AttributeError``. Pass via the ``convolved=`` / ``confounds=`` constructor kwargs, or use ``.append(other, axis=1)`` (raw DataFrames are auto-marked as confounds; ``as_confounds=True`` promotes a DesignMatrix's columns).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> HRF-convolve events files by default

    - **Breaking:** DesignMatrix(events_path, run_length=N, TR=t) previously returned boxcar regressors; now returns HRF-convolved by default. Callers that relied on the boxcar output (e.g., to build interaction terms before convolution) need to add hrf_model=None.
- <span class="badge badge-feature">Feature</span> with_columns + pl.Expr in __setitem__
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> find_spikes returns DesignMatrix

    - **Breaking:** Code that did `find_spikes(data).iloc[:, 1:]` or `spikes.drop('TR', axis=1)` will break — drop the index manipulation and use the returned DesignMatrix directly via .append(spikes, axis=1).
- <span class="badge badge-feature">Feature</span> list_resources() for HF dataset discoverability
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> rebuild iplot() as anywidget viewer with 4D step-through

    - **Breaking:** `iplot(surface=True, anatomical=...)` → `iplot(view='surface', bg_img=...)`. The `view=` kwarg is canonical with `view='ortho'` (default). `bg_img=` matches nilearn naming. The `nltools[interactive_plots]` extra is gone — `anywidget` is now a hard dep.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> iplot threshold panel + drag-end render fix

    - **Breaking:** BrainViewerWidget.threshold / threshold_min / threshold_max / threshold_step traitlets are gone — replaced by lower / upper / vmax_abs / data_min / data_max / pct_table_*. Direct widget users need to update; iplot() callers passing threshold= are unaffected.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> rewrite BrainData.predict() with kwargs API + Predict dataclass

    - **Breaking:** - BrainData.predict(y=) returns Predict dataclass instead of BrainData;   result['weight_map'] → result.weight_map. - estimator= renamed to model= (mirrors bd.fit(model=)). - 'ridge' shortcut now means Ridge regressor; for RidgeClassifier use   'ridge_classifier'. - scoring='auto' default replaces hardcoded 'accuracy'. - BrainData.cv() and the .cv().normalize().reduce().pipe().predict()   fluent chain removed; pass cv=, standardize=, reduce='pca', and/or   model=Pipeline(...) on bd.predict() instead. - n_jobs default 1 (was -1) for searchlight/roi.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> brain-space Predict fields + ROI per-fold scores + decoding tutorial

    - **Breaking:** - Predict.weight_map / fold_weight_maps / final_weight_map / accuracy_map   are BrainData, not ndarray. Numpy via .data. - bd.predict(..., inplace=True) attaches with predict_ prefix   (bd.predict_weight_map, etc.), not flat names. - ROI dispatch repurposes scores/mean_score/std_score with array shapes   (n_folds,n_rois)/(n_rois,) instead of leaving them None. - model=Pipeline(...) auto-flips standardize to False (with warning);   pass standardize=True explicitly to keep the old wrapping behavior.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> ROI dispatch produces voxel-space weight maps
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> spatial_scale axis for ROI/searchlight RSA + predict rename

    - **Breaking:** predict() no longer accepts method= for spatial scope; use spatial_scale=. method= is reserved for algorithm choice.
- <span class="badge badge-feature">Feature</span> implement distance(searchlight) and align(roi)
- <span class="badge badge-feature">Feature</span> cluster reports with anatomical labeling
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> rebuild BrainData.iplot() on ipyniivue (niivue)

    - **Breaking:** BrainData.iplot() signature changed. Removed: mode=, units=, cut_coords=, symmetric_cmap=, and view="surface" (raises, pointing to view="render" / plot_flatmap / plot_surf). cmap default RdBu_r -> warm (matplotlib names auto-mapped with a warning). lower/upper reinterpreted as window endpoints. New: atlas=, opacity=, outline=. **kwargs now forward to ipyniivue.NiiVue(). Returns ipyniivue.NiiVue, not BrainViewerWidget.
- <span class="badge badge-feature">Feature</span> add marimo->ipynb converter for JupyterLite tutorials
- <span class="badge badge-feature">Feature</span> add poe docs-jupyterlite task to build the tutorial JupyterLite bundle
- <span class="badge badge-feature">Feature</span> make GLM + MVPA tutorials run in JupyterLite on trimmed HF data
- <span class="badge badge-feature">Feature</span> add "Try it live" JupyterLite nav link + deploy-ready build
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> iplot() colorbar + interactive threshold slider

    - **Breaking:** iplot() now returns an ipywidgets.VBox by default (with .viewer and .threshold_slider) instead of an ipyniivue.NiiVue. Access the widget via .viewer, or pass controls=False for the bare NiiVue.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> host pain dataset on HF so fetch_pain() works in Pyodide

    - **Breaking:** fetch_pain() signature changed from (data_dir=None, verbose=1) to (verbose=0). The data_dir arg is removed — cache location is now governed by huggingface_hub (HF_HOME). fetch_pain().X columns changed from the raw Neurovault metadata to the curated schema above.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> host emotion-rating dataset on HF for fetch_emotion_ratings()

    - **Breaking:** fetch_emotion_ratings() signature changed from (data_dir=None, verbose=1) to (verbose=0); data_dir removed (HF cache governs location). X columns are the curated Neurovault table keyed by a new `filename` column.
- <span class="badge badge-feature">Feature</span> serve tutorials as interactive marimo/WASM notebooks
- <span class="badge badge-feature">Feature</span> land core helpers, __init__, indexing, parallel _apply
- <span class="badge badge-feature">Feature</span> implement load/unload/write/read/cleanup + memory_estimate
- <span class="badge badge-feature">Feature</span> GLM fit + HDF5 bundles + compute_contrasts
- <span class="badge badge-feature">Feature</span> reductions, perm tests, ISC, align, from_bids/from_glob, predict dispatch
- <span class="badge badge-feature">Feature</span> ridge fit bundles + bc.predict(X_new=) per-subject path
- <span class="badge badge-feature">Feature</span> stage encoding + ISC datasets for nltools/niftis
- <span class="badge badge-feature">Feature</span> wire max_gpu_memory_gb into ridge_svd GPU batching (F022)
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> user-controlled scale/standardize preprocessing

    - **Breaking:** BrainData.fit / BrainCollection.fit drop `scale_value`; `scale` default changes from True (grand-mean) to 'auto' (off). GLM no longer mean-scales by default (betas now in native units unless scale=True; t/z/p unaffected). Ridge now z-scores its targets by default. Bundle schema 1->2 (old fit caches invalid).
- <span class="badge badge-feature">Feature</span> GLM predict(X)/coef_ parity + nilearn report (F182)
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> append(axis=1) refuses bitwise-duplicate columns

    - **Breaking:** DesignMatrix.append(axis=1) raises ValueError when an appended column's values are bitwise identical to another column's, where it previously appended silently. Drop or modify one of the columns first.
- <span class="badge badge-feature">Feature</span> make the rank-deficiency warning diagnostic and actionable
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> machine-enforce the canonical kwarg vocabulary from api-vocabulary.yml

    - **Breaking:** BrainData.fit defaults progress_bar=False and no longer inherits bd.verbose when unset (verbose is reserved for log-level only). SphereNeighborhoods.iter_neighborhoods takes progress_bar keyword-only. Deliberate deviations are now recorded as manifest exemptions with reasons (BrainData.predict n_jobs=1 memory guard; align n_iter solver iterations).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> one canonical tail= vocabulary across every p-value (#474)

    - **Breaking:** the v0.5 -1/'upper'/'lower' public forms now raise ValueError (negate the data / swap groups / flip the contrast for the negative direction; _compute_pvalue keeps them internally for forced-tail sites). BrainData.ttest and Adjacency.ttest previously ignored tail= on the parametric path (always two-sided); tail now maps onto scipy's alternative=, and the z map matches the requested tail.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> reserve the .nl_ namespace for generated columns

    - **Breaking:** all generated DesignMatrix / find_spikes column names gained the `.nl_` prefix, and run-separated columns changed shape from `{run}_{col}` to `.nl_r{run}_{col}`. Code selecting generated columns by name must be updated; see the (reserved-column-prefix) section of the migration guide.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> one core GPU execution layer — measured budgets, OOM recovery, run-or-raise

    - **Breaking:** SRM/DetSRM.fit() no longer accepts max_gpu_memory_gb and parallel='gpu' raises; LocalAlignment rejects unknown parallel= values and gpu-without-torch; max_gpu_memory_gb defaults changed from 4.0 to None (measured) across inference/ridge/braindata entry points; seeded null distributions from BrainCollection.permutation_test/ permutation_test2 change (engine RNG replaces the hand-rolled loop). Migration guide: (gpu-execution-layer).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> carve out predict_group(); remove the legacy cv() pipeline

    - **Breaking:** BrainCollection.predict(y=...) raises (use predict_group); BrainCollection.cv() and BrainCollectionPipeline are removed. Migration guide: (predict-group).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> iplot robust autoscaling + shared zero-aware percentile thresholds

    - **Breaking:** threshold(upper="98%") results change on any map containing zeros (percentile now over nonzero voxels); iplot's default window is autoscaled rather than min/max. Migration guide: (iplot-autoscale).
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> sklearn-style cv names — 'logo'/'loo' replace 'loso'/'loro'

    - **Breaking:** cv='loso'/'loro' removed; use cv='logo' with groups=. predict_group(cv=<int>) with a classifier now stratifies folds.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> predict() decodes against the stored .Y slot — labels travel with the data

    - **Breaking:** predict(cv=<int>, groups=...) now group-aware; previously groups was ignored for int cv specs.
- <span class="badge badge-feature">Feature</span> PredictCollection — per-subject decoding results container
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> predict(y=) maps per-subject decoding — closes the #478 map-reduce gap

    - **Breaking:** predict(y=...) previously raised (and before that, aggregated across subjects — that operation is predict_group since the phase-1 carve-out). predict() with no arguments now decodes stored .Y instead of raising for a missing X_new.
- ⚠ **Breaking** <span class="badge badge-feature">Feature</span> predict_group permutation null for roi and searchlight

    - **Breaking:** Predict.permutation_pvalue is no longer always float | None — a per-ROI ndarray for spatial_scale='roi' and a BrainData map for 'searchlight' (both previously crashed, so no working code changes meaning). Whole-brain results are numerically unchanged for a given random_state.
- <span class="badge badge-feature">Feature</span> warn on near-collinear full-rank designs

### Improvements
- <span class="badge badge-improvement">Improvement</span> refactor and improve brain data dunder math. improve first tutorial
- <span class="badge badge-improvement">Improvement</span> refactor onsets_to_dm to wrap new nilearn functionality instead
- <span class="badge badge-improvement">Improvement</span> Convert .shape(), .isempty(), .dtype() to properties + quick fixes
- <span class="badge badge-improvement">Improvement</span> Convert shape() and isempty() to properties
- <span class="badge badge-improvement">Improvement</span> Reorganize Brain_Data tests into class-based structure
- <span class="badge badge-improvement">Improvement</span> Reorganize Adjacency tests into class-based structure
- <span class="badge badge-improvement">Improvement</span> Reorganize Design_Matrix tests into class-based structure
- <span class="badge badge-improvement">Improvement</span> Organize test_stats.py with section headers and docstrings
- <span class="badge badge-improvement">Improvement</span> Organize test suite into subdirectories following architectural patterns
- <span class="badge badge-improvement">Improvement</span> Integrate Glm model into Brain_Data.regress()
- <span class="badge badge-improvement">Improvement</span> Optimize DesignMatrix with idiomatic Polars patterns
- <span class="badge badge-improvement">Improvement</span> Optimize DesignMatrix with selectors and enhanced errors
- <span class="badge badge-improvement">Improvement</span> Consolidate design_matrix files and remove old implementation
- <span class="badge badge-improvement">Improvement</span> Remove dead Design_Matrix_Series code
- <span class="badge badge-improvement">Improvement</span> Standardize on DesignMatrix naming throughout codebase
- <span class="badge badge-improvement">Improvement</span> Complete Polars optimization with native resampling
- <span class="badge badge-improvement">Improvement</span> refactor inference tests and add two sample statistica correcteness tests
- <span class="badge badge-improvement">Improvement</span> improve isc group
- <span class="badge badge-improvement">Improvement</span> refactor tests
- <span class="badge badge-improvement">Improvement</span> refactor tests
- <span class="badge badge-improvement">Improvement</span> refactoring tests
- <span class="badge badge-improvement">Improvement</span> improve resampling
- <span class="badge badge-improvement">Improvement</span> move Adjacency to subdirectory structure
- <span class="badge badge-improvement">Improvement</span> Complete P0 deprecations for v0.6.0 release
- <span class="badge badge-improvement">Improvement</span> split brain_data.py into braindata/ subpackage with facade pattern
- <span class="badge badge-improvement">Improvement</span> split design_matrix.py into design_matrix/ subpackage
- <span class="badge badge-improvement">Improvement</span> rename design_matrix/ to designmatrix/
- <span class="badge badge-improvement">Improvement</span> split adjacency/__init__.py into submodules
- <span class="badge badge-improvement">Improvement</span> split collection.py into collection/ subpackage
- <span class="badge badge-improvement">Improvement</span> remove leading underscores from module filenames
- <span class="badge badge-improvement">Improvement</span> restructure data/ to use only subpackage dirs
- <span class="badge badge-improvement">Improvement</span> clean BrainData facade — no private methods, alphabetical ordering
- <span class="badge badge-improvement">Improvement</span> clean Adjacency facade — no private methods, alphabetical ordering
- <span class="badge badge-improvement">Improvement</span> restructure BrainData tests into braindata/ subdir
- <span class="badge badge-improvement">Improvement</span> clean DesignMatrix facade — no private methods, alphabetical ordering
- <span class="badge badge-improvement">Improvement</span> restructure Adjacency tests into adjacency/ subdir
- <span class="badge badge-improvement">Improvement</span> restructure DesignMatrix tests into designmatrix/ subdir
- <span class="badge badge-improvement">Improvement</span> rename tests/shell → tests/data, tests/data → tests/fixtures
- <span class="badge badge-improvement">Improvement</span> restructure model tests into models/ subdir
- <span class="badge badge-improvement">Improvement</span> move models tests from tests/core/ to tests/models/
- <span class="badge badge-improvement">Improvement</span> move Simulator and Roc into nltools/data/ packages
- <span class="badge badge-improvement">Improvement</span> move neighborhoods and cache into data/braindata/
- <span class="badge badge-improvement">Improvement</span> move check_brain_data helpers into braindata/utils.py
- <span class="badge badge-improvement">Improvement</span> consolidate template resolution into MNI_Template_Factory
- <span class="badge badge-improvement">Improvement</span> create nltools/io/ package, move HDF5 and file_reader into it
- <span class="badge badge-improvement">Improvement</span> add tests/io_tests/ with h5 and file_reader test classes
- <span class="badge badge-improvement">Improvement</span> move BrainData Args to class docstring, drop params-move logic
- <span class="badge badge-improvement">Improvement</span> reorganize nltools/stats.py into focused subpackage
- <span class="badge badge-improvement">Improvement</span> move stats tests to tests/stats/ for flatter hierarchy
- <span class="badge badge-improvement">Improvement</span> add stats/permutation.py facade and update imports
- <span class="badge badge-improvement">Improvement</span> reorganize nltools/plotting.py into focused subpackage
- <span class="badge badge-improvement">Improvement</span> consolidate alignment algorithms and move validation into inference
- <span class="badge badge-improvement">Improvement</span> consolidate ridge/backends into top-level Backend class
- <span class="badge badge-improvement">Improvement</span> remove deprecated neurovault download shims
- <span class="badge badge-improvement">Improvement</span> replace prefs.MNI_Template with templates/ module
- <span class="badge badge-improvement">Improvement</span> lazy-import pandas across nltools modules
- <span class="badge badge-improvement">Improvement</span> rewrite plotting/adjacency label-distance and silhouette as polars-native
- <span class="badge badge-improvement">Improvement</span> polars-first stats.outliers.zscore and stats.intersubject input boundaries
- <span class="badge badge-improvement">Improvement</span> sweep pandas out of scattered small modules
- <span class="badge badge-improvement">Improvement</span> relocate backends to algorithms/ and thread resolve_backend
- <span class="badge badge-improvement">Improvement</span> validate_frame returns polars, accepts pandas/numpy/csv
- <span class="badge badge-improvement">Improvement</span> BrainData.X/.Y are polars DataFrames, never None
- <span class="badge badge-improvement">Improvement</span> drop pandas from braindata/analysis.py
- <span class="badge badge-improvement">Improvement</span> drop vestigial pandas branch in parse_contrast_string
- <span class="badge badge-improvement">Improvement</span> rewrite BrainData h5 persistence on h5py + polars
- <span class="badge badge-improvement">Improvement</span> Simulator.create_data builds polars frames and writes via pl
- <span class="badge badge-improvement">Improvement</span> Adjacency.Y invariant is polars via property setter
- <span class="badge badge-improvement">Improvement</span> drop pandas from Adjacency CSV read/write
- <span class="badge badge-improvement">Improvement</span> rewrite Adjacency h5 persistence on h5py + polars
- <span class="badge badge-improvement">Improvement</span> numpy-native Adjacency label-distance and cluster helpers
- <span class="badge badge-improvement">Improvement</span> BrainCollection.metadata invariant is polars
- <span class="badge badge-improvement">Improvement</span> build BrainCollection result metadata as polars
- <span class="badge badge-improvement">Improvement</span> rename plotting functions to plot_* convention
- <span class="badge badge-improvement">Improvement</span> lean on nilearn for HRFs and resample plumbing
- <span class="badge badge-improvement">Improvement</span> drop BrainData.nifti_masker, use functional apply_mask/unmask
- <span class="badge badge-improvement">Improvement</span> drop Simulator.nifti_masker, use functional apply_mask/unmask
- <span class="badge badge-improvement">Improvement</span> drop pandas-compat shims from stats and DesignMatrix
- <span class="badge badge-improvement">Improvement</span> drop legacy PyTables h5 support and plotting underscore re-exports
- <span class="badge badge-improvement">Improvement</span> drop glover_hrf lambda wrapper in onsets_to_dm
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> canonical trailing-kwarg order across facades
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> replace **kwargs passthroughs with explicit signatures (1/2)

    - **Breaking:** Remove **kwargs catch-all from user-facing methods that delegate internally; expose previously hidden kwargs explicitly.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> replace **kwargs passthroughs with explicit signatures (2/2)

    - **Breaking:** Expose previously hidden permutation and bootstrap kwargs explicitly instead of forwarding via **kwargs.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> keyword-only marker for __init__ and complex methods

    - **Breaking:** Enforce keyword-only args after the primary data arg for all four data-class __init__ methods, plus selected public methods with many optional kwargs. Callers passing these positionally must update.
- <span class="badge badge-improvement">Improvement</span> route progress_bar to nilearn verbose
- <span class="badge badge-improvement">Improvement</span> simplify extract_roi PCA, stop mutating caller headers
- <span class="badge badge-improvement">Improvement</span> broaden check_brain_data to accept Niimg-like inputs
- <span class="badge badge-improvement">Improvement</span> use np.trunc instead of deprecated np.fix
- <span class="badge badge-improvement">Improvement</span> expand ruff with UP/C4/PIE/RUF022/RET/SIM families
- <span class="badge badge-improvement">Improvement</span> improve first tutorial and make plotting fixes
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> polars passthrough, unify data attr, drop details()

    - **Breaking:** DesignMatrix._df is now .data. DesignMatrix.details() is removed — use print(dm) or repr(dm) instead.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> drop fetch_haxby; migrate GLM tutorial to localizer

    - **Breaking:** nltools.datasets.fetch_haxby removed. Use nilearn directly (fetch_localizer_first_level, fetch_spm_auditory, etc.) and convert events.tsv to a DesignMatrix with nltools.io.onsets_to_dm.
- <span class="badge badge-improvement">Improvement</span> rename fetch_nifti → fetch_resource
- <span class="badge badge-improvement">Improvement</span> extract indexing helpers to indexing.py
- <span class="badge badge-improvement">Improvement</span> extract aggregation helpers to aggregation.py
- <span class="badge badge-improvement">Improvement</span> extract conversions to conversions.py
- <span class="badge badge-improvement">Improvement</span> move memory_estimate/load/unload to io.py
- <span class="badge badge-improvement">Improvement</span> consolidate stateless helpers in core.py
- <span class="badge badge-improvement">Improvement</span> drop unused TypeVar import
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> scaffold v0.6.0 BrainCollection redesign

    - **Breaking:** BrainCollection is being rewritten from scratch. Removes FittedBrainCollection, BrainCollectionCVResult (CV results are now plain BrainData), fit_glm/fit_from_events/fit_ridge (one .fit() instead), to_stacked/from_stacked/to_list/to_tensor, axis= on reductions, output=/save= on fit, checkpoint(), and the per-axis _map_axis0/1/2 / _aggregate_axis0/1/2 handlers. Shape semantics drop the 3D framing; mean()/std()/etc. collapse subjects only. Old test files for the removed APIs are deleted.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> tighten decompose signature

    - **Breaking:** BrainData.decompose and analysis.decompose are now keyword-only after the data arg. Positional calls beyond `self`/`bd` (which were silently swallowed by the old `*args`) now raise TypeError.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> drop final_weight_map, weight_map = all-data refit

    - **Breaking:** - Predict.final_weight_map field removed (folded into weight_map). - Predict.final_estimator renamed to Predict.estimator. - bd.predict(refit=...) kwarg removed; the all-data fit is always-on. - bd.predict(inplace=True) now attaches bd.predict_estimator (was   bd.predict_final_estimator) and no longer attaches predict_final_*.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> generalize .plot() into method= dispatcher; add .corr()

    - **Breaking:** DesignMatrix.plot()'s first positional arg is now `method` (was `figsize`). Pass figsize=... as a keyword. dm.plot() with no args is unchanged.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> rename BrainCollectionPipeline.normalize → standardize

    - **Breaking:** ``BrainCollection.cv(...).normalize(...)`` is now ``standardize(...)``.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> BrainCollectionPipeline.predict returns BrainData

    - **Breaking:** ``BrainCollectionPipeline.predict`` no longer returns ``BrainCollectionCVResult``; that class is removed. Downstream code reading ``.mean_score`` / ``.scores`` continues to work since the attributes live on the BrainData now (note: ``.scores`` is renamed to ``.cv_scores`` to mirror the BrainData CV path).
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> align bc.predict with new BD.predict + GLM tutorial

    - **Breaking:** BrainCollection.predict kwargs estimator=/return_weights= removed; use model= to match BrainData.predict.
- <span class="badge badge-improvement">Improvement</span> memoize fetch_resource, skip network revalidation
- <span class="badge badge-improvement">Improvement</span> add coalesced_gc() and wrap masking-heavy operations
- <span class="badge badge-improvement">Improvement</span> validate GLM-fit result mask once per map-list
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> remove superseded standalone Pipeline/MultiSubject orchestration

    - **Breaking:** nltools.pipelines.Pipeline, MultiSubjectPipeline, NestedCVScheme, the *Terminal and *Result classes, and PooledData.cv() are removed. Multi-subject CV now lives on BrainCollection (bc.cv().standardize().reduce().predict()); custom single-dataset preprocessing uses model=make_pipeline(...) on BrainData.predict().
- <span class="badge badge-improvement">Improvement</span> extract API-doc postprocess into its own module
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> remove dead standalone pipeline surface

    - **Breaking:** nltools.pipelines no longer exports AlignStep, FittedAlign, or Terminal.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> strip ICC functionality entirely (F012/F048/F140/F177/F194)

    - **Breaking:** BrainData.icc(), nltools.stats.compute_icc, and nltools.algorithms.inference.compute_icc_voxelwise are removed. Compute ICC externally (e.g. pingouin.intraclass_corr) on extracted values. See the migration guide.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> canonical-kwarg sweep — banned-kwarg renames + **kwargs hygiene

    - **Breaking:** renamed public kwargs — permutation tests parallel=->device=, BrainCollection.align scheme=->spatial_scale=, regress mode=->method=, BrainCollectionPipeline.predict algorithm=->method=, compute_contrasts contrast_type=->method=. DesignMatrix.up/downsample and Roc.plot no longer accept **kwargs.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> add keyword-only `*` marker to 61 public functions (F-kwonly)

    - **Breaking:** kwargs on the following are now keyword-only (positional calls TypeError). Callers overwhelmingly already used keywords; positional callers and facade→backing delegations were converted.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> rename compute_contrasts statistic selector method→statistic

    - **Breaking:** compute_contrasts(..., method='t'|'beta'|'all') → statistic=.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> canonicalize remaining kwarg names across facades

    - **Breaking:** many public kwargs renamed to the v0.6.0 canonical vocabulary (see list above). Notably: DesignMatrix.clean now defaults to silent (progress_bar=False, was verbose=True); BrainData.standardize uses suppress_warnings=False (was verbose=True); BrainCollection.isc_test uses n_samples=; BrainCollection.align uses roi_mask=; Roc uses method=; Adjacency.cluster_summary/similarity/plot_mds and stats.isc/compute_similarity kwargs renamed.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> close AUDIT-0.6.0 dead-code bucket (23 findings)

    - **Breaking:** BrainData.regress(), BrainData.predict_multi(), StatResult.to_nifti(), and FittedBrainCollection are removed. Use fit(model='glm', X=...) for regression and predict(y=...) for whole-brain MVPA (see migration guide).
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> remove orphaned pool.py two-stage aggregation (F118/F111)

    - **Breaking:** nltools.pipelines no longer exports PooledData, StatResult, or ResultDict; the two-stage bc.fit(...).pool() GLM aggregation workflow is removed.
- <span class="badge badge-improvement">Improvement</span> functional GLM map extraction, no per-contrast round-trips
- <span class="badge badge-improvement">Improvement</span> functional compute_contrasts, no Nifti round-trip
- <span class="badge badge-improvement">Improvement</span> annotate models layer + fix Any/hint gaps (F107/F009/F017/F046/F085)
- <span class="badge badge-improvement">Improvement</span> single-source helpers + frozen dataclass cleanups (F006/F061/F154/F183/F192)
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> co-locate pipeline primitives under data/collection/pipesteps

    - **Breaking:** nltools.pipelines is removed; import the primitives from nltools.data.collection.pipesteps (or its .base/.cv/.steps submodules) instead.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> canonicalize spatial_scale vocab in LocalAlignment

    - **Breaking:** LocalAlignment now takes spatial_scale=('searchlight'|'roi') and roi_mask= instead of scheme=('searchlight'|'piecewise') and parcellation=. No compat aliases. BrainCollection.align(spatial_scale='whole_brain') now raises NotImplementedError instead of leaking a LocalAlignment ValueError.
- <span class="badge badge-improvement">Improvement</span> remove 7 dead validation/utility functions
- <span class="badge badge-improvement">Improvement</span> rewrite as a speed+memory harness over the current API
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> canonicalize device selection to device= across facades

    - **Breaking:** Ridge(backend=) -> Ridge(device=); BrainData.fit(model='ridge') backend=/parallel= in **kwargs now raise TypeError (use device=); BrainData.bootstrap(backend=) -> device=; nltools.stats.phase_randomize(backend=) -> device=.
- <span class="badge badge-improvement">Improvement</span> run pairwise ISC bootstrap on the GPU
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> rebuild BrainData.iplot() on a self-owned niivue anywidget

    - **Breaking:** BrainData.iplot() now returns a NiivueViewer (anywidget.AnyWidget) instead of an ipyniivue.NiiVue / ipywidgets.VBox. The `.viewer` and `.threshold_slider` attributes are gone; the threshold window is reactive via the `cal_min`/`cal_max` traits. `ipyniivue` is no longer a dependency.
- <span class="badge badge-improvement">Improvement</span> make find_spikes deduplication unconditional; drop clean=
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> stop cleaning the design in fit(); warn on rank deficiency

    - **Breaking:** `BrainData.fit()` no longer accepts `design_clean`, `design_clean_thresh`, `design_clean_exclude_confounds`, or `design_clean_fill_na`, and no longer runs `DesignMatrix.clean()` implicitly. It estimates exactly the design it is given. Callers who want columns dropped call `DesignMatrix.clean()` explicitly. `fit()` now emits a UserWarning when the design matrix is rank deficient.
- <span class="badge badge-improvement">Improvement</span> make maybe_tqdm/make_progress_bar the single library-wide progress mechanism
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> enforce the keyword-only marker convention uniformly across the package

    - **Breaking:** options on the newly-marked public functions must now be passed by keyword (e.g. KFoldStratified(5, True) -> KFoldStratified(5, shuffle=True)).
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> remove nltools.stats; consolidate the functional core into nltools.algorithms

    - **Breaking:** nltools.stats is removed. Implementations moved: corrections.py, outliers.py, regression.py as-is; timeseries.py -> algorithms/signal.py and correlation.py -> algorithms/similarity.py (avoids name echoes with the inference engine's modules); intersubject.py -> algorithms/inference/; alignment.py -> algorithms/alignment/procrustes.py. stats/permutation.py is deleted outright — the nltools.algorithms permutation exports ARE the algorithms.inference engine functions (identity, no wrapper layer), so facade/engine drift is structurally impossible.
    - **Breaking:** the inference engine speaks the canonical device= vocabulary directly. parallel= -> device= on every algorithms.inference entry point (values unchanged: 'cpu' | 'gpu' | None), validate_parallel_parameter -> validate_device_parameter, and result dicts report a 'device' key instead of 'parallel'. phase_randomize(backend='numpy'|'torch') is now keyword-only device='cpu'|'gpu'|'auto' with warn-and-fallback on unknown values. The ridge and alignment layers keep parallel= internally (documented Backend abstraction); facades still translate at those boundaries.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> canonicalize the ISC vocabulary (summary=/metric=/null_dist)

    - **Breaking:** isc_permutation_test / isc_group_permutation_test rename metric= (the 'median'|'mean' central-tendency choice) to summary=, and sim_metric= (the similarity metric) to metric= — `metric` now means the same thing here as everywhere else in the API. isc_group() and BrainCollection.isc/.isc_test likewise take summary= instead of metric= (isc() already used summary=).
    - **Breaking:** every ISC result — engines, the isc/isc_group wrappers, and BrainCollection — exposes the null under the engine-standard 'null_dist' key; the legacy 'null_distribution' key is removed.
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> canonicalize cluster_summary and extract_roi kwargs

    - **Breaking:** Adjacency.cluster_summary renames method= -> summary= ('mean'|'median'|None central tendency) and its old summary= (within/between scope) -> scope=. BrainData.extract_roi renames metric= -> method= ('mean'|'median'|'pca' selects an extraction variant; metric stays reserved for similarity metrics). Closes the last two mean/median-vocabulary violations flagged in the #474 consolidation follow-up.
- <span class="badge badge-improvement">Improvement</span> isc_test p-values via the shared _compute_pvalue helper
- <span class="badge badge-improvement">Improvement</span> thread tail through the bootstrap engines; drop the facade closures
- <span class="badge badge-improvement">Improvement</span> one GPU bootstrap driver, two thin wrappers
- <span class="badge badge-improvement">Improvement</span> stop materializing np.abs(arr) three times per iplot window
- <span class="badge badge-improvement">Improvement</span> one shared manifest module for the lint-api trio
- ⚠ **Breaking** <span class="badge badge-improvement">Improvement</span> pare the alignment package back to the v0.5.1 surface

    - Removes `HyperAlignment`, `LocalAlignment` and `RoiNeighborhoods`, the `parallel=`/`n_jobs=`/`pad_samples=` knobs on `SRM`/`DetSRM`, `BrainData.align(spatial_scale='searchlight', radius=)`, and `backends.auto_n_jobs_for_arrays` — all 0.6.0-dev-only names no v0.5.1 user can see. The Procrustes template loop moves into `nltools.algorithms.alignment.procrustes` as the internal `_hyperalign`, which `align(method='procrustes')` calls; every value it returns is numerically unchanged.
    - Retains the F001 fix: `_hyperalign` zero-pads the feature axis up to the largest subject instead of truncating to the smallest, so no subject's features are silently dropped. Subjects with unequal sample counts now raise, as they did in v0.5.1.
    - **Breaking:** `BrainData.align(target, method='procrustes')` stores `transformation_matrix` transposed, so back-projection is `transformed @ T.T` — the rule its v0.5.1 docstring already documented and its code did not honor, and the rule `nltools.algorithms.align` uses. `nltools.algorithms.align(..., method='procrustes')` on numpy input is transposed to match. `transformed`, `common_model`, `disparity` and `scale` are byte-identical on every path.
    - **Breaking:** `nltools.algorithms.align(..., method='procrustes', axis=1)` on `BrainData` input now raises `ValueError`. The axis=1 transform spans images on both axes, so it has no voxel axis to be returned on; the call previously produced a `BrainData` whose matrix width did not match its own mask. numpy input at `axis=1` is unaffected.

### Bug Fixes
- <span class="badge badge-fix">Bug Fix</span> fix formatting
- <span class="badge badge-fix">Bug Fix</span> fix warnings 1
- <span class="badge badge-fix">Bug Fix</span> fix warnings 2
- <span class="badge badge-fix">Bug Fix</span> fix us few more things
- <span class="badge badge-fix">Bug Fix</span> fix neurovault downloaders and add tests
- <span class="badge badge-fix">Bug Fix</span> fix tutorial and log
- <span class="badge badge-fix">Bug Fix</span> fix nilearn resampling messages
- <span class="badge badge-fix">Bug Fix</span> fix Brain_Data eq. Update tests
- <span class="badge badge-fix">Bug Fix</span> fixup collection and tutorial v1
- <span class="badge badge-fix">Bug Fix</span> fix up collection issue
- <span class="badge badge-fix">Bug Fix</span> Handle model attributes in Brain_Data.copy() to prevent pickle errors
- <span class="badge badge-fix">Bug Fix</span> Complete Round 1 audit fixes - 4 critical bugs resolved
- <span class="badge badge-fix">Bug Fix</span> Achieve perfect backward compatibility via deterministic RNG pattern
- <span class="badge badge-fix">Bug Fix</span> Achieve perfect cross-backend determinism for all permutation tests
- <span class="badge badge-fix">Bug Fix</span> fix statistical correctness of inference module
- <span class="badge badge-fix">Bug Fix</span> fix docs
- <span class="badge badge-fix">Bug Fix</span> fix ruff complaints
- <span class="badge badge-fix">Bug Fix</span> fix adjacenecy bootstrap. Add gpu optimized correlation and timeseries functions
- <span class="badge badge-fix">Bug Fix</span> fix/add gpu optimizations
- <span class="badge badge-fix">Bug Fix</span> fix up bootstrapping with ridge
- <span class="badge badge-fix">Bug Fix</span> fix up docs for algorithms
- <span class="badge badge-fix">Bug Fix</span> fix cross-validation
- <span class="badge badge-fix">Bug Fix</span> fix brain-data fit progress bar
- <span class="badge badge-fix">Bug Fix</span> fix resampling
- <span class="badge badge-fix">Bug Fix</span> fix beads git tracking and add GitHub issues
- <span class="badge badge-fix">Bug Fix</span> fix Brain_Data squeeze flattening single-item lists (#449)
- <span class="badge badge-fix">Bug Fix</span> fix nilearn darkness parameter deprecation warning
- <span class="badge badge-fix">Bug Fix</span> fix nilearn nearest interpolation deprecation warning
- <span class="badge badge-fix">Bug Fix</span> fix NiftiMasker mask warning in simulator
- <span class="badge badge-fix">Bug Fix</span> fix memory resampling warnings in tests
- <span class="badge badge-fix">Bug Fix</span> convert Backend object to parallel string in Ridge CV calls
- <span class="badge badge-fix">Bug Fix</span> handle NaN values in Adjacency.similarity() (#432)
- <span class="badge badge-fix">Bug Fix</span> resolve API mismatches in BrainData and test expectations
- <span class="badge badge-fix">Bug Fix</span> prevent fitted model state from propagating to copies
- <span class="badge badge-fix">Bug Fix</span> resolve pandas deprecation and logic bug in Adjacency
- <span class="badge badge-fix">Bug Fix</span> replace legacy tier1/tier2 markers with slow/gpu
- <span class="badge badge-fix">Bug Fix</span> Adjacency.regress() now correctly sets is_single_matrix for single-regressor DesignMatrix case
- <span class="badge badge-fix">Bug Fix</span> ISC calculation in align() now correctly handles all axis/data_type combos
- <span class="badge badge-fix">Bug Fix</span> atlas/label data now correctly uses nearest-neighbor interpolation (#446)
- <span class="badge badge-fix">Bug Fix</span> Adjacency.similarity() NaN handling with perm_type='2d' (#432)
- <span class="badge badge-fix">Bug Fix</span> Brain_Data.threshold() now works when upper=0 or lower=0 (#370)
- <span class="badge badge-fix">Bug Fix</span> fix docs build: add jupytext config for .py tutorials
- <span class="badge badge-fix">Bug Fix</span> Adjacency.shape returns (n_nodes, n_nodes) for API consistency
- <span class="badge badge-fix">Bug Fix</span> Remove incorrect \@pytest.mark.slow markers from fast tests
- <span class="badge badge-fix">Bug Fix</span> MultiSubjectPipeline.align() now works with LOSO CV (nltools-7j3g)
- <span class="badge badge-fix">Bug Fix</span> Skip CI for beads sync commits, skip surface tests when files missing
- <span class="badge badge-fix">Bug Fix</span> CI test failures - FittedBrainCollection, thresholds, tolerance
- <span class="badge badge-fix">Bug Fix</span> More robust CI tests - tolerance, nan handling, constant input checks
- <span class="badge badge-fix">Bug Fix</span> Use unbiased sigma estimator in regression (GH #287)
- <span class="badge badge-fix">Bug Fix</span> Correct type annotations for ty type checker
- <span class="badge badge-fix">Bug Fix</span> relax SVD reconstruction tolerance for float32 precision
- <span class="badge badge-fix">Bug Fix</span> seed RNG in SVD test to eliminate flaky precision failures
- <span class="badge badge-fix">Bug Fix</span> configure ty type checker and resolve 18 type errors
- <span class="badge badge-fix">Bug Fix</span> fixup tutorials
- <span class="badge badge-fix">Bug Fix</span> resolve ty type-checking errors
- <span class="badge badge-fix">Bug Fix</span> guard scale_data grand-mean branch against divide-by-zero
- <span class="badge badge-fix">Bug Fix</span> fix up docs and poe commands
- <span class="badge badge-fix">Bug Fix</span> fix toc
- <span class="badge badge-fix">Bug Fix</span> update ridge benchmark to parallel API
- <span class="badge badge-fix">Bug Fix</span> rewrite mvpa_roi for correct per-ROI MVPA decoding
- <span class="badge badge-fix">Bug Fix</span> serialize polars frames to h5 via Arrow IPC
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> make out-of-mask voxels transparent in BrainData.plot

    - **Breaking:** `plot(threshold=t)` is now nilearn's absolute-value transparency cutoff (hide voxels with |value| < t) instead of a signed remap to upper/lower. Negative thresholds raise. Use `upper=` / `lower=` for one-sided data thresholding.
- <span class="badge badge-fix">Bug Fix</span> make slices + flatmap plotting usable again
- <span class="badge badge-fix">Bug Fix</span> fixup plotting and add plot_surf()
- <span class="badge badge-fix">Bug Fix</span> fix up design matrix, autoscale for plotting, include constant for .add_dct_basis by default to match .add_poly
- <span class="badge badge-fix">Bug Fix</span> fix up design mat
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> compute_contrasts returns real t-stats, not beta sums

    - **Breaking:** compute_contrasts default behavior now returns t-statistics. Code that treated the old return as effect sizes must pass contrast_type="beta" explicitly.
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> return matplotlib Figure from BrainData/DesignMatrix .plot()

    - **Breaking:** BrainData.plot() returns matplotlib.figure.Figure, not a nilearn Display. DesignMatrix.plot() returns matplotlib.figure.Figure, not Axes.
- <span class="badge badge-fix">Bug Fix</span> honor fit_intercept and CV splitters; add BrainData.size
- <span class="badge badge-fix">Bug Fix</span> re-stub predict_multi; bounds-trim default slice cut_coords
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> always suffix convolved columns with _c{i}

    - **Breaking:** column lookups by trial-type name after a `.convolve()` chain need `_c0` appended (notably `compute_contrasts("A - B")` becomes `compute_contrasts("A_c0 - B_c0")`).
- <span class="badge badge-fix">Bug Fix</span> make .convolve() idempotent over already-convolved columns
- <span class="badge badge-fix">Bug Fix</span> relax accidental numpy floor; pin nilearn in pyodide smoke test
- <span class="badge badge-fix">Bug Fix</span> restore the interactive_plots optional extra (ipywidgets)
- <span class="badge badge-fix">Bug Fix</span> deterministic cell ids for generated tutorial notebooks
- <span class="badge badge-fix">Bug Fix</span> deterministic tiebreak for same-second step subdirs
- <span class="badge badge-fix">Bug Fix</span> strip leaked RST directives from re-exported docstrings
- <span class="badge badge-fix">Bug Fix</span> correct Attributes-section removal over-match in API generation
- <span class="badge badge-fix">Bug Fix</span> correct invalid p-values/CIs and permutation nulls (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> crashes/silent-empty on realistic inputs (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> silent metadata loss and threshold/stack bugs (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> seaborn 0.13.2 crashes, dropped returns, triangle swap (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> always-raising ctor, copy-paste + wrong SE bugs (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> schema check + non-integer downsample ratios (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> __all__ attribute errors and mutable default (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> filter crash, resource leaks, CV-index + MVPA seed bugs (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> NaN-poisoning, silent-wrong + crash bugs (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> n_subjects referenced nonexistent BrainCollection.n_images (0.6.0 audit)
- <span class="badge badge-fix">Bug Fix</span> restore ISC bootstrap null centering dropped in refactor (F066)
- <span class="badge badge-fix">Bug Fix</span> HyperAlignment auto_pad zero-pads instead of truncating (F001)
- <span class="badge badge-fix">Bug Fix</span> implement PooledData.repool for real fitted_state shapes (F111)
- <span class="badge badge-fix">Bug Fix</span> share one integer cluster_id space between peaks and clusters (F043)
- <span class="badge badge-fix">Bug Fix</span> complete parallel→device + contrast_type→method renames missed in 9b1b0eb4
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> drop invalid permutation CVScheme, add predict(n_permute=) null (F112)

    - **Breaking:** CVScheme no longer accepts scheme='permutation' / cv(method='permutation'). Use BrainCollectionPipeline.predict(n_permute=N) for the permutation-accuracy null.
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> stop advertising kwargs that do nothing (F068/F021/F182)

    - **Breaking:** BrainCollection.isc/isc_test no longer accept radius_mm, device, n_jobs or progress_bar (they were silently ignored); roi_mask now actually scopes the computation and the returned maps carry the ROI mask, so results from code that passed roi_mask will CHANGE (previously whole-brain). The ridge solvers no longer accept n_jobs.
- <span class="badge badge-fix">Bug Fix</span> apply_mask inherits target space for raw Niimg masks
- <span class="badge badge-fix">Bug Fix</span> clearer errors + robust input handling (F098/F031/F159/F157)
- <span class="badge badge-fix">Bug Fix</span> to_nifti preserves data precision instead of quantizing to mask dtype
- <span class="badge badge-fix">Bug Fix</span> align feeds LocalAlignment correct orientation + wire cache= (F073)
- <span class="badge badge-fix">Bug Fix</span> repair docstring rendering bugs in API reference
- <span class="badge badge-fix">Bug Fix</span> make marimo-WASM tutorials boot in Pyodide
- <span class="badge badge-fix">Bug Fix</span> run GPU legs on CUDA hosts, not just MPS
- <span class="badge badge-fix">Bug Fix</span> stream leave-one-out ISC instead of materializing all subjects
- <span class="badge badge-fix">Bug Fix</span> correct regress standard errors, all_same, copy() docs
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> wire GPU into pairwise ISC; fail fast on unsupported metric

    - **Breaking:** isc_permutation_test(parallel='gpu', summary_statistic='pairwise', sim_metric != 'correlation') now raises ValueError instead of silently running on CPU.
- <span class="badge badge-fix">Bug Fix</span> sequential micropip install for marimo-WASM; adopt niivue viewer
- <span class="badge badge-fix">Bug Fix</span> use Python # comments in Pyodide micropip.install string
- <span class="badge badge-fix">Bug Fix</span> make h5py a core dependency; BrainCollection.fit() requires it
- <span class="badge badge-fix">Bug Fix</span> stop find_spikes() emitting duplicate spike regressors
- <span class="badge badge-fix">Bug Fix</span> a design matrix with no regressors keeps its row count
- <span class="badge badge-fix">Bug Fix</span> make n_rows survive copies and reject conflicting values
- <span class="badge badge-fix">Bug Fix</span> point the rank-deficiency warning at regularization, not deletion
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> add progress_bar to the permutation and bootstrap family

    - **Breaking:** progress bars are now off by default. isc_permutation_test and isc_group_permutation_test previously defaulted to progress_bar=True and now default to False; every other function in the family previously had no way to disable its bar. Pass progress_bar=True to restore the old output.
- <span class="badge badge-fix">Bug Fix</span> thread progress_bar through the bootstrap and Adjacency stat facades
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> make options keyword-only across the inference layer

    - **Breaking:** options must now be passed by keyword. `one_sample_permutation_test(data, 5000)` becomes `one_sample_permutation_test(data, n_permute=5000)`. The leading data arguments remain positional.
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> make write() and the file constructor round-trip

    - **Breaking:** `write()` to a `.csv` now emits comma-separated data (it emitted tabs before, which its own reader could not parse). Code parsing nltools-written `.csv` files with an explicit tab delimiter must switch to comma or pass `sep="\t"` to `write()`.
- <span class="badge badge-fix">Bug Fix</span> compare OOM-recovered results at float32-ulp tolerance, not bitwise
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> timeseries GPU draws match CPU exactly; conjugate pairing fixed in batched phase randomization

    - **Breaking:** GPU null distributions for timeseries_correlation_permutation_ test change (they now equal the CPU nulls for a given seed).
- <span class="badge badge-fix">Bug Fix</span> relay worker warnings to the parent — deduplicated, categories preserved
- <span class="badge badge-fix">Bug Fix</span> adjacency plots rendered twice
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> GPU Spearman ranks ties correctly; device validation is run-or-raise

    - **Breaking:** GPU-computed Spearman correlations and null distributions over tied data (integer ratings, discrete scores) change — they were wrong and now match the CPU/scipy path; continuous untied data is unaffected. phase_randomize with an invalid device= now raises ValueError instead of warning and falling back to CPU.
- <span class="badge badge-fix">Bug Fix</span> one-tailed z maps stay finite; single shared z-from-p helper
- ⚠ **Breaking** <span class="badge badge-fix">Bug Fix</span> ttest(popmean=X, permutation=True) tests mean != popmean

    - **Breaking:** BrainData.ttest(popmean=X, permutation=True) with X != 0 returns different (now correct) p-values and a popmean-subtracted 'mean' map. popmean=0 calls are numerically unchanged.
- <span class="badge badge-fix">Bug Fix</span> string class labels decode and persist end to end
- <span class="badge badge-fix">Bug Fix</span> predict worker closures no longer capture the collection
- <span class="badge badge-fix">Bug Fix</span> classify .h5 items by bundle_kind, not bare suffix
- <span class="badge badge-fix">Bug Fix</span> predict-bundle model_spec is a real refit spec, not a repr
- <span class="badge badge-fix">Bug Fix</span> plot_between_label_distance crashed on its default permutation path
- <span class="badge badge-fix">Bug Fix</span> validate the separator-recovery re-parse instead of trusting the header hint
- <span class="badge badge-fix">Bug Fix</span> translate pre-.nl_ generated names when loading a legacy h5
- <span class="badge badge-fix">Bug Fix</span> make the GPU bootstrap per-sample hooks private
- <span class="badge badge-fix">Bug Fix</span> qualify vocabulary suppressions by module path
- <span class="badge badge-fix">Bug Fix</span> move check_kwonly's inline EXEMPT dict into the vocabulary manifest
- <span class="badge badge-fix">Bug Fix</span> move the _NullProgressBar nosemgrep suppressions into the semgrep config
- <span class="badge badge-fix">Bug Fix</span> serialize in-memory masks in BrainData h5 files
- <span class="badge badge-fix">Bug Fix</span> GPU bootstrap guard accepts torch-cuda
- <span class="badge badge-fix">Bug Fix</span> keep refit alpha indices on the compute device
- <span class="badge badge-fix">Bug Fix</span> cap measured-budget batch sizing at a saturation ceiling
- <span class="badge badge-fix">Bug Fix</span> bench_inference GPU leg passed the harness probe string as device=
- <span class="badge badge-fix">Bug Fix</span> provenance label auto-detects the GPU; record the benchmarked commit

### Documentation
- <span class="badge badge-docs">Docs</span> Streamline CLAUDE.md and add token-efficient pytest guidance
- <span class="badge badge-docs">Docs</span> Update documentation to reflect completed test suite refactoring
- <span class="badge badge-docs">Docs</span> Update documentation for R², effect variance, and filter method
- <span class="badge badge-docs">Docs</span> Update nilearn-log.md with Phase 1 & 2 completion status
- <span class="badge badge-docs">Docs</span> Update nilearn-log.md with Phase 3 completion status
- <span class="badge badge-docs">Docs</span> Update REFACTORING_PLAN.md with Priority 2.5 completion
- <span class="badge badge-docs">Docs</span> Add systematic benchmarking framework and update project specs
- <span class="badge badge-docs">Docs</span> Add remaining v0.6.0 tasks to refactoring plan
- <span class="badge badge-docs">Docs</span> Improve API documentation infrastructure and organization
- <span class="badge badge-docs">Docs</span> Refactor documentation into focused, purpose-built files
- <span class="badge badge-docs">Docs</span> suppress sphinx build warnings with exclude patterns
- <span class="badge badge-docs">Docs</span> eliminate all Sphinx build warnings (45→0)
- <span class="badge badge-docs">Docs</span> docs scaffold
- <span class="badge badge-docs">Docs</span> Document parallel testing safety with pytest-xdist
- <span class="badge badge-docs">Docs</span> Convert all docstrings from NumPy to Google style
- <span class="badge badge-docs">Docs</span> Enforce parallel-first and permission-gated tier2 testing
- <span class="badge badge-docs">Docs</span> Update refactor docs and archive completed research
- <span class="badge badge-docs">Docs</span> Complete documentation update for GPU-accelerated inference module
- <span class="badge badge-docs">Docs</span> add GitHub issues audit for v0.6.0 planning
- <span class="badge badge-docs">Docs</span> add update notices to tutorials using deprecated localizer dataset
- <span class="badge badge-docs">Docs</span> document API issues in migration guide
- <span class="badge badge-docs">Docs</span> Add BrainCollection tutorial
- <span class="badge badge-docs">Docs</span> Update migration guide and API docs for v0.6.0
- <span class="badge badge-docs">Docs</span> Add encoding models tutorial (08_encoding_models.py)
- <span class="badge badge-docs">Docs</span> Consolidate group_analysis + thresholding tutorials
- <span class="badge badge-docs">Docs</span> Heavy prune 01_glm.py tutorial
- <span class="badge badge-docs">Docs</span> Prune tutorials removing pedagogy, keeping practical code
- <span class="badge badge-docs">Docs</span> Add CHANGELOG.md with pipeline infrastructure release notes
- <span class="badge badge-docs">Docs</span> Add Pipeline workflow tutorials
- <span class="badge badge-docs">Docs</span> Document predict algorithms and class_weight='balanced' (GH #182, #177)
- <span class="badge badge-docs">Docs</span> Add comprehensive v0.6.0 codebase audit
- <span class="badge badge-docs">Docs</span> Fix all 236 documentation build warnings
- <span class="badge badge-docs">Docs</span> standardize docstrings to Google-style and fill gaps
- <span class="badge badge-docs">Docs</span> add API doc pages for pipelines, simulator, neighborhoods, and cache
- <span class="badge badge-docs">Docs</span> fix migration guide inaccuracies and add missing SRM docs
- <span class="badge badge-docs">Docs</span> clarify mask handling behavior in migration guide
- <span class="badge badge-docs">Docs</span> organize BrainData API page into navigable grouped sections
- <span class="badge badge-docs">Docs</span> fix docstrings across braindata subpackage
- <span class="badge badge-docs">Docs</span> fix docstrings and add API pages for design_matrix subpackage
- <span class="badge badge-docs">Docs</span> add API pages for collection subpackage, clean up braindata imports
- <span class="badge badge-docs">Docs</span> migrate from Jupyter Book v1 to v2 (mystmd)
- <span class="badge badge-docs">Docs</span> rewrite BrainData tutorial with execution support
- <span class="badge badge-docs">Docs</span> add class renames, import paths, and ttest removal to migration guide
- <span class="badge badge-docs">Docs</span> rewrite DesignMatrix tutorial with execution support
- <span class="badge badge-docs">Docs</span> rewrite Adjacency tutorial with execution support
- <span class="badge badge-docs">Docs</span> improve BrainData API page layout and griffe2md postprocessing
- <span class="badge badge-docs">Docs</span> apply same API page improvements to DesignMatrix class
- <span class="badge badge-docs">Docs</span> apply same API page improvements to Adjacency class
- <span class="badge badge-docs">Docs</span> remove unused paired .py files for rewritten tutorials
- <span class="badge badge-docs">Docs</span> apply same API page improvements to BrainCollection class
- <span class="badge badge-docs">Docs</span> rewrite BrainCollection tutorial with execution support
- <span class="badge badge-docs">Docs</span> suppress progress bars in BrainCollection tutorial
- <span class="badge badge-docs">Docs</span> docs updates
- <span class="badge badge-docs">Docs</span> add gallery index page for API Classes sidebar section
- <span class="badge badge-docs">Docs</span> include tutorial pages as hidden TOC entries
- <span class="badge badge-docs">Docs</span> docs updates
- <span class="badge badge-docs">Docs</span> capture v0.6.0 API conventions and breaking-commit format
- <span class="badge badge-docs">Docs</span> sync migration guide with last month of breaking commits
- <span class="badge badge-docs">Docs</span> fix tutorial kwarg names and ridge-regression xref
- <span class="badge badge-docs">Docs</span> regenerate from current source
- <span class="badge badge-docs">Docs</span> use load_haxby_example in first-level GLM
- <span class="badge badge-docs">Docs</span> merge group analysis into the GLM tutorial
- <span class="badge badge-docs">Docs</span> add explicit decompose mentions
- <span class="badge badge-docs">Docs</span> PPI worked example using v0.6.0 idioms
- <span class="badge badge-docs">Docs</span> loading canonical brain images section
- <span class="badge badge-docs">Docs</span> regenerate api docs to pick up docstring drift
- <span class="badge badge-docs">Docs</span> switch tutorial atlas from k200 to k50 for faster builds
- <span class="badge badge-docs">Docs</span> refresh Multivariate_Prediction migration guidance
- <span class="badge badge-docs">Docs</span> mark Multivariate_Prediction migration done
- <span class="badge badge-docs">Docs</span> enable RSA tutorial and rewrite for current API
- <span class="badge badge-docs">Docs</span> rewrite RSA tutorial for trial-level RSA in MNI space
- <span class="badge badge-docs">Docs</span> add marimo→myst pipeline and GLM tutorial template
- <span class="badge badge-docs">Docs</span> consolidate workflows into 4 standardized notebooks
- <span class="badge badge-docs">Docs</span> remove broken BrainCollection basics card
- <span class="badge badge-docs">Docs</span> align API doc generation with the uv-cleanup module layout
- <span class="badge badge-docs">Docs</span> standardize docstrings to Google/Markdown, automate changelog, fix migration guide
- <span class="badge badge-docs">Docs</span> fix broken cross-reference links in generated API docs
- <span class="badge badge-docs">Docs</span> document iplot() colorbar + threshold slider; ipywidgets in JupyterLite
- <span class="badge badge-docs">Docs</span> explicit page-scoped MyST targets to silence heading-ref warnings
- <span class="badge badge-docs">Docs</span> silence remaining mystmd build warnings (frontmatter, grid, docstrings)
- <span class="badge badge-docs">Docs</span> regenerate API reference (fetch_pain signature + docstring fixes)
- <span class="badge badge-docs">Docs</span> update SPEC status header to reflect implemented state
- <span class="badge badge-docs">Docs</span> regenerate API reference after BrainCollection bring-over
- <span class="badge badge-docs">Docs</span> regenerate collection_core after seq-tiebreak
- <span class="badge badge-docs">Docs</span> regenerate API reference for statistic rename + permutation removal
- <span class="badge badge-docs">Docs</span> fix docstring/RST-leakage bucket + regenerate API reference
- <span class="badge badge-docs">Docs</span> regenerate API reference (F068/F021/F182 + owed drift)
- <span class="badge badge-docs">Docs</span> wire encoding + isc notebooks for in-browser WASM data (#3673)
- <span class="badge badge-docs">Docs</span> commit the 0.6.0 pre-release hygiene audit record
- <span class="badge badge-docs">Docs</span> fix stale Adjacency.similarity/regress/isc claims
- ⚠ **Breaking** <span class="badge badge-docs">Docs</span> reorder summary tables to Parameters/Attributes/Classes/Methods

    - **Breaking:** visible API-doc section order changes across many pages.
- <span class="badge badge-docs">Docs</span> reconcile Args/Returns docstrings with actual signatures
- <span class="badge badge-docs">Docs</span> tidy module docstrings (dedupe, drop leftover headings)
- <span class="badge badge-docs">Docs</span> tidy first-line docstring summaries for griffe tables
- <span class="badge badge-docs">Docs</span> add BrainCollection basics notebook
- <span class="badge badge-docs">Docs</span> normalize docstring style (Note: header, typos, models blank line)
- <span class="badge badge-docs">Docs</span> add interactive design tour + wire standalone-page build
- <span class="badge badge-docs">Docs</span> add static-markdown tutorial build mode (default) + simplify docs poe tasks
- <span class="badge badge-docs">Docs</span> trim CLAUDE.md to load-bearing guidance
- <span class="badge badge-docs">Docs</span> clean up and fix stale references
- <span class="badge badge-docs">Docs</span> remind to use vendored nilearn/marimo skills
- <span class="badge badge-docs">Docs</span> delete superseded SPEC.md and ridge design docs; repoint to docs/development
- <span class="badge badge-docs">Docs</span> reconcile docstrings/comments with implementation across data/stats
- <span class="badge badge-docs">Docs</span> fix similarity result-key docs to 'correlation'; add semgrep guard
- <span class="badge badge-docs">Docs</span> reconcile remaining algorithm-layer docstrings with implementation
- <span class="badge badge-docs">Docs</span> generate canonical-kwarg vocab from a single source
- <span class="badge badge-docs">Docs</span> regenerate API docs (griffe2md)
- <span class="badge badge-docs">Docs</span> integrate pikachu CUDA run, make perf doc host-aware
- <span class="badge badge-docs">Docs</span> correct the nilearn cluster-forming threshold scale
- <span class="badge badge-docs">Docs</span> batched regeneration — API sources, changelog, tail-docstring cleanup
- <span class="badge badge-docs">Docs</span> per-subject predict + sklearn cv names — migration guide, execution model, vocabulary
- ⚠ **Breaking** <span class="badge badge-docs">Docs</span> plain marimo notebooks, executed previews; defer browser support to 0.6.1

    - **Breaking:** removed nltools.templates.seed_resources and the Pyodide/IDBFS fetch path, nltools.datasets.PAIN_RESOURCES / EMOTION_METADATA / emotion_resources, scripts/build_marimo_wasm.py, the docs-wasm and test-pyodide poe tasks, the pyodide CI job, and nltools/tests/pyodide. All of it is preserved on the 0.6.1-browser branch.
- <span class="badge badge-docs">Docs</span> regenerate changelog for the commits since the batched docs pass
- <span class="badge badge-docs">Docs</span> close the gaps found by the breaking-commit audit
- <span class="badge badge-docs">Docs</span> regenerate API sources and changelog for the review-fix commits
- <span class="badge badge-docs">Docs</span> refresh pikachu CUDA baseline at 55e44f06
- <span class="badge badge-docs">Docs</span> regenerate API reference and changelog for the CUDA-verification and benchmark commits


## 0.5.0 (2023-10-31)

### Bug Fixes
- <span class="badge badge-fix">Bug Fix</span> fix documentation build errors
- <span class="badge badge-fix">Bug Fix</span> fix testing bug
- <span class="badge badge-fix">Bug Fix</span> fix #413
- <span class="badge badge-fix">Bug Fix</span> fix up test bugs, support pandas 2.0, pin numpy until we replace deepdish, use only pip for GA
- <span class="badge badge-fix">Bug Fix</span> fix #409
- <span class="badge badge-fix">Bug Fix</span> fix #392
- <span class="badge badge-fix">Bug Fix</span> fix up docs testing. merge cron and push GA files
- <span class="badge badge-fix">Bug Fix</span> fix bug in downloading data


## 0.4.6 (2022-08-15)

### Improvements
- <span class="badge badge-improvement">Improvement</span> refactored trim and winsorize to single subfunction
- <span class="badge badge-improvement">Improvement</span> refactored correlation_permutation with combined case testing.
- <span class="badge badge-improvement">Improvement</span> refactored tests
- <span class="badge badge-improvement">Improvement</span> refactored code using sourcery
- <span class="badge badge-improvement">Improvement</span> refactored code using sourcery
- <span class="badge badge-improvement">Improvement</span> refactored roi_to_brain to be much faster

### Bug Fixes
- <span class="badge badge-fix">Bug Fix</span> fixed bug
- <span class="badge badge-fix">Bug Fix</span> fixed bug
- <span class="badge badge-fix">Bug Fix</span> fixed predict problems
- <span class="badge badge-fix">Bug Fix</span> fixed bug with plotting with no Xval
- <span class="badge badge-fix">Bug Fix</span> fixed bug
- <span class="badge badge-fix">Bug Fix</span> fix error related to email alert
- <span class="badge badge-fix">Bug Fix</span> fix an undefined variable; format the code layout
- <span class="badge badge-fix">Bug Fix</span> fix undefined attribute
- <span class="badge badge-fix">Bug Fix</span> fix numpy's ValueError: The truth value of an array with more...
- <span class="badge badge-fix">Bug Fix</span> fixed dist_from_hyperplane_plot bug
- <span class="badge badge-fix">Bug Fix</span> fixed tests
- <span class="badge badge-fix">Bug Fix</span> fixed simulator test
- <span class="badge badge-fix">Bug Fix</span> fixed pytests
- <span class="badge badge-fix">Bug Fix</span> fixed cross_validation bug from updating to sklearn 0.17
- <span class="badge badge-fix">Bug Fix</span> fixed bug in analysis.Predict and updated tutorials
- <span class="badge badge-fix">Bug Fix</span> fixed bug when self.Y is empty
- <span class="badge badge-fix">Bug Fix</span> fixed bug in similarity metric
- <span class="badge badge-fix">Bug Fix</span> fixed small bug
- <span class="badge badge-fix">Bug Fix</span> fixed bug in Brain_Data.bootstrap() method related to indexing
- <span class="badge badge-fix">Bug Fix</span> fixed bug with Brain_Data.similarity
- <span class="badge badge-fix">Bug Fix</span> fixed roc figure return type
- <span class="badge badge-fix">Bug Fix</span> fixed icc bugs
- <span class="badge badge-fix">Bug Fix</span> fixed bug on Brain_Data list import
- <span class="badge badge-fix">Bug Fix</span> fixed extract_roi bug
- <span class="badge badge-fix">Bug Fix</span> fixed bugs with permutation ttest
- <span class="badge badge-fix">Bug Fix</span> fixed cross-val bug and added test
- <span class="badge badge-fix">Bug Fix</span> fixed typo
- <span class="badge badge-fix">Bug Fix</span> fixed bug
- <span class="badge badge-fix">Bug Fix</span> fixed bug with upload naming
- <span class="badge badge-fix">Bug Fix</span> fixed bug in Adjacency import function
- <span class="badge badge-fix">Bug Fix</span> fixed bugs
- <span class="badge badge-fix">Bug Fix</span> fixed downsample bug
- <span class="badge badge-fix">Bug Fix</span> fixed bugs in tutorials
- <span class="badge badge-fix">Bug Fix</span> fixed bugs in cross-validation
- <span class="badge badge-fix">Bug Fix</span> fixed bug with apply_mask
- <span class="badge badge-fix">Bug Fix</span> fixed bug with apply mask
- <span class="badge badge-fix">Bug Fix</span> fixed test tolerance
- <span class="badge badge-fix">Bug Fix</span> fixed test
- <span class="badge badge-fix">Bug Fix</span> fixed bug with Adjacency.append()
- <span class="badge badge-fix">Bug Fix</span> fix auto-rounding bug in extract_roi
- <span class="badge badge-fix">Bug Fix</span> fixed bug with reading directed flat adjacency data
- <span class="badge badge-fix">Bug Fix</span> fix mn_score bug in Roc.calculate for forced choice
- <span class="badge badge-fix">Bug Fix</span> fix forced_choice_idx in Roc.calculate function
- <span class="badge badge-fix">Bug Fix</span> fix the line up issue in analysis.py line 95
- <span class="badge badge-fix">Bug Fix</span> fixed roc forced choice plotting bug
- <span class="badge badge-fix">Bug Fix</span> fixed bug in roc forced choice plotting
- <span class="badge badge-fix">Bug Fix</span> fixed bug with create sphere and mask
- <span class="badge badge-fix">Bug Fix</span> fixed neurovault_upload method
- <span class="badge badge-fix">Bug Fix</span> fixed tests
- <span class="badge badge-fix">Bug Fix</span> fixed tests
- <span class="badge badge-fix">Bug Fix</span> fixed bug with cross-validation in predict
- <span class="badge badge-fix">Bug Fix</span> fixed bug in simulator
- <span class="badge badge-fix">Bug Fix</span> fixed bug in test_analysis
- <span class="badge badge-fix">Bug Fix</span> fixed typo
- <span class="badge badge-fix">Bug Fix</span> fixed roc forced choice accuracy
- <span class="badge badge-fix">Bug Fix</span> fixed kwargs bug
- <span class="badge badge-fix">Bug Fix</span> fix issue #152, rename Stimulus to Stim to keep the naming consistent
- <span class="badge badge-fix">Bug Fix</span> fixed bug that requires creating a copy of input data on align function.
- <span class="badge badge-fix">Bug Fix</span> fixed Brain_Data.threshold bug
- <span class="badge badge-fix">Bug Fix</span> fixed random seed issue with permutations/bootstraps and joblib
- <span class="badge badge-fix">Bug Fix</span> fixed check_random_state imports
- <span class="badge badge-fix">Bug Fix</span> fixed transform pairwise and added tests
- <span class="badge badge-fix">Bug Fix</span> fixed bug in test and crucial missing line in function
- <span class="badge badge-fix">Bug Fix</span> fixed nilearn version dependency
- <span class="badge badge-fix">Bug Fix</span> fixed bug regression bug with 2d arrays
- <span class="badge badge-fix">Bug Fix</span> fixed bugs in regression
- <span class="badge badge-fix">Bug Fix</span> fixed missing plot call
- <span class="badge badge-fix">Bug Fix</span> fixed check_brain_data bug
- <span class="badge badge-fix">Bug Fix</span> fixed typos
- <span class="badge badge-fix">Bug Fix</span> fixed bug with social relations model
- <span class="badge badge-fix">Bug Fix</span> fixed bug in fetch_localizer
- <span class="badge badge-fix">Bug Fix</span> fixed predict Y warnings.
- <span class="badge badge-fix">Bug Fix</span> fixed pearsonr bug
- <span class="badge badge-fix">Bug Fix</span> fixed bug with glover_hrf function
- <span class="badge badge-fix">Bug Fix</span> fixed bug with labels in Adjacency.plot_silhouette
- <span class="badge badge-fix">Bug Fix</span> fixed issues with labels in plot_mds
- <span class="badge badge-fix">Bug Fix</span> fixed silent errors to roi_to_brain
- <span class="badge badge-fix">Bug Fix</span> fixed doc string for roi_to_brain
- <span class="badge badge-fix">Bug Fix</span> fixed extract_roi bug and added new functionality
- <span class="badge badge-fix">Bug Fix</span> fixed import error with ipywidgets
- <span class="badge badge-fix">Bug Fix</span> fixed isc bugs
- <span class="badge badge-fix">Bug Fix</span> fixed smooth bug
- <span class="badge badge-fix">Bug Fix</span> fixed codacy recs.
- <span class="badge badge-fix">Bug Fix</span> fixed new smooth test.
- <span class="badge badge-fix">Bug Fix</span> fixed align check.
- <span class="badge badge-fix">Bug Fix</span> fixed spacing.
- <span class="badge badge-fix">Bug Fix</span> fix ga. fix bug in design matrix repr. pin pandas version until deepdish updates.
- <span class="badge badge-fix">Bug Fix</span> fixes #364
- <span class="badge badge-fix">Bug Fix</span> fixed broken tests
- <span class="badge badge-fix">Bug Fix</span> fixed cluster_summary in adjacency tutorial
- <span class="badge badge-fix">Bug Fix</span> fix ci badge. try to fix failing gallery build on ga
- <span class="badge badge-fix">Bug Fix</span> fix #396, fix #398, remove uneccesary files for doc build
- <span class="badge badge-fix">Bug Fix</span> fix nilearn warnings and onsets_to_dm warnings


