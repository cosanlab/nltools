# nltools 0.5.x → 0.6.0 Migration

Birds-eye comparison of user-facing API between the previous release (`cosanlab/nltools@master`) and the current `uv-cleanup` branch. One table per top-level module. Columns: **Old** | **New** | **Status**.

Status legend: `renamed`, `removed`, `moved`, `signature-changed`, `added`, `unchanged`.

---

## `nltools/__init__.py` (top-level exports)

| Old | New | Status |
|---|---|---|
| `Roc` | `Roc` | moved (`analysis.py` → `data/roc/`) |
| `set_cv` | — | removed |
| `Brain_Data` | `BrainData` | renamed |
| `Adjacency` | `Adjacency` | unchanged name (internals changed) |
| `Groupby` | — | removed |
| `Design_Matrix` | `DesignMatrix` | renamed |
| `Design_Matrix_Series` | — | removed |
| `Simulator` | `Simulator` | moved (`simulator.py` → `data/simulator/`) |
| — | `SimulateGrid` | added (moved from `simulator.py`) |
| `MNI_Template` | — | removed (replaced by `templates/`) |
| `resolve_mni_path` | — | removed (replaced by `templates/`) |
| `expand_mask` | `expand_mask` | unchanged |
| `collapse_mask` | `collapse_mask` | unchanged |
| `create_sphere` | `create_sphere` | unchanged |
| `SRM` | `SRM` | moved (`external/` → `algorithms/alignment/`) |
| `DetSRM` | `DetSRM` | moved (`external/` → `algorithms/alignment/`) |
| — | `BrainSpaceConfig` | added |
| — | `get_brainspace` | added |
| — | `set_brainspace` | added |
| — | `reset_brainspace` | added |
| — | `with_brainspace` | added |
| `data` (submodule) | `data` | unchanged |
| `datasets` (submodule) | `datasets` | unchanged |
| `analysis` (submodule) | — | removed |
| `cross_validation` (submodule) | `cross_validation` | unchanged |
| `plotting` (submodule) | `plotting` | unchanged (now package) |
| `stats` (submodule) | `stats` | unchanged (now package) |
| `utils` (submodule) | `utils` | unchanged |
| `file_reader` (submodule) | — | moved → `io/file_reader.py` |
| `mask` (submodule) | `mask` | unchanged |
| `prefs` (submodule) | — | removed (replaced by `templates/`) |
| `external` (submodule) | — | removed (moved to `algorithms/`) |
| — | `io` (submodule) | added |
| — | `templates` (submodule) | added |

---

## `nltools/analysis.py` → `nltools/data/roc/`

| Old | New | Status |
|---|---|---|
| `Roc` | `Roc` | moved (`analysis.py` → `data/roc/`) |
| `Roc.calculate` | `Roc.calculate` | unchanged |
| `Roc.plot` | `Roc.plot` | unchanged |
| `Roc.summary` | `Roc.summary` | unchanged |

---

## `nltools/cross_validation.py`

| Old | New | Status |
|---|---|---|
| `KFoldStratified` | `KFoldStratified` | unchanged |
| `KFoldStratified.split` | `KFoldStratified.split` | unchanged |
| `set_cv` | — | removed |

---

## `nltools/datasets.py`

| Old | New | Status |
|---|---|---|
| `download_nifti` | `download_nifti` | unchanged |
| `fetch_pain` | `fetch_pain` | unchanged |
| `fetch_emotion_ratings` | `fetch_emotion_ratings` | unchanged |
| `download_collection` | `fetch_neurovault_collection` | renamed |
| `get_collection_image_metadata` | — | removed |
| — | `fetch_haxby` | added |

---

## `nltools/file_reader.py` → `nltools/io/file_reader.py`

| Old | New | Status |
|---|---|---|
| `onsets_to_dm` | `onsets_to_dm` | moved + signature-changed |

**`onsets_to_dm` signature changes:**

- `sampling_freq=1/tr` → `TR=tr` (pass the TR directly; sampling frequency derived internally).
- `hrf_model` now defaults to `'glover'` — the returned `DesignMatrix` is already HRF-convolved. Pass `hrf_model=None` for raw onset regressors (old behaviour).
- `run_length` is still accepted and still required for single-run inputs.
- Event DataFrame must use BIDS-standard lowercase columns: `onset`, `duration`, `trial_type` (old code using `Onset`/`Duration`/`Stim` will raise `ValueError: The provided events data has no onset column`).
- Output column names are the raw `trial_type` values (e.g. `horizontal_checkerboard`) — the old `_c0` suffix is gone.
- An intercept column named `constant` is added automatically.

---

## `nltools/mask.py`

| Old | New | Status |
|---|---|---|
| `create_sphere` | `create_sphere` | unchanged |
| `expand_mask` | `expand_mask` | unchanged |
| `collapse_mask` | `collapse_mask` | unchanged |
| `roi_to_brain` | `roi_to_brain` | unchanged |

---

## `nltools/plotting.py` → `nltools/plotting/`

| Old | New | Status |
|---|---|---|
| `plot_brain` | — | removed |
| `plot_t_brain` | — | removed |
| `plot_interactive_brain` | `plot_interactive_brain` | moved → `plotting/brain.py` |
| — | `surface_plot` | added |
| — | `plot_flatmap` | added |
| `plot_stacked_adjacency` | `plot_stacked_adjacency` | moved → `plotting/adjacency.py` |
| `plot_mean_label_distance` | `plot_mean_label_distance` | moved → `plotting/adjacency.py` |
| `plot_between_label_distance` | `plot_between_label_distance` | moved → `plotting/adjacency.py` |
| `plot_silhouette` | `plot_silhouette` | moved → `plotting/adjacency.py` |
| `dist_from_hyperplane_plot` | `dist_from_hyperplane_plot` | moved → `plotting/prediction.py` |
| `scatterplot` | `scatterplot` | moved → `plotting/prediction.py` |
| `probability_plot` | `probability_plot` | moved → `plotting/prediction.py` |
| `roc_plot` | `roc_plot` | moved → `plotting/prediction.py` |
| `component_viewer` | `component_viewer` | moved → `plotting/decomposition.py` |

---

## `nltools/prefs.py` → `nltools/templates/`

| Old | New | Status |
|---|---|---|
| `MNI_Template` | — | removed |
| `resolve_mni_path` | — | removed |
| — | `BrainSpaceConfig` | added |
| — | `get_brainspace` | added |
| — | `set_brainspace` | added |
| — | `reset_brainspace` | added |
| — | `with_brainspace` | added |
| — | `TemplateMatch` | added |
| — | `match_resolution` | added |
| — | `get_bg_image` | added |
| — | `resolve_paths` | added |
| — | `resolve_template_name` | added |

---

## `nltools/simulator.py` → `nltools/data/simulator/`

| Old | New | Status |
|---|---|---|
| `Simulator` | `Simulator` | moved |
| `Simulator.create_cov_data` | `Simulator.create_cov_data` | unchanged |
| `Simulator.create_data` | `Simulator.create_data` | unchanged |
| `Simulator.create_ncov_data` | `Simulator.create_ncov_data` | unchanged |
| `Simulator.gaussian` | `Simulator.gaussian` | unchanged |
| `Simulator.n_spheres` | `Simulator.n_spheres` | unchanged |
| `Simulator.normal_noise` | `Simulator.normal_noise` | unchanged |
| `Simulator.sphere` | `Simulator.sphere` | unchanged |
| `Simulator.to_nifti` | `Simulator.to_nifti` | unchanged |
| `SimulateGrid` | `SimulateGrid` | moved |
| `SimulateGrid.add_signal` | `SimulateGrid.add_signal` | unchanged |
| `SimulateGrid.create_mask` | `SimulateGrid.create_mask` | unchanged |
| `SimulateGrid.fit` | `SimulateGrid.fit` | unchanged |
| `SimulateGrid.plot_grid_simulation` | `SimulateGrid.plot_grid_simulation` | unchanged |
| `SimulateGrid.run_multiple_simulations` | `SimulateGrid.run_multiple_simulations` | unchanged |
| `SimulateGrid.threshold_simulation` | `SimulateGrid.threshold_simulation` | unchanged |

---

## `nltools/stats.py` → `nltools/stats/`

| Old | New | Status |
|---|---|---|
| `pearson` | — | removed |
| `zscore` | `zscore` | moved → `stats/outliers.py`; **returns polars DataFrame** (was pandas). Call `.to_pandas()` to restore prior behaviour. |
| `fdr` | `fdr` | moved → `stats/corrections.py` |
| `holm_bonf` | `holm_bonf` | moved → `stats/corrections.py` |
| `threshold` | `threshold` | moved → `stats/corrections.py` |
| `multi_threshold` | `multi_threshold` | moved → `stats/corrections.py` |
| `winsorize` | `winsorize` | moved → `stats/outliers.py` |
| `trim` | `trim` | moved → `stats/outliers.py` |
| `find_spikes` | `find_spikes` | moved → `stats/outliers.py`; **returns polars DataFrame** (was pandas). Call `.to_pandas()` before using `.iloc`/other pandas-only methods. |
| `calc_bpm` | `calc_bpm` | moved → `stats/timeseries.py` |
| `downsample` | `downsample` | moved → `stats/timeseries.py` |
| `upsample` | `upsample` | moved → `stats/timeseries.py` |
| `make_cosine_basis` | `make_cosine_basis` | moved → `stats/timeseries.py` |
| `fisher_r_to_z` | `fisher_r_to_z` | moved → `stats/correlation.py` |
| `fisher_z_to_r` | `fisher_z_to_r` | moved → `stats/correlation.py` |
| `correlation` | — | removed |
| — | `compute_similarity` | added → `stats/correlation.py` |
| — | `compute_multivariate_similarity` | added → `stats/correlation.py` |
| — | `compute_icc` | added → `stats/correlation.py` |
| `transform_pairwise` | `transform_pairwise` | moved → `stats/correlation.py` |
| `one_sample_permutation` | `one_sample_permutation_test` | renamed + moved → `stats/permutation.py` (delegates to `algorithms/inference/`) |
| `two_sample_permutation` | `two_sample_permutation_test` | renamed + moved |
| `correlation_permutation` | `correlation_permutation_test` | renamed + moved |
| `matrix_permutation` | `matrix_permutation_test` | renamed + moved |
| — | `timeseries_correlation_permutation_test` | added |
| `circle_shift` | `circle_shift` | moved → `stats/permutation.py` |
| `phase_randomize` | `phase_randomize` | moved → `stats/permutation.py` |
| `double_center` | `double_center` | moved → `stats/permutation.py` |
| `u_center` | `u_center` | moved → `stats/permutation.py` |
| `distance_correlation` | `distance_correlation` | moved → `stats/permutation.py` |
| `align` | `align` | moved → `stats/alignment.py` |
| `align_states` | `align_states` | moved → `stats/alignment.py` |
| `procrustes` | `procrustes` | moved → `stats/alignment.py` |
| `procrustes_distance` | `procrustes_distance` | moved → `stats/alignment.py` |
| `isc` | `isc` | moved → `stats/intersubject.py` |
| `isc_group` | `isc_group` | moved → `stats/intersubject.py` |
| `isfc` | `isfc` | moved → `stats/intersubject.py` |
| `isps` | `isps` | moved → `stats/intersubject.py` |
| `regress` | `regress` | **restored** in `stats/regression.py` as OLS-only (legacy `mode='robust'`/`'arma'` removed). Returns the same `(b, se, t, p, df, res)` tuple. For 4D brain data use `BrainData.fit(model='glm', X=dm)` + `.compute_contrasts(...)`. |
| `regress_permutation` | — | removed |
| `summarize_bootstrap` | — | removed |
| `MAX_INT` (const) | — | removed |

---

## `nltools/utils.py`

| Old | New | Status |
|---|---|---|
| `get_resource_path` | `get_resource_path` | unchanged |
| `attempt_to_import` | `attempt_to_import` | unchanged |
| `all_same` | `all_same` | unchanged |
| `concatenate` | `concatenate` | unchanged |
| `AmbiguityError` | — | removed |
| `to_h5` | — | moved → `io/h5.py` |
| `get_anatomical` | — | removed |
| `get_mni_from_img_resolution` | — | removed |
| `set_algorithm` | — | removed |
| `set_decomposition_algorithm` | — | removed |
| `isiterable` | — | removed |
| `check_brain_data` | — | removed |
| `check_brain_data_is_single` | — | removed |
| `check_square_numpy_matrix` | — | removed |
| `generate_jitter` | — | removed |

---

## `nltools/data/brain_data.py::Brain_Data` → `nltools/data/braindata/::BrainData`

| Old | New | Status |
|---|---|---|
| `Brain_Data` | `BrainData` | renamed |
| `.shape` | `.shape` (property) | unchanged |
| `.dtype` | `.dtype` (property) | unchanged |
| — | `.X` (property) | added |
| — | `.Y` (property) | added |
| `.isempty` | `.is_empty` (property) | renamed |
| `.empty` | `.create_empty` | renamed |
| `.mean` | `.mean` | unchanged |
| `.median` | `.median` | unchanged |
| `.std` | `.std` | unchanged |
| `.sum` | `.sum` | unchanged |
| `.to_nifti` | `.to_nifti` | unchanged |
| `.write` | `.write` | unchanged |
| `.scale` | `.scale` | unchanged |
| `.plot` | `.plot` | unchanged |
| `.iplot` | — | removed |
| — | `.plot_flatmap` | added |
| `.regress` | — | removed — the no-arg `.regress()` path now raises `NotImplementedError`. Use `.fit(model='glm', X=dm)`, then access `.glm_betas`, `.glm_t`, `.glm_p`, `.glm_se`, `.glm_residual`, `.glm_r2` on the fitted `BrainData`; compute contrasts with `.compute_contrasts(vector_or_string)`. |
| `.randomise` | — | removed |
| `.ttest` | `.ttest` | restored — one-sample voxelwise t-test across images. Returns `dict` (`{"t", "p"}`, or `{"mean", "p"}` when `permutation=True`) matching `Adjacency.ttest` convention. `.ttest2(other)` added for two-sample tests. |
| `.append` | `.append` | unchanged |
| `.similarity` | `.similarity` | unchanged |
| `.distance` | `.distance` | unchanged |
| `.multivariate_similarity` | `.multivariate_similarity` | unchanged |
| `.predict` | `.predict` | unchanged |
| `.predict_multi` | — | removed |
| — | `.fit` | added |
| — | `.cv` | added |
| — | `.compute_contrasts` | added |
| `.apply_mask` | `.apply_mask` | unchanged |
| `.extract_roi` | `.extract_roi` | unchanged |
| `.icc` | `.icc` | unchanged |
| `.detrend` | `.detrend` | unchanged |
| `.copy` | `.copy` | unchanged |
| `.upload_neurovault` | `.upload_neurovault` | unchanged |
| `.r_to_z` | `.r_to_z` | unchanged |
| `.z_to_r` | `.z_to_r` | unchanged |
| `.filter` | `.filter` | unchanged |
| `.astype` | `.astype` | unchanged |
| `.standardize` | `.standardize` | unchanged |
| `.groupby` | — | removed |
| `.aggregate` | — | removed |
| `.threshold` | `.threshold` | unchanged |
| `.regions` | `.regions` | unchanged |
| `.transform_pairwise` | `.transform_pairwise` | unchanged |
| `.bootstrap` | `.bootstrap` | unchanged |
| `.decompose` | `.decompose` | unchanged |
| `.align` | `.align` | unchanged |
| `.smooth` | `.smooth` | unchanged |
| `.find_spikes` | `.find_spikes` | unchanged |
| `.temporal_resample` | `.temporal_resample` | unchanged |
| — | `.resample_to` | added |

---

## `nltools/data/brain_data.py::Groupby`

| Old | New | Status |
|---|---|---|
| `Groupby` | — | removed |
| `Groupby.apply` | — | removed |
| `Groupby.combine` | — | removed |
| `Groupby.split` | — | removed |

---

## `nltools/data/adjacency.py::Adjacency` → `nltools/data/adjacency/::Adjacency`

| Old | New | Status |
|---|---|---|
| `Adjacency` | `Adjacency` | unchanged name |
| `.shape` | `.shape` (property) | unchanged |
| `.square_shape` | `.to_square` | renamed |
| — | `.n_nodes` (property) | added |
| — | `.vector_shape` (property) | added |
| — | `.Y` (property) | added |
| `.isempty` | `.is_empty` (property) | renamed |
| `.squareform` | `.squareform` | unchanged |
| `.plot` | `.plot` | unchanged |
| `.mean` | `.mean` | unchanged |
| `.median` | `.median` | unchanged |
| `.std` | `.std` | unchanged |
| `.sum` | `.sum` | unchanged |
| `.copy` | `.copy` | unchanged |
| `.append` | `.append` | unchanged |
| `.write` | `.write` | unchanged |
| `.similarity` | `.similarity` | unchanged |
| `.distance` | `.distance` | unchanged |
| `.r_to_z` | `.r_to_z` | unchanged |
| `.z_to_r` | `.z_to_r` | unchanged |
| `.threshold` | `.threshold` | unchanged |
| `.to_graph` | `.to_graph` | unchanged |
| `.ttest` | `.ttest` | unchanged |
| `.plot_label_distance` | `.plot_label_distance` | unchanged |
| `.stats_label_distance` | `.stats_label_distance` | unchanged |
| `.plot_silhouette` | `.plot_silhouette` | unchanged |
| `.plot_mds` | `.plot_mds` | unchanged |
| `.bootstrap` | `.bootstrap` | unchanged |
| `.isc` | — | removed |
| `.isc_group` | — | removed |
| `.distance_to_similarity` | `.distance_to_similarity` | unchanged |
| `.cluster_summary` | `.cluster_summary` | unchanged |
| `.regress` | `.regress` | unchanged |
| `.social_relations_model` | `.social_relations_model` | unchanged |
| `.generate_permutations` | `.generate_permutations` | unchanged |

---

## `nltools/data/design_matrix.py::Design_Matrix` → `nltools/data/designmatrix/::DesignMatrix`

| Old | New | Status |
|---|---|---|
| `Design_Matrix` | `DesignMatrix` | renamed |
| `Design_Matrix_Series` | — | removed |
| `.details` | `.details` | unchanged |
| `.append` | `.append` | signature-changed (now supports `axis=1` column concat) |
| `.vif` | `.vif` | unchanged |
| `.heatmap` | — | removed — use `.plot()` (heatmap-style) or `sns.heatmap(dm.to_pandas())` |
| `.info` | — | removed — use `.details()` |
| `.head` | — | removed — use `.to_pandas().head()` |
| `.corr` | — | removed — use `.to_pandas().corr()` |
| `.iloc` | — | removed — use `.to_pandas().iloc` |
| `.plot(ax=...)` | `.plot(figsize=...)` | signature-changed — `.plot()` creates its own figure; passing `ax=` conflicts with the internal `sns.heatmap` call |
| `.convolve` | `.convolve` | unchanged |
| `.downsample` | `.downsample` | unchanged |
| `.upsample` | `.upsample` | unchanged |
| `.zscore` | `.zscore` | unchanged |
| `.add_poly` | `.add_poly` | unchanged |
| `.add_dct_basis` | `.add_dct_basis` | unchanged |
| `.replace_data` | `.replace_data` | unchanged |
| `.clean` | `.clean` | unchanged |
| — | `.shape` (property) | added |
| — | `.columns` (property) | added |
| — | `.is_empty` (property) | added |
| — | `.copy` | added |
| — | `.drop` | added |
| — | `.fillna` | added |
| — | `.reset_index` | added |
| — | `.standardize` | added |
| — | `.sum` | added |
| — | `.to_numpy` | added |
| — | `.to_pandas` | added |
| — | `.write` | added |

---

## `nltools/data/` — new collection / results classes

| Old | New | Status |
|---|---|---|
| — | `BrainCollection` | added (`data/collection/`) |
| — | `BrainCollection.{align, anova, compute_contrasts, cv, detrend, filter, fit, fit_from_events, fit_glm, fit_ridge, from_bids, from_glob, from_stacked, is_loaded, isc, isc_test, iter_batches, load, map, mask, max, mean, median, memory_estimate, metadata, min, n_images, n_voxels, permutation_test, permutation_test2, predict, select_feature, shape, smooth, standardize, std, sum, threshold, to_list, to_stacked, to_tensor, ttest, ttest2, unload, var, write}` | added |
| — | `Fit` (frozen dataclass, `data/fitresults/`) | added |
| — | `Fit.asdict`, `Fit.available` | added |

---

## `nltools/external/` → `nltools/algorithms/`

| Old | New | Status |
|---|---|---|
| `external.SRM` | `algorithms.alignment.SRM` | moved |
| `external.DetSRM` | `algorithms.alignment.DetSRM` | moved |
| `external.hrf.spm_hrf` | `algorithms.hrf.spm_hrf` | moved |
| `external.hrf.glover_hrf` | `algorithms.hrf.glover_hrf` | moved |
| `external.hrf.spm_time_derivative` | `algorithms.hrf.spm_time_derivative` | moved |
| `external.hrf.glover_time_derivative` | `algorithms.hrf.glover_time_derivative` | moved |
| `external.hrf.spm_dispersion_derivative` | `algorithms.hrf.spm_dispersion_derivative` | moved |
| `SRM.fit`, `SRM.transform`, `SRM.transform_subject` | same | unchanged |
| `DetSRM.fit`, `DetSRM.transform`, `DetSRM.transform_subject` | same | unchanged |
| — | `algorithms.alignment.HyperAlignment` | added |
| — | `HyperAlignment.{fit, transform, transform_subject, common_model_}` | added |
| — | `algorithms.alignment.LocalAlignment` | added |
| — | `LocalAlignment.{fit, fit_transform, transform, iter_neighborhoods}` | added |

---

## `nltools/algorithms/` — new (no pre-0.6 analogue)

| Old | New | Status |
|---|---|---|
| — | `algorithms.ridge.solve_ridge_cv` | added |
| — | `algorithms.ridge.solve_banded_ridge_cv` | added |
| — | `algorithms.ridge.ridge_svd` | added |
| — | `algorithms.ridge.ridge_cv` | added |
| — | `algorithms.backends.Backend` (class) | added |
| — | `algorithms.backends.resolve_backend` | added |
| — | `algorithms.backends.assert_array_almost_equal` | added |
| — | `algorithms.backends.check_gpu_available` | added |
| — | `algorithms.backends.auto_select_backend` | added |
| — | `algorithms.random.get_random_state` | added |
| — | `algorithms.random.generate_seeds` | added |
| — | `algorithms.random.generate_sign_flips` | added |
| — | `algorithms.random.generate_bootstrap_indices` | added |
| — | `algorithms.shape_utils.extract_triangle_elements` | added |
| — | `algorithms.shape_utils.permute_matrix_symmetric` | added |
| — | `algorithms.shape_utils.ensure_2d` | added |
| — | `algorithms.shape_utils.batch_or_skip` | added |
| — | `algorithms.inference.one_sample_permutation_test` | added |
| — | `algorithms.inference.two_sample_permutation_test` | added |
| — | `algorithms.inference.correlation_permutation_test` | added |
| — | `algorithms.inference.timeseries_correlation_permutation_test` | added |
| — | `algorithms.inference.matrix_permutation_test` | added |
| — | `algorithms.inference.isc_permutation_test` | added |
| — | `algorithms.inference.isc_group_permutation_test` | added |
| — | `algorithms.inference.circle_shift` | added |
| — | `algorithms.inference.phase_randomize` | added |
| — | `algorithms.inference.double_center` | added |
| — | `algorithms.inference.u_center` | added |
| — | `algorithms.inference.distance_correlation` | added |
| — | `algorithms.inference.compute_icc_voxelwise` | added |
| — | `algorithms.inference.OnlineBootstrapStats` (class) | added |

---

## `nltools/io/` — new

| Old | New | Status |
|---|---|---|
| `utils.to_h5` | `io.h5.to_h5` | moved |
| — | `io.h5.is_h5_path` | added |
| — | `io.h5.load_brain_data_h5` | added |
| `file_reader.onsets_to_dm` | `io.file_reader.onsets_to_dm` | moved |

---

## `nltools/models/` — new

| Old | New | Status |
|---|---|---|
| — | `BaseModel` (`models/base.py`) | added |
| — | `BaseModel.{fit, predict, score}` | added |
| — | `Ridge` (`models/ridge.py`) | added |
| — | `Ridge.{fit, predict, score}` | added |
| — | `Glm` (`models/glm.py`) | added |
| — | `Glm.{fit, predict, score, compute_contrast, design_matrices_, glm_, residuals}` | added |

---

## `nltools/pipelines/` — new

| Old | New | Status |
|---|---|---|
| — | `Pipeline` | added |
| — | `CVScheme`, `CVSchemeImpl`, `NestedCVScheme` | added |
| — | `FittedStack` | added |
| — | `TransformStep`, `FittedTransform`, `Terminal` (protocols) | added |
| — | `NormalizeStep`, `ReduceStep`, `PipeStep` | added |
| — | `AlignStep`, `FittedAlign` | added |
| — | `PredictTerminal`, `ISCTerminal`, `RSATerminal` | added |
| — | `CVResult`, `FoldResult`, `ISCResult`, `RSAResult`, `PermutationResult` | added |
| — | `MultiSubjectPipeline` | added |
| — | `PooledData`, `StatResult`, `ResultDict` | added |

---

## `nltools/templates/` — new

| Old | New | Status |
|---|---|---|
| `prefs.MNI_Template` | — | removed |
| `prefs.resolve_mni_path` | — | removed |
| — | `BrainSpaceConfig` | added |
| — | `get_brainspace` | added |
| — | `set_brainspace` | added |
| — | `reset_brainspace` | added |
| — | `with_brainspace` | added |
| — | `TemplateMatch` | added |
| — | `match_resolution` | added |
| — | `get_bg_image` | added |
| — | `resolve_paths` | added |
| — | `resolve_template_name` | added |
