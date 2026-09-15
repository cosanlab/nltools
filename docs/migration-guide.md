# Migrating from v0.5.1

nltools 0.6.0 delegates to nilearn, scikit-learn and polars instead of
re-implementing them. The data classes keep their jobs, but their names, keyword
vocabulary and return types line up with each other and with the libraries
underneath: `Brain_Data` is `BrainData`, `nltools.stats` is `nltools.algorithms`,
prediction and regression return typed records instead of loose dictionaries, and
one keyword name means one thing everywhere. There are no compatibility aliases —
a v0.5.1 name either has a listed replacement or is gone.
`BrainCollection` and collection-level execution are deferred to 0.6.1; in 0.6.0,
apply `BrainData` methods per subject and stack the results with
[`concatenate`](api/nltools.md).

## Quick reference

| Area | v0.5.1 | v0.6.0 | Note |
| --- | --- | --- | --- |
| Data classes | `Brain_Data`, `Design_Matrix` | `BrainData`, `DesignMatrix` | PEP 8 names; no aliases |
| Data classes | `brain.shape()`, `brain.isempty()` | `brain.shape`, `brain.is_empty` | Properties, not methods |
| Data classes | `brain.empty()` | `brain.create_empty()` | `empty` was ambiguous with the predicate |
| Data classes | `brain.nifti_masker` | `nilearn.masking.apply_mask(img, brain.mask)` | Removed |
| Data classes | `brain.apply_mask(mask, resample_mask_to_brain=True)` | `brain.apply_mask(mask)` | The mask must already sit on the data's grid; `resample()` first |
| Data classes | `brain.threshold(2, -2)`, `brain.filter(0.5, 0.008)` | `brain.threshold(upper=2, lower=-2)`, `brain.filter(sampling_freq=0.5, high_pass=0.008)` | Options are keyword-only after the data argument |
| Data classes | `brain.decompose(algorithm='ica')` | `brain.decompose(method='ica')` | `method` names the algorithm; `metric` is reserved for distances |
| Data classes | `brain.append(data, **kwargs)` | `brain.append(data, *, ignore_attrs=False)` | Extra keywords are no longer forwarded to pandas |
| Data classes | `adjacency.shape()` (vector), `square_shape()` | `adjacency.shape` (square), `adjacency.vector_shape` | The logical shape is the default |
| Data classes | `Groupby`, `brain.groupby()`, `brain.aggregate()` | Removed, no successor | Loop over `extract_roi` output |
| Data classes | `brain.icc()` | Removed, no successor | Use `pingouin.intraclass_corr` |
| Data classes | `nltools.prefs.MNI_Template` | `set_brainspace()`, `get_brainspace()`, `reset_brainspace()`, `with_brainspace()` | Functions, not a mutable singleton |
| Data classes | Legacy (≤ 0.5.1) HDF5 files load | `BrainData(path)` raises `ValueError` | Export to NIfTI or CSV under 0.5.1 before upgrading |
| Data classes | Resampling to a resolution kept only the positive-coordinate octant — 8,039 of 238,955 voxels on the 2 mm MNI mask | `brain.resample(resolution=3)` keeps all 70,831 voxels | The target grid encloses the source field of view instead of starting at world zero |
| Data classes | `len(single_image)` was its voxel count and `single_image[0]` raised | `len()` is 1, and a single image is indexable and iterable | Inherited v0.5.1 behaviour |
| Data classes | A one-voxel mask squeezed three images into one three-voxel image | The voxel axis is kept; only a leading singleton image axis collapses | Affects objects whose mask holds a single voxel |
| Data classes | `brain[0] = other` matched metadata columns by position and turned a mixed `Int64`/`String` frame into `Object` | Columns are matched by name and keep their dtypes; a name mismatch raises | Inherited v0.5.1 behaviour; the check is the one `append` already applied |
| Data classes | `X` or `Y` with a row count that did not match the data was accepted | Construction and assignment raise `ValueError` | The mismatch used to surface later, from an unrelated operation |
| Data classes | `BrainData(other, mask=submask)` reinterpreted the packed columns under the new mask | The values are re-extracted onto the new support | Voxels keep their locations; a wider mask widens with zeros |
| Data classes | `BrainData([bd1, bd2])` dropped `.X`/`.Y` and ignored `mask=` | Row metadata is concatenated and `mask=` is applied | Same result as `concatenate` followed by the mask |
| Data classes | `BrainData(path.h5, mask=other)` reinterpreted the stored columns under the new mask | They are re-extracted onto that mask, and a stored fit is dropped when the voxel axis changes | The warning now says the stored mask describes the stored columns |
| Data classes | `a + b`, `a.append(b)` and `a[0] = b[0]` combined objects whose voxels were different voxels | Each raises `ValueError` unless the two grids and supports match | Inherited v0.5.1 laxity; use `resample()` and `apply_mask()` to establish a common voxel axis |
| Data classes | `brain.threshold(lower=3, cluster_threshold=50)` returned the positive tail, and `upper=3` kept the negative one too | The clustered result keeps the same tail as the unclustered one | nilearn's `two_sided` keys off the cutoff's sign, not the tail you asked for |
| Data classes | `brain.standardize()` cast the result back to integer or boolean input dtype | Integer and boolean data comes back floating | A centred int32 column of `[0, 1]` was `[0, 0]`; float32 input is unchanged |
| Data classes | `brain.filter(runs=…, sample_mask=…)` gave every run's rows the first run's `X` and `Y` | Row metadata follows the rows nilearn returns | `clean` cleans one run at a time in `np.unique(runs)` order |
| Design matrix | pandas subclass | polars-backed class | `.to_numpy()`, `.with_columns()`, `.corr()` |
| Design matrix | `Design_Matrix_Series` | Removed, no successor | A single column is a `polars.Series` |
| Design matrix | `dm.polys` | `dm.confounds` | Read-only; set columns as confounds when you append them |
| Design matrix | `dm.zscore(columns=…)` | `dm.standardize(method='zscore', columns=…)` | Keyword-only; the default `method='center'` centers only |
| Design matrix | `dm.convolve(conv_func='hrf')` | `dm.convolve(kernel='glover')` | Six nilearn HRF names or an array; `'hrf'` is gone |
| Design matrix | `dm.convolved` listed the source columns | `dm.convolved` lists the `<col>_c{i}` outputs | The suffix and the dropped source column are unchanged |
| Design matrix | Convolved values from a nipy-derived kernel | nilearn's `compute_regressor` at `oversampling=50` | Every beta, t and contrast moves ([#492](https://github.com/cosanlab/nltools/issues/492)) |
| Design matrix | `poly_0`, `cosine_1`, `global_spike1` | `.nl_poly_0`, `.nl_cosine_1`, `.nl_global_spike1` | Generated columns live in the reserved `.nl_` namespace |
| Design matrix | `dm.vif(exclude_polys=…)`, `dm.clean(exclude_polys=…, verbose=…)` | `dm.vif(exclude_confounds=…)`, `dm.clean(exclude_confounds=…, progress_bar=…)` | Renamed with `.polys` |
| Design matrix | `dm.append(dm=…)` | `dm.append(data=…)` | `data` is the second operand on all three data classes |
| Design matrix | `dm.heatmap()` | `dm.plot()` | One plotting method name per class |
| Design matrix | `find_spikes` emitted duplicate regressors | One regressor per spike | The design is full rank again |
| Design matrix | `dm.convolve()` silently replaced an existing `<col>_c0` column | `convolve` raises `ValueError` naming the column | Rename or drop the column before convolving |
| Design matrix | `find_spikes` put a `diff_spike` indicator on the volume *before* the frame-to-frame jump | The indicator marks the volume the jump moved into | The framewise-displacement convention; regenerated confound regressors change, and a spike that used to collide with a global detection may now add a column |
| Design matrix | `find_spikes` raised on a 4D NIfTI with a singleton spatial axis (a single-slice acquisition) | It returns the spike design | Only trailing singleton axes are squeezed, so `(x, y, z, t, 1)` input still works |
| GLM | `brain.regress(mode='ols')` → dict | `brain.fit(model='glm', X=dm)`, read `brain.model`, then `compute_contrasts` | The fit is a frozen `FitResult` with `betas`, `predicted`, `residual` and `r2`; t/p are per contrast, not per regressor |
| GLM | `nltools.stats.regress(X, Y, mode=…)` | `nltools.algorithms.regress(X, Y, *, stats=…, tail=…)` | OLS was the only working mode; robust/ARMA are gone |
| GLM | `brain.randomise(...)` | `brain.ttest(permutation=True)` | Voxelwise permutation on the entry point that already existed |
| GLM | `adjacency.regress(X, mode='ols')` | `adjacency.regress(X)` | Same removal as the standalone `regress`; `tail` is keyword-only |
| GLM | Before this fix, a ridge `FitResult` scored a non-finite voxel `0.0` | That voxel's `r2` is `NaN` | v0.5.1 had no ridge `FitResult`; a finite constant target still scores exactly `0.0` |
| Prediction | `brain.predict(algorithm='svm', cv_dict=…)` → dict | `brain.predict(y=…, estimator=…, cv=…)` → `PredictResult` | Frozen record with `weight_map`, `scores`, `predictions` |
| Prediction | `brain.predict_multi(...)` | `brain.predict(spatial_scale='roi'\|'searchlight')` | One entry point, three spatial scales |
| Prediction | `set_cv(Y, cv_dict)` | `cv=<int>` or an sklearn splitter, plus `groups=` | Removed; an int is that many unshuffled stratified folds |
| Prediction | `algorithm='svm'`, `'logistic'`, `'svr'` | `estimator='linear_svc'`, `'logistic_regression'`, `'linear_svr'` | Abbreviations are rejected by name; `linear`, `lassopcr` and the `*CV` variants are gone |
| Prediction | `algorithm='ridge'` fitted `Ridge()` at its default penalty | `estimator='ridge'` fits `RidgeCV` over a 1e-3…1e6 grid | Pass `estimator_kwargs={'alphas': …}` for a fixed penalty |
| Prediction | `Roc(threshold_type='optimal_overall')` | `Roc(method='optimal_overall')` | Keyword-only, and `calculate` takes `method=` too |
| Prediction | `Roc` reported metrics from non-finite decision values | `Roc` raises `ValueError` | One NaN score made every criterion NaN, so nothing counted as misclassified |
| Prediction | An `(n, 1)` score column reported accuracy 0.5 on perfectly separated data | The column is read as one value per observation | `misclass` came back as an `(n, n)` matrix |
| Prediction | Integer forced-choice scores reported accuracy 0.0 for a perfect subject | Scores are coerced to float | Pair centering was truncated back into the integer array |
| Prediction | AUC came from a 50-points-per-observation grid: tied scores gave 0.0, perfectly ordered data 0.75 | AUC comes from the data's own operating points, matching `sklearn.metrics.roc_curve` | `class_thr` and `criterion_values` move with it; `criterion_values` now ends at `np.inf`, the corner `class_thr` is never chosen from |
| Prediction | Forced-choice `calculate()` gave one answer on the first call and another from the second on | Every call gives the same answer | Pair centering is derived per call instead of overwriting `input_values` |
| Prediction | Forced-choice accuracy depended on the row order of the two classes | Errors are paired by subject id | A reordering of the same observations gave 0.0 where 0.5 is correct |
| Prediction | An unknown `scoring` name returned an all-NaN `score_map` for searchlight and ROI decoding | Raises, naming the scorer | Whole-brain decoding already raised on the identical call |
| Similarity | `brain.similarity(image=…, method=…)` | `brain.similarity(data, *, metric=…)` | `metric` is the similarity metric everywhere |
| Similarity | `adjacency.similarity(perm_type=…, ignore_diagonal=…)` | `adjacency.similarity(data, *, method=…, include_diag=…)` | Polarity is flipped, so the default now excludes the diagonal |
| Similarity | `adjacency.cluster_summary(metric=…, summary=…)` | `adjacency.cluster_summary(summary=…, scope=…)` | `summary` is the central tendency; `scope` is within/between |
| Similarity | `brain.extract_roi(metric=…)` | `brain.extract_roi(method=…)` | `metric` is reserved for distances |
| Similarity | `brain.multivariate_similarity(images, method='ols')` | `brain.multivariate_similarity(images, tail=2)` | OLS was the only mode |
| Similarity | Manual per-ROI loop to paint an RSA map | `brain.distance(spatial_scale='roi', roi_mask=atlas)` then `roi_to_brain_from_atlas` | Explicit atlas mapping |
| Similarity | `brain.similarity(other)` compared array position against array position whenever the two masks kept the same number of voxels | Masks that differ are intersected first | Equal support size is not equal support |
| Similarity | `brain.extract_roi(atlas)` raised when the atlas's labels filled its mask | The atlas is classified by its nonzero labels | A `{1, 2}` atlas with no background is an atlas, not a binary mask |
| Alignment | `from nltools.external import SRM, DetSRM` | `brain.align(method='probabilistic_srm'\|'deterministic_srm')` | The estimators are internal |
| Alignment | Procrustes back-projection was `transformed @ T` | `transformed @ T.T` | `transformation_matrix` is stored as `transformed = original @ T` |
| Alignment | Procrustes aligned `Brain_Data` subjects with different voxel counts by zero-padding the feature axis | `align(method='procrustes')` raises; pass the `.data` arrays to get the padded result | The padded result has no mask that can describe it |
| Alignment | An SRM `n_features` above a subject's voxel count silently fit a smaller model (`deterministic_srm`) or failed with a numpy broadcast error (`probabilistic_srm`) | Both SRM estimators raise, naming the subject and both counts | The returned model was never the one that was asked for |
| Alignment | `align_states(..., replace_zero_variance=True)` on integer maps raised `ValueError: matrix contains invalid numeric entries` | The maps are converted to float, so the replacement noise survives | The non-replacement path still keeps the caller's dtype |
| Statistics | `nltools.stats` | `nltools.algorithms` | Same functions, one namespace |
| Statistics | `one_sample_permutation`, `two_sample_permutation`, `correlation_permutation`, `matrix_permutation` | Same names with a `_test` suffix | Keyword-only after the data arguments |
| Statistics | `adjacency.generate_permutations(n_perm=…)` | `generate_permutations(n_permute=…)` | One spelling for a permutation count everywhere |
| Statistics | `return_perms=True` → `out['perm_dist']` | `return_null=True` → `out['null_dist']` | One name for the null distribution |
| Statistics | `correlation_permutation`, `matrix_permutation` defaulted to `metric='spearman'` | Both default to `metric='pearson'` | Pass `metric='spearman'` to reproduce v0.5.1 numbers |
| Statistics | `correlation_permutation(method='circle_shift'\|'phase_randomize')` | Removed | Shift or randomize with `circle_shift`/`phase_randomize` before the test |
| Statistics | `brain.ttest(threshold_dict={'fdr': .05})` | `brain.ttest(popmean=…, permutation=…)` then `nltools.algorithms.threshold` | Thresholding is its own step |
| Statistics | `adjacency.ttest(permutation=…, **kwargs)` | Same keywords as `BrainData.ttest` | Returns `mean`, `t`, `z`, `p` |
| Statistics | `brain.bootstrap('mean', save_weights=…)` → dict | `brain.bootstrap('mean', return_samples=…)` → `BootstrapResult` | `estimate`, `standard_error`, `ci_lower`, `ci_upper` |
| Statistics | `summarize_bootstrap`, `regress_permutation` | Removed | The bootstrap entry points return the summary |
| Statistics | `double_center`, `u_center` | Removed | Internal steps of `distance_correlation` |
| Statistics | `adjacency.isc()`, `adjacency.isc_group()` | `nltools.algorithms.isc()`, `isc_group()` | Standalone functions on arrays |
| Statistics | `transform_pairwise` alternated its sign flip over candidate pairs, leaving the classes unbalanced | The `+1` and `-1` classes come out balanced | Pairs the group and target filters skip no longer consume an alternation slot |
| Statistics | `adjacency.bootstrap` on a single matrix resampled its edges and returned a smaller matrix | Raises `ValueError` | It resamples matrices, so it needs at least two, like `adjacency.ttest` |
| Statistics | The social relations model took its grand mean before masking the diagonal | Every SRM average excludes the diagonal | Only a directed matrix whose diagonal holds self-ratings moves; a NaN diagonal is unchanged |
| Statistics | The social relations model paired each cell with the wrong reciprocal cell | Each dyad is paired with its own transpose | Variance components, reciprocity, reliabilities and the total variance all move |
| Statistics | The SRM summary printed two-sided p-values above one when `t` was negative | `2 * sf(abs(t), df)` | Printed covariance rows only; no returned value changes |
| Statistics | `brain.bootstrap('mean')` on a single image resampled its voxels as observations | Raises `ValueError` | The old result was one scalar wrapped in the source's full mask, which could not be written or plotted |
| Statistics | `exclude_self_corr` masked every resampled pair whose correlation reached 1 | Only the pairs a resampled subject forms with itself are masked | Two distinct subjects that genuinely correlate at +1 or -1 now count, and identity is the only correct reading for `metric='euclidean'` or `'cosine'` |
| Statistics | An ISC bootstrap counted undefined draws in the p-value denominator and let one of them make the whole interval `NaN` | Undefined draws are dropped per feature: the denominator is that feature's valid-draw count and the interval is a NaN-aware percentile | The same data gave p = 1/6 as 2-D input and p = 1/21 once a singleton voxel axis was added; `return_null=True` now hands back every draw, `NaN` where a draw was undefined, rather than a shortened array |
| Statistics | A feature whose ISC was undefined reported `p = 1 / (n + 1)`, the smallest p the correction can produce, with an interval taken over the resamples that survived | `p` and both bounds are `NaN` for that feature | Every flat or masked-out voxel used to surface as the most significant hit in a whole-brain map |
| Statistics | `holm_bonf(np.array([0.03, 0.04]))` returned `0.04` | It returns `-1` | Holm is a step-down procedure, so the walk stops at the first p above its boundary; the new threshold is never larger than the old. `fdr` is a step-up procedure and is unchanged |
| Statistics | `threshold(stat, p)` resolved the two images spatially, through the NIfTI grid | `threshold` and `multi_threshold` raise `ValueError` unless the two images cover the same voxels | The masked-array rewrite applied the p-values by array position, so two images with disjoint masks of equal size thresholded the wrong voxels |
| Statistics | `multi_threshold` returned a cumulative map for a multi-image `t_map`; `threshold` returned an image-count vector when nothing survived | Both keep the input's `(n_images, n_voxels)` shape | 0.6.0 raised a broadcast `ValueError` on the multi-image path ([#304](https://github.com/cosanlab/nltools/issues/304), [#372](https://github.com/cosanlab/nltools/issues/372)) |
| Statistics | `winsorize(x, cutoff={'quantile': [0, .75]})` on `[nan, 0, 1, 2, 100]` gave `[nan, 0, 1, 2, 26.5]` | The same values | The polars rewrite returned all-NaN; quantile cutoffs skip missing values again, so `winsorize(trim(x))` is meaningful |
| Statistics | `winsorize([0, 1, 2, 3, 4], cutoff={'quantile': [.25, .75]}, replace_with_cutoff=False)` returned all `2.0` | It returns `[1, 1, 2, 3, 3]` | A value sitting exactly on a cutoff is not an outlier, so it is its own closest existing value |
| Statistics | `regress` reported `df = n - p`, so `X = ones((4, 2))`, `Y = 0..3` gave `df=2`, `se=sqrt(5/32)`, `p=0.198` | `df = n - rank(X)`: `df=3`, `se=sqrt(5/48)` | A duplicated regressor no longer changes the inference; full-rank designs are bit-identical. `Adjacency.regress` reports the same df |
| Statistics | An integer design overflowed in `X.T @ X`: a column of `100000` gave `t=0.710`, `p=0.529` where the same design in float gives `t=3.873`, `p=0.030` | `X` and `Y` are promoted to float first | Only designs large enough to overflow move; 0/1 dummies are exact either way |
| Statistics | `regress` on an undefined fit (a NaN in `Y`) returned `t=0`, `p=1` next to a NaN coefficient | `t` and `p` are NaN | A perfect fit, whose standard error is finite but near zero, still scores `t=0`, `p=1` |
| Statistics | `downsample(range(10), sampling_freq=5, target=2, target_type='hz')` returned `[0.5, 2.5, 4.5, 7.5]` | It returns `[1, 3.5, 6, 8.5]` | Rows bin by `floor(row / n_samples)`, the rule `DesignMatrix.downsample` already used; integer ratios are unchanged |
| Statistics | `upsample` dropped non-numeric columns with a `UserWarning` | The same, naming the dropped columns | The polars rewrite raised `ValueError: could not convert string to float`; a frame with no numeric column now raises instead |
| Statistics | `make_cosine_basis(nsamples, sampling_freq=…)` | `make_cosine_basis(nsamples, sampling_interval=…)` | The value was always the sampling interval in seconds (SPM's `RT`); the name now says so. Positional calls are unchanged |
| Plotting | `brain.plot(view=…, threshold_upper=…, axes=…)` | `brain.plot(method=…, upper=…, ax=…)` | `ax` is the matplotlib spelling on every class |
| Plotting | `adjacency.plot(limit, axes, *args)` | `adjacency.plot(*, limit=3, ax=None)` | Keyword-only; no positional passthrough |
| Plotting | `plot_brain`, `plot_t_brain`, `plot_interactive_brain` | Removed | `BrainData.plot(method='glass'\|'mni'\|'full')` and `BrainData.iplot` |
| Plotting | `roc_plot`, `scatterplot`, `probability_plot`, `dist_from_hyperplane_plot` | `Roc.plot()`, `BrainData.predict(plot=True)` | Drawn by the object that holds the results |
| Plotting | `plot_mean_label_distance`, `plot_between_label_distance`, `plot_silhouette` | `Adjacency.plot_label_distance` and friends | Methods on `Adjacency` |
| Plotting | `plot_stacked_adjacency(a1, a2)` | `a1.plot_stacked(a2)` | A method on `Adjacency`; each triangle keeps its own scale instead of being normalized onto a shared one |
| Plotting | `brain.iplot(threshold=0, surface=…)` | `brain.iplot(view=…, threshold=…, atlas=…)` | Rebuilt on niivue |
| Plotting | Glass brains hid a percentile of voxels (nilearn's `threshold='auto'`) | `brain.plot(method='glass')` draws every voxel | The colorbar keeps its 0 tick; pass `threshold=` for a cutoff |
| Plotting | `adjacency.plot()` drew every matrix on a sequential ramp | Matrices whose off-diagonal values cross zero use `RdBu_r`, centered at 0 with symmetric limits | One-signed matrices are unchanged; `cmap`, `center`, `vmin`, `vmax` still win |
| Plotting | `adjacency.squareform()` always wrote a zero diagonal | The diagonal follows `matrix_type`: 1 for a similarity, 0 for a distance | `Adjacency(sim.squareform())` now round-trips as a similarity |
| Plotting | `adjacency.plot()` on a stack with shared labels tick-labelled panel *i* with label *i* alone | Every panel carries all the node labels | A nested per-matrix label grid is unchanged |
| Plotting | `brain.plot(method='glass'\|'slices', ax=…)` ignored `ax` and drew into a figure of its own | `ax` reaches nilearn as `axes=` | The caller's figure is returned and left open |
| Plotting | `brain.predict(plot=True)` drew the ROC and margin figures but not the weight map | All three figures are drawn | The weight map was drawn and immediately closed |
| IO | `onsets_to_dm(f, sampling_freq, run_length)` | `DesignMatrix(events_path, run_length=…, TR=…)` | HRF-convolves by default; `hrf_model=None` for boxcars |
| IO | `from nltools.external import glover_hrf` | `from nilearn.glm.first_level import glover_hrf` | The five HRF wrappers were pass-throughs |
| Datasets | `fetch_pain(data_dir=…, resume=…, verbose=1)` | `fetch_pain(verbose=0)` | Caching is handled for you; same for `fetch_emotion_ratings` |
| Datasets | `download_collection`, `get_collection_image_metadata` | `fetch_neurovault_collection(collection_id)` | One function |
| Datasets | `get_anatomical()` | `nilearn.datasets.load_mni152_brain_mask()` | Removed |
| Simulator | `create_data` returned data on the template mask | The data keeps the `brain_mask` the `Simulator` was built with | Only affects a custom `brain_mask`; the default path is byte-identical |
| Simulator | `create_ncov_data` sized every covariance block by the row region | Each block is sized by its own region | The matrix was asymmetric for regions of unequal size, so simulated values move |
| Simulator | `create_ncov_data` broadcast one noise image across every repetition | Noise is drawn per repetition, as in `create_cov_data` | `sigma` now varies within subject outside the regions |
| Simulator | `SimulateGrid` signal masks were a pixel wider than `signal_width` at mismatched parity, and the hit rates divided by `signal_width**2` | The mask is exactly `signal_width` wide and both rates divide by its size | The true-positive rate could read 1.78; same-parity widths, including every default, are unchanged |
| Simulator | `SimulateGrid`'s false discovery rate counted positive discoveries only, and was NaN with none | Every non-zero discovery counts; an empty map scores 0.0 | `multiple_fdr` no longer carries NaN into its mean |
| Simulator | `SimulateGrid.add_signal` left the previous fit in place and `fit()` then raised | `add_signal` clears the fit, so `fit()` runs again | Statistics no longer describe data the object has replaced |
| Simulator | `plot_grid_simulation` redrew a cached thresholded map | It re-thresholds with the arguments it was given | The middle panel and its caption describe the same threshold |
| Simulator | `corrected_threshold` recorded `fdr(p)` at the default q of 0.05 | It records the cutoff at the requested q | The recorded value used to be -1 for maps with surviving pixels |
| Masks | `collapse_mask` returned an all-zero template-length vector for custom-space data and kept overlaps shared by fewer than all masks | It collapses on the input's own voxel axis and drops every overlapping voxel | `to_nifti()` on a custom-space result used to raise; 2-mask and disjoint inputs are unchanged |
| Masks | `expand_mask` rounded the caller's mask to int32 in place | The input is left untouched | The output's int32 labels are unchanged |
| Masks | `roi_to_brain_from_atlas(roi_labels=[0, …])` painted label 0 with that parcel's value | A 0 in `roi_labels` raises `ValueError` | Label 0 is background and receives `fill` |
| Cluster reports | `peaks` took `volume_mm3` and `n_voxels` from nilearn's 6-connected cluster sizes | Peak rows carry their parent cluster's 26-connected extent | A join of `peaks` and `clusters` no longer reports two sizes for one cluster |
| Cluster reports | `cluster_report(stat_threshold=t)` kept voxels equal to `t` | The threshold is exclusive | Matches `get_clusters_table`, which drives the peak table |
| Brain space | Voxel size was read off the affine's diagonal | Voxel size is the affine's column norms (`nibabel.affines.voxel_sizes`) | A rotated or permuted isotropic MNI image is recognized instead of refused; axis-aligned affines are unchanged |

## Renames and import paths

Old class names raise `ImportError`; moved modules raise `ModuleNotFoundError`.
Two find-and-replace passes cover the class renames:

```bash
sd 'Brain_Data' 'BrainData' **/*.py
sd 'Design_Matrix' 'DesignMatrix' **/*.py
```

| v0.5.1 import | v0.6.0 import |
| --- | --- |
| `from nltools.data import Brain_Data, Design_Matrix` | `from nltools import BrainData, DesignMatrix` |
| `from nltools.simulator import Simulator, SimulateGrid` | `from nltools import Simulator, SimulateGrid` |
| `from nltools.analysis import Roc` | `from nltools import Roc` |
| `from nltools.stats import …` | `from nltools.algorithms import …` |
| `from nltools.file_reader import onsets_to_dm` | `DesignMatrix(events_path, …)`, or `from nltools.io import events_to_dm` |
| `from nltools.external import glover_hrf, spm_hrf, …` | `from nilearn.glm.first_level import glover_hrf, spm_hrf, …` |
| `from nltools.external import SRM, DetSRM` | `BrainData.align(...)`, or `from nltools.algorithms import align` |
| `from nltools.utils import concatenate` | `from nltools import concatenate` |
| `from nltools.utils import get_resource_path` | `from nltools.datasets import get_resource_path` |
| `from nltools.utils import get_anatomical, all_same` | Removed |
| `from nltools.cross_validation import set_cv` | Removed — pass `cv=` and `groups=` to `BrainData.predict` |
| `from nltools.prefs import MNI_Template` | `from nltools import set_brainspace, get_brainspace, reset_brainspace, with_brainspace` |
| `from nltools.plotting import plot_brain, roc_plot, …` | Methods on the data classes; `component_viewer` is the one standalone plot left |

The four v0.5.1 mask functions keep their home in [`nltools.mask`](api/mask.md),
joined by `roi_to_brain_from_atlas`.

## Examples

Every example below runs against the same simulated data.

```python exec="on" source="above" result="text" session="mig"
import matplotlib
import numpy as np

matplotlib.use("Agg")  # this page is rendered without a display
import matplotlib.pyplot as plt

from nltools import Adjacency, BrainData, DesignMatrix, Simulator
from nltools.mask import create_sphere

roi = create_sphere([0, -18, 18], radius=8)
brain = Simulator(random_state=0).create_data([0.0, 1.0] * 10, 1.0, reps=1).apply_mask(roi)
conditions = np.asarray(brain.Y).ravel()
print(brain.shape, conditions[:6])
```

### Data classes

Shape and emptiness are properties rather than methods.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: brain.shape(), brain.isempty(), brain.empty()
print(brain.shape, brain.is_empty, brain.create_empty().shape)
```

`Adjacency.shape` is the logical square shape; the packed vector length moved to
`vector_shape`.

```python exec="on" source="above" result="text" session="mig"
rng = np.random.default_rng(1)
square = rng.normal(size=(10, 10))
square = (square + square.T) / 2
np.fill_diagonal(square, 0)
adjacency = Adjacency(square, matrix_type="similarity")

# v0.5.1: adjacency.shape() was (45,) and adjacency.square_shape() was (10, 10)
print(adjacency.shape, adjacency.vector_shape)
```

### Design matrices and GLM

`DesignMatrix` is backed by polars. Column assignment goes through
`with_columns`, and `to_numpy()` is the escape hatch. Generated columns —
intercepts, polynomial trends, cosine bases, spike regressors — carry the
reserved `.nl_` prefix so they can never collide with a column of yours.

```python exec="on" source="above" result="text" session="mig"
import polars as pl

design = DesignMatrix({"stim": np.tile([0.0, 1.0], 10)}, sampling_freq=0.5)
design = design.add_poly(1)

# v0.5.1: dm['stim_sq'] = dm['stim'] ** 2
design = design.with_columns((pl.col("stim") ** 2).alias("stim_sq"))

# v0.5.1: ['stim', 'stim_sq', 'poly_0', 'poly_1'] and dm.polys
print(design.columns, design.confounds)
```

`standardize(method='zscore')` replaces `zscore()`. Called without `columns=`
it standardizes the non-confound columns only, where `zscore()` rescaled the
intercept and drift terms along with everything else.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: design.zscore(columns=['stim'])
standardized = design.standardize(method="zscore", columns=["stim"])
print(dict(zip(standardized.columns, standardized.to_numpy().std(axis=0).round(2))))
```

Convolution names the kernel with `kernel=`. The output column keeps the
`_c{i}` suffix it had in v0.5.1, and `.convolved` now lists those output names
rather than the source names.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: DesignMatrix(...).convolve(conv_func='hrf'); .convolved was ['stim']
convolved = DesignMatrix({"stim": np.tile([0.0, 1.0], 10)}, sampling_freq=0.5).convolve()
print(convolved.columns, convolved.convolved)
```

The kernel itself is nilearn's. `convolve()` calls
[`compute_regressor`](https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.compute_regressor.html)
at `oversampling=50`, and building a `DesignMatrix` from an events file calls
`make_first_level_design_matrix`, so both produce exactly what a nilearn
`FirstLevelModel` would. v0.5.1's kernel peaked around 8 s instead of 5 s at
TR = 2 ([#492](https://github.com/cosanlab/nltools/issues/492)), so every
convolved regressor — and every beta, t and contrast downstream — changes. Array
kernels are unaffected.

Reading an events file is now the constructor's job; `onsets_to_dm` is gone.

```python exec="on" source="above" result="text" session="mig"
from pathlib import Path

from nltools.datasets import get_resource_path

events = Path(get_resource_path()) / "onsets_example.csv"

# v0.5.1: onsets_to_dm(events, sampling_freq=0.5, run_length=200)
events_design = DesignMatrix(events, run_length=200, TR=2.0)
print(events_design.shape, events_design.columns[:3])

# Raw boxcars, for a design you convolve yourself (PPI, FIR)
print(DesignMatrix(events, run_length=200, TR=2.0, hrf_model=None).columns[:3])
```

`BrainData.regress()` is gone. Fit the model, then ask for the contrasts you
want: t, p and standard errors are per contrast, not per regressor.

```python exec="on" source="above" result="text" session="mig"
fit_design = DesignMatrix({"stim": conditions}, sampling_freq=0.5).add_poly(0)

# v0.5.1: brain.X = df; out = brain.regress(); out['beta'], out['t'], out['p']
fitted = brain.fit(model="glm", X=fit_design)
print(fitted.model.betas.shape, round(float(fitted.model.r2.data.mean()), 4))

result = fitted.compute_contrasts({"stim": [1, 0]}, inference=True)["stim"]
print(type(result).__name__, result.statistic.shape, result.degrees_of_freedom)
```

The standalone OLS helper survives the move to `nltools.algorithms` but loses
`mode=`; the robust and ARMA fits are gone.

```python exec="on" source="above" result="text" session="mig"
from nltools.algorithms import regress

X = np.column_stack([np.ones(20), conditions])

# v0.5.1: regress(X, Y, mode='ols')
betas, se, t, p, df, residuals = regress(X, rng.normal(size=(20, 5)))
print(betas.shape, df[0])
```

### Prediction

`predict` takes the labels, the estimator and the cross-validation scheme as
three separate keywords and returns a frozen `PredictResult` record. There is no
`cv_dict` and no `set_cv`: pass an integer (that many unshuffled stratified
folds), or any sklearn splitter, plus `groups=` when folds must respect subjects.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: out = brain.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 3})
predicted = brain.predict(y=conditions, estimator="linear_svc", cv=3)
print(type(predicted).__name__, predicted.weight_map.shape, len(predicted.scores))
```

`spatial_scale` replaces `predict_multi`: the same call runs whole-brain, per-ROI
or searchlight decoding. `estimator='ridge'` and `'ridge_classifier'` now fit
`RidgeCV`/`RidgeClassifierCV` over a 1e-3…1e6 grid inside the per-fold scaler
pipeline instead of a single default penalty; pass
`estimator_kwargs={'alphas': [1.0]}` or your own sklearn object to pin one.

`Roc` spells its threshold rule `method=`, and its arguments are keyword-only.

```python exec="on" source="above" result="text" session="mig"
from nltools import Roc

# v0.5.1: Roc(scores, labels, threshold_type='optimal_overall')
roc = Roc(
    input_values=np.asarray(predicted.predictions, dtype=float),
    binary_outcome=conditions.astype(bool),
    method="optimal_overall",
)
roc.calculate()
print(round(roc.accuracy, 3), round(roc.auc, 3))
```

### Similarity and RSA

The compared object is `data` on both classes, and the metric is `metric=`.
`Adjacency.similarity` renames `perm_type=` to `method=` and flips the diagonal
flag to `include_diag=`.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: brain.similarity(image=brain[0], method='correlation')
print(np.round(brain.similarity(brain[0], metric="correlation")[:3], 3))

other = rng.normal(size=(10, 10))
other = (other + other.T) / 2
np.fill_diagonal(other, 0)

# v0.5.1: adjacency.similarity(other, perm_type='2d', ignore_diagonal=True)
out = adjacency.similarity(
    Adjacency(other, matrix_type="similarity"),
    method="2d",
    metric="spearman",
    include_diag=False,
    n_permute=100,
    random_state=0,
)
print({k: round(float(v), 3) for k, v in out.items()})
```

`cluster_summary` moves the central tendency to `summary=` and the
within/between choice to `scope=`.

```python exec="on" source="above" result="text" session="mig"
labels = ["left"] * 5 + ["right"] * 5

# v0.5.1: adjacency.cluster_summary(clusters=labels, metric='mean', summary='within')
summary = adjacency.cluster_summary(clusters=labels, summary="mean", scope="within")
print({k: round(v, 3) for k, v in summary.items()})
```

For a searchlight or ROI RSA, `BrainData.distance(spatial_scale='roi',
roi_mask=atlas)` builds the `Adjacency` stack that the v0.5.1 workflow assembled
by hand, and `nltools.mask.roi_to_brain_from_atlas` paints the reduced values
back onto the atlas.

### Alignment

Both alignment entry points now store `transformation_matrix` in one
orientation, `transformed = original @ T`, so back-projection is
`transformed @ T.T` everywhere. v0.5.1's `BrainData.align` stored the transpose
of this, contradicting its own docstring.

```python exec="on" source="above" result="text" session="mig"
target = Simulator(random_state=1).create_data([0.0, 1.0] * 10, 1.0, reps=1).apply_mask(roi)
aligned = brain.align(target, method="procrustes")

# v0.5.1: aligned['transformed'].data @ aligned['transformation_matrix'].data
recovered = aligned["transformed"].data @ aligned["transformation_matrix"].data.T
centered = brain.data - brain.data.mean(axis=0)
print(round(float(np.corrcoef(recovered.ravel(), centered.ravel())[0, 1]), 6))
```

Shared response models arrive through the same method:
`brain.align(target, method='probabilistic_srm')` or `'deterministic_srm'`, and
`nltools.algorithms.align` aligns a list of subjects. The `SRM` and `DetSRM`
classes are no longer importable — the method builds them for you.

### Statistics and inference

Everything from `nltools.stats` lives in `nltools.algorithms`. The four
permutation tests gained a `_test` suffix, everything after the data arguments is
keyword-only, and `return_perms=` became `return_null=`.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: from nltools.stats import one_sample_permutation
#         one_sample_permutation(x, n_permute=200, return_perms=True)
from nltools.algorithms import one_sample_permutation_test

print({k: round(v, 3) for k, v in one_sample_permutation_test(
    rng.normal(size=50), n_permute=200, random_state=0
).items()})
```

`ttest` takes a `popmean` and an optional permutation test, and no longer
thresholds for you; `nltools.algorithms.threshold` is the separate step.

```python exec="on" source="above" result="text" session="mig"
from nltools.algorithms import fdr, threshold

# v0.5.1: brain.ttest(threshold_dict={'fdr': .05})
t_test = brain.ttest(popmean=0.0)
print(sorted(t_test), t_test["t"].shape)

q = fdr(t_test["p"].data, q=0.05)
print(round(q, 5), threshold(t_test["t"], t_test["p"], thr=q).shape)
```

`Adjacency.ttest` returns the same four maps as `BrainData.ttest` — the mean is
its own key, and `t` is the t-statistic rather than the edgewise mean.

Bootstrapping returns a `BootstrapResult`: an estimate, a standard error, and a
central percentile interval at `confidence_level`. `save_weights=` became
`return_samples=`.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: out = brain.bootstrap('mean', n_samples=1000); out['Z'], out['p']
boot = brain.bootstrap("mean", n_samples=1000, confidence_level=0.95, random_state=0)
print(type(boot).__name__, boot.estimate.shape, round(float(boot.ci_lower.mean()), 3))
```

Intersubject correlation is a function, not an `Adjacency` method:
`nltools.algorithms.isc`, `isc_group`, `isfc` and `isps` take arrays and return
dictionaries.

### Plotting

The standalone plotters are gone; each class draws itself, and the axis keyword
is `ax` everywhere, matching matplotlib.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: design.heatmap(); adjacency.plot(limit=1, axes=ax)
figure = design.plot()
adjacency.plot(limit=1)

# v0.5.1: plot_brain(brain.mean(), how='glass', thr_upper=1)
brain.mean().plot(method="glass", upper=1)
plt.close("all")
print(type(figure).__name__)
```

`BrainData.plot` renames `threshold_upper`/`threshold_lower` to `upper`/`lower`
and moves v0.5.1's `view='glass'|'mni'|'full'` onto `method=`; `view=` is now the
slice axis. `iplot` is rebuilt on niivue and takes `view=`, `threshold=`,
`atlas=` and an `autoscale`/`symmetric` display pair.
[`component_viewer`](api/plotting.md) is the only standalone plotting function
left.

### IO and datasets

The bundled-dataset fetchers cache to a shared location of their own, so
`data_dir=` and `resume=` are gone and `verbose` is quiet by default. These two
examples download data, so they are not executed when this page is built.

```python
from nltools.datasets import fetch_pain, fetch_neurovault_collection

# v0.5.1: fetch_pain(data_dir='~/nltools_data', resume=True, verbose=1)
pain = fetch_pain()

# v0.5.1: download_collection(collection=504); get_collection_image_metadata(504)
metadata, files = fetch_neurovault_collection(504)
```

Atlases and bundled resources are looked up by name with
[`load_atlas`, `list_atlases`, `fetch_resource` and `list_resources`](api/datasets.md).
HDF5 files written by 0.5.1 or earlier no longer load: export them to NIfTI or
CSV under 0.5.1 first, then read them back in 0.6.0.

## Removals and their replacements

| v0.5.1 | What to do instead |
| --- | --- |
| `Groupby`, `Brain_Data.groupby()`, `Brain_Data.aggregate()` | Removed, no successor — extract each region with `BrainData.extract_roi` and work with the returned array |
| `Design_Matrix_Series` | Removed, no successor — indexing a `DesignMatrix` column returns a `polars.Series` |
| `set_cv` | Pass `cv=<int>` or an sklearn splitter to `BrainData.predict`, with `groups=` when folds must hold out subjects |
| `Roc(threshold_type=…)` | `Roc(method=…)`, keyword-only |
| `plot_brain`, `plot_t_brain` | `BrainData.plot(method='glass'\|'mni'\|'full')`; run the t-test yourself with `BrainData.ttest` |
| `plot_interactive_brain` | Removed, no successor — `BrainData.iplot` |
| `plot_stacked_adjacency` | `Adjacency.plot_stacked(data, ...)` — the same two triangles, plus titles, node labels and a per-triangle color scale |
| `fetch_pain(data_dir=…, resume=…, verbose=1)` | `fetch_pain(verbose=0)` — caching is internal |
| `Brain_Data.icc()`, `compute_icc` | Removed, no successor — `pingouin.intraclass_corr` on `brain.data` |
| `double_center`, `u_center` | Removed — internal steps of `nltools.algorithms.distance_correlation` |
| `Brain_Data.nifti_masker` | `nilearn.masking.apply_mask(img, brain.mask)` |
| `summarize_bootstrap`, `regress_permutation` | The bootstrap methods return a `BootstrapResult`; permute with `nltools.algorithms.one_sample_permutation_test` |
| `nltools.utils.all_same`, `get_anatomical`, `set_algorithm` | Removed — `numpy`, `nilearn.datasets`, and the `estimator=` keyword cover them |

## What did not change

`BrainData`, `Adjacency` and `DesignMatrix` still mean what they meant in v0.5.1,
and the methods that were already well named — `threshold`, `detrend`, `filter`,
`r_to_z`, `distance`, `to_nifti`, `write` — keep their names and their results.
Masks are still nibabel images, `BrainData.data` is still a 2-D NumPy array of
images by voxels, `Y` and `X` still hold labels and designs, and
`KFoldStratified` and the four v0.5.1 functions in
[`nltools.mask`](api/mask.md) are unchanged. Reading a NIfTI file, a list of
files, or a NeuroVault URL into `BrainData` works as before, and now accepts
`Path` objects too.
