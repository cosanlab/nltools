# Migrating from v0.5.1

nltools 0.6.0 stops re-implementing what nilearn, scikit-learn and polars already
do well and delegates to them instead. The data classes keep their jobs, but their
names, keyword vocabulary and return types are now consistent with each other and
with the libraries underneath: `Brain_Data` is `BrainData`, `nltools.stats` is
`nltools.algorithms`, prediction and regression return typed records instead of
loose dictionaries, and one keyword name means one thing everywhere. There are no
compatibility aliases — a v0.5.1 name either has a listed replacement or is gone.
`BrainCollection` and collection-level execution are deferred to 0.6.1; in 0.6.0,
apply `BrainData` methods per subject and stack the results with
[`concatenate`](api/nltools.md).

## Quick reference

| Area | v0.5.1 | v0.6.0 | Note |
| --- | --- | --- | --- |
| Data classes | `Brain_Data`, `Design_Matrix` | `BrainData`, `DesignMatrix` | PEP 8 names; no aliases |
| Data classes | `brain.shape()`, `brain.isempty()` | `brain.shape`, `brain.is_empty` | Properties, not methods |
| Data classes | `brain.empty()` | `brain.create_empty()` | `empty` was ambiguous with the predicate |
| Data classes | `brain.smooth(6)` mutates | `brain.smooth(6)` returns a copy | Every transform returns a new object |
| Data classes | `brain.nifti_masker` | `nilearn.masking.apply_mask(img, brain.mask)` | Removed |
| Data classes | `brain.apply_mask(mask, resample_mask_to_brain=True)` | `brain.apply_mask(mask)` | The mask must already sit on the data's grid; `resample()` first |
| Data classes | `brain.resample(img=target)` took the target's support | `brain.resample(img=target)` takes only the grid | The source mask is carried over |
| Data classes | `brain.append(data, **kwargs)` | `brain.append(data, *, ignore_attrs=False)` | Extra keywords are no longer forwarded to pandas |
| Data classes | `adjacency.shape()` (vector), `square_shape()` | `adjacency.shape` (square), `adjacency.vector_shape` | The logical shape is the default |
| Data classes | `Groupby`, `brain.groupby()`, `brain.aggregate()` | Removed, no successor | Loop over `extract_roi` output |
| Data classes | `brain.icc()` | Removed, no successor | Use `pingouin.intraclass_corr` |
| Data classes | `nltools.prefs.MNI_Template` | `set_brainspace()`, `get_brainspace()`, `reset_brainspace()`, `with_brainspace()` | Functions, not a mutable singleton |
| Data classes | Legacy (≤ 0.5.1) HDF5 files load | `BrainData(path)` raises `ValueError` | Export to NIfTI or CSV under 0.5.1 before upgrading |
| Design matrix | pandas subclass | polars-backed class | `.to_numpy()`, `.with_columns()`, `.corr()` |
| Design matrix | `Design_Matrix_Series` | Removed, no successor | A single column is a `polars.Series` |
| Design matrix | `dm.polys` | `dm.confounds` | Read-only; set columns as confounds when you append them |
| Design matrix | `dm.zscore(columns=…)` | `dm.standardize(method='zscore', columns=…)` | Keyword-only; the default `method='center'` centers only |
| Design matrix | `dm.convolve(conv_func='hrf')` | `dm.convolve(kernel='glover')` | Six nilearn HRF names or an array; `'hrf'` is gone |
| Design matrix | 1-D kernel kept the column name | Always `<col>_c{i}` | The source column is dropped |
| Design matrix | Convolved values from a nipy-derived kernel | nilearn's `compute_regressor` at `oversampling=50` | Every beta, t and contrast moves ([#492](https://github.com/cosanlab/nltools/issues/492)) |
| Design matrix | `poly_0`, `cosine_1`, `global_spike1` | `.nl_poly_0`, `.nl_cosine_1`, `.nl_global_spike1` | Generated columns live in the reserved `.nl_` namespace |
| Design matrix | `dm.vif(exclude_polys=…)`, `dm.clean(exclude_polys=…, verbose=…)` | `dm.vif(exclude_confounds=…)`, `dm.clean(exclude_confounds=…, progress_bar=…)` | Renamed with `.polys` |
| Design matrix | `dm.append(dm=…)` | `dm.append(data=…)` | `data` is the second operand on all three data classes |
| Design matrix | `dm.heatmap()` | `dm.plot()` | One plotting method name per class |
| Design matrix | `find_spikes` emitted duplicate regressors | One regressor per spike | The design is full rank again |
| GLM | `brain.regress(mode='ols')` → dict | `brain.fit(model='glm', X=dm)` then `compute_contrasts` | t/p are per contrast, not per regressor |
| GLM | `nltools.stats.regress(X, Y, mode=…)` | `nltools.algorithms.regress(X, Y, *, stats=…, tail=…)` | OLS was the only working mode; robust/ARMA are gone |
| GLM | `brain.randomise(...)` | `brain.ttest(permutation=True)` | Voxelwise permutation on the entry point that already existed |
| GLM | `fit` cleaned the design implicitly | `fit` estimates the design you pass | Call `dm.clean()` yourself |
| Prediction | `brain.predict(algorithm='svm', cv_dict=…)` → dict | `brain.predict(y=…, estimator=…, cv=…)` → `Predict` | Frozen record with `weight_map`, `scores`, `predictions` |
| Prediction | `brain.predict_multi(...)` | `brain.predict(spatial_scale='roi'\|'searchlight')` | One entry point, three spatial scales |
| Prediction | `set_cv(Y, cv_dict)` | `cv=<int>` or an sklearn splitter, plus `groups=` | Removed; an int is that many unshuffled stratified folds |
| Prediction | `algorithm='ridge'` fitted `Ridge()` at its default penalty | `estimator='ridge'` fits `RidgeCV` over a 1e-3…1e6 grid | Pass `estimator_kwargs={'alphas': …}` for a fixed penalty |
| Prediction | `Roc(threshold_type='optimal_overall')` | `Roc(method='optimal_overall')` | Keyword-only, and `calculate` takes `method=` too |
| Similarity | `brain.similarity(image=…, method=…)` | `brain.similarity(data, *, metric=…)` | `metric` is the similarity metric everywhere |
| Similarity | `adjacency.similarity(perm_type=…, ignore_diagonal=…)` | `adjacency.similarity(data, *, method=…, include_diag=…)` | Polarity of the diagonal flag is flipped |
| Similarity | `adjacency.cluster_summary(metric=…, summary=…)` | `adjacency.cluster_summary(summary=…, scope=…)` | `summary` is the central tendency; `scope` is within/between |
| Similarity | `brain.extract_roi(metric=…)` | `brain.extract_roi(method=…)` | `metric` is reserved for distances |
| Similarity | `brain.multivariate_similarity(images, method='ols')` | `brain.multivariate_similarity(images, tail=2)` | OLS was the only mode |
| Similarity | Manual per-ROI loop to paint an RSA map | `brain.distance(spatial_scale='roi', roi_mask=atlas)` then `roi_to_brain_from_atlas` | Explicit atlas mapping |
| Alignment | `from nltools.external import SRM, DetSRM` | `brain.align(method='probabilistic_srm'\|'deterministic_srm')` | The estimators are internal |
| Alignment | Procrustes back-projection was `transformed @ T` | `transformed @ T.T` | `transformation_matrix` is stored as `transformed = original @ T` |
| Statistics | `nltools.stats` | `nltools.algorithms` | Same functions, one namespace |
| Statistics | `one_sample_permutation`, `two_sample_permutation`, `correlation_permutation`, `matrix_permutation` | Same names with a `_test` suffix | Keyword-only after the data arguments |
| Statistics | `n_perm=` | `n_permute=` | Including `adjacency.generate_permutations` |
| Statistics | `show_progress=True` | `progress_bar=False` | Renamed, and off by default |
| Statistics | `brain.ttest(threshold_dict={'fdr': .05})` | `brain.ttest(popmean=…, permutation=…)` then `nltools.algorithms.threshold` | Thresholding is its own step |
| Statistics | `adjacency.ttest(permutation=…, **kwargs)` | Same keywords as `BrainData.ttest` | Returns `mean`, `t`, `z`, `p` |
| Statistics | `brain.bootstrap('mean', save_weights=…)` → dict | `brain.bootstrap('mean', return_samples=…)` → `BootstrapResult` | `estimate`, `standard_error`, `ci_lower`, `ci_upper` |
| Statistics | `summarize_bootstrap`, `regress_permutation` | Removed | The bootstrap entry points return the summary |
| Statistics | `double_center`, `u_center` | Removed | Internal steps of `distance_correlation` |
| Statistics | `adjacency.isc()`, `adjacency.isc_group()` | `nltools.algorithms.isc()`, `isc_group()` | Standalone functions on arrays |
| Plotting | `brain.plot(view=…, threshold_upper=…, axes=…)` | `brain.plot(method=…, upper=…, ax=…)` | `ax` is the matplotlib spelling on every class |
| Plotting | `adjacency.plot(limit, axes, *args)` | `adjacency.plot(*, limit=3, ax=None)` | Keyword-only; no positional passthrough |
| Plotting | `plot_brain`, `plot_t_brain`, `plot_interactive_brain` | Removed | `BrainData.plot(method='glass'\|'mni'\|'full')` and `BrainData.iplot` |
| Plotting | `roc_plot`, `scatterplot`, `probability_plot`, `dist_from_hyperplane_plot` | `Roc.plot()`, `BrainData.predict(plot=True)` | Drawn by the object that holds the results |
| Plotting | `plot_mean_label_distance`, `plot_between_label_distance`, `plot_silhouette` | `Adjacency.plot_label_distance` and friends | Methods on `Adjacency` |
| Plotting | `plot_stacked_adjacency` | Removed, no successor | Plot the two matrices side by side |
| Plotting | `brain.iplot(threshold=0, surface=…)` | `brain.iplot(view=…, threshold=…, atlas=…)` | Rebuilt on niivue |
| IO | `onsets_to_dm(f, sampling_freq, run_length)` | `DesignMatrix(events_path, run_length=…, TR=…)` | HRF-convolves by default; `hrf_model=None` for boxcars |
| IO | `from nltools.external import glover_hrf` | `from nilearn.glm.first_level import glover_hrf` | The six HRF wrappers were pass-throughs |
| Datasets | `fetch_pain(data_dir=…, resume=…, verbose=1)` | `fetch_pain(verbose=0)` | Caching is handled for you; same for `fetch_emotion_ratings` |
| Datasets | `download_collection`, `get_collection_image_metadata` | `fetch_neurovault_collection(collection_id)` | One function |
| Datasets | `get_anatomical()` | `nilearn.datasets.load_mni152_brain_mask()` | Removed |

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

Shape and emptiness are properties, and transforms return a new object instead of
mutating the one you called them on.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: brain.shape(), brain.isempty(), brain.empty()
print(brain.shape, brain.is_empty, brain.create_empty().shape)

# v0.5.1: brain.smooth(6) modified brain in place and returned None
print(brain.smooth(6) is brain)
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

Convolution names the kernel with `kernel=` and always suffixes the output
column, so a design says which regressors have been convolved.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: DesignMatrix(...).convolve(conv_func='hrf') kept the name 'stim'
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
print(fitted.glm_betas.shape, round(float(fitted.glm_r2.data.mean()), 4))

result = fitted.compute_contrasts({"stim": [1, 0]}, inference=True)["stim"]
print(type(result).__name__, result.statistic.shape, result.degrees_of_freedom)
```

The standalone OLS helper survives the move to `nltools.algorithms` but loses
`mode=`: OLS was its only supported value.

```python exec="on" source="above" result="text" session="mig"
from nltools.algorithms import regress

X = np.column_stack([np.ones(20), conditions])

# v0.5.1: regress(X, Y, mode='ols')
betas, se, t, p, df, residuals = regress(X, rng.normal(size=(20, 5)))
print(betas.shape, df[0])
```

### Prediction

`predict` takes the labels, the estimator and the cross-validation scheme as
three separate keywords and returns a frozen `Predict` record. There is no
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
permutation tests gained a `_test` suffix, `n_perm=` became `n_permute=`, and
`show_progress=` became `progress_bar=`, which is off by default.

```python exec="on" source="above" result="text" session="mig"
# v0.5.1: from nltools.stats import one_sample_permutation
#         one_sample_permutation(x, n_perm=200, show_progress=True)
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

# v0.5.1: plot_brain(brain.mean(), how='glass', thr_upper=2)
brain.mean().plot(method="glass", upper=2)
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
| `plot_interactive_brain`, `plot_stacked_adjacency` | Removed, no successor — `BrainData.iplot`, and plot two `Adjacency` objects separately |
| `fetch_pain(data_dir=…, resume=…, verbose=1)` | `fetch_pain(verbose=0)` — caching is internal |
| `Brain_Data.icc()`, `compute_icc` | Removed, no successor — `pingouin.intraclass_corr` on `brain.data` |
| `double_center`, `u_center` | Removed — internal steps of `nltools.algorithms.distance_correlation` |
| `Brain_Data.nifti_masker` | `nilearn.masking.apply_mask(img, brain.mask)` |
| `summarize_bootstrap`, `regress_permutation` | The bootstrap methods return a `BootstrapResult`; permute with `nltools.algorithms.one_sample_permutation_test` |
| `nltools.utils.all_same`, `get_anatomical`, `set_algorithm` | Removed — `numpy`, `nilearn.datasets`, and the `estimator=` keyword cover them |

## What did not change

`BrainData`, `Adjacency` and `DesignMatrix` still mean what they meant in v0.5.1,
and the methods that were already well named — `threshold`, `extract_roi`,
`detrend`, `filter`, `r_to_z`, `distance`, `to_nifti`, `write` — keep their names
and their results. Masks are still nibabel images, `BrainData.data` is still a
2-D NumPy array of images by voxels, `Y` and `X` still hold labels and designs,
and `KFoldStratified` and the four v0.5.1 functions in
[`nltools.mask`](api/mask.md) are unchanged. Reading a NIfTI file, a list of files, or a NeuroVault URL into
`BrainData` works exactly as before — and now accepts `Path` objects too.
