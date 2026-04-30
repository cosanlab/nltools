# nilearn.decoding — Decoding (MVPA)

Decoding tools and algorithms. Wraps scikit-learn classifiers/regressors with neuroimaging-specific feature selection (ANOVA screening), spatial regularization (graph-net, TV-L1), ensemble clustering (FREM), and searchlight analysis. All estimators take Niimg-like inputs directly and return weight maps as `Nifti1Image`.

**Source:** https://nilearn.github.io/dev/modules/decoding.html

## When to use

- Standard MVPA (cross-validated classification/regression) on whole brain or ROI: `Decoder`, `DecoderRegressor`
- Stable, interpretable weight maps via ensemble clustering: `FREMClassifier`, `FREMRegressor`
- Sparse, spatially smooth weight maps with structured priors: `SpaceNetClassifier`, `SpaceNetRegressor`
- Local information mapping (per-voxel accuracy in a sphere): `SearchLight`

## Inventory

### Classes

| Class | Purpose |
|---|---|
| `Decoder` | A wrapper for popular classification strategies in neuroimaging. |
| `DecoderRegressor` | A wrapper for popular regression strategies in neuroimaging. |
| `FREMClassifier` | State of the art decoding scheme applied to usual classifiers. |
| `FREMRegressor` | State of the art decoding scheme applied to usual regression estimators. |
| `SpaceNetClassifier` | Classification learners with sparsity and spatial priors. |
| `SpaceNetRegressor` | Regression learners with sparsity and spatial priors. |
| `SearchLight` | Implement search_light analysis using an arbitrary type of classifier. |

## Estimator string options

Classification (`Decoder`, `FREMClassifier`):
`'svc' | 'svc_l1' | 'logistic' | 'logistic_l1' | 'ridge_classifier' | 'dummy_classifier'`

Regression (`DecoderRegressor`, `FREMRegressor`):
`'svr' | 'ridge' | 'ridge_regressor' | 'dummy_regressor'`

SpaceNet penalties: `'graph-net' | 'tv-l1'`.

## Decoder

```python
Decoder(
    estimator='svc',
    mask=None,                  # Niimg-like, masker, or None (auto NiftiMasker)
    cv=None,                    # int, CV splitter, or None (default StratifiedKFold)
    param_grid=None,            # dict for inner GridSearchCV
    clustering_percentile=100,  # 100 = no ReNA clustering
    screening_percentile=20,    # ANOVA feature selection cutoff (%)
    scoring=None,               # 'accuracy'|'roc_auc'|'f1'|... or callable
    smoothing_fwhm=None,
    standardize='zscore_sample',
    target_affine=None,
    target_shape=None,
    mask_strategy='background',
    low_pass=None,
    high_pass=None,
    t_r=None,
    memory=None,
    memory_level=0,
    n_jobs=1,
    verbose=0,
)
```

Post-fit attributes:
- `cv_scores_` — dict of per-class lists of fold scores
- `coef_img_` — dict mapping class label -> `Nifti1Image` weight map
- `coef_` — averaged weights across folds
- `intercept_`
- `cv_params_` — best hyperparameters per fold
- `mask_img_`
- `classes_`
- `dummy_output_` (for dummy estimators)

Key methods: `fit(X, y, groups=None)`, `predict(X)`, `score(X, y)`, `decision_function(X)`.

## DecoderRegressor

Same constructor as `Decoder` but `estimator` defaults to `'svr'` and the default scoring is regression-appropriate (`'r2'`).

Post-fit attributes:
- `cv_scores_` — list of per-fold scores
- `coef_img_` — dict (single key) -> `Nifti1Image`
- `coef_`, `intercept_`, `cv_params_`, `mask_img_`

## FREMClassifier

Fast Regularized Ensemble of Models. Combines ReNA fast clustering with cross-validated linear classifiers, then averages weights for stability.

```python
FREMClassifier(
    estimator='svc',
    mask=None,
    cv=30,                      # ensemble size; ~30 folds recommended
    param_grid=None,
    clustering_percentile=10,   # ReNA reduction (% of voxels kept)
    screening_percentile=20,
    scoring=None,
    smoothing_fwhm=None,
    standardize='zscore_sample',
    target_affine=None,
    target_shape=None,
    mask_strategy='background',
    low_pass=None,
    high_pass=None,
    t_r=None,
    memory=None,
    memory_level=0,
    n_jobs=1,
    verbose=0,
)
```

Post-fit attributes: same as `Decoder` (`coef_img_`, `cv_scores_`, etc.).

## FREMRegressor

Same constructor as `FREMClassifier` with regression `estimator` defaults (`'svr'` / `'ridge'`).

## SpaceNetClassifier

Linear classifier with structured spatial sparsity priors (Graph-Net or TV-L1).

```python
SpaceNetClassifier(
    penalty='graph-net',        # 'graph-net'|'tv-l1'
    loss=None,                  # 'logistic' (default) for classifier
    is_classif=True,
    l1_ratios=0.5,              # float or list
    alphas=None,                # regularization path
    n_alphas=10,
    mask=None,
    target_affine=None,
    target_shape=None,
    low_pass=None,
    high_pass=None,
    t_r=None,
    max_iter=200,
    tol=5e-4,
    memory=None,
    memory_level=1,
    standardize=True,
    verbose=1,
    n_jobs=1,
    cv=8,
    fit_intercept=True,
    screening_percentile=20.0,
    debias=False,
)
```

Post-fit attributes:
- `coef_img_` — `Nifti1Image`
- `coef_`, `intercept_`
- `mask_img_`
- `cv_scores_`
- `best_model_params_`
- `alpha_grids_`, `screening_percentile_`
- `classes_`

## SpaceNetRegressor

Same as `SpaceNetClassifier` with `is_classif=False` and regression loss.

## SearchLight

Voxel-by-voxel local decoding: a classifier is fit on the voxels within a sphere centered at every voxel of interest.

```python
SearchLight(
    mask_img=None,              # required: brain mask
    process_mask_img=None,      # subset of voxels to use as sphere centers (default = mask_img)
    radius=2.0,                 # mm
    estimator='svc',
    n_jobs=1,
    scoring=None,
    cv=None,
    verbose=0,
)
```

Post-fit attribute:
- `scores_img_` — `Nifti1Image` of per-voxel cross-validated scores
- `masked_scores_` — 1D array of scores at process-mask voxels

Key methods: `fit(imgs, y, groups=None)`.

## Common patterns

### Whole-brain classification

```python
from nilearn.decoding import Decoder

decoder = Decoder(
    estimator='svc', mask=mask_img, cv=5,
    screening_percentile=5, scoring='accuracy',
    smoothing_fwhm=4, standardize='zscore_sample',
)
decoder.fit(fmri_imgs, y=labels)
print(decoder.cv_scores_)
weights_face = decoder.coef_img_['face']
```

### Continuous target regression

```python
from nilearn.decoding import DecoderRegressor

reg = DecoderRegressor(estimator='ridge', scoring='r2', cv=5,
                       mask=mask_img, standardize='zscore_sample')
reg.fit(fmri_imgs, y=age)
weight_map = reg.coef_img_[next(iter(reg.coef_img_))]
```

### FREM ensemble (more stable maps)

```python
from nilearn.decoding import FREMClassifier

frem = FREMClassifier(
    estimator='svc',
    cv=30,
    clustering_percentile=10,
    screening_percentile=20,
    standardize='zscore_sample',
    mask=mask_img,
    n_jobs=1,
)
frem.fit(fmri_imgs, y=labels)
```

### Spatially regularized classifier

```python
from nilearn.decoding import SpaceNetClassifier

sn = SpaceNetClassifier(
    penalty='graph-net',
    l1_ratios=0.5,
    cv=8,
    screening_percentile=20,
    mask=mask_img,
    standardize=True,
)
sn.fit(fmri_imgs, y=labels)
acc_map = sn.coef_img_
```

### SearchLight

```python
from nilearn.decoding import SearchLight

sl = SearchLight(
    mask_img=mask_img,
    process_mask_img=roi_mask,   # restrict centers
    radius=5.6,                  # mm
    estimator='svc',
    cv=5,
    n_jobs=1,
)
sl.fit(fmri_img, y=labels)
acc_img = sl.scores_img_
```

## Gotchas

- `n_jobs=-1` can hang or OOM on large data. Start with `n_jobs=1` and scale up after profiling.
- `screening_percentile` is on the percentile scale (0-100). Lower = more aggressive ANOVA pre-selection. Default `20` keeps the top 20% of voxels.
- `clustering_percentile` (FREM only) reduces voxels via ReNA before fitting; very low values (e.g. `10`) speed up training and improve stability.
- `Decoder.coef_img_` is a **dict** keyed by class label, not a single image. `DecoderRegressor.coef_img_` is also a dict (single key).
- For multi-class problems, `Decoder` fits one-vs-rest classifiers and exposes one weight map per class.
- `SearchLight.radius` is in **mm**, not voxels. Common values: 4-10 mm.
- `SearchLight` does not standardize by default — wrap data through a masker first if needed.
- Pass `groups=` to `fit` when CV must respect subject/run boundaries (e.g. `LeaveOneGroupOut`).
- `mask` accepts a Niimg-like or a fitted/unfitted masker; the masker route lets you reuse cleaning settings across analyses.

## See also

- `references/maskers.md` — pass `mask=NiftiMasker(...)` to share cleaning config
- `references/decomposition.md` — alternate dimensionality reduction (CanICA/DictLearning)
- Source: https://nilearn.github.io/dev/modules/decoding.html
