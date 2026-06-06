---
file_format: mystnb
authors:
  - Eshin Jolly
kernelspec:
  name: python3
  display_name: Python 3
execute:
  allow_errors: false
---

```{code-cell} python
:tags: [remove-cell]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

# Decoding (MVPA)

## Introduction

The GLM tutorial asks **"where in the brain is activity different between conditions?"**. Decoding flips the question:

> **Can the spatial pattern of activity *predict* which condition was being viewed?**

Same data, different lens. Instead of one regression per voxel, we run one classifier across all voxels at once, and we judge it on **held-out** data — if it generalizes, the pattern carries information.

Once you have a working decoder, two interpretation goals follow naturally — and this tutorial walks through both:

1. **Whole-brain decoding.** One classifier sees every voxel. The output is one *weight map* across the whole brain telling you which voxels contributed (with caveats — we'll get to those).
2. **ROI decoding.** Train one classifier per region using a parcellation atlas. The output is one *accuracy* per region, so you can see *where* in the brain the pattern is informative without trusting raw classifier weights.

We'll use `BrainData.predict()` for both — same call, different `method=`.

```{code-cell} python
from nltools.data import BrainData
```

## Load Haxby (one subject)

The classic MVPA dataset: one subject (`sub-2`) viewing 8 object categories across 12 runs. We'll restrict to **face vs house** — the two conditions with the strongest, best-understood signal — and let `BrainData` auto-resample to MNI 3mm:

```{code-cell} python
from nilearn.datasets import fetch_haxby

HAXBY = fetch_haxby(subjects=[2], verbose=0)
brain = BrainData(HAXBY.func[0])
print(f"brain.shape: {brain.shape}  (TRs, voxels)")
print(f"mask zooms : {brain.mask.header.get_zooms()}")
```

The label file ships per-TR condition labels and a run-id (`chunks`) per TR:

```{code-cell} python
labels = pd.read_csv(HAXBY.session_target[0], sep=" ")
print(f"total TRs: {len(labels)}")
print(f"conditions: {sorted(labels['labels'].unique())}")
print(f"runs: {sorted(labels['chunks'].unique())}")
```

Filter to the face & house TRs and build a binary label vector (1 = face, 0 = house). Boolean-indexing a `BrainData` slices the timeseries the same way it slices a numpy array:

```{code-cell} python
keep = labels["labels"].isin(["face", "house"]).to_numpy()
y = (labels.loc[keep, "labels"] == "face").astype(int).to_numpy()
trials = brain[keep]
print(f"trials.shape: {trials.shape}  (n_trials, n_voxels)")
print(f"class balance: {np.bincount(y)}  (house, face)")
```

```{code-cell} python
trials[0].plot(method="slices", title="Example face trial (one TR)")
```

## Whole-brain decoding

`BrainData.predict()` mirrors `.fit()`: one call, one frozen result dataclass. The dispatch is set by `method=`, the algorithm by `model=`. For whole-brain MVPA with a linear SVM:

```{code-cell} python
result_wb = trials.predict(
    y=y,
    spatial_scale="whole_brain",
    model="svm",
    cv=5,
)
print("Populated fields:", result_wb.available())
```

`result_wb` is a `Predict` dataclass — frozen, every field defaults to `None`, only the fields relevant to this call are populated. `available()` lists them so you don't have to guess what dispatch path filled which slot.

### How well? — per-fold scores

`scores` is the score per fold; `mean_score` and `std_score` are the cross-fold summary. For a binary classifier with `scoring='auto'`, "score" resolves to **accuracy**:

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3))
folds = list(range(1, len(result_wb.scores) + 1))
ax.bar(folds, result_wb.scores, color="C0")
ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="chance")
ax.set_ylim(0, 1)
ax.set_xlabel("CV fold")
ax.set_ylabel("accuracy")
ax.set_title(
    f"Whole-brain SVM: face vs house\n"
    f"mean = {result_wb.mean_score:.3f} ± {result_wb.std_score:.3f}"
)
ax.legend(loc="lower right")
fig
```

### Where does the signal live? — the weight map

`weight_map` is the publishable map: a `BrainData` of `.coef_` from a single classifier fit on **all** the data, after CV scored its generalization. Cross-validation tells you how well the model generalizes; the all-data fit gives you the model itself, with nothing held out. Same mask as `trials`, so `.plot()` just works:

```{code-cell} python
result_wb.weight_map.plot(
    method="slices",
    title="SVM weight map: + favors face, − favors house",
    cmap="RdBu_r",
    colorbar=True,
)
```

> **Interpretation caveat.** Raw classifier weights are *not* statistical maps. A voxel with a large positive weight is one the classifier *uses* to push toward "face" — but a voxel with weight ≈ 0 might still carry information that other voxels already supply. For a principled "activation pattern" view, see Haufe et al. 2014 (the `weight × cov(X)` transformation).

### How stable is the map across folds?

The all-data fit is the publishable estimate, but it doesn't tell you *whether the same voxels would have been weighted similarly* on a different subset of trials. `fold_weight_maps` is the stack of per-fold `.coef_` vectors — a `BrainData` of shape `(n_folds, n_voxels)` — and the across-fold standard deviation is a quick stability proxy: low values mean the voxel's contribution is robust across CV splits.

```{code-cell} python
fold_std = result_wb.fold_weight_maps.data.std(axis=0)
stability = BrainData(fold_std, mask=trials.mask)
stability.plot(
    method="slices",
    title="Across-fold std of weights (lower = more stable)",
    cmap="viridis",
    colorbar=True,
)
```

### Applying the trained model to new data

`estimator` is the fitted sklearn object — same one whose coefficients are in `weight_map`. Use it directly with `.predict()` on a new design matrix:

```{code-cell} python
print(type(result_wb.estimator).__name__)
# new_predictions = result_wb.estimator.predict(new_trials.data)
```

## ROI decoding with the bundled k50 atlas

The whole-brain map answers *"where do voxels contribute when one classifier sees them all together?"*. ROI decoding answers a different question: **which regions, on their own, carry enough information to discriminate?** That's a more honest "where" — each region competes only against chance, not against its neighbours.

nltools ships parcellation atlases at every supported template/resolution. Our data is in 3mm default-template space (the auto-resampled output above), so we'll grab the matching k50:

```{code-cell} python
from nltools.templates import fetch_resource

atlas_path = fetch_resource("masks/default/3mm-MNI152-2009fsl-k50.nii.gz")
atlas = BrainData(atlas_path)
atlas.plot(method="slices", title="k50 atlas (50-region parcellation)")
```

Pass the atlas to `predict(spatial_scale="roi", roi_mask=...)`. This trains one classifier per parcel and returns a single `accuracy_map` — every voxel inside parcel *i* is filled with parcel *i*'s cross-validated accuracy. Voxels outside any parcel are `NaN`:

```{code-cell} python
result_roi = trials.predict(
    y=y,
    spatial_scale="roi",
    roi_mask=atlas_path,
    model="svm",
    cv=5,
    n_jobs=4,
)
print("Populated fields:", result_roi.available())
print(f"scores shape    : {result_roi.scores.shape}  (n_folds, n_rois)")
print(f"mean_score shape: {result_roi.mean_score.shape}  (one accuracy per parcel)")
```

On `spatial_scale="roi"` the score fields are repurposed for the per-parcel layout: `scores` is `(n_folds, n_rois)`, `mean_score` and `std_score` are `(n_rois,)`, and `roi_labels` carries the atlas integer IDs in the same order. `accuracy_map` is a `BrainData` brain-space view of `mean_score` — every voxel inside a parcel shows that parcel's mean accuracy.

ROI dispatch also produces `weight_map`, `fold_weight_maps`, and a per-parcel `estimator` dict. Because the atlas is a label image (each voxel belongs to exactly one parcel), per-parcel `coef_` vectors slot back into voxel space disjointly — same shape as the whole-brain map, just composed of independently-trained pieces.

### Per-region accuracy as a brain map

```{code-cell} python
result_roi.accuracy_map.plot(
    method="slices",
    title="ROI decoding accuracy (chance = 0.5)",
    cmap="RdBu_r",
    vmin=0.3,
    vmax=0.7,
    colorbar=True,
)
```

Regions tinted red beat chance for face/house; blue underperform chance (in a small dataset, that's almost entirely noise). Ventro-temporal cortex should light up clearly — that's where face- and place-selective patches live.

### Top-N regions by accuracy

`mean_score` plus `roi_labels` makes ranking trivial — sort the indices, then index both arrays in lockstep:

```{code-cell} python
top_n = 15
order = np.argsort(result_roi.mean_score)[::-1][:top_n]
top_acc = result_roi.mean_score[order]
top_labels = result_roi.roi_labels[order]

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(range(top_n), top_acc[::-1], color="C3")
ax.axvline(0.5, color="k", linestyle="--", linewidth=1, label="chance")
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"parcel {label}" for label in top_labels][::-1])
ax.set_xlabel("accuracy")
ax.set_title(f"Top {top_n} parcels (out of {len(result_roi.roi_labels)})")
ax.legend(loc="lower right")
fig
```

### ROI weight map

`weight_map` is the same shape as the whole-brain version, but each voxel's value comes from its parcel's all-data fit. **Magnitudes are not directly comparable across parcels** — different parcels see different voxel sets, so their `coef_` distributions live on different scales. Within-parcel ranking is meaningful, and the brain-space view is right for visualizing where each region's classifier is putting its weight:

```{code-cell} python
result_roi.weight_map.plot(
    method="slices",
    title="ROI weight map: per-parcel coefs (+ favors face, − favors house)",
    cmap="RdBu_r",
    colorbar=True,
)
```

Per-parcel estimators are kept in `result_roi.estimator` (a dict keyed by atlas label) — apply any one of them to new data via `result_roi.estimator[label].predict(...)`.

## Plug in any sklearn estimator

String shortcuts (`'svm'`, `'logistic'`, `'lda'`, `'ridge_classifier'`, etc.) cover the common cases. For anything custom — feature selection, custom preprocessing, hyperparameter search — pass any sklearn estimator or `Pipeline` directly to `model=`. When `predict()` detects a `Pipeline`, it automatically flips `standardize=False` so it doesn't wrap a second `StandardScaler` around your pipeline (a one-shot warning prints to confirm):

```{code-cell} python
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

pipe = make_pipeline(
    StandardScaler(),
    SelectKBest(k=500),
    LinearSVC(max_iter=5000),
)
result_pipe = trials.predict(y=y, spatial_scale="whole_brain", cv=5, model=pipe)
print(f"500-voxel pipeline: mean accuracy = {result_pipe.mean_score:.3f}")
print(f"Populated fields  : {result_pipe.available()}")
```

`weight_map` is *not* populated here — `predict()` only back-projects coefficients from linear models with a direct voxel-to-coefficient correspondence. A `SelectKBest` feature mask plus a downstream classifier breaks that link, so the field stays `None`. The cross-validated accuracy is still the right thing to read.

## Putting it all together

| Goal | Call | Result fields |
|---|---|---|
| Whole-brain accuracy + weight map | `bd.predict(y=y, spatial_scale="whole_brain", model="svm", cv=K)` | `predictions`, `scores` (n_folds,), scalar `mean_score`/`std_score`, `weight_map` (all-data fit, BrainData), `fold_weight_maps` (BrainData stack for stability), `estimator` (fitted sklearn) |
| Per-region accuracy + per-parcel weights | `bd.predict(y=y, spatial_scale="roi", roi_mask=atlas, model="svm", cv=K)` | `scores` (n_folds, n_rois), `mean_score`/`std_score` (n_rois,), `roi_labels`, `accuracy_map` / `weight_map` / `fold_weight_maps` (BrainData), `estimator` (dict keyed by label) |
| Custom preprocessing | `bd.predict(y=y, model=Pipeline(...), ...)` | `standardize` auto-flipped to `False`; CV fields populated, weight_map None when feature selection breaks back-projection |
| Result attached to `bd` | add `inplace=True` | fields become `bd.predict_*` attributes |

> **Whole-brain vs ROI — what each tells you**
>
> A whole-brain weight map is the *joint* contribution of every voxel within one model. It reflects the geometry of the data the classifier saw, not which voxels are independently predictive. ROI accuracy answers the cleaner "is this region informative on its own?" question. Use both when you can — agreement across the two views is much more convincing than either alone.

## Next Steps

- **Searchlight**: roving sphere instead of fixed parcels — `bd.predict(spatial_scale="searchlight", radius_mm=8.0)`.
- **Multi-subject decoding** with leave-one-subject-out CV — see [Multi-Subject Decoding](08_multi_subject_decoding.md).
- **Encoding models**: the inverse view — predict voxel responses from features rather than features from voxels — see [Encoding](02_encoding.md).
- **RSA**: characterize the *structure* of representations, not just their separability — see [RSA](04_rsa.md).
