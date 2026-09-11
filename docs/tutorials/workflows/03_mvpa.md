---
# AUTO-GENERATED from docs/tutorials/workflows/03_mvpa.py by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe docs-generate`.
kernelspec:
  name: python3
  display_name: Python 3
edit_url: https://github.com/cosanlab/nltools/edit/master/docs/tutorials/workflows/03_mvpa.py
source_url: https://github.com/cosanlab/nltools/blob/master/docs/tutorials/workflows/03_mvpa.py
downloads:
  - url: https://molab.marimo.io/github/cosanlab/nltools/blob/master/docs/tutorials/workflows/03_mvpa.py
    title: Open in molab
  - url: https://github.com/cosanlab/nltools/blob/master/docs/tutorials/workflows/03_mvpa.py
    title: Source notebook (03_mvpa.py)
---

# Multivariate Pattern Analysis

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/cosanlab/nltools/blob/master/docs/tutorials/workflows/03_mvpa.py)

````{tip} Run this tutorial
This page is rendered from the [marimo](https://marimo.io) notebook [`docs/tutorials/workflows/03_mvpa.py`](https://github.com/cosanlab/nltools/blob/master/docs/tutorials/workflows/03_mvpa.py). Click the badge to run it in the cloud (free, no install), or locally: download `03_mvpa.py` and run `uvx marimo edit --sandbox 03_mvpa.py`. Outputs below were baked in at build time.
````

**What it answers.** Does the *distributed pattern* of activity across many voxels carry information about the conditions — beyond what any single voxel shows? Two complementary approaches:

- **Decoding** — can a classifier *predict* the condition from the pattern? (cross-validated accuracy)
- **RSA** — what is the *geometry* of the patterns relative to one another, and does it match a hypothesis? (a representational dissimilarity matrix compared to a model)

Both run at three spatial scales via `spatial_scale=` — `'whole_brain'`, `'roi'`, `'searchlight'` — and either within a subject (here) or across subjects (loop + group test). For the theory, see [dartbrains](https://dartbrains.org).
<!---->
**How it works.** Both approaches operate on the same patterns; they differ in the question. Decoding fits a classifier across voxels and scores it on held-out data. RSA turns patterns into a distance matrix (the RDM) and correlates that geometry with a model RDM. The `spatial_scale=` switch is shared: whole-brain uses every voxel jointly, ROI runs the analysis per parcel, and searchlight sweeps a roving sphere.

We use the classic **Haxby** dataset — one subject viewing 8 object categories — for both.

```{code-cell} python3
import numpy as np
import pandas as pd
from joblib import Memory

from nltools.data import Adjacency, BrainData
from nltools.templates import fetch_resource

memory = Memory(".cache/tutorials", verbose=0)
```

## Decoding

Haxby ships in **subject space** (no MNI normalization, anisotropic 3.5 × 3.75 × 3.75 mm voxels), so we load it with the dataset's own brain mask to stay on its native grid, and plot on the subject's anatomical via `bg_img=`. Then restrict to **face vs. house** — the strongest, best-understood contrast. Boolean-indexing a `BrainData` slices its timeseries like a numpy array.

```{code-cell} python3
from nilearn.datasets import fetch_haxby

HAXBY = fetch_haxby(subjects=[2], verbose=0)

LABELS = pd.read_csv(HAXBY.session_target[0], sep=r"\s+")["labels"].to_numpy()

brain = BrainData(HAXBY.func[0], mask=HAXBY.mask)
keep = np.isin(LABELS, ["face", "house"])
trials = brain[keep]
y = (LABELS[keep] == "face").astype(int)
print(
    f"trials: {trials.shape}  (n_trials, n_voxels)   classes (house, face): {np.bincount(y)}"
)
```

### Whole-brain

`BrainData.predict()` mirrors `.fit()`: one call, one frozen `Predict` result. `cv` scores generalization; `weight_map` is the classifier refit on all the data (the publishable map). For a linear SVM:

```{code-cell} python3
decode_wb = trials.predict(
    y=y, spatial_scale="whole_brain", estimator="linear_svc", cv=5
)
print(
    f"whole-brain accuracy: {decode_wb.mean_score:.3f} ± {decode_wb.std_score:.3f}  (chance 0.5)"
)
decode_wb.weight_map.plot(
    method="slices",
    bg_img=HAXBY.anat[0],
    title="SVM weights: + favors face, − favors house",
    cmap="RdBu_r",
    colorbar=True,
)
```

> Raw classifier weights are *not* a statistical map — a near-zero weight can still carry information other voxels already supply (see Haufe et al., 2014). For a cleaner "where", decode per region.

### ROI

`spatial_scale="roi"` with a parcellation trains one classifier per parcel and returns a `score_map` — every voxel in parcel *i* filled with parcel *i*'s cross-validated accuracy. We use the bundled k50 atlas. It is defined in MNI space, and `roi_mask=` resamples it onto this subject's grid by header affine alone — a grid change, not a spatial normalization — so its parcel boundaries are only approximate for this un-normalized subject.

```{code-cell} python3
atlas_path = fetch_resource("masks/default/3mm-MNI152-2009fsl-k50.nii.gz")
decode_roi = trials.predict(
    y=y,
    spatial_scale="roi",
    roi_mask=atlas_path,
    estimator="linear_svc",
    cv=5,
    n_jobs=4,
)
print(
    f"per-parcel accuracy: {decode_roi.mean_score.shape[0]} parcels, best = {decode_roi.mean_score.max():.3f}"
)
decode_roi.score_map.plot(
    method="slices",
    bg_img=HAXBY.anat[0],
    title="ROI decoding accuracy (chance 0.5)",
    cmap="RdBu_r",
    vmin=0.3,
    vmax=0.7,
    colorbar=True,
)
```

Ventro-temporal cortex lights up — where face- and place-selective patches live. ROI accuracy answers "is this region informative *on its own*?", a cleaner "where" than the joint whole-brain weights.

### Searchlight

A roving sphere: one classifier per voxel-neighborhood, giving a per-voxel accuracy map. Same call, `spatial_scale="searchlight"`. This fits thousands of classifiers, so it's the slow one — we cache it.

```{code-cell} python3
@memory.cache
def searchlight_decode(radius):
    return trials.predict(
        y=y,
        spatial_scale="searchlight",
        radius=radius,
        estimator="linear_svc",
        cv=5,
        n_jobs=-1,
    )

decode_sl = searchlight_decode(8.0)
decode_sl.score_map.plot(
    method="slices",
    bg_img=HAXBY.anat[0],
    title="Searchlight decoding accuracy (8 mm sphere)",
    cmap="hot",
    colorbar=True,
)
```

## RSA

Decoding asks *can we separate* the conditions; RSA asks *what is their geometry*. Build one pattern per category (mean BOLD across that category's TRs, shifted 2 TRs for the hemodynamic lag), then turn the patterns into a representational dissimilarity matrix (RDM).

```{code-cell} python3
conditions = [c for c in sorted(set(LABELS)) if c != "rest"]
shifted = np.roll(LABELS, 2)  # align BOLD to stimulus (~5s HRF lag)
patterns = np.vstack([brain.data[shifted == c].mean(axis=0) for c in conditions])
category_patterns = BrainData(patterns, mask=brain.mask)
print(
    f"category patterns: {category_patterns.shape}  ({len(conditions)} categories)"
)
```

```{code-cell} python3
rdm = category_patterns.distance(metric="correlation")
rdm.labels = conditions
rdm.plot(cmap="RdBu_r")
```

The RDM is the full geometry — every pairwise dissimilarity at once. To test a hypothesis, build a model RDM (here: animate `face`/`cat` vs. the rest) and correlate the two with a Mantel permutation test.

```{code-cell} python3
animate = np.array([c in ("face", "cat") for c in conditions])
model_rdm = Adjacency(
    (animate[:, None] != animate[None, :]).astype(float),
    matrix_type="distance",
    labels=conditions,
)
rsa_wb = rdm.similarity(
    model_rdm, metric="spearman", n_permute=1000, random_state=0
)
print(
    f"whole-brain RSA (animacy): rho = {rsa_wb['correlation']:.3f}  p = {rsa_wb['p']:.3f}"
)
```

To examine animacy structure by region, `spatial_scale="roi"` computes one
RDM per parcel. `roi_to_brain_from_atlas` then paints each parcel's correlation with
the model back into brain space. Align the atlas first and retain the sorted
nonzero labels inside the source mask. These labels give the returned RDM order:

```{code-cell} python3
atlas_path_rsa = fetch_resource("masks/default/3mm-MNI152-2009fsl-k50.nii.gz")
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask
from nltools.mask import roi_to_brain_from_atlas

rsa_atlas = resample_to_img(
    atlas_path_rsa, category_patterns.mask, interpolation="nearest",
    force_resample=True, copy_header=True,
)
rsa_labels = np.unique(apply_mask(rsa_atlas, category_patterns.mask).astype(int))
rsa_labels = rsa_labels[rsa_labels != 0]
roi_rdms = category_patterns.distance(
    metric="correlation", spatial_scale="roi", roi_mask=rsa_atlas
)
roi_scores = roi_rdms.similarity(model_rdm, metric="spearman", method=None)
rsa_map = roi_to_brain_from_atlas(
    np.array([score["correlation"] for score in roi_scores]),
    atlas=rsa_atlas, source_mask=category_patterns.mask, roi_labels=rsa_labels,
)
rsa_map.plot(
    method="slices",
    bg_img=HAXBY.anat[0],
    title="ROI RSA: where category geometry matches animacy",
    cmap="RdBu_r",
    colorbar=True,
)
```

## Recap

Both approaches share the `spatial_scale=` axis, and both extend across subjects (loop per subject, then a group test on accuracies or projected RSA maps).

| | Decoding | RSA |
|---|---|---|
| Question | Can we predict the condition? | What's the representational geometry? |
| Whole-brain | `bd.predict(y=, spatial_scale="whole_brain")` | `bd.distance(metric="correlation")` → `.similarity(model)` |
| ROI | `bd.predict(y=, spatial_scale="roi", roi_mask=)` | `bd.distance(..., spatial_scale="roi", roi_mask=)` → `.similarity(model)` → `roi_to_brain_from_atlas(...)` |
| Searchlight | `bd.predict(y=, spatial_scale="searchlight", radius=)` | `bd.distance(..., spatial_scale="searchlight", radius=)` |
| Custom model | pass any sklearn estimator to `estimator=` | any `metric=` (`spearman`/`pearson`) |

**Next steps**

- [GLM analysis](01_glm.md) — the mass-univariate "where", and how to build single-trial designs.
- [Encoding models](02_encoding.md) — predict brain activity from stimulus features.
- [Inter-subject correlation](04_isc.md) — shared responses across people.
