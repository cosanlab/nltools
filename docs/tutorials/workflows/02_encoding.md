---
# AUTO-GENERATED from docs/tutorials/workflows/02_encoding.py by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe docs-generate`.
kernelspec:
  name: python3
  display_name: Python 3
edit_url: https://github.com/cosanlab/nltools/edit/master/docs/tutorials/workflows/02_encoding.py
source_url: https://github.com/cosanlab/nltools/blob/master/docs/tutorials/workflows/02_encoding.py
downloads:
  - url: https://molab.marimo.io/github/cosanlab/nltools/blob/master/docs/tutorials/workflows/02_encoding.py
    title: Open in molab
  - url: https://github.com/cosanlab/nltools/blob/master/docs/tutorials/workflows/02_encoding.py
    title: Source notebook (02_encoding.py)
---

# Encoding Models

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/cosanlab/nltools/blob/master/docs/tutorials/workflows/02_encoding.py)

````{tip} Run this tutorial
This page is rendered from the [marimo](https://marimo.io) notebook [`docs/tutorials/workflows/02_encoding.py`](https://github.com/cosanlab/nltools/blob/master/docs/tutorials/workflows/02_encoding.py). Click the badge to run it in the cloud (free, no install), or locally: download `02_encoding.py` and run `uvx marimo edit --sandbox 02_encoding.py`. Outputs below were baked in at build time.
````

**What it answers.** How much of each voxel's response can a stimulus feature space explain — on *held-out* data? An encoding model is the inverse of decoding: instead of predicting the stimulus from the brain, you predict the brain from features of the stimulus, and score each voxel by its cross-validated R².

For the theory, see the encoding-model material in [naturalistic-data](https://naturalistic-data.org). This tutorial is about *running* one in nltools.
<!---->
**How it works.** Compared with the [GLM](01_glm.md), an encoding model flips the question and the machinery:

- **GLM** assumes a canonical HRF, uses a few categorical regressors, and asks *which voxels respond* (β / t / p).
- **Encoding** uses many features (often hundreds), lets the data estimate the response shape, and asks *how well features predict each voxel* (R² on data the model never saw).

Two ideas make it work: a **FIR (finite impulse response)** feature bank — lagged copies of the stimulus, so ridge learns the per-voxel HRF instead of assuming one — and **ridge regularization with per-voxel α**, since hundreds of features would make ordinary least squares overfit. We compare an optimistic in-sample fit against an honest held-out one.

```{code-cell} python3
import numpy as np
from joblib import Memory

from nltools.data import BrainData
from nltools.utils import concatenate

# Memoize the (slow) multi-run load to disk (.cache/ is git-ignored).
memory = Memory(".cache/tutorials", verbose=0)
```

## How to do it

We use the classic **Miyawaki 2008** dataset — one subject viewing 10×10 binary contrast figures while we record from visual cortex. The stimulus is naturally 100-dimensional, so encoding is the right tool. The data ships in *subject-native space* (anisotropic voxels), so we pass `mask=` to skip MNI resampling and plot on the bundled anatomical with slice views (glass-brain/MNI views would misalign).

```{code-cell} python3
from nilearn.datasets import fetch_miyawaki2008

DATASET = fetch_miyawaki2008(verbose=0)

@memory.cache
def load_runs(n_runs: int):
    """Concatenate the first n_runs of BOLD and stack the matching stimulus."""
    runs = [BrainData(DATASET.func[i], mask=DATASET.mask) for i in range(n_runs)]
    bold = concatenate(runs)
    stim = np.vstack(
        [
            np.loadtxt(DATASET.label[i], delimiter=",", dtype=int)
            for i in range(n_runs)
        ]
    ).astype(float)
    stim[stim < 0] = 0.0  # rest frames → no-patch
    return bold, stim

bold, stim = load_runs(8)
print(f"bold: {bold.shape}  (TRs, voxels)   stim: {stim.shape}  (TRs, 100 patches)")
```

Each on-frame is a 10×10 binary contrast figure — these 100 patches are our features:

```{code-cell} python3
import matplotlib.pyplot as plt

# Each stimulus is held on screen for several TRs, so show the first TR of
# four *different* stimuli rather than four copies of the same one.
on_frames = (stim.sum(axis=1) > 0).nonzero()[0]
first_seen = sorted(np.unique(stim[on_frames], axis=0, return_index=True)[1])
stim_fig, stim_axes = plt.subplots(1, 4, figsize=(9, 2.6))
for stim_ax, frame in zip(stim_axes, on_frames[first_seen[:4]]):
    stim_ax.imshow(stim[frame].reshape(10, 10), cmap="gray", vmin=0, vmax=1)
    stim_ax.set_title(f"TR {frame}")
    stim_ax.axis("off")
_ = stim_fig.suptitle("Stimulus frames (10×10 binary contrast)", y=1.05)
```

### Build the feature matrix (FIR lag bank)

For each of the 100 patches we add copies at lags of 1, 2, and 3 TRs, giving `100 × 3 = 300` features. The weights ridge learns across those lags *are* an estimated impulse response per voxel — no canonical HRF assumed. Going from 100 to 300 features for ~1000 samples is exactly why we need ridge.

```{code-cell} python3
def lag_features(X, lags):
    """Stack lagged copies of X horizontally; pad early TRs with zeros."""
    n_tr, n_patch = X.shape
    out = np.zeros((n_tr, n_patch * len(lags)))
    for j, lag in enumerate(lags):
        out[lag:, j * n_patch : (j + 1) * n_patch] = X[: n_tr - lag]
    return out

X_fir = lag_features(stim, [1, 2, 3])
print(f"X_fir: {X_fir.shape}  (3 lags × 100 patches = 300 features)")
```

### Fit ridge: in-sample vs. held-out

Standard encoding preprocessing: z-score each voxel explicitly, since `fit` never preprocesses the response and ridge fits no intercept. A fixed-α fit with no CV gives `ridge_r2` — an *in-sample* R², which is optimistically biased.

```{code-cell} python3
# `fit` never preprocesses the response, so standardize explicitly: ridge
# fits no intercept, and a shared alpha should regularize voxels comparably.
# The standardized data gets its own name so every later cell that needs it
# depends on it by name rather than on `bold` having been mutated.
bold_z = bold.standardize(method="zscore")
bold_z.fit(model="ridge", X=X_fir, ridge_alpha=1.0)
in_sample = bold_z.ridge_r2.data.ravel()
print(f"in-sample R²  — mean {in_sample.mean():.3f}  max {in_sample.max():.3f}")
```

The honest version holds out the last run entirely and fits on the other seven. A *sequence* of candidate alphas plus a `ridge_cv` sweeps the grid and picks the best α **per voxel** (`ridge_per_target_alpha=True`, the default) — high-SNR visual voxels want little regularization, noisier voxels want more. Scoring the fitted model on the untouched run gives a genuinely out-of-sample R². Both blocks are standardized on their own statistics — explicitly, since `fit` does no preprocessing — because ridge fits no intercept and a new run carries its own offset.

```{code-cell} python3
from sklearn.model_selection import KFold

ALPHAS = np.logspace(-1, 4, 20)
n_test = bold_z.shape[0] // 8  # last of the 8 concatenated runs
train, test = slice(0, -n_test), slice(-n_test, None)

trained = (
    bold_z[train]
    .standardize(method="zscore")
    .fit(
        model="ridge",
        X=X_fir[train],
        ridge_alpha=ALPHAS,
        ridge_cv=KFold(n_splits=5, shuffle=True, random_state=0),
        inplace=False,
    )
)
held_out = bold_z[test].standardize(method="zscore")
held_out_r2 = trained.model_.score(X_fir[test], held_out.data)
print(
    f"held-out R²  — median {np.median(held_out_r2):.3f}  "
    f"max {held_out_r2.max():.3f}"
)
print(
    f"voxels with held-out R² > 0.10: "
    f"{(held_out_r2 > 0.10).sum()} / {held_out_r2.size}"
)
```

```{code-cell} python3
held_out_map = bold_z.ridge_r2.copy()
# Most voxels do not track the stimulus at all, so their held-out R² is
# negative. Floor the map at zero and let the threshold hide the rest —
# a diverging map here would be a wall of colour with no signal in it.
held_out_map.data = np.clip(held_out_r2, 0, None).reshape(1, -1)
held_out_map.plot(
    method="slices",
    view="z",
    cut_coords=[[-12, -6, 0, 6, 12]],
    bg_img=DATASET.background,
    title="Ridge held-out R² (per-voxel α, FIR lags 1–3; R² > 0.05)",
    cmap="viridis",
    threshold=0.05,
)
```

Held-out R² is smaller than in-sample (a new run is harder) and concentrates in the visual cortex that actually tracks the stimulus. Finally, the spread of selected α confirms why per-voxel regularization matters — voxels disagree:

```{code-cell} python3
best_alpha = np.asarray(trained.model_.alpha_).ravel()
# One bin per grid value, edges at the midpoints between neighbouring alphas.
log_grid = np.log10(ALPHAS)
step = log_grid[1] - log_grid[0]
edges = np.concatenate([log_grid - step / 2, [log_grid[-1] + step / 2]])
alpha_fig, alpha_ax = plt.subplots(figsize=(7, 3))
alpha_ax.hist(np.log10(best_alpha), bins=edges, color="steelblue")
alpha_ax.set_xlabel(r"$\log_{10}(\alpha)$ selected per voxel")
alpha_ax.set_ylabel("voxel count")
_ = alpha_ax.set_title("Per-voxel ridge α — voxels disagree on regularization")
```

## Recap

| Stage | What it does | Key API |
|---|---|---|
| Load runs | Concatenate BOLD + stimulus across runs | `concatenate([...])` |
| Features | FIR lag bank (learn the HRF, don't assume it) | `lag_features(stim, [1, 2, 3])` |
| In-sample fit | Fixed-α ridge → optimistic R² | `bold_z.fit(model="ridge", X=, ridge_alpha=1.0)` |
| Honest fit | Per-voxel α via CV, scored on a held-out run | `bold_z[train].fit(model="ridge", X=, ridge_alpha=ALPHAS, ridge_cv=KFold(5), inplace=False)` |
| Inspect | Held-out R² map + selected α per voxel | `trained.model_.score(X_test, Y_test)`, `trained.model_.alpha_` |

**Next steps**

- [GLM analysis](01_glm.md) — the inferential counterpart: which voxels respond.
- [Multivariate pattern analysis](03_mvpa.md) — decode the stimulus from brain patterns.
- [Inter-subject correlation](04_isc.md) — shared responses to naturalistic stimuli.
