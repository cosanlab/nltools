import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Voxelwise Encoding Models

    ## Introduction

    A **voxelwise encoding model** (VEM) flips the GLM question on its head:

    - **GLM** asks *which voxels respond significantly* to a small set of categorical regressors. Output: β / t / p maps.
    - **VEM** asks *how much variance in each voxel can a feature space explain on held-out data*. Output: a per-voxel cross-validated R² map.

    Practically that means: many features (often hundreds), ridge regularization (so OLS doesn't melt), per-voxel α selection (so each voxel picks the regularization that works for its noise level), and an honest train/test split.

    In this tutorial we'll use the classic **Miyawaki 2008** visual reconstruction dataset — one subject viewing 10×10 binary contrast figures while we record from visual cortex. The stimulus is naturally 100-dimensional, so VEM is the right tool.

    > **Heads up — native space.** Miyawaki ships in *subject native space* (anisotropic 3.3 × 3.6 × 6.4 mm voxels), not MNI. We'll do the whole tutorial without resampling, using slice plots with the bundled subject anatomical as the background. Glass-brain / flatmap views require MNI-aligned data and will refuse to render here — that's intentional, and the error messages will point you at the right view.
    """)
    return


@app.cell
def _():
    from nltools.data import BrainData

    return (BrainData,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the data

    The dataset bundles 32 runs of BOLD data, a visual-cortex mask, the subject's anatomical (`d.background`) on the **same native grid as the funcs**, and per-frame stimulus labels. We pass `mask=d.mask` to `BrainData` so it skips its usual MNI auto-resample and keeps the native affine.
    """)
    return


@app.cell
def _():
    from nilearn.datasets import fetch_miyawaki2008

    DATASET = fetch_miyawaki2008(verbose=0)
    print(f"runs: {len(DATASET.func)}")
    print(f"label files: {len(DATASET.label)}")
    print(f"native voxel size: {(3.3, 3.6, 6.4)} mm  (anisotropic)")
    return (DATASET,)


@app.cell
def _(BrainData, DATASET):
    brain = BrainData(DATASET.func[0], mask=DATASET.mask)
    print(f"brain.shape: {brain.shape}  (timepoints, voxels)")
    print(f"mask zooms: {brain.mask.header.get_zooms()}")
    brain
    return (brain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sanity check: visualize one BOLD volume

    On native data we use slice plots with the bundled `d.background` as the underlay. This works because the stat map and the background share an affine — no resampling, no MNI assumption.
    """)
    return


@app.cell
def _(DATASET, brain):
    brain[0].plot(
        method="slices",
        view="z",
        cut_coords=[[-12, -6, 0, 6, 12]],
        bg_img=DATASET.background,
        title="sub-01 run-01 frame 0",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What if we try a glass-brain plot?

    Glass brain draws an MNI-shape outline. With native-space data, the stat map and the outline don't share a coordinate system, so any rendering would be visually misleading. nltools refuses outright and points you at the right view:
    """)
    return


@app.cell
def _(brain):
    try:
        brain[0].plot(method="glass")
    except ValueError as e:
        print(f"ValueError: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stack runs into one design

    For VEM we want a long timeseries (more samples → better-conditioned ridge). We'll concatenate the first **8 runs** into one `BrainData` (1040 TRs total) and stack the matching 100-D stimulus per timepoint from the bundled label CSVs. Rest frames are coded `-1`; we map them to `0` (no-patch).
    """)
    return


@app.cell
def _(BrainData, DATASET, np):
    from nltools.utils import concatenate

    N_RUNS = 8
    runs = [BrainData(DATASET.func[i], mask=DATASET.mask) for i in range(N_RUNS)]
    bold = concatenate(runs)
    stim = np.vstack(
        [np.loadtxt(DATASET.label[i], delimiter=",", dtype=int) for i in range(N_RUNS)]
    ).astype(float)
    stim[stim < 0] = 0.0  # rest frames → no-patch
    print(f"bold.shape: {bold.shape}  (TRs, voxels)")
    print(f"stim.shape: {stim.shape}  (TRs, 10x10 patches)")
    print(f"frames with any patch on: {(stim.sum(axis=1) > 0).sum()} / {stim.shape[0]}")
    return bold, stim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The stimulus on a single on-frame is a 10×10 binary contrast figure. Plot a few to ground intuition:
    """)
    return


@app.cell
def _(plt, stim):
    on_frames = (stim.sum(axis=1) > 0).nonzero()[0]
    _fig, _axes = plt.subplots(1, 4, figsize=(10, 3))
    for _ax, _idx in zip(_axes, on_frames[:4]):
        _ax.imshow(stim[_idx].reshape(10, 10), cmap="gray", vmin=0, vmax=1)
        _ax.set_title(f"TR {_idx}")
        _ax.axis("off")
    _fig.suptitle("Stimulus frames (10×10 binary contrast)", y=1.02)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the feature matrix: FIR lag bank

    In `01_glm.py` we *assumed* a canonical HRF and `convolve()`'d each regressor with it. The encoding-model way is more flexible: feed the model **lagged copies of the stimulus** at several delays and let ridge learn the per-voxel impulse response from the data. This is the **FIR (finite impulse response)** trick.

    Concretely: for each of the 100 patches, we create copies at lags 1, 2, 3 TRs. The model sees `100 × 3 = 300` features per timepoint. The recovered weight pattern across lags *is* an estimated HRF for that voxel, no canonical-shape assumption baked in.

    Why this motivates ridge: we just went from 100 to 300 features for 1040 samples. OLS would happily overfit. Ridge keeps things honest.
    """)
    return


@app.cell
def _(np, stim):
    def lag_features(X, lags):
        """Stack lagged copies of X horizontally; pad early TRs with zeros."""
        T, P = X.shape
        out = np.zeros((T, P * len(lags)), dtype=X.dtype)
        for j, k in enumerate(lags):
            out[k:, j * P : (j + 1) * P] = X[: T - k]
        return out

    LAGS = [1, 2, 3]
    X_fir = lag_features(stim, LAGS)
    print(
        f"X_fir.shape: {X_fir.shape}  ({len(LAGS)} lags × 100 patches = 300 features)"
    )
    return (X_fir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A first ridge fit (fixed α, in-sample R²)

    Two pieces of standard VEM preprocessing first:

    1. **Demean BOLD per voxel** — raw BOLD has a large positive offset; ridge with `fit_intercept=False` (the default in nltools' SVD path) needs zero-centered targets.
    2. **Skip `bd.fit`'s grand-mean PSC scaling** (`scale=False`) — that's a GLM convention, not a VEM one.

    Then the simplest call: fixed regularization, no cross-validation. The resulting `ridge_scores` map is **in-sample R²**, optimistically biased — we'll fix that with CV next. `bd.fit(model="ridge", ...)` populates three attributes on `bold`:

    - `bold.ridge_weights` — `(n_features, n_voxels)` β map
    - `bold.ridge_fitted_values` — `(n_TRs, n_voxels)` reconstructed BOLD
    - `bold.ridge_scores` — `(1, n_voxels)` per-voxel R²
    """)
    return


@app.cell
def _(X_fir, bold, np):
    bold.data = bold.data - bold.data.mean(axis=0, keepdims=True)  # voxelwise demean
    bold.fit(model="ridge", X=X_fir, alpha=1.0, scale=False)
    _r2 = bold.ridge_scores.data.ravel()
    print(f"ridge_weights: {bold.ridge_weights.shape}")
    print(f"ridge_scores : {bold.ridge_scores.shape}")
    print(
        f"in-sample R² — mean: {_r2.mean():.3f}  max: {_r2.max():.3f}  "
        f"95th pct: {np.percentile(_r2, 95):.3f}"
    )
    return


@app.cell
def _(DATASET, bold):
    bold.ridge_scores.plot(
        method="slices",
        view="z",
        cut_coords=[[-12, -6, 0, 6, 12]],
        bg_img=DATASET.background,
        title="Ridge in-sample R² (α=1.0, FIR lags 1–3)",
        cmap="hot",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-validated, per-voxel α — the VEM workhorse

    The in-sample map is optimistic. The honest version: `alpha="auto"` plus `cv=K` triggers nltools' `solve_ridge_cv`, which sweeps a grid of α values and **picks the best α independently for each voxel** (controlled by `local_alpha=True`, the default).

    This is what makes nltools' Ridge a VEM tool rather than a sklearn `RidgeCV`: V1 voxels with high SNR want little regularization, IFG voxels with low SNR want a lot, and you don't have to commit to one number.

    The `cv_results_` dict carries:

    - `mean_score` — `(n_voxels,)` honest cross-validated R² per voxel
    - `scores` — `(n_folds, n_voxels)` per-fold R² (for variability)
    - `best_alpha` — `(n_voxels,)` selected α per voxel
    - `predictions` — held-out predicted BOLD as a `BrainData`
    """)
    return


@app.cell
def _(X_fir, bold, np):
    from sklearn.model_selection import KFold

    ALPHAS = np.logspace(-1, 4, 20)
    # Shuffled K-fold so each fold sees stimulus from across the whole
    # session. With only 8 runs, leave-one-run-out under-uses the data
    # for α selection — covered as an aside below.
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    bold.fit(
        model="ridge",
        X=X_fir,
        alpha="auto",
        alphas=ALPHAS,
        cv=cv,
        scale=False,
    )
    _cv = bold.cv_results_["mean_score"]
    print(f"alpha grid: {ALPHAS.min():.2g} … {ALPHAS.max():.2g} ({len(ALPHAS)} values)")
    print(
        f"CV R² (mean across folds) — mean: {_cv.mean():.3f}  "
        f"max: {_cv.max():.3f}  95th pct: {np.percentile(_cv, 95):.3f}"
    )
    print(f"voxels with CV R² > 0.10: {(_cv > 0.10).sum()} / {_cv.size}")
    return (ALPHAS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Honest R² map

    Notice the values are smaller than the in-sample fit (as expected — held-out is harder), and the spatial pattern is more concentrated in the parts of visual cortex that actually track this stimulus.
    """)
    return


@app.cell
def _(DATASET, bold):
    cv_score_map = bold.ridge_scores.copy()
    cv_score_map.data = bold.cv_results_["mean_score"].reshape(1, -1)
    cv_score_map.plot(
        method="slices",
        view="z",
        cut_coords=[[-12, -6, 0, 6, 12]],
        bg_img=DATASET.background,
        title="Ridge CV R² (α auto, per-voxel, FIR lags 1–3)",
        cmap="hot",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per-voxel α distribution

    The whole point of `local_alpha=True` is that V1 vs higher-order voxels can land on very different α values. A histogram of the selected α (in log space) shows that spread directly:
    """)
    return


@app.cell
def _(ALPHAS, bold, np, plt):
    best_alpha = bold.cv_results_["best_alpha"]
    if np.isscalar(best_alpha):
        best_alpha = np.full(bold.ridge_scores.size, best_alpha)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.hist(
        np.log10(np.asarray(best_alpha).ravel()), bins=len(ALPHAS), color="steelblue"
    )
    _ax.set_xlabel(r"$\log_{10}(\alpha)$ selected per voxel")
    _ax.set_ylabel("voxel count")
    _ax.set_title("Per-voxel ridge α — voxels disagree on regularization")
    _fig
    return


if __name__ == "__main__":
    app.run()
