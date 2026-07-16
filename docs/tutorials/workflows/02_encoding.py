# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.0.2",
#     "nibabel",
#     "nilearn==0.13.1",
#     "scikit-learn",
#     "scipy",
#     "pandas",
#     "polars",
#     "seaborn",
#     "matplotlib",
#     "joblib>=1.5.3",
#     "huggingface-hub",
#     "pynv",
#     "pyodide-http; sys_platform == 'emscripten'",
# ]
# ///
import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys

    IN_WASM = sys.platform == "emscripten"
    return (IN_WASM,)


@app.cell(hide_code=True)
async def _(IN_WASM):
    # In-browser only: install the nltools dev wheel before any nltools import runs.
    # The PyPI stack micropip-installs from the PEP 723 header automatically; nltools is
    # not on PyPI at this dev version, so we install the build-hosted wheel by absolute
    # URL. `wasm_ready` is threaded into the nltools-importing cells to force ordering.
    #
    # NOTE: this notebook's dataset (nilearn Miyawaki) is not yet hosted for in-browser
    # use — the data cell below runs locally but not in WASM until a trimmed subset is
    # seeded from HF (tracked follow-up). The kernel + nltools still boot in the browser.
    wasm_ready = True
    if IN_WASM:
        import micropip
        import js

        _ = await micropip.install(
            js.location.origin + "__NLTOOLS_WHEEL_URL__", deps=False
        )
    return (wasm_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Encoding Models

    **What it answers.** How much of each voxel's response can a stimulus feature space explain — on *held-out* data? An encoding model is the inverse of decoding: instead of predicting the stimulus from the brain, you predict the brain from features of the stimulus, and score each voxel by its cross-validated R².

    For the theory, see the encoding-model material in [naturalistic-data](https://naturalistic-data.org). This tutorial is about *running* one in nltools.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **How it works.** Compared with the [GLM](workflows-01_glm.html), an encoding model flips the question and the machinery:

    - **GLM** assumes a canonical HRF, uses a few categorical regressors, and asks *which voxels respond* (β / t / p).
    - **Encoding** uses many features (often hundreds), lets the data estimate the response shape, and asks *how well features predict each voxel* (cross-validated R²).

    Two ideas make it work: a **FIR (finite impulse response)** feature bank — lagged copies of the stimulus, so ridge learns the per-voxel HRF instead of assuming one — and **ridge regularization with per-voxel α**, since hundreds of features would make ordinary least squares overfit. We compare an optimistic in-sample fit against an honest cross-validated one.
    """
    )
    return


@app.cell
def _(wasm_ready):
    _ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
    import numpy as np
    from joblib import Memory

    from nltools.data import BrainData
    from nltools.utils import concatenate

    # Memoize the (slow) multi-run load to disk (.cache/ is git-ignored).
    memory = Memory(".cache/tutorials", verbose=0)
    return BrainData, Memory, concatenate, memory, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## How to do it

    We use the classic **Miyawaki 2008** dataset — one subject viewing 10×10 binary contrast figures while we record from visual cortex. The stimulus is naturally 100-dimensional, so encoding is the right tool. The data ships in *subject-native space* (anisotropic voxels), so we pass `mask=` to skip MNI resampling and plot on the bundled anatomical with slice views (glass-brain/MNI views would misalign).
    """
    )
    return


@app.cell
def _(BrainData, concatenate, memory, np):
    from nilearn.datasets import fetch_miyawaki2008

    DATASET = fetch_miyawaki2008(verbose=0)

    @memory.cache
    def load_runs(n_runs: int):
        """Concatenate the first n_runs of BOLD and stack the matching stimulus."""
        runs = [BrainData(DATASET.func[i], mask=DATASET.mask) for i in range(n_runs)]
        bold = concatenate(runs)
        stim = np.vstack(
            [np.loadtxt(DATASET.label[i], delimiter=",", dtype=int) for i in range(n_runs)]
        ).astype(float)
        stim[stim < 0] = 0.0  # rest frames → no-patch
        return bold, stim

    bold, stim = load_runs(8)
    print(f"bold: {bold.shape}  (TRs, voxels)   stim: {stim.shape}  (TRs, 100 patches)")
    return DATASET, bold, stim


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Each on-frame is a 10×10 binary contrast figure — these 100 patches are our features:
    """
    )
    return


@app.cell
def _(stim):
    import matplotlib.pyplot as plt

    on_frames = (stim.sum(axis=1) > 0).nonzero()[0]
    stim_fig, stim_axes = plt.subplots(1, 4, figsize=(9, 2.6))
    for stim_ax, frame in zip(stim_axes, on_frames[:4]):
        stim_ax.imshow(stim[frame].reshape(10, 10), cmap="gray", vmin=0, vmax=1)
        stim_ax.set_title(f"TR {frame}")
        stim_ax.axis("off")
    stim_fig.suptitle("Stimulus frames (10×10 binary contrast)", y=1.05)
    stim_fig
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Build the feature matrix (FIR lag bank)

    For each of the 100 patches we add copies at lags of 1, 2, and 3 TRs, giving `100 × 3 = 300` features. The weights ridge learns across those lags *are* an estimated impulse response per voxel — no canonical HRF assumed. Going from 100 to 300 features for ~1000 samples is exactly why we need ridge.
    """
    )
    return


@app.cell
def _(np, stim):
    def lag_features(X, lags):
        """Stack lagged copies of X horizontally; pad early TRs with zeros."""
        n_tr, n_patch = X.shape
        out = np.zeros((n_tr, n_patch * len(lags)))
        for j, lag in enumerate(lags):
            out[lag:, j * n_patch : (j + 1) * n_patch] = X[: n_tr - lag]
        return out

    X_fir = lag_features(stim, [1, 2, 3])
    print(f"X_fir: {X_fir.shape}  (3 lags × 100 patches = 300 features)")
    return (X_fir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Fit ridge: in-sample vs. cross-validated

    Standard encoding preprocessing: demean each voxel (ridge here has no intercept) and skip the GLM's percent-signal scaling (`scale=False`). A fixed-α fit with no CV gives `ridge_scores` — an *in-sample* R², which is optimistically biased.
    """
    )
    return


@app.cell
def _(X_fir, bold, np):
    bold.data = bold.data - bold.data.mean(axis=0, keepdims=True)  # voxelwise demean
    bold.fit(model="ridge", X=X_fir, alpha=1.0, scale=False)
    in_sample = bold.ridge_scores.data.ravel()
    print(f"in-sample R²  — mean {in_sample.mean():.3f}  max {in_sample.max():.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The honest version: `alpha="auto"` + `cv=K` sweeps an α grid and picks the best α **per voxel** (`local_alpha=True`, the default) — high-SNR visual voxels want little regularization, noisier voxels want more. `cv_results_` then carries the cross-validated `mean_score`, per-fold `scores`, and selected `best_alpha`.
    """
    )
    return


@app.cell
def _(X_fir, bold, np):
    from sklearn.model_selection import KFold

    ALPHAS = np.logspace(-1, 4, 20)
    bold.fit(
        model="ridge",
        X=X_fir,
        alpha="auto",
        alphas=ALPHAS,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        scale=False,
    )
    cv_r2 = bold.cv_results_["mean_score"]
    print(f"CV R²        — mean {cv_r2.mean():.3f}  max {cv_r2.max():.3f}")
    print(f"voxels with CV R² > 0.10: {(cv_r2 > 0.10).sum()} / {cv_r2.size}")
    return (ALPHAS,)


@app.cell
def _(DATASET, bold):
    cv_map = bold.ridge_scores.copy()
    cv_map.data = bold.cv_results_["mean_score"].reshape(1, -1)
    cv_map.plot(
        method="slices",
        view="z",
        cut_coords=[[-12, -6, 0, 6, 12]],
        bg_img=DATASET.background,
        title="Ridge cross-validated R² (per-voxel α, FIR lags 1–3)",
        cmap="hot",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    CV R² is smaller than in-sample (held-out is harder) and concentrates in the visual cortex that actually tracks the stimulus. Finally, the spread of selected α confirms why per-voxel regularization matters — voxels disagree:
    """
    )
    return


@app.cell
def _(ALPHAS, bold, np, plt):
    best_alpha = np.asarray(bold.cv_results_["best_alpha"]).ravel()
    alpha_fig, alpha_ax = plt.subplots(figsize=(7, 3))
    alpha_ax.hist(np.log10(best_alpha), bins=len(ALPHAS), color="steelblue")
    alpha_ax.set_xlabel(r"$\log_{10}(\alpha)$ selected per voxel")
    alpha_ax.set_ylabel("voxel count")
    alpha_ax.set_title("Per-voxel ridge α — voxels disagree on regularization")
    alpha_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Recap

    | Stage | What it does | Key API |
    |---|---|---|
    | Load runs | Concatenate BOLD + stimulus across runs | `concatenate([...])` |
    | Features | FIR lag bank (learn the HRF, don't assume it) | `lag_features(stim, [1, 2, 3])` |
    | In-sample fit | Fixed-α ridge → optimistic R² | `bold.fit(model="ridge", X=, alpha=1.0, scale=False)` |
    | Honest fit | Per-voxel α via CV → out-of-sample R² | `bold.fit(model="ridge", X=, alpha="auto", alphas=, cv=)` |
    | Inspect | CV R² map + selected α per voxel | `bold.cv_results_["mean_score"]`, `["best_alpha"]` |

    **Next steps**

    - [GLM analysis](workflows-01_glm.html) — the inferential counterpart: which voxels respond.
    - [Multivariate pattern analysis](workflows-03_mvpa.html) — decode the stimulus from brain patterns.
    - [Inter-subject correlation](workflows-04_isc.html) — shared responses to naturalistic stimuli.
    """
    )
    return


if __name__ == "__main__":
    app.run()
