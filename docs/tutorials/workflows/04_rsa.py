import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Representational Similarity Analysis (RSA)

    ## Introduction

    Decoding asks **"can the spatial pattern predict the condition?"**. RSA flips the question once more:

    > **What is the *structure* of the patterns relative to one another?**

    Instead of training a classifier, we compute a **representational dissimilarity matrix (RDM)** — pairwise distances between item patterns — and ask whether that geometry matches a hypothesis (e.g. *same-condition trials cluster together*).

    The big win: the RDM is a common currency. Once your brain patterns and your hypothesis are both expressed as RDMs, comparing them is a single correlation. The same shape can also relate brain data to behavior, model layers, or another subject — anything you can turn into a distance matrix.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import json
    from nilearn.datasets import fetch_language_localizer_demo_dataset
    from nilearn.interfaces.bids import get_bids_files
    from nltools.data import BrainData, Adjacency, DesignMatrix

    return (
        Adjacency,
        BrainData,
        DesignMatrix,
        Path,
        fetch_language_localizer_demo_dataset,
        get_bids_files,
        json,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the language localizer dataset

    We'll use the same dataset as the [GLM tutorial](01_glm.md) — preprocessed, MNI-space, 10 subjects watching alternating blocks of `language` (sentences) and `string` (consonant strings). With only two categories there's no condition-level geometry to compare, so we'll work at the **trial level**: each subject sees 12 language blocks + 12 string blocks = 24 trials, and the question becomes whether each trial's spatial pattern groups by its category.
    """)
    return


@app.cell
def _(Path, fetch_language_localizer_demo_dataset, get_bids_files, json):
    DATASET = fetch_language_localizer_demo_dataset(verbose=0)
    datapath = Path(DATASET["data_dir"])

    def get_sub_files(sub: str) -> dict:
        """Return one subject's BOLD, events, confounds, and TR from BIDS."""
        events = get_bids_files(datapath, file_tag="events", file_type="tsv", sub_label=sub)[0]
        bold = get_bids_files(
            datapath / "derivatives", file_tag="bold", file_type="nii.gz", sub_label=sub
        )[0]
        sidecar = get_bids_files(
            datapath / "derivatives", file_tag="bold", file_type="json", sub_label=sub
        )[0]
        confounds = get_bids_files(
            datapath / "derivatives", file_type="tsv", modality_folder="func", sub_label=sub
        )[0]
        return {
            "bold": bold,
            "events": events,
            "confounds": confounds,
            "TR": json.loads(Path(sidecar).read_text())["RepetitionTime"],
        }

    return (get_sub_files,)


@app.cell
def _(BrainData, get_sub_files):
    s1 = get_sub_files("01")
    brain = BrainData(s1["bold"])
    print(f"brain.shape: {brain.shape}  TR={s1['TR']}s")
    return brain, s1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build a single-trial design (LSA)

    A standard first-level GLM collapses all `language` trials into one regressor and all `string` trials into another. For RSA we want **one pattern per trial**, so we rename each trial type to be unique — the 12-language + 12-string design becomes 24 separate regressors. This is the **Least-Squares All (LSA)** estimator:
    """)
    return


@app.cell
def _(pd, s1):
    ev = pd.read_csv(s1["events"], sep="\t").sort_values("onset").reset_index(drop=True)
    ev["trial_type"] = [
        f"{row.trial_type}_t{i:02d}" for i, row in enumerate(ev.itertuples(), start=1)
    ]
    trial_names = ev["trial_type"].tolist()
    trial_labels = [name.rsplit("_", 1)[0] for name in trial_names]
    print(ev.head())
    print(
        f"\n{len(trial_names)} trial regressors, "
        f"{sum(c == 'language' for c in trial_labels)} language + "
        f"{sum(c == 'string' for c in trial_labels)} string"
    )
    return ev, trial_labels, trial_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Write the renamed events to a temp TSV so `DesignMatrix` can read it like any BIDS file, then build the design matrix with confounds and polynomial drift:
    """)
    return


@app.cell
def _(DesignMatrix, brain, ev, s1):
    import tempfile

    events_lsa = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
    ev.to_csv(events_lsa.name, sep="\t", index=False)

    trials_dm = DesignMatrix(events_lsa.name, run_length=brain.shape[0], TR=s1["TR"])
    confounds_dm = DesignMatrix(s1["confounds"], run_length="infer", TR=s1["TR"])
    designmat = trials_dm.append(confounds_dm, axis=1, as_confounds=True).add_poly(2)
    print(
        f"Design matrix: {designmat.shape}  "
        f"(trial regressors + 6 motion + drift)"
    )
    return (designmat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fit the GLM and take the first 24 beta maps — one whole-brain pattern per trial:
    """)
    return


@app.cell
def _(brain, designmat, pd, trial_labels, trial_names):
    brain.fit(X=designmat)
    n_trials = len(trial_names)
    trial_betas = brain.glm_betas[:n_trials]
    trial_betas.Y = pd.DataFrame({"trial": trial_names, "condition": trial_labels})
    print(f"trial_betas.shape: {trial_betas.shape}  (n_trials, n_voxels)")
    return (trial_betas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute the RDM

    `BrainData.distance(metric='correlation')` returns an `Adjacency` whose entry *(i,j)* is `1 − corr(pattern_i, pattern_j)` — the canonical RSA dissimilarity:
    """)
    return


@app.cell
def _(trial_betas, trial_names):
    rdm = trial_betas.distance(metric="correlation")
    rdm.labels = trial_names
    print(f"rdm.shape: {rdm.shape}  (matrix_type={rdm.matrix_type})")
    return (rdm,)


@app.cell
def _(plt, rdm):
    fig, ax = plt.subplots(figsize=(6, 5))
    rdm.plot(cmap="RdBu_r", axes=ax)
    ax.set_title("Whole-brain trial RDM (1 − Pearson r)")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A categorical model

    Build a hypothesis RDM where same-condition pairs have distance 0 and different-condition pairs have distance 1:
    """)
    return


@app.cell
def _(Adjacency, np, trial_labels, trial_names):
    labels_arr = np.array(trial_labels)
    model_rdm = (labels_arr[:, None] != labels_arr[None, :]).astype(float)
    category_model = Adjacency(model_rdm, matrix_type="distance", labels=trial_names)
    return (category_model,)


@app.cell
def _(category_model, plt):
    fig, ax = plt.subplots(figsize=(6, 5))
    category_model.plot(vmin=0, vmax=1, cmap="RdBu_r", axes=ax)
    ax.set_title("Model RDM: same condition = 0, different = 1")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The checkerboard pattern reflects the alternating block design — every trial has a different-condition neighbour on each side and same-condition partners two trials away.

    ## Whole-brain RSA

    Compare the brain RDM and the model RDM. Spearman rank correlation with a Mantel-style permutation null:
    """)
    return


@app.cell
def _(category_model, rdm):
    result_wb = rdm.similarity(
        category_model, metric="spearman", n_permute=1000, random_state=0
    )
    print(
        f"Whole-brain RSA: rho = {result_wb['correlation']:.3f}  p = {result_wb['p']:.4f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The whole-brain trial RDM is dominated by global BOLD fluctuations and adjacent-trial autocorrelation — most voxels don't carry language-vs-string structure. The signal lives regionally, so we'll zoom in.

    ## ROI RSA

    `spatial_scale='roi'` with a parcellation atlas returns *one RDM per region* — same data, sliced by ROI. We'll use the bundled k50 atlas (50 cortical ROIs in default 3mm MNI space, also used in the decoding tutorial):
    """)
    return


@app.cell
def _(trial_betas):
    from nltools.templates import fetch_resource

    atlas_path = fetch_resource("masks/default/3mm-MNI152-2009fsl-k50.nii.gz")

    roi_rdms = trial_betas.distance(
        metric="correlation", spatial_scale="roi", roi_mask=atlas_path
    )
    print(f"roi_rdms: {len(roi_rdms)} RDMs, each {roi_rdms[0].shape}")
    return (roi_rdms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comparing each ROI's RDM to the model and projecting the correlations back to brain space is a one-liner — `project=True` returns a `BrainData` where every voxel inside ROI *k* gets ROI *k*'s correlation with the model:
    """)
    return


@app.cell
def _(category_model, roi_rdms):
    rsa_map = roi_rdms.similarity(
        category_model, metric="spearman", permutation_method=None, project=True
    )
    print(f"rsa_map.shape: {rsa_map.shape}")
    rsa_map.plot(
        method="slices",
        title="ROI RSA: where same-condition trials cluster",
        cmap="RdBu_r",
        colorbar=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Regions where same-condition trials are reliably more similar than between-condition trials show up bright — those are the ROIs whose representational geometry matches the categorical hypothesis.

    ## Inspect the top ROI

    Pull the RDM for the best-matching ROI and look at the trial-level structure directly:
    """)
    return


@app.cell
def _(category_model, np, plt, roi_rdms, trial_names):
    roi_rsa = roi_rdms.similarity(
        category_model, metric="spearman", permutation_method=None
    )
    rsa_corrs = np.array([r["correlation"] for r in roi_rsa])
    top_roi = int(np.argmax(rsa_corrs))
    print(f"Best ROI: {top_roi}  (rho = {rsa_corrs[top_roi]:.3f})")

    best_rdm = roi_rdms[top_roi]
    best_rdm.labels = trial_names

    fig, ax = plt.subplots(figsize=(6, 5))
    best_rdm.plot(cmap="RdBu_r", axes=ax)
    ax.set_title(f"Top ROI ({top_roi}): trial RDM")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The block structure is now visible — within-language and within-string pairs (small diagonal blocks of dark cells) are more similar than cross-condition pairs (the lighter off-diagonal blocks).

    ## Putting it together

    | Step | Call | Returns |
    |---|---|---|
    | LSA design | `DesignMatrix(events_lsa, run_length=..., TR=...)` (unique trial_types) | `DesignMatrix` (one regressor per trial) |
    | Trial-level fit | `brain.fit(X=designmat); brain.glm_betas[:n_trials]` | `BrainData` (n_trials, n_voxels) |
    | Whole-brain RDM | `trial_betas.distance(metric='correlation')` | `Adjacency` (matrix_type='distance') |
    | ROI RDMs | `trial_betas.distance(metric='correlation', spatial_scale='roi', roi_mask=atlas)` | `Adjacency` with n_rois matrices |
    | Compare to model | `rdm.similarity(model, metric='spearman', n_permute=K)` | `dict` with `correlation`, `p` |
    | Project to brain | `roi_rdms.similarity(model, project=True)` | `BrainData` of per-voxel correlations |

    > **Why trial-level instead of condition-level?** With only two categories a condition-level RDM collapses to a single off-diagonal value — no geometry to compare. Trial-level betas give us 24 items and a rich 24×24 RDM. The cost is per-trial noise; the win is real hypothesis structure. When you have many conditions, mean-of-trials or per-condition GLM contrasts are the natural choice instead.

    ## Next Steps

    - **Searchlight RSA**: roving sphere — `trial_betas.distance(spatial_scale='searchlight', radius_mm=8.0)` returns one RDM per searchlight, ready to compare to a model RDM voxelwise.
    - **Decoding view**: instead of distance structure, can a classifier separate the conditions? — see [Decoding](05_decoding.md).
    """)
    return


if __name__ == "__main__":
    app.run()
