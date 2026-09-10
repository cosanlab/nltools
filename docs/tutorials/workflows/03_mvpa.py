# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Multivariate Pattern Analysis — marimo notebook. Source of truth for the docs page; rendered to MyST by scripts/marimo_to_myst.py.
import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Multivariate Pattern Analysis

    **What it answers.** Does the *distributed pattern* of activity across many voxels carry information about the conditions — beyond what any single voxel shows? Two complementary approaches:

    - **Decoding** — can a classifier *predict* the condition from the pattern? (cross-validated accuracy)
    - **RSA** — what is the *geometry* of the patterns relative to one another, and does it match a hypothesis? (a representational dissimilarity matrix compared to a model)

    Both run at three spatial scales via `spatial_scale=` — `'whole_brain'`, `'roi'`, `'searchlight'` — and either within a subject (here) or across subjects (loop + group test). For the theory, see [dartbrains](https://dartbrains.org).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **How it works.** Both approaches operate on the same patterns; they differ in the question. Decoding fits a classifier across voxels and scores it on held-out data. RSA turns patterns into a distance matrix (the RDM) and correlates that geometry with a model RDM. The `spatial_scale=` switch is shared: whole-brain uses every voxel jointly, ROI runs the analysis per parcel, and searchlight sweeps a roving sphere.

    We use the classic **Haxby** dataset — one subject viewing 8 object categories — for both.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from joblib import Memory

    from nltools.data import Adjacency, BrainData
    from nltools.templates import fetch_resource

    memory = Memory(".cache/tutorials", verbose=0)
    return Adjacency, BrainData, fetch_resource, memory, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Decoding

    Haxby ships in **subject space** (no MNI normalization, anisotropic 3.5 × 3.75 × 3.75 mm voxels), so we load it with the dataset's own brain mask to stay on its native grid, and plot on the subject's anatomical via `bg_img=`. Then restrict to **face vs. house** — the strongest, best-understood contrast. Boolean-indexing a `BrainData` slices its timeseries like a numpy array.
    """
    )
    return


@app.cell
def _(BrainData, np, pd):
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
    return HAXBY, LABELS, brain, trials, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Whole-brain

    `BrainData.predict()` mirrors `.fit()`: one call, one frozen `Predict` result. `cv` scores generalization; `weight_map` is the classifier refit on all the data (the publishable map). For a linear SVM:
    """
    )
    return


@app.cell
def _(HAXBY, trials, y):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    > Raw classifier weights are *not* a statistical map — a near-zero weight can still carry information other voxels already supply (see Haufe et al., 2014). For a cleaner "where", decode per region.

    ### ROI

    `spatial_scale="roi"` with a parcellation trains one classifier per parcel and returns a `score_map` — every voxel in parcel *i* filled with parcel *i*'s cross-validated accuracy. We use the bundled k50 atlas. It is defined in MNI space, and `roi_mask=` resamples it onto this subject's grid by header affine alone — a grid change, not a spatial normalization — so its parcel boundaries are only approximate for this un-normalized subject.
    """
    )
    return


@app.cell
def _(HAXBY, fetch_resource, trials, y):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Ventro-temporal cortex lights up — where face- and place-selective patches live. ROI accuracy answers "is this region informative *on its own*?", a cleaner "where" than the joint whole-brain weights.

    ### Searchlight

    A roving sphere: one classifier per voxel-neighborhood, giving a per-voxel accuracy map. Same call, `spatial_scale="searchlight"`. This fits thousands of classifiers, so it's the slow one — we cache it.
    """
    )
    return


@app.cell
def _(HAXBY, memory, trials, y):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RSA

    Decoding asks *can we separate* the conditions; RSA asks *what is their geometry*. Build one pattern per category (mean BOLD across that category's TRs, shifted 2 TRs for the hemodynamic lag), then turn the patterns into a representational dissimilarity matrix (RDM).
    """
    )
    return


@app.cell
def _(BrainData, LABELS, brain, np):
    conditions = [c for c in sorted(set(LABELS)) if c != "rest"]
    shifted = np.roll(LABELS, 2)  # align BOLD to stimulus (~5s HRF lag)
    patterns = np.vstack([brain.data[shifted == c].mean(axis=0) for c in conditions])
    category_patterns = BrainData(patterns, mask=brain.mask)
    print(
        f"category patterns: {category_patterns.shape}  ({len(conditions)} categories)"
    )
    return category_patterns, conditions


@app.cell
def _(category_patterns, conditions):
    rdm = category_patterns.distance(metric="correlation")
    rdm.labels = conditions
    rdm.plot(cmap="RdBu_r")
    return (rdm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The RDM is the full geometry — every pairwise dissimilarity at once. To test a hypothesis, build a model RDM (here: animate `face`/`cat` vs. the rest) and correlate the two with a Mantel permutation test.
    """
    )
    return


@app.cell
def _(Adjacency, conditions, np, rdm):
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
    return (model_rdm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To examine animacy structure by region, `spatial_scale="roi"` computes one
    RDM per parcel. `roi_to_brain_from_atlas` then paints each parcel's correlation with
    the model back into brain space. Align the atlas first and retain the sorted
    nonzero labels inside the source mask. These labels give the returned RDM order:
    """
    )
    return


@app.cell
def _(HAXBY, category_patterns, fetch_resource, model_rdm, np):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


if __name__ == "__main__":
    app.run()
