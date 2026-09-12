# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Univariate regression — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Univariate Regression

    A two-level general linear model asks, one voxel at a time, whether activity
    tracks something you measured. The first level fits a separate regression
    for each subject and keeps its effect map. The second level tests those maps
    across subjects.

    Here the measured variable is pain: 28 subjects, each with three images at
    low, medium and high thermal intensity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the data

    `fetch_pain` returns all 84 images as one `BrainData`, with the
    per-image metadata in `.X`. The `joblib` cache keeps the docs build from
    re-reading them on every run; you can call `fetch_pain()` directly.
    """)
    return


@app.cell
def _():
    import numpy as np
    from joblib import Memory

    from nltools.algorithms import fdr, threshold
    from nltools.data import DesignMatrix
    from nltools.datasets import fetch_pain
    from nltools import concatenate

    memory = Memory(".tutorial-cache", verbose=0)

    data = memory.cache(fetch_pain)()
    data.X.select("SubjectID", "PainIntensity", "PainLevel").head()
    return DesignMatrix, concatenate, data, fdr, np, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First level: one regression per subject

    Each subject's design has two columns: pain intensity coded 1/2/3, and an
    intercept that `add_poly(0)` appends. With three images per subject this is
    the smallest model that can express "activity rises with intensity" — an
    illustration, not a design you would run a study on.
    """)
    return


@app.cell
def _(DesignMatrix, data):
    def subject_design(images):
        """Pain intensity plus an intercept, one row per image."""
        return DesignMatrix({"Pain": images.X["PainLevel"].to_list()}).add_poly(0)

    subject_design(data[data.X["SubjectID"] == 1]).plot(title="Design for one subject")
    return (subject_design,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `fit(model="glm", X=design)` runs ordinary least squares at every voxel and
    stores the fit on the object. `compute_contrasts("Pain")` then returns the
    effect map for that column — the slope of activity on intensity. Feed effect
    maps to the group test, never first-level t-maps: a t divides the effect by a
    standard error that varies across subjects for reasons unrelated to pain.

    `concatenate` stacks the 28 maps into one `(subjects, voxels)` object.
    """)
    return


@app.cell
def _(concatenate, data, subject_design):
    subject_ids = data.X["SubjectID"].unique(maintain_order=True).to_list()

    slope_maps = []
    for subject in subject_ids:
        images = data[data.X["SubjectID"] == subject]
        images.fit(model="glm", X=subject_design(images))
        slope_maps.append(images.compute_contrasts("Pain"))

    slopes = concatenate(slope_maps)
    slopes
    return slopes, subject_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Second level: a one-sample test across subjects

    `ttest` tests each voxel's mean against zero and returns `mean`, `t`, `z` and
    `p` as unthresholded maps. Correction comes afterwards: `fdr` returns the
    p-value cutoff that controls the false discovery rate at `q`, and `threshold`
    keeps the voxels of one map whose p-values in another clear that cutoff.
    """)
    return


@app.cell
def _(fdr, np, slopes, threshold):
    slope_stats = slopes.ttest()
    slope_p = np.asarray(slope_stats["p"].data)
    slope_cutoff = fdr(slope_p, q=0.001)

    print(f"FDR q < 0.001 keeps p < {slope_cutoff:.2e}")
    print(f"surviving voxels: {(slope_p < slope_cutoff).sum()} of {slope_p.size}")

    threshold(slope_stats["t"], slope_stats["p"], thr=slope_cutoff).plot(
        title="Pain slope across subjects (t, FDR q < 0.001)"
    )
    return (slope_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most of the brain survives, which is what a whole-brain thermal-pain
    manipulation does: the intensities differ in arousal and attention as well as
    in nociception, and those effects are widespread.

    ## The same question as a linear contrast

    Three equally spaced levels invite a simpler route. Weight the images
    −1 / 0 / +1 and add them up, and you have one contrast map per subject with
    no model fitted at all. Multiplying a `BrainData` by a vector of length
    `n_images` does exactly that weighted sum.
    """)
    return


@app.cell
def _(concatenate, data, subject_ids):
    contrast_maps = []
    for contrast_subject in subject_ids:
        contrast_images = data[data.X["SubjectID"] == contrast_subject]
        weights = contrast_images.X["PainLevel"].to_numpy() - 2
        contrast_maps.append(contrast_images * weights)

    contrasts = concatenate(contrast_maps)
    contrasts
    return (contrasts,)


@app.cell
def _(contrasts, fdr, np, threshold):
    contrast_stats = contrasts.ttest()
    contrast_p = np.asarray(contrast_stats["p"].data)
    contrast_cutoff = fdr(contrast_p, q=0.001)

    threshold(contrast_stats["t"], contrast_stats["p"], thr=contrast_cutoff).plot(
        title="Linear pain contrast across subjects (t, FDR q < 0.001)"
    )
    return (contrast_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two maps are not merely similar, they are the same map. With equally
    spaced predictor values the regression slope is the linear contrast divided
    by a constant, and dividing every subject's map by the same constant leaves
    the one-sample t unchanged. Both are undefined — NaN — at the voxels that
    hold zero in every image, which is why the comparison allows NaN to match
    NaN.
    """)
    return


@app.cell
def _(contrast_stats, np, slope_stats):
    print(
        "t-maps identical:",
        np.allclose(contrast_stats["t"].data, slope_stats["t"].data, equal_nan=True),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | Step | Call |
    |---|---|
    | Build a subject design | `DesignMatrix({...}).add_poly(0)` |
    | Fit every voxel | `images.fit(model="glm", X=design)` |
    | Effect map for one column | `images.compute_contrasts("Pain")` |
    | Weighted sum instead of a fit | `images * weights` |
    | Stack subjects | `concatenate(maps)` |
    | Group test | `stack.ttest()` → `mean`, `t`, `z`, `p` |
    | Correction | `fdr(p, q=...)` then `threshold(t, p, thr=...)` |

    **Next steps**

    - [Multivariate Prediction](02_multivariate_prediction.md) — ask what the
      whole pattern predicts instead of testing voxels one at a time.
    - [GLM Analysis](../workflows/01_glm.md) — the same two levels on raw
      timeseries, with HRF convolution and nuisance regression.
    """)
    return


if __name__ == "__main__":
    app.run()
