# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# GLM Analysis — marimo notebook. Source of truth for the docs page; rendered to MyST by scripts/marimo_to_myst.py.
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
    # GLM Analysis

    **What it answers.** *Where* in the brain does activity track your task design? The general linear model (GLM) is the mass-univariate workhorse of task fMRI: fit one regression per voxel, then test contrasts between conditions. Use it when you have a known design and want a statistical map of effects.

    For the underlying theory, see the GLM chapters in [dartbrains](https://dartbrains.org). This tutorial is about *running* the analysis in nltools.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **How it works.** A GLM analysis runs in two stages:

    - **First level (single subject).** Regress each voxel's timeseries on the design matrix **X** → one β (and t) per regressor. A *contrast* is a linear combination of βs, giving a per-subject effect map.
    - **Second level (group).** Stack the per-subject contrast **effect-size** maps and run a one-sample test across subjects.

    Feed *effect sizes* (βs), not first-level t-maps, into the group test: a first-level t is `β / SE(β)`, and SE varies across subjects for reasons unrelated to the effect (scan length, motion). Stacking t-maps would conflate effect magnitude with first-level precision.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    from joblib import Memory
    from scipy.signal import detrend

    from nltools.data import BrainData, DesignMatrix
    from nltools.algorithms import fdr, threshold
    from nltools.utils import concatenate

    # Memoize per-subject fits to disk (.cache/ is git-ignored) so re-running
    # the notebook reloads results instead of refitting every voxel.
    memory = Memory(".cache/tutorials", verbose=0)
    return BrainData, DesignMatrix, concatenate, detrend, fdr, memory, np, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## How to do it

    We use the **language localizer demo** from `nilearn` — 10 subjects viewing blocks of sentences (`language`) vs. consonant strings (`string`). Each subject's BIDS derivatives give us three files: the preprocessed BOLD, an events TSV, and a confounds TSV.

    The BOLD is already MNI-normalized, but on a 4.5 mm grid rather than the bundled 1/2/3 mm template grids. Interpolating every subject up to 3 mm would add voxels without adding information, so we analyze on the data's own grid: one MNI152 brain mask resampled down to it, shared by every subject so their maps stack directly, and the MNI152 T1 as the plotting background (`bg_img=`).
    """
    )
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from nilearn.datasets import (
        fetch_language_localizer_demo_dataset,
        load_mni152_brain_mask,
        load_mni152_template,
    )
    from nilearn.image import resample_to_img
    from nilearn.interfaces.bids import get_bids_files

    DATASET = fetch_language_localizer_demo_dataset(verbose=0)
    DATA_DIR = Path(DATASET["data_dir"])

    def get_sub_files(sub: str) -> dict:
        """Resolve one subject's BOLD, events, confounds, and TR from BIDS."""
        derivatives = DATA_DIR / "derivatives"
        sidecar = get_bids_files(
            derivatives, file_tag="bold", file_type="json", sub_label=sub
        )[0]
        return {
            "bold": get_bids_files(
                derivatives, file_tag="bold", file_type="nii.gz", sub_label=sub
            )[0],
            "events": get_bids_files(
                DATA_DIR, file_tag="events", file_type="tsv", sub_label=sub
            )[0],
            "confounds": get_bids_files(
                derivatives, file_type="tsv", modality_folder="func", sub_label=sub
            )[0],
            "TR": json.loads(Path(sidecar).read_text())["RepetitionTime"],
        }

    # All subjects share one 4.5 mm MNI grid; nearest-neighbour keeps the mask binary.
    MNI_MASK = resample_to_img(
        load_mni152_brain_mask(), get_sub_files("01")["bold"], interpolation="nearest"
    )
    MNI_T1 = load_mni152_template(resolution=2)
    return MNI_MASK, MNI_T1, get_sub_files


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### First level (single subject)

    The recipe for one subject: load the BOLD with the shared mask, build the design, and fit. Building a `DesignMatrix` from a BIDS events file creates boxcar regressors and **convolves them with the canonical (Glover) HRF for you** — columns come back as `language_c0` / `string_c0` (pass `hrf_model=None` for raw boxcars to `.convolve()` yourself). We append the six motion parameters as nuisance columns and add polynomial drift terms. Motion estimates drift slowly themselves, so we detrend them first — otherwise the drift would be modeled twice, once by the polynomials and again by the motion columns, and the two sets of regressors would be nearly collinear. Wrapping it in `memory.cache` means each subject is fit once, then reloaded from disk.
    """
    )
    return


@app.cell
def _(BrainData, DesignMatrix, MNI_MASK, detrend, get_sub_files, memory):
    @memory.cache
    def first_level(sub: str, contrast: str = "language_c0 - string_c0"):
        """Fit one subject's GLM; return its design and the contrast bundle.

        We return only the lightweight design and contrast maps (not the
        fitted model, which carries residuals and a copy of the data) so the
        on-disk cache stays small.
        """
        f = get_sub_files(sub)
        brain = BrainData(f["bold"], mask=MNI_MASK)
        events = DesignMatrix(f["events"], run_length=brain.shape[0], TR=f["TR"])
        motion = DesignMatrix(f["confounds"], run_length="infer", TR=f["TR"])
        motion = DesignMatrix(
            detrend(motion.to_numpy(), axis=0), columns=motion.columns, TR=f["TR"]
        )
        design = events.append(motion, axis=1, as_confounds=True).add_poly(2)
        brain.fit(X=design)
        return brain.design_matrix, brain.compute_contrasts(contrast, statistic="all")

    return (first_level,)


@app.cell
def _(first_level):
    design, contrasts = first_level("01")
    design.plot()  # the design we just fit
    return (contrasts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The helper returns the `language > string` contrast as a bundle — `beta`, `t`, `z`, `p`, `se` — computed in one call with `statistic="all"`, so we can threshold the t-map here *and* reuse the β map for the group analysis below.
    """
    )
    return


@app.cell
def _(MNI_T1, contrasts):
    contrasts["t"].plot(
        method="slices",
        threshold=3.09,
        bg_img=MNI_T1,
        title="sub-01: language > string (t)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Even at one subject the left-lateralized fronto-temporal language network is visible (`|t| > 3.09`, uncorrected p ≈ 0.002 two-tailed).

    ### Second level (group)

    The same cached recipe runs per subject, returning one **effect-size** (β) map each. We loop over eight of the ten demo subjects.
    """
    )
    return


@app.cell
def _(first_level):
    SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08"]
    beta_maps = []
    for sub in SUBJECTS:
        _, sub_contrasts = first_level(sub)
        beta_maps.append(sub_contrasts["beta"])
    return (beta_maps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `concatenate` stacks the per-subject maps into one `(n_subjects, n_voxels)` `BrainData`. `BrainData.ttest` runs a voxelwise one-sample test, returning the effect-size `mean`, the parametric `t`, a signed `z`, and `p`. `nltools.algorithms.threshold` keeps the `z` values whose `p` clears a cutoff — here voxelwise `p < 0.001`.
    """
    )
    return


@app.cell
def _(MNI_T1, beta_maps, concatenate, threshold):
    group = concatenate(beta_maps)
    group_result = group.ttest()
    group_z = threshold(group_result["z"], group_result["p"], thr=0.001)
    group_z.plot(
        method="slices",
        bg_img=MNI_T1,
        title="Group: language > string (voxelwise p < 0.001)",
    )
    return (group_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Multiple-comparisons correction

    That `p < 0.001` map is *uncorrected* — it ignores that we ran tens of thousands of tests. `nltools.algorithms.fdr` returns the p-threshold controlling the false-discovery rate. Whole-brain correction is stringent: with eight subjects, few or no voxels survive FDR or Bonferroni even though dozens pass the uncorrected threshold — exactly the inflation that correction guards against. Restricting the search to an ROI (see the [MVPA tutorial](03_mvpa.md)) recovers power.
    """
    )
    return


@app.cell
def _(fdr, group_result, np):
    p_values = np.asarray(group_result["p"].data)
    n_voxels = p_values.size
    fdr_thr = fdr(p_values, q=0.05)
    bonf_thr = 0.05 / n_voxels

    n_uncorrected = int((p_values < 0.001).sum())
    n_fdr = int((p_values <= fdr_thr).sum()) if fdr_thr > 0 else 0
    n_bonferroni = int((p_values < bonf_thr).sum())

    print(f"voxels surviving, out of {n_voxels}:")
    print(f"  uncorrected (p < 0.001):  {n_uncorrected:5d}")
    print(f"  FDR (q = 0.05):           {n_fdr:5d}")
    print(f"  Bonferroni (p < 0.05/N):  {n_bonferroni:5d}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Recap

    | Stage | What it does | Key API |
    |---|---|---|
    | Build design | BIDS events → HRF-convolved regressors + confounds + drift | `DesignMatrix(events, run_length=, TR=)`, `.append(confounds, axis=1, as_confounds=True)`, `.add_poly()` |
    | First level | OLS at every voxel | `brain.fit(X=design)` |
    | Contrast | Linear combination of βs (effect size + inference) | `brain.compute_contrasts("A - B", statistic="all")` |
    | Stack subjects | Concatenate first-level β maps | `concatenate([...])` |
    | Group test | Voxelwise one-sample t-test → `{mean, t, z, p}` | `group.ttest()` |
    | Correction | FDR threshold | `nltools.algorithms.fdr`, `nltools.algorithms.threshold` |

    The per-subject loop is the explicit path; [`BrainCollection`](../basics/04_brain_collection.md) wraps the same per-subject fit → contrast → group test into parallel, cached calls (`bc.fit(...)`, `bc.compute_contrasts(...)`, `bc.ttest()`).

    **Next steps**

    - [Encoding models](02_encoding.md) — predict brain activity *from* stimulus features (GLM vs. Ridge).
    - [Multivariate pattern analysis](03_mvpa.md) — decode conditions and compare representational geometry.
    - [Inter-subject correlation](04_isc.md) — shared responses to naturalistic stimuli.
    """
    )
    return


if __name__ == "__main__":
    app.run()
