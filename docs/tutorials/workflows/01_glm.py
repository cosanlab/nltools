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
def _(wasm_ready):
    _ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
    import numpy as np
    from joblib import Memory

    from nltools.data import BrainData, DesignMatrix
    from nltools.stats import fdr, threshold
    from nltools.utils import concatenate

    # Memoize per-subject fits to disk (.cache/ is git-ignored) so re-running
    # the notebook reloads results instead of refitting every voxel.
    memory = Memory(".cache/tutorials", verbose=0)
    return BrainData, DesignMatrix, concatenate, fdr, memory, np, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## How to do it

    We use the **language localizer demo** from `nilearn` — 10 subjects viewing blocks of sentences (`language`) vs. consonant strings (`string`). Each subject's BIDS derivatives give us three files: the preprocessed BOLD, an events TSV, and a confounds TSV. In the browser (Pyodide), the same analysis uses a trimmed copy of the eight subjects fitted below so it runs without a server.
    """
    )
    return


@app.cell
async def _(IN_WASM, wasm_ready):
    _ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
    import json
    from pathlib import Path

    from nilearn.datasets import fetch_language_localizer_demo_dataset
    from nilearn.interfaces.bids import get_bids_files
    from nltools.templates import fetch_resource, seed_resources

    if IN_WASM:
        _pyodide_subjects = [f"{_subject:02d}" for _subject in range(1, 9)]

        def _resource_paths(sub: str) -> dict:
            stem = f"sub-{sub}_task-languagelocalizer"
            return {
                "bold": f"tutorials/glm/derivatives/sub-{sub}/func/{stem}_desc-preproc_bold.nii.gz",
                "sidecar": f"tutorials/glm/derivatives/sub-{sub}/func/{stem}_desc-preproc_bold.json",
                "confounds": f"tutorials/glm/derivatives/sub-{sub}/func/{stem}_desc-confounds_regressors.tsv",
                "events": f"tutorials/glm/sub-{sub}/func/{stem}_events.tsv",
            }

        _glm_resources = [
            _relpath
            for _sub in _pyodide_subjects
            for _relpath in _resource_paths(_sub).values()
        ] + [
            # MNI templates the resample + slice plots fetch — pre-seed in Pyodide.
            # Both 2mm (BrainData default brainspace) and 3mm are covered.
            "default/2mm-MNI152-2009fsl-mask.nii.gz",
            "default/2mm-MNI152-2009fsl-brain.nii.gz",
            "default/2mm-MNI152-2009fsl-T1.nii.gz",
            "default/3mm-MNI152-2009fsl-mask.nii.gz",
            "default/3mm-MNI152-2009fsl-brain.nii.gz",
            "default/3mm-MNI152-2009fsl-T1.nii.gz",
        ]
        await seed_resources(_glm_resources)

        _pyodide_files = {}
        for _sub in _pyodide_subjects:
            _relpaths = _resource_paths(_sub)
            _sidecar = fetch_resource(_relpaths["sidecar"])
            _pyodide_files[_sub] = {
                "bold": fetch_resource(_relpaths["bold"]),
                "events": fetch_resource(_relpaths["events"]),
                "confounds": fetch_resource(_relpaths["confounds"]),
                "TR": json.loads(Path(_sidecar).read_text())["RepetitionTime"],
            }

        def get_sub_files(sub: str) -> dict:
            """Resolve one subject's trimmed browser-ready tutorial files."""
            return _pyodide_files[sub]

    else:
        DATASET = fetch_language_localizer_demo_dataset(verbose=0)
        DATA_DIR = Path(DATASET["data_dir"])

        def get_sub_files(sub: str) -> dict:
            """Resolve one subject's BOLD, events, confounds, and TR from BIDS."""
            derivatives = DATA_DIR / "derivatives"
            sidecar = get_bids_files(derivatives, file_tag="bold", file_type="json", sub_label=sub)[0]
            return {
                "bold": get_bids_files(derivatives, file_tag="bold", file_type="nii.gz", sub_label=sub)[0],
                "events": get_bids_files(DATA_DIR, file_tag="events", file_type="tsv", sub_label=sub)[0],
                "confounds": get_bids_files(derivatives, file_type="tsv", modality_folder="func", sub_label=sub)[0],
                "TR": json.loads(Path(sidecar).read_text())["RepetitionTime"],
            }

    return (get_sub_files,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### First level (single subject)

    The recipe for one subject: load the BOLD (`BrainData` resamples to standard MNI automatically), build the design, and fit. Building a `DesignMatrix` from a BIDS events file creates boxcar regressors and **convolves them with the canonical (Glover) HRF for you** — columns come back as `language_c0` / `string_c0` (pass `hrf_model=None` for raw boxcars to `.convolve()` yourself). We append the motion confounds as nuisance columns and add polynomial drift. Wrapping it in `memory.cache` means each subject is fit once, then reloaded from disk.
    """
    )
    return


@app.cell
def _(BrainData, DesignMatrix, get_sub_files, memory):
    @memory.cache
    def first_level(sub: str, contrast: str = "language_c0 - string_c0"):
        """Fit one subject's GLM; return its design and the contrast bundle.

        We return only the lightweight design and contrast maps (not the
        fitted model, which carries residuals and a copy of the data) so the
        on-disk cache stays small.
        """
        f = get_sub_files(sub)
        brain = BrainData(f["bold"])
        events = DesignMatrix(f["events"], run_length=brain.shape[0], TR=f["TR"])
        confounds = DesignMatrix(f["confounds"], run_length="infer", TR=f["TR"])
        brain.fit(X=events.append(confounds, axis=1, as_confounds=True).add_poly(2))
        return brain.design_matrix, brain.compute_contrasts(contrast, contrast_type="all")

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
    The helper returns the `language > string` contrast as a bundle — `beta`, `t`, `z`, `p`, `se` — computed in one call with `contrast_type="all"`, so we can threshold the t-map here *and* reuse the β map for the group analysis below.
    """
    )
    return


@app.cell
def _(contrasts):
    contrasts["t"].plot(method="slices", threshold=3.09, title="sub-01: language > string (t)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Even at one subject the left-lateralized fronto-temporal language network is visible (`|t| > 3.09`, two-tailed p ≈ 0.001).

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
    `concatenate` stacks the per-subject maps into one `(n_subjects, n_voxels)` `BrainData`. `BrainData.ttest` runs a voxelwise one-sample test, returning the effect-size `mean`, the parametric `t`, a signed `z`, and `p`. `nltools.stats.threshold` keeps the `z` values whose `p` clears a cutoff — here voxelwise `p < 0.001`.
    """
    )
    return


@app.cell
def _(beta_maps, concatenate, threshold):
    group = concatenate(beta_maps)
    group_result = group.ttest()
    group_z = threshold(group_result["z"], group_result["p"], thr=0.001)
    group_z.plot(method="slices", title="Group: language > string (voxelwise p < 0.001)")
    return (group_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Multiple-comparisons correction

    That `p < 0.001` map is *uncorrected* — it ignores that we ran tens of thousands of tests. `nltools.stats.fdr` returns the p-threshold controlling the false-discovery rate. Whole-brain correction is stringent: on a ten-subject demo, far fewer voxels survive than at the uncorrected threshold — exactly the inflation that correction guards against. Restricting the search to an ROI (see the [MVPA tutorial](workflows-03_mvpa.html)) recovers power.
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
    | Contrast | Linear combination of βs (effect size + inference) | `brain.compute_contrasts("A - B", contrast_type="all")` |
    | Stack subjects | Concatenate first-level β maps | `concatenate([...])` |
    | Group test | Voxelwise one-sample t-test → `{mean, t, z, p}` | `group.ttest()` |
    | Correction | FDR threshold | `nltools.stats.fdr`, `nltools.stats.threshold` |

    The per-subject loop is the explicit path; `BrainCollection` will wrap multi-subject fitting into a single call once it lands on this branch.

    **Next steps**

    - [Encoding models](workflows-02_encoding.html) — predict brain activity *from* stimulus features (GLM vs. Ridge).
    - [Multivariate pattern analysis](workflows-03_mvpa.html) — decode conditions and compare representational geometry.
    - [Inter-subject correlation](workflows-04_isc.html) — shared responses to naturalistic stimuli.
    """
    )
    return


if __name__ == "__main__":
    app.run()
