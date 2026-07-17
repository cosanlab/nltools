# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Only marimo + the emscripten HTTP shim load from this header. nltools and its whole
#     # runtime stack are micropip-installed by the IN_WASM setup cell (UNPINNED, so Pyodide's
#     # bundled builds win) — see that cell. Listing the stack here too makes marimo's header
#     # auto-install redundantly pull unpinned latest scikit-learn/scipy/pandas/matplotlib,
#     # which drag in `packaging>=26` (absent in Pyodide 0.27.7) and error out.
#     "marimo",
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


@app.cell(hide_code=True)
def _():
    import sys

    IN_WASM = sys.platform == "emscripten"
    return (IN_WASM,)


@app.cell(hide_code=True)
async def _(IN_WASM):
    # In-browser only: install nltools + its full runtime stack before any nltools import
    # runs, then hand `wasm_ready` to every nltools-importing cell to force ordering. We
    # can't rely on marimo's PEP 723 header auto-install alone: it races cell execution and
    # marimo never re-runs a cell that already failed with ModuleNotFoundError.
    #
    # The dataset (nilearn development_fmri) is hosted under tutorials/isc/ in the
    # nltools/niftis HF dataset; the data cell seeds a light 6-subject subset into the
    # IDBFS cache in the browser and reads the full 12 from local nilearn otherwise.
    wasm_ready = True
    if IN_WASM:
        import micropip
        import js

        # Install the stack UNPINNED so micropip takes Pyodide's bundled builds (pinning
        # to nltools' host versions, e.g. joblib>=1.5.3, fails against Pyodide's bundled
        # joblib). nilearn is the exception: 0.14+ needs packaging>=26 (absent in Pyodide
        # 0.27.7), so pin the last 0.13.x. numpy/scipy/pandas/sklearn/matplotlib come in
        # transitively at their bundled versions.
        await micropip.install(
            [
                "nibabel",
                "nilearn==0.13.1",
                "seaborn",
                "polars",
                "pynv",
                "ipyniivue",
                "ipywidgets",
                "huggingface-hub",
                "anywidget",
            ]
        )
        # deps=False installs the wheel without re-checking nltools' own version pins.
        await micropip.install(
            js.location.origin + "__NLTOOLS_WHEEL_URL__", deps=False
        )
    return (wasm_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Inter-Subject Correlation

    **What it answers.** Which brain regions respond *consistently across people* to a shared naturalistic stimulus (a movie, a story)? There's no explicit design matrix to model — instead, ISC uses other subjects' responses as the model, asking where the stimulus drives a common, time-locked signal.

    For the theory, see the ISC material in [naturalistic-data](https://naturalistic-data.org). This tutorial runs one in nltools.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **How it works.** ISC runs in two stages, like a GLM's first/second level:

    - **Compute** the cross-subject similarity per region. Either **pairwise** (correlate every pair of subjects, then summarize) or **leave-one-out** (correlate each subject against the mean of the others). LOO is larger because each subject is compared to a denoised group average.
    - **Group inference** on whether that similarity exceeds chance, via a permutation/bootstrap test that respects the temporal structure.
    """
    )
    return


@app.cell
def _(wasm_ready):
    _ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
    import warnings

    import numpy as np
    from joblib import Memory

    from nltools.algorithms.inference.isc import isc_permutation_test
    from nltools.data import BrainData
    from nltools.mask import roi_to_brain_from_atlas
    from nltools.templates import fetch_resource, seed_resources

    memory = Memory(".cache/tutorials", verbose=0)
    warnings.filterwarnings("ignore", message="Cannot detect name collisions")
    return (
        BrainData,
        fetch_resource,
        isc_permutation_test,
        memory,
        np,
        roi_to_brain_from_atlas,
        seed_resources,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## How to do it

    We use nilearn's **development_fmri** dataset — children and adults watching the same short Pixar movie. For each subject we extract a region-mean timeseries with the bundled k50 atlas, giving one `(timepoints, regions)` array per subject; stacking them is the `(timepoints, subjects, regions)` input ISC expects. (In a full analysis you'd regress the provided confounds first.)
    """
    )
    return


@app.cell(hide_code=True)
async def _(IN_WASM, wasm_ready, fetch_resource, seed_resources):
    # In-browser only: seed a light 6-subject movie subset + the ROI atlas, and
    # wrap the BOLD in a Bunch mimicking nilearn's fetch_development_fmri().
    # `browser_movie` stays None locally, where the visible cell below loads from
    # nilearn. (6 matches N_SUBJECTS in the browser; see the visible cell.)
    # Imports/vars are underscore-aliased to stay cell-local.
    _ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
    browser_movie = None
    if IN_WASM:
        from sklearn.utils import Bunch as _Bunch

        _subs = [f"{_i:02d}" for _i in range(1, 7)]
        _isc_resources = [
            f"tutorials/isc/sub-{_s}_task-pixar_bold.nii.gz" for _s in _subs
        ] + ["masks/default/3mm-MNI152-2009fsl-k50.nii.gz"]
        await seed_resources(_isc_resources)
        browser_movie = _Bunch(
            func=[
                fetch_resource(f"tutorials/isc/sub-{_s}_task-pixar_bold.nii.gz")
                for _s in _subs
            ]
        )
    return (browser_movie,)


@app.cell
def _(IN_WASM, browser_movie, BrainData, fetch_resource, memory, np):
    from nilearn.datasets import fetch_development_fmri

    # In-browser a light 6-subject subset streams from HF; locally use the full 12.
    N_SUBJECTS = 6 if IN_WASM else 12
    if IN_WASM:
        DATA = browser_movie
    else:
        DATA = fetch_development_fmri(n_subjects=N_SUBJECTS, verbose=0)

    ATLAS = fetch_resource("masks/default/3mm-MNI152-2009fsl-k50.nii.gz")

    @memory.cache
    def region_timeseries(n_subjects):
        """Region-mean timeseries per subject (slow load; cached to disk)."""
        # extract_roi returns (n_regions, n_timepoints); transpose to (time, region).
        return [BrainData(DATA.func[i]).extract_roi(ATLAS).T for i in range(n_subjects)]

    series = region_timeseries(N_SUBJECTS)
    isc_data = np.stack(series, axis=1)  # (timepoints, subjects, regions)
    print(f"ISC input: {isc_data.shape}  (timepoints, subjects, regions)")
    return ATLAS, isc_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Compute ISC + group inference

    `isc_permutation_test` does both stages in one call: it computes the per-region ISC (`summary_statistic="pairwise"`) and returns a permutation p-value per region.
    """
    )
    return


@app.cell
def _(isc_data, isc_permutation_test, np):
    pairwise = isc_permutation_test(
        isc_data,
        summary_statistic="pairwise",
        metric="median",
        n_permute=1000,
        random_state=0,
        progress_bar=False,
    )
    isc_values = np.asarray(pairwise["isc"])
    p_values = np.asarray(pairwise["p"])
    print(
        f"pairwise ISC — median {np.median(isc_values):.3f}, max {isc_values.max():.3f}"
    )
    print(
        f"regions significant (p < 0.05): {(p_values < 0.05).sum()} / {isc_values.size}"
    )
    return (isc_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Paint the per-region ISC back onto the brain with `roi_to_brain_from_atlas`. Sensory regions that track the movie's audio and visuals should show the highest synchrony:
    """
    )
    return


@app.cell
def _(ATLAS, isc_values, roi_to_brain_from_atlas):
    from nilearn.image import math_img

    brain_mask = math_img("img > 0", img=ATLAS)  # binary mask defining the output grid
    isc_map = roi_to_brain_from_atlas(isc_values, atlas=ATLAS, source_mask=brain_mask)
    isc_map.plot(
        method="slices",
        title="Inter-subject correlation (pairwise, per region)",
        cmap="hot",
        colorbar=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Pairwise vs. leave-one-out

    The two summary statistics rank regions almost identically, but LOO values are systematically larger — each subject is compared against a less noisy group mean:
    """
    )
    return


@app.cell
def _(isc_data, isc_permutation_test, isc_values, np):
    import matplotlib.pyplot as plt

    loo = isc_permutation_test(
        isc_data,
        summary_statistic="leave-one-out",
        metric="median",
        n_permute=1000,
        random_state=0,
        progress_bar=False,
    )
    loo_values = np.asarray(loo["isc"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(isc_values, loo_values, alpha=0.7)
    lims = [
        min(isc_values.min(), loo_values.min()),
        max(isc_values.max(), loo_values.max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.set_xlabel("pairwise ISC")
    ax.set_ylabel("leave-one-out ISC")
    ax.set_title("Pairwise vs. leave-one-out (per region)")
    ax.legend()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Recap

    | Stage | What it does | Key API |
    |---|---|---|
    | Region timeseries | Extract region means per subject, stack to `(time, subjects, regions)` | `BrainData(func).extract_roi(atlas).T` |
    | Compute + test | Per-region ISC + permutation p-value | `isc_permutation_test(data, summary_statistic="pairwise", n_permute=)` |
    | Leave-one-out | Each subject vs. the group mean | `summary_statistic="leave-one-out"` |
    | Project to brain | Paint per-region values onto voxels | `roi_to_brain_from_atlas(values, atlas=, source_mask=)` |

    **Next steps**

    - [GLM analysis](workflows-01_glm.html) — model a known design instead of using subjects as each other's model.
    - [Encoding models](workflows-02_encoding.html) — predict brain activity from explicit stimulus features.
    - [Multivariate pattern analysis](workflows-03_mvpa.html) — decode conditions and compare representational geometry.
    """
    )
    return


if __name__ == "__main__":
    app.run()
