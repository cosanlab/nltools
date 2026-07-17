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
# DesignMatrix basics — runs entirely in the browser via marimo + Pyodide.
# Source of truth for the docs tutorial; exported to WASM by
# scripts/build_marimo_wasm.py. `nltools` is injected at build time (file:// wheel
# for --execute) and micropip-installed from a hosted URL in the browser.

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # DesignMatrix Basics

        The `DesignMatrix` class is the core data structure for working with
        csv/tsv/dataframes that capture your experimental design (e.g. a GLM analysis) or
        a voxel-wise model (e.g. encoding models, group analysis). It's backed by `polars`
        internally for fast operations but accepts pandas DataFrames, dicts, and numpy
        arrays as input.

        /// admonition | Running live in your browser
        This page **is** a running notebook — the cells below execute in a Pyodide kernel
        inside the page. The first load boots the kernel and installs the scientific
        stack (about a minute; cached afterwards). Edit any cell and re-run to explore.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import sys

    IN_WASM = sys.platform == "emscripten"
    return (IN_WASM,)


@app.cell(hide_code=True)
async def _(IN_WASM):
    # In-browser only: install the nltools dev wheel (the PyPI stack is micropip-installed
    # from the PEP 723 header automatically). Resolve against the shared worker origin.
    if IN_WASM:
        import micropip
        import js

        _ = await micropip.install(
            js.location.origin + "__NLTOOLS_WHEEL_URL__", deps=False
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Basics

        Let's build a small toy design matrix to learn the basics.
        """
    )
    return


@app.cell
def _():
    from nltools.data import DesignMatrix
    import numpy as np

    # A toy blocked design: 4 conditions, 22 TRs, 2s TR (sampling_freq = 0.5 Hz)
    dm = DesignMatrix(
        np.array(
            [
                [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0],
                [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1],
                [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                [0, 0, 0, 0], [0, 0, 0, 0],
            ]
        ),
        columns=["face_A", "face_B", "house_A", "house_B"],
        sampling_freq=0.5,
    )
    return DesignMatrix, dm, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "`DesignMatrix` behaves like a `polars` DataFrame, so familiar methods work — "
        "`.head()`, `.tail()`, `.select()`, etc."
    )
    return


@app.cell
def _(dm):
    dm
    return


@app.cell
def _(dm):
    # First few rows
    dm.head()
    return


@app.cell
def _(dm):
    # Specific columns
    dm.select("face_A", "face_B").tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "Visualize it as an SPM-style heatmap — rows are time-points, columns are "
        "regressors:"
    )
    return


@app.cell
def _(dm):
    dm.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## HRF convolution

        The hemodynamic response function (HRF) models the sluggish BOLD response to
        neural activity. `.convolve()` applies it to your task columns, renaming the
        convolved columns with a `_c0` suffix (`_c1`, `_c2`, … for multiple kernels) so
        they can be referenced deterministically. Notice how the regressors are delayed
        and smeared in time:
        """
    )
    return


@app.cell
def _(dm):
    dm.convolve().plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "`.plot(method='timeseries')` draws regressors as line plots. Passing the same "
        "`ax` to a second call overlays the convolved version on the original:"
    )
    return


@app.cell
def _(dm):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(figsize=(8, 4))
    dm.plot(method="timeseries", columns=["face_A"], ax=_ax)
    dm.convolve().plot(method="timeseries", columns=["face_A_c0"], ax=_ax)
    _fig
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating drift regressors

        `DesignMatrix` offers two equivalent ways to add low-frequency "nuisance"
        regressors for a GLM: `.add_poly()` and `.add_dct_basis()`.

        ### Polynomials

        Legendre polynomials capture low-frequency trends by order — 0 = intercept,
        1 = linear, 2 = quadratic, and so on:
        """
    )
    return


@app.cell
def _(dm):
    # Up to 4th-order polynomials
    dm.add_poly(order=4).plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### DCT high-pass filter

        A common SPM alternative is a set of discrete-cosine filters. `duration` sets the
        high-pass cutoff in seconds:
        """
    )
    return


@app.cell
def _(dm):
    # A 20s cutoff is roughly equivalent to the polynomials above for this design
    dm.add_dct_basis(duration=20).plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Multicollinearity diagnostics

        In classic GLM analysis it's **essential** to keep excessive multicollinearity out
        of your design matrix, so voxel beta-estimates stay stable. `DesignMatrix` gives
        you two tools: `.vif()` and `.clean()`.

        ### Variance Inflation Factor (VIF)

        VIF measures how much each regressor's variance is inflated by correlation with the
        others; values ≥ 5 are classically cause for caution:
        """
    )
    return


@app.cell
def _(dm):
    dm.vif()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Visualize a correlation matrix of the columns with `.plot(method='corr')`:")
    return


@app.cell
def _(dm):
    dm.plot(method="corr")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "`.corr()` returns an nltools `Adjacency` (a labeled similarity matrix), so you "
        "can hand it to any of the `Adjacency` tools:"
    )
    return


@app.cell
def _(dm):
    dm.corr()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Cleaning correlated columns

        Let's build a degenerate design by duplicating the columns:
        """
    )
    return


@app.cell
def _(dm):
    # Duplicate the design under new names, then append column-wise (axis=1)
    dm2 = dm.copy()
    dm2.columns = ["car_A", "car_B", "dog_A", "dog_B"]
    duplicated_dm = dm.append(dm2, axis=1)
    duplicated_dm.plot()
    return (duplicated_dm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("The duplicated columns are perfectly correlated:")
    return


@app.cell
def _(duplicated_dm):
    duplicated_dm.plot(method="corr")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("So the VIFs are essentially infinite:")
    return


@app.cell
def _(duplicated_dm):
    duplicated_dm.vif()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "`.clean()` automatically drops columns whose correlation exceeds a threshold:"
    )
    return


@app.cell
def _(duplicated_dm):
    duplicated_dm.clean(thresh=0.99).plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Combining runs

        `.append(axis=0)` stacks design matrices vertically — e.g. concatenating runs.
        Polynomial columns are kept separate per run by default (`keep_separate=True`),
        while task regressors are stacked so a single estimate is computed across runs:
        """
    )
    return


@app.cell
def _(dm):
    # Two "runs", each with its own drift terms
    run1 = dm.copy().add_poly(order=2)
    run2 = dm.copy().add_poly(order=2)
    combined = run1.append(run2, axis=0)
    combined.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mixing task regressors with external confounds

        Real GLM workflows combine HRF-convolved task regressors with confound regressors
        from preprocessing — head motion, spike regressors, CSF/WM signals, physio. The
        canonical pattern is `.append(axis=1)`: it accepts a `DesignMatrix` *or* a raw
        pandas/polars DataFrame, automatically marks the appended columns as confounds
        (so `.convolve()` skips them and they stay separate per run on a later vertical
        append), and merges the `convolved`/`confounds` metadata correctly.
        """
    )
    return


@app.cell
def _(dm):
    # 1. Convolve task regressors — convolved columns get a `_c0` suffix; `.convolved` tracks them
    dm_task = dm.convolve()
    print(dm_task)
    return (dm_task,)


@app.cell
def _(dm_task, np):
    import pandas as pd

    # 2. Confounds typically arrive as pandas DataFrames from your preprocessing pipeline
    _n_tr = dm_task.shape[0]
    _rng = np.random.default_rng(0)
    motion = pd.DataFrame(
        _rng.normal(size=(_n_tr, 6)),
        columns=[f"motion_{ax}" for ax in ["tx", "ty", "tz", "rx", "ry", "rz"]],
    )
    csf = pd.DataFrame({"csf": _rng.normal(size=_n_tr)})
    spikes = pd.DataFrame(
        {f"spike_{i}": (np.arange(_n_tr) == i * 5).astype(float) for i in range(2)}
    )
    return csf, motion, spikes


@app.cell
def _(csf, dm_task, motion, spikes):
    # 3. Append them all at once, then add drift. Raw DataFrames are auto-wrapped and
    #    their columns tracked as confounds — no pd.concat round-trip needed.
    dm_full = dm_task.append([motion, csf, spikes], axis=1).add_poly(order=2)
    print(dm_full)
    return (dm_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "`dm_full.convolved` records the HRF-convolved task regressors; "
        "`dm_full.confounds` records the motion / spike / CSF / drift columns. Both are "
        "managed by `.convolve()` / `.append()` / `.add_poly()` and are read-only "
        "properties (pass `convolved=` / `confounds=` to the constructor to set initial "
        "state directly)."
    )
    return


@app.cell
def _(dm_full):
    dm_full.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "If your confounds are already a `DesignMatrix`, pass them the same way — "
        "`as_confounds=True` is the explicit knob to mark its columns as confounds even "
        "when its own `confounds` list is empty:"
    )
    return


@app.cell
def _(DesignMatrix, dm, dm_task, motion):
    motion_dm = DesignMatrix(motion, sampling_freq=dm.sampling_freq)
    dm_full2 = dm_task.append(motion_dm, axis=1, as_confounds=True)
    print(dm_full2.confounds)
    return


if __name__ == "__main__":
    app.run()
