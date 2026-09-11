# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Design matrices — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Design Matrices

    `DesignMatrix` is a dataframe that knows it describes a timeseries. It carries
    a sampling frequency, tracks which of its columns are convolved task
    regressors and which are confounds, and offers the operations a GLM needs:
    HRF convolution, drift terms, run-wise concatenation, and collinearity
    diagnostics.

    The layout is the usual machine-learning one, observations by features: TRs by
    conditions plus nuisance regressors for a first-level analysis, or
    participants by conditions for a second-level one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build one by hand

    A toy design for one participant: 22 TRs at 1.5 s, four stimulus conditions,
    each on for 2 TRs with a TR of rest between events. `sampling_freq` is in
    hertz, so it is `1 / TR`.
    """)
    return


@app.cell
def _():
    import numpy as np

    from nltools.data import DesignMatrix

    TR = 1.5

    dm = DesignMatrix(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
        columns=["face_A", "face_B", "house_A", "house_B"],
        sampling_freq=1.0 / TR,
    )
    dm.head()
    return DesignMatrix, dm


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Printing it shows what the object knows about itself: sampling frequency,
    shape, and — once there are any — which columns are convolved and which are
    confounds.
    """)
    return


@app.cell
def _(dm):
    print(dm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `.plot()` draws the SPM/FSL-style heatmap, TRs down and regressors across:
    """)
    return


@app.cell
def _(dm):
    dm.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Drift and baseline regressors

    ### Legendre polynomials

    `add_poly` adds orthogonal Legendre polynomials on the interval -1 to 1, the
    same convention other packages use. Order 2 with `include_lower=True` gives an
    intercept, a linear trend and a quadratic trend.
    """)
    return


@app.cell
def _(dm):
    dm_with_nuisance = dm.add_poly(2, include_lower=True)
    dm_with_nuisance.plot()
    return (dm_with_nuisance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Three columns appeared, and the object now lists them as confounds:
    """)
    return


@app.cell
def _(dm_with_nuisance):
    print(dm_with_nuisance)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Discrete cosine basis

    `add_dct_basis` is the other standard choice: a bank of cosine filters acting
    as a high-pass filter. `duration` sets the cutoff period in seconds and
    defaults to 180 s; 20 s suits this very short toy run.
    """)
    return


@app.cell
def _(dm):
    dm.add_dct_basis(duration=20).plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convolution

    `convolve` applies a hemodynamic response function to the task regressors and
    skips the confound columns, so drift terms are left alone. The default kernel
    is a Glover HRF; pass a 1-D array for your own kernel, or a 2-D array to
    convolve with several at once. Convolved columns get a `_c0` suffix (`_c1`,
    `_c2`, … for further kernels) so you can reference them by name.
    """)
    return


@app.cell
def _(dm_with_nuisance):
    convolved = dm_with_nuisance.convolve()
    print(convolved)
    convolved.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `DesignMatrix` also upsamples, downsamples and z-scores; the
    [`DesignMatrix` reference](../../api/data/design_matrix.md) has the full list.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read a design from files

    ### From an onsets file

    Pass a path to a 2- or 3-column onsets file and `DesignMatrix` builds the
    regressors for you. `run_length` is the number of TRs and `TR` the repetition
    time. The constructor convolves by default; `hrf_model=None` gives raw boxcars
    instead, which is what you want when you plan to build interaction terms
    first. nltools ships an example file where each event lasts 10 s.
    """)
    return


@app.cell
def _(DesignMatrix):
    import os

    from nltools.utils import get_resource_path

    RUN_TR = 2.0
    onsets_file = os.path.join(get_resource_path(), "onsets_example.csv")

    onsets_dm = DesignMatrix(
        onsets_file, run_length=160, TR=RUN_TR, hrf_model=None
    ).add_poly(1)
    onsets_dm.plot()
    return RUN_TR, get_resource_path, onsets_file, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From any table

    Anything `polars` or `pandas` can read becomes a `DesignMatrix` — you only
    have to supply the sampling frequency. Here is a motion-realignment file from
    a preprocessing pipeline. `.plot()` forwards keyword arguments to seaborn, so
    `vmin`/`vmax` rescale the color range to the size of these regressors.
    """)
    return


@app.cell
def _(DesignMatrix, RUN_TR, get_resource_path, os):
    import pandas as pd

    covariates_file = os.path.join(get_resource_path(), "covariates_example.csv")
    covariates = DesignMatrix(pd.read_csv(covariates_file), sampling_freq=1.0 / RUN_TR)
    covariates.plot(vmin=-1, vmax=1)
    return covariates_file, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Combining runs

    ### Stacking vertically

    An experiment with several runs needs one design per run, stacked. Run
    differences have to be absorbed by run-specific baselines, so `append` keeps
    polynomial columns separate per run automatically and renames them to say
    which run they came from.
    """)
    return


@app.cell
def _(dm_with_nuisance):
    two_runs = dm_with_nuisance.append(dm_with_nuisance, axis=0)
    print(two_runs.columns)
    two_runs.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Keeping other columns separate

    Task regressors are stacked, so one coefficient is estimated across runs. To
    estimate a column per run instead, name it in `unique_cols`. A leading or
    trailing `*` is a wildcard, so `"house*"` separates both house conditions.
    """)
    return


@app.cell
def _(dm_with_nuisance):
    split_houses = dm_with_nuisance.append(
        dm_with_nuisance, axis=0, unique_cols=["house*"]
    )
    print(split_houses.columns)
    split_houses.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A realistic multi-run design

    Putting it together: for each run, read the onsets, read the confounds, add
    that run's drift terms, join the two side by side, then stack the runs while
    keeping the confounds run-specific. Four runs of the same example files stand
    in for a real experiment here.

    `add_dct_basis(include_constant=False)` is deliberate — `add_poly(1)` already
    contributed an intercept, and asking for a second one would be redundant.
    """)
    return


@app.cell
def _(DesignMatrix, RUN_TR, covariates_file, onsets_file, pd):
    all_runs = DesignMatrix(sampling_freq=1.0 / RUN_TR)

    for _run in range(4):
        # 1. Task regressors, HRF-convolved by the constructor
        _task = DesignMatrix(onsets_file, run_length=160, TR=RUN_TR)

        # 2. Confounds for this run, with drift and high-pass terms
        _confounds = DesignMatrix(
            pd.read_csv(covariates_file), sampling_freq=1.0 / RUN_TR
        ).fillna(0)
        _confounds = _confounds.add_poly(1).add_dct_basis(include_constant=False)

        # 3. Join them side by side, then stack onto the master design
        _full = _task.append(_confounds, axis=1)
        all_runs = all_runs.append(_full, axis=0, unique_cols=list(_confounds.columns))

    print(all_runs)
    all_runs.plot(vmin=-1, vmax=1)
    return (all_runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Reading the heatmap left to right: conditions of interest, stacked across all
    four runs; then each run's own confounds; then each run's drift and baseline
    terms.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnostics

    A design with two columns that say nearly the same thing cannot be estimated
    stably. `clean` drops any column correlated at or above `thresh` (0.95 by
    default) with a column before it.
    """)
    return


@app.cell
def _(all_runs):
    cleaned = all_runs.clean()
    print(f"{all_runs.shape[1]} columns -> {cleaned.shape[1]} after cleaning")
    cleaned.plot(vmin=-1, vmax=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The dropped columns are polynomials and cosine filters that duplicate one
    another. In practice pick one family or the other; both were used above to
    show the diagnostic working.

    `vif()` is the companion check: it reports each regressor's variance inflation
    factor, and values at or above 5 are the classic warning sign.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimating the model

    A finished design becomes the `X` of a `BrainData` object holding the EPI data
    for those runs, and the regression is one call. This is not run here because
    the notebook has no four-run dataset attached; the
    [GLM tutorial](../workflows/01_glm.md) works through a real one.

    ```python
    from nltools.data import BrainData

    brains = BrainData(["run_1.nii.gz", "run_2.nii.gz", "run_3.nii.gz", "run_4.nii.gz"])
    brains.X = cleaned

    results = brains.regress()
    ```

    That produces a beta, t, and p image per column of the design matrix.
    """)
    return


if __name__ == "__main__":
    app.run()
