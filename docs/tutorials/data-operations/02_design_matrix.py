# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Working with DesignMatrix — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Working with DesignMatrix

    A `DesignMatrix` is a table that knows it describes a timeseries. It carries
    a sampling frequency, tracks which of its columns are HRF-convolved task
    regressors and which are nuisance confounds, and offers what a GLM needs:
    convolution, drift terms, run-wise concatenation and collinearity
    diagnostics. The layout is observations by features — TRs by regressors for
    a first-level analysis, participants by conditions for a second-level one.
    `polars` backs it: unknown attributes forward to the DataFrame underneath,
    so `select`, `filter` and `slice` work and hand back a `DesignMatrix`, and
    `.data` is that DataFrame.

    ## Build one by hand

    The constructor takes a dict of columns, a `polars` or `pandas` DataFrame, a
    NumPy array with `columns=`, or a file path. `sampling_freq` is in hertz —
    one over the TR — and `TR=` is the same number spelled the other way; pass
    exactly one of them. A toy blocked design below: four conditions, 22 TRs, a
    2 s TR, each condition on for two TRs.

    Printing reports the sampling frequency, the shape, and — once there are any
    — which columns are convolved and which are confounds. Indexing with one
    name gives a `polars` Series and with a list of names a `DesignMatrix`
    carrying the metadata along. Row selectors like `head` return a
    `DesignMatrix` too, so `.data` is how you look at the numbers.

    `plot` draws the SPM-style heatmap by default, TRs down and regressors
    across, each column rescaled by its L2 norm so regressors of different
    native magnitude stay comparable (`rescale=False` keeps the raw range).
    `method='corr'` draws the column correlation matrix instead, and
    `method='timeseries'` draws line plots. Keywords the signature does not name
    go to `seaborn.heatmap` or to matplotlib. `corr` returns that correlation
    matrix as an `Adjacency` for anything the heatmap does not cover:
    """)
    return


@app.cell
def _():
    from nltools.data import DesignMatrix

    _blocks = {"face_A": 2, "face_B": 5, "house_A": 8, "house_B": 11}
    dm = DesignMatrix(
        {
            name: [1.0 if onset <= tr < onset + 2 else 0.0 for tr in range(22)]
            for name, onset in _blocks.items()
        },
        sampling_freq=0.5,
    )

    print(dm)
    return DesignMatrix, dm


@app.cell
def _(dm):
    print(f"dm['face_A'] is a {type(dm['face_A']).__name__}")
    print(f"dm[['face_A', 'face_B']] is a {type(dm[['face_A', 'face_B']]).__name__}")
    return


@app.cell
def _(dm):
    dm.head().data
    return


@app.cell
def _(dm):
    dm.plot(title="Toy blocked design")
    return


@app.cell
def _(dm):
    dm.plot(method="corr", title="Column correlations")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convolution

    The hemodynamic response function models the sluggish BOLD answer to neural
    activity. `convolve` replaces each task column with its convolved version
    and renames it `<column>_c0`: the source column is dropped and `.convolved`
    records the new name. Confound columns are skipped, so drift terms are left
    alone, and so is anything already in `.convolved`: convolving an HRF-shaped
    signal a second time describes nothing, so naming such a column in
    `columns=` raises rather than doing it.

    `kernel` defaults to `'glover'`; the other five names are `'glover_time'`,
    `'glover_dispersion'`, `'spm'`, `'spm_time'` and `'spm_dispersion'`. A named
    model hands each column to nilearn's `compute_regressor` as events — one per
    nonzero sample, each lasting one TR and scaled by that sample's value —
    convolved at 50x oversampling and resampled onto the TR grid, which is what
    a nilearn `FirstLevelModel` computes from the same events.

    Pass an array instead of a name for your own kernel: 1-D for a single
    kernel, applied with `numpy.convolve` and truncated back to the run length;
    2-D as samples by kernels, giving `_c0`, `_c1`, … per source column, which
    is how an FIR basis is written. Timing finer than a TR is gone by the time a
    column exists — for that, hand the events table to the constructor and let
    it convolve, as in [Read a design from files](#read-a-design-from-files):
    """)
    return


@app.cell
def _(dm):
    convolved = dm.convolve()

    print(convolved)
    return (convolved,)


@app.cell
def _(dm):
    import numpy as np

    _decay = np.exp(-np.arange(0, 24, 2) / 6.0)
    print(dm.convolve(kernel=_decay).columns)
    print(dm.convolve(kernel=np.column_stack([_decay, _decay[::-1]])).columns)
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One boxcar and its convolved version, drawn on the same axis:
    """)
    return


@app.cell
def _(convolved, dm):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(figsize=(8, 3))
    dm.plot(method="timeseries", columns=["face_A"], ax=_ax)
    convolved.plot(method="timeseries", columns=["face_A_c0"], ax=_ax, title="face_A")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Drift and baseline regressors

    Scanner signal drifts over a run, and a GLM that does not model the drift
    charges it to the task regressors. Two standard families do the job, and
    both mark their columns as confounds so `convolve` skips them.

    `add_poly` adds Legendre polynomials evaluated over -1 to 1: order 0 is the
    intercept, 1 a linear trend, 2 a quadratic, and `include_lower=True` (the
    default) adds every order up to the one you ask for. `add_dct_basis` adds a
    discrete cosine basis that acts as a high-pass filter; `duration` is the
    cutoff period in seconds, 180 by default, and together with the run length
    it fixes how many bases you get. The basis omits the constant per SPM
    convention, so nltools re-adds it unless `include_constant=False`. Every
    column nltools generates lives in a reserved `.nl_` namespace —
    `.nl_poly_2`, `.nl_cosine_1` — so your own regressors can be named anything
    without colliding.

    Pick one family, not both. Polynomials are the simpler choice for a short
    run; the cosine basis states its cutoff in seconds, which is easier to match
    against a preprocessing pipeline that already high-pass filtered the data at
    a stated period. Using both makes columns that say the same thing, and
    [Diagnostics](#diagnostics) shows what that costs:
    """)
    return


@app.cell
def _(dm):
    poly_drift = dm.add_poly(order=2)

    print(poly_drift)
    return (poly_drift,)


@app.cell
def _(poly_drift):
    poly_drift.plot(title="Legendre polynomials to order 2")
    return


@app.cell
def _(dm):
    cosine_drift = dm.add_dct_basis(duration=20)

    print(cosine_drift)
    return (cosine_drift,)


@app.cell
def _(cosine_drift):
    cosine_drift.plot(title="Cosine basis, 20 s cutoff")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read a design from files

    A `.csv` or `.tsv` path is read as a BIDS events file when it carries
    `onset` and `duration` columns, and as a plain table otherwise. An events
    file needs `run_length`, the number of TRs in the run, alongside the
    sampling frequency. Each `trial_type` becomes one regressor, HRF-convolved
    by the constructor with the model named in `hrf_model` — `'glover'` by
    default, `None` for raw boxcars. Convolving here keeps onsets that fall
    between TRs, which sampling onto the grid first would quantize away.

    `events_to_dm` is that conversion without the convolution, for events
    already in memory. It returns a `polars` DataFrame of boxcars, one column
    per `trial_type` and no intercept; `add_poly(0)` is where an intercept comes
    from. The example file below holds 39 events across 13 conditions:
    """)
    return


@app.cell
def _(DesignMatrix):
    from pathlib import Path

    import polars as pl

    from nltools.datasets import get_resource_path
    from nltools.io import events_to_dm

    resources = Path(get_resource_path())
    events_file = resources / "onsets_example.csv"
    events = pl.read_csv(events_file)

    events.head(3)
    return events, events_file, events_to_dm, pl, resources


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both conversions, on that file:
    """)
    return


@app.cell
def _(DesignMatrix, events, events_file, events_to_dm):
    boxcars = events_to_dm(events, run_length=160, sampling_freq=0.5)
    run_task = DesignMatrix(events_file, run_length=160, sampling_freq=0.5)

    print(f"events_to_dm gives a {type(boxcars).__name__} of shape {boxcars.shape}")
    print(run_task)
    return (run_task,)


@app.cell
def _(run_task):
    run_task.plot(title="One run of task regressors")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every other table is read as it stands, one row per TR. `run_length='infer'`
    accepts whatever row count the file has; an events file rejects it, since
    its rows are events rather than TRs. Empty cells arrive as nulls, so a
    motion file whose first row has no derivatives needs `fillna`. `vmin` and
    `vmax` reach `seaborn.heatmap` and set a color range these small regressors
    are visible in:
    """)
    return


@app.cell
def _(DesignMatrix, resources):
    confounds_file = resources / "covariates_example.csv"
    run_confounds = DesignMatrix(
        confounds_file, run_length="infer", sampling_freq=0.5
    ).fillna(0)

    print(run_confounds)
    return confounds_file, run_confounds


@app.cell
def _(run_confounds):
    run_confounds.plot(vmin=-1, vmax=1, title="One run of motion confounds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Combining runs

    `append(axis=0)` stacks runs. Task regressors stack with them, so one
    coefficient is estimated across the whole experiment, but a run's baseline
    and drift cannot be shared: `keep_separate=True`, the default, renames each
    confound into the run-separated part of the reserved namespace —
    `.nl_r0_poly_0`, `.nl_r1_poly_0` — giving every run its own. Name a column
    in `unique_cols` to separate it the same way — a leading or trailing `*` is
    a wildcard, so `'house*'` covers both house conditions.

    Add drift terms to each run before stacking. `add_poly` and `add_dct_basis`
    refuse a design that already carries run-separated drift, because a global
    trend on top of per-run ones is ambiguous:
    """)
    return


@app.cell
def _(convolved):
    single_run = convolved.add_poly(order=1)
    two_runs = single_run.append(single_run, axis=0)
    split_houses = single_run.append(single_run, axis=0, unique_cols=["house*"])

    print(two_runs)
    print(f"with unique_cols=['house*']: {split_houses.columns}")
    return single_run, two_runs


@app.cell
def _(two_runs):
    two_runs.plot(title="Two runs, separate baselines")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `append(axis=1)` joins columns side by side instead. `as_confounds=True`
    marks everything the other matrix contributes as a confound, which is what
    makes `convolve` skip those columns and a later vertical append separate
    them per run; a raw `polars` DataFrame appended this way is marked
    automatically. `.confounds` and `.convolved` are read-only, managed by these
    methods — the constructor's `confounds=` and `convolved=` are how you set
    initial state.

    The whole recipe, once. Per run: read the events, read the confounds, fill
    their nulls, add that run's drift terms, join the two side by side as
    confounds, and stack the result onto the design so far. Four runs of the
    same two example files stand in for a real experiment. Both drift families
    go in, against the advice above, so that [Diagnostics](#diagnostics) has a
    real redundancy to find. `include_constant=False` keeps the cosine basis
    from adding a second intercept on top of the one `add_poly(1)` already
    contributed; asking for one anyway warns and skips it:
    """)
    return


@app.cell
def _(DesignMatrix, confounds_file, events_file):
    all_runs = DesignMatrix(sampling_freq=0.5)

    for _run in range(4):
        _task = DesignMatrix(events_file, run_length=160, sampling_freq=0.5)
        _confounds = (
            DesignMatrix(confounds_file, run_length="infer", sampling_freq=0.5)
            .fillna(0)
            .add_poly(order=1)
            .add_dct_basis(include_constant=False)
        )
        _full = _task.append(_confounds, axis=1, as_confounds=True)
        all_runs = all_runs.append(_full, axis=0)

    print(f"{all_runs.shape[0]} TRs x {all_runs.shape[1]} columns")
    print(f"task: {all_runs.columns[:2]} ... {all_runs.columns[12]}")
    print(f"run 0 confounds: {all_runs.confounds[:2]} ... {all_runs.confounds[29]}")
    return (all_runs,)


@app.cell
def _(all_runs):
    all_runs.plot(vmin=-1, vmax=1, title="Four runs")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reading that heatmap left to right: the thirteen conditions, stacked across
    all four runs, then one block per run — that run's motion confounds, then
    its drift and baseline terms.

    ## Diagnostics

    Two columns saying nearly the same thing cannot be estimated stably. `vif`
    reports each regressor's variance inflation factor — the diagonal of the
    inverted correlation matrix, the definition R and MATLAB use — over the
    non-confound columns by default. Values at or above 5 are the classic
    warning sign. It returns a plain array in column order, and `None` when the
    design is singular outright.

    `clean` walks the columns in order and drops the second of a pair whose
    absolute correlation reaches `thresh`, 0.95 by default, keeping the first.
    Planting a near-copy of one task regressor shows both at work: the copy and
    the original come back with VIFs above a thousand, and `clean` removes the
    copy. It also removes each run's lowest-frequency cosine, a near duplicate
    of that run's linear polynomial — the price of using both drift families
    above:
    """)
    return


@app.cell
def _(all_runs, np, pl):
    _rng = np.random.default_rng(0)
    planted = all_runs.with_columns(
        CoachTaylor_copy=pl.col("CoachTaylor_c0")
        + _rng.normal(0, 0.01, len(all_runs))
    )
    task_columns = [c for c in planted.columns if c not in planted.confounds]

    for _name, _value in sorted(
        zip(task_columns, planted.vif()), key=lambda pair: -pair[1]
    )[:3]:
        print(f"VIF {_value:8.1f}  {_name}")
    return (planted,)


@app.cell
def _(planted):
    cleaned = planted.clean()

    print(f"{planted.shape[1]} columns -> {cleaned.shape[1]} after clean()")
    print(f"dropped: {[c for c in planted.columns if c not in cleaned.columns]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimating the model

    A finished design meets the data in `BrainData.fit`. The 22-TR run below is
    simulated — a sphere whose signal follows the `face_A` regressor, plus noise
    — so the design and the data belong to each other. `fit(model='glm',
    X=design)` leaves a `FitResult` on `.model`: `betas`, one map per design
    column in column order, alongside `predicted`, `residual`, `r2` and the
    design itself. `compute_contrasts`
    takes a string naming design columns and returns the effect map;
    `inference=True` returns a `ContrastResult` carrying the t, z and one-sided
    p maps alongside it. The univariate GLM tutorial takes a real dataset
    through the whole first-level workflow; this is only the handoff:
    """)
    return


@app.cell
def _(single_run):
    from nltools.data import Simulator

    brain = Simulator(random_state=0).create_data(
        single_run["face_A_c0"].to_list(), 1.0, radius=10
    )
    brain.fit(model="glm", X=single_run)

    print(brain)
    print(f"{brain.model.betas.shape[0]} beta maps for {single_run.shape[1]} columns")
    return (brain,)


@app.cell
def _(brain):
    brain.compute_contrasts("face_A_c0").plot(title="face_A effect")
    return


if __name__ == "__main__":
    app.run()
