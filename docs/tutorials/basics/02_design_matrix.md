---
# AUTO-GENERATED from 02_design_matrix.py by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe docs-generate`.
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# DesignMatrix Basics

:::{tip} Interactive version
The outputs below are pre-computed. [**Open this tutorial as a live notebook →**](/tutorials/basics-02_design_matrix.html) to run and edit every cell in your browser (via marimo + WebAssembly).
:::

The `DesignMatrix` class is the core data structure for working with
csv/tsv/dataframes that capture your experimental design (e.g. a GLM analysis) or
a voxel-wise model (e.g. encoding models, group analysis). It's backed by `polars`
internally for fast operations but accepts pandas DataFrames, dicts, and numpy
arrays as input.

```{code-cell} python3
:tags: [remove-input]
import sys

IN_WASM = sys.platform == "emscripten"
```

```{code-cell} python3
:tags: [remove-input]
# In-browser only: install nltools + its full runtime stack before any nltools import
# runs, then hand `wasm_ready` to every nltools-importing cell to force ordering. We
# can't rely on marimo's PEP 723 header auto-install alone: it races cell execution and
# marimo never re-runs a cell that already failed with ModuleNotFoundError. Resolve the
# wheel against the shared worker origin.
wasm_ready = True
if IN_WASM:
    import asyncio

    import micropip
    import js

    async def _pip(reqs, **kw):
        # Install packages ONE AT A TIME instead of a single concurrent
        # micropip.install([...]) call. The big concurrent batch download
        # occasionally returns a truncated wheel (BadZipFile); micropip then
        # caches the corrupt bytes so an in-session retry keeps failing — and
        # marimo never re-runs an errored cell, permanently bricking the
        # page. Sequential installs keep peak download concurrency low and
        # sidestep the corruption; a per-package retry still rides out
        # ordinary network blips. (see nltools#455 investigation)
        items = [reqs] if isinstance(reqs, str) else list(reqs)
        for _item in items:
            for _attempt in range(3):
                try:
                    await micropip.install(_item, **kw)
                    break
                except Exception:  # noqa: BLE001
                    if _attempt == 2:
                        raise
                    await asyncio.sleep(0.75 * (_attempt + 1))

    # Install the stack UNPINNED so micropip takes Pyodide's bundled builds (pinning to
    # nltools' host versions, e.g. joblib>=1.5.3, fails against Pyodide's bundled
    # joblib). nilearn is the exception: 0.14+ needs packaging>=26 (absent in Pyodide
    # 0.27.7), so pin the last 0.13.x. numpy/scipy/pandas/sklearn/matplotlib come in
    # transitively at their bundled versions.
    await _pip(
        [
            "nibabel",
            "nilearn==0.13.1",
            "seaborn",
            "polars",
            "pynv",
            "huggingface-hub",
            "anywidget",
        ]
    )
    # deps=False installs the wheel without re-checking nltools' own version pins.
    await _pip(
        js.location.origin + "__NLTOOLS_WHEEL_URL__", deps=False
    )
```

## Basics

Let's build a small toy design matrix to learn the basics.

```{code-cell} python3
_ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
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
```

`DesignMatrix` behaves like a `polars` DataFrame, so familiar methods work — `.head()`, `.tail()`, `.select()`, etc.

```{code-cell} python3
dm
```

```{code-cell} python3
# First few rows
dm.head()
```

```{code-cell} python3
# Specific columns
dm.select("face_A", "face_B").tail()
```

Visualize it as an SPM-style heatmap — rows are time-points, columns are regressors:

```{code-cell} python3
dm.plot()
```

## HRF convolution

The hemodynamic response function (HRF) models the sluggish BOLD response to
neural activity. `.convolve()` applies it to your task columns, renaming the
convolved columns with a `_c0` suffix (`_c1`, `_c2`, … for multiple kernels) so
they can be referenced deterministically. Notice how the regressors are delayed
and smeared in time:

```{code-cell} python3
dm.convolve().plot()
```

`.plot(method='timeseries')` draws regressors as line plots. Passing the same `ax` to a second call overlays the convolved version on the original:

```{code-cell} python3
import matplotlib.pyplot as plt

_fig, _ax = plt.subplots(figsize=(8, 4))
dm.plot(method="timeseries", columns=["face_A"], ax=_ax)
dm.convolve().plot(method="timeseries", columns=["face_A_c0"], ax=_ax)
_fig
```

## Creating drift regressors

`DesignMatrix` offers two equivalent ways to add low-frequency "nuisance"
regressors for a GLM: `.add_poly()` and `.add_dct_basis()`.

### Polynomials

Legendre polynomials capture low-frequency trends by order — 0 = intercept,
1 = linear, 2 = quadratic, and so on:

```{code-cell} python3
# Up to 4th-order polynomials
dm.add_poly(order=4).plot()
```

### DCT high-pass filter

A common SPM alternative is a set of discrete-cosine filters. `duration` sets the
high-pass cutoff in seconds:

```{code-cell} python3
# A 20s cutoff is roughly equivalent to the polynomials above for this design
dm.add_dct_basis(duration=20).plot()
```

## Multicollinearity diagnostics

In classic GLM analysis it's **essential** to keep excessive multicollinearity out
of your design matrix, so voxel beta-estimates stay stable. `DesignMatrix` gives
you two tools: `.vif()` and `.clean()`.

### Variance Inflation Factor (VIF)

VIF measures how much each regressor's variance is inflated by correlation with the
others; values ≥ 5 are classically cause for caution:

```{code-cell} python3
dm.vif()
```

Visualize a correlation matrix of the columns with `.plot(method='corr')`:

```{code-cell} python3
dm.plot(method="corr")
```

`.corr()` returns an nltools `Adjacency` (a labeled similarity matrix), so you can hand it to any of the `Adjacency` tools:

```{code-cell} python3
dm.corr()
```

### Cleaning correlated columns

Let's build a degenerate design by duplicating the columns:

```{code-cell} python3
# Duplicate the design under new names, then append column-wise (axis=1)
dm2 = dm.copy()
dm2.columns = ["car_A", "car_B", "dog_A", "dog_B"]
duplicated_dm = dm.append(dm2, axis=1)
duplicated_dm.plot()
```

The duplicated columns are perfectly correlated:

```{code-cell} python3
duplicated_dm.plot(method="corr")
```

So the VIFs are essentially infinite:

```{code-cell} python3
duplicated_dm.vif()
```

`.clean()` automatically drops columns whose correlation exceeds a threshold:

```{code-cell} python3
duplicated_dm.clean(thresh=0.99).plot()
```

## Combining runs

`.append(axis=0)` stacks design matrices vertically — e.g. concatenating runs.
Polynomial columns are kept separate per run by default (`keep_separate=True`),
while task regressors are stacked so a single estimate is computed across runs:

```{code-cell} python3
# Two "runs", each with its own drift terms
run1 = dm.copy().add_poly(order=2)
run2 = dm.copy().add_poly(order=2)
combined = run1.append(run2, axis=0)
combined.plot()
```

## Mixing task regressors with external confounds

Real GLM workflows combine HRF-convolved task regressors with confound regressors
from preprocessing — head motion, spike regressors, CSF/WM signals, physio. The
canonical pattern is `.append(axis=1)`: it accepts a `DesignMatrix` *or* a raw
pandas/polars DataFrame, automatically marks the appended columns as confounds
(so `.convolve()` skips them and they stay separate per run on a later vertical
append), and merges the `convolved`/`confounds` metadata correctly.

```{code-cell} python3
# 1. Convolve task regressors — convolved columns get a `_c0` suffix; `.convolved` tracks them
dm_task = dm.convolve()
print(dm_task)
```

```{code-cell} python3
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
```

```{code-cell} python3
# 3. Append them all at once, then add drift. Raw DataFrames are auto-wrapped and
#    their columns tracked as confounds — no pd.concat round-trip needed.
dm_full = dm_task.append([motion, csf, spikes], axis=1).add_poly(order=2)
print(dm_full)
```

`dm_full.convolved` records the HRF-convolved task regressors; `dm_full.confounds` records the motion / spike / CSF / drift columns. Both are managed by `.convolve()` / `.append()` / `.add_poly()` and are read-only properties (pass `convolved=` / `confounds=` to the constructor to set initial state directly).

```{code-cell} python3
dm_full.plot()
```

If your confounds are already a `DesignMatrix`, pass them the same way — `as_confounds=True` is the explicit knob to mark its columns as confounds even when its own `confounds` list is empty:

```{code-cell} python3
motion_dm = DesignMatrix(motion, sampling_freq=dm.sampling_freq)
dm_full2 = dm_task.append(motion_dm, axis=1, as_confounds=True)
print(dm_full2.confounds)
```
