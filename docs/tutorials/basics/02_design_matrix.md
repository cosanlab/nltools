---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# DesignMatrix Basics

## Introduction

The `DesignMatrix` class is the core data structure in for working with csv/tsv/dataframes that capture your experimental design (e.g. GLM analysis) or voxel-wise model (e.g. encoding models, group-analysis). `DesignMatrix` is backed by `polars` internally for fast operations, but accepts pandas DataFrames, dicts, and numpy arrays as input.

## Basics 

Lets build a small toy design matrix to learn the basics.

```{code-cell} python3
from nltools.data import DesignMatrix
import numpy as np

# Create a sample design matrix
dm = DesignMatrix(np.array([
                            [0,0,0,0],
                            [0,0,0,0],
                            [1,0,0,0],
                            [1,0,0,0],
                            [0,0,0,0],
                            [0,1,0,0],
                            [0,1,0,0],
                            [0,0,0,0],
                            [0,0,1,0],
                            [0,0,1,0],
                            [0,0,0,0],
                            [0,0,0,1],
                            [0,0,0,1],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0]
                            ]),
                            columns=['face_A','face_B','house_A','house_B'],
                            sampling_freq = 0.5, # 2s TR
                            )
```

`DesignMatrix` works just like a `polars` Dataframe so you can use familiar methods to inspect the data: `.head()`, `.tail()`, etc. 

```{code-cell} python3
dm
```

Get the first few rows:

```{code-cell} python3
dm.head()
```

Or specific columns:

```{code-cell} python3
dm.select('face_A', 'face_B').tail()
```

You can also visualize the design matrix as an SPM-style heatmap, where rows are time-points and columns are regressors:

```{code-cell} python3
dm.plot();
```

## HRF Convolution

The hemodynamic response function (HRF) models the sluggish BOLD signal response to neural activity. You can use `.convolve()` to apply this function to all your columns of interest. Convolved columns are renamed with a `_c0` suffix (and `_c1`, `_c2`, … if you pass multiple kernels) so callers can refer to them deterministically. Notice how the columns are delayed in time:

```{code-cell} python3
dm.convolve().plot();
```

Use `.plot(method='timeseries')` to view regressor time courses as lines. Passing the same `ax` to a second call overlays the convolved version on top of the original:

```{code-cell} python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
dm.plot(method='timeseries', columns=['face_A'], ax=ax)
dm.convolve().plot(method='timeseries', columns=['face_A_c0'], ax=ax);
```

## Creating Drift Regressors

`DesignMatrix` supports two equivalent approaches for creating "nuisance" regressors to capture low-frequency signals for use in GLM analysis: `.add_poly()` and `.add_dct_basis()`. 

### Polynomials

Legendre polynomials capture low-frequency trends defined by order:

- 0 = intercept
- 1 = linear trend
- 2 = quadratic trend, etc.

```{code-cell} python3
# Add up to 4th-order polynomials
dm.add_poly(order=4).plot();
```

### DCT High-Pass Filter

An alternative to polynomials, commonly used in SPM is a set of discrete-cosine filters. The `duration` parameter sets the high-pass cutoff (in seconds).

```{code-cell} python3
# Using a 20s cut-off is equivalent to above for this design
dm.add_dct_basis(duration=20).plot();
```

## Multicollinearity Diagnostics



### Cleaning Correlated Columns

It's **incredibly** important in classic GLM analysis to ensure don't have excessive multi-collinearity in your design matrix to ensure stable voxel beta-estimates. `DesignMatrix` offers a few tools to help: `.vif()` and `.clean()`

### Variance Inflation Factor (VIF)

VIF measures how much each regressor's variance is inflated by correlation with other regressors with values >= 5 classically cause for caution:

```{code-cell} python3
dm.vif()
```

We can also visualize a correlation matrix of the columns with `.plot(method='corr')`:

```{code-cell} python
dm.plot(method='corr');
```

`.corr()` itself returns an nltools `Adjacency` (a similarity matrix carrying the column labels), so you can feed it to any of the `Adjacency` tools:

```{code-cell} python
dm.corr()
```

Let's see a degenerate design:

```{code-cell} python
# Just duplicate the design
dm2 = dm.copy()
dm2.columns = ['car_A', 'car_B', 'dog_A', 'dog_B']

# And horizontally/column-wise append it (axis = 1)
duplicated_dm = dm.append(dm2, axis=1)
duplicated_dm.plot();
```

Columns are perfectly correlated:

```{code-cell} python
duplicated_dm.plot(method='corr');
```

And we can't even compute VIFs (they're essentially infinite):

```{code-cell} python
duplicated_dm.vif()
```

Instead we can use `.clean()` to automatically removes columns with correlations above a threshold:

```{code-cell} python3
duplicated_dm.clean(thresh=.99).plot();
```

## Combining Runs

We saw `.append()` above which can be used to combined multiple `DesignMatrix`. Use `.append(axis=0)` will combine vertically/row-wise for stacking design matrices across runs. Polynomial columns are automatically separated per run with `keep_separate=True` (default):

```{code-cell} python3
# Create two "runs"
run1 = dm.copy()
run2 = dm.copy()

# Drift for run 1
run1 = run1.add_poly(order=2)
# Drift for run 2
run2 = run2.add_poly(order=2)

# Append
combined = run1.append(run2, axis=0)
combined.plot();
```

Notice how all `poly` columns are kept separated but `face` and `house` regressors have been stacked so that a single estimate is computed across runs.

## Mixing Task Regressors with External Confounds

Real GLM workflows usually combine HRF-convolved task regressors with confound regressors that come from external preprocessing — head motion parameters, spike regressors, CSF/WM signals, physio. The canonical pattern is `.append(axis=1)`: it accepts a `DesignMatrix` *or* a raw pandas / polars DataFrame, automatically marks the appended columns as confounds (so they're skipped by `.convolve()` and kept separate per run on a later vertical append), and merges the `convolved` / `confounds` metadata correctly.

```{code-cell} python3
import pandas as pd

# 1. Convolve task regressors. Convolved columns get a `_c0` suffix; .convolved tracks them.
dm_task = dm.convolve()
print(dm_task)
```

```{code-cell} python3
# 2. Confound regressors typically arrive as pandas DataFrames from your preprocessing pipeline.
n_tr = dm_task.shape[0]
rng = np.random.default_rng(0)
motion = pd.DataFrame(
    rng.normal(size=(n_tr, 6)),
    columns=[f'motion_{ax}' for ax in ['tx','ty','tz','rx','ry','rz']],
)
csf = pd.DataFrame({'csf': rng.normal(size=n_tr)})
spikes = pd.DataFrame({f'spike_{i}': (np.arange(n_tr) == i*5).astype(float) for i in range(2)})

# 3. Append them all at once, then add drift terms. No `pd.concat` round-trip needed —
#    raw DataFrames are auto-wrapped and their columns are tracked as confounds.
dm_full = dm_task.append([motion, csf, spikes], axis=1).add_poly(order=2)
print(dm_full)
```

`dm_full.convolved` records the HRF-convolved task regressors, `dm_full.confounds` records the motion / spike / CSF / drift columns. Both are managed by `.convolve()` / `.append()` / `.add_poly()`; they're read-only properties — pass `convolved=` or `confounds=` to the constructor if you ever need to set initial state directly.

```{code-cell} python3
dm_full.plot();
```

If your confounds are already a `DesignMatrix` (e.g. you built them with `.add_poly()` or read them in via the file-path constructor), pass them in the same way — `as_confounds=True` is the explicit knob to mark a `DesignMatrix`'s columns as confounds even when its own `confounds` list is empty:

```{code-cell} python3
motion_dm = DesignMatrix(motion, sampling_freq=dm.sampling_freq)
dm_full2 = dm_task.append(motion_dm, axis=1, as_confounds=True)
print(dm_full2.confounds)
```
