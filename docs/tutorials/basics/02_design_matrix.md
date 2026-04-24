---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# DesignMatrix Basics

## Introduction

The `DesignMatrix` class is the core data structure in for working with csv/tsv/dataframes that capture your experimental design (e.g. GLM analysis) or voxel-wise model (e.g. encoding models, group-analysis). `DesignMatrix` is backed by `polars` internally for fast operations, but accepts pandas DataFrames, dicts, and numpy arrays as input.

Lets load an included dataset to inspect a pre-built `DesignMatrix`:

```{code-cell} python3
import matplotlib.pyplot as plt
from nltools.datasets import fetch_haxby
from nltools.data import DesignMatrix

# Get all subjects
brains, designs = fetch_haxby()

# Just a single subject
s1_brain, s1_design = brains[0], designs[0]

s1_design
```

`DesignMatrix` works just like a `polars` Dataframe so you can use familiar methods to inspect the data: `.head`, `.tail`, etc. 

```{code-cell} python3
s1_design.head()
```

## HRF Convolution

The hemodynamic response function (HRF) models the sluggish BOLD signal response to neural activity. In nltools, you create a boxcar stimulus timecourse and then convolve it with the canonical HRF.

```{code-cell} python3
# Define event timing
onsets = [20, 40, 60, 80, 100, 120]  # seconds
durations = [2] * len(onsets)         # seconds

# Create stimulus boxcar at TR resolution
stim = np.zeros(n_samples)
for onset, dur in zip(onsets, durations):
    start_idx = int(onset / TR)
    end_idx = int((onset + dur) / TR)
    if start_idx < n_samples:
        stim[start_idx:min(end_idx, n_samples)] = 1

# Convolve with canonical HRF
dm_hrf = DesignMatrix({"task": stim}, sampling_freq=sampling_freq)
dm_hrf = dm_hrf.convolve("hrf")
print(f"HRF-convolved: {dm_hrf.shape}")
```

```{code-cell} python3
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(stim)
axes[0].set_ylabel("Stimulus")
axes[0].set_title("Original Boxcar")

axes[1].plot(dm_hrf["task"].to_numpy())
axes[1].set_ylabel("BOLD Signal")
axes[1].set_xlabel("Volume Number")
axes[1].set_title("HRF-Convolved Regressor")

# Mark event onsets
for onset in onsets:
    for ax in axes:
        ax.axvline(onset / TR, color="r", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()
```

### Selective Convolution

When your design matrix has both task and nuisance columns, use the `columns` argument to convolve only task regressors:

```{code-cell} python3
dm_mixed = DesignMatrix(
    {"task": stim, "motion_x": np.random.randn(n_samples) * 0.5},
    sampling_freq=sampling_freq,
)
dm_mixed = dm_mixed.convolve("hrf", columns=["task"])
print("Only 'task' was convolved, 'motion_x' left unchanged")
```

### Parametric Modulation

Modulate a regressor by a trial-level variable (e.g., reaction time) to capture parametric effects:

```{code-cell} python3
param_onsets = [20, 40, 60, 80, 100]
reaction_times = [0.5, 0.7, 0.6, 0.9, 0.55]

# Create impulse regressors at event onsets
stim_main = np.zeros(n_samples)
stim_param = np.zeros(n_samples)

for onset, rt in zip(param_onsets, reaction_times):
    idx = int(onset / TR)
    if idx < n_samples:
        stim_main[idx] = 1
        stim_param[idx] = rt

dm_param = DesignMatrix(
    {"task_main": stim_main, "task_x_rt": stim_param},
    sampling_freq=sampling_freq,
)
dm_param = dm_param.convolve("hrf")

corr = np.corrcoef(
    dm_param["task_main"].to_numpy(), dm_param["task_x_rt"].to_numpy()
)[0, 1]
print(f"Correlation between main effect and parametric modulator: {corr:.3f}")
```

## Drift Regressors

fMRI data contains low-frequency drift that must be modeled or filtered out.

### Polynomial Drift

Legendre polynomials capture low-frequency trends. Order 0 = intercept, 1 = linear, 2 = quadratic, etc.

```{code-cell} python3
dm_drift = DesignMatrix({"placeholder": np.zeros(n_samples)}, sampling_freq=sampling_freq)
dm_drift = dm_drift.add_poly(order=3, include_lower=True)
dm_drift = dm_drift.drop("placeholder")
print(f"Polynomial columns: {dm_drift.columns}")
```

```{code-cell} python3
fig, ax = plt.subplots(figsize=(12, 4))
for col in dm_drift.columns:
    ax.plot(dm_drift[col].to_numpy(), label=col)
ax.set_xlabel("Volume Number")
ax.set_ylabel("Amplitude")
ax.set_title("Legendre Polynomial Drift Regressors")
ax.legend()
plt.tight_layout()
plt.show()
```

### DCT High-Pass Filter

An alternative to polynomials, commonly used in SPM. The `duration` parameter sets the high-pass cutoff (in seconds).

```{code-cell} python3
dm_dct = DesignMatrix({"placeholder": np.zeros(n_samples)}, sampling_freq=sampling_freq)
dm_dct = dm_dct.add_dct_basis(duration=128)
dm_dct = dm_dct.drop("placeholder")
print(f"DCT basis functions: {dm_dct.shape[1]} columns (128s high-pass)")
```

## Building a Complete Design Matrix

A real analysis combines task regressors, motion parameters, drift, and an intercept:

```{code-cell} python3
# Simulate motion parameters
np.random.seed(42)
motion_params = {
    "trans_x": np.random.randn(n_samples) * 0.5,
    "trans_y": np.random.randn(n_samples) * 0.3,
    "trans_z": np.random.randn(n_samples) * 0.4,
    "rot_x": np.random.randn(n_samples) * 0.01,
    "rot_y": np.random.randn(n_samples) * 0.01,
    "rot_z": np.random.randn(n_samples) * 0.01,
}

# Build the full design matrix
task_onsets = [20, 40, 60, 80, 100, 120, 140]
task_stim = np.zeros(n_samples)
for onset in task_onsets:
    start_idx = int(onset / TR)
    end_idx = int((onset + 2) / TR)
    if start_idx < n_samples:
        task_stim[start_idx:min(end_idx, n_samples)] = 1

dm_full = DesignMatrix(
    {"task": task_stim, **motion_params},
    sampling_freq=sampling_freq,
)

# Convolve only task column
dm_full = dm_full.convolve("hrf", columns=["task"])

# Add polynomial drift (order 2 includes intercept, linear, quadratic)
dm_full = dm_full.add_poly(order=2, include_lower=True)

print(f"Complete design matrix: {dm_full.shape}")
print(f"Columns: {dm_full.columns}")
```

## Visualization

### Heatmap

The `plot()` method shows the design matrix in SPM-style format:

```{code-cell} python3
dm_full.plot(figsize=(8, 8))
plt.title("Complete Design Matrix")
plt.show()
```

### Details

Printing (or `repr`-ing) a `DesignMatrix` shows a summary of its structure:

```{code-cell} python3
print(dm_full)
```

## Multicollinearity Diagnostics

### Variance Inflation Factor (VIF)

VIF measures how much each regressor's variance is inflated by correlation with other regressors

```{code-cell} python3
vif = dm_full.vif()
print("Variance Inflation Factors:")
print(vif)
```

### Cleaning Correlated Columns

`clean()` automatically removes columns with correlation above a threshold:

```{code-cell} python3
dm_cleaned = dm_full.clean(thresh=0.95)
print(f"Before: {dm_full.shape[1]} columns → After: {dm_cleaned.shape[1]} columns")
```

## Combining Runs

Use `append(axis=0)` to stack design matrices across runs. Polynomial columns are automatically separated per run with `keep_separate=True` (default):

```{code-cell} python3
# Create two "runs"
run1 = DesignMatrix({"task": np.random.randn(100)}, sampling_freq=sampling_freq)
run1 = run1.add_poly(order=1, include_lower=True)

run2 = DesignMatrix({"task": np.random.randn(100)}, sampling_freq=sampling_freq)
run2 = run2.add_poly(order=1, include_lower=True)

combined = run1.append(run2, axis=0)
print(f"Run 1: {run1.shape} → Combined: {combined.shape}")
print(f"Columns: {combined.columns}")
```

Notice how `poly_0` and `poly_1` are duplicated per run — this ensures each run gets its own intercept and drift terms.
