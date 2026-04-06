---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
execute:
  skip: true
---

# DesignMatrix Basics

The `DesignMatrix` class for building fMRI experimental designs.

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import os

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from nltools.data import DesignMatrix
```

## Create Basic Design Matrix

```{code-cell} python3
sampling_freq = 0.5  # TR = 2.0 seconds
n_samples = 150
TR = 1 / sampling_freq

data = pd.DataFrame(
    {"intercept": np.ones(n_samples), "linear_trend": np.linspace(0, 1, n_samples)}
)

dm = DesignMatrix(data, sampling_freq=sampling_freq)
print(f"Shape: {dm.shape}")
print(f"Columns: {list(dm.columns)}")
```

## HRF Convolution

```{code-cell} python3
# Define event timing
onsets = [20, 40, 60, 80, 100, 120]  # seconds
durations = [2] * len(onsets)

# Create stimulus boxcar
stim = np.zeros(n_samples)
for onset, dur in zip(onsets, durations):
    start_idx = int(onset / TR)
    end_idx = int((onset + dur) / TR)
    if start_idx < n_samples:
        stim[start_idx : min(end_idx, n_samples)] = 1

# Convolve with HRF
dm_hrf = DesignMatrix(pd.DataFrame({"task": stim}), sampling_freq=sampling_freq)
dm_hrf = dm_hrf.convolve("hrf")
print(f"HRF-convolved: {dm_hrf.shape}")
```

```{code-cell} python3
# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(stim)
axes[0].set_ylabel("Stimulus")
axes[0].set_title("Original Boxcar")
axes[1].plot(dm_hrf["task"].to_numpy())
axes[1].set_ylabel("HRF Convolved")
axes[1].set_xlabel("Volume Number")
plt.tight_layout()
plt.close()
```

## Parametric Modulation

```{code-cell} python3
onsets = [20, 40, 60, 80, 100]
durations = [1] * len(onsets)
reaction_times = [0.5, 0.7, 0.6, 0.9, 0.55]

stim_main = np.zeros(n_samples)
stim_param = np.zeros(n_samples)

for onset, dur, rt in zip(onsets, durations, reaction_times):
    start_idx = int(onset / TR)
    end_idx = int((onset + dur) / TR)
    if start_idx < n_samples:
        stim_main[start_idx : min(end_idx, n_samples)] = 1
        stim_param[start_idx : min(end_idx, n_samples)] = rt

dm_param = DesignMatrix(
    pd.DataFrame({"task_main": stim_main, "task_x_rt": stim_param}),
    sampling_freq=sampling_freq,
)
dm_param = dm_param.convolve("hrf")

corr = np.corrcoef(dm_param["task_main"].to_numpy(), dm_param["task_x_rt"].to_numpy())[
    0, 1
]
print(f"Correlation: {corr:.3f}")
```

## Polynomial Drift

```{code-cell} python3
dm_drift = DesignMatrix(
    pd.DataFrame({"placeholder": np.zeros(n_samples)}), sampling_freq=sampling_freq
)
dm_drift = dm_drift.add_poly(order=3, include_lower=True)
dm_drift = dm_drift.drop("placeholder")

print(f"Drift regressors: {list(dm_drift.columns)}")
```

## DCT High-Pass Filter

```{code-cell} python3
dm_dct = DesignMatrix(
    pd.DataFrame({"placeholder": np.zeros(n_samples)}), sampling_freq=sampling_freq
)
dm_dct = dm_dct.add_dct_basis(duration=128)
dm_dct = dm_dct.drop("placeholder")

print(f"DCT basis: {dm_dct.shape[1]} columns")
```

## Motion Regressors

```{code-cell} python3
motion_params = pd.DataFrame(
    {
        "trans_x": np.random.randn(n_samples) * 0.5,
        "trans_y": np.random.randn(n_samples) * 0.3,
        "trans_z": np.random.randn(n_samples) * 0.4,
        "rot_x": np.random.randn(n_samples) * 0.01,
        "rot_y": np.random.randn(n_samples) * 0.01,
        "rot_z": np.random.randn(n_samples) * 0.01,
    }
)

dm_motion = DesignMatrix(motion_params, sampling_freq=sampling_freq)
print(f"Motion: {dm_motion.shape}")
```

## Complete Design Matrix

```{code-cell} python3
# Task regressor
task_onsets = [20, 40, 60, 80, 100, 120, 140]
task_stim = np.zeros(n_samples)
for onset in task_onsets:
    start_idx = int(onset / TR)
    end_idx = int((onset + 2) / TR)
    if start_idx < n_samples:
        task_stim[start_idx : min(end_idx, n_samples)] = 1

# Build full design
dm_full = DesignMatrix(
    pd.DataFrame({"task": task_stim, **motion_params}), sampling_freq=sampling_freq
)
dm_full = dm_full.convolve("hrf", columns=["task"])
dm_full = dm_full.add_poly(order=2, include_lower=True)
dm_full["intercept"] = 1.0

print(f"Complete: {dm_full.shape}")
print(f"Columns: {list(dm_full.columns)}")
```

```{code-cell} python3
# Heatmap
dm_full.heatmap(figsize=(10, 8))
plt.title("Complete Design Matrix")
plt.close()
```

## VIF (Multicollinearity Check)

```{code-cell} python3
vif = dm_full.vif()
print("Variance Inflation Factors:")
print(vif)

if (vif > 10).any():
    print("\n⚠️  High multicollinearity:")
    print(vif[vif > 10])
else:
    print("\n✓ No problematic multicollinearity (VIF < 10)")
```

## Block Design

```{code-cell} python3
block_duration = 20
n_blocks = 5

block_stim = np.zeros(n_samples)
for i in range(n_blocks):
    onset = i * (block_duration * 2)
    start_idx = int(onset / TR)
    end_idx = int((onset + block_duration) / TR)
    if start_idx < n_samples:
        block_stim[start_idx : min(end_idx, n_samples)] = 1

dm_block = DesignMatrix(
    pd.DataFrame({"task_block": block_stim}), sampling_freq=sampling_freq
)
dm_block = dm_block.convolve("hrf")
dm_block["intercept"] = 1.0

print(f"Block design: {dm_block.shape}")
```

## Event-Related Design

```{code-cell} python3
np.random.seed(42)
n_trials = 20
min_iti, max_iti = 4, 12

event_onsets = [10]
for _ in range(n_trials - 1):
    iti = np.random.uniform(min_iti, max_iti)
    event_onsets.append(event_onsets[-1] + iti)

event_n_samples = int(event_onsets[-1] / TR) + 20
event_stim = np.zeros(event_n_samples)
for onset in event_onsets:
    idx = int(onset / TR)
    if idx < event_n_samples:
        event_stim[idx] = 1

dm_event = DesignMatrix(
    pd.DataFrame({"task_event": event_stim}), sampling_freq=sampling_freq
)
dm_event = dm_event.convolve("hrf")
dm_event["intercept"] = 1.0

print(f"Event-related design: {dm_event.shape}")
```

## File I/O

```{code-cell} python3
dm_full._df.write_csv("/tmp/design_matrix.csv")
print("Saved to /tmp/design_matrix.csv")

dm_loaded = DesignMatrix(pl.read_csv("/tmp/design_matrix.csv"), sampling_freq=0.5)
assert dm_loaded.shape == dm_full.shape
print("✓ Data verified")
```

```{code-cell} python3
# Clean up
if os.path.exists("/tmp/design_matrix.csv"):
    os.remove("/tmp/design_matrix.csv")
```
