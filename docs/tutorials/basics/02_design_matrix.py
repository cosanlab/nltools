# %% [markdown]
# # DesignMatrix Basics
#
# This tutorial covers the `DesignMatrix` class for building fMRI experimental designs.
#
# ## Learning Objectives
#
# By the end of this tutorial, you will be able to:
# - Create `DesignMatrix` objects for fMRI analysis
# - Add task regressors with HRF convolution
# - Include nuisance covariates (motion, drift, etc.)
# - Visualize and diagnose design matrices
# - Check for multicollinearity
# - Build design matrices for common experimental designs

# %% [markdown]
# ## Introduction
#
# The `DesignMatrix` class represents the design matrix (X) in the General Linear Model:
#
# **Y = Xβ + ε**
#
# Key components:
# - **Task regressors**: Experimental conditions convolved with HRF
# - **Nuisance regressors**: Motion, drift, physiological noise
# - **Interactions**: Task × continuous moderators

# %%
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import DesignMatrix

# %% [markdown]
# ## Creating a Basic Design Matrix

# %%
# Design matrix parameters
sampling_freq = 0.5  # TR = 2.0 seconds → sampling_freq = 1/TR
n_samples = 150  # Number of volumes
TR = 1 / sampling_freq

# Create design matrix from DataFrame
data = pd.DataFrame(
    {"intercept": np.ones(n_samples), "linear_trend": np.linspace(0, 1, n_samples)}
)

dm = DesignMatrix(data, sampling_freq=sampling_freq)

print(f"Design matrix shape: {dm.shape}")
print(f"Columns: {list(dm.columns)}")

# %% [markdown]
# ## HRF Convolution
#
# The hemodynamic response function (HRF) models the sluggish BOLD response
# to neural activity. To create HRF-convolved regressors:
#
# 1. Create a stimulus timecourse (boxcar or impulse)
# 2. Use `.convolve()` to convolve with the canonical HRF

# %%
# Define event timing
onsets = [20, 40, 60, 80, 100, 120]  # Event times in seconds
durations = [2] * len(onsets)  # Event durations in seconds

# Create stimulus timecourse (boxcar)
stim = np.zeros(n_samples)
for onset, dur in zip(onsets, durations):
    start_idx = int(onset / TR)
    end_idx = int((onset + dur) / TR)
    if start_idx < n_samples:
        stim[start_idx : min(end_idx, n_samples)] = 1

# Create design matrix and convolve
dm_hrf = DesignMatrix(pd.DataFrame({"task": stim}), sampling_freq=sampling_freq)
dm_hrf = dm_hrf.convolve("hrf")

print(f"Design matrix with HRF-convolved regressor: {dm_hrf.shape}")

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(stim)
axes[0].set_ylabel("Stimulus")
axes[0].set_title("Original Boxcar")

axes[1].plot(dm_hrf["task"].to_numpy())
axes[1].set_ylabel("HRF Convolved")
axes[1].set_xlabel("Volume Number")
axes[1].set_title("After HRF Convolution")

plt.tight_layout()
plt.close()

# %% [markdown]
# ### Parametric Modulation

# %%
# Modulate regressor by trial-level variable (e.g., reaction time)
onsets = [20, 40, 60, 80, 100]
durations = [1] * len(onsets)
reaction_times = [0.5, 0.7, 0.6, 0.9, 0.55]  # Modulation values

# Create main effect timecourse
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

print(f"Parametric design matrix: {dm_param.shape}")
corr = np.corrcoef(dm_param["task_main"].to_numpy(), dm_param["task_x_rt"].to_numpy())[
    0, 1
]
print(f"Correlation between regressors: {corr:.3f}")

# %% [markdown]
# ## Polynomial Drift Regressors
#
# fMRI data contains low-frequency drift that must be modeled.

# %%
# Add polynomial drift terms (linear, quadratic, cubic)
dm_drift = DesignMatrix(
    pd.DataFrame({"placeholder": np.zeros(n_samples)}), sampling_freq=sampling_freq
)
dm_drift = dm_drift.add_poly(order=3, include_lower=True)

# Remove placeholder
dm_drift = dm_drift.drop("placeholder")

print(f"Drift regressors: {list(dm_drift.columns)}")

# Visualize
fig, axes = plt.subplots(len(dm_drift.columns), 1, figsize=(12, 6), sharex=True)
for i, col in enumerate(dm_drift.columns):
    axes[i].plot(dm_drift[col].to_numpy())
    axes[i].set_ylabel(col)
axes[-1].set_xlabel("Volume Number")
plt.suptitle("Polynomial Drift Regressors")
plt.tight_layout()
plt.close()

# %% [markdown]
# ## DCT Basis for High-Pass Filtering
#
# Alternative to polynomial drift, commonly used in SPM.

# %%
# Add DCT basis set for drift modeling
dm_dct = DesignMatrix(
    pd.DataFrame({"placeholder": np.zeros(n_samples)}), sampling_freq=sampling_freq
)
dm_dct = dm_dct.add_dct_basis(duration=128)  # 128-second high-pass filter
dm_dct = dm_dct.drop("placeholder")

print(f"DCT basis functions: {dm_dct.shape[1]} columns")

# %% [markdown]
# ## Motion Regressors

# %%
# Simulate 6 motion parameters (3 translation + 3 rotation)
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

print(f"Motion design matrix: {dm_motion.shape}")

# Visualize motion parameters
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(dm_motion[["trans_x", "trans_y", "trans_z"]].to_numpy())
axes[0].set_ylabel("Translation (mm)")
axes[0].legend(["X", "Y", "Z"])
axes[0].set_title("Translation Parameters")

axes[1].plot(dm_motion[["rot_x", "rot_y", "rot_z"]].to_numpy())
axes[1].set_ylabel("Rotation (radians)")
axes[1].set_xlabel("Volume Number")
axes[1].legend(["X", "Y", "Z"])
axes[1].set_title("Rotation Parameters")

plt.tight_layout()
plt.close()

# %% [markdown]
# ## Complete Design Matrix
#
# Putting it all together: task + motion + drift + intercept

# %%
# Create task regressor
task_onsets = [20, 40, 60, 80, 100, 120, 140]
task_stim = np.zeros(n_samples)
for onset in task_onsets:
    start_idx = int(onset / TR)
    end_idx = int((onset + 2) / TR)
    if start_idx < n_samples:
        task_stim[start_idx : min(end_idx, n_samples)] = 1

# Build complete design matrix
dm_full = DesignMatrix(
    pd.DataFrame({"task": task_stim, **motion_params}), sampling_freq=sampling_freq
)

# Convolve task column only
dm_full = dm_full.convolve("hrf", columns=["task"])

# Add polynomial drift
dm_full = dm_full.add_poly(order=2, include_lower=True)

# Add intercept
dm_full["intercept"] = 1.0

print(f"Complete design matrix: {dm_full.shape}")
print(f"Columns: {list(dm_full.columns)}")

# %% [markdown]
# ## Visualization and Diagnostics
#
# ### Heatmap Visualization

# %%
# Visualize full design matrix
dm_full.heatmap(figsize=(10, 8))
plt.title("Complete Design Matrix")
plt.close()

# %% [markdown]
# ### Variance Inflation Factor (VIF)
#
# VIF > 10 indicates problematic multicollinearity.

# %%
# Compute VIF for each regressor
vif = dm_full.vif()

print("Variance Inflation Factors:")
print(vif)

# Flag problematic regressors
if (vif > 10).any():
    print("\n⚠️  Warning: High multicollinearity detected in:")
    print(vif[vif > 10])
else:
    print("\n✓ No problematic multicollinearity (all VIF < 10)")

# %% [markdown]
# ## Common Experimental Designs
#
# ### Block Design

# %%
# Alternating blocks of task and rest
block_duration = 20  # seconds
n_blocks = 5

block_stim = np.zeros(n_samples)
for i in range(n_blocks):
    onset = i * (block_duration * 2)  # Task + rest
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

# %% [markdown]
# ### Event-Related Design

# %%
# Brief events with randomized inter-trial intervals
np.random.seed(42)
n_trials = 20
min_iti = 4  # Minimum inter-trial interval (seconds)
max_iti = 12  # Maximum ITI

# Generate event onsets
event_onsets = [10]  # First event at 10s
for _ in range(n_trials - 1):
    iti = np.random.uniform(min_iti, max_iti)
    event_onsets.append(event_onsets[-1] + iti)

# Create stimulus timecourse
event_n_samples = int(event_onsets[-1] / TR) + 20  # Extra samples for HRF tail
event_stim = np.zeros(event_n_samples)
for onset in event_onsets:
    idx = int(onset / TR)
    if idx < event_n_samples:
        event_stim[idx] = 1  # Impulse events

dm_event = DesignMatrix(
    pd.DataFrame({"task_event": event_stim}), sampling_freq=sampling_freq
)
dm_event = dm_event.convolve("hrf")
dm_event["intercept"] = 1.0

print(f"Event-related design: {dm_event.shape}")

# %% [markdown]
# ## Saving and Loading

# %%
# Save design matrix as CSV (access underlying Polars DataFrame)
import polars as pl  # noqa: E402

dm_full._df.write_csv("/tmp/design_matrix.csv")
print("Saved to /tmp/design_matrix.csv")

# Load design matrix
dm_loaded = DesignMatrix(pl.read_csv("/tmp/design_matrix.csv"), sampling_freq=0.5)

print(f"Loaded design matrix: {dm_loaded.shape}")
assert dm_loaded.shape == dm_full.shape, "Loaded matrix should match original"
print("✓ Data verified")

# %% [markdown]
# ## Summary
#
# In this tutorial, you learned how to:
# - ✓ Create `DesignMatrix` objects with appropriate sampling parameters
# - ✓ Add task regressors with HRF convolution using `.convolve()`
# - ✓ Include parametric modulators for trial-level effects
# - ✓ Model low-frequency drift with polynomials or DCT
# - ✓ Add motion and nuisance regressors
# - ✓ Diagnose multicollinearity with VIF
# - ✓ Build designs for block and event-related paradigms
#
# ## Next Steps
#
# - **[Adjacency Basics](03_adjacency)**: Connectivity and similarity matrices
# - **[GLM Workflow](../workflows/01_glm)**: Using design matrices in regression

# %%
# Clean up
import os  # noqa: E402

if os.path.exists("/tmp/design_matrix.csv"):
    os.remove("/tmp/design_matrix.csv")
