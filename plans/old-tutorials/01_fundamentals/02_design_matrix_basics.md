# DesignMatrix Basics

## Learning Objectives

By the end of this tutorial, you will be able to:
- Create `DesignMatrix` objects for fMRI analysis
- Add task regressors with HRF convolution
- Include nuisance covariates (motion, drift, etc.)
- Visualize and diagnose design matrices
- Check for multicollinearity
- Build design matrices for common experimental designs

## Introduction

The `DesignMatrix` class represents the design matrix (X) in the General Linear Model: **Y = Xβ + ε**

Key components:
- **Task regressors**: Experimental conditions convolved with HRF
- **Nuisance regressors**: Motion, drift, physiological noise
- **Interactions**: Task × continuous moderators

**This tutorial covers everything you need to know about design matrices in nltools.**

## Creating a Basic Design Matrix

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import DesignMatrix

# Create design matrix from sampling parameters
sampling_freq = 0.5  # TR = 2.0 seconds → sampling_freq = 1/TR
n_samples = 150      # Number of volumes

dm = DesignMatrix(
    sampling_freq=sampling_freq,
    n_samples=n_samples
)

print(dm)
print(f"Shape: {dm.shape}")
```

## Adding Columns

```python
# Add a column of ones (intercept)
dm['intercept'] = 1

# Add a continuous regressor
dm['age'] = np.random.randn(n_samples)

# Add multiple columns
predictors = pd.DataFrame({
    'task_A': np.random.randn(n_samples),
    'task_B': np.random.randn(n_samples)
})
dm = dm.append(predictors, axis=1)

print(f"Design matrix with columns: {dm.shape}")
print(dm.head())
```

## HRF Convolution

The hemodynamic response function (HRF) models the sluggish BOLD response to neural activity.

### Canonical HRF

```python
# Create event regressor with HRF convolution
onsets = [20, 40, 60, 80, 100, 120]  # Event times in seconds
durations = [2] * len(onsets)         # Event durations in seconds

# Create fresh design matrix for this example
dm_hrf = DesignMatrix(sampling_freq=0.5, n_samples=150)

# Add event regressor (automatically convolved)
dm_hrf = dm_hrf.add_event_regressor(
    onsets=onsets,
    durations=durations,
    name='task'
)

print(f"Design matrix with HRF-convolved regressor: {dm_hrf.shape}")

# Visualize the regressor
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dm_hrf['task'].values, linewidth=2)
ax.set_xlabel('Volume Number')
ax.set_ylabel('Task Regressor')
ax.set_title('HRF-Convolved Task Regressor')

# Mark original event onsets
for onset in onsets:
    ax.axvline(onset / 2.0, color='r', linestyle='--', alpha=0.5)

plt.show()
```

### Parametric Modulation

```python
# Modulate regressor by trial-level variable (e.g., reaction time)

# Event times and modulator values
onsets = [20, 40, 60, 80, 100]
durations = [1] * len(onsets)
reaction_times = [0.5, 0.7, 0.6, 0.9, 0.55]  # Seconds

dm_param = DesignMatrix(sampling_freq=0.5, n_samples=150)

# Add main effect
dm_param = dm_param.add_event_regressor(
    onsets=onsets,
    durations=durations,
    name='task_main'
)

# Add parametric modulation
dm_param = dm_param.add_event_regressor(
    onsets=onsets,
    durations=durations,
    amplitude=reaction_times,  # Modulate by RT
    name='task_x_rt'
)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(dm_param['task_main'].values)
axes[0].set_ylabel('Main Effect')
axes[0].set_title('Task Main Effect')

axes[1].plot(dm_param['task_x_rt'].values)
axes[1].set_ylabel('Parametric Modulation')
axes[1].set_xlabel('Volume Number')
axes[1].set_title('Task × Reaction Time')

plt.tight_layout()
plt.show()

print(f"Correlation between regressors: {dm_param['task_main'].corr(dm_param['task_x_rt']):.3f}")
```

## Polynomial Drift Regressors

fMRI data contains low-frequency drift that must be modeled.

```python
# Add polynomial drift terms (linear, quadratic, cubic)
dm_drift = DesignMatrix(sampling_freq=0.5, n_samples=150)

# Add drift regressors
dm_drift = dm_drift.add_poly(order=3, include_lower=True)

print(f"Drift regressors: {dm_drift.columns.tolist()}")

# Visualize
dm_drift.plot(figsize=(12, 6))
plt.suptitle('Polynomial Drift Regressors')
plt.show()
```

## Discrete Cosine Transform (DCT) Basis

Alternative to polynomial drift, used in SPM.

```python
# Add DCT basis set for drift modeling
dm_dct = DesignMatrix(sampling_freq=0.5, n_samples=150)

# DCT with 128-second high-pass filter (SPM default)
dm_dct = dm_dct.add_dct_basis(duration=128)

print(f"DCT basis functions: {dm_dct.shape[1]}")

# Visualize
dm_dct.plot(figsize=(12, 8))
plt.suptitle('DCT Basis Set (128s high-pass)')
plt.show()
```

## Motion Regressors

```python
# Load motion parameters from preprocessing
# In practice: motion = pd.read_csv('confounds.tsv', sep='\t')

# Simulate 6 motion parameters (3 translation + 3 rotation)
n_vols = 150
motion_params = {
    'trans_x': np.random.randn(n_vols) * 0.5,
    'trans_y': np.random.randn(n_vols) * 0.3,
    'trans_z': np.random.randn(n_vols) * 0.4,
    'rot_x': np.random.randn(n_vols) * 0.01,
    'rot_y': np.random.randn(n_vols) * 0.01,
    'rot_z': np.random.randn(n_vols) * 0.01
}

dm_motion = DesignMatrix(pd.DataFrame(motion_params), sampling_freq=0.5)

print(f"Motion design matrix: {dm_motion.shape}")

# Visualize motion parameters
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Translation
axes[0].plot(dm_motion[['trans_x', 'trans_y', 'trans_z']].values)
axes[0].set_ylabel('Translation (mm)')
axes[0].legend(['X', 'Y', 'Z'])
axes[0].set_title('Translation Parameters')

# Rotation
axes[1].plot(dm_motion[['rot_x', 'rot_y', 'rot_z']].values)
axes[1].set_ylabel('Rotation (radians)')
axes[1].set_xlabel('Volume Number')
axes[1].legend(['X', 'Y', 'Z'])
axes[1].set_title('Rotation Parameters')

plt.tight_layout()
plt.show()
```

### Framewise Displacement

```python
# Compute framewise displacement (FD) as motion summary
trans = dm_motion[['trans_x', 'trans_y', 'trans_z']].values
rot = dm_motion[['rot_x', 'rot_y', 'rot_z']].values

# Convert rotations to mm (assume 50mm brain radius)
rot_mm = rot * 50

# Compute absolute displacement from previous frame
fd = np.sum(np.abs(np.diff(trans, axis=0)), axis=1) + \
     np.sum(np.abs(np.diff(rot_mm, axis=0)), axis=1)

# Prepend 0 for first volume
fd = np.concatenate([[0], fd])

# Visualize
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(fd, 'k-', linewidth=2)
ax.axhline(y=0.5, color='r', linestyle='--', label='FD > 0.5mm threshold')
ax.set_xlabel('Volume Number')
ax.set_ylabel('Framewise Displacement (mm)')
ax.set_title('Motion Quality Control')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

print(f"Volumes with FD > 0.5mm: {(fd > 0.5).sum()}")
```

## Complete Design Matrix

Putting it all together: task + motion + drift + intercept

```python
# Create comprehensive design matrix
dm_full = DesignMatrix(sampling_freq=0.5, n_samples=150)

# 1. Task regressors
task_onsets = [20, 40, 60, 80, 100, 120, 140]
dm_full = dm_full.add_event_regressor(
    onsets=task_onsets,
    durations=[2] * len(task_onsets),
    name='task'
)

# 2. Motion parameters
for col in motion_params.keys():
    dm_full[col] = motion_params[col]

# 3. Polynomial drift
dm_full = dm_full.add_poly(order=2, include_lower=True)

# 4. Intercept
dm_full = dm_full.add_intercept()

print(f"Complete design matrix: {dm_full.shape}")
print(f"Columns: {dm_full.columns.tolist()}")
```

## Visualization and Diagnostics

### Heatmap Visualization

```python
# Visualize full design matrix
dm_full.heatmap(figsize=(10, 8))
plt.title('Complete Design Matrix')
plt.show()
```

### Correlation Matrix

```python
# Check correlation between regressors
dm_full.corr_matrix(figsize=(10, 8))
plt.title('Regressor Correlation Matrix')
plt.show()
```

### Variance Inflation Factor (VIF)

VIF > 10 indicates problematic multicollinearity.

```python
# Compute VIF for each regressor
vif = dm_full.vif()

print("Variance Inflation Factors:")
print(vif)

# Visualize VIF
fig, ax = plt.subplots(figsize=(10, 6))
vif.plot(kind='bar', ax=ax)
ax.axhline(y=10, color='r', linestyle='--', label='VIF = 10 threshold')
ax.set_ylabel('VIF')
ax.set_title('Multicollinearity Check')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Flag problematic regressors
if (vif > 10).any():
    print("\n⚠️  Warning: High multicollinearity detected in:")
    print(vif[vif > 10])
else:
    print("\n✓ No problematic multicollinearity (all VIF < 10)")
```

## Common Experimental Designs

### Block Design

```python
# Alternating blocks of task and rest

block_duration = 20  # seconds
n_blocks = 5

onsets = []
durations = []

for i in range(n_blocks):
    onset = i * (block_duration * 2)  # Task + rest
    onsets.append(onset)
    durations.append(block_duration)

dm_block = DesignMatrix(sampling_freq=0.5, n_samples=200)
dm_block = dm_block.add_event_regressor(
    onsets=onsets,
    durations=durations,
    name='task_block'
)
dm_block = dm_block.add_intercept()

dm_block.plot(figsize=(12, 4))
plt.title('Block Design')
plt.show()
```

### Event-Related Design

```python
# Brief events with randomized inter-trial intervals

np.random.seed(42)
n_trials = 20
min_iti = 4   # Minimum inter-trial interval (seconds)
max_iti = 12  # Maximum ITI

# Generate event onsets
onsets = [10]  # First event at 10s
for _ in range(n_trials - 1):
    iti = np.random.uniform(min_iti, max_iti)
    onsets.append(onsets[-1] + iti)

dm_event = DesignMatrix(sampling_freq=0.5, n_samples=300)
dm_event = dm_event.add_event_regressor(
    onsets=onsets,
    durations=[1] * n_trials,  # Brief 1s events
    name='task_event'
)
dm_event = dm_event.add_intercept()

dm_event.plot(figsize=(12, 4))
plt.title('Event-Related Design')
plt.show()
```

### Mixed Design (Block + Event)

```python
# Combination of sustained blocks and brief events

dm_mixed = DesignMatrix(sampling_freq=0.5, n_samples=200)

# Blocks
block_onsets = [20, 100]
dm_mixed = dm_mixed.add_event_regressor(
    onsets=block_onsets,
    durations=[30, 30],
    name='context_block'
)

# Events within blocks
event_onsets = [25, 30, 35, 105, 110, 115]
dm_mixed = dm_mixed.add_event_regressor(
    onsets=event_onsets,
    durations=[1] * len(event_onsets),
    name='probe_event'
)

dm_mixed = dm_mixed.add_intercept()

dm_mixed.plot(figsize=(12, 6))
plt.suptitle('Mixed Design (Block + Event)')
plt.show()
```

## Convolving Custom HRF

```python
# Use custom HRF (e.g., for different populations or brain regions)

# Create a simple gamma function HRF
def custom_hrf(t, peak=6, undershoot=16):
    """Custom HRF based on gamma functions."""
    return (t**(peak-1) * np.exp(-t)) - 0.35 * (t**(undershoot-1) * np.exp(-t/1.5))

# Generate HRF
t_hrf = np.arange(0, 30, 2.0)  # 0-30s with TR=2s
hrf_values = custom_hrf(t_hrf)

# Visualize custom HRF
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_hrf, hrf_values, 'b-', linewidth=2)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Amplitude')
ax.set_title('Custom HRF')
ax.grid(True, alpha=0.3)
plt.show()

# TODO: Convolve with custom HRF
# dm_custom = dm.convolve(hrf=hrf_values)
print("Note: Custom HRF convolution to be implemented")
```

## Saving and Loading

```python
# Save design matrix
dm_full.to_csv('design_matrix.csv', index=False)

# Load design matrix
dm_loaded = DesignMatrix(pd.read_csv('design_matrix.csv'), sampling_freq=0.5)

print(f"Loaded design matrix: {dm_loaded.shape}")
assert dm_loaded.shape == dm_full.shape, "Loaded matrix should match original"
```

## Sanity Checks

```python
# Check design matrix properties
assert dm_full.shape[0] == n_samples, "Rows should match number of volumes"
assert 'intercept' in dm_full.columns, "Should include intercept"

# Check that regressors are not constant (except intercept)
for col in dm_full.columns:
    if col != 'intercept':
        assert dm_full[col].std() > 0, f"{col} should not be constant"

# Check for NaN values
assert not dm_full.isnull().any().any(), "Should not contain NaN values"

# Check VIF
assert (vif < 100).all(), "Extreme multicollinearity detected (VIF > 100)"

print("✓ All sanity checks passed!")
```

## Summary

In this tutorial, you learned how to:
- ✓ Create `DesignMatrix` objects with appropriate sampling parameters
- ✓ Add task regressors with automatic HRF convolution
- ✓ Include parametric modulators for trial-level effects
- ✓ Model low-frequency drift with polynomials or DCT
- ✓ Add motion and nuisance regressors
- ✓ Diagnose multicollinearity with VIF and correlation matrices
- ✓ Visualize design matrices with heatmaps and line plots
- ✓ Build designs for block, event-related, and mixed paradigms

## Next Steps

- **Tutorial: First-Level GLM** - Use design matrices in regression analyses
- **Tutorial: Advanced Design Matrices** - Interactions, basis sets, FIR models
- **API Reference**: Deep dive into all `DesignMatrix` methods

## Clean Up

```python
import os
if os.path.exists('design_matrix.csv'):
    os.remove('design_matrix.csv')
```
