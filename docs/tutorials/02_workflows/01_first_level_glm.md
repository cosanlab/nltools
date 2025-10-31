# First-Level GLM Analysis

## Learning Objectives

By the end of this tutorial, you will be able to:
- Load fMRI timeseries data for a single subject
- Create a design matrix with HRF-convolved regressors
- Add motion and nuisance covariates
- Run a General Linear Model (GLM) regression
- Extract and interpret beta coefficients and t-statistics
- Visualize statistical maps
- Save results for group-level analysis

## Introduction

The General Linear Model (GLM) is the foundation of fMRI analysis. In a first-level (single-subject) GLM:
1. We model the BOLD timeseries as a linear combination of experimental conditions
2. We convolve experimental events with the hemodynamic response function (HRF)
3. We estimate beta coefficients for each condition
4. We compute statistics (t-values, p-values) for each voxel

**This tutorial demonstrates a complete first-level GLM workflow.**

## The Research Question

We have a pain perception study where participants experienced thermal stimulation at three intensity levels (low, medium, high) while in the scanner. We want to:
1. Model the BOLD response to each pain level
2. Test which brain regions respond to pain
3. Examine the linear relationship between pain intensity and brain activity

## Load fMRI Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import BrainData, DesignMatrix
from nltools.datasets import fetch_pain

# Load single-subject timeseries
# In practice, you'd load your own data:
# data = BrainData('sub-01_task-pain_bold.nii.gz')
data = fetch_pain()[:60]  # Use first 60 volumes as example

print(f"Data shape: {data.shape()}")
print(f"TR: 2.0 seconds (typical)")
```

## Load Experimental Events

```python
# In practice, load from BIDS events.tsv file:
# events = pd.read_csv('sub-01_task-pain_events.tsv', sep='\t')

# For this tutorial, create example events
events = pd.DataFrame({
    'onset': [10, 30, 50, 70, 90, 110, 130, 150],  # seconds
    'duration': [5, 5, 5, 5, 5, 5, 5, 5],           # seconds
    'trial_type': ['low', 'med', 'high', 'low', 'med', 'high', 'low', 'med'],
    'intensity': [3, 5, 7, 3, 5, 7, 3, 5]           # pain rating
})

print(events)
```

## Create Design Matrix

### Step 1: Initialize Design Matrix

```python
# Create basic design matrix from sampling frequency
sampling_freq = 1 / 2.0  # TR = 2.0 seconds
n_volumes = len(data)

dm = DesignMatrix(
    sampling_freq=sampling_freq,
    n_samples=n_volumes
)

print(f"Design matrix shape: {dm.shape}")
```

### Step 2: Add Task Regressors with HRF Convolution

```python
# Add regressor for each pain level
for pain_level in ['low', 'med', 'high']:
    # Get onsets for this condition
    condition_events = events[events['trial_type'] == pain_level]
    onsets = condition_events['onset'].values
    durations = condition_events['duration'].values

    # Add to design matrix with HRF convolution
    dm = dm.add_event_regressor(
        onsets=onsets,
        durations=durations,
        name=f'pain_{pain_level}'
    )

print(f"Design matrix with task regressors: {dm.shape}")
dm.plot()
```

### Step 3: Add Parametric Modulator

```python
# Add parametric modulator for pain intensity
# This tests for linear relationship between intensity and BOLD

all_onsets = events['onset'].values
all_durations = events['duration'].values
intensity_values = events['intensity'].values

dm = dm.add_event_regressor(
    onsets=all_onsets,
    durations=all_durations,
    amplitude=intensity_values,  # Modulate by intensity
    name='pain_intensity'
)

print(f"Design matrix with parametric modulator: {dm.shape}")
```

### Step 4: Add Motion Regressors

```python
# In practice, load motion parameters from preprocessing:
# motion = pd.read_csv('sub-01_task-pain_confounds.tsv', sep='\t')
# motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

# For this tutorial, simulate motion parameters
n_motion_params = 6
motion = pd.DataFrame(
    np.random.randn(n_volumes, n_motion_params) * 0.1,
    columns=[f'motion_{i}' for i in range(n_motion_params)]
)

# Add motion regressors
for col in motion.columns:
    dm[col] = motion[col].values

print(f"Design matrix with motion: {dm.shape}")
```

### Step 5: Add Intercept

```python
# Add intercept (baseline) term
dm = dm.add_intercept()

print(f"Final design matrix: {dm.shape}")
dm.heatmap()
```

### Design Matrix Diagnostics

```python
# Check for multicollinearity
vif = dm.vif()
print("Variance Inflation Factors:")
print(vif)

# VIF > 10 indicates problematic multicollinearity
if (vif > 10).any():
    print("⚠️  Warning: High multicollinearity detected!")
else:
    print("✓ VIF values look good")

# Visualize correlation structure
dm.corr_matrix()
```

## Run GLM Regression

```python
# Attach design matrix to data
data.X = dm

# Run regression
results = data.regress()

print("Regression complete!")
print(f"Beta coefficients shape: {len(results['beta'])}")
print(f"T-statistics shape: {len(results['t'])}")
```

## Examine Results

### Beta Coefficients

```python
# Beta coefficients for each regressor
beta_names = dm.columns

# Visualize beta for low pain condition
beta_low = results['beta'][beta_names.index('pain_low')]
beta_low.plot(title='Beta: Low Pain')

# Visualize beta for high pain condition
beta_high = results['beta'][beta_names.index('pain_high')]
beta_high.plot(title='Beta: High Pain')

# Visualize parametric effect of intensity
beta_intensity = results['beta'][beta_names.index('pain_intensity')]
beta_intensity.plot(title='Beta: Pain Intensity Effect')
```

### T-Statistics

```python
# T-statistics test if betas are significantly different from zero

# T-stat for pain intensity effect
t_intensity = results['t'][beta_names.index('pain_intensity')]
t_intensity.plot(title='T-statistic: Pain Intensity Effect')

# Threshold at t > 3.3 (approximately p < 0.001, uncorrected)
t_thresh = t_intensity.threshold(lower=3.3)
t_thresh.plot(title='Thresholded T-statistic (t > 3.3)')

# Count significant voxels
n_sig = (t_thresh.data > 0).sum()
print(f"Significant voxels (uncorrected): {n_sig}")
```

### R² (Model Fit)

```python
# R² indicates how well the model fits the data
r2 = results['r2']
r2.plot(title='Model R²')

print(f"Mean R²: {r2.mean():.3f}")
print(f"Median R²: {np.median(r2.data[r2.data > 0]):.3f}")
```

## Contrasts

### Define Contrasts

```python
# Contrast: High pain > Low pain
contrast_high_vs_low = np.zeros(len(beta_names))
contrast_high_vs_low[beta_names.index('pain_high')] = 1
contrast_high_vs_low[beta_names.index('pain_low')] = -1

# Compute contrast
# TODO: This will be available in v0.6.0
# contrast_result = data.compute_contrast(contrast_high_vs_low)
# contrast_result['t'].plot(title='T-stat: High > Low Pain')

# For now, compute manually
beta_contrast = results['beta'][beta_names.index('pain_high')] - \
                results['beta'][beta_names.index('pain_low')]
beta_contrast.plot(title='Contrast: High > Low Pain')
```

### Multiple Comparisons Correction

```python
# Apply FDR correction to t-statistics
# TODO: Update when ttest() with FDR is implemented
# t_fdr = t_intensity.threshold_fdr(q=0.05)

# For now, use simple threshold
t_thresh_strict = t_intensity.threshold(lower=4.0)
t_thresh_strict.plot(title='Conservative Threshold (t > 4.0)')
```

## Extract ROI Timecourses

```python
# Create ROI from thresholded activation
roi_mask = t_intensity.threshold(upper='95%', binarize=True)
print(f"ROI contains {roi_mask.data.sum():.0f} voxels")

# Extract mean timecourse from ROI
roi_timecourse = data.extract_roi(roi_mask)

# Plot timecourse with events overlaid
fig, ax = plt.subplots(figsize=(12, 4))

# Plot timecourse
time_points = np.arange(len(roi_timecourse)) * 2.0  # TR = 2.0s
ax.plot(time_points, roi_timecourse, 'k-', linewidth=2)

# Overlay event onsets
for _, event in events.iterrows():
    color = {'low': 'blue', 'med': 'orange', 'high': 'red'}[event['trial_type']]
    ax.axvspan(event['onset'], event['onset'] + event['duration'],
               alpha=0.3, color=color)

ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Mean BOLD Signal')
ax.set_title('ROI Timecourse with Task Events')
ax.legend(['BOLD Signal', 'Low Pain', 'Med Pain', 'High Pain'])
plt.show()
```

## Visualize Predicted vs Actual

```python
# Get predicted timeseries for a single voxel
voxel_idx = 1000  # Example voxel

# Predicted = Design matrix × Beta coefficients
predicted = dm.values @ results['beta'].data[:, voxel_idx]
actual = data.data[:, voxel_idx]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_points, actual, 'k-', alpha=0.5, label='Actual')
ax.plot(time_points, predicted, 'r-', linewidth=2, label='Predicted')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('BOLD Signal')
ax.set_title(f'Model Fit: Voxel {voxel_idx}')
ax.legend()
plt.show()

# Compute correlation
from scipy.stats import pearsonr
r, p = pearsonr(actual, predicted)
print(f"Voxel {voxel_idx} - Pearson r: {r:.3f}, p-value: {p:.3e}")
```

## Save Results for Group Analysis

```python
# Save beta images for group-level analysis
# Typically save one beta image per condition

beta_low.write('sub-01_beta_pain-low.nii.gz')
beta_high.write('sub-01_beta_pain-high.nii.gz')
beta_intensity.write('sub-01_beta_pain-intensity.nii.gz')

# Save t-statistics
t_intensity.write('sub-01_tstat_pain-intensity.nii.gz')

print("✓ Results saved for group-level analysis")
```

## Sanity Checks

```python
# Check that regression worked correctly
assert len(results['beta']) == dm.shape[1], "Should have one beta per regressor"
assert results['beta'][0].shape() == data[0].shape(), "Betas should match data shape"

# Check that residuals are smaller than original signal
residual_var = results['residual'].std().mean()
original_var = data.std().mean()
assert residual_var < original_var, "Residuals should be smaller than original"

# Check R² is bounded [0, 1]
assert (r2.data >= 0).all() and (r2.data <= 1).all(), "R² should be in [0, 1]"

print("✓ All sanity checks passed!")
```

## Summary

In this tutorial, you learned how to:
- ✓ Load fMRI timeseries data
- ✓ Create event files for experimental conditions
- ✓ Build a design matrix with HRF-convolved task regressors
- ✓ Add motion and nuisance covariates
- ✓ Run GLM regression with `regress()`
- ✓ Extract and visualize beta coefficients and t-statistics
- ✓ Compute contrasts between conditions
- ✓ Extract ROI timecourses
- ✓ Validate model fit with R²
- ✓ Save results for group-level analysis

## Next Steps

- **Tutorial: Group-Level Analysis** - Combine results across subjects with t-tests
- **Tutorial: ROI Analysis** - Deep dive into region-of-interest analyses
- **Advanced**: Multiple comparison correction strategies
- **Advanced**: More complex design matrices (e.g., basis sets, temporal derivatives)

## Clean Up

```python
import os
for fname in ['sub-01_beta_pain-low.nii.gz',
              'sub-01_beta_pain-high.nii.gz',
              'sub-01_beta_pain-intensity.nii.gz',
              'sub-01_tstat_pain-intensity.nii.gz']:
    if os.path.exists(fname):
        os.remove(fname)
```
