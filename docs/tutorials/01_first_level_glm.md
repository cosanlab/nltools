# First-Level GLM Analysis with Haxby Dataset

## Learning Objectives

By the end of this tutorial, you will be able to:
- Load and inspect fMRI data using `BrainData`
- Create and manipulate design matrices using `DesignMatrix`
- Understand multi-run concatenation and nuisance regressor handling
- Fit first-level GLM models to single-subject data
- Extract and interpret beta coefficients and t-statistics
- Prepare data for group-level analysis

## Introduction and Setup

The General Linear Model (GLM) is the foundation of fMRI analysis. In a first-level (single-subject) GLM:
1. We model the BOLD timeseries as a linear combination of experimental conditions
2. We convolve experimental events with the hemodynamic response function (HRF)
3. We estimate beta coefficients for each condition
4. We compute statistics (t-values, p-values) for each voxel

**This tutorial demonstrates a complete first-level GLM workflow using the Haxby 2001 dataset**, a well-known dataset featuring multiple runs and multiple experimental conditions (faces, houses, objects, etc.).

```python
# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltools.data import BrainData, DesignMatrix
from nltools.datasets import fetch_haxby

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
```

## Loading Single Subject Data

The `fetch_haxby()` function downloads and loads the Haxby dataset from nilearn. For a single subject, it returns lists of `BrainData` and `DesignMatrix` objects, one for each run.

```python
# Load all runs for subject 1
brain_data_list, design_matrix_list = fetch_haxby(n_subjects=1, verbose=1)

print(f"Number of runs: {len(brain_data_list)}")
print(f"Run 0 shape: {brain_data_list[0].shape}")
print(f"Run 1 shape: {brain_data_list[1].shape}")
```

The return structure is:
- `brain_data_list`: List of `BrainData` objects, one per run
- `design_matrix_list`: List of `DesignMatrix` objects, one per run

Let's inspect the first run's data:

```python
# Inspect first run
run0 = brain_data_list[0]
print(f"Run 0: {run0.shape[0]} timepoints × {run0.shape[1]} voxels")

# Basic BrainData operations
print(f"Mean signal: {run0.mean():.2f}")
print(f"Standard deviation: {run0.std():.2f}")

# Indexing: Access specific timepoints
print(f"First TR shape: {run0[0].shape}")
print(f"First 10 TRs shape: {run0[:10].shape}")

# Basic statistics
mean_timeseries = run0.mean(axis=1)  # Mean across voxels for each timepoint
print(f"Mean timeseries shape: {mean_timeseries.shape}")
```

## Working with DesignMatrix

The design matrices returned by `fetch_haxby()` are already HRF-convolved and ready to use. Let's inspect their structure:

```python
# Inspect first run's design matrix
dm0 = design_matrix_list[0]

print(f"Design matrix shape: {dm0.shape}")
print(f"Sampling frequency (1/TR): {dm0.sampling_freq} Hz")
print(f"TR: {1/dm0.sampling_freq:.2f} seconds")
print(f"\nCondition columns:")
for col in dm0.columns:
    print(f"  - {col}")
```

DesignMatrix provides several useful methods:

```python
# Access specific columns
face_regressor = dm0['face']
print(f"Face regressor shape: {face_regressor.shape}")
print(f"Face regressor statistics:")
print(f"  Mean: {face_regressor.mean():.4f}")
print(f"  Std: {face_regressor.std():.4f}")
print(f"  Min: {face_regressor.min():.4f}")
print(f"  Max: {face_regressor.max():.4f}")

# Visualize design matrix structure
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Heatmap of design matrix
sns.heatmap(
    dm0.values.T, 
    ax=axes[0],
    cmap='RdBu_r',
    center=0,
    cbar_kws={'label': 'Regressor value'},
    xticklabels=50,
    yticklabels=dm0.columns
)
axes[0].set_title('Design Matrix Heatmap (Run 0)')
axes[0].set_xlabel('Timepoint (TR)')
axes[0].set_ylabel('Regressor')

# Time series of key regressors
timepoints = np.arange(dm0.shape[0]) * (1/dm0.sampling_freq)
for condition in ['face', 'house', 'scrambledpix']:
    axes[1].plot(timepoints, dm0[condition], label=condition, linewidth=2)

axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Regressor value')
axes[1].set_title('HRF-Convolved Regressors (Run 0)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

Check correlations between regressors:

```python
# Compute correlation matrix manually
corr_matrix = np.corrcoef(dm0.values.T)  # Transpose: columns become rows for corrcoef
print("Correlation matrix shape:", corr_matrix.shape)

# Visualize correlation
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    square=True,
    vmin=-1,
    vmax=1,
    xticklabels=dm0.columns,
    yticklabels=dm0.columns
)
plt.title('Regressor Correlation Matrix (Run 0)')
plt.tight_layout()
plt.show()
```

## Concatenating Multiple Runs

For a complete first-level analysis, we need to concatenate data across runs. This is a **critical step** that requires proper handling of nuisance regressors (polynomials, intercepts) that should be separated per run.

### Concatenating BrainData

There are several ways to concatenate `BrainData` objects:

```python
# Option 1: Pass list to BrainData constructor (recommended)
concatenated_brain = BrainData(brain_data_list)

print(f"Concatenated shape: {concatenated_brain.shape}")
print(f"Sum of individual runs: {sum(bd.shape[0] for bd in brain_data_list)}")
assert concatenated_brain.shape[0] == sum(bd.shape[0] for bd in brain_data_list), \
    "Concatenated shape should match sum of runs"

# Option 2: Use append() method (chaining)
concatenated_brain_v2 = brain_data_list[0].append(brain_data_list[1])
for bd in brain_data_list[2:]:
    concatenated_brain_v2 = concatenated_brain_v2.append(bd)

# Option 3: Use utility function
from nltools.utils import concatenate
concatenated_brain_v3 = concatenate(brain_data_list)

# All methods should produce the same result
assert concatenated_brain.shape == concatenated_brain_v2.shape
assert concatenated_brain.shape == concatenated_brain_v3.shape
```

### Concatenating DesignMatrix

DesignMatrix concatenation requires special care. When `keep_separate=True` (the default), polynomial/nuisance regressors are automatically separated per run:

```python
# Concatenate design matrices with automatic polynomial separation
dm_concatenated = design_matrix_list[0]
for dm in design_matrix_list[1:]:
    dm_concatenated = dm_concatenated.append(dm, axis=0, keep_separate=True)

print(f"Concatenated design matrix shape: {dm_concatenated.shape}")
print(f"Number of runs: {len(design_matrix_list)}")
print(f"\nColumns after concatenation:")
for col in dm_concatenated.columns:
    print(f"  - {col}")
```

**Why `keep_separate=True`?** Each run needs its own intercept and polynomial drift terms because:
- Baseline signal can differ between runs
- Scanner drift is independent per run
- We want to model these run-specific effects separately

You'll notice that polynomial columns are automatically renamed:
- `0_poly_0`, `1_poly_0`, etc. for intercepts
- `0_poly_1`, `1_poly_1`, etc. for linear trends
- Condition columns (face, house, etc.) remain shared across runs

### Adding Polynomial Regressors

If polynomial regressors aren't already present, add them:

```python
# Check if polynomials already exist
if not dm_concatenated.polys:
    # Add polynomial regressors (order 2 = intercept + linear + quadratic)
    dm_concatenated = dm_concatenated.add_poly(order=2)
    print("Added polynomial regressors")
else:
    print(f"Polynomial regressors already present: {dm_concatenated.polys}")

# Alternative: Add DCT basis for high-pass filtering
# dm_concatenated = dm_concatenated.add_dct_basis(duration=180)  # 180 second cutoff
```

### Visualizing Concatenated Design Matrix

Visualize the run structure:

```python
# Create visualization showing run boundaries
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Heatmap with run boundaries
sns.heatmap(
    dm_concatenated.values.T,
    ax=axes[0],
    cmap='RdBu_r',
    center=0,
    cbar_kws={'label': 'Regressor value'},
    xticklabels=50,
    yticklabels=dm_concatenated.columns
)

# Add vertical lines for run boundaries
run_lengths = [dm.shape[0] for dm in design_matrix_list]
run_boundaries = np.cumsum(run_lengths)
for boundary in run_boundaries[:-1]:
    axes[0].axvline(x=boundary, color='yellow', linewidth=2, linestyle='--', alpha=0.7)

axes[0].set_title('Concatenated Design Matrix with Run Boundaries')
axes[0].set_xlabel('Timepoint (TR)')
axes[0].set_ylabel('Regressor')
axes[0].text(0.02, 0.98, 'Yellow lines = run boundaries', 
             transform=axes[0].transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot polynomial regressors to show separation
poly_cols = [col for col in dm_concatenated.columns if 'poly' in col]
if poly_cols:
    for col in poly_cols[:6]:  # Show first 6 polynomial columns
        axes[1].plot(dm_concatenated[col], label=col, linewidth=1.5, alpha=0.7)
    
    # Add run boundaries
    for boundary in run_boundaries[:-1]:
        axes[1].axvline(x=boundary, color='red', linewidth=1, linestyle='--', alpha=0.5)
    
    axes[1].set_xlabel('Timepoint (TR)')
    axes[1].set_ylabel('Regressor value')
    axes[1].set_title('Polynomial Regressors (Separated by Run)')
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Fitting the GLM Model

Now we can fit the GLM to the concatenated data:

```python
# Fit GLM model
print("Fitting GLM model...")
concatenated_brain.fit(
    model='glm',
    X=dm_concatenated,
    noise_model='ar1'  # Accounts for temporal autocorrelation in fMRI data
)
print("GLM fitting complete!")

# Verify model attributes are set
assert hasattr(concatenated_brain, 'model_'), "Model should be stored"
assert hasattr(concatenated_brain, 'glm_betas'), "Beta coefficients should be available"
assert hasattr(concatenated_brain, 'glm_t'), "T-statistics should be available"
```

**Key parameters explained:**
- `model='glm'`: Use the GLM model
- `X=`: Design matrix with proper nuisance regressors (polynomials separated per run)
- `noise_model='ar1'`: Account for temporal autocorrelation (AR1 model is standard for fMRI)

## Inspecting Results

After fitting, GLM results are stored as attributes on the `BrainData` object:

```python
# Extract beta coefficients
betas = concatenated_brain.glm_betas  # BrainData object: (n_regressors, n_voxels)
print(f"Beta coefficients shape: {betas.shape}")
print(f"Number of regressors: {betas.shape[0]}")
print(f"Number of voxels: {betas.shape[1]}")

# Each beta map corresponds to one regressor (in order of design matrix columns)
print(f"\nRegressor order matches design matrix:")
for i, col_name in enumerate(dm_concatenated.columns[:10]):  # Show first 10
    print(f"  Beta[{i:2d}] = {col_name}")
```

### Accessing Specific Condition Betas

```python
# Find column index for a condition of interest
condition_name = 'face'
if condition_name in dm_concatenated.columns:
    condition_idx = dm_concatenated.columns.index(condition_name)
    beta_face = betas[condition_idx]  # Access by index
    
    print(f"\nBeta map for '{condition_name}':")
    print(f"  Shape: {beta_face.shape}")
    print(f"  Mean: {beta_face.mean():.4f}")
    print(f"  Std: {beta_face.std():.4f}")
    print(f"  Min: {beta_face.min():.4f}")
    print(f"  Max: {beta_face.max():.4f}")
    
    # Visualize beta map
    beta_face.plot(title=f'Beta Coefficients: {condition_name}', colorbar=True)
```

### Extract T-Statistics and P-Values

```python
# Extract t-statistics
t_stats = concatenated_brain.glm_t
print(f"T-statistics shape: {t_stats.shape}")

# Extract p-values
p_vals = concatenated_brain.glm_p
print(f"P-values shape: {p_vals.shape}")

# Access t-statistics for face condition
if condition_name in dm_concatenated.columns:
    t_face = t_stats[condition_idx]
    p_face = p_vals[condition_idx]
    
    print(f"\nT-statistics for '{condition_name}':")
    print(f"  Mean: {t_face.mean():.4f}")
    print(f"  Std: {t_face.std():.4f}")
    print(f"  Significant voxels (p < 0.001, uncorrected): {(p_face.data < 0.001).sum()}")
    
    # Visualize t-statistic map
    t_face.plot(title=f'T-statistics: {condition_name}', colorbar=True)
    
    # Thresholded map
    t_thresh = t_face.threshold(lower=3.3)  # Approximately p < 0.001
    t_thresh.plot(title=f'Thresholded T-statistics (t > 3.3): {condition_name}', colorbar=True)
```

### Model Fit Diagnostics

```python
# Residuals
residuals = concatenated_brain.glm_residual
print(f"Residuals shape: {residuals.shape}")
print(f"Residual std: {residuals.std().mean():.4f}")

# R² values
r2 = concatenated_brain.glm_r2
print(f"R² shape: {r2.shape}")
print(f"Mean R²: {r2.mean():.4f}")
print(f"Median R²: {np.median(r2.data[r2.data > 0]):.4f}")

# Visualize R² map
r2.plot(title='Model R²', colorbar=True)

# Predicted values
predicted = concatenated_brain.glm_predicted
print(f"Predicted values shape: {predicted.shape}")
```

### Iterate Over All Conditions

```python
# Extract beta maps for all conditions
condition_cols = [col for col in dm_concatenated.columns 
                  if col not in dm_concatenated.polys]  # Exclude polynomial columns

print(f"\nExtracting beta maps for {len(condition_cols)} conditions:")
beta_maps = {}
for col_name in condition_cols:
    if col_name in dm_concatenated.columns:
        idx = dm_concatenated.columns.index(col_name)
        beta_maps[col_name] = betas[idx]
        print(f"  ✓ {col_name}")

# Visualize multiple conditions
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (cond_name, beta_map) in enumerate(list(beta_maps.items())[:6]):
    beta_map.plot(ax=axes[i], title=f'Beta: {cond_name}', colorbar=True)

plt.tight_layout()
plt.show()
```

## Group-Level Preparation

To prepare for group-level analysis, we need to fit GLM models for each subject and extract beta maps:

```python
# Load all subjects
print("Loading all subjects...")
all_subjects_brain, all_subjects_dm = fetch_haxby(n_subjects='all', verbose=1)

print(f"Number of subjects: {len(all_subjects_brain)}")
print(f"Number of runs per subject: {len(all_subjects_brain[0])}")

# Understand nested structure:
# - Outer list: subjects (6 subjects)
# - Inner list: runs per subject
```

### Fit GLM for Each Subject

```python
# Loop through subjects and fit GLM for each
all_beta_maps = {}
condition_name = 'face'  # Condition of interest

for subj_idx, (brain_runs, dm_runs) in enumerate(zip(all_subjects_brain, all_subjects_dm)):
    print(f"\nProcessing subject {subj_idx + 1}...")
    
    # Concatenate runs for this subject
    brain_concat = BrainData(brain_runs)  # Constructor takes list
    
    # Concatenate design matrices with polynomial separation
    dm_concat = dm_runs[0]
    for dm in dm_runs[1:]:
        dm_concat = dm_concat.append(dm, axis=0, keep_separate=True)
    
    # Add polynomial regressors if needed
    if not dm_concat.polys:
        dm_concat = dm_concat.add_poly(order=2)
    
    # Fit GLM
    brain_concat.fit(model='glm', X=dm_concat, noise_model='ar1')
    
    # Extract beta for condition of interest
    if condition_name in dm_concat.columns:
        condition_idx = dm_concat.columns.index(condition_name)
        beta_map = brain_concat.glm_betas[condition_idx]  # Access by index
        
        # Store for this subject
        all_beta_maps[f'subject_{subj_idx+1}'] = beta_map
        print(f"  ✓ Extracted beta map for '{condition_name}'")
    else:
        print(f"  ⚠ Condition '{condition_name}' not found in design matrix")

print(f"\n✓ Processed {len(all_beta_maps)} subjects")
```

### Concatenate Beta Maps Across Subjects

```python
# Convert to list of BrainData objects
beta_list = [all_beta_maps[key] for key in sorted(all_beta_maps.keys())]

# Concatenate into single BrainData for group analysis
group_beta_map = BrainData(beta_list)  # Constructor takes list

print(f"Group beta map shape: {group_beta_map.shape}")
print(f"Expected: ({len(beta_list)} subjects, {beta_list[0].shape[1]} voxels)")
assert group_beta_map.shape[0] == len(beta_list), "Shape should match number of subjects"

# Optional: Add metadata for group analysis
group_beta_map.X = pd.DataFrame({
    'subject_id': [f'sub-{i+1:02d}' for i in range(len(group_beta_map))]
})
print(f"\nMetadata shape: {group_beta_map.X.shape}")
print(f"Subject IDs: {group_beta_map.X['subject_id'].tolist()}")
```

The group beta map is now ready for group-level analysis (e.g., one-sample t-test, group comparisons).

## Summary

In this tutorial, you learned how to:
- ✓ Load and inspect fMRI data with `BrainData`
- ✓ Create and visualize design matrices with `DesignMatrix`
- ✓ Properly concatenate multiple runs with separated nuisance regressors
- ✓ Fit GLM models using `.fit(model='glm', X=design_matrix, noise_model='ar1')`
- ✓ Extract and interpret beta coefficients, t-statistics, and p-values
- ✓ Prepare data for group-level analysis by fitting GLMs per subject

### Key Takeaways

1. **Multi-run concatenation**: Always use `keep_separate=True` when concatenating design matrices across runs to properly separate polynomial/nuisance regressors
2. **Noise model**: Use `noise_model='ar1'` for fMRI data to account for temporal autocorrelation
3. **Beta maps**: GLM results are stored as `BrainData` objects, which can be concatenated for group analysis
4. **Accessing results**: Beta maps are accessed by index: `betas[condition_idx]` where `condition_idx = dm.columns.index(condition_name)`

### Next Steps

- **Tutorial: Group-Level Analysis** - Combine results across subjects with t-tests
- **Tutorial: ROI Analysis** - Deep dive into region-of-interest analyses
- **Advanced**: Multiple comparison correction strategies
- **Advanced**: More complex design matrices (basis sets, temporal derivatives)
