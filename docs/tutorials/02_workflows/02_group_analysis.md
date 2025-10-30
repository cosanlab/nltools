# Group-Level Analysis

## Learning Objectives

By the end of this tutorial, you will be able to:
- Load first-level beta images from multiple subjects
- Perform one-sample t-tests at the group level
- Apply multiple comparison correction (FDR, FWE)
- Threshold statistical maps appropriately
- Interpret and visualize group results
- Extract effect sizes and confidence intervals
- Report results following best practices

## Introduction

After running first-level GLMs for individual subjects, we combine results across subjects using **group-level inference**. This allows us to:
1. Identify brain regions that show consistent activation across the group
2. Make population-level inferences (not just subject-specific)
3. Account for between-subject variability
4. Control for multiple comparisons across voxels

**This tutorial demonstrates a complete group-level analysis workflow.**

## The Research Question

We have 28 subjects who completed a pain perception task. Each subject has first-level beta coefficients representing their brain response to pain stimulation. We want to test:
1. Which brain regions show consistent pain responses across subjects?
2. What is the effect size of the pain response?
3. Which voxels survive multiple comparison correction?

## Load First-Level Results

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import Brain_Data
from nltools.datasets import fetch_pain

# In practice, load first-level beta images:
# subjects = []
# for sub_id in range(1, 29):
#     beta = Brain_Data(f'sub-{sub_id:02d}_beta_pain-intensity.nii.gz')
#     subjects.append(beta)
#
# group_data = Brain_Data(subjects)

# For this tutorial, use example data
group_data = fetch_pain()
print(f"Group data shape: {group_data.shape()}")
print(f"Number of subjects: {len(group_data)}")
```

## Data Quality Checks

```python
# Check for outliers or missing data
global_means = group_data.mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution of global means
axes[0].hist(global_means, bins=20, edgecolor='black')
axes[0].set_xlabel('Global Mean Intensity')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Subject Means')

# Identify potential outliers
mean_global = np.mean(global_means)
std_global = np.std(global_means)
outliers = np.abs(global_means - mean_global) > 3 * std_global

# Box plot
axes[1].boxplot(global_means)
axes[1].set_ylabel('Global Mean Intensity')
axes[1].set_title('Subject Global Means')
if outliers.any():
    axes[1].scatter(np.where(outliers)[0] + 1, global_means[outliers],
                    c='red', s=100, marker='x', label='Potential Outliers')
    axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Potential outliers: {outliers.sum()}")
if outliers.any():
    print(f"Outlier indices: {np.where(outliers)[0]}")
```

## Visualize Individual Subjects

```python
# Look at spatial patterns for a few subjects
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i in range(6):
    group_data[i].plot(axes=axes[i//3, i%3], title=f'Subject {i+1}')
plt.tight_layout()
plt.show()
```

## Group Mean and Variability

```python
# Compute group mean
group_mean = group_data.mean()
group_mean.plot(title='Group Mean Activation')

# Compute group standard deviation
group_std = group_data.std()
group_std.plot(title='Between-Subject Variability')

# Compute coefficient of variation (CV = std / mean)
cv = group_std / group_mean
cv.plot(title='Coefficient of Variation')

print(f"Mean activation: {group_mean.mean():.4f}")
print(f"Mean variability: {group_std.mean():.4f}")
```

## One-Sample T-Test

```python
# Test if group mean is significantly different from zero
# H0: population mean = 0
# H1: population mean ≠ 0

# TODO: Update when ttest() is fully implemented
# t_results = group_data.ttest()

# For now, compute manually
n_subjects = len(group_data)
mean_image = group_data.mean()
std_image = group_data.std()

# T-statistic = mean / (std / sqrt(n))
t_stat = mean_image / (std_image / np.sqrt(n_subjects))

print(f"T-statistic shape: {t_stat.shape()}")
print(f"Mean t-value: {t_stat.mean():.2f}")
print(f"Max t-value: {t_stat.data.max():.2f}")

# Visualize t-statistics
t_stat.plot(title='One-Sample T-Test (Uncorrected)')
```

### Compute P-Values

```python
from scipy import stats

# Degrees of freedom
df = n_subjects - 1

# Convert t-statistics to p-values (two-tailed)
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat.data), df))

# Create Brain_Data object for p-values
p_map = Brain_Data(p_values, mask=t_stat.mask)

# Negative log p-values for visualization
neg_log_p = Brain_Data(-np.log10(p_values), mask=t_stat.mask)
neg_log_p.plot(title='-log10(p-value)')

# Count voxels at different thresholds
p_001 = (p_values < 0.001).sum()
p_005 = (p_values < 0.05).sum()
print(f"Voxels p < 0.001 (uncorrected): {p_001}")
print(f"Voxels p < 0.05 (uncorrected): {p_005}")
```

## Multiple Comparison Correction

### False Discovery Rate (FDR)

```python
# FDR controls the expected proportion of false discoveries
# More powerful than Bonferroni, appropriate for exploratory analyses

from statsmodels.stats.multitest import multipletests

# Apply FDR correction (Benjamini-Hochberg)
q = 0.05  # Desired FDR level

# Flatten p-values for FDR correction
p_flat = p_values[~np.isnan(p_values)]
reject, p_corrected, _, _ = multipletests(p_flat, alpha=q, method='fdr_bh')

# Reconstruct brain image
p_fdr = np.full_like(p_values, np.nan)
p_fdr[~np.isnan(p_values)] = p_corrected

fdr_map = Brain_Data(p_fdr < q, mask=t_stat.mask)
n_sig_fdr = fdr_map.data.sum()

print(f"Significant voxels (FDR q < {q}): {n_sig_fdr}")
fdr_map.plot(title=f'FDR Corrected (q < {q})')
```

### Family-Wise Error (FWE) Correction

```python
# FWE controls the probability of ANY false positive
# More conservative, appropriate for confirmatory analyses

# Bonferroni correction
n_voxels = (~np.isnan(p_values)).sum()
p_bonferroni = 0.05 / n_voxels

bonf_map = Brain_Data(p_values < p_bonferroni, mask=t_stat.mask)
n_sig_bonf = bonf_map.data.sum()

print(f"Bonferroni threshold: p < {p_bonferroni:.2e}")
print(f"Significant voxels (Bonferroni): {n_sig_bonf}")

if n_sig_bonf > 0:
    bonf_map.plot(title='Bonferroni Corrected (p < 0.05)')
else:
    print("No voxels survive Bonferroni correction")
```

### Cluster-Level Inference

```python
# Alternative: threshold at uncorrected p, then cluster correction
# TODO: Implement cluster-level inference in v0.6.1

# For now, use simple cluster size threshold
t_thresh = t_stat.threshold(lower=3.3)  # Approximately p < 0.001

# Count clusters
# This is a placeholder - proper cluster inference requires:
# 1. Smoothness estimation
# 2. Random field theory or permutation testing
# 3. Cluster extent threshold

print("Note: Proper cluster-level inference not yet implemented")
print("Use nilearn.glm for cluster correction")
```

## Thresholding and Visualization

```python
# Create multiple views with different thresholds

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Uncorrected p < 0.001
t_unc = t_stat.threshold(lower=3.3)
t_unc.plot(axes=axes[0, 0], title='Uncorrected p < 0.001 (t > 3.3)')

# FDR q < 0.05
if n_sig_fdr > 0:
    t_fdr = Brain_Data(t_stat.data * fdr_map.data, mask=t_stat.mask)
    t_fdr.plot(axes=axes[0, 1], title='FDR q < 0.05')
else:
    axes[0, 1].text(0.5, 0.5, 'No significant voxels',
                    ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('FDR q < 0.05')

# Top 5% of voxels
t_top5 = t_stat.threshold(upper='95%')
t_top5.plot(axes=axes[1, 0], title='Top 5% of Voxels')

# Effect size (Cohen\'s d = mean / std)
cohens_d = mean_image / std_image
cohens_d.plot(axes=axes[1, 1], title='Effect Size (Cohen\'s d)')

plt.tight_layout()
plt.show()
```

## Extract Peak Coordinates

```python
# Find peak voxels (local maxima in t-statistic map)
# TODO: Implement peak detection in v0.6.1

# For now, find global maximum
max_idx = np.nanargmax(t_stat.data)
max_t = t_stat.data[max_idx]

# Convert to MNI coordinates
# This requires converting from flattened index to 3D coordinates
# then applying the affine transformation
print(f"Peak t-value: {max_t:.2f}")
print("Note: Full peak detection not yet implemented")
```

## Region of Interest (ROI) Analysis

```python
# Extract mean values from anatomical ROI
# In practice, load an atlas:
# from nltools.mask import expand_mask
# atlas = Brain_Data('atlas.nii.gz')

# Create ROI from thresholded activation
roi_mask = t_stat.threshold(upper='95%', binarize=True)
print(f"ROI size: {roi_mask.data.sum():.0f} voxels")

# Extract values from ROI for each subject
roi_values = []
for i in range(len(group_data)):
    subj_data = group_data[i].apply_mask(roi_mask)
    roi_mean = subj_data.mean()
    roi_values.append(roi_mean)

roi_values = np.array(roi_values)

# Statistical summary
print(f"ROI mean: {roi_values.mean():.4f}")
print(f"ROI std: {roi_values.std():.4f}")
print(f"ROI SEM: {roi_values.std() / np.sqrt(len(roi_values)):.4f}")

# 95% confidence interval
ci_lower = roi_values.mean() - 1.96 * (roi_values.std() / np.sqrt(len(roi_values)))
ci_upper = roi_values.mean() + 1.96 * (roi_values.std() / np.sqrt(len(roi_values)))
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

## Visualize Group Results

```python
# Bar plot with error bars
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Violin plot
axes[0].violinplot([roi_values], positions=[1], showmeans=True)
axes[0].scatter(np.ones_like(roi_values), roi_values, alpha=0.5)
axes[0].set_ylabel('ROI Beta Value')
axes[0].set_title('Distribution of ROI Values Across Subjects')
axes[0].set_xticks([1])
axes[0].set_xticklabels(['Pain Response'])

# Bar plot with 95% CI
axes[1].bar([1], [roi_values.mean()], yerr=[(ci_upper - ci_lower) / 2],
            capsize=10, color='steelblue', alpha=0.7)
axes[1].set_ylabel('Mean Beta Value')
axes[1].set_title('Group Mean ± 95% CI')
axes[1].set_xticks([1])
axes[1].set_xticklabels(['Pain Response'])
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

## Effect Size Estimation

```python
# Cohen's d for group (mean / std)
cohens_d_roi = roi_values.mean() / roi_values.std()
print(f"Cohen's d: {cohens_d_roi:.3f}")

# Interpret effect size
if abs(cohens_d_roi) < 0.2:
    interpretation = "small"
elif abs(cohens_d_roi) < 0.5:
    interpretation = "small to medium"
elif abs(cohens_d_roi) < 0.8:
    interpretation = "medium to large"
else:
    interpretation = "large"

print(f"Effect size interpretation: {interpretation}")

# Hedge's g (corrected for small sample bias)
hedges_g = cohens_d_roi * (1 - (3 / (4 * (n_subjects - 1) - 1)))
print(f"Hedge's g: {hedges_g:.3f}")
```

## Reporting Results

```python
# Generate text for methods/results section
t_value, p_value = stats.ttest_1samp(roi_values, 0)

report = f"""
Results Summary:
----------------
Sample size: N = {n_subjects}
Mean activation: {roi_values.mean():.4f} ± {roi_values.std():.4f} (M ± SD)
95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
One-sample t-test: t({df}) = {t_value:.2f}, p = {p_value:.4e}
Effect size: Cohen's d = {cohens_d_roi:.3f}, Hedge's g = {hedges_g:.3f}
Number of significant voxels (FDR q < 0.05): {n_sig_fdr}
"""

print(report)
```

## Save Group Results

```python
# Save statistical maps for publication
group_mean.write('group_mean_pain-response.nii.gz')
t_stat.write('group_tstat_pain-response.nii.gz')

# Save thresholded maps
if n_sig_fdr > 0:
    t_fdr_masked = Brain_Data(t_stat.data * fdr_map.data, mask=t_stat.mask)
    t_fdr_masked.write('group_tstat_pain-response_fdr-q05.nii.gz')

print("✓ Group results saved")
```

## Sanity Checks

```python
# Verify statistical computations
assert len(roi_values) == n_subjects, "ROI values should match number of subjects"
assert not np.isnan(roi_values).any(), "ROI values should not contain NaN"

# Check t-statistic computation
manual_t = roi_values.mean() / (roi_values.std() / np.sqrt(n_subjects))
assert abs(manual_t - t_value) < 1e-6, "T-statistic should match manual calculation"

# Check that FDR is less conservative than Bonferroni
if n_sig_fdr > 0 and n_sig_bonf > 0:
    assert n_sig_fdr >= n_sig_bonf, "FDR should be less conservative than Bonferroni"

print("✓ All sanity checks passed!")
```

## Summary

In this tutorial, you learned how to:
- ✓ Load and quality-check first-level results from multiple subjects
- ✓ Compute group-level statistics (mean, std, t-tests)
- ✓ Apply multiple comparison correction (FDR, FWE/Bonferroni)
- ✓ Threshold and visualize group statistical maps
- ✓ Extract ROI values and compute confidence intervals
- ✓ Calculate and interpret effect sizes (Cohen's d, Hedge's g)
- ✓ Report results following best practices

## Next Steps

- **Tutorial: ROI Analysis** - Deep dive into region-based analyses
- **Tutorial: Two-Sample T-Tests** - Compare groups (e.g., patients vs. controls)
- **Advanced**: Regression with covariates (age, sex, etc.)
- **Advanced**: Mixed-effects models for repeated measures

## Clean Up

```python
import os
for fname in ['group_mean_pain-response.nii.gz',
              'group_tstat_pain-response.nii.gz',
              'group_tstat_pain-response_fdr-q05.nii.gz']:
    if os.path.exists(fname):
        os.remove(fname)
```
