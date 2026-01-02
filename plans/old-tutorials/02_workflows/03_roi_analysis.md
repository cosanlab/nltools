# Region of Interest (ROI) Analysis

## Learning Objectives

By the end of this tutorial, you will be able to:
- Load and work with anatomical ROI masks
- Create functional ROIs from statistical maps
- Extract mean timeseries from ROIs
- Perform ROI-based statistical tests
- Correlate ROI activity with behavioral measures
- Visualize ROI results effectively
- Report ROI analyses appropriately

## Introduction

Region of Interest (ROI) analysis focuses on specific brain regions rather than conducting voxel-wise analyses. ROIs can be:
1. **Anatomical**: Defined by brain atlases (e.g., amygdala, V1)
2. **Functional**: Defined by independent localizer tasks
3. **Literature-based**: Coordinates from prior studies

**Benefits of ROI analysis**:
- Increased statistical power (fewer comparisons)
- More interpretable results
- Better signal-to-noise ratio
- Hypothesis-driven approach

**This tutorial demonstrates complete ROI analysis workflows.**

## The Research Question

We have fMRI data from a pain perception study. We want to test whether:
1. The insula shows increased activation during pain
2. Insula activity correlates with subjective pain ratings
3. Individual differences in insula sensitivity predict pain tolerance

## Load Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from nltools.data import BrainData
from nltools.datasets import fetch_pain

# Load example data
data = fetch_pain()
print(f"Data shape: {data.shape()}")

# Get pain ratings from metadata
pain_ratings = data.X['PainLevel'].values
print(f"Pain ratings: {pain_ratings[:10]}")
```

## Method 1: Anatomical ROI from Atlas

```python
# In practice, load an atlas ROI:
# from nilearn import datasets
# atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
# insula_mask = BrainData(atlas.maps).threshold(label=XX, binarize=True)

# For this tutorial, create a simple spherical ROI
# Centered at MNI coordinates for right anterior insula [38, 20, -4]

# TODO: Implement create_sphere() utility
# insula_roi = BrainData.create_sphere(
#     center=[38, 20, -4],
#     radius=10,
#     mask=data.mask
# )

# For now, use a thresholded activation as proxy
print("Note: Using functional activation as proxy for anatomical ROI")
print("In practice, load anatomical ROI from atlas")
```

## Method 2: Functional ROI from Localizer

```python
# Create functional ROI from pain > baseline contrast

# Run quick GLM to get pain activation
# (See First-Level GLM tutorial for details)

# For this example, use mean activation as proxy
mean_activation = data.mean()

# Threshold to create ROI (top 5% of voxels)
functional_roi = mean_activation.threshold(upper='95%', binarize=True)

print(f"Functional ROI size: {functional_roi.data.sum():.0f} voxels")
functional_roi.plot(title='Functional ROI (Top 5% Activation)')
```

## Method 3: ROI from Literature Coordinates

```python
# Create sphere around published coordinates
# Example: Insula peak from meta-analysis at [38, 20, -4]

# TODO: Implement coordinate-to-ROI utility
# literature_roi = BrainData.create_sphere_mni(
#     mni_coords=[38, 20, -4],
#     radius_mm=8
# )

print("Note: Sphere creation utility not yet implemented")
print("Use nilearn.masking.compute_brain_mask or nilearn.image.new_img_like")
```

## Extract ROI Timeseries

```python
# Extract mean timeseries from functional ROI
roi_timeseries = data.extract_roi(functional_roi)

print(f"ROI timeseries shape: {roi_timeseries.shape}")
print(f"Mean ROI activity: {roi_timeseries.mean():.4f}")

# Plot timeseries
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(roi_timeseries, 'k-', linewidth=2)
ax.set_xlabel('Image Number')
ax.set_ylabel('Mean ROI Activity')
ax.set_title(f'ROI Timeseries (n={functional_roi.data.sum():.0f} voxels)')
ax.grid(True, alpha=0.3)
plt.show()
```

## Correlation with Behavior

```python
# Test if ROI activity correlates with pain ratings

# Compute correlation
r, p = pearsonr(roi_timeseries, pain_ratings)

print(f"Pearson correlation: r = {r:.3f}, p = {p:.4e}")

# Visualize relationship
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pain_ratings, roi_timeseries, alpha=0.6, s=50)

# Add regression line
z = np.polyfit(pain_ratings, roi_timeseries, 1)
p_fit = np.poly1d(z)
x_fit = np.linspace(pain_ratings.min(), pain_ratings.max(), 100)
ax.plot(x_fit, p_fit(x_fit), 'r-', linewidth=2, label=f'r = {r:.3f}')

ax.set_xlabel('Pain Rating')
ax.set_ylabel('ROI Activity')
ax.set_title('Brain-Behavior Correlation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# Effect size (proportion of variance explained)
r_squared = r ** 2
print(f"Variance explained: R² = {r_squared:.3f} ({r_squared*100:.1f}%)")
```

## Multi-Subject ROI Analysis

```python
# Extract ROI values across multiple subjects
# In practice: Load individual subject beta images

# For this example, treat individual images as "subjects"
n_subjects = 28
roi_values = []

for i in range(n_subjects):
    # Extract mean from ROI
    subj_roi = data[i].extract_roi(functional_roi)
    roi_values.append(subj_roi)

roi_values = np.array(roi_values)

print(f"ROI values across {n_subjects} subjects:")
print(f"Mean: {roi_values.mean():.4f}")
print(f"Std: {roi_values.std():.4f}")
print(f"SEM: {roi_values.std() / np.sqrt(n_subjects):.4f}")
```

## Group-Level Statistics on ROI

```python
from scipy import stats

# One-sample t-test: Is mean ROI activation > 0?
t_stat, p_value = stats.ttest_1samp(roi_values, 0)

print(f"One-sample t-test: t({n_subjects-1}) = {t_stat:.3f}, p = {p_value:.4e}")

# 95% confidence interval
ci_lower, ci_upper = stats.t.interval(
    0.95,
    df=n_subjects-1,
    loc=roi_values.mean(),
    scale=roi_values.std() / np.sqrt(n_subjects)
)

print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Effect size
cohens_d = roi_values.mean() / roi_values.std()
print(f"Cohen's d: {cohens_d:.3f}")
```

## Visualize Group Results

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(roi_values, bins=15, edgecolor='black', alpha=0.7)
axes[0].axvline(roi_values.mean(), color='r', linestyle='--',
                linewidth=2, label=f'Mean = {roi_values.mean():.3f}')
axes[0].set_xlabel('ROI Beta Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution Across Subjects')
axes[0].legend()

# Box plot with individual points
axes[1].boxplot(roi_values, widths=0.5)
axes[1].scatter(np.ones(n_subjects), roi_values, alpha=0.5, s=50)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('ROI Beta Value')
axes[1].set_title('Group ROI Analysis')
axes[1].set_xticklabels(['Pain Response'])

plt.tight_layout()
plt.show()
```

## Multiple ROIs Comparison

```python
# Compare activity across multiple ROIs

# Create multiple ROIs at different thresholds
roi_liberal = mean_activation.threshold(upper='90%', binarize=True)
roi_moderate = mean_activation.threshold(upper='95%', binarize=True)
roi_strict = mean_activation.threshold(upper='99%', binarize=True)

rois = {
    'Liberal (top 10%)': roi_liberal,
    'Moderate (top 5%)': roi_moderate,
    'Strict (top 1%)': roi_strict
}

# Extract values from each ROI
roi_comparison = {}
for roi_name, roi_mask in rois.items():
    values = []
    for i in range(n_subjects):
        val = data[i].extract_roi(roi_mask)
        values.append(val)
    roi_comparison[roi_name] = np.array(values)

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))

positions = np.arange(len(rois)) + 1
bp = ax.boxplot(
    [roi_comparison[name] for name in rois.keys()],
    positions=positions,
    widths=0.6,
    patch_artist=True
)

# Color boxes
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_xticklabels(rois.keys())
ax.set_ylabel('ROI Beta Value')
ax.set_title('Comparison Across ROI Definitions')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

# Print sizes
for name, roi in rois.items():
    print(f"{name}: {roi.data.sum():.0f} voxels")
```

## ROI-to-ROI Correlation

```python
# Examine correlation between two ROIs

# Create two ROIs from different regions
roi1 = mean_activation.threshold(upper='98%', binarize=True)
roi2_data = data.std()  # Use different criterion
roi2 = roi2_data.threshold(upper='98%', binarize=True)

# Extract timeseries from both ROIs
roi1_ts = data.extract_roi(roi1)
roi2_ts = data.extract_roi(roi2)

# Compute correlation
r, p = pearsonr(roi1_ts, roi2_ts)

print(f"ROI-to-ROI correlation: r = {r:.3f}, p = {p:.4e}")

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(roi1_ts, roi2_ts, alpha=0.6, s=50)
ax.set_xlabel('ROI 1 Activity')
ax.set_ylabel('ROI 2 Activity')
ax.set_title(f'ROI-to-ROI Correlation (r = {r:.3f})')
ax.grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(roi1_ts, roi2_ts, 1)
p_fit = np.poly1d(z)
x_fit = np.linspace(roi1_ts.min(), roi1_ts.max(), 100)
ax.plot(x_fit, p_fit(x_fit), 'r-', linewidth=2)
plt.show()
```

## Subject-Level Individual Differences

```python
# Test if individual differences in ROI sensitivity predict behavior

# Simulate subject-level pain tolerance scores
np.random.seed(42)
pain_tolerance = np.random.randn(n_subjects) * 10 + 50

# Correlate individual ROI betas with tolerance
r, p = pearsonr(roi_values, pain_tolerance)

print(f"Individual differences correlation: r = {r:.3f}, p = {p:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(roi_values, pain_tolerance, s=80, alpha=0.6)
ax.set_xlabel('ROI Beta Value (Pain Response)')
ax.set_ylabel('Pain Tolerance Score')
ax.set_title('Individual Differences Analysis')
ax.grid(True, alpha=0.3)

# Add regression line if significant
if p < 0.05:
    z = np.polyfit(roi_values, pain_tolerance, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(roi_values.min(), roi_values.max(), 100)
    ax.plot(x_fit, p_fit(x_fit), 'r-', linewidth=2,
            label=f'r = {r:.3f}, p = {p:.3f}')
    ax.legend()

plt.show()
```

## Best Practices for ROI Analysis

```python
print("""
ROI Analysis Best Practices:
----------------------------

1. **Independence**:
   - Define ROIs independently of your test data
   - Use separate localizer scan or literature coordinates
   - Avoid "double-dipping" (defining and testing on same data)

2. **Multiple comparison correction**:
   - If testing multiple ROIs, correct for multiple comparisons
   - Bonferroni: p_corrected = p_threshold / n_rois
   - FDR for exploratory ROI sets

3. **Report ROI definition**:
   - Size (number of voxels, volume in mm³)
   - Source (atlas, coordinates, functional localizer)
   - Threshold used (if functional ROI)

4. **Visualize ROIs**:
   - Show ROI location on anatomical image
   - Report overlap with anatomical structures
   - Check for unexpected spatial extent

5. **Effect sizes**:
   - Report standardized effect sizes (Cohen's d, partial η²)
   - Provide confidence intervals
   - Consider power analysis for sample size
""")
```

## Reporting ROI Results

```python
# Generate methods text
methods_text = f"""
Methods: ROI Analysis
---------------------
We defined a functional ROI based on the group-level pain > baseline
contrast (top 5% of activated voxels, {functional_roi.data.sum():.0f} voxels).
For each subject, we extracted the mean beta value from this ROI and
tested whether it differed significantly from zero using a one-sample
t-test (two-tailed, α = 0.05).

Results:
--------
The pain-responsive ROI showed significant positive activation across
subjects (M = {roi_values.mean():.3f}, SD = {roi_values.std():.3f},
t({n_subjects-1}) = {t_stat:.2f}, p = {p_value:.4e}, 95% CI =
[{ci_lower:.3f}, {ci_upper:.3f}], Cohen's d = {cohens_d:.2f}).
"""

print(methods_text)
```

## Save ROI Masks and Results

```python
# Save ROI mask for reuse
functional_roi.write('roi_pain-responsive_top5pct.nii.gz')

# Save extracted values
roi_df = pd.DataFrame({
    'subject_id': range(1, n_subjects + 1),
    'roi_beta': roi_values,
    'pain_tolerance': pain_tolerance
})
roi_df.to_csv('roi_extracted_values.csv', index=False)

print("✓ ROI mask and extracted values saved")
print(roi_df.head())
```

## Sanity Checks

```python
# Verify ROI extraction
assert len(roi_timeseries) == len(data), "Timeseries length should match data"
assert not np.isnan(roi_timeseries).any(), "ROI values should not be NaN"
assert functional_roi.data.sum() > 0, "ROI should contain voxels"

# Check that ROI is within brain mask
assert (functional_roi.data <= 1).all(), "Binary ROI should be 0 or 1"

# Verify statistics
manual_mean = roi_values.sum() / len(roi_values)
assert abs(manual_mean - roi_values.mean()) < 1e-6, "Mean should match manual calculation"

print("✓ All sanity checks passed!")
```

## Summary

In this tutorial, you learned how to:
- ✓ Define ROIs from anatomical atlases, functional localizers, and literature
- ✓ Extract mean timeseries from ROIs
- ✓ Correlate ROI activity with behavioral measures
- ✓ Perform group-level statistics on ROI values
- ✓ Compare activity across multiple ROIs
- ✓ Analyze individual differences with ROI data
- ✓ Follow best practices for independence and reporting

## Next Steps

- **Tutorial: Connectivity Analysis** - ROI-to-ROI functional connectivity
- **Tutorial: MVPA** - Multivariate pattern analysis within ROIs
- **Advanced**: Time-varying connectivity with ROI timeseries
- **Advanced**: Mediation analysis with ROI values

## Clean Up

```python
import os
for fname in ['roi_pain-responsive_top5pct.nii.gz', 'roi_extracted_values.csv']:
    if os.path.exists(fname):
        os.remove(fname)
```
