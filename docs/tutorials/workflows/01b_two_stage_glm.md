---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
execute:
  skip: true
---

# Two-Stage GLM Analysis

Two-stage (also called "summary statistics") analysis is the standard approach
for group-level fMRI inference. In the first stage, you fit a GLM to each
subject separately. In the second stage, you perform statistical tests across
subjects on the first-level parameter estimates (betas).

nltools provides a fluent API for this workflow using the **pool infrastructure**.

## Learning Objectives

- Understand two-stage GLM analysis
- Use `bc.fit().pool()` to aggregate first-level results
- Perform group t-tests with `pool.fit(model='ttest')`
- Run multiple contrasts efficiently
- Apply FDR/Bonferroni thresholding

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from nltools.pipelines.pool import PooledData, StatResult, ResultDict
```

## The Two-Stage Workflow

```
Stage 1: First-Level (per subject)
┌─────────────────────────────────────────┐
│  Subject 1: GLM → betas (face, house)   │
│  Subject 2: GLM → betas (face, house)   │
│  Subject 3: GLM → betas (face, house)   │
│  ...                                     │
└─────────────────────────────────────────┘
                   ↓
             pool(param='beta')
                   ↓
Stage 2: Second-Level (across subjects)
┌─────────────────────────────────────────┐
│  t-test on (face - house) contrast      │
│  → t-map, p-map                         │
│  → threshold(method='fdr')              │
└─────────────────────────────────────────┘
```

With a real `BrainCollection`, the workflow looks like:

```python
result = (
    bc.fit(model='glm', X=design_matrices)
    .pool(param='beta')
    .fit(model='ttest', contrast='face-house')
)
```

In this tutorial, we'll work directly with `PooledData` to understand
the second-stage mechanics.

## Creating Pooled Data

`PooledData` represents first-level parameter estimates aggregated across
subjects. The data has shape `(n_subjects, n_conditions, n_voxels)`.

```{code-cell} python3
# Simulate first-level betas for 8 subjects, 3 conditions, 100 voxels
np.random.seed(42)

n_subjects = 8
n_conditions = 3
n_voxels = 100

# Create beta data with signal in first 20 voxels
betas = np.random.randn(n_subjects, n_conditions, n_voxels) * 0.5

# Add consistent signal across subjects:
# - Condition 0 (face): strong activation in voxels 0-20
# - Condition 1 (house): moderate activation in voxels 0-20
# - Condition 2 (object): weak activation
betas[:, 0, :20] += 1.5  # Face: strong
betas[:, 1, :20] += 0.5  # House: moderate
betas[:, 2, :20] += 0.2  # Object: weak

# Create PooledData
pool = PooledData(
    data=betas,
    param="beta",
    condition_names=["face", "house", "object"]
)

print(f"PooledData shape: {pool.shape}")
print(f"  n_subjects: {pool.n_subjects}")
print(f"  n_conditions: {pool.n_conditions}")
print(f"  n_voxels: {pool.n_voxels}")
print(f"  conditions: {pool.condition_names}")
```

## One-Sample T-Test on a Contrast

The most common second-level analysis: test whether a contrast (e.g., face - house)
is significantly different from zero across subjects.

```{code-cell} python3
# Test: Is face activation > house activation?
result = pool.fit(model="ttest", contrast="face-house")

print(f"Result type: {type(result).__name__}")
print(f"Contrast: {result.contrast}")
print(f"T-map shape: {result.t_map.shape}")
print(f"P-map shape: {result.p_map.shape}")
```

## Interpreting StatResult

`StatResult` contains the statistical maps and metadata.

```{code-cell} python3
# Look at the t-values
print("T-value statistics:")
print(f"  Min: {result.t_map.min():.2f}")
print(f"  Max: {result.t_map.max():.2f}")
print(f"  Mean: {result.t_map.mean():.2f}")

# The first 20 voxels should have higher t-values (that's where we added signal)
mean_signal_region = np.mean(result.t_map[:20])
mean_noise_region = np.mean(result.t_map[20:])

print(f"\nSignal region (voxels 0-20): mean t = {mean_signal_region:.2f}")
print(f"Noise region (voxels 20+):   mean t = {mean_noise_region:.2f}")
```

```{code-cell} python3
# Visualize the t-map
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(range(n_voxels), result.t_map, color=['steelblue' if i < 20 else 'gray' for i in range(n_voxels)])
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(y=2, color='red', linestyle='--', label='t=2 threshold')
ax.set_xlabel('Voxel')
ax.set_ylabel('T-value')
ax.set_title('Face - House Contrast (T-values)')
ax.legend()
plt.tight_layout()
plt.close()

print("Plot created: T-values across voxels")
```

## Thresholding Statistical Maps

Apply multiple comparison correction with `.threshold()`.

```{code-cell} python3
# FDR correction (less conservative)
thresholded_fdr = result.threshold(method="fdr", alpha=0.05)

n_sig_fdr = np.sum(thresholded_fdr.t_map != 0)
print(f"FDR (q=0.05): {n_sig_fdr} significant voxels")

# Bonferroni correction (more conservative)
thresholded_bonf = result.threshold(method="bonferroni", alpha=0.05)

n_sig_bonf = np.sum(thresholded_bonf.t_map != 0)
print(f"Bonferroni (p=0.05): {n_sig_bonf} significant voxels")

# Uncorrected threshold
thresholded_unc = result.threshold(method="uncorrected", alpha=0.001)

n_sig_unc = np.sum(thresholded_unc.t_map != 0)
print(f"Uncorrected (p=0.001): {n_sig_unc} significant voxels")
```

## Multiple Contrasts

Often you want to test several contrasts. Pass a list to get a `ResultDict`.

```{code-cell} python3
# Test all pairwise comparisons
results = pool.fit(
    model="ttest",
    contrasts=["face-house", "face-object", "house-object"]
)

print(f"Result type: {type(results).__name__}")
print(f"Contrasts: {list(results.keys())}")

# Access individual results
for contrast, stat in results.items():
    n_sig = np.sum(stat.threshold(method="fdr").t_map != 0)
    print(f"  {contrast}: {n_sig} significant voxels (FDR)")
```

## Batch Thresholding

Apply the same threshold to all contrasts at once.

```{code-cell} python3
# Threshold all results with FDR
thresholded_all = results.threshold_all(method="fdr", alpha=0.05)

print("Batch thresholding applied:")
for contrast, stat in thresholded_all.items():
    n_sig = np.sum(stat.t_map != 0)
    print(f"  {contrast}: {n_sig} voxels survive FDR")
```

## ANOVA: Testing Multiple Conditions

When you have 3+ conditions and want to test for any difference, use ANOVA.

```{code-cell} python3
# Create data with 4 conditions
np.random.seed(42)
betas_4cond = np.random.randn(12, 4, 80)

# Add graded signal: each condition has different activation in first 10 voxels
for i in range(4):
    betas_4cond[:, i, :10] += i * 0.5  # Increasing means

pool_4cond = PooledData(
    data=betas_4cond,
    param="beta",
    condition_names=["cond1", "cond2", "cond3", "cond4"]
)

# Run ANOVA
anova_result = pool_4cond.fit(model="anova")

print(f"ANOVA result:")
print(f"  F-map shape: {anova_result.f_map.shape}")
print(f"  Max F: {anova_result.f_map.max():.2f}")

# Signal region should have higher F-values
mean_f_signal = np.mean(anova_result.f_map[:10])
mean_f_noise = np.mean(anova_result.f_map[10:])
print(f"  Mean F (signal): {mean_f_signal:.2f}")
print(f"  Mean F (noise): {mean_f_noise:.2f}")
```

## Paired T-Test

For within-subject comparisons (e.g., pre vs post).

```{code-cell} python3
# Create pre/post data
np.random.seed(42)
betas_paired = np.random.randn(15, 2, 60)

# Post > Pre in first 15 voxels
betas_paired[:, 1, :15] += 1.0

pool_paired = PooledData(
    data=betas_paired,
    param="beta",
    condition_names=["pre", "post"]
)

# Paired t-test
paired_result = pool_paired.fit(model="paired_ttest")

print(f"Paired t-test:")
print(f"  T-map range: [{paired_result.t_map.min():.2f}, {paired_result.t_map.max():.2f}]")

# Signal in first 15 voxels
mean_t_signal = np.mean(paired_result.t_map[:15])
mean_t_noise = np.mean(paired_result.t_map[15:])
print(f"  Mean t (signal): {mean_t_signal:.2f}")
print(f"  Mean t (noise): {mean_t_noise:.2f}")
```

## Two-Sample T-Test (Group Comparison)

Compare two groups (e.g., patients vs controls).

```{code-cell} python3
# Create single-condition data for 20 subjects
np.random.seed(42)
betas_groups = np.random.randn(20, 100)

# First 10 subjects (patients) have stronger activation
betas_groups[:10, :25] += 1.0

# Create PooledData (squeeze to 2D for single condition)
pool_groups = PooledData(data=betas_groups, param="beta")

# Group labels: 0=patient, 1=control
groups = np.array([0] * 10 + [1] * 10)

# Two-sample t-test
group_result = pool_groups.fit(model="ttest", X=groups)

print(f"Two-sample t-test (patients vs controls):")
print(f"  T-map range: [{group_result.t_map.min():.2f}, {group_result.t_map.max():.2f}]")

# Signal in first 25 voxels
mean_t_signal = np.mean(np.abs(group_result.t_map[:25]))
mean_t_noise = np.mean(np.abs(group_result.t_map[25:]))
print(f"  Mean |t| (signal): {mean_t_signal:.2f}")
print(f"  Mean |t| (noise): {mean_t_noise:.2f}")
```

## Contrast Syntax

Contrasts are specified as strings with condition names:

| Syntax | Meaning |
|--------|---------|
| `"A-B"` | A minus B |
| `"A+B-C-D"` | (A + B) minus (C + D) |
| `"A"` | Just A (vs implicit baseline of 0) |

Condition names must match those in `PooledData.condition_names`.

```{code-cell} python3
# Examples of valid contrasts
pool_demo = PooledData(
    data=np.random.randn(10, 4, 50),
    param="beta",
    condition_names=["face", "house", "object", "scrambled"]
)

# Simple subtraction
r1 = pool_demo.fit(model="ttest", contrast="face-house")
print(f"face-house: contrast = {r1.contrast}")

# Multiple additions/subtractions
r2 = pool_demo.fit(model="ttest", contrast="face+house-object-scrambled")
print(f"face+house-object-scrambled: computed successfully")

# Just one condition (vs 0)
r3 = pool_demo.fit(model="ttest", contrast="face")
print(f"face: tests whether face betas differ from 0")
```

## Full Workflow Example

Putting it all together with realistic analysis steps.

```{code-cell} python3
# Simulate a complete experiment:
# - 10 subjects
# - 3 conditions: face, house, scrambled
# - 200 voxels (FFA-like region in first 30)

np.random.seed(123)
n_subj = 10
n_vox = 200

# Create realistic betas
betas_exp = np.random.randn(n_subj, 3, n_vox) * 0.3

# Add face-selective signal (face > house > scrambled in "FFA")
betas_exp[:, 0, :30] += 2.0 + np.random.randn(n_subj, 30) * 0.5  # Face
betas_exp[:, 1, :30] += 1.0 + np.random.randn(n_subj, 30) * 0.5  # House
betas_exp[:, 2, :30] += 0.2 + np.random.randn(n_subj, 30) * 0.5  # Scrambled

# Create PooledData
pool_exp = PooledData(
    data=betas_exp,
    param="beta",
    condition_names=["face", "house", "scrambled"]
)

print("=" * 50)
print("EXPERIMENT: Face Processing Study")
print("=" * 50)
print(f"Subjects: {pool_exp.n_subjects}")
print(f"Conditions: {pool_exp.condition_names}")
print(f"Voxels: {pool_exp.n_voxels}")

# Run key contrasts
print("\n--- Statistical Tests ---")

# 1. Face-selective response
face_vs_scram = pool_exp.fit(model="ttest", contrast="face-scrambled")
face_vs_scram_thr = face_vs_scram.threshold(method="fdr", alpha=0.05)
n_ffa = np.sum(face_vs_scram_thr.t_map != 0)
print(f"Face > Scrambled: {n_ffa} voxels (FDR q=0.05)")

# 2. Face vs House (face selectivity)
face_vs_house = pool_exp.fit(model="ttest", contrast="face-house")
face_vs_house_thr = face_vs_house.threshold(method="fdr", alpha=0.05)
n_face_sel = np.sum(face_vs_house_thr.t_map != 0)
print(f"Face > House: {n_face_sel} voxels (FDR q=0.05)")

# 3. Any category effect (ANOVA)
anova = pool_exp.fit(model="anova")
print(f"ANOVA max F: {anova.f_map.max():.1f}")

print("\n--- Interpretation ---")
print(f"FFA-like region (voxels 0-30): {n_ffa} responsive to faces")
print("Analysis complete!")
```

## Summary

| Method | Description |
|--------|-------------|
| `PooledData(data, condition_names)` | Create pooled first-level data |
| `pool.fit(model='ttest', contrast='A-B')` | One-sample t-test on contrast |
| `pool.fit(model='ttest', contrasts=[...])` | Multiple contrasts → ResultDict |
| `pool.fit(model='ttest', X=groups)` | Two-sample t-test |
| `pool.fit(model='paired_ttest')` | Paired t-test (2 conditions) |
| `pool.fit(model='anova')` | F-test across 3+ conditions |
| `result.threshold(method='fdr')` | Apply FDR correction |
| `results.threshold_all(method='fdr')` | Batch threshold all contrasts |

## The Full Pipeline

With a real `BrainCollection`:

```python
# Complete two-stage workflow
result = (
    bc.fit(model='glm', X=design_matrices)
    .pool(param='beta')
    .fit(model='ttest', contrast='face-house')
    .threshold(method='fdr', alpha=0.05)
)

# Save to NIfTI
result.to_nifti().to_filename('face_vs_house_fdr.nii.gz')
```

## Next Steps

- **[Multi-Subject Decoding](08_multi_subject_decoding)**: LOSO CV with alignment
- **[ISC Analysis](09_isc_analysis)**: Inter-subject correlation
- **[RSA Analysis](10_rsa_analysis)**: Representational similarity
