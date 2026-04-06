---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Inter-Subject Correlation (ISC) Analysis

Inter-Subject Correlation measures the similarity of brain responses across
subjects viewing the same naturalistic stimuli (movies, audio stories, etc.).
High ISC indicates that subjects share a common neural response to the stimulus.

## Learning Objectives

- Understand what ISC measures
- Use `isc_permutation_test()` for statistical inference
- Choose between leave-one-out and pairwise methods
- Apply different null hypothesis methods (bootstrap, circle_shift)

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from nltools.algorithms.inference.isc import isc_permutation_test
```

## What is ISC?

When subjects view the same movie or listen to the same story, some brain
regions show correlated time courses across subjects:

```
Subject 1: [t1, t2, t3, t4, ...]  ─┐
Subject 2: [t1, t2, t3, t4, ...]  ─┼─> Correlated? → ISC
Subject 3: [t1, t2, t3, t4, ...]  ─┘
```

- **High ISC**: Region responds reliably to stimulus (e.g., visual cortex)
- **Low ISC**: Response varies across subjects (e.g., mind-wandering)

## Summary Statistics

Two methods to summarize cross-subject correlations:

- **Leave-One-Out (LOO)**: Correlate each subject with the mean of others
- **Pairwise**: Correlate all pairs of subjects, take median

## Simulating Naturalistic Data

Let's create data where some "voxels" have shared signal (high ISC) and
others have independent noise (low ISC).

```{code-cell} python3
np.random.seed(42)

n_timepoints = 100  # e.g., 100 TRs of movie
n_subjects = 15
n_voxels = 50

# Create shared signal (the "stimulus response")
shared_signal = np.random.randn(n_timepoints)

# Create data: first 20 voxels have shared signal, rest is noise
data = np.zeros((n_timepoints, n_subjects, n_voxels))

for subj in range(n_subjects):
    # Voxels 0-19: shared signal + noise (high ISC)
    for v in range(20):
        data[:, subj, v] = shared_signal + np.random.randn(n_timepoints) * 0.5

    # Voxels 20-49: pure noise (low ISC)
    for v in range(20, n_voxels):
        data[:, subj, v] = np.random.randn(n_timepoints)

print(f"Data shape: {data.shape}")
print(f"  Timepoints: {n_timepoints}")
print(f"  Subjects: {n_subjects}")
print(f"  Voxels: {n_voxels}")
print(f"  High-ISC voxels: 0-19")
print(f"  Low-ISC voxels: 20-49")
```

## Basic ISC Analysis

Run ISC with permutation testing for statistical significance.

```{code-cell} python3
# Run ISC analysis
result = isc_permutation_test(
    data,
    summary_statistic="pairwise",  # Use pairwise correlations
    n_permute=1000,                # Number of permutations
    random_state=42,
    progress_bar=False
)

print("ISC Results:")
print(f"  ISC values shape: {result['isc'].shape}")
print(f"  P-values shape: {result['p'].shape}")
print(f"  CI shape: ({result['ci'][0].shape}, {result['ci'][1].shape})")
```

## Interpreting Results

The result dictionary contains:

| Key | Description |
|-----|-------------|
| `isc` | ISC value per voxel |
| `p` | P-value per voxel (one-tailed, positive ISC) |
| `ci` | 95% confidence interval (lower, upper) |

```{code-cell} python3
# Look at ISC values
print("ISC values by region:")
print(f"  High-ISC voxels (0-19): mean ISC = {result['isc'][:20].mean():.3f}")
print(f"  Low-ISC voxels (20-49): mean ISC = {result['isc'][20:].mean():.3f}")

# Look at p-values
sig_voxels = np.sum(result['p'] < 0.05)
sig_high_isc = np.sum(result['p'][:20] < 0.05)
sig_low_isc = np.sum(result['p'][20:] < 0.05)

print(f"\nSignificant voxels (p < 0.05):")
print(f"  Total: {sig_voxels}/{n_voxels}")
print(f"  High-ISC region: {sig_high_isc}/20")
print(f"  Low-ISC region: {sig_low_isc}/30")
```

## Visualizing ISC Results

```{code-cell} python3
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# ISC values
ax1 = axes[0]
colors = ['steelblue' if i < 20 else 'gray' for i in range(n_voxels)]
ax1.bar(range(n_voxels), result['isc'], color=colors)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_ylabel('ISC')
ax1.set_title('Inter-Subject Correlation by Voxel')
ax1.set_xlim(-1, n_voxels)

# P-values (log scale)
ax2 = axes[1]
ax2.bar(range(n_voxels), -np.log10(result['p'] + 1e-10), color=colors)
ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
ax2.set_xlabel('Voxel')
ax2.set_ylabel('-log10(p)')
ax2.set_title('Statistical Significance')
ax2.legend()
ax2.set_xlim(-1, n_voxels)

plt.tight_layout()
plt.close()

print("Plot created: ISC values and p-values")
```

## Leave-One-Out vs Pairwise

| Method | Description | Best For |
|--------|-------------|----------|
| **Leave-One-Out** | Correlate each subject with group mean | Larger samples, robust |
| **Pairwise** | Median of all pairwise correlations | Smaller samples |

```{code-cell} python3
# Compare methods on the same data
result_loo = isc_permutation_test(
    data,
    summary_statistic="leave-one-out",
    n_permute=500,
    random_state=42,
    progress_bar=False
)

result_pairwise = isc_permutation_test(
    data,
    summary_statistic="pairwise",
    n_permute=500,
    random_state=42,
    progress_bar=False
)

print("Comparison of ISC methods:")
print(f"  LOO - High-ISC mean: {result_loo['isc'][:20].mean():.3f}")
print(f"  LOO - Low-ISC mean: {result_loo['isc'][20:].mean():.3f}")
print(f"  Pairwise - High-ISC mean: {result_pairwise['isc'][:20].mean():.3f}")
print(f"  Pairwise - Low-ISC mean: {result_pairwise['isc'][20:].mean():.3f}")
```

## Null Hypothesis Methods

Different methods for generating the null distribution:

| Method | Description | Preserves |
|--------|-------------|-----------|
| `bootstrap` | Resample subjects | Default, general purpose |
| `circle_shift` | Circularly shift time series | Autocorrelation |
| `phase_randomize` | Randomize phases in Fourier domain | Power spectrum |

```{code-cell} python3
# Bootstrap (default)
result_boot = isc_permutation_test(
    data[:, :, :5],  # Subset for speed
    method="bootstrap",
    n_permute=500,
    random_state=42,
    progress_bar=False
)

# Circle shift (preserves autocorrelation)
result_circle = isc_permutation_test(
    data[:, :, :5],
    method="circle_shift",
    n_permute=500,
    random_state=42,
    progress_bar=False
)

print("Null hypothesis methods:")
print(f"  Bootstrap - p-values: {result_boot['p']}")
print(f"  Circle shift - p-values: {result_circle['p']}")
```

## Single Voxel Analysis

For quick exploration, analyze a single voxel/ROI.

```{code-cell} python3
# Extract data for one high-ISC voxel
single_voxel_data = data[:, :, 0]  # Shape: (timepoints, subjects)

result_single = isc_permutation_test(
    single_voxel_data,
    n_permute=2000,
    random_state=42,
    return_null=True,
    progress_bar=False
)

print(f"Single voxel analysis:")
print(f"  ISC = {float(result_single['isc']):.3f}")
print(f"  p-value = {float(result_single['p']):.4f}")
print(f"  95% CI = [{float(result_single['ci'][0]):.3f}, {float(result_single['ci'][1]):.3f}]")
```

## Getting the Null Distribution

Use `return_null=True` to get the full bootstrap distribution.

```{code-cell} python3
# Visualize null distribution
fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(result_single['null_dist'], bins=50, alpha=0.7, color='gray', label='Null distribution')
ax.axvline(x=result_single['isc'], color='red', linewidth=2, label=f"Observed ISC = {result_single['isc']:.3f}")
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('ISC')
ax.set_ylabel('Count')
ax.set_title('ISC Permutation Test: Null Distribution')
ax.legend()

plt.tight_layout()
plt.close()

print("Plot created: Null distribution with observed ISC")
```

## Similarity Metrics

Besides correlation, ISC can use other similarity metrics.

```{code-cell} python3
# Compare similarity metrics
metrics = ["correlation", "spearman", "cosine"]

print("ISC with different similarity metrics:")
for metric in metrics:
    result = isc_permutation_test(
        data[:, :, :5],
        summary_statistic="pairwise",
        sim_metric=metric,
        n_permute=200,
        random_state=42,
        progress_bar=False
    )
    print(f"  {metric:12s}: mean ISC = {result['isc'].mean():.3f}")
```

## Best Practices

1. **Use enough permutations**: 5000+ for publication-quality p-values
2. **Choose appropriate method**:
   - Bootstrap for general use
   - Circle shift for autocorrelated data (fMRI)
   - Phase randomize for frequency-domain analyses
3. **Multiple comparison correction**: Apply FDR for voxel-wise maps
4. **Report effect sizes**: ISC values, not just p-values

## Summary

| Function | Description |
|----------|-------------|
| `isc_permutation_test(data)` | Run ISC with permutation testing |
| `summary_statistic='pairwise'` | Median of all pairwise correlations |
| `summary_statistic='leave-one-out'` | Correlate each with group mean |
| `method='bootstrap'` | Subject resampling (default) |
| `method='circle_shift'` | Preserve autocorrelation |
| `return_null=True` | Get full null distribution |

## The Full Workflow

```python
from nltools.algorithms.inference.isc import isc_permutation_test

# Run ISC analysis
result = isc_permutation_test(
    data,                           # (timepoints, subjects, voxels)
    summary_statistic="pairwise",
    method="bootstrap",
    n_permute=5000,
    random_state=42
)

# Apply FDR correction
from scipy.stats import false_discovery_control
p_fdr = false_discovery_control(result['p'])
significant = p_fdr < 0.05

print(f"ISC = {result['isc'].mean():.3f}")
print(f"Significant voxels: {significant.sum()}")
```

## Next Steps

- **[RSA Analysis](10_rsa_analysis)**: Representational similarity
- **[Multi-Subject Decoding](08_multi_subject_decoding)**: LOSO CV
- **[Pipeline Basics](06_pipeline_basics)**: Single-subject pipelines
