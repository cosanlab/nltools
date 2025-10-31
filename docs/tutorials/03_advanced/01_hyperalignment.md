# Hyperalignment for Multi-Subject Analysis

## Learning Objectives

By the end of this tutorial, you will be able to:
- Understand the principles of functional alignment
- Apply hyperalignment to multi-subject fMRI data
- Validate alignment quality
- Use hyperalignment for improved group analyses
- Apply learned transformations to new data

## Introduction

**Hyperalignment** functionally aligns brain responses across subjects by finding transformations that maximize shared response patterns. Unlike anatomical alignment (e.g., warping to MNI space), hyperalignment aligns *functional* responses.

**Key benefits**:
- Improved between-subject correspondence
- Better group-level pattern analysis
- Enables training classifiers on one subject, testing on others
- Useful for naturalistic stimuli (movies, stories)

**This tutorial demonstrates hyperalignment workflows with nltools.**

## The Problem: Inter-Subject Variability

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import BrainData
from nltools.datasets import fetch_pain
from nltools.external import Hyperalignment

# Load multi-subject data
data = fetch_pain()
print(f"Data shape: {data.shape()}")

# Split into subjects (assuming 3 volumes per subject)
n_subjects = 28
vols_per_subj = 3

subjects = []
for i in range(n_subjects):
    start_idx = i * vols_per_subj
    end_idx = start_idx + vols_per_subj
    subjects.append(data[start_idx:end_idx])

print(f"Number of subjects: {len(subjects)}")
print(f"Volumes per subject: {subjects[0].shape()[0]}")
```

## Visualize Pre-Alignment Similarity

```python
# Compute inter-subject correlation before alignment

# Get one volume from each subject
subj_patterns = [subj[0] for subj in subjects[:10]]  # Use first 10 subjects

# Compute pairwise correlations
n_subj = len(subj_patterns)
isc_pre = np.zeros((n_subj, n_subj))

for i in range(n_subj):
    for j in range(n_subj):
        # Correlation between subjects
        isc_pre[i, j] = np.corrcoef(
            subj_patterns[i].data.flatten(),
            subj_patterns[j].data.flatten()
        )[0, 1]

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(isc_pre, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
ax.set_title('Inter-Subject Correlation (Pre-Alignment)')
ax.set_xlabel('Subject')
ax.set_ylabel('Subject')
plt.colorbar(im, ax=ax, label='Correlation')
plt.tight_layout()
plt.show()

# Mean off-diagonal correlation
off_diag = isc_pre[np.triu_indices(n_subj, k=1)]
print(f"Mean ISC (pre-alignment): {off_diag.mean():.3f}")
```

## Apply Hyperalignment

```python
# Initialize hyperalignment
hyper = Hyperalignment(
    n_iter=10,     # Number of iterations
    level=1        # Complexity level
)

# Fit hyperalignment to subject data
# Convert subjects to format expected by hyperalignment
# Each subject should be timepoints × voxels

subj_data_list = [subj.data for subj in subjects]

# Fit hyperalignment
hyper.fit(subj_data_list)

print("✓ Hyperalignment complete!")
print(f"Learned {len(hyper.transformations)} transformation matrices")
```

## Transform Data to Common Space

```python
# Apply learned transformations to align subjects

aligned_subjects = []
for i, subj in enumerate(subjects):
    # Transform to common space
    aligned = hyper.transform([subj.data])[0]

    # Create BrainData object
    aligned_bd = BrainData(
        data=aligned,
        mask=subj.mask
    )
    aligned_subjects.append(aligned_bd)

print(f"Aligned {len(aligned_subjects)} subjects to common space")
```

## Visualize Post-Alignment Similarity

```python
# Compute inter-subject correlation after alignment

aligned_patterns = [aligned_subjects[i][0] for i in range(10)]

# Compute pairwise correlations
isc_post = np.zeros((n_subj, n_subj))

for i in range(n_subj):
    for j in range(n_subj):
        isc_post[i, j] = np.corrcoef(
            aligned_patterns[i].data.flatten(),
            aligned_patterns[j].data.flatten()
        )[0, 1]

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(isc_pre, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
axes[0].set_title('Pre-Alignment ISC')
axes[0].set_xlabel('Subject')
axes[0].set_ylabel('Subject')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(isc_post, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
axes[1].set_title('Post-Alignment ISC')
axes[1].set_xlabel('Subject')
axes[1].set_ylabel('Subject')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# Compare mean ISC
off_diag_post = isc_post[np.triu_indices(n_subj, k=1)]
print(f"Mean ISC (pre-alignment):  {off_diag.mean():.3f}")
print(f"Mean ISC (post-alignment): {off_diag_post.mean():.3f}")
print(f"Improvement: {(off_diag_post.mean() - off_diag.mean()):.3f}")
```

## Validation: Leave-One-Out Classification

```python
# Test if hyperalignment improves cross-subject generalization

# TODO: Implement MVPA classification example
# This would show that classifiers trained on aligned data
# generalize better to held-out subjects

print("Note: Cross-subject classification validation to be added")
```

## Apply to New Data

```python
# Apply learned transformations to independent data

# Simulate new data for same subjects
new_data_list = [np.random.randn(*subj.data.shape) for subj in subjects]

# Transform using previously learned mappings
aligned_new = []
for i, new_subj_data in enumerate(new_data_list):
    transformed = hyper.transform([new_subj_data], subject_idx=[i])[0]
    aligned_new.append(transformed)

print("✓ Applied hyperalignment to new data")
```

## Best Practices

```python
print("""
Hyperalignment Best Practices:
------------------------------

1. **Data Requirements**:
   - Rich, complex stimuli (movies, narratives)
   - Sufficient data per subject (>100 timepoints recommended)
   - Common stimulus across subjects

2. **Preprocessing**:
   - Standard preprocessing (realignment, normalization)
   - High-pass filtering
   - Z-score within session
   - Consider spatial smoothing carefully

3. **Validation**:
   - Quantify improvement in ISC
   - Test cross-subject decoding
   - Compare to anatomical alignment

4. **Applications**:
   - Group-level MVPA
   - Intersubject pattern analysis
   - Shared response modeling
   - Naturalistic data analysis

5. **Limitations**:
   - Requires same stimulus across subjects
   - Computationally intensive
   - May overfit with limited data
""")
```

## Summary

In this tutorial, you learned how to:
- ✓ Understand the motivation for functional alignment
- ✓ Apply hyperalignment to multi-subject fMRI data
- ✓ Quantify alignment quality with inter-subject correlation
- ✓ Transform new data using learned mappings
- ✓ Follow best practices for hyperalignment

## Next Steps

- **Tutorial: Shared Response Model** - Alternative alignment method
- **Tutorial: MVPA** - Use hyperalignment for cross-subject decoding
- **Advanced**: Searchlight hyperalignment
- **Advanced**: Time-segment matching for naturalistic data

## Further Reading

- Haxby et al. (2011). A common, high-dimensional model of the representational space in human ventral temporal cortex. Neuron.
- Guntupalli et al. (2016). A model of representational spaces in human cortex. Cerebral Cortex.
