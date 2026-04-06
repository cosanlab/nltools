---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
execute:
  skip: true
---

# Multi-Subject Classification with LOSO CV

When classifying subjects into groups (e.g., patients vs. controls), we use
**Leave-One-Subject-Out (LOSO) cross-validation**. This is the gold standard
for evaluating classifiers that generalize to new individuals.

## Learning Objectives

- Use `MultiSubjectPipeline` for group classification
- Understand Leave-One-Subject-Out (LOSO) cross-validation
- Compare different classifiers on multi-subject data
- Interpret cross-subject decoding results

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from nltools.pipelines.multi_subject import MultiSubjectPipeline
from nltools.pipelines.cv import CVScheme
```

## The Subject Classification Problem

Consider a typical clinical neuroimaging study:

```
Subject 1 (Patient):  [brain patterns] → Patient
Subject 2 (Control):  [brain patterns] → Control
Subject 3 (Patient):  [brain patterns] → Patient
...
```

**LOSO cross-validation** tests if the classifier can correctly identify
a held-out subject's group based on their brain patterns.

```
Fold 1: Train on S2,S3,S4,S5,S6 → Test on S1
Fold 2: Train on S1,S3,S4,S5,S6 → Test on S2
Fold 3: Train on S1,S2,S4,S5,S6 → Test on S3
...
```

## Creating Multi-Subject Data

Let's simulate data from two groups with distinct brain patterns.

```{code-cell} python3
np.random.seed(42)

n_subjects = 8  # 4 per group
n_timepoints = 40  # e.g., 40 TRs of resting state
n_voxels = 100

# Create subject data with group differences
subject_data = []
for subj in range(n_subjects):
    # Base pattern with noise
    data = np.random.randn(n_timepoints, n_voxels) * 0.5

    # Group A (subjects 0-3): higher activation in first 30 voxels
    # Group B (subjects 4-7): lower activation in first 30 voxels
    if subj < 4:
        data[:, :30] += 1.0  # Group A effect
    else:
        data[:, :30] -= 0.5  # Group B effect

    subject_data.append(data)

# Subject-level labels: one label per subject
# Group A = 0, Group B = 1
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

print(f"Number of subjects: {len(subject_data)}")
print(f"Data shape per subject: {subject_data[0].shape}")
print(f"Subject labels: {labels}")
print(f"  Group A (label=0): {np.sum(labels == 0)} subjects")
print(f"  Group B (label=1): {np.sum(labels == 1)} subjects")
```

## Creating a LOSO Pipeline

`CVScheme(scheme='loso')` creates Leave-One-Subject-Out cross-validation.

```{code-cell} python3
# Create LOSO CV scheme
cv_loso = CVScheme(scheme="loso")

print(f"CV scheme: {cv_loso.scheme}")
print(f"Number of folds: {n_subjects} (one per subject)")
```

## Basic Multi-Subject Classification

```{code-cell} python3
# Create pipeline with LOSO CV
pipeline = MultiSubjectPipeline(
    data=subject_data,
    cv=cv_loso
)

print(f"Pipeline: {pipeline}")
print(f"  n_subjects: {pipeline.n_subjects}")
print(f"  n_steps: {pipeline.n_steps}")
```

```{code-cell} python3
# Run classification with normalization
result = (
    pipeline
    .normalize()
    .predict(y=labels, algorithm="svm")
)

print(f"\nResults:")
print(f"  Mean accuracy: {result.mean_score:.1%}")
print(f"  Std accuracy: {result.std_score:.1%}")
print(f"  Per-fold scores: {[f'{s:.1%}' for s in result.scores]}")
```

## Comparing Algorithms

Try different classification algorithms to find the best performer.

```{code-cell} python3
algorithms = ["ridge", "svm"]
results = {}

for algo in algorithms:
    result = (
        MultiSubjectPipeline(data=subject_data, cv=cv_loso)
        .normalize()
        .predict(y=labels, algorithm=algo)
    )
    results[algo] = result
    print(f"{algo:10s}: {result.mean_score:.1%} (+/- {result.std_score:.1%})")
```

## Adding Dimensionality Reduction

For high-dimensional data, PCA can improve classification.

```{code-cell} python3
# Pipeline with PCA
result_pca = (
    MultiSubjectPipeline(data=subject_data, cv=cv_loso)
    .normalize()
    .reduce(n_components=20)
    .predict(y=labels, algorithm="svm")
)

print(f"\nWith PCA (20 components):")
print(f"  Accuracy: {result_pca.mean_score:.1%}")
print(f"  Per-fold: {[f'{s:.1%}' for s in result_pca.scores]}")
```

## Real-World Example: Patient vs Control

A common use case for LOSO is distinguishing clinical groups.

```{code-cell} python3
np.random.seed(123)

# 12 subjects: 6 patients, 6 controls
n_subj = 12
n_timepoints = 50
n_vox = 80

# Simulate group differences in functional connectivity patterns
subject_data_clinical = []
for s in range(n_subj):
    # Base pattern with subject-specific noise
    data = np.random.randn(n_timepoints, n_vox) * 0.5

    # Patients (subjects 0-5): reduced connectivity in frontal regions
    # Controls (subjects 6-11): normal connectivity
    if s < 6:
        data[:, :20] -= 0.8  # Reduced activation
        data[:, :20] += np.random.randn(n_timepoints, 20) * 0.3
    else:
        data[:, :20] += 0.8  # Normal activation
        data[:, :20] += np.random.randn(n_timepoints, 20) * 0.3

    subject_data_clinical.append(data)

# Subject-level labels: 0=patient, 1=control
labels_clinical = np.array([0] * 6 + [1] * 6)

print(f"Subjects: {len(subject_data_clinical)}")
print(f"  Patients (label=0): {np.sum(labels_clinical == 0)}")
print(f"  Controls (label=1): {np.sum(labels_clinical == 1)}")
```

```{code-cell} python3
# Run classification
print("\n" + "=" * 50)
print("CLINICAL CLASSIFICATION (Patient vs Control)")
print("=" * 50)

result_clinical = (
    MultiSubjectPipeline(data=subject_data_clinical, cv=CVScheme(scheme="loso"))
    .normalize()
    .reduce(n_components=30)
    .predict(y=labels_clinical, algorithm="svm")
)

print(f"Accuracy: {result_clinical.mean_score:.1%} (chance = 50%)")
print(f"Per-subject results:")
for i, score in enumerate(result_clinical.scores):
    group = "Patient" if labels_clinical[i] == 0 else "Control"
    correct = "✓" if score > 0.5 else "✗"
    print(f"  Subject {i+1} ({group}): {correct}")
```

## Visualizing Results

```{code-cell} python3
# Bar plot of per-fold accuracy
fig, ax = plt.subplots(figsize=(10, 4))

x = np.arange(len(result_clinical.scores))
colors = ['coral' if labels_clinical[i] == 0 else 'steelblue' for i in x]

ax.bar(x, [s * 100 for s in result_clinical.scores], color=colors)
ax.axhline(y=50, color='red', linestyle='--', label='Chance')
ax.axhline(y=result_clinical.mean_score * 100, color='green', linestyle='-',
           linewidth=2, label=f'Mean: {result_clinical.mean_score:.1%}')
ax.set_xlabel('Subject (held out)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('LOSO Cross-Validation: Per-Subject Accuracy')
ax.set_xticks(x)
ax.set_xticklabels([f'S{i+1}' for i in x])
ax.legend()
ax.set_ylim(0, 110)

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='coral', label='Patient'),
                   Patch(facecolor='steelblue', label='Control')]
ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0],
          loc='upper right')

plt.tight_layout()
plt.close()

print("\nPlot created: LOSO per-subject accuracy")
```

## Understanding LOSO Results

LOSO provides several insights:

1. **Mean accuracy**: Overall classifier performance
2. **Per-fold scores**: Which subjects are harder to classify
3. **Consistency**: Low variance = robust classifier

### Interpreting the Scores

- Each fold tests on one subject, training on all others
- A score of 1.0 means the held-out subject was correctly classified
- A score of 0.0 means the subject was misclassified
- For binary classification, 0.5 is chance level

## Pipeline Steps

The `MultiSubjectPipeline` supports these preprocessing steps:

| Method | Description |
|--------|-------------|
| `.normalize()` | Z-score each subject's data |
| `.reduce(n=50)` | PCA dimensionality reduction |
| `.pipe(transformer)` | Custom sklearn transformer |

All transforms are fit on training subjects and applied to the test subject.

## Summary

| Method | Description |
|--------|-------------|
| `MultiSubjectPipeline(data, cv)` | Create multi-subject pipeline |
| `CVScheme(scheme='loso')` | Leave-one-subject-out CV |
| `.normalize()` | Z-score each subject's data |
| `.reduce(n_components=50)` | PCA dimensionality reduction |
| `.predict(y, algorithm='svm')` | Execute and get results |

## The Full Workflow

```python
# Complete multi-subject classification pipeline
result = (
    MultiSubjectPipeline(data=subject_data, cv=CVScheme(scheme='loso'))
    .normalize()
    .reduce(n_components=50)
    .predict(y=labels, algorithm='svm')
)

print(f"Cross-subject accuracy: {result.mean_score:.1%}")
```

## Next Steps

- **[ISC Analysis](09_isc_analysis)**: Inter-subject correlation
- **[RSA Analysis](10_rsa_analysis)**: Representational similarity
- **[Pipeline Basics](06_pipeline_basics)**: Single-subject CV pipelines
