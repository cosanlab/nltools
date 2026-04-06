---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Pipeline Basics

nltools v0.6.0 introduces a **fluent pipeline API** for building cross-validated
analysis workflows. Instead of configuring everything in a single function call,
you chain operations together in a readable, composable way.

## Learning Objectives

- Understand the fluent pipeline pattern
- Use `.cv()` to create cross-validation pipelines
- Chain preprocessing steps (`.normalize()`, `.reduce()`)
- Execute predictions with `.predict()`
- Interpret `CVResult` objects

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from nltools.data import BrainData
from nltools.datasets import fetch_haxby
```

## Load Data

We'll use the Haxby dataset - a classic fMRI study of visual object recognition.

```{code-cell} python3
data, design = fetch_haxby(n_subjects=1, verbose=0)
brain = data[0]  # First run

print(f"Data shape: {brain.shape} (timepoints x voxels)")
```

## The Old Way vs The New Way

### Old API (still works, but deprecated for MVPA)

```python
# Everything in one call - hard to read, hard to extend
result = brain.predict(y=labels, cv=5, standardize=True, method='whole_brain')
```

### New Fluent API

```python
# Chain operations clearly - easy to read, easy to extend
result = (
    brain.cv(k=5)
    .normalize()
    .predict(y=labels, algorithm='svm')
)
```

The new API separates concerns: CV setup, preprocessing, and prediction are
distinct steps that can be mixed and matched.

## Creating a Cross-Validation Pipeline

The `.cv()` method creates a pipeline with cross-validation context.

```{code-cell} python3
# Create binary classification labels
# First half = category A, second half = category B
n_samples = brain.shape[0]
labels = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))

print(f"Labels: {len(labels)} samples, classes: {np.unique(labels)}")
```

```{code-cell} python3
# Create a 5-fold CV pipeline
pipeline = brain.cv(k=5)

print(f"Pipeline: {pipeline}")
print(f"Number of steps: {pipeline.n_steps}")
```

## Adding Preprocessing Steps

Chain preprocessing steps with method calls. Each step returns a new pipeline
(immutable pattern), so you can build pipelines incrementally.

```{code-cell} python3
# Add normalization (z-score each voxel)
pipeline_normalized = brain.cv(k=5).normalize()
print(f"After normalize: {pipeline_normalized.n_steps} steps")

# Add dimensionality reduction (PCA to 50 components)
pipeline_reduced = brain.cv(k=5).normalize().reduce(n_components=50)
print(f"After reduce: {pipeline_reduced.n_steps} steps")
```

## Executing the Pipeline

Terminal methods like `.predict()` execute the full pipeline and return results.

```{code-cell} python3
# Full pipeline: CV -> normalize -> reduce -> predict
import warnings

# Suppress the deprecation warning for demo purposes
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

    result = (
        brain.cv(k=5)
        .normalize()
        .reduce(n_components=50)
        .predict(y=labels, algorithm='svm')
    )

print(f"Result type: {type(result).__name__}")
print(f"Mean accuracy: {result.mean_score:.1%}")
print(f"Std accuracy: {result.std_score:.1%}")
```

## Understanding CVResult

The `CVResult` object contains detailed information about each fold.

```{code-cell} python3
# Per-fold scores
print("Per-fold scores:")
for i, score in enumerate(result.scores):
    print(f"  Fold {i+1}: {score:.1%}")
```

```{code-cell} python3
# All predictions (in original sample order)
predictions = result.predictions
print(f"\nPredictions shape: {predictions.shape}")
print(f"Prediction distribution: {np.bincount(predictions.astype(int))}")
```

## Available Algorithms

The `.predict()` method supports these algorithms:

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| `'ridge'` | Ridge regression | Continuous outcomes (default) |
| `'lasso'` | Lasso regression | Sparse solutions |
| `'svr'` | Support Vector Regression | Nonlinear regression |
| `'svm'` | Support Vector Classification | Binary/multiclass |

```{code-cell} python3
# Try different algorithms
algorithms = ['ridge', 'svm', 'svr', 'lasso']

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for algo in algorithms:
        result = (
            brain.cv(k=3)
            .normalize()
            .reduce(n_components=30)
            .predict(y=labels, algorithm=algo)
        )
        print(f"{algo:10s}: {result.mean_score:.1%} (+/- {result.std_score:.1%})")
```

## CV Schemes

Different cross-validation schemes are available:

- `k=5` or `scheme='kfold'`: Standard k-fold
- `scheme='loro'`: Leave-one-run-out (requires `groups`)
- `scheme='bootstrap'`: Bootstrap with out-of-bag testing

```{code-cell} python3
# Leave-one-run-out CV (if you have run labels)
# Simulating run labels: 4 runs of equal length
n_runs = 4
run_labels = np.repeat(np.arange(n_runs), n_samples // n_runs)

# Pad if needed
if len(run_labels) < n_samples:
    run_labels = np.concatenate([run_labels, [n_runs - 1] * (n_samples - len(run_labels))])

print(f"Run labels: {np.bincount(run_labels)}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    result_loro = (
        brain.cv(scheme='loro', groups=run_labels)
        .normalize()
        .predict(y=labels, algorithm='svm')
    )

print(f"LORO accuracy: {result_loro.mean_score:.1%} ({len(result_loro.scores)} folds)")
```

## Preprocessing Options

### Normalization Methods

```{code-cell} python3
# Z-score normalization (default)
pipeline1 = brain.cv(k=3).normalize(method='zscore')

# Min-max normalization
pipeline2 = brain.cv(k=3).normalize(method='minmax')

print("Normalization methods: 'zscore' (default), 'minmax'")
```

### Dimensionality Reduction Methods

```{code-cell} python3
# PCA (default)
pipeline1 = brain.cv(k=3).reduce(method='pca', n_components=50)

# ICA
pipeline2 = brain.cv(k=3).reduce(method='ica', n_components=20)

print("Reduction methods: 'pca' (default), 'ica'")
```

## Custom Transformers

Use `.pipe()` to add any sklearn-compatible transformer.

```{code-cell} python3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add custom sklearn transformers
pipeline = (
    brain.cv(k=3)
    .pipe(StandardScaler())
    .pipe(PCA(n_components=50))
)

print(f"Custom pipeline: {pipeline.n_steps} steps")
```

## Benefits of the Fluent API

1. **Readable**: Each step is clearly visible
2. **Composable**: Mix and match preprocessing steps
3. **Reusable**: Save pipeline configurations
4. **Testable**: Each step can be tested independently
5. **Extensible**: Add custom transformers with `.pipe()`

The pipeline handles train/test separation correctly - all transforms are
fit on training data and applied to test data, preventing data leakage.

## Summary

| Method | Description |
|--------|-------------|
| `.cv(k=5)` | Create CV pipeline with k folds |
| `.normalize()` | Add z-score normalization |
| `.reduce(n=50)` | Add PCA dimensionality reduction |
| `.pipe(transformer)` | Add custom sklearn transformer |
| `.predict(y, algorithm)` | Execute pipeline and get results |

## Next Steps

- **[Two-Stage GLM](07_two_stage_glm)**: Group-level analysis with pooling
- **[Multi-Subject Decoding](08_multi_subject_decoding)**: LOSO with alignment
- **[ISC Analysis](09_isc_analysis)**: Inter-subject correlation
