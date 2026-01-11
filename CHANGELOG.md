# Changelog

All notable changes to nltools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Pipeline Infrastructure (New)
Fluent API for building data processing pipelines with cross-validation support.

- **Core Pipeline Classes**
  - `Pipeline`: Base pipeline with immutable step accumulation and CV context
  - `CVScheme`: Cross-validation configuration (kfold, loso, loro, bootstrap)
  - `FittedStack`: Collection of fitted transforms for inverse transform support

- **Transform Steps**
  - `NormalizeStep`: Z-score and min-max normalization with inverse transform
  - `ReduceStep`: PCA and ICA dimensionality reduction
  - `PipeStep`: Wrapper for arbitrary sklearn transformers
  - `AlignStep`: Cross-subject alignment via SRM or HyperAlignment

- **Terminal Methods**
  - `PredictTerminal`: Classification/regression with 7 algorithms (ridge, lasso, svm, logistic, rf, svr, elastic)
  - Statistical tests: t-test (one-sample, two-sample, paired), ANOVA, with FDR/Bonferroni correction

- **Multi-Subject Support**
  - `MultiSubjectPipeline`: LOSO and run-based CV for group analyses
  - `BrainCollection.cv()`: Entry point for multi-subject pipelines
  - `BrainCollection.fit().pool()`: Two-stage analysis workflow

- **Pool Infrastructure**
  - `PooledData`: Aggregated multi-subject data for group-level analysis
  - `StatResult`: Statistical maps with `threshold()` method
  - `ResultDict`: Batch operations on multiple contrasts

- **Results**
  - `CVResult`: Aggregated CV results with scores, predictions, inverse transform
  - `FoldResult`: Per-fold results with fitted transforms

#### Example Workflows

```python
# Single-subject CV with preprocessing
result = (
    brain.cv(k=5)
    .normalize()
    .reduce(n_components=50)
    .predict(y, algorithm='svm')
)

# Two-stage GLM (group analysis)
result = (
    bc.fit(model='glm', X=designs)
    .pool(param='beta')
    .fit(model='ttest', contrast='face-house')
)

# LOSO with alignment
result = (
    MultiSubjectPipeline(data=subjects, cv=CVScheme(scheme='loso'))
    .normalize()
    .align(method='srm', n_features=50)
    .predict(y=labels, algorithm='svm')
)

# Multiple contrasts
results = pool.fit(model='ttest', contrasts=['A-B', 'A-C', 'B-C'])
results['A-B'].threshold(method='fdr', alpha=0.05)
```

### Changed
- `BrainData` and `BrainCollection` now support `.cv()` method for pipeline entry
- `BrainCollection.fit()` returns `FittedBrainCollection` with `.pool()` method

### Notes
- Pipeline infrastructure adds 153 new tests
- Searchlight/piecewise alignment schemes pending LocalAlignment implementation
