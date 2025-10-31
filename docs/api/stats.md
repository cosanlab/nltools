# `nltools.stats`

Core statistical functions for neuroimaging analyses.

## Overview

The `nltools.stats` module provides a comprehensive collection of statistical functions for analyzing neuroimaging data. It includes univariate and multivariate methods, permutation tests, alignment algorithms, and specialized functions for timeseries analysis.

## Key Functions

**Regression & GLM**
- `regress()` - Univariate and multivariate regression
- `align()` - Functional alignment (Procrustes, SRM)

**Correlation & Similarity**
- `correlation()` - Pearson/Spearman correlation
- `distance_correlation()` - Distance correlation
- `isc()` - Inter-subject correlation
- `isc_group()` - Group-level ISC
- `isfc()` - Inter-subject functional connectivity
- `isps()` - Inter-subject phase synchrony

**Permutation Tests**
- `one_sample_permutation()` - One-sample permutation test
- `two_sample_permutation()` - Two-sample permutation test
- `correlation_permutation()` - Correlation permutation test
- `matrix_permutation()` - Matrix permutation test

**Data Transformation**
- `fisher_r_to_z()` - Fisher r-to-z transformation
- `fisher_z_to_r()` - Fisher z-to-r transformation
- `zscore()` - Z-score normalization
- `threshold()` - Statistical thresholding
- `fdr()` - False discovery rate correction

**Signal Processing**
- `downsample()` - Temporal downsampling
- `upsample()` - Temporal upsampling
- `find_spikes()` - Spike detection in timeseries

## Quick Start

```python
from nltools.stats import (
    regress,
    correlation_permutation,
    isc,
    fisher_r_to_z
)

# Regression
stats = regress(X, y, mode='ols')

# Permutation test
p_value = correlation_permutation(x, y, n_permutations=5000)

# Inter-subject correlation
isc_values = isc(data, metric='correlation')

# Fisher transformation
z_values = fisher_r_to_z(r_values)
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.stats
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - BrainData.fit() and BrainData.regress()
- {doc}`models` - Ridge and other model classes
- {doc}`algorithms` - Optimized algorithms (HyperAlignment, SRM)
- {doc}`analysis` - ROC analysis tools