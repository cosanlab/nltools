# `nltools.stats`

Core statistical functions for neuroimaging analyses.

## Overview

The `nltools.stats` module provides a comprehensive collection of statistical functions for analyzing neuroimaging data. It includes univariate and multivariate methods, permutation tests, alignment algorithms, and specialized functions for timeseries analysis.

**Note**: Several functions have been migrated to the `nltools.algorithms.inference` module for improved performance and GPU acceleration. Wrappers are maintained for backward compatibility.

## Key Functions

**Regression & GLM**
- `align()` - Functional alignment (Procrustes, SRM)

**Correlation & Similarity**
- `distance_correlation()` - Distance correlation (re-exported from inference module)
- `isc()` - Inter-subject correlation (wrapper for inference module)
- `isc_group()` - Group-level ISC (wrapper for inference module)
- `isfc()` - Inter-subject functional connectivity (uses inference module internally)
- `isps()` - Inter-subject phase synchrony

**Permutation Tests** (wrappers for inference module)
- `one_sample_permutation()` - One-sample permutation test (wrapper)
- `two_sample_permutation()` - Two-sample permutation test (wrapper)
- `correlation_permutation()` - Correlation permutation test (wrapper)
- `matrix_permutation()` - Matrix permutation test (wrapper)

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

**Matrix Utilities** (re-exported from inference module)
- `double_center()` - Double centering for distance matrices
- `u_center()` - U-centering for distance correlation

## Migration Notes

**Functions Removed** (use alternatives):
- `regress()` → Use `nltools.models.Glm` or `BrainData.fit(model='glm')`
- `regress_permutation()` → Use `one_sample_permutation_test()` from inference module
- `correlation()` → Use `correlation_permutation_test()` from inference module
- `pearson()` → Use `scipy.stats.pearsonr` or `correlation_permutation_test()`

**Recommended Migration**:
For new code, use the inference module directly for better performance (4-8× CPU, 10-100× GPU speedup):
```python
from nltools.algorithms.inference import (
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    isc_permutation_test,
    isc_group_permutation_test,
)
```

See the [Migration Guide](../migration-guide.md#pattern-9-stats-py-inference-module-migration) for details.

## Quick Start

```python
from nltools.stats import (
    correlation_permutation,  # Wrapper for inference module
    isc,  # Wrapper for inference module
    fisher_r_to_z
)

# Correlation permutation test (uses inference module internally)
p_value = correlation_permutation(x, y, n_permutations=5000)

# Inter-subject correlation (uses inference module internally)
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

- {doc}`data/brain_data` - BrainData.fit() for GLM analysis
- {doc}`models` - Ridge and other model classes
- {doc}`algorithms` - Optimized algorithms (HyperAlignment, SRM, Inference)
- {doc}`analysis` - ROC analysis tools
- {doc}`../migration-guide` - Migration guide for v0.6.0 changes