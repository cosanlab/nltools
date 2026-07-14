"""nltools.stats — Statistical utilities for neuroimaging analysis.

This package provides standalone statistical functions organized into
focused submodules:

- **corrections**: Multiple comparison corrections (FDR, Holm-Bonferroni, thresholding)
- **outliers**: Outlier detection, winsorizing, z-scoring
- **timeseries**: Temporal signal processing (resampling, filtering, basis functions)
- **correlation**: Similarity metrics, Fisher transforms, ICC
- **alignment**: Data alignment (SRM, Procrustes, state alignment)
- **intersubject**: ISC, ISFC, ISPS

All public functions are re-exported here for convenience:

```python
from nltools.stats import fdr, zscore, isc  # all work
```
"""

from .corrections import fdr, holm_bonf, threshold, multi_threshold
from .outliers import zscore, winsorize, trim, find_spikes
from .timeseries import downsample, upsample, calc_bpm, make_cosine_basis
from .correlation import (
    fisher_r_to_z,
    fisher_z_to_r,
    compute_similarity,
    compute_multivariate_similarity,
    compute_icc,
    transform_pairwise,
)
from .alignment import align, procrustes, procrustes_distance, align_states
from .intersubject import isc, isc_group, isfc, isps
from .regression import regress
from .permutation import (
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    timeseries_correlation_permutation_test,
    circle_shift,
    phase_randomize,
    matrix_permutation_test,
    double_center,
    u_center,
    distance_correlation,
)

__all__ = [
    # alignment
    "align",
    "align_states",
    "calc_bpm",
    "circle_shift",
    "compute_icc",
    "compute_multivariate_similarity",
    "compute_similarity",
    "correlation_permutation_test",
    "distance_correlation",
    "double_center",
    # timeseries
    "downsample",
    # corrections
    "fdr",
    "find_spikes",
    # correlation
    "fisher_r_to_z",
    "fisher_z_to_r",
    "holm_bonf",
    # intersubject
    "isc",
    "isc_group",
    "isfc",
    "isps",
    "make_cosine_basis",
    "matrix_permutation_test",
    "multi_threshold",
    # permutation
    "one_sample_permutation_test",
    "phase_randomize",
    "procrustes",
    "procrustes_distance",
    # regression
    "regress",
    "threshold",
    "timeseries_correlation_permutation_test",
    "transform_pairwise",
    "trim",
    "two_sample_permutation_test",
    "u_center",
    "upsample",
    "winsorize",
    # outliers
    "zscore",
]
