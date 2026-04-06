"""
nltools.stats — Statistical utilities for neuroimaging analysis.

This package provides standalone statistical functions organized into
focused submodules:

- **corrections**: Multiple comparison corrections (FDR, Holm-Bonferroni, thresholding)
- **outliers**: Outlier detection, winsorizing, z-scoring
- **timeseries**: Temporal signal processing (resampling, filtering, basis functions)
- **correlation**: Similarity metrics, Fisher transforms, ICC
- **alignment**: Data alignment (SRM, Procrustes, state alignment)
- **intersubject**: ISC, ISFC, ISPS

All public functions are re-exported here for convenience::

    from nltools.stats import fdr, zscore, isc  # all work
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

__all__ = [
    # corrections
    "fdr",
    "holm_bonf",
    "threshold",
    "multi_threshold",
    # outliers
    "zscore",
    "winsorize",
    "trim",
    "find_spikes",
    # timeseries
    "downsample",
    "upsample",
    "calc_bpm",
    "make_cosine_basis",
    # correlation
    "fisher_r_to_z",
    "fisher_z_to_r",
    "compute_similarity",
    "compute_multivariate_similarity",
    "compute_icc",
    "transform_pairwise",
    # alignment
    "align",
    "procrustes",
    "procrustes_distance",
    "align_states",
    # intersubject
    "isc",
    "isc_group",
    "isfc",
    "isps",
]
