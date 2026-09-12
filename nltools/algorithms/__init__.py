"""nltools.algorithms — the functional core of nltools.

Every user-facing statistical function and algorithm is importable flat from
here (`from nltools.algorithms import fdr, zscore, isc`), organized into
focused submodules underneath:

- **corrections**: multiple-comparison corrections (FDR, Holm-Bonferroni, thresholding)
- **outliers**: outlier detection, winsorizing, z-scoring
- **signal**: temporal signal processing (resampling, filtering, basis functions)
- **similarity**: similarity metrics and Fisher transforms
- **regression**: standalone OLS regression on numpy arrays
- **alignment**: SRM/DetSRM and the functional `align`/`procrustes` entry
  points
- **inference**: permutation tests, bootstrap resampling, and intersubject
  statistics (ISC/ISFC/ISPS), parallelized across joblib workers
- **hrf**: hemodynamic response functions

Ridge regression lives in `nltools.models.Ridge`, which delegates its numerics
to the Himalaya library.
"""

__all__ = [
    "SRM",
    "DetSRM",
    "SphereNeighborhoods",
    "align",
    "align_states",
    "calc_bpm",
    "circle_shift",
    "compute_multivariate_similarity",
    "compute_searchlight_neighborhoods",
    "compute_similarity",
    "correlation_permutation_test",
    "distance_correlation",
    "double_center",
    "downsample",
    "fdr",
    "find_spikes",
    "fisher_r_to_z",
    "fisher_z_to_r",
    "glover_dispersion_derivative",
    "glover_hrf",
    "glover_time_derivative",
    "holm_bonf",
    "isc",
    "isc_group",
    "isc_group_permutation_test",
    "isc_permutation_test",
    "isfc",
    "isps",
    "make_cosine_basis",
    "matrix_permutation_test",
    "multi_threshold",
    "one_sample_permutation_test",
    "phase_randomize",
    "procrustes",
    "procrustes_distance",
    "regress",
    "spm_dispersion_derivative",
    "spm_hrf",
    "spm_time_derivative",
    "threshold",
    "timeseries_correlation_permutation_test",
    "transform_pairwise",
    "trim",
    "two_sample_permutation_test",
    "u_center",
    "upsample",
    "winsorize",
    "zscore",
]

from .alignment import (
    DetSRM,
    SRM,
    align,
    align_states,
    procrustes,
    procrustes_distance,
)
from .corrections import fdr, holm_bonf, multi_threshold, threshold
from ..data.braindata.neighborhoods import (
    SphereNeighborhoods,
    compute_searchlight_neighborhoods,
)
from .hrf import (
    glover_dispersion_derivative,
    glover_hrf,
    glover_time_derivative,
    spm_dispersion_derivative,
    spm_hrf,
    spm_time_derivative,
)
from .inference import (
    circle_shift,
    correlation_permutation_test,
    distance_correlation,
    double_center,
    isc_group_permutation_test,
    isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    phase_randomize,
    timeseries_correlation_permutation_test,
    two_sample_permutation_test,
    u_center,
)

# Imported from the submodule, not the `inference` package namespace: exporting
# the `isc` *function* there would shadow the `inference.isc` engine *module*.
from .inference.intersubject import isc, isc_group, isfc, isps
from .outliers import find_spikes, trim, winsorize, zscore
from .regression import regress
from .signal import calc_bpm, downsample, make_cosine_basis, upsample
from .similarity import (
    compute_multivariate_similarity,
    compute_similarity,
    fisher_r_to_z,
    fisher_z_to_r,
    transform_pairwise,
)
