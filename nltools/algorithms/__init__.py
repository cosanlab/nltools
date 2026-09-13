"""nltools.algorithms — the functional core of nltools.

Every user-facing statistical function and algorithm is importable flat from
here (`from nltools.algorithms import fdr, zscore, isc`), organized into
focused submodules underneath:

- **corrections**: multiple-comparison corrections (FDR, Holm-Bonferroni, thresholding)
- **outliers**: outlier detection, winsorizing, z-scoring
- **signal**: temporal signal processing (resampling, filtering, basis functions)
- **similarity**: similarity metrics and Fisher transforms
- **regression**: standalone OLS regression on numpy arrays
- **alignment**: the `align`/`procrustes` entry points and the shared-response
  estimators behind them
- **inference**: permutation tests, bootstrap resampling, and intersubject
  statistics (ISC/ISFC/ISPS), parallelized across joblib workers
- **backends**: device selection and memory budgeting for the ridge paths;
  `check_gpu_available` is how you ask before requesting `device='gpu'`

Ridge regression lives in `nltools.models.Ridge`, which delegates its numerics
to the Himalaya library and is reached through `BrainData.fit(model='ridge')`.
"""

__all__ = [
    "align",
    "align_states",
    "calc_bpm",
    "check_gpu_available",
    "circle_shift",
    "compute_searchlight_neighborhoods",
    "compute_similarity",
    "correlation_permutation_test",
    "distance_correlation",
    "downsample",
    "fdr",
    "find_spikes",
    "fisher_r_to_z",
    "fisher_z_to_r",
    "holm_bonf",
    "isc",
    "isc_group",
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
    "threshold",
    "transform_pairwise",
    "trim",
    "two_sample_permutation_test",
    "upsample",
    "winsorize",
    "zscore",
]

from .alignment import (
    align,
    align_states,
    procrustes,
    procrustes_distance,
)
from .backends import check_gpu_available
from .corrections import fdr, holm_bonf, multi_threshold, threshold
from .neighborhoods import compute_searchlight_neighborhoods
from .inference import (
    circle_shift,
    correlation_permutation_test,
    distance_correlation,
    matrix_permutation_test,
    one_sample_permutation_test,
    phase_randomize,
    two_sample_permutation_test,
)

# Imported from the submodule, not the `inference` package namespace: exporting
# the `isc` *function* there would shadow the `inference.isc` engine *module*.
from .inference.intersubject import isc, isc_group, isfc, isps
from .outliers import find_spikes, trim, winsorize, zscore
from .regression import regress
from .signal import calc_bpm, downsample, make_cosine_basis, upsample
from .similarity import (
    compute_similarity,
    fisher_r_to_z,
    fisher_z_to_r,
    transform_pairwise,
)
