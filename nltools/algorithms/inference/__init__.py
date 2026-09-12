"""Permutation tests, bootstrap resampling, and intersubject statistics.

Every test here runs on plain numpy arrays and returns a dict of results. The
one-, two-sample, correlation, matrix, and timeseries permutation tests share
one execution model: permutations run on joblib workers, `n_jobs` sets how
many, and a given `random_state` gives the same result at every worker count.
The intersubject statistics (`isc`, `isc_group`, `isfc`, `isps` in
`nltools.algorithms`) are built on the same engine.

Examples:
    ```python
    import numpy as np
    from nltools.algorithms.inference import one_sample_permutation_test

    data = np.random.randn(30)  # 30 subjects
    result = one_sample_permutation_test(data, n_permute=5000)
    result["p"]  # → two-sided p-value

    # Voxel-wise test
    data = np.random.randn(30, 50000)  # 30 subjects, 50K voxels
    result = one_sample_permutation_test(data, n_permute=10000)
    (result["p"] < 0.05).sum()  # → number of significant voxels
    ```

Note:
    These are the functional core. The data classes wrap them —
    `BrainData.ttest`, `BrainData.bootstrap`, `Adjacency.ttest` — and handle
    masking and result reshaping for you.
"""

# Import public API functions
from .one_sample import one_sample_permutation_test
from .two_sample import two_sample_permutation_test
from .correlation import correlation_permutation_test
from .timeseries import (
    circle_shift,
    phase_randomize,
    timeseries_correlation_permutation_test,
)
from .matrix import (
    matrix_permutation_test,
    double_center,
    u_center,
    distance_correlation,
)

# NOTE: the user-facing intersubject statistics (`isc`, `isc_group`, `isfc`,
# `isps`) live in `.intersubject` and are exported flat from
# `nltools.algorithms` — re-exporting the `isc` *function* here would shadow
# the `.isc` engine *module* on this package.
from .isc import isc_permutation_test, isc_group_permutation_test


# Define public exports
__all__ = [
    "circle_shift",
    "correlation_permutation_test",
    "distance_correlation",
    "double_center",
    "isc_group_permutation_test",
    "isc_permutation_test",
    "matrix_permutation_test",
    "one_sample_permutation_test",
    "phase_randomize",
    "timeseries_correlation_permutation_test",
    "two_sample_permutation_test",
    "u_center",
]
