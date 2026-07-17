"""Permutation tests for statistical inference.

User-facing API for permutation-based hypothesis testing. Each function
delegates to the optimized engine in ``nltools.algorithms.inference`` which
provides CPU-parallel and optional GPU-accelerated backends.

Functions:
    - ``one_sample_permutation_test``: Sign-flipping test (mean ≠ 0)
    - ``two_sample_permutation_test``: Group label shuffling (group difference)
    - ``correlation_permutation_test``: Index permutation (correlation ≠ 0)
    - ``timeseries_correlation_permutation_test``: Autocorrelation-preserving test
    - ``circle_shift``: Circular shift for time-series surrogate data
    - ``phase_randomize``: FFT-based phase randomization for surrogate data
    - ``matrix_permutation_test``: Mantel test for matrix correlation
    - ``double_center``: Double-center a distance matrix
    - ``u_center``: Bias-corrected (U-centered) distance matrix
    - ``distance_correlation``: Distance correlation for multivariate dependence
"""

__all__ = [
    "circle_shift",
    "correlation_permutation_test",
    "distance_correlation",
    "double_center",
    "matrix_permutation_test",
    "one_sample_permutation_test",
    "phase_randomize",
    "timeseries_correlation_permutation_test",
    "two_sample_permutation_test",
    "u_center",
]

import numpy as np
from typing import Literal


def one_sample_permutation_test(
    data: np.ndarray,
    *,
    n_permute: int = 5000,
    tail: int | str = 2,
    return_null: bool = False,
    device: str | None = "cpu",
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    random_state: int | None = None,
) -> dict:
    """One-sample permutation test using sign-flipping.

    Tests whether the mean of *data* is significantly different from zero
    by randomly flipping the sign of each observation. Permutation-test
    equivalent of a one-sample t-test.

    Args:
        data: Data to test.
            Shape ``(n_samples,)`` for a single feature, or
            ``(n_samples, n_features)`` for voxel-wise testing.
        n_permute: Number of permutations (default 5000).
        tail: Test sidedness.
            ``2`` / ``'two'`` — two-tailed (mean ≠ 0);
            ``1`` / ``'upper'`` — upper one-tailed (mean > 0);
            ``-1`` / ``'lower'`` — lower one-tailed (mean < 0).
        return_null: If True, include the full null distribution in results.
        device: Parallelization backend.
            ``'cpu'`` — joblib (default, 4–8× speedup);
            ``'gpu'`` — PyTorch GPU (fastest for large problems);
            ``None`` — single-threaded NumPy (debugging).
        n_jobs: CPU cores for ``device='cpu'`` (default −1 = all).
        max_gpu_memory_gb: GPU memory budget in GB (default 4.0).
        random_state: Seed for reproducibility.

    Returns:
        dict with keys ``'mean'``, ``'p'``, ``'parallel'``,
        and optionally ``'null_dist'``.
    """
    from nltools.algorithms.inference.one_sample import (
        one_sample_permutation_test as _engine,
    )

    return _engine(
        data,
        n_permute=n_permute,
        tail=tail,
        return_null=return_null,
        parallel=device,
        n_jobs=n_jobs,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=random_state,
    )


def two_sample_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int = 5000,
    tail: int | str = 2,
    return_null: bool = False,
    device: str | None = "cpu",
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    random_state: int | None = None,
) -> dict:
    """Two-sample permutation test using group label shuffling.

    Tests whether two groups differ in mean by randomly permuting group
    labels. Permutation-test equivalent of an independent-samples t-test.

    Args:
        data1: Group 1 data, shape ``(n1,)`` or ``(n1, n_features)``.
        data2: Group 2 data, shape ``(n2,)`` or ``(n2, n_features)``.
        n_permute: Number of permutations (default 5000).
        tail: Test sidedness (see `one_sample_permutation_test`).
        return_null: If True, include the full null distribution.
        device: Parallelization backend (``'cpu'``, ``'gpu'``, or ``None``).
        n_jobs: CPU cores for ``device='cpu'`` (default −1 = all).
        max_gpu_memory_gb: GPU memory budget in GB (default 4.0).
        random_state: Seed for reproducibility.

    Returns:
        dict with keys ``'mean'`` (observed group difference), ``'p'``,
        ``'parallel'``, and optionally ``'null_dist'``.
    """
    from nltools.algorithms.inference.two_sample import (
        two_sample_permutation_test as _engine,
    )

    return _engine(
        data1,
        data2,
        n_permute=n_permute,
        tail=tail,
        return_null=return_null,
        parallel=device,
        n_jobs=n_jobs,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=random_state,
    )


def correlation_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int = 5000,
    metric: str = "pearson",
    tail: int | str = 2,
    return_null: bool = False,
    device: str | None = "cpu",
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    random_state: int | None = None,
) -> dict:
    """Correlation permutation test.

    Tests whether the correlation between two variables is significantly
    different from zero by permuting one variable's indices.

    Args:
        data1: First variable, shape ``(n_samples,)``.
        data2: Second variable, shape ``(n_samples,)``.
        n_permute: Number of permutations (default 5000).
        metric: Correlation type — ``'pearson'``, ``'spearman'``, or ``'kendall'``.
        tail: Test sidedness (see `one_sample_permutation_test`).
        return_null: If True, include the full null distribution.
        device: Parallelization backend (``'cpu'``, ``'gpu'``, or ``None``).
        n_jobs: CPU cores for ``device='cpu'`` (default −1 = all).
        max_gpu_memory_gb: GPU memory budget in GB (default 4.0).
        random_state: Seed for reproducibility.

    Returns:
        dict with keys ``'r'`` (observed correlation), ``'p'``,
        ``'parallel'``, and optionally ``'null_dist'``.
    """
    from nltools.algorithms.inference.correlation import (
        correlation_permutation_test as _engine,
    )

    return _engine(
        data1,
        data2,
        n_permute=n_permute,
        metric=metric,
        tail=tail,
        return_null=return_null,
        parallel=device,
        n_jobs=n_jobs,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=random_state,
    )


def timeseries_correlation_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    method: Literal["circle_shift", "phase_randomize"] = "circle_shift",
    n_permute: int = 5000,
    metric: Literal["pearson", "spearman", "kendall"] = "pearson",
    tail: int = 2,
    device: str | None = "cpu",
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    return_null: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> dict:
    """Time-series correlation permutation test.

    Tests correlation significance while preserving temporal autocorrelation
    using either circular shifting or phase randomization of one variable.

    Args:
        data1: First time-series, shape ``(n_timepoints,)``.
        data2: Second time-series, shape ``(n_timepoints,)``.
        method: Surrogate-data method — ``'circle_shift'`` (preserves
            autocorrelation) or ``'phase_randomize'`` (preserves power spectrum).
        n_permute: Number of permutations (default 5000).
        metric: Correlation type — ``'pearson'``, ``'spearman'``, or ``'kendall'``.
        tail: Test sidedness (1, 2, or −1).
        device: Parallelization backend (``'cpu'``, ``'gpu'``, or ``None``).
        n_jobs: CPU cores for ``device='cpu'`` (default −1 = all).
        max_gpu_memory_gb: GPU memory budget in GB (default 4.0).
        return_null: If True, include the full null distribution.
        random_state: Seed for reproducibility.

    Returns:
        dict with keys ``'r'`` (observed correlation), ``'p'``,
        ``'parallel'``, and optionally ``'null_dist'``.

    References:
        Theiler et al. (1992). Testing for nonlinearity in time series.
        *Physica D*, 58, 77–94.

        Lancaster et al. (2018). Surrogate data for hypothesis testing.
        *Physics Reports*, 748, 1–60.
    """
    from nltools.algorithms.inference.timeseries import (
        timeseries_correlation_permutation_test as _engine,
    )

    return _engine(
        data1,
        data2,
        method=method,
        n_permute=n_permute,
        metric=metric,
        tail=tail,
        parallel=device,
        n_jobs=n_jobs,
        max_gpu_memory_gb=max_gpu_memory_gb,
        return_null=return_null,
        random_state=random_state,
    )


def circle_shift(
    data: np.ndarray,
    shift_amount: int | np.ndarray | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Circular shift for time-series data.

    Shifts the data circularly (wrapping around), preserving temporal
    autocorrelation structure. Useful for generating surrogate data in
    time-series permutation tests.

    Args:
        data: Input time-series, shape ``(n_timepoints,)`` or
            ``(n_timepoints, n_features)``.
        shift_amount: Number of positions to shift. If None, a random
            shift is drawn from ``[1, n_timepoints−1]``.
        random_state: Seed for reproducibility (only used when
            *shift_amount* is None).

    Returns:
        Circularly shifted array, same shape as *data*.
    """
    from nltools.algorithms.inference.timeseries import circle_shift as _engine

    return _engine(data, shift_amount=shift_amount, random_state=random_state)


def phase_randomize(
    data: np.ndarray,
    backend: str | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """FFT-based phase randomization for time-series data.

    Randomizes the phase of the FFT while preserving the power spectrum,
    producing a surrogate time-series with the same spectral characteristics
    but destroyed temporal correlations.

    Args:
        data: Input time-series, shape ``(n_timepoints,)`` or
            ``(n_timepoints, n_features)``.
        backend: Computation backend (``None`` for NumPy, ``'torch'`` for GPU).
        random_state: Seed for reproducibility.

    Returns:
        Phase-randomized array, same shape as *data*.
    """
    from nltools.algorithms.inference.timeseries import (
        phase_randomize as _engine,
    )

    return _engine(data, backend=backend, random_state=random_state)


def matrix_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int = 5000,
    metric: str = "pearson",
    how: str = "upper",
    include_diag: bool = False,
    tail: int | str = 2,
    device: str | None = "cpu",
    n_jobs: int = -1,
    return_null: bool = False,
    random_state: int | None = None,
) -> dict:
    """Matrix permutation test (Mantel test).

    Tests whether two square matrices are correlated by permuting rows and
    columns of one matrix simultaneously (preserving symmetry).

    Args:
        data1: First square matrix, shape ``(n, n)``.
        data2: Second square matrix, shape ``(n, n)``.
        n_permute: Number of permutations (default 5000).
        metric: Correlation type — ``'pearson'``, ``'spearman'``, or ``'kendall'``.
        how: Which matrix elements to use — ``'upper'``, ``'lower'``, or ``'full'``.
        include_diag: Whether to include diagonal elements.
        tail: Test sidedness (see `one_sample_permutation_test`).
        device: Parallelization backend (``'cpu'`` or ``None``; GPU not supported).
        n_jobs: CPU cores for ``device='cpu'`` (default −1 = all).
        return_null: If True, include the full null distribution.
        random_state: Seed for reproducibility.

    Returns:
        dict with keys ``'r'`` (observed matrix correlation), ``'p'``,
        ``'parallel'``, and optionally ``'null_dist'``.

    References:
        Mantel (1967). The detection of disease clustering and a generalized
        regression approach. *Cancer Research*, 27, 209–220.

        Chen et al. (2016). Untangling correlations at the group level.
        *NeuroImage*, 142, 248–259.
    """
    from nltools.algorithms.inference.matrix import (
        matrix_permutation_test as _engine,
    )

    return _engine(
        data1,
        data2,
        n_permute=n_permute,
        metric=metric,
        how=how,
        include_diag=include_diag,
        tail=tail,
        parallel=device,
        n_jobs=n_jobs,
        return_null=return_null,
        random_state=random_state,
    )


def double_center(mat: np.ndarray) -> np.ndarray:
    """Double-center a distance matrix.

    Subtracts row means, column means, and adds back the grand mean so that
    all row and column means of the result are zero.

    Args:
        mat: Square matrix, shape ``(n, n)``.

    Returns:
        Double-centered matrix, same shape.
    """
    from nltools.algorithms.inference.matrix import double_center as _engine

    return _engine(mat)


def u_center(mat: np.ndarray) -> np.ndarray:
    """U-center a distance matrix.

    Bias-corrected form of double centering that yields an unbiased estimate
    of the squared distance covariance (Székely & Rizzo, 2014).

    Args:
        mat: Square matrix, shape ``(n, n)``.

    Returns:
        U-centered matrix, same shape.
    """
    from nltools.algorithms.inference.matrix import u_center as _engine

    return _engine(mat)


def distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    bias_corrected: bool = True,
    ttest: bool = False,
) -> dict:
    """Distance correlation for multivariate dependence.

    Computes the distance correlation between two arrays, which can detect
    both linear and non-linear associations.

    Args:
        x: First variable, shape ``(n,)`` or ``(n, p)``.
        y: Second variable, shape ``(n,)`` or ``(n, q)``.
        bias_corrected: Use bias-corrected (U-centered) estimator (default True).
        ttest: If True, include a t-test for the hypothesis dcor = 0.

    Returns:
        dict with key ``'r'`` (distance correlation), and optionally
        ``'t'``, ``'p'``, ``'df'`` when ``ttest=True``.

    References:
        Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
        Measuring and testing dependence by correlation of distances.
        *Annals of Statistics*, 35(6), 2769–2794.
    """
    from nltools.algorithms.inference.matrix import (
        distance_correlation as _engine,
    )

    return _engine(x, y, bias_corrected=bias_corrected, ttest=ttest)
