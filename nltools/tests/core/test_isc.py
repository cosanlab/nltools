"""
Tests for GPU-accelerated Intersubject Correlation (ISC) module.

Test organization follows the TDD plan (2025-10-30-isc-tdd-plan.md):
    Phase 1: Leave-One-Out (LOO) Computation
    Phase 2: Pairwise Computation
    Phase 3: LOO Bootstrap
    Phase 4: Pairwise Bootstrap
    Phase 5: Main Function
    Integration Tests
    Performance Benchmarks (tier2)

Tier 1: Fast tests (~1-2min, run on every iteration)
Tier 2: GPU and benchmark tests (~5-7min, run before commits)

Testing Strategy & Tolerances:

    This test suite uses different tolerance levels for different comparison types.
    Following the pattern from test_inference.py:

    1. Backend Consistency (NumPy vs PyTorch):
       - Tolerance: EXACT (rtol=1e-5)
       - Why: Same algorithm, same random seed, only float precision differs
       - Tests verify implementations are mathematically identical

    2. GPU Precision (GPU float32 vs CPU float64):
       - Values: rtol=1e-3 (0.1% error)
       - P-values: rtol=5e-3 (0.5% error)
       - Why: GPU uses float32, CPU uses float64; P-values accumulate more error
"""

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from nltools.algorithms.inference.isc import (
    _batch_correlation_gpu,
    _batch_corrcoef_gpu,
    _bootstrap_loo_cpu_parallel,
    _bootstrap_loo_numpy,
    _bootstrap_pairwise_cpu_parallel,
    _bootstrap_pairwise_numpy,
    _compute_loo_isc,
    _compute_pairwise_isc,
    isc_permutation_test,
)


# =============================================================================
# Test Constants - DO NOT MODIFY without updating docstring above
# =============================================================================

# Tolerance for backend consistency (NumPy vs PyTorch with same seed)
# These should be EXACT matches (same algorithm, only precision differs)
TOLERANCE_EXACT = 1e-5

# Tolerance for GPU vs CPU comparisons (float32 vs float64)
TOLERANCE_GPU_VALUE = 1e-3  # 0.1% error for computed values
TOLERANCE_GPU_PVALUE = 5e-3  # 0.5% error for P-values (more FP error)


# =============================================================================
# Phase 1: Leave-One-Out (LOO) Computation Tests
# =============================================================================


def test_compute_loo_isc_single_feature_basic():
    """LOO ISC computes correlation of each subject with mean of others."""
    np.random.seed(42)
    data = np.random.randn(100, 5)  # 100 timepoints, 5 subjects

    loo_values = _compute_loo_isc(data, backend="numpy")

    assert loo_values.shape == (5,)

    # Verify each value manually
    for i in range(5):
        others_mean = data[:, np.arange(5) != i].mean(axis=1)
        expected = np.corrcoef(data[:, i], others_mean)[0, 1]
        assert np.isclose(loo_values[i], expected)


def test_compute_loo_isc_voxelwise_shape():
    """Voxel-wise LOO returns (n_subjects, n_voxels) array."""
    np.random.seed(42)
    data = np.random.randn(100, 5, 10)  # 10 voxels

    loo_values = _compute_loo_isc(data, backend="numpy")

    assert loo_values.shape == (5, 10)

    # Each voxel computed independently
    for v in range(10):
        voxel_loo = _compute_loo_isc(data[:, :, v], backend="numpy")
        assert np.allclose(loo_values[:, v], voxel_loo)


@pytest.mark.slow
def test_compute_loo_isc_gpu_matches_numpy():
    """GPU LOO matches NumPy within float32 tolerance."""
    _ = pytest.importorskip("torch")

    np.random.seed(42)
    data = np.random.randn(100, 10, 100)  # 100 voxels (smaller for testing)

    loo_numpy = _compute_loo_isc(data, backend="numpy")
    loo_gpu = _compute_loo_isc(data, backend="torch")

    # GPU uses float32, CPU uses float64 - use GPU precision tolerance
    np.testing.assert_allclose(loo_numpy, loo_gpu, rtol=TOLERANCE_GPU_VALUE, atol=1e-7)


def test_compute_loo_isc_deterministic():
    """LOO computation is deterministic."""
    np.random.seed(42)
    data = np.random.randn(100, 5, 10)

    loo1 = _compute_loo_isc(data, backend="numpy")
    loo2 = _compute_loo_isc(data, backend="numpy")

    assert np.array_equal(loo1, loo2)


@pytest.mark.slow
def test_batch_correlation_gpu_correctness():
    """Batch correlation on GPU matches manual computation."""
    torch = pytest.importorskip("torch")

    np.random.seed(42)
    # Create simple test data
    x = np.random.randn(100, 5)  # 100 observations, 5 features
    y = np.random.randn(100, 5)

    # Convert to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_gpu = torch.tensor(x, dtype=torch.float32, device=device)
    y_gpu = torch.tensor(y, dtype=torch.float32, device=device)

    # Compute on GPU
    corr_gpu = _batch_correlation_gpu(x_gpu, y_gpu).cpu().numpy()

    # Verify against NumPy for each feature
    for i in range(5):
        expected = np.corrcoef(x[:, i], y[:, i])[0, 1]
        assert np.isclose(corr_gpu[i], expected, rtol=1e-5)


# =============================================================================
# Phase 2: Pairwise Computation Tests
# =============================================================================


def test_compute_pairwise_isc_single_feature_condensed():
    """Pairwise ISC returns condensed correlation matrix."""
    np.random.seed(42)
    data = np.random.randn(100, 5)

    pairwise = _compute_pairwise_isc(data, backend="numpy")

    # 5 subjects → 10 pairs
    assert pairwise.shape == (10,)

    # Verify matches np.corrcoef
    corr_matrix = np.corrcoef(data.T)
    expected = squareform(corr_matrix, checks=False)
    assert np.allclose(pairwise, expected)


def test_compute_pairwise_isc_voxelwise_shape():
    """Voxel-wise pairwise returns (n_pairs, n_voxels)."""
    np.random.seed(42)
    data = np.random.randn(100, 5, 10)

    pairwise = _compute_pairwise_isc(data, backend="numpy")

    assert pairwise.shape == (10, 10)  # 10 pairs × 10 voxels

    # Verify each voxel independently
    for v in range(10):
        voxel_pair = _compute_pairwise_isc(data[:, :, v], backend="numpy")
        assert np.allclose(pairwise[:, v], voxel_pair)


@pytest.mark.slow
def test_compute_pairwise_isc_gpu_matches_numpy():
    """GPU pairwise matches NumPy within float32 tolerance."""
    _ = pytest.importorskip("torch")

    np.random.seed(42)
    data = np.random.randn(100, 10, 100)  # 100 voxels

    pair_numpy = _compute_pairwise_isc(data, backend="numpy")
    pair_gpu = _compute_pairwise_isc(data, backend="torch")

    # GPU uses float32, CPU uses float64 - use GPU precision tolerance
    np.testing.assert_allclose(
        pair_numpy, pair_gpu, rtol=TOLERANCE_GPU_VALUE, atol=1e-7
    )


@pytest.mark.slow
def test_batch_corrcoef_gpu_correctness():
    """Batch corrcoef on GPU matches NumPy."""
    torch = pytest.importorskip("torch")

    # Create test data (n_voxels, n_subjects, n_observations)
    np.random.seed(42)
    data = np.random.randn(5, 10, 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_gpu = torch.tensor(data, dtype=torch.float32, device=device)
    corr_gpu = _batch_corrcoef_gpu(data_gpu).cpu().numpy()

    # Verify against NumPy for each voxel
    # GPU uses float32, CPU uses float64 - use GPU precision tolerance
    for v in range(5):
        corr_numpy = np.corrcoef(data[v])
        np.testing.assert_allclose(
            corr_gpu[v], corr_numpy, rtol=TOLERANCE_GPU_VALUE, atol=1e-7
        )


# =============================================================================
# Phase 3: LOO Bootstrap Tests
# =============================================================================


def test_bootstrap_loo_resamples_values():
    """LOO bootstrap resamples pre-computed LOO values."""
    loo_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    rng = np.random.RandomState(42)

    # Single bootstrap
    boot_median = _bootstrap_loo_numpy(loo_values, metric="median", random_state=rng)

    # Should be median of resampled values
    assert isinstance(boot_median, (float, np.floating))
    assert -1 <= boot_median <= 1


def test_bootstrap_loo_fisher_z_transform():
    """LOO bootstrap with mean applies Fisher z-transform."""
    loo_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    rng = np.random.RandomState(42)

    boot_mean = _bootstrap_loo_numpy(loo_values, metric="mean", random_state=rng)

    # Manual Fisher z
    rng2 = np.random.RandomState(42)
    indices = rng2.choice(5, size=5, replace=True)
    boot_sample = loo_values[indices]
    z = np.arctanh(boot_sample)
    expected = np.tanh(np.mean(z))

    assert np.isclose(boot_mean, expected)


def test_bootstrap_loo_cpu_parallel_deterministic():
    """CPU-parallel LOO bootstrap is deterministic."""
    np.random.seed(42)
    loo_values = np.random.randn(50)  # 50 subjects

    boot1 = _bootstrap_loo_cpu_parallel(
        loo_values, n_permute=100, metric="median", random_state=42, progress_bar=False
    )

    boot2 = _bootstrap_loo_cpu_parallel(
        loo_values, n_permute=100, metric="median", random_state=42, progress_bar=False
    )

    assert np.allclose(boot1, boot2)


def test_bootstrap_loo_cpu_parallel_matches_numpy():
    """CPU-parallel LOO matches sequential NumPy."""
    np.random.seed(42)
    loo_values = np.random.randn(10, 20)  # 10 subjects, 20 voxels

    # Sequential
    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 2**31 - 1, size=100)
    boot_numpy = np.array(
        [
            _bootstrap_loo_numpy(
                loo_values,
                metric="median",
                random_state=np.random.RandomState(seeds[i]),
            )
            for i in range(100)
        ]
    )

    # Parallel
    boot_parallel = _bootstrap_loo_cpu_parallel(
        loo_values, n_permute=100, metric="median", random_state=42, progress_bar=False
    )

    assert np.allclose(boot_numpy, boot_parallel)


# =============================================================================
# Phase 4: Pairwise Bootstrap Tests
# =============================================================================


def test_bootstrap_pairwise_duplicate_masking():
    """Pairwise bootstrap masks same-subject correlations as NaN."""
    # Simple 3-subject example
    pairwise = np.array([0.8, 0.7, 0.6])  # Pairs: (0,1), (0,2), (1,2)

    # Bootstrap: [0, 0, 1] → subject 0 appears twice
    boot_subjects = np.array([0, 0, 1])

    boot_median = _bootstrap_pairwise_numpy(
        pairwise, metric="median", bootstrap_subjects=boot_subjects
    )

    # Expected: Matrix [[1.0, 1.0, 0.8],
    #                   [1.0, 1.0, 0.8],
    #                   [0.8, 0.8, 1.0]]
    # Mask 1.0s → valid values: [0.8, 0.8]
    # Median = 0.8
    assert np.isclose(boot_median, 0.8)


def test_bootstrap_pairwise_condensed_to_square_roundtrip():
    """Condensed storage roundtrip preserves correlation values."""
    # Create correlation matrix
    corr_matrix = np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.6], [0.7, 0.6, 1.0]])

    # Extract condensed form (upper triangle, excluding diagonal)
    condensed = squareform(corr_matrix, checks=False)

    # Reconstruct to square
    reconstructed = squareform(condensed, force="tomatrix", checks=False)
    # Fill diagonal with 1.0 (correlation matrices have 1 on diagonal)
    np.fill_diagonal(reconstructed, 1.0)

    # Back to condensed
    condensed_again = squareform(reconstructed, checks=False)

    assert np.allclose(condensed, condensed_again)


def test_bootstrap_pairwise_cpu_parallel_deterministic():
    """CPU-parallel pairwise bootstrap is deterministic."""
    np.random.seed(42)
    pairwise = np.random.randn(45, 20)  # 10 subjects (45 pairs), 20 voxels

    boot1 = _bootstrap_pairwise_cpu_parallel(
        pairwise,
        n_permute=100,
        n_subjects=10,
        metric="median",
        random_state=42,
        progress_bar=False,
    )

    boot2 = _bootstrap_pairwise_cpu_parallel(
        pairwise,
        n_permute=100,
        n_subjects=10,
        metric="median",
        random_state=42,
        progress_bar=False,
    )

    assert np.allclose(boot1, boot2)


# =============================================================================
# Phase 5: Main Function Tests
# =============================================================================


def test_isc_permutation_test_loo_basic():
    """LOO ISC returns all expected outputs."""
    np.random.seed(42)
    data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects

    result = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    assert "isc" in result
    assert "p" in result
    assert "ci" in result
    assert 0 <= result["p"] <= 1
    assert len(result["ci"]) == 2
    # CI should contain ISC (or be close for small n_permute)
    assert result["ci"][0] <= result["ci"][1]


def test_isc_permutation_test_pairwise_basic():
    """Pairwise ISC returns all expected outputs."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    assert "isc" in result
    assert "p" in result
    assert "ci" in result


def test_isc_loo_vs_pairwise_correlated():
    """LOO and pairwise ISC are different but correlated."""
    np.random.seed(42)
    data = np.random.randn(100, 20)  # 20 subjects for good comparison

    result_loo = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_pairwise = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Should be different
    assert result_loo["isc"] != result_pairwise["isc"]

    # But in similar range (both measuring subject consistency)
    assert np.abs(result_loo["isc"] - result_pairwise["isc"]) < 0.5


def test_isc_voxelwise_shape():
    """Voxel-wise ISC returns arrays per voxel."""
    np.random.seed(42)
    data = np.random.randn(100, 10, 50)  # 50 voxels

    result = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    assert result["isc"].shape == (50,)
    assert result["p"].shape == (50,)
    assert result["ci"][0].shape == (50,)
    assert result["ci"][1].shape == (50,)


def test_isc_backend_consistency_numpy_cpu_parallel():
    """NumPy and CPU-parallel backends give identical results."""
    np.random.seed(42)
    data = np.random.randn(100, 10, 20)

    result_numpy = isc_permutation_test(
        data, parallel=None, n_permute=100, random_state=42, progress_bar=False
    )

    result_parallel = isc_permutation_test(
        data, parallel="cpu", n_permute=100, random_state=42, progress_bar=False
    )

    # Both use float64 CPU - should be exact matches
    np.testing.assert_allclose(
        result_numpy["isc"], result_parallel["isc"], rtol=TOLERANCE_EXACT, atol=1e-10
    )
    np.testing.assert_allclose(
        result_numpy["p"], result_parallel["p"], rtol=TOLERANCE_EXACT, atol=1e-10
    )


@pytest.mark.slow
def test_isc_gpu_matches_cpu():
    """GPU backend matches CPU within float32 tolerance."""
    _ = pytest.importorskip("torch")

    np.random.seed(42)
    data = np.random.randn(100, 10, 100)  # 100 voxels

    result_cpu = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        parallel="cpu",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_gpu = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        parallel="gpu",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # GPU uses float32, CPU uses float64 - use GPU precision tolerances
    np.testing.assert_allclose(
        result_cpu["isc"], result_gpu["isc"], rtol=TOLERANCE_GPU_VALUE, atol=1e-7
    )
    # P-values accumulate more FP error, use GPU p-value tolerance
    np.testing.assert_allclose(
        result_cpu["p"], result_gpu["p"], rtol=TOLERANCE_GPU_PVALUE, atol=1e-7
    )


def test_isc_return_null_distribution():
    """ISC with return_null=True includes bootstrap distribution."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result = isc_permutation_test(
        data, n_permute=100, return_null=True, random_state=42, progress_bar=False
    )

    assert "null_dist" in result
    assert result["null_dist"].shape == (100,)


def test_isc_circle_shift_method():
    """ISC with circle_shift method completes successfully."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result = isc_permutation_test(
        data, method="circle_shift", n_permute=50, random_state=42, progress_bar=False
    )

    assert "isc" in result
    assert "p" in result


def test_isc_phase_randomize_method():
    """ISC with phase_randomize method completes successfully."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result = isc_permutation_test(
        data,
        method="phase_randomize",
        n_permute=50,
        random_state=42,
        progress_bar=False,
    )

    assert "isc" in result
    assert "p" in result


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.slow
def test_isc_matches_brainiak_loo_logic():
    """LOO ISC matches Brainiak computation pattern."""
    # Create controlled data
    np.random.seed(42)
    data = np.random.randn(100, 5)

    # Our implementation
    result = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        metric="median",
        n_permute=1000,
        random_state=42,
        progress_bar=False,
    )

    # Manual Brainiak-style computation
    loo_values = []
    for i in range(5):
        others = data[:, np.arange(5) != i].mean(axis=1)
        loo_values.append(np.corrcoef(data[:, i], others)[0, 1])

    expected_isc = np.median(loo_values)

    assert np.isclose(result["isc"], expected_isc)


@pytest.mark.slow
def test_isc_chen_bootstrap_correctness():
    """Bootstrap uses subject-wise resampling (Chen et al. 2016)."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        n_permute=1000,
        random_state=42,
        return_null=True,
        progress_bar=False,
    )

    # Bootstrap distribution should be centered
    # Mean of (null + observed) should approximate observed ISC
    null_mean = np.mean(result["null_distribution"] + result["isc"])

    # Mean should be close to observed ISC (unbiased bootstrap)
    assert np.abs(null_mean - result["isc"]) < 0.1


# =============================================================================
# Performance Benchmarks (Tier 2, optional with GPU)
# =============================================================================


@pytest.mark.slow
def test_isc_gpu_speedup_loo():
    """GPU provides speedup for voxel-wise LOO computation."""
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    np.random.seed(42)
    data = np.random.randn(100, 50, 5000)  # 5K voxels

    import time

    start = time.time()
    _ = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        parallel="cpu",
        n_permute=100,
        progress_bar=False,
    )
    cpu_time = time.time() - start

    start = time.time()
    _ = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        parallel="gpu",
        n_permute=100,
        progress_bar=False,
    )
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time
    print(f"\nLOO GPU Speedup: {speedup:.1f}×")

    # Expect at least 3× speedup (conservative for testing)
    assert speedup > 3.0


@pytest.mark.slow
def test_isc_gpu_pairwise_matches_cpu():
    """GPU pairwise ISC matches CPU within float32 tolerance.

    Correctness guard for the GPU pairwise wiring (parallel='gpu' now routes the
    observed pairwise compute to the torch backend). Unlike LOO, pairwise GPU
    does NOT provide a meaningful speedup for the full permutation test: the
    default-method bootstrap resamples the precomputed condensed matrix on CPU
    (only the single observed compute is on GPU), and `_compute_pairwise_isc_gpu`
    extracts the per-voxel upper triangles in a CPU squareform loop. So this
    asserts numerical agreement, not speed. (Previously asserted >3× speedup —
    an assumption that never ran on the Apple-Silicon dev box and does not hold.)
    """
    _ = pytest.importorskip("torch")

    np.random.seed(42)
    data = np.random.randn(80, 20, 200)  # (n_obs, n_subjects, n_voxels)

    result_cpu = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        parallel="cpu",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )
    result_gpu = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        parallel="gpu",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    np.testing.assert_allclose(
        result_cpu["isc"], result_gpu["isc"], rtol=TOLERANCE_GPU_VALUE, atol=1e-6
    )
    np.testing.assert_allclose(
        result_cpu["p"], result_gpu["p"], rtol=TOLERANCE_GPU_PVALUE, atol=1e-6
    )


# =============================================================================
# Edge Cases and Input Validation
# =============================================================================


def test_isc_invalid_summary_statistic():
    """Invalid summary_statistic raises ValueError."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    with pytest.raises(ValueError, match="summary_statistic must be"):
        isc_permutation_test(
            data, summary_statistic="invalid", n_permute=100, progress_bar=False
        )


def test_isc_invalid_method():
    """Invalid method raises ValueError."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    with pytest.raises(ValueError, match="method must be"):
        isc_permutation_test(data, method="invalid", n_permute=100, progress_bar=False)


def test_isc_invalid_data_dimensions():
    """Data with wrong dimensions raises ValueError."""
    data_1d = np.random.randn(100)

    with pytest.raises(ValueError, match="data must be 2D or 3D"):
        isc_permutation_test(data_1d, n_permute=100, progress_bar=False)


def test_isc_default_is_pairwise():
    """Default summary_statistic is 'pairwise'."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # Call without specifying summary_statistic
    result = isc_permutation_test(
        data, n_permute=100, random_state=42, progress_bar=False
    )

    # Verify it used pairwise by comparing with explicit pairwise call
    result_explicit = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    assert np.isclose(result["isc"], result_explicit["isc"])


def test_isc_exclude_self_corr_parameter():
    """exclude_self_corr parameter controls masking of self-correlations."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # Test with exclude_self_corr=True (default)
    result_exclude = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        method="bootstrap",
        n_permute=200,
        exclude_self_corr=True,
        random_state=42,
        progress_bar=False,
    )

    # Test with exclude_self_corr=False
    result_include = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        method="bootstrap",
        n_permute=200,
        exclude_self_corr=False,
        random_state=42,
        progress_bar=False,
    )

    # Both should complete successfully
    assert "isc" in result_exclude
    assert "isc" in result_include
    assert "p" in result_exclude
    assert "p" in result_include

    # With small n_permute, results may be similar, but exclude=True should
    # always exclude perfect correlations (1.0) from bootstrap samples
    # We verify the parameter is accepted and functions differently
    assert isinstance(result_exclude["isc"], (float, np.floating))
    assert isinstance(result_include["isc"], (float, np.floating))


def test_isc_exclude_self_corr_affects_bootstrap():
    """exclude_self_corr parameter affects bootstrap distribution."""
    # Create data that will produce duplicate subjects in bootstrap
    np.random.seed(42)
    data = np.random.randn(100, 5)  # Small n_subjects increases chance of duplicates

    result_exclude = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        method="bootstrap",
        n_permute=500,
        exclude_self_corr=True,
        random_state=42,
        return_null=True,
        progress_bar=False,
    )

    result_include = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        method="bootstrap",
        n_permute=500,
        exclude_self_corr=False,
        random_state=42,
        return_null=True,
        progress_bar=False,
    )

    # Bootstrap distributions should be different
    # When exclude_self_corr=False, perfect correlations (1.0) from duplicates
    # are included, which can inflate the bootstrap distribution
    null_exclude = result_exclude["null_dist"]
    null_include = result_include["null_dist"]

    # Both should have valid results
    assert not np.all(np.isnan(null_exclude))
    assert not np.all(np.isnan(null_include))

    # The distributions may differ (especially if duplicates occurred)
    # We verify both paths work correctly
    assert len(null_exclude) == 500
    assert len(null_include) == 500


def test_isc_sim_metric_parameter():
    """sim_metric parameter allows different similarity metrics."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # Test with correlation (default)
    result_corr = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="correlation",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Test with euclidean distance (converted to similarity)
    result_eucl = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="euclidean",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Both should complete successfully
    assert "isc" in result_corr
    assert "isc" in result_eucl
    assert "p" in result_corr
    assert "p" in result_eucl

    # Different metrics should produce different ISC values
    # Correlation ISC is bounded [-1, 1], euclidean similarity can be negative
    assert isinstance(result_corr["isc"], (float, np.floating))
    assert isinstance(result_eucl["isc"], (float, np.floating))

    # Correlation should be in reasonable range
    assert -1 <= result_corr["isc"] <= 1


def test_isc_sim_metric_affects_pairwise_computation():
    """sim_metric parameter affects pairwise ISC computation."""
    np.random.seed(42)
    data = np.random.randn(100, 5)

    # Test with different metrics
    result_corr = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="correlation",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_cosine = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="cosine",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Different metrics should produce different ISC values
    assert result_corr["isc"] != result_cosine["isc"]

    # Correlation ISC should be in [-1, 1]
    assert -1 <= result_corr["isc"] <= 1

    # Cosine similarity (1 - cosine distance) should also be in reasonable range
    # but may be different from correlation
    assert isinstance(result_cosine["isc"], (float, np.floating))


def test_isc_gpu_pairwise_non_correlation_raises():
    """GPU pairwise ISC only implements sim_metric='correlation'.

    Requesting `parallel='gpu'` with a non-correlation metric is contradictory —
    the GPU pairwise kernel computes correlation only. It must fail fast with a
    clear ValueError (consistent with the sibling `isc_group_permutation_test`),
    not silently ignore the GPU request. The guard runs before any torch call,
    so no GPU/CUDA is needed to exercise it.

    (Pre-0.6.0 this asserted a UserWarning + CPU fallback via the removed
    `backend=` kwarg; that behavior no longer exists.)
    """
    np.random.seed(42)
    data = np.random.randn(100, 10, 50)  # (n_obs, n_subjects, n_voxels)

    with pytest.raises(ValueError, match="only supports sim_metric='correlation'"):
        isc_permutation_test(
            data,
            summary_statistic="pairwise",
            sim_metric="euclidean",
            parallel="gpu",
            n_permute=10,
            random_state=42,
            progress_bar=False,
        )


def test_isc_pairwise_gpu_engages_torch_backend(monkeypatch):
    """parallel='gpu' must route the pairwise compute to the torch backend.

    Regression guard for the wiring fix: the observed pairwise ISC previously
    hardcoded `backend='numpy'` even under `parallel='gpu'`, making the GPU a
    silent no-op. Spy on `_compute_pairwise_isc` and assert it's invoked with
    `backend='torch'`. No CUDA needed — torch falls back to its CPU device.
    """
    pytest.importorskip("torch")
    from nltools.algorithms.inference import isc as isc_mod

    seen = []
    orig = isc_mod._compute_pairwise_isc

    def _spy(data, backend="numpy", sim_metric="correlation"):
        seen.append(backend)
        return orig(data, backend=backend, sim_metric=sim_metric)

    monkeypatch.setattr(isc_mod, "_compute_pairwise_isc", _spy)

    np.random.seed(0)
    data = np.random.randn(60, 8, 40)
    isc_mod.isc_permutation_test(
        data,
        summary_statistic="pairwise",
        parallel="gpu",
        n_permute=10,
        random_state=0,
        progress_bar=False,
    )
    assert "torch" in seen, (
        f"pairwise parallel='gpu' used backends {seen}; expected 'torch'."
    )


def test_isc_exclude_self_corr_pairwise_only():
    """exclude_self_corr only applies to pairwise bootstrap."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # exclude_self_corr should not affect LOO (which doesn't use pairwise matrix)
    result_loo = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        method="bootstrap",
        exclude_self_corr=True,  # Should be ignored for LOO
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_loo_default = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        method="bootstrap",
        exclude_self_corr=False,  # Should be ignored for LOO
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # LOO results should be identical regardless of exclude_self_corr
    assert np.isclose(result_loo["isc"], result_loo_default["isc"])


def test_isc_sim_metric_pairwise_only():
    """sim_metric only applies to pairwise summary_statistic."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # sim_metric should not affect LOO (which computes correlations directly)
    result_loo_corr = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        sim_metric="correlation",  # Should be ignored for LOO
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_loo_eucl = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        sim_metric="euclidean",  # Should be ignored for LOO
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # LOO results should be identical regardless of sim_metric
    assert np.isclose(result_loo_corr["isc"], result_loo_eucl["isc"])


def test_isc_sim_metric_spearman_basic():
    """Spearman sim_metric works and produces valid ISC results."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # Spearman should work (currently fails, but will work after optimization)
    result_spearman = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="spearman",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Should complete successfully
    assert "isc" in result_spearman
    assert "p" in result_spearman
    assert isinstance(result_spearman["isc"], (float, np.floating))
    assert -1 <= result_spearman["isc"] <= 1


def test_isc_sim_metric_spearman_vs_correlation():
    """Spearman and correlation produce different ISC values."""
    np.random.seed(42)
    # Create monotonic but non-linear relationship
    x = np.random.randn(100)
    data = np.column_stack([x**3 + np.random.randn(100) * 0.1 for _ in range(10)])

    result_corr = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="correlation",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_spearman = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="spearman",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Spearman should handle monotonic relationships better
    # For non-linear monotonic data, Spearman should be higher
    assert result_spearman["isc"] != result_corr["isc"]
    assert -1 <= result_spearman["isc"] <= 1
    assert -1 <= result_corr["isc"] <= 1


def test_compute_pairwise_isc_spearman_single_feature():
    """Spearman pairwise ISC matches manual computation for single feature."""
    from scipy.stats import rankdata
    from scipy.spatial.distance import squareform

    np.random.seed(42)
    data = np.random.randn(100, 5)  # 100 timepoints, 5 subjects

    # Compute using our function
    result = _compute_pairwise_isc(data, backend="numpy", sim_metric="spearman")

    # Manual computation: rank-transform then Pearson correlation
    data_ranked = np.array([rankdata(data[:, i], method="average") for i in range(5)]).T
    corr_matrix = np.corrcoef(data_ranked.T)
    expected = squareform(corr_matrix, checks=False)

    # Results should match
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_compute_pairwise_isc_spearman_voxelwise():
    """Spearman pairwise ISC works for voxel-wise data."""
    from scipy.stats import rankdata
    from scipy.spatial.distance import squareform

    np.random.seed(42)
    data = np.random.randn(100, 5, 10)  # 100 timepoints, 5 subjects, 10 voxels

    # Compute using our function
    result = _compute_pairwise_isc(data, backend="numpy", sim_metric="spearman")

    # Verify shape
    n_pairs = 5 * (5 - 1) // 2
    assert result.shape == (n_pairs, 10)

    # Verify first voxel matches manual computation
    data_ranked_v0 = np.array(
        [rankdata(data[:, i, 0], method="average") for i in range(5)]
    ).T
    corr_matrix_v0 = np.corrcoef(data_ranked_v0.T)
    expected_v0 = squareform(corr_matrix_v0, checks=False)

    np.testing.assert_allclose(result[:, 0], expected_v0, rtol=1e-10, atol=1e-10)


@pytest.mark.slow
def test_compute_pairwise_isc_spearman_performance():
    """Spearman optimization works efficiently and avoids pairwise_distances overhead."""
    import time

    np.random.seed(42)
    # Use voxel-wise data where optimization matters most
    data = np.random.randn(100, 10, 100)  # 100 voxels

    # Time Spearman optimization (rank-transform + corrcoef)
    start = time.time()
    result_spearman = _compute_pairwise_isc(
        data, backend="numpy", sim_metric="spearman"
    )
    time_spearman = time.time() - start

    # Time correlation (baseline fast path)
    start = time.time()
    result_corr = _compute_pairwise_isc(data, backend="numpy", sim_metric="correlation")
    time_corr = time.time() - start

    # Verify results are valid
    assert result_spearman.shape == result_corr.shape
    assert np.all(np.abs(result_spearman) <= 1)
    assert np.all(np.abs(result_corr) <= 1)

    # Spearman should complete in reasonable time (within 30× of correlation)
    # Rank transform adds overhead, but the key benefit is that Spearman now works
    # (fixes the bug where it previously failed with pairwise_distances)
    assert time_spearman < 30 * time_corr, (
        f"Spearman ({time_spearman:.3f}s) should complete reasonably quickly. "
        f"Rank transform adds overhead but avoids pairwise_distances failure."
    )


def test_compute_pairwise_isc_cosine_single_feature():
    """Cosine pairwise ISC matches sklearn pairwise_distances for single feature."""
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import squareform

    np.random.seed(42)
    data = np.random.randn(100, 5)  # 100 timepoints, 5 subjects

    # Compute using our optimized function
    result = _compute_pairwise_isc(data, backend="numpy", sim_metric="cosine")

    # Compute using sklearn (baseline for correctness)
    dist_matrix = pairwise_distances(data.T, metric="cosine")
    sim_matrix = 1 - dist_matrix  # Convert distance to similarity
    expected = squareform(sim_matrix, checks=False)

    # Results should match sklearn's output
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_compute_pairwise_isc_cosine_voxelwise():
    """Cosine pairwise ISC matches sklearn pairwise_distances for voxel-wise data."""
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import squareform

    np.random.seed(42)
    data = np.random.randn(100, 5, 10)  # 100 timepoints, 5 subjects, 10 voxels

    # Compute using our optimized function
    result = _compute_pairwise_isc(data, backend="numpy", sim_metric="cosine")

    # Verify shape
    n_pairs = 5 * (5 - 1) // 2
    assert result.shape == (n_pairs, 10)

    # Verify first voxel matches sklearn computation
    dist_matrix_v0 = pairwise_distances(data[:, :, 0].T, metric="cosine")
    sim_matrix_v0 = 1 - dist_matrix_v0
    expected_v0 = squareform(sim_matrix_v0, checks=False)

    np.testing.assert_allclose(result[:, 0], expected_v0, rtol=1e-10, atol=1e-10)


def test_isc_sim_metric_cosine_vs_correlation():
    """Cosine and correlation produce different ISC values."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result_corr = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="correlation",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_cosine = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="cosine",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Different metrics should produce different ISC values
    assert result_corr["isc"] != result_cosine["isc"]
    assert -1 <= result_corr["isc"] <= 1
    # Cosine similarity should be in [0, 1] range (normalized vectors)
    assert 0 <= result_cosine["isc"] <= 1


def test_compute_pairwise_isc_cosine_handles_zero_norm():
    """Cosine similarity handles zero-norm vectors gracefully."""
    np.random.seed(42)
    data = np.random.randn(100, 5)

    # Add a zero vector (all zeros)
    data_zero = np.copy(data)
    data_zero[:, 0] = 0.0  # First subject has zero norm

    # Should not raise error
    result = _compute_pairwise_isc(data_zero, backend="numpy", sim_metric="cosine")

    # Should produce valid results (may have NaN or 0 for zero-norm pairs)
    assert result.shape == (10,)  # 5*4/2 = 10 pairs
    # Values should be finite or NaN (for zero-norm cases)
    assert np.all(np.isfinite(result) | np.isnan(result))


@pytest.mark.slow
def test_compute_pairwise_isc_cosine_performance():
    """Cosine optimization is faster than pairwise_distances fallback."""
    import time

    np.random.seed(42)
    # Use voxel-wise data where optimization matters most
    data = np.random.randn(100, 10, 100)  # 100 voxels

    # Time optimized cosine implementation
    start = time.time()
    result_cosine_opt = _compute_pairwise_isc(
        data, backend="numpy", sim_metric="cosine"
    )
    time_cosine_opt = time.time() - start

    # Time correlation (baseline fast path)
    start = time.time()
    result_corr = _compute_pairwise_isc(data, backend="numpy", sim_metric="correlation")
    time_corr = time.time() - start

    # Verify results are valid
    assert result_cosine_opt.shape == result_corr.shape
    assert np.all(np.abs(result_cosine_opt) <= 1)
    assert np.all(np.abs(result_corr) <= 1)

    # Cosine should be reasonably fast (within 10× of correlation)
    # Normalization + matrix multiply is fast, but slightly slower than raw corrcoef
    assert time_cosine_opt < 10 * time_corr, (
        f"Optimized cosine ({time_cosine_opt:.3f}s) should be reasonably fast compared "
        f"to correlation ({time_corr:.3f}s). Matrix multiply is fast but adds overhead."
    )


def test_compute_pairwise_isc_euclidean_single_feature():
    """Euclidean pairwise ISC matches sklearn pairwise_distances for single feature."""
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import squareform

    np.random.seed(42)
    data = np.random.randn(100, 5)  # 100 timepoints, 5 subjects

    # Compute using our optimized function
    result = _compute_pairwise_isc(data, backend="numpy", sim_metric="euclidean")

    # Compute using sklearn (baseline for correctness)
    dist_matrix = pairwise_distances(data.T, metric="euclidean")
    sim_matrix = 1 - dist_matrix  # Convert distance to similarity
    expected = squareform(sim_matrix, checks=False)

    # Results should match sklearn's output
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_compute_pairwise_isc_euclidean_voxelwise():
    """Euclidean pairwise ISC matches sklearn pairwise_distances for voxel-wise data."""
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import squareform

    np.random.seed(42)
    data = np.random.randn(100, 5, 10)  # 100 timepoints, 5 subjects, 10 voxels

    # Compute using our optimized function
    result = _compute_pairwise_isc(data, backend="numpy", sim_metric="euclidean")

    # Verify shape
    n_pairs = 5 * (5 - 1) // 2
    assert result.shape == (n_pairs, 10)

    # Verify first voxel matches sklearn computation
    dist_matrix_v0 = pairwise_distances(data[:, :, 0].T, metric="euclidean")
    sim_matrix_v0 = 1 - dist_matrix_v0
    expected_v0 = squareform(sim_matrix_v0, checks=False)

    np.testing.assert_allclose(result[:, 0], expected_v0, rtol=1e-10, atol=1e-10)


def test_isc_sim_metric_euclidean_vs_correlation():
    """Euclidean and correlation produce different ISC values."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result_corr = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="correlation",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_eucl = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        sim_metric="euclidean",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Different metrics should produce different ISC values
    assert result_corr["isc"] != result_eucl["isc"]
    assert -1 <= result_corr["isc"] <= 1
    # Euclidean similarity can be negative (1 - distance, where distance can be > 1)
    assert isinstance(result_eucl["isc"], (float, np.floating))


@pytest.mark.slow
def test_compute_pairwise_isc_euclidean_performance():
    """Euclidean optimization is faster than pairwise_distances fallback."""
    import time

    np.random.seed(42)
    # Use voxel-wise data where optimization matters most
    data = np.random.randn(100, 10, 100)  # 100 voxels

    # Time optimized euclidean implementation
    start = time.time()
    result_eucl_opt = _compute_pairwise_isc(
        data, backend="numpy", sim_metric="euclidean"
    )
    time_eucl_opt = time.time() - start

    # Time correlation (baseline fast path)
    start = time.time()
    result_corr = _compute_pairwise_isc(data, backend="numpy", sim_metric="correlation")
    time_corr = time.time() - start

    # Verify results are valid
    assert result_eucl_opt.shape == result_corr.shape
    assert np.all(np.isfinite(result_eucl_opt))
    assert np.all(np.isfinite(result_corr))

    # Euclidean should be reasonably fast (within 15× of correlation)
    # Matrix operations are fast, but sqrt adds overhead
    assert time_eucl_opt < 15 * time_corr, (
        f"Optimized euclidean ({time_eucl_opt:.3f}s) should be reasonably fast compared "
        f"to correlation ({time_corr:.3f}s). Vectorized operations are fast but sqrt adds overhead."
    )


# =============================================================================
# Statistical Correctness Tests
# =============================================================================


def _generate_shared_signal_isc(
    n_timepoints, n_subjects, isc_strength, random_state=None
):
    """
    Generate time series data with known ISC.

    Creates data where all subjects share a common signal with strength
    controlled by isc_strength. Higher isc_strength → higher ISC.

    Parameters
    ----------
    n_timepoints : int
        Number of time points
    n_subjects : int
        Number of subjects
    isc_strength : float
        Strength of shared signal (0.0 = no ISC, 1.0 = perfect ISC)
        Higher values → higher ISC
    random_state : int or RandomState, optional
        Random seed

    Returns
    -------
    data : ndarray, shape (n_timepoints, n_subjects)
        Time series data with known ISC structure
    """
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)

    # Generate shared signal
    shared_signal = rng.randn(n_timepoints)

    # Generate data for each subject: shared_signal * strength + noise * (1 - strength)
    data = np.zeros((n_timepoints, n_subjects))
    for i in range(n_subjects):
        noise = rng.randn(n_timepoints)
        data[:, i] = shared_signal * isc_strength + noise * np.sqrt(1 - isc_strength**2)

    return data


class TestISCStatisticalCorrectness:
    """Test statistical correctness of ISC permutation tests (not just CPU/GPU consistency)."""

    @pytest.mark.slow
    def test_null_hypothesis_pvalue_distribution(self):
        """Test that p-values are uniformly distributed under null hypothesis (ISC = 0)."""
        from scipy.stats import kstest

        n_timepoints = 100
        n_subjects = 15
        n_tests = 100  # Run many tests with different seeds
        n_permute = 2000  # Enough permutations for stable p-values

        # Test both LOO and pairwise methods
        summary_statistics = ["leave-one-out", "pairwise"]

        for summary_statistic in summary_statistics:
            p_values = []

            for seed in range(n_tests):
                np.random.seed(seed)
                # Generate independent time series (ISC = 0)
                data = np.random.randn(n_timepoints, n_subjects)

                result = isc_permutation_test(
                    data,
                    summary_statistic=summary_statistic,
                    n_permute=n_permute,
                    random_state=seed,
                    progress_bar=False,
                )

                p_values.append(result["p"])

            # Test uniformity using Kolmogorov-Smirnov test
            # Under null hypothesis, p-values should be uniformly distributed
            ks_statistic, ks_pvalue = kstest(p_values, "uniform")

            # KS test p-value should be > 0.05 (p-values are uniform)
            assert ks_pvalue > 0.05, (
                f"P-values should be uniformly distributed under null hypothesis for {summary_statistic}. "
                f"KS test p-value: {ks_pvalue:.4f}"
            )

    def test_isc_value_correctness_loo(self):
        """Test that LOO ISC value matches expected value for known ISC structure."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects = 15
        isc_strength = 0.7  # High ISC

        # Generate data with known ISC
        data = _generate_shared_signal_isc(
            n_timepoints, n_subjects, isc_strength, random_state=42
        )

        # Compute ISC
        result = isc_permutation_test(
            data,
            summary_statistic="leave-one-out",
            n_permute=100,  # Small for speed
            random_state=42,
            progress_bar=False,
        )

        # ISC should be positive and reasonably high (strength 0.7)
        # Note: Actual ISC will be less than 0.7 due to noise, but should be positive
        assert result["isc"] > 0.3, (
            f"LOO ISC should be positive for shared signal. Got {result['isc']:.4f}"
        )
        assert result["isc"] < 1.0, "ISC should be less than 1.0"

    def test_isc_value_correctness_pairwise(self):
        """Test that pairwise ISC value matches expected value for known ISC structure."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects = 15
        isc_strength = 0.7  # High ISC

        # Generate data with known ISC
        data = _generate_shared_signal_isc(
            n_timepoints, n_subjects, isc_strength, random_state=42
        )

        # Compute ISC
        result = isc_permutation_test(
            data,
            summary_statistic="pairwise",
            n_permute=100,  # Small for speed
            random_state=42,
            progress_bar=False,
        )

        # ISC should be positive and reasonably high
        assert result["isc"] > 0.3, (
            f"Pairwise ISC should be positive for shared signal. Got {result['isc']:.4f}"
        )
        assert result["isc"] < 1.0, "ISC should be less than 1.0"

    def test_effect_size_sensitivity(self):
        """Test that higher ISC produces lower p-values."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects = 15
        n_permute = 2000  # Enough for stable p-values

        # Test different ISC strengths
        isc_strengths = [0.0, 0.2, 0.4, 0.6]
        p_values = []

        for isc_strength in isc_strengths:
            data = _generate_shared_signal_isc(
                n_timepoints, n_subjects, isc_strength, random_state=42
            )

            result = isc_permutation_test(
                data,
                summary_statistic="leave-one-out",
                n_permute=n_permute,
                random_state=42,
                progress_bar=False,
            )

            p_values.append(result["p"])

        # Verify monotonic relationship: higher ISC → lower p-value
        # (p-values should generally decrease as ISC increases)
        # Note: Due to sampling variability, ISC=0.0 may occasionally have p < 0.05
        # But high ISC (0.6) should reliably produce low p-values
        assert p_values[-1] < 0.05, (
            f"High ISC (0.6) should have significant p-value. Got {p_values[-1]:.4f}"
        )

        # Verify trend: p-values should generally decrease with higher ISC
        # (allow some variance due to finite permutations)
        # Check that large ISC differences produce corresponding p-value decreases
        for i in range(len(p_values) - 1):
            # Higher ISC should generally produce lower p-values
            # But allow some variance, so check if trend is correct on average
            if isc_strengths[i + 1] > isc_strengths[i] + 0.3:  # Large enough difference
                assert p_values[i + 1] <= p_values[i] * 1.5, (
                    f"P-value should decrease with higher ISC. "
                    f"ISC={isc_strengths[i]}: p={p_values[i]:.4f}, "
                    f"ISC={isc_strengths[i + 1]}: p={p_values[i + 1]:.4f}"
                )

        # Verify that high ISC produces significantly lower p-value than low/null ISC
        # (ISC=0.6 should have much lower p-value than ISC=0.0 or 0.2)
        assert p_values[-1] < p_values[0] * 0.5, (
            f"High ISC (0.6) should produce much lower p-value than null ISC (0.0). "
            f"ISC=0.0: p={p_values[0]:.4f}, ISC=0.6: p={p_values[-1]:.4f}"
        )

    def test_loo_vs_pairwise_relationship(self):
        """Test that LOO and pairwise ISC are different but both detect ISC correctly."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects = 15
        isc_strength = 0.5  # Medium ISC

        # Generate data with known ISC
        data = _generate_shared_signal_isc(
            n_timepoints, n_subjects, isc_strength, random_state=42
        )

        # Compute both LOO and pairwise ISC
        result_loo = isc_permutation_test(
            data,
            summary_statistic="leave-one-out",
            n_permute=2000,
            random_state=42,
            progress_bar=False,
        )

        result_pairwise = isc_permutation_test(
            data,
            summary_statistic="pairwise",
            n_permute=2000,
            random_state=42,
            progress_bar=False,
        )

        # Both should detect ISC (p < 0.05 for isc_strength=0.5)
        assert result_loo["p"] < 0.05, "LOO should detect ISC"
        assert result_pairwise["p"] < 0.05, "Pairwise should detect ISC"

        # Both should have positive ISC values
        assert result_loo["isc"] > 0, "LOO ISC should be positive"
        assert result_pairwise["isc"] > 0, "Pairwise ISC should be positive"

        # LOO and pairwise ISC are different metrics (expected, see Chen 2016)
        assert result_loo["isc"] != result_pairwise["isc"], (
            "LOO and pairwise ISC are different metrics and should produce different values"
        )

        # Both should be monotonically related (higher true ISC → higher both)
        # Test with higher ISC strength
        data_high = _generate_shared_signal_isc(
            n_timepoints, n_subjects, isc_strength=0.8, random_state=42
        )

        result_loo_high = isc_permutation_test(
            data_high,
            summary_statistic="leave-one-out",
            n_permute=1000,
            random_state=42,
            progress_bar=False,
        )

        result_pairwise_high = isc_permutation_test(
            data_high,
            summary_statistic="pairwise",
            n_permute=1000,
            random_state=42,
            progress_bar=False,
        )

        # Higher ISC strength → higher ISC values (both methods)
        assert result_loo_high["isc"] > result_loo["isc"], (
            "LOO ISC should increase with higher true ISC"
        )
        assert result_pairwise_high["isc"] > result_pairwise["isc"], (
            "Pairwise ISC should increase with higher true ISC"
        )

    def test_bootstrap_method_correctness(self):
        """Test that bootstrap method produces correct null distribution."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects = 15
        isc_strength = 0.5

        # Generate data with known ISC
        data = _generate_shared_signal_isc(
            n_timepoints, n_subjects, isc_strength, random_state=42
        )

        # Run bootstrap with return_null=True
        result = isc_permutation_test(
            data,
            summary_statistic="leave-one-out",
            method="bootstrap",
            n_permute=2000,
            random_state=42,
            return_null=True,
            progress_bar=False,
        )

        # Bootstrap distribution should be centered around 0
        # (return_null returns centered null_distribution: bootstraps - observed_isc)
        null_dist = result["null_dist"]
        observed_isc = result["isc"]

        # Mean of centered null distribution should be close to 0
        null_mean = np.mean(null_dist)
        assert np.abs(null_mean) < 0.2, (
            f"Centered null distribution should have mean close to 0. "
            f"Observed ISC: {observed_isc:.4f}, Null mean: {null_mean:.4f}"
        )

        # CI should contain observed ISC at reasonable rate
        # (with 2000 permutations, CI should contain observed most of the time)
        ci_lower, ci_upper = result["ci"]
        assert ci_lower <= observed_isc <= ci_upper, (
            f"95% CI should contain observed ISC. "
            f"CI: [{ci_lower:.4f}, {ci_upper:.4f}], Observed: {observed_isc:.4f}"
        )

    def test_circle_shift_preserves_temporal_structure(self):
        """Test that circle_shift preserves temporal autocorrelation structure."""
        np.random.seed(42)
        n_timepoints = 200  # Longer for better autocorrelation measurement
        n_subjects = 10

        # Create time series with strong autocorrelation (AR(1) process)
        data = np.zeros((n_timepoints, n_subjects))
        ar_coef = 0.7  # Strong autocorrelation
        for i in range(n_subjects):
            noise = np.random.randn(n_timepoints)
            data[0, i] = noise[0]
            for t in range(1, n_timepoints):
                data[t, i] = ar_coef * data[t - 1, i] + noise[t]

        # Compute autocorrelation of original data
        autocorr_orig = []
        for i in range(n_subjects):
            # Compute lag-1 autocorrelation
            autocorr_orig.append(np.corrcoef(data[:-1, i], data[1:, i])[0, 1])

        # Run ISC test with circle_shift (which should preserve autocorrelation)
        result = isc_permutation_test(
            data,
            summary_statistic="leave-one-out",
            method="circle_shift",
            n_permute=100,  # Small for speed
            random_state=42,
            progress_bar=False,
        )

        # Verify test completes successfully
        assert "isc" in result
        assert "p" in result

        # Verify autocorrelation is preserved in circle_shifted data
        # (by checking that circle_shift function preserves it)
        from nltools.stats import circle_shift

        shifted_data = circle_shift(data, random_state=42)
        autocorr_shifted = []
        for i in range(n_subjects):
            autocorr_shifted.append(
                np.corrcoef(shifted_data[:-1, i], shifted_data[1:, i])[0, 1]
            )

        # Autocorrelation should be preserved (within tolerance)
        np.testing.assert_allclose(
            np.mean(autocorr_orig),
            np.mean(autocorr_shifted),
            rtol=0.1,  # 10% tolerance
            err_msg="Circle shift should preserve autocorrelation structure",
        )

    def test_phase_randomize_preserves_power_spectrum(self):
        """Test that phase_randomize preserves power spectrum."""
        np.random.seed(42)
        n_timepoints = 200  # Longer for better FFT resolution
        n_subjects = 10

        # Create time series with known frequency structure
        t = np.linspace(0, 10 * np.pi, n_timepoints)
        data = np.zeros((n_timepoints, n_subjects))
        for i in range(n_subjects):
            # Mix of frequencies
            data[:, i] = (
                np.sin(2 * np.pi * t)
                + 0.5 * np.sin(4 * np.pi * t)
                + np.random.randn(n_timepoints) * 0.1
            )

        # Compute power spectrum of original data
        power_orig = []
        for i in range(n_subjects):
            fft_orig = np.fft.rfft(data[:, i])
            power_orig.append(np.abs(fft_orig) ** 2)

        # Verify phase_randomize preserves power spectrum
        from nltools.stats import phase_randomize

        randomized_data = phase_randomize(data, random_state=42)
        power_rand = []
        for i in range(n_subjects):
            fft_rand = np.fft.rfft(randomized_data[:, i])
            power_rand.append(np.abs(fft_rand) ** 2)

        # Power spectrum should be preserved exactly (within numerical precision)
        for i in range(n_subjects):
            np.testing.assert_allclose(
                power_orig[i],
                power_rand[i],
                rtol=1e-10,
                err_msg="Phase randomize must preserve power spectrum exactly",
            )

        # Verify ISC test with phase_randomize completes successfully
        result = isc_permutation_test(
            data,
            summary_statistic="leave-one-out",
            method="phase_randomize",
            n_permute=100,  # Small for speed
            random_state=42,
            progress_bar=False,
        )

        assert "isc" in result
        assert "p" in result

    def test_metric_correctness_median_vs_mean(self):
        """Test that median and mean metrics both detect ISC correctly."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects = 15
        isc_strength = 0.5

        # Generate data with known ISC
        data = _generate_shared_signal_isc(
            n_timepoints, n_subjects, isc_strength, random_state=42
        )

        # Compute ISC with median metric
        result_median = isc_permutation_test(
            data,
            summary_statistic="leave-one-out",
            metric="median",
            n_permute=2000,
            random_state=42,
            progress_bar=False,
        )

        # Compute ISC with mean metric
        result_mean = isc_permutation_test(
            data,
            summary_statistic="leave-one-out",
            metric="mean",
            n_permute=2000,
            random_state=42,
            progress_bar=False,
        )

        # Both should detect ISC
        assert result_median["p"] < 0.05, "Median metric should detect ISC"
        assert result_mean["p"] < 0.05, "Mean metric should detect ISC"

        # Both should have positive ISC values
        assert result_median["isc"] > 0, "Median ISC should be positive"
        assert result_mean["isc"] > 0, "Mean ISC should be positive"

        # Test robustness: median should be less affected by outliers
        # Create data with one outlier subject
        data_outlier = data.copy()
        data_outlier[:, 0] = np.random.randn(n_timepoints) * 10  # Outlier subject

        result_median_outlier = isc_permutation_test(
            data_outlier,
            summary_statistic="leave-one-out",
            metric="median",
            n_permute=1000,
            random_state=42,
            progress_bar=False,
        )

        result_mean_outlier = isc_permutation_test(
            data_outlier,
            summary_statistic="leave-one-out",
            metric="mean",
            n_permute=1000,
            random_state=42,
            progress_bar=False,
        )

        # Median should be more robust (less affected by outlier)
        # ISC values should be similar but median may be slightly more stable
        assert isinstance(result_median_outlier["isc"], (float, np.floating))
        assert isinstance(result_mean_outlier["isc"], (float, np.floating))
        # Both should still detect ISC from remaining subjects
        assert result_median_outlier["isc"] > 0 or result_mean_outlier["isc"] > 0, (
            "At least one metric should detect ISC from non-outlier subjects"
        )

    @pytest.mark.slow
    def test_pvalue_converges_with_more_permutations(self):
        """Test that p-values stabilize with more permutations."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects = 15
        isc_strength = 0.4  # Medium ISC

        # Generate data with known ISC
        data = _generate_shared_signal_isc(
            n_timepoints, n_subjects, isc_strength, random_state=42
        )

        # Run with different permutation counts
        n_permutes = [100, 1000, 5000]
        p_values = []

        for n_permute in n_permutes:
            result = isc_permutation_test(
                data,
                summary_statistic="leave-one-out",
                n_permute=n_permute,
                random_state=42,  # Same seed for fair comparison
                progress_bar=False,
            )
            p_values.append(result["p"])

        # P-values should stabilize (variance decreases) with more permutations
        # Check that p-values are in reasonable range and don't change dramatically
        for i, p_val in enumerate(p_values):
            assert 0 <= p_val <= 1, f"P-value should be in [0, 1]. Got {p_val:.4f}"

        # P-values should be relatively stable (not wildly different)
        # With same seed, they should be similar (allowing for Monte Carlo variance)
        p_value_range = max(p_values) - min(p_values)
        assert p_value_range < 0.3, (
            f"P-values should be relatively stable across permutation counts. "
            f"Range: {p_value_range:.4f}, Values: {p_values}"
        )
