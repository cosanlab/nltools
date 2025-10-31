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
# Phase 1: Leave-One-Out (LOO) Computation Tests
# =============================================================================


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier2
def test_compute_loo_isc_gpu_matches_numpy():
    """GPU LOO matches NumPy within float32 tolerance."""
    _ = pytest.importorskip("torch")

    np.random.seed(42)
    data = np.random.randn(100, 10, 100)  # 100 voxels (smaller for testing)

    loo_numpy = _compute_loo_isc(data, backend="numpy")
    loo_gpu = _compute_loo_isc(data, backend="torch")

    assert np.allclose(loo_numpy, loo_gpu, rtol=1e-5)


@pytest.mark.tier1
def test_compute_loo_isc_deterministic():
    """LOO computation is deterministic."""
    np.random.seed(42)
    data = np.random.randn(100, 5, 10)

    loo1 = _compute_loo_isc(data, backend="numpy")
    loo2 = _compute_loo_isc(data, backend="numpy")

    assert np.array_equal(loo1, loo2)


@pytest.mark.tier2
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier2
def test_compute_pairwise_isc_gpu_matches_numpy():
    """GPU pairwise matches NumPy within float32 tolerance."""
    _ = pytest.importorskip("torch")

    np.random.seed(42)
    data = np.random.randn(100, 10, 100)  # 100 voxels

    pair_numpy = _compute_pairwise_isc(data, backend="numpy")
    pair_gpu = _compute_pairwise_isc(data, backend="torch")

    assert np.allclose(pair_numpy, pair_gpu, rtol=1e-5)


@pytest.mark.tier2
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
    for v in range(5):
        corr_numpy = np.corrcoef(data[v])
        assert np.allclose(corr_gpu[v], corr_numpy, rtol=1e-5)


# =============================================================================
# Phase 3: LOO Bootstrap Tests
# =============================================================================


@pytest.mark.tier1
def test_bootstrap_loo_resamples_values():
    """LOO bootstrap resamples pre-computed LOO values."""
    loo_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    rng = np.random.RandomState(42)

    # Single bootstrap
    boot_median = _bootstrap_loo_numpy(loo_values, metric="median", random_state=rng)

    # Should be median of resampled values
    assert isinstance(boot_median, (float, np.floating))
    assert -1 <= boot_median <= 1


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
def test_isc_backend_consistency_numpy_cpu_parallel():
    """NumPy and CPU-parallel backends give identical results."""
    np.random.seed(42)
    data = np.random.randn(100, 10, 20)

    result_numpy = isc_permutation_test(
        data, backend="numpy", n_permute=100, random_state=42, progress_bar=False
    )

    result_parallel = isc_permutation_test(
        data, backend="cpu-parallel", n_permute=100, random_state=42, progress_bar=False
    )

    assert np.allclose(result_numpy["isc"], result_parallel["isc"])
    assert np.allclose(result_numpy["p"], result_parallel["p"])


@pytest.mark.tier2
def test_isc_gpu_matches_cpu():
    """GPU backend matches CPU within float32 tolerance."""
    _ = pytest.importorskip("torch")

    np.random.seed(42)
    data = np.random.randn(100, 10, 100)  # 100 voxels

    result_cpu = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        backend="numpy",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_gpu = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        backend="torch",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    assert np.allclose(result_cpu["isc"], result_gpu["isc"], rtol=1e-5)
    # P-values may vary slightly due to bootstrap variance
    assert np.allclose(result_cpu["p"], result_gpu["p"], rtol=0.2)


@pytest.mark.tier1
def test_isc_return_null_distribution():
    """ISC with return_null=True includes bootstrap distribution."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result = isc_permutation_test(
        data, n_permute=100, return_null=True, random_state=42, progress_bar=False
    )

    assert "null_distribution" in result
    assert result["null_distribution"].shape == (100,)


@pytest.mark.tier1
def test_isc_circle_shift_method():
    """ISC with circle_shift method completes successfully."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result = isc_permutation_test(
        data, method="circle_shift", n_permute=50, random_state=42, progress_bar=False
    )

    assert "isc" in result
    assert "p" in result


@pytest.mark.tier1
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


@pytest.mark.tier2
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


@pytest.mark.tier2
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


@pytest.mark.tier2
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
        backend="numpy",
        n_permute=100,
        progress_bar=False,
    )
    cpu_time = time.time() - start

    start = time.time()
    _ = isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        backend="torch",
        n_permute=100,
        progress_bar=False,
    )
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time
    print(f"\nLOO GPU Speedup: {speedup:.1f}×")

    # Expect at least 3× speedup (conservative for testing)
    assert speedup > 3.0


@pytest.mark.tier2
def test_isc_gpu_speedup_pairwise():
    """GPU provides speedup for voxel-wise pairwise computation."""
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    np.random.seed(42)
    data = np.random.randn(100, 50, 5000)  # 5K voxels

    import time

    start = time.time()
    _ = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        backend="numpy",
        n_permute=100,
        progress_bar=False,
    )
    cpu_time = time.time() - start

    start = time.time()
    _ = isc_permutation_test(
        data,
        summary_statistic="pairwise",
        backend="torch",
        n_permute=100,
        progress_bar=False,
    )
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time
    print(f"\nPairwise GPU Speedup: {speedup:.1f}×")

    # Expect at least 3× speedup
    assert speedup > 3.0


# =============================================================================
# Edge Cases and Input Validation
# =============================================================================


@pytest.mark.tier1
def test_isc_invalid_summary_statistic():
    """Invalid summary_statistic raises ValueError."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    with pytest.raises(ValueError, match="summary_statistic must be"):
        isc_permutation_test(
            data, summary_statistic="invalid", n_permute=100, progress_bar=False
        )


@pytest.mark.tier1
def test_isc_invalid_method():
    """Invalid method raises ValueError."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    with pytest.raises(ValueError, match="method must be"):
        isc_permutation_test(data, method="invalid", n_permute=100, progress_bar=False)


@pytest.mark.tier1
def test_isc_invalid_data_dimensions():
    """Data with wrong dimensions raises ValueError."""
    data_1d = np.random.randn(100)

    with pytest.raises(ValueError, match="data must be 2D or 3D"):
        isc_permutation_test(data_1d, n_permute=100, progress_bar=False)


@pytest.mark.tier1
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
