"""Tests for matrix permutation tests (Mantel test) and utilities."""

import pytest
import numpy as np

from nltools.algorithms.inference import matrix_permutation_test
from nltools.tests.core.test_inference import (
    N_PERMUTE_STATS_COMPARISON,
    TOLERANCE_STATS_DETERMINISTIC,
    TOLERANCE_STATS_PVALUE,
)
from nltools.stats import matrix_permutation


class TestMatrixHelpers:
    """Test helper functions for matrix permutation."""

    def test_extract_upper_triangle(self):
        """Test extraction of upper triangle elements."""
        from nltools.algorithms.inference.matrix import _extract_matrix_elements

        # Create a simple 5×5 matrix
        matrix = np.arange(25).reshape(5, 5)

        # Extract upper triangle
        elements = _extract_matrix_elements(matrix, how="upper")

        # Should have n*(n-1)/2 = 5*4/2 = 10 elements
        assert len(elements) == 10

        # Verify correct elements (manually check a few)
        # Upper triangle indices: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        expected = np.array([1, 2, 3, 4, 7, 8, 9, 13, 14, 19])
        np.testing.assert_array_equal(elements, expected)

    def test_extract_lower_triangle(self):
        """Test extraction of lower triangle elements."""
        from nltools.algorithms.inference.matrix import _extract_matrix_elements

        matrix = np.arange(25).reshape(5, 5)
        elements = _extract_matrix_elements(matrix, how="lower")

        # Should have n*(n-1)/2 = 10 elements
        assert len(elements) == 10

        # Lower triangle indices: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), (4,0), (4,1), (4,2), (4,3)
        expected = np.array([5, 10, 11, 15, 16, 17, 20, 21, 22, 23])
        np.testing.assert_array_equal(elements, expected)

    def test_extract_full_matrix_no_diag(self):
        """Test extraction of full matrix without diagonal."""
        from nltools.algorithms.inference.matrix import _extract_matrix_elements

        matrix = np.arange(25).reshape(5, 5)
        elements = _extract_matrix_elements(matrix, how="full", include_diag=False)

        # Should have n*n - n = 25 - 5 = 20 elements (all except diagonal)
        assert len(elements) == 20

        # Should be concatenation of upper and lower triangles
        upper = np.array([1, 2, 3, 4, 7, 8, 9, 13, 14, 19])
        lower = np.array([5, 10, 11, 15, 16, 17, 20, 21, 22, 23])
        expected = np.concatenate([upper, lower])
        np.testing.assert_array_equal(elements, expected)

    def test_extract_full_matrix_with_diag(self):
        """Test extraction of full matrix with diagonal."""
        from nltools.algorithms.inference.matrix import _extract_matrix_elements

        matrix = np.arange(25).reshape(5, 5)
        elements = _extract_matrix_elements(matrix, how="full", include_diag=True)

        # Should have all n*n = 25 elements
        assert len(elements) == 25

        # Should be raveled matrix
        expected = matrix.ravel()
        np.testing.assert_array_equal(elements, expected)

    def test_permute_matrix_symmetric(self):
        """Test symmetric row+column permutation."""
        from nltools.algorithms.inference.matrix import _permute_matrix_symmetric

        # Create a simple matrix with identifiable structure
        matrix = np.arange(16).reshape(4, 4)

        # Identity permutation should leave matrix unchanged
        perm_identity = np.arange(4)
        result = _permute_matrix_symmetric(matrix, perm_identity)
        np.testing.assert_array_equal(result, matrix)

        # Known permutation: reverse order
        perm_reverse = np.array([3, 2, 1, 0])
        result = _permute_matrix_symmetric(matrix, perm_reverse)

        # Manually verify: should reverse both rows and columns
        expected = np.array(
            [[15, 14, 13, 12], [11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]]
        )
        np.testing.assert_array_equal(result, expected)

        # Verify shape preserved
        assert result.shape == matrix.shape

    def test_permute_matrix_preserves_symmetry(self):
        """Test that symmetric permutation preserves matrix symmetry."""
        from nltools.algorithms.inference.matrix import _permute_matrix_symmetric

        # Create a symmetric matrix
        matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])

        # Apply any permutation
        perm = np.array([2, 0, 1])
        result = _permute_matrix_symmetric(matrix, perm)

        # Result should still be symmetric
        np.testing.assert_array_equal(result, result.T)

    def test_compute_matrix_correlation_pearson(self):
        """Test Pearson correlation computation."""
        from nltools.algorithms.inference.matrix import _compute_matrix_correlation

        # Identical matrices should have r = 1.0
        matrix = np.random.randn(5, 5)
        r = _compute_matrix_correlation(matrix, matrix, metric="pearson")
        assert abs(r - 1.0) < 1e-10

        # Negated matrices should have r = -1.0
        r = _compute_matrix_correlation(matrix, -matrix, metric="pearson")
        assert abs(r - (-1.0)) < 1e-10

        # Uncorrelated random matrices should have |r| < 1.0
        np.random.seed(42)
        m1 = np.random.randn(10, 10)
        m2 = np.random.randn(10, 10)
        r = _compute_matrix_correlation(m1, m2, metric="pearson")
        assert abs(r) < 1.0

    def test_compute_matrix_correlation_spearman(self):
        """Test Spearman correlation computation."""
        from nltools.algorithms.inference.matrix import _compute_matrix_correlation

        # Identical matrices should have r = 1.0
        matrix = np.random.randn(5, 5)
        r = _compute_matrix_correlation(matrix, matrix, metric="spearman")
        assert abs(r - 1.0) < 1e-10

        # Monotonic relationship should have high Spearman
        m1 = np.arange(25).reshape(5, 5).astype(float)
        m2 = m1**2  # Monotonic but not linear
        r = _compute_matrix_correlation(m1, m2, metric="spearman")
        assert r > 0.99  # Should be very high for monotonic relationship

    def test_compute_matrix_correlation_kendall(self):
        """Test Kendall correlation computation."""
        from nltools.algorithms.inference.matrix import _compute_matrix_correlation

        # Identical matrices should have tau = 1.0
        matrix = np.random.randn(5, 5)
        tau = _compute_matrix_correlation(matrix, matrix, metric="kendall")
        assert abs(tau - 1.0) < 1e-10

        # Negated matrices should have tau ≈ -1.0
        tau = _compute_matrix_correlation(matrix, -matrix, metric="kendall")
        assert abs(tau - (-1.0)) < 1e-10

    def test_compute_matrix_correlation_all_extraction_modes(self):
        """Test that correlation works with all extraction modes."""
        from nltools.algorithms.inference.matrix import _compute_matrix_correlation

        # Create symmetric matrices (typical use case)
        np.random.seed(42)
        m1 = np.random.randn(6, 6)
        m1 = (m1 + m1.T) / 2  # Make symmetric
        m2 = np.random.randn(6, 6)
        m2 = (m2 + m2.T) / 2

        # All modes should work
        r_upper = _compute_matrix_correlation(m1, m2, how="upper")
        r_lower = _compute_matrix_correlation(m1, m2, how="lower")
        r_full_no_diag = _compute_matrix_correlation(
            m1, m2, how="full", include_diag=False
        )
        r_full_with_diag = _compute_matrix_correlation(
            m1, m2, how="full", include_diag=True
        )

        # For symmetric matrices, upper and lower should be identical
        assert abs(r_upper - r_lower) < 1e-10

        # Full modes should give reasonable correlations
        assert isinstance(r_full_no_diag, float)
        assert isinstance(r_full_with_diag, float)


class TestMatrixPermutationCPUParallel:
    """Test CPU-parallel implementation of matrix permutation."""

    def test_basic_functionality(self):
        """Test basic matrix permutation computation."""
        from nltools.algorithms.inference.matrix import _matrix_permutation_cpu_parallel

        # Create two correlated matrices
        np.random.seed(42)
        n = 20
        # Generate data matrix where rows are observations and columns are features
        data1 = np.random.randn(30, n)  # 30 observations, 20 features
        data2 = np.random.randn(30, n)
        # Compute correlation matrices (n×n)
        m1 = np.corrcoef(data1.T)  # Transpose to get features × features correlation
        m2 = np.corrcoef(data2.T)

        result = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=100,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )

        # Check result structure
        assert "correlation" in result
        assert "p" in result
        assert "backend" in result
        assert "null_dist" in result

        # Check types
        assert isinstance(result["correlation"], float)
        assert isinstance(result["p"], float)
        assert isinstance(result["backend"], str)
        assert isinstance(result["null_dist"], np.ndarray)

        # P-value should be in valid range
        assert 0 < result["p"] < 1

        # Null distribution should have correct size
        assert len(result["null_dist"]) == 100

    def test_determinism(self):
        """Test that same seed produces identical results."""
        from nltools.algorithms.inference.matrix import _matrix_permutation_cpu_parallel

        np.random.seed(42)
        n = 15
        m1 = np.random.randn(n, n)
        m2 = np.random.randn(n, n)

        # Run twice with same seed
        result1 = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=200,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )

        result2 = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=200,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )

        # Results should be EXACTLY identical (0.000% variance)
        assert result1["correlation"] == result2["correlation"]
        assert result1["p"] == result2["p"]
        np.testing.assert_array_equal(result1["null_dist"], result2["null_dist"])

    def test_parallel_consistency(self):
        """Test that n_jobs=1 and n_jobs=-1 produce identical results."""
        from nltools.algorithms.inference.matrix import _matrix_permutation_cpu_parallel

        np.random.seed(42)
        n = 12
        m1 = np.random.randn(n, n)
        m2 = np.random.randn(n, n)

        # Run with n_jobs=1
        result_serial = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=150,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )

        # Run with n_jobs=-1 (all cores)
        result_parallel = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=150,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=-1,
            random_state=42,
        )

        # Results should be identical
        assert result_serial["correlation"] == result_parallel["correlation"]
        assert result_serial["p"] == result_parallel["p"]
        np.testing.assert_array_equal(
            result_serial["null_dist"], result_parallel["null_dist"]
        )

    def test_return_null_distribution(self):
        """Test that null distribution is returned when requested."""
        from nltools.algorithms.inference.matrix import _matrix_permutation_cpu_parallel

        np.random.seed(42)
        m1 = np.random.randn(10, 10)
        m2 = np.random.randn(10, 10)

        # With return_null=True
        result = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=100,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )
        assert "null_dist" in result
        assert len(result["null_dist"]) == 100

        # Without return_null
        result = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=100,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=False,
            n_jobs=1,
            random_state=42,
        )
        assert "null_dist" not in result


class TestMatrixPermutationMain:
    """Test main matrix_permutation_test function."""

    def test_input_validation_non_square(self):
        """Test that non-square matrices are rejected."""

        m1 = np.random.randn(5, 6)  # Not square
        m2 = np.random.randn(5, 6)

        with pytest.raises(ValueError, match="must be square"):
            matrix_permutation_test(m1, m2, n_permute=100)

    def test_input_validation_mismatched_sizes(self):
        """Test that mismatched matrix sizes are rejected."""

        m1 = np.random.randn(5, 5)
        m2 = np.random.randn(6, 6)

        with pytest.raises(ValueError, match="must have same shape"):
            matrix_permutation_test(m1, m2, n_permute=100)

    def test_input_validation_invalid_metric(self):
        """Test that invalid metric is rejected."""

        m1 = np.random.randn(5, 5)
        m2 = np.random.randn(5, 5)

        with pytest.raises(ValueError, match="metric must be"):
            matrix_permutation_test(m1, m2, metric="invalid")

    def test_input_validation_invalid_how(self):
        """Test that invalid 'how' parameter is rejected."""

        m1 = np.random.randn(5, 5)
        m2 = np.random.randn(5, 5)

        with pytest.raises(ValueError, match="how must be"):
            matrix_permutation_test(m1, m2, how="invalid")

    def test_all_extraction_modes(self):
        """Test that all extraction modes work correctly."""

        np.random.seed(42)
        m1 = np.random.randn(8, 8)
        m2 = np.random.randn(8, 8)

        # All modes should work
        result_upper = matrix_permutation_test(
            m1, m2, how="upper", n_permute=100, random_state=42, n_jobs=1
        )
        result_lower = matrix_permutation_test(
            m1, m2, how="lower", n_permute=100, random_state=42, n_jobs=1
        )
        result_full_no_diag = matrix_permutation_test(
            m1,
            m2,
            how="full",
            include_diag=False,
            n_permute=100,
            random_state=42,
            n_jobs=1,
        )
        result_full_with_diag = matrix_permutation_test(
            m1,
            m2,
            how="full",
            include_diag=True,
            n_permute=100,
            random_state=42,
            n_jobs=1,
        )

        # All should return valid results
        assert 0 < result_upper["p"] < 1
        assert 0 < result_lower["p"] < 1
        assert 0 < result_full_no_diag["p"] < 1
        assert 0 < result_full_with_diag["p"] < 1

    def test_all_metrics(self):
        """Test that all correlation metrics work correctly."""

        np.random.seed(42)
        m1 = np.random.randn(8, 8)
        m2 = np.random.randn(8, 8)

        # All metrics should work
        result_pearson = matrix_permutation_test(
            m1, m2, metric="pearson", n_permute=100, random_state=42, n_jobs=1
        )
        result_spearman = matrix_permutation_test(
            m1, m2, metric="spearman", n_permute=100, random_state=42, n_jobs=1
        )
        result_kendall = matrix_permutation_test(
            m1, m2, metric="kendall", n_permute=100, random_state=42, n_jobs=1
        )

        # All should return valid results
        assert 0 < result_pearson["p"] < 1
        assert 0 < result_spearman["p"] < 1
        assert 0 < result_kendall["p"] < 1

    def test_include_diag_parameter(self):
        """Test that include_diag parameter works correctly."""

        np.random.seed(42)
        m1 = np.random.randn(8, 8)
        m2 = np.random.randn(8, 8)

        # Should work with both values
        result_no_diag = matrix_permutation_test(
            m1,
            m2,
            how="full",
            include_diag=False,
            n_permute=100,
            random_state=42,
            n_jobs=1,
        )
        result_with_diag = matrix_permutation_test(
            m1,
            m2,
            how="full",
            include_diag=True,
            n_permute=100,
            random_state=42,
            n_jobs=1,
        )

        # Both should return valid results
        assert 0 < result_no_diag["p"] < 1
        assert 0 < result_with_diag["p"] < 1

        # Results might differ since different elements are used
        # (Just verify they both work, not that they match)


class TestMatrixPermutationCorrectness:
    """Test statistical correctness of matrix permutation."""

    def test_identical_matrices(self):
        """Test that identical matrices produce perfect correlation and significant p-value."""

        # Create a matrix and test against itself
        np.random.seed(42)
        matrix = np.random.randn(20, 20)

        result = matrix_permutation_test(
            matrix, matrix, n_permute=500, random_state=42, n_jobs=1
        )

        # Correlation should be perfect
        assert abs(result["correlation"] - 1.0) < 1e-10

        # P-value should be highly significant
        assert result["p"] < 0.05

    def test_uncorrelated_matrices(self):
        """Test that uncorrelated random matrices produce non-significant p-value."""

        # Create two independent random matrices
        np.random.seed(42)
        m1 = np.random.randn(25, 25)
        m2 = np.random.randn(25, 25)

        result = matrix_permutation_test(
            m1, m2, n_permute=500, random_state=42, n_jobs=1
        )

        # Correlation should be weak
        assert abs(result["correlation"]) < 0.3

        # P-value should be non-significant
        assert result["p"] > 0.05

    def test_null_distribution_centered(self):
        """Test that null distribution is centered near zero for uncorrelated matrices."""

        np.random.seed(42)
        m1 = np.random.randn(20, 20)
        m2 = np.random.randn(20, 20)

        result = matrix_permutation_test(
            m1, m2, n_permute=1000, return_null=True, random_state=42, n_jobs=1
        )

        null_mean = np.mean(result["null_dist"])
        null_std = np.std(result["null_dist"])

        # Mean should be close to 0
        assert abs(null_mean) < 0.05

        # Standard deviation should be reasonable
        assert 0.01 < null_std < 0.15

    def test_backward_compatibility_stats_py(self):
        """Test backward compatibility with stats.py matrix_permutation.

        Note: Expect ~1-2% variance due to different RNG pattern
        (independent RandomState per permutation vs shared RNG state).
        Both implementations are statistically correct.
        """

        # Create test matrices
        np.random.seed(42)
        n = 15
        m1 = np.random.randn(n, n)
        m2 = np.random.randn(n, n)

        # New implementation
        result_new = matrix_permutation_test(
            m1,
            m2,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            random_state=42,
            n_jobs=1,
        )

        # Old implementation (stats.py)
        result_old = matrix_permutation(
            m1,
            m2,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            random_state=42,
            n_jobs=1,
        )

        # Observed correlation should be identical (deterministic computation)
        np.testing.assert_allclose(
            result_new["correlation"],
            result_old["correlation"],
            rtol=TOLERANCE_STATS_DETERMINISTIC,
        )

        # P-values may differ slightly (~1-2%) due to RNG pattern
        # This is expected and acceptable (see DESIGN.md)
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE
        )


# ============================================================================
# Matrix Utility Functions Tests (double_center, u_center, distance_correlation)
# ============================================================================


class TestDoubleCenter:
    """Test double_center function."""

    def test_double_center_basic(self):
        """Test basic double-centering operation."""
        from nltools.algorithms.inference.matrix import double_center

        # Create a simple matrix
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = double_center(mat)

        # After double-centering, row and column means should be zero
        assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)
        assert result.shape == mat.shape

    def test_double_center_symmetric(self):
        """Test double-centering on symmetric matrix."""
        from nltools.algorithms.inference.matrix import double_center

        np.random.seed(42)
        mat = np.random.randn(5, 5)
        mat = (mat + mat.T) / 2  # Make symmetric

        result = double_center(mat)

        # Should preserve symmetry
        assert np.allclose(result, result.T, atol=1e-10)
        assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)

    def test_double_center_raises_on_1d(self):
        """Test that double_center raises error on 1D input."""
        from nltools.algorithms.inference.matrix import double_center

        with pytest.raises(ValueError, match="Array should be 2d"):
            double_center(np.array([1, 2, 3]))


class TestUCenter:
    """Test u_center function."""

    def test_u_center_basic(self):
        """Test basic u-centering operation."""
        from nltools.algorithms.inference.matrix import u_center

        np.random.seed(42)
        mat = np.random.randn(5, 5)

        result = u_center(mat)

        # Diagonal should be zero
        assert np.allclose(np.diag(result), 0, atol=1e-10)
        assert result.shape == mat.shape

    def test_u_center_symmetric(self):
        """Test u-centering on symmetric matrix."""
        from nltools.algorithms.inference.matrix import u_center

        np.random.seed(42)
        mat = np.random.randn(5, 5)
        mat = (mat + mat.T) / 2  # Make symmetric

        result = u_center(mat)

        # Should preserve symmetry
        assert np.allclose(result, result.T, atol=1e-10)
        # Diagonal should be zero
        assert np.allclose(np.diag(result), 0, atol=1e-10)

    def test_u_center_diagonal_zero(self):
        """Test that u_center sets diagonal to zero."""
        from nltools.algorithms.inference.matrix import u_center

        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = u_center(mat)

        # Diagonal should be explicitly zero
        assert np.allclose(np.diag(result), 0, atol=1e-10)

    def test_u_center_raises_on_1d(self):
        """Test that u_center raises error on 1D input."""
        from nltools.algorithms.inference.matrix import u_center

        with pytest.raises(ValueError, match="Array should be 2d"):
            u_center(np.array([1, 2, 3]))


class TestDistanceCorrelation:
    """Test distance_correlation function."""

    def test_distance_correlation_basic(self):
        """Test basic distance correlation computation."""
        from nltools.algorithms.inference.matrix import distance_correlation

        np.random.seed(42)
        n = 20
        x = np.random.randn(n, 3)
        y = x + np.random.randn(n, 3) * 0.1  # Strongly correlated

        result = distance_correlation(x, y, bias_corrected=True, ttest=False)

        assert "dcorr" in result
        assert 0 <= result["dcorr"] <= 1
        assert result["dcorr"] > 0.5  # Should be high for correlated data

    def test_distance_correlation_bias_corrected(self):
        """Test distance correlation with bias correction."""
        from nltools.algorithms.inference.matrix import distance_correlation

        np.random.seed(42)
        n = 20
        x = np.random.randn(n, 3)
        y = x + np.random.randn(n, 3) * 0.1

        result_bias = distance_correlation(x, y, bias_corrected=True)
        result_no_bias = distance_correlation(x, y, bias_corrected=False)

        assert "dcorr" in result_bias
        assert "dcorr" in result_no_bias
        assert "dcorr_squared" in result_bias
        assert "dcorr_squared" not in result_no_bias

    def test_distance_correlation_with_ttest(self):
        """Test distance correlation with t-test."""
        from nltools.algorithms.inference.matrix import distance_correlation

        np.random.seed(42)
        n = 20
        x = np.random.randn(n, 3)
        y = x + np.random.randn(n, 3) * 0.1

        result = distance_correlation(x, y, bias_corrected=True, ttest=True)

        assert "dcorr" in result
        assert "t" in result
        assert "p" in result
        assert "df" in result
        assert 0 <= result["p"] <= 1
        assert result["df"] > 0

    def test_distance_correlation_1d_arrays(self):
        """Test distance correlation with 1D arrays."""
        from nltools.algorithms.inference.matrix import distance_correlation

        np.random.seed(42)
        n = 20
        x = np.random.randn(n)
        y = x + np.random.randn(n) * 0.1

        result = distance_correlation(x, y, bias_corrected=True)

        assert "dcorr" in result
        assert 0 <= result["dcorr"] <= 1

    def test_distance_correlation_independent(self):
        """Test distance correlation with independent data."""
        from nltools.algorithms.inference.matrix import distance_correlation

        np.random.seed(42)
        n = 20
        x = np.random.randn(n, 3)
        y = np.random.randn(n, 3)  # Independent

        result = distance_correlation(x, y, bias_corrected=True)

        assert "dcorr" in result
        # Should be low but not necessarily zero
        assert result["dcorr"] < 0.5

    def test_distance_correlation_ttest_requires_bias_corrected(self):
        """Test that ttest requires bias_corrected=True."""
        from nltools.algorithms.inference.matrix import distance_correlation

        np.random.seed(42)
        n = 20
        x = np.random.randn(n, 3)
        y = np.random.randn(n, 3)

        with pytest.raises(ValueError, match="bias_corrected must be true"):
            distance_correlation(x, y, bias_corrected=False, ttest=True)

    def test_distance_correlation_raises_on_3d(self):
        """Test that distance_correlation raises error on 3D input."""
        from nltools.algorithms.inference.matrix import distance_correlation

        np.random.seed(42)
        x = np.random.randn(5, 5, 5)
        y = np.random.randn(5, 5, 5)

        with pytest.raises(ValueError, match="Both arrays must be 1d or 2d"):
            distance_correlation(x, y)


class TestMatrixUtilitiesIntegration:
    """Test that matrix utilities work together correctly."""

    def test_double_center_vs_u_center(self):
        """Test that double_center and u_center produce different results."""
        from nltools.algorithms.inference.matrix import double_center, u_center

        np.random.seed(42)
        mat = np.random.randn(5, 5)

        dc_result = double_center(mat)
        uc_result = u_center(mat)

        # Results should be different
        assert not np.allclose(dc_result, uc_result, atol=1e-10)

        # Both should have same shape
        assert dc_result.shape == uc_result.shape

    def test_distance_correlation_uses_centering(self):
        """Test that distance_correlation correctly uses centering functions."""
        from nltools.algorithms.inference.matrix import distance_correlation

        np.random.seed(42)
        n = 20
        x = np.random.randn(n, 3)
        y = x + np.random.randn(n, 3) * 0.1

        # Both methods should work
        result_bias = distance_correlation(x, y, bias_corrected=True)
        result_no_bias = distance_correlation(x, y, bias_corrected=False)

        assert "dcorr" in result_bias
        assert "dcorr" in result_no_bias
        # Results should be different
        assert not np.allclose(result_bias["dcorr"], result_no_bias["dcorr"], atol=1e-6)

    def test_backward_compatibility_import_from_stats(self):
        """Test that functions can still be imported from stats.py for backward compatibility."""
        # After migration, these should still be importable from stats.py
        from nltools.stats import double_center, u_center, distance_correlation

        assert callable(double_center)
        assert callable(u_center)
        assert callable(distance_correlation)

        # Verify they produce same results
        np.random.seed(42)
        mat = np.random.randn(5, 5)
        result1 = double_center(mat)
        result2 = u_center(mat)

        assert result1.shape == mat.shape
        assert result2.shape == mat.shape
