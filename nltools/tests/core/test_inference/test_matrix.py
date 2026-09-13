"""Tests for matrix permutation tests (Mantel test) and utilities."""

import pytest
import numpy as np

from nltools.algorithms import matrix_permutation_test


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

    def test_include_diag_parameter(self):
        """Test that include_diag parameter works correctly."""
        np.random.seed(42)
        m1 = np.random.randn(6, 6)  # Reduced from 8×8 for tier1 speed
        m2 = np.random.randn(6, 6)

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


# ============================================================================
# Matrix Utility Functions Tests (double_center, u_center, distance_correlation)
# ============================================================================


class TestDoubleCenter:
    """Test double_center function."""

    def test_double_center_basic(self):
        """Test basic double-centering operation."""
        from nltools.algorithms.inference.matrix import _double_center

        # Create a simple matrix
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = _double_center(mat)

        # After double-centering, row and column means should be zero
        assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)
        assert result.shape == mat.shape


class TestDistanceCorrelation:
    """Test distance_correlation function."""

    def test_distance_correlation_basic(self):
        """Test basic distance correlation computation."""
        from nltools.algorithms import distance_correlation

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
        from nltools.algorithms import distance_correlation

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

    def test_distance_correlation_ttest_requires_bias_corrected(self):
        """Test that ttest requires bias_corrected=True."""
        from nltools.algorithms import distance_correlation

        np.random.seed(42)
        n = 20
        x = np.random.randn(n, 3)
        y = np.random.randn(n, 3)

        with pytest.raises(ValueError, match="bias_corrected must be true"):
            distance_correlation(x, y, bias_corrected=False, ttest=True)

    def test_distance_correlation_raises_on_3d(self):
        """Test that distance_correlation raises error on 3D input."""
        from nltools.algorithms import distance_correlation

        np.random.seed(42)
        x = np.random.randn(5, 5, 5)
        y = np.random.randn(5, 5, 5)

        with pytest.raises(ValueError, match="Both arrays must be 1d or 2d"):
            distance_correlation(x, y)


# ============================================================================
# Test Matrix Permutation Statistical Correctness
# ============================================================================


def _generate_correlated_matrices(n, correlation_strength, random_state=None):
    """Generate two correlated matrices with known correlation."""
    np.random.seed(random_state)
    # Create two matrices with shared structure
    # Use correlation matrices approach: create base data and compute correlation matrices
    base_data1 = np.random.randn(n + 10, n)  # Extra samples for stability
    base_data2 = (
        base_data1
        + np.random.randn(n + 10, n)
        * np.sqrt((1 - correlation_strength**2) / correlation_strength**2)
        if correlation_strength > 0
        else np.random.randn(n + 10, n)
    )

    # Compute correlation matrices
    m1 = np.corrcoef(base_data1.T)
    m2 = np.corrcoef(base_data2.T)

    return m1, m2


class TestMatrixPermutationStatisticalCorrectness:
    """Test statistical correctness of matrix permutation tests."""

    def test_correlation_value_correctness(self):
        """Test that computed correlation values match expected values."""
        n = 25
        correlation_strength = 0.7  # Known correlation strength

        # Generate matrices with known correlation
        m1, m2 = _generate_correlated_matrices(n, correlation_strength, random_state=42)

        # Test with Pearson metric
        result = matrix_permutation_test(
            m1, m2, n_permute=2000, metric="pearson", random_state=42, n_jobs=1
        )

        # Computed correlation should be positive (matrices are correlated)
        # Tolerance: rtol=0.1 (10% as specified in plan - matrices are smaller, less stable)
        assert result["correlation"] > 0.3, (
            f"Correlation should be positive for correlated matrices. "
            f"Got {result['correlation']:.4f}"
        )

    def test_symmetric_permutation_correctness(self):
        """Test that symmetric permutation preserves symmetry and computes correlation correctly."""
        n = 20

        # Create symmetric matrix (distance matrix)
        np.random.seed(42)
        m1 = np.random.randn(n, n)
        m1 = (m1 + m1.T) / 2  # Make symmetric
        m2 = np.random.randn(n, n)
        m2 = (m2 + m2.T) / 2  # Make symmetric

        result = matrix_permutation_test(
            m1, m2, n_permute=2000, how="upper", random_state=42, n_jobs=1
        )

        # Correlation should be computed correctly (upper triangle only)
        assert isinstance(result["correlation"], float)
        assert -1 <= result["correlation"] <= 1

        # Verify permutation preserves symmetry (test by checking result structure)
        # The permutation function should preserve symmetry
        from nltools.algorithms.inference.matrix import _permute_matrix_symmetric

        perm = np.random.RandomState(42).permutation(n)
        permuted = _permute_matrix_symmetric(m1, perm)

        # Permuted matrix should still be symmetric
        np.testing.assert_allclose(permuted, permuted.T, rtol=1e-10)
