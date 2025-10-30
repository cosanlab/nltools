"""
Test suite for Design_Matrix class.

Follows "imperative shell" pattern: tests focus on transform operations
and resampling functionality. Organized into logical sections for clarity.
"""

import numpy as np
from nltools.data import Design_Matrix
from nltools.algorithms.hrf import glover_hrf


class TestDesignMatrix:
    """Test Design_Matrix class - focus on transform operations."""

    # ==================== Transform Operations ====================

    def test_add_poly(self, sim_design_matrix):
        """Test polynomial basis expansion."""
        matp = sim_design_matrix.add_poly(2)
        assert matp.shape[1] == 7
        assert sim_design_matrix.add_poly(2, include_lower=False).shape[1] == 5

    def test_add_dct_basis(self, sim_design_matrix):
        """Test discrete cosine transform basis addition."""
        matpd = sim_design_matrix.add_dct_basis()
        assert matpd.shape[1] == 15

    def test_convolve(self, sim_design_matrix):
        """Test HRF convolution."""
        TR = 2.0
        assert sim_design_matrix.convolve().shape == sim_design_matrix.shape
        hrf = glover_hrf(TR, oversampling=1.0)
        assert (
            sim_design_matrix.convolve(conv_func=np.column_stack([hrf, hrf])).shape[1]
            == sim_design_matrix.shape[1] + 4
        )

    def test_zscore(self, sim_design_matrix):
        """Test selective z-scoring of columns."""
        import numpy as np

        matz = sim_design_matrix.zscore(columns=["face_A", "face_B"])
        # Check unchanged columns are identical (Polars-compatible)
        assert np.allclose(
            matz[["house_A", "house_B"]].to_numpy(),
            sim_design_matrix[["house_A", "house_B"]].to_numpy(),
        )

    def test_replace(self, sim_design_matrix):
        """Test data replacement."""
        assert (
            sim_design_matrix.replace_data(np.zeros((500, 4))).shape
            == sim_design_matrix.shape
        )

    # ==================== Resampling ====================

    def test_upsample(self, sim_design_matrix):
        """Test upsampling to higher frequency."""
        newTR = 1.0
        target = 1.0 / newTR
        assert (
            sim_design_matrix.upsample(target).shape[0]
            == sim_design_matrix.shape[0] * 2 - target * 2
        )

    def test_downsample(self, sim_design_matrix):
        """Test downsampling to lower frequency."""
        newTR = 4.0
        target = 1.0 / newTR
        assert (
            sim_design_matrix.downsample(target).shape[0]
            == sim_design_matrix.shape[0] / 2
        )

    # ==================== Utilities ====================

    def test_vif(self, sim_design_matrix):
        """Test VIF calculation and clean method."""
        matpd = sim_design_matrix.add_poly(2).add_dct_basis()
        assert all(matpd.vif() < 2.0)
        assert not all(matpd.vif(exclude_polys=False) < 2.0)
        matc = matpd.clean()
        assert matc.shape[1] == 16

    def test_clean(self, sim_design_matrix):
        """Test automatic collinearity removal."""
        # Drop correlated column (Polars-compatible)
        # Create a copy and add a duplicate of the first column
        first_col_name = sim_design_matrix.columns[0]
        first_col_data = sim_design_matrix[first_col_name]

        # Create new DesignMatrix with added correlated column
        corr_cols = sim_design_matrix.copy()
        corr_cols["new_col"] = first_col_data

        out = corr_cols.clean(verbose=True)
        assert out.shape[1] < corr_cols.shape[1]

        # Test for bug #413 about args combinations
        out = corr_cols.clean(fill_na=None, exclude_polys=True, verbose=True)
        assert out.shape[1] < corr_cols.shape[1]

        # Note: Testing duplicate column names is not applicable with Polars
        # Polars doesn't allow duplicate column names (raises DuplicateError on concat)
        # This is actually a feature - prevents accidental duplicate columns
        # Skip this test case for Polars implementation

    def test_append(self, sim_design_matrix):
        """Test appending design matrices with various column handling options."""
        mats = sim_design_matrix.append(sim_design_matrix)
        assert mats.shape[0] == sim_design_matrix.shape[0] * 2
        # Keep polys separate by default
        assert (mats.shape[1] - 4) == (sim_design_matrix.shape[1] - 4) * 2

        # Otherwise stack them
        mats = sim_design_matrix.append(sim_design_matrix, keep_separate=False)
        assert mats.shape[1] == sim_design_matrix.shape[1]
        assert mats.shape[0] == sim_design_matrix.shape[0] * 2

        # Keep a single stimulus column separate
        assert (
            sim_design_matrix.append(sim_design_matrix, unique_cols=["face_A"]).shape[1]
            == 5
        )

        # Keep a common stimulus class separate
        assert (
            sim_design_matrix.append(sim_design_matrix, unique_cols=["face*"]).shape[1]
            == 6
        )

        # Keep a common stimulus class and a different single stim separate
        assert (
            sim_design_matrix.append(
                sim_design_matrix, unique_cols=["face*", "house_A"]
            ).shape[1]
            == 7
        )

        # Keep multiple stimulus class separate
        assert (
            sim_design_matrix.append(
                sim_design_matrix, unique_cols=["face*", "house*"]
            ).shape[1]
            == 8
        )

        # Growing a multi-run design matrix; keeping things separate
        num_runs = 4
        all_runs = Design_Matrix(sampling_freq=0.5)
        for i in range(num_runs):
            run = Design_Matrix(
                np.array(
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                    ]
                ),
                sampling_freq=0.5,
                columns=["stim_A", "stim_B", "cond_C", "cond_D"],
            )
            run = run.add_poly(2)
            all_runs = all_runs.append(run, unique_cols=["stim*", "cond*"])
        assert all_runs.shape == (44, 28)
