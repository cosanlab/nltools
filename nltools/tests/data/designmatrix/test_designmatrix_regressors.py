import numpy as np
import pytest
from nltools.data.designmatrix import DesignMatrix


class TestDesignMatrixConvolution:
    """
    Test HRF convolution functionality.

    Behavioral contract:
    - Default convolution uses Glover HRF
    - Can provide custom kernels
    - Polynomial columns excluded from convolution
    - .convolved metadata tracks which columns were convolved
    """

    def test_convolve_with_default_hrf_delays_response(self):
        """
        Default HRF convolution should delay and smooth response.

        Expected behavior:
        - Peak shifts later in time (HRF peaks ~5-6s after stimulus)
        - Signal is smoothed (convolution blurs sharp edges)

        Use case: Model hemodynamic response in fMRI
        """
        # Box-car stimulus: on at TRs 2-4, off otherwise
        dm = DesignMatrix(
            {"stim": [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]},
            sampling_freq=0.5,  # 2s TR
        )
        dm_conv = dm.convolve()

        # Peak should shift later due to HRF delay
        original_peak_idx = dm["stim"].arg_max()
        convolved_peak_idx = dm_conv["stim"].arg_max()

        assert convolved_peak_idx > original_peak_idx, (
            "HRF convolution should delay peak response"
        )

    # NOTE: DesignMatrix may not currently have this feature, but we've always wanted to support multiple kernels as well passed in a list of inputs that automatically creates columns * kernels new convovled columns. See if it's easy to support this
    def test_convolve_with_custom_kernel(self):
        """
        Convolution with custom kernel should use provided function.

        Expected behavior:
        - Custom kernel applied via convolution
        - Results differ from default HRF

        Use case: Model non-canonical HRF, or other response functions (SCR, pupil)
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0, 0]}, sampling_freq=1)

        # Custom kernel: simple 3-point average (box-car smoothing)
        kernel = np.array([0.33, 0.33, 0.33])
        dm_conv = dm.convolve(conv_func=kernel)

        # First value should be smoothed
        assert dm_conv["stim"].to_list()[0] == pytest.approx(0.33, abs=0.01)

    def test_convolve_ignores_polynomial_columns(self):
        """
        Convolution should skip columns marked as polynomials.

        Expected behavior:
        - Columns in .confounds list are NOT convolved
        - Only stimulus columns are convolved

        Rationale: Confounds (intercept, drift, motion, …) represent baseline, not stimulus
        """
        dm = DesignMatrix(
            {"stim": [1, 0, 0, 0], "intercept": [1, 1, 1, 1]}, sampling_freq=1
        )
        dm.confounds = ["intercept"]

        dm_conv = dm.convolve()

        # Intercept should be unchanged
        assert dm_conv["intercept"].to_list() == [1, 1, 1, 1], (
            "Polynomial columns should not be convolved"
        )

    def test_convolve_specific_columns_only(self):
        """
        Can specify which columns to convolve.

        Expected behavior:
        - Only specified columns are convolved
        - Other columns unchanged

        Use case: Convolve task regressors but not parametric modulators
        """
        dm = DesignMatrix(
            {"stim_A": [1, 0, 0, 0], "stim_B": [0, 1, 0, 0], "baseline": [1, 1, 1, 1]},
            sampling_freq=1,
        )

        dm_conv = dm.convolve(columns=["stim_A"])

        # Only stim_A should be different
        assert dm_conv["stim_A"].to_list() != dm["stim_A"].to_list(), (
            "Specified column should be convolved"
        )
        assert dm_conv["stim_B"].to_list() == dm["stim_B"].to_list(), (
            "Unspecified column should be unchanged"
        )
        assert dm_conv["baseline"].to_list() == dm["baseline"].to_list()

    def test_convolve_updates_metadata(self):
        """
        Convolution should update .convolved metadata.

        Expected behavior:
        - .convolved list contains names of convolved columns
        - Metadata persists in returned DesignMatrix

        Use case: Track which regressors have been convolved
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0]}, sampling_freq=1)
        dm_conv = dm.convolve(columns=["stim"])

        assert "stim" in dm_conv.convolved, (
            "Convolved columns should be tracked in metadata"
        )

    # NOTE: see earlier note
    def test_convolve_with_multiple_kernels(self):
        """
        Support convolution with multiple kernels (2D array).

        Expected behavior:
        - Each column convolved with multiple kernels
        - New columns created for each kernel variant
        - Column names suffixed with kernel index (e.g., 'stim_c0', 'stim_c1')

        Use case: FIR models, temporal derivatives
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0, 0, 0]}, sampling_freq=1)

        # Two simple kernels
        kernels = np.array(
            [
                [1.0, 0.5, 0.0],  # Kernel 0: quick rise
                [0.0, 0.5, 1.0],  # Kernel 1: delayed rise
            ]
        ).T  # Shape: (3, 2) - samples x kernels

        dm_conv = dm.convolve(conv_func=kernels)

        # Should create stim_c0 and stim_c1
        assert "stim_c0" in dm_conv.columns
        assert "stim_c1" in dm_conv.columns


class TestDesignMatrixPolynomials:
    """
    Test polynomial and DCT basis function addition.

    Behavioral contract:
    - add_poly() adds Legendre polynomials (orthogonal on [-1, 1])
    - add_dct_basis() adds discrete cosine transform basis (high-pass filter)
    - .confounds metadata tracks polynomial / DCT columns (alongside other nuisance regressors)
    - Polynomials are NOT duplicated if added twice
    """

    def test_add_poly_creates_legendre_polynomials(self):
        """
        .add_poly(order=2) should add polynomials of order 0, 1, 2.

        Expected behavior:
        - Creates poly_0 (intercept), poly_1 (linear), poly_2 (quadratic)
        - All columns present in output
        - .confounds metadata updated

        Use case: Model baseline and slow drift in fMRI
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0] * 10}, sampling_freq=1)
        dm_poly = dm.add_poly(order=2, include_lower=True)

        # Should add 3 polynomial columns
        assert dm_poly.shape[1] == 4, "Should have stim + 3 polynomials"
        assert "poly_0" in dm_poly.columns
        assert "poly_1" in dm_poly.columns
        assert "poly_2" in dm_poly.columns

        # Metadata should track polynomials
        assert set(dm_poly.confounds) == {"poly_0", "poly_1", "poly_2"}

    def test_add_poly_intercept_is_constant(self):
        """
        poly_0 (order=0) should be constant intercept term.

        Expected behavior:
        - Mean ≈ 1.0 (or some constant)
        - Variance ≈ 0 (constant across rows)

        Rationale: Legendre polynomial of order 0 is constant
        """
        dm = DesignMatrix({"stim": [1, 2, 3, 4]}, sampling_freq=1)
        dm_poly = dm.add_poly(order=0)

        # Intercept should be constant (very low variance)
        poly_0 = dm_poly["poly_0"]
        assert poly_0.std() < 1e-10, "Intercept should have near-zero variance"

    def test_add_poly_linear_trend(self):
        """
        poly_1 (order=1) should be linear trend.

        Expected behavior:
        - Monotonic increase or decrease
        - First and last values have opposite signs (scaled -1 to 1)

        Use case: Model linear drift in signal
        """
        dm = DesignMatrix(np.zeros((20, 1)), sampling_freq=1, columns=["stim"])
        dm_poly = dm.add_poly(order=1, include_lower=False)

        poly_1 = dm_poly["poly_1"]

        # Should be monotonic (always increasing or decreasing)
        diffs = np.diff(poly_1.to_numpy())
        assert np.all(diffs > 0) or np.all(diffs < 0), "Should be monotonic"

    def test_add_poly_without_lower_terms(self):
        """
        include_lower=False should add only specified order.

        Expected behavior:
        - Only poly_2 added, not poly_0 or poly_1

        Use case: Add specific polynomial without lower orders
        """
        dm = DesignMatrix({"stim": [1, 2, 3, 4]}, sampling_freq=1)
        dm_poly = dm.add_poly(order=2, include_lower=False)

        assert dm_poly.shape[1] == 2, "Should have stim + poly_2 only"
        assert "poly_2" in dm_poly.columns
        assert "poly_0" not in dm_poly.columns
        assert "poly_1" not in dm_poly.columns

    def test_add_poly_idempotent(self):
        """
        Adding same polynomial twice should skip (no duplicates).

        Expected behavior:
        - Second call to add_poly(order=1) does nothing
        - Column count unchanged
        - Warning message printed (optional)

        Rationale: Prevents accidental duplication
        """
        dm = DesignMatrix({"stim": [1, 2, 3, 4]}, sampling_freq=1)
        dm1 = dm.add_poly(order=1)
        dm2 = dm1.add_poly(order=1)

        assert dm1.shape == dm2.shape, "Should not duplicate polynomials"
        assert dm1.columns == dm2.columns

    def test_add_dct_basis_creates_cosine_filters(self):
        """
        .add_dct_basis() should add discrete cosine basis functions.

        Expected behavior:
        - Multiple cosine_* columns added
        - Number of bases depends on duration and sampling_freq
        - .confounds metadata updated

        Use case: High-pass filtering (SPM-style)
        """
        dm = DesignMatrix(
            np.zeros((100, 1)),
            sampling_freq=0.5,  # 2s TR
            columns=["stim"],
        )

        dm_dct = dm.add_dct_basis(duration=60)  # 60s filter

        # Should add multiple cosine basis functions
        cosine_cols = [c for c in dm_dct.columns if "cosine" in c]
        assert len(cosine_cols) > 1, "Should add multiple DCT bases"

        # Metadata should track
        assert "cosine_1" in dm_dct.confounds

    def test_add_dct_basis_drop_parameter(self):
        """
        drop parameter should exclude low-frequency bases.

        Expected behavior:
        - drop=2 skips first 2 basis functions (constant and slowest)
        - Remaining bases start from index 3

        Use case: Remove very slow drifts beyond typical DCT filtering
        """
        dm = DesignMatrix(np.zeros((100, 1)), sampling_freq=0.5, columns=["stim"])

        # Drop first 2 bases (including constant, like SPM)
        dm_dct = dm.add_dct_basis(duration=60, drop=2)

        # Should not have cosine_1 or cosine_2
        assert "cosine_1" not in dm_dct.columns
        assert "cosine_2" not in dm_dct.columns
        # Should have higher-order bases
        assert (
            "cosine_3" in dm_dct.columns or "cosine_1" in dm_dct.columns
        )  # Depends on numbering convention
