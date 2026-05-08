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
    - Convolved columns are always renamed to ``<col>_c{i}``; the source
      column is dropped. ``.convolved`` lists post-suffix names.
    """

    def test_convolve_with_default_hrf_delays_response(self):
        """
        Default HRF convolution should delay and smooth response.

        Expected behavior:
        - Peak shifts later in time (HRF peaks ~5-6s after stimulus)
        - Signal is smoothed (convolution blurs sharp edges)
        - Output column is renamed to ``stim_c0`` (always-suffix policy)

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
        convolved_peak_idx = dm_conv["stim_c0"].arg_max()

        assert convolved_peak_idx > original_peak_idx, (
            "HRF convolution should delay peak response"
        )

    def test_convolve_with_custom_kernel(self):
        """
        Convolution with custom kernel should use provided function.

        Expected behavior:
        - Custom kernel applied via convolution
        - Results differ from default HRF
        - Output column renamed to ``stim_c0``

        Use case: Model non-canonical HRF, or other response functions (SCR, pupil)
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0, 0]}, sampling_freq=1)

        # Custom kernel: simple 3-point average (box-car smoothing)
        kernel = np.array([0.33, 0.33, 0.33])
        dm_conv = dm.convolve(conv_func=kernel)

        # First value should be smoothed
        assert dm_conv["stim_c0"].to_list()[0] == pytest.approx(0.33, abs=0.01)

    def test_convolve_drops_source_columns(self):
        """
        Convolution drops the un-convolved source column.

        Expected behavior:
        - After ``dm.convolve()``, ``stim`` no longer exists; ``stim_c0`` does.
        - Holds for both 1-D and 2-D kernels.

        Rationale: Callers want the convolved regressor in place of the
        boxcar; leaving both around bloats the design and breaks downstream
        column lookups.
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0, 0]}, sampling_freq=1)
        dm_conv = dm.convolve(conv_func=np.array([0.5, 0.5]))

        assert "stim" not in dm_conv.columns
        assert "stim_c0" in dm_conv.columns

    def test_convolve_ignores_polynomial_columns(self):
        """
        Convolution should skip columns marked as polynomials.

        Expected behavior:
        - Columns in .confounds list are NOT convolved (no suffix added)
        - Only stimulus columns are convolved

        Rationale: Confounds (intercept, drift, motion, …) represent baseline, not stimulus
        """
        dm = DesignMatrix(
            {"stim": [1, 0, 0, 0], "intercept": [1, 1, 1, 1]},
            sampling_freq=1,
            confounds=["intercept"],
        )

        dm_conv = dm.convolve()

        # Intercept should be unchanged AND keep its name (no suffix)
        assert "intercept" in dm_conv.columns
        assert dm_conv["intercept"].to_list() == [1, 1, 1, 1], (
            "Confound columns should not be convolved"
        )
        # And the stim column should be renamed
        assert "stim_c0" in dm_conv.columns
        assert "stim" not in dm_conv.columns

    def test_convolve_specific_columns_only(self):
        """
        Can specify which columns to convolve.

        Expected behavior:
        - Only specified columns are convolved (and renamed ``_c0``)
        - Other columns unchanged (no suffix)

        Use case: Convolve task regressors but not parametric modulators
        """
        dm = DesignMatrix(
            {"stim_A": [1, 0, 0, 0], "stim_B": [0, 1, 0, 0], "baseline": [1, 1, 1, 1]},
            sampling_freq=1,
        )

        dm_conv = dm.convolve(columns=["stim_A"])

        # stim_A is convolved → renamed; stim_B / baseline untouched
        assert "stim_A" not in dm_conv.columns
        assert "stim_A_c0" in dm_conv.columns
        assert dm_conv["stim_B"].to_list() == dm["stim_B"].to_list(), (
            "Unspecified column should be unchanged"
        )
        assert dm_conv["baseline"].to_list() == dm["baseline"].to_list()

    def test_convolve_updates_metadata(self):
        """
        Convolution should update .convolved metadata to post-suffix names.

        Expected behavior:
        - ``.convolved`` lists the actual column names in the output
          (post-suffix), not the source names
        - Metadata persists in returned DesignMatrix

        Use case: Track which regressors have been convolved, and let
        ``.append()``'s rename path find them in the dataframe.
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0]}, sampling_freq=1)
        dm_conv = dm.convolve(columns=["stim"])

        assert dm_conv.convolved == ["stim_c0"]

    def test_convolved_metadata_survives_multirun_append(self):
        """
        Regression: ``.convolved`` entries must be real column names so that
        vertical ``.append()`` rename map (``"col" -> "{run}_col"``) keeps
        metadata in sync with the dataframe.

        Before the always-suffix fix, ``.convolved`` carried pre-suffix names
        that didn't exist in the dataframe, so ``append()`` couldn't rename
        them and metadata silently drifted.
        """
        dm1 = DesignMatrix({"stim": [1, 0, 0, 0]}, sampling_freq=1).convolve()
        dm2 = DesignMatrix({"stim": [0, 0, 1, 0]}, sampling_freq=1).convolve()

        out = dm1.append(dm2, axis=0, unique_cols=["stim_c0"])

        # Both runs' convolved columns exist in the dataframe under
        # run-prefixed names, AND .convolved tracks them.
        assert "0_stim_c0" in out.columns
        assert "1_stim_c0" in out.columns
        assert set(out.convolved) == {"0_stim_c0", "1_stim_c0"}

    def test_convolve_is_idempotent_on_already_convolved(self):
        """Calling .convolve() again on a DM whose experimental regressors are
        all already convolved is a no-op (with a warning), not a re-convolution.

        Regression: previously convolve blindly appended ``_c0`` to every
        non-confound column, so ``language_c0`` became ``language_c0_c0`` —
        breaking downstream contrast strings written against the first-pass
        names. This contract bites file-loaded DMs in particular: events.tsv
        loads auto-convolve at construction (matching nilearn's default),
        and tutorials commonly chain ``.add_poly().convolve()`` afterwards.
        """
        import warnings

        dm = DesignMatrix({"stim": [1, 0, 1, 0]}, sampling_freq=1)
        dm1 = dm.convolve()
        # Sanity: first convolve produces the conventional _c0 suffix
        assert "stim_c0" in dm1.columns
        assert dm1.convolved == ["stim_c0"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dm2 = dm1.convolve()
        assert "stim_c0_c0" not in dm2.columns
        assert "stim_c0" in dm2.columns
        assert dm2.convolved == ["stim_c0"]
        assert any("no-op" in str(x.message) for x in w), (
            "Expected a no-op warning when nothing is left to convolve"
        )

    def test_convolve_refuses_explicit_already_convolved_column(self):
        """Explicit ``columns=`` cannot name an already-convolved column.

        ``_c0_c0`` has no biological meaning (HRF-shaped signal convolved
        with another kernel ≠ any real neural/hemodynamic process), and the
        only situations this call shape arises in practice are user typos /
        caller bugs / ill-defined "use a different kernel" intent. Raising
        keeps the column-name space well-defined: a column named ``stim_c{i}``
        always means "convolved exactly once".
        """
        dm = DesignMatrix({"stim": [1, 0, 1, 0]}, sampling_freq=1).convolve()
        with pytest.raises(ValueError, match="already-convolved"):
            dm.convolve(columns=["stim_c0"], conv_func=np.array([0.5, 0.5]))

    def test_convolve_partial_with_new_event_column(self):
        """When some experimental regressors are already convolved and a fresh
        un-convolved column is added (e.g., via ``.append()``), the next
        ``.convolve()`` should convolve only the new one and preserve the
        existing convolved columns + their metadata.
        """
        import polars as pl

        dm = DesignMatrix({"stim_a": [1, 0, 0, 0]}, sampling_freq=1).convolve()
        # Inject a fresh boxcar regressor (skipping the .append() machinery
        # to keep this focused on .convolve()'s partial-convolve path)
        dm_with_b = DesignMatrix(
            dm.data.with_columns(pl.Series("stim_b", [0, 1, 0, 0])),
            sampling_freq=1,
            convolved=dm.convolved,
        )
        dm_done = dm_with_b.convolve()
        assert "stim_a_c0" in dm_done.columns  # preserved
        assert "stim_b_c0" in dm_done.columns  # newly convolved
        assert set(dm_done.convolved) == {"stim_a_c0", "stim_b_c0"}

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
