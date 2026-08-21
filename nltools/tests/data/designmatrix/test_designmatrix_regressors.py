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
        vertical ``.append()`` rename map (``"col" -> ".nl_r{run}_col"``) keeps
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
        assert ".nl_r0_stim_c0" in out.columns
        assert ".nl_r1_stim_c0" in out.columns
        assert set(out.convolved) == {".nl_r0_stim_c0", ".nl_r1_stim_c0"}

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
        - Creates .nl_poly_0 (intercept), .nl_poly_1 (linear), .nl_poly_2 (quadratic)
        - All columns present in output
        - .confounds metadata updated

        Use case: Model baseline and slow drift in fMRI
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0] * 10}, sampling_freq=1)
        dm_poly = dm.add_poly(order=2, include_lower=True)

        # Should add 3 polynomial columns
        assert dm_poly.shape[1] == 4, "Should have stim + 3 polynomials"
        assert ".nl_poly_0" in dm_poly.columns
        assert ".nl_poly_1" in dm_poly.columns
        assert ".nl_poly_2" in dm_poly.columns

        # Metadata should track polynomials
        assert set(dm_poly.confounds) == {".nl_poly_0", ".nl_poly_1", ".nl_poly_2"}

    def test_add_poly_intercept_is_constant(self):
        """
        .nl_poly_0 (order=0) should be constant intercept term.

        Expected behavior:
        - Mean ≈ 1.0 (or some constant)
        - Variance ≈ 0 (constant across rows)

        Rationale: Legendre polynomial of order 0 is constant
        """
        dm = DesignMatrix({"stim": [1, 2, 3, 4]}, sampling_freq=1)
        dm_poly = dm.add_poly(order=0)

        # Intercept should be constant (very low variance)
        intercept = dm_poly[".nl_poly_0"]
        assert intercept.std() < 1e-10, "Intercept should have near-zero variance"

    def test_add_poly_linear_trend(self):
        """
        .nl_poly_1 (order=1) should be linear trend.

        Expected behavior:
        - Monotonic increase or decrease
        - First and last values have opposite signs (scaled -1 to 1)

        Use case: Model linear drift in signal
        """
        dm = DesignMatrix(np.zeros((20, 1)), sampling_freq=1, columns=["stim"])
        dm_poly = dm.add_poly(order=1, include_lower=False)

        linear = dm_poly[".nl_poly_1"]

        # Should be monotonic (always increasing or decreasing)
        diffs = np.diff(linear.to_numpy())
        assert np.all(diffs > 0) or np.all(diffs < 0), "Should be monotonic"

    def test_add_poly_without_lower_terms(self):
        """
        include_lower=False should add only specified order.

        Expected behavior:
        - Only .nl_poly_2 added, not .nl_poly_0 or .nl_poly_1

        Use case: Add specific polynomial without lower orders
        """
        dm = DesignMatrix({"stim": [1, 2, 3, 4]}, sampling_freq=1)
        dm_poly = dm.add_poly(order=2, include_lower=False)

        assert dm_poly.shape[1] == 2, "Should have stim + .nl_poly_2 only"
        assert ".nl_poly_2" in dm_poly.columns
        assert ".nl_poly_0" not in dm_poly.columns
        assert ".nl_poly_1" not in dm_poly.columns

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
        assert ".nl_cosine_1" in dm_dct.confounds

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

        # Should not have .nl_cosine_1 or .nl_cosine_2
        assert ".nl_cosine_1" not in dm_dct.columns
        assert ".nl_cosine_2" not in dm_dct.columns
        # Should have higher-order bases
        assert (
            ".nl_cosine_3" in dm_dct.columns or ".nl_cosine_1" in dm_dct.columns
        )  # Depends on numbering convention


class TestReservedPrefixNaming:
    """Machinery-generated columns live in the reserved ``.nl_`` namespace.

    Behavioral contract (v0.6.0):
    - Every column name DesignMatrix machinery invents — polynomial drift,
      DCT cosines, and run-separated variants — starts with ``.nl_``, a
      prefix users won't plausibly use for their own regressors.
    - Run separation renames a column ``col`` to ``.nl_r{run}_{base}`` where
      ``base`` is ``col`` with any leading ``.nl_`` stripped, so generated
      names never stack prefixes (``.nl_poly_0`` -> ``.nl_r0_poly_0``).
    - Detection of machinery-generated columns keys on this prefix, never on
      heuristics over user-controlled names (underscore counts, regexes on
      arbitrary text).
    """

    def test_add_poly_names_use_reserved_prefix(self):
        dm = DesignMatrix({"stim": [0, 1, 0, 1, 0, 1]}, sampling_freq=1)
        out = dm.add_poly(order=2, include_lower=True)
        assert {".nl_poly_0", ".nl_poly_1", ".nl_poly_2"} <= set(out.columns)
        assert {".nl_poly_0", ".nl_poly_1", ".nl_poly_2"} <= set(out.confounds)

    def test_add_dct_basis_names_use_reserved_prefix(self):
        dm = DesignMatrix(np.zeros((100, 1)), sampling_freq=0.5, columns=["stim"])
        out = dm.add_dct_basis(duration=60)
        cosine_cols = [c for c in out.columns if "cosine" in c]
        assert cosine_cols, "expected DCT basis columns"
        assert all(c.startswith(".nl_cosine_") for c in cosine_cols)
        assert ".nl_cosine_0" in out.columns  # include_constant=True default

    def test_run_separated_names_use_reserved_prefix(self):
        run1 = DesignMatrix(
            {"stim": [0, 1, 0, 1], "motion_x": [0.1, 0.2, 0.1, 0.3]},
            sampling_freq=1,
            confounds=["motion_x"],
        ).add_poly(0)
        run2 = DesignMatrix(
            {"stim": [1, 0, 1, 0], "motion_x": [0.4, 0.1, 0.2, 0.5]},
            sampling_freq=1,
            confounds=["motion_x"],
        ).add_poly(0)
        multi = run1.append(run2, axis=0)

        assert ".nl_r0_poly_0" in multi.columns
        assert ".nl_r1_poly_0" in multi.columns
        # User confounds get run-separated into the reserved namespace too:
        # the run-prefixed variant is a machinery-generated name.
        assert ".nl_r0_motion_x" in multi.columns
        assert ".nl_r1_motion_x" in multi.columns
        # No double-prefixed names anywhere.
        assert not any(c.count(".nl_") > 1 for c in multi.columns)

    def test_third_run_append_continues_numbering(self):
        def make_run(vals):
            return DesignMatrix({"stim": vals}, sampling_freq=1).add_poly(0)

        multi = make_run([0, 1]).append(make_run([1, 0]), axis=0)
        three = multi.append(make_run([1, 1]), axis=0)
        assert ".nl_r2_poly_0" in three.columns


class TestDriftGuardRunSeparation:
    """add_poly / add_dct_basis refuse designs with run-separated drift terms.

    Behavioral contract:
    - A design carrying run-separated polynomial OR cosine drift columns
      (``.nl_r{i}_poly_{j}`` / ``.nl_r{i}_cosine_{j}``) refuses BOTH adders:
      adding a global drift term on top of per-run ones is ambiguous.
    - Detection keys on the reserved prefix, so ordinary user confounds that
      merely contain underscores (24-parameter motion expansions like
      ``trans_x_sq``) never false-positive. Regression for the dartbrains
      seed-connectivity design that could not add drift terms at all.
    """

    @staticmethod
    def _multi_with_poly():
        run1 = DesignMatrix({"stim": [0, 1, 0, 1]}, sampling_freq=1).add_poly(0)
        run2 = DesignMatrix({"stim": [1, 0, 1, 0]}, sampling_freq=1).add_poly(0)
        return run1.append(run2, axis=0)

    @staticmethod
    def _multi_with_cosine():
        run1 = DesignMatrix(
            np.random.default_rng(0).standard_normal((50, 1)),
            sampling_freq=0.5,
            columns=["stim"],
        ).add_dct_basis(duration=60)
        run2 = DesignMatrix(
            np.random.default_rng(1).standard_normal((50, 1)),
            sampling_freq=0.5,
            columns=["stim"],
        ).add_dct_basis(duration=60)
        return run1.append(run2, axis=0)

    def test_raises_on_run_separated_polynomials(self):
        multi = self._multi_with_poly()
        with pytest.raises(ValueError, match="[Rr]un-separated"):
            multi.add_poly(order=1)
        with pytest.raises(ValueError, match="[Rr]un-separated"):
            multi.add_dct_basis(duration=4)

    def test_raises_on_run_separated_cosines(self):
        multi = self._multi_with_cosine()
        with pytest.raises(ValueError, match="[Rr]un-separated"):
            multi.add_poly(order=1)
        with pytest.raises(ValueError, match="[Rr]un-separated"):
            multi.add_dct_basis(duration=60)

    @pytest.mark.parametrize(
        "confound",
        [
            "trans_x_sq",
            "rot_x_diff",
            "rot_x_diff_sq",
            "a_b_c",
            "my_poly_thing",
            "my_cosine_thing",
        ],
    )
    def test_no_false_positive_on_underscored_confounds(self, confound):
        rng = np.random.default_rng(0)
        dm = DesignMatrix(
            {"stim": [0.0, 1, 0, 1, 0, 1], confound: rng.standard_normal(6)},
            sampling_freq=1,
            confounds=[confound],
        )
        out = dm.add_poly(order=1, include_lower=True)
        assert ".nl_poly_1" in out.columns
        out2 = dm.add_dct_basis(duration=4)
        assert any("cosine" in c for c in out2.columns)

    def test_motion_confounds_then_add_poly(self):
        """End-to-end repro of the dartbrains failure: 24-param motion + drift."""
        rng = np.random.default_rng(0)
        n = 20
        task = DesignMatrix(
            {"stim": rng.integers(0, 2, n).astype(float)}, sampling_freq=0.5
        )
        base_names = [f"{k}_{ax}" for k in ("trans", "rot") for ax in ("x", "y", "z")]
        motion_names = (
            base_names
            + [f"{b}_sq" for b in base_names]
            + [f"{b}_diff" for b in base_names]
            + [f"{b}_diff_sq" for b in base_names]
        )
        motion = DesignMatrix(
            {name: rng.standard_normal(n) for name in motion_names},
            sampling_freq=0.5,
        )
        dm = task.append(motion, axis=1, as_confounds=True).add_poly(
            order=2, include_lower=True
        )
        assert {".nl_poly_0", ".nl_poly_1", ".nl_poly_2"} <= set(dm.columns)
