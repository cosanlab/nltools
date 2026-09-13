"""Tests for nltools.algorithms.outliers — outlier detection and robust statistics."""

import numpy as np
import polars as pl
import pytest

from nltools.algorithms.outliers import trim, winsorize, find_spikes, zscore


class TestWinsorize:
    """Test winsorizing outlier handling."""

    def test_std_replace_with_cutoff(self, outlier_data):
        """Winsorize by std, replacing outliers with cutoff values."""
        out = winsorize(outlier_data, cutoff={"std": [2, 2]}, replace_with_cutoff=True)
        out = out.to_numpy().squeeze()
        expected = np.array(
            [
                92.0,
                19.0,
                101.0,
                58.0,
                556.97961997,
                91.0,
                26.0,
                78.0,
                10.0,
                13.0,
                -40.0,
                101.0,
                86.0,
                85.0,
                15.0,
                89.0,
                89.0,
                28.0,
                -5.0,
                41.0,
            ]
        )
        assert np.round(np.mean(out)) == np.round(np.mean(expected))


class TestTrim:
    """Test trimming outliers to null (as opposed to winsorize's clamping)."""

    def test_std_nulls_outliers(self, outlier_data):
        """Std trim nulls the extreme high outlier and leaves the rest intact."""
        out = trim(outlier_data, cutoff={"std": [2, 2]})
        out_vals = out["x"].to_list()
        # Only 1053 exceeds mean ± 2·std (it inflates std enough to keep -40 in).
        assert out["x"].null_count() == 1
        assert out_vals[4] is None


class TestZscore:
    """Test z-score normalization."""

    def test_zscore_polars_dataframe(self):
        """Z-scoring a polars DataFrame returns a polars DataFrame."""
        data = pl.DataFrame(
            {"a": np.random.randn(100), "b": np.random.randn(100) * 2 + 5}
        )
        result = zscore(data)
        assert isinstance(result, pl.DataFrame)
        for col in result.columns:
            np.testing.assert_almost_equal(result[col].mean(), 0, decimal=10)
            np.testing.assert_almost_equal(result[col].std(), 1, decimal=10)

    def test_zscore_rejects_invalid_input(self):
        """Z-scoring a non-DataFrame/Series input raises."""
        with pytest.raises(ValueError, match="Polars or pandas"):
            zscore([1, 2, 3])


class TestFindSpikes:
    """Test spike detection in neuroimaging data.

    `find_spikes` returns a `DesignMatrix` with spike indicator columns
    pre-marked as confounds. The legacy `TR` index column is dropped (row
    position is the time axis in the Polars-backed DM). Pass `TR=` or
    `sampling_freq=` to make `.convolve()` / `.append()` happy downstream.
    """

    @pytest.fixture
    def spike_nifti(self):
        """Tiny synthetic 4D nifti with two known global spikes."""
        import nibabel as nib

        rng = np.random.default_rng(0)
        n_tr = 30
        data = rng.standard_normal((4, 4, 4, n_tr))
        # Inject two global spikes well above 3σ.
        data[..., 5] += 50
        data[..., 20] += 50
        return nib.Nifti1Image(data, affine=np.eye(4))

    def test_returns_designmatrix(self, spike_nifti):
        from nltools.data import DesignMatrix

        dm = find_spikes(spike_nifti)
        assert isinstance(dm, DesignMatrix)

    def test_spike_columns_marked_as_confounds(self, spike_nifti):
        dm = find_spikes(spike_nifti)
        spike_cols = [c for c in dm.columns if "spike" in c]
        assert spike_cols, "expected at least one spike column"
        for c in spike_cols:
            assert c in dm.confounds

    def test_row_count_matches_input(self, spike_nifti):
        dm = find_spikes(spike_nifti)
        assert dm.shape[0] == 30  # n_tr

    def test_sampling_freq_kwarg_propagates(self, spike_nifti):
        dm = find_spikes(spike_nifti, sampling_freq=0.5)
        assert dm.sampling_freq == 0.5

    @pytest.mark.slow
    def test_find_spikes_brain_data(self):
        """Find spikes in simulated BrainData (slow — uses Simulator)."""
        from nltools.data import DesignMatrix
        from nltools.data.simulator import Simulator
        from nltools.mask import create_sphere

        sim = Simulator()
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data([0, 1], 1, reps=50, output_dir=None).apply_mask(s1)

        dm = find_spikes(d1)
        assert isinstance(dm, DesignMatrix)
        assert dm.shape[0] == len(d1)


class TestFindSpikesDeduplication:
    """`find_spikes` must not emit two indicators for the same TR.

    The global-signal and frame-difference detectors run independently, so a
    single bad volume is routinely caught by both. Each detection became its
    own one-hot column, producing exactly duplicated regressors and a rank
    deficient design — nltools manufacturing the very degeneracy that
    `BrainData.fit()` now warns about.
    """

    @pytest.fixture
    def colliding_nifti(self):
        """4D nifti whose spikes trip the global *and* difference detectors.

        A single-volume intensity jump raises the global mean for that TR and
        also the frame-to-frame difference around it, so both detectors fire.
        """
        import nibabel as nib

        rng = np.random.default_rng(0)
        n_tr = 40
        data = rng.standard_normal((4, 4, 4, n_tr))
        for t in (7, 21, 33):
            data[..., t] += 60
        return nib.Nifti1Image(data, affine=np.eye(4))

    @pytest.fixture
    def spike_nifti(self):
        """Two well-separated global spikes; the detectors should not collide."""
        import nibabel as nib

        rng = np.random.default_rng(0)
        n_tr = 30
        data = rng.standard_normal((4, 4, 4, n_tr))
        data[..., 5] += 50
        data[..., 20] += 50
        return nib.Nifti1Image(data, affine=np.eye(4))

    @staticmethod
    def _flagged_trs(dm):
        arr = dm.to_numpy()
        return [tuple(np.flatnonzero(arr[:, i])) for i in range(arr.shape[1])]

    def test_no_duplicate_regressors_by_default(self, colliding_nifti):
        dm = find_spikes(
            colliding_nifti, global_spike_cutoff=1.0, diff_spike_cutoff=1.0
        )
        flagged = self._flagged_trs(dm)
        assert len(flagged) == len(set(flagged)), (
            f"duplicate spike regressors: {flagged}"
        )

    def test_result_is_full_rank_by_default(self, colliding_nifti):
        dm = find_spikes(
            colliding_nifti, global_spike_cutoff=1.0, diff_spike_cutoff=1.0
        )
        X = dm.to_numpy()
        assert np.linalg.matrix_rank(X) == X.shape[1]

    def test_dedup_keeps_the_global_detection(self, colliding_nifti):
        """When both detectors flag a TR, the .nl_global_spike name is retained.

        The columns are bitwise identical, so this is purely about which name
        survives — but it must be deterministic, not insertion-ordered.
        Collisions are discovered by running each detector on its own.
        """
        dm = find_spikes(
            colliding_nifti, global_spike_cutoff=1.0, diff_spike_cutoff=1.0
        )
        global_only = find_spikes(
            colliding_nifti, global_spike_cutoff=1.0, diff_spike_cutoff=None
        )
        diff_only = find_spikes(
            colliding_nifti, global_spike_cutoff=None, diff_spike_cutoff=1.0
        )
        global_trs = {t for (t,) in self._flagged_trs(global_only)}
        diff_trs = {t for (t,) in self._flagged_trs(diff_only)}
        collided = global_trs & diff_trs
        assert collided, "fixture invariant: expected at least one collision"
        kept_by_tr = dict(zip((t for (t,) in self._flagged_trs(dm)), dm.columns))
        for tr in collided:
            assert kept_by_tr[tr].startswith(".nl_global_spike")


class TestFindSpikesNoSpikes:
    """A subject with no detected spikes must not break the design build.

    `find_spikes` returning a column-less DesignMatrix is the correct answer,
    but it still describes a specific number of timepoints. Polars derives
    height from columns, so a naive empty frame reports 0 rows — which made
    `.append()` raise "All Design Matrices must have the same number of rows!"
    and took down the whole first-level design for any clean subject.
    """

    @pytest.fixture
    def clean_nifti(self):
        import nibabel as nib

        rng = np.random.default_rng(0)
        return nib.Nifti1Image(rng.standard_normal((4, 4, 4, 60)), affine=np.eye(4))

    def test_no_spikes_appends_as_noop(self, clean_nifti):
        """The empty result must compose with a real design matrix."""
        from nltools.data import DesignMatrix

        rng = np.random.default_rng(0)
        task = DesignMatrix({"task": rng.standard_normal(60)}, sampling_freq=0.5)
        spikes = find_spikes(
            clean_nifti, global_spike_cutoff=100, diff_spike_cutoff=100, TR=2.0
        )
        out = task.append(spikes, axis=1, as_confounds=True)
        assert out.shape == (60, 1)
        assert out.columns == ["task"]

    def test_no_spikes_is_empty_property(self, clean_nifti):
        """No columns means no regressors, even though rows are known."""
        dm = find_spikes(clean_nifti, global_spike_cutoff=100, diff_spike_cutoff=100)
        assert dm.is_empty


class TestFindSpikesReservedPrefix:
    """Spike indicator columns live in the reserved ``.nl_`` namespace.

    `find_spikes` generates column names the user never chose, so they follow
    the same reserved-prefix convention as DesignMatrix's own machinery
    (``.nl_poly_*``, ``.nl_cosine_*``): ``.nl_global_spike{n}`` and
    ``.nl_diff_spike{n}``.
    """

    @pytest.fixture
    def spike_nifti(self):
        import nibabel as nib

        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 4, 4, 30))
        data[..., 5] += 50
        data[..., 20] += 50
        return nib.Nifti1Image(data, affine=np.eye(4))

    def test_spike_columns_use_reserved_prefix(self, spike_nifti):
        dm = find_spikes(spike_nifti)
        assert dm.shape[1] > 0
        assert all(
            c.startswith((".nl_global_spike", ".nl_diff_spike")) for c in dm.columns
        )
