"""Round-trip tests for DesignMatrix.write() and the file constructor.

Behavioral contract:
- Text formats are symmetric: whatever ``.write()`` produces for a given
  extension, the constructor reads back. The separator follows the extension
  (``.csv`` -> comma, everything else -> tab) on both sides.
- HDF5 is the metadata-preserving format: data, column names, sampling_freq,
  ``.convolved`` / ``.confounds``, ``.multi``, and the recorded row count of a
  column-less matrix all survive the round trip, and reading one back needs no
  ``run_length`` / ``sampling_freq`` (the file carries what it needs).
"""

import numpy as np
import polars as pl
import pytest
from nltools.data.designmatrix import DesignMatrix


@pytest.fixture
def dm():
    rng = np.random.default_rng(0)
    return DesignMatrix(
        {"cond_a": rng.standard_normal(12), "cond_b": rng.standard_normal(12)},
        sampling_freq=0.5,
    ).add_poly(1)


class TestTextRoundTrip:
    """.write() and the constructor agree on the separator for every extension.

    The writer used to default to tab regardless of extension while the reader
    picked the separator from the extension, so a ``.csv`` written by nltools
    read back as a single mashed column named ``'cond_a\\tcond_b'``.
    """

    @pytest.mark.parametrize("ext", [".csv", ".tsv", ".txt"])
    def test_round_trip_preserves_columns_and_values(self, dm, tmp_path, ext):
        path = tmp_path / f"design{ext}"
        dm.write(path)

        back = DesignMatrix(path, sampling_freq=0.5, run_length="infer")

        assert back.columns == dm.columns
        np.testing.assert_allclose(back.to_numpy(), dm.to_numpy())

    def test_csv_is_comma_separated_on_disk(self, dm, tmp_path):
        path = tmp_path / "design.csv"
        dm.write(path)

        header = path.read_text().splitlines()[0]
        assert "," in header
        assert "\t" not in header

    def test_tsv_is_tab_separated_on_disk(self, dm, tmp_path):
        path = tmp_path / "design.tsv"
        dm.write(path)

        header = path.read_text().splitlines()[0]
        assert "\t" in header

    def test_explicit_sep_overrides_the_extension(self, dm, tmp_path):
        """An explicit sep= still wins — callers may want a non-standard file."""
        path = tmp_path / "design.tsv"
        dm.write(path, sep=",")

        header = path.read_text().splitlines()[0]
        assert "," in header
        assert "\t" not in header

    def test_reader_recovers_from_a_mismatched_separator(self, dm, tmp_path):
        """A tab-separated .csv (what the old writer produced) still loads.

        Files written by earlier versions carry the wrong separator for their
        extension; falling back rather than handing back one mashed column
        keeps them readable. The recovery is a silent reinterpretation of the
        file, so it must announce itself with a warning.
        """
        path = tmp_path / "legacy.csv"
        dm.write(path, sep="\t")

        with pytest.warns(UserWarning, match="separator"):
            back = DesignMatrix(path, sampling_freq=0.5, run_length="infer")

        assert back.columns == dm.columns
        np.testing.assert_allclose(back.to_numpy(), dm.to_numpy())

    def test_single_column_with_alternate_delimiter_in_name_stays_one_column(
        self, tmp_path
    ):
        """A legitimate one-column .tsv whose header contains a comma is not re-read as CSV.

        The recovery heuristic fires on "one column whose name contains the
        alternate delimiter", but that is not proof of a mismatched separator:
        `onset,ms` is a perfectly valid single column name. Re-parsing such a
        file as CSV splits the header into `onset` + an all-null `ms`, silently
        corrupting the data — the re-parse must be validated and rejected.
        """
        path = tmp_path / "one_column.tsv"
        path.write_text("onset,ms\n1.0\n2.0\n3.0\n")

        back = DesignMatrix(path, sampling_freq=0.5, run_length="infer")

        assert back.columns == ["onset,ms"]
        np.testing.assert_allclose(back.to_numpy().ravel(), [1.0, 2.0, 3.0])


class TestH5RoundTrip:
    """HDF5 preserves the whole object, not just the numbers."""

    def test_data_and_columns_round_trip(self, dm, tmp_path):
        path = tmp_path / "design.h5"
        dm.write(path)

        back = DesignMatrix(path)

        assert back.columns == dm.columns
        np.testing.assert_allclose(back.to_numpy(), dm.to_numpy())

    def test_needs_no_run_length_or_sampling_freq(self, dm, tmp_path):
        """An .h5 is a serialized DesignMatrix, not a table needing interpretation."""
        path = tmp_path / "design.h5"
        dm.write(path)

        back = DesignMatrix(path)

        assert back.sampling_freq == 0.5

    def test_metadata_round_trips(self, tmp_path):
        rng = np.random.default_rng(0)
        dm = (
            DesignMatrix({"stim": rng.standard_normal(10)}, sampling_freq=0.5)
            .convolve()
            .add_poly(0)
        )
        path = tmp_path / "design.h5"
        dm.write(path)

        back = DesignMatrix(path)

        assert back.convolved == dm.convolved
        assert back.confounds == dm.confounds
        assert back.multi == dm.multi

    def test_multi_run_flag_round_trips(self, tmp_path):
        def run(seed):
            rng = np.random.default_rng(seed)
            return DesignMatrix(
                {"stim": rng.standard_normal(6)}, sampling_freq=1
            ).add_poly(0)

        multi = run(0).append(run(1), axis=0)
        path = tmp_path / "multi.h5"
        multi.write(path)

        back = DesignMatrix(path)

        assert back.multi is True
        assert back.columns == multi.columns
        assert back.confounds == multi.confounds

    def test_reserved_prefix_column_names_survive(self, dm, tmp_path):
        path = tmp_path / "design.h5"
        dm.write(path)

        back = DesignMatrix(path)

        assert ".nl_poly_0" in back.columns
        assert ".nl_poly_1" in back.columns

    def test_integer_columns_keep_their_dtype(self, tmp_path):
        """Spike indicators are integer one-hots; a round trip shouldn't float them."""
        dm = DesignMatrix(
            {"stim": [0.5, 1.5, 2.5], ".nl_global_spike1": [0, 1, 0]},
            sampling_freq=1,
        )
        path = tmp_path / "design.h5"
        dm.write(path)

        back = DesignMatrix(path)

        assert (
            back.data.schema[".nl_global_spike1"] == dm.data.schema[".nl_global_spike1"]
        )

    def test_column_less_matrix_keeps_its_row_count(self, tmp_path):
        """find_spikes() on a clean subject: no columns, but a real height."""
        dm = DesignMatrix(pl.DataFrame(), sampling_freq=0.5, n_rows=25)
        path = tmp_path / "empty.h5"
        dm.write(path)

        back = DesignMatrix(path)

        assert back.shape == (25, 0)

    def test_explicit_kwargs_override_stored_metadata(self, dm, tmp_path):
        path = tmp_path / "design.h5"
        dm.write(path)

        back = DesignMatrix(path, sampling_freq=2.0)

        assert back.sampling_freq == 2.0


class TestH5LegacyLayout:
    """Files written by the pre-reader h5 writer still load.

    That writer stored a plain float matrix in ``data`` alongside an ``S``-typed
    ``columns`` dataset. Nothing could read them back, but they exist on disk.
    """

    def test_reads_legacy_numeric_layout(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "legacy.h5"
        values = np.arange(6, dtype=float).reshape(3, 2)

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=values)
            f.create_dataset(
                "columns", data=np.array(["stim", ".nl_poly_0"], dtype="S")
            )
            meta = f.create_group("metadata")
            meta.attrs["sampling_freq"] = 0.5
            meta.attrs["convolved"] = np.array([], dtype="S")
            meta.attrs["confounds"] = np.array([".nl_poly_0"], dtype="S")
            meta.attrs["multi"] = False
            meta.attrs["obj_type"] = "design_matrix"

        back = DesignMatrix(path)

        assert back.columns == ["stim", ".nl_poly_0"]
        np.testing.assert_allclose(back.to_numpy(), values)
        assert back.sampling_freq == 0.5
        assert back.confounds == [".nl_poly_0"]

    @staticmethod
    def _write_legacy_h5(path, values, columns, *, confounds=(), multi=False):
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=values)
            f.create_dataset("columns", data=np.array(columns, dtype="S"))
            meta = f.create_group("metadata")
            meta.attrs["sampling_freq"] = 0.5
            meta.attrs["convolved"] = np.array([], dtype="S")
            meta.attrs["confounds"] = np.array(list(confounds), dtype="S")
            meta.attrs["multi"] = multi
            meta.attrs["obj_type"] = "design_matrix"

    def test_translates_legacy_generated_names(self, tmp_path):
        """Pre-`.nl_` generated names load into the reserved namespace (A-2).

        Recognition keys solely on the `.nl_` prefix, so a legacy file's
        `poly_0` / `cosine_1` / `global_spike1` must be translated at load
        time or every downstream recognizer treats them as user columns.
        """
        pytest.importorskip("h5py")
        path = tmp_path / "legacy.h5"
        rng = np.random.default_rng(0)
        n = 20
        values = np.column_stack(
            [
                rng.standard_normal(n),
                rng.standard_normal(n),
                np.ones(n),
                np.cos(np.linspace(0, np.pi, n)),
                (np.arange(n) == 3).astype(float),
            ]
        )
        self._write_legacy_h5(
            path,
            values,
            ["stim_a", "stim_b", "poly_0", "cosine_1", "global_spike1"],
            confounds=["poly_0", "cosine_1", "global_spike1"],
        )

        back = DesignMatrix(path)

        assert back.columns == [
            "stim_a",
            "stim_b",
            ".nl_poly_0",
            ".nl_cosine_1",
            ".nl_global_spike1",
        ]
        assert back.confounds == [".nl_poly_0", ".nl_cosine_1", ".nl_global_spike1"]
        # vif(exclude_confounds=False) must drop the all-ones legacy intercept
        # instead of producing a singular/NaN correlation matrix.
        vif = back.vif(exclude_confounds=False)
        assert vif is not None
        assert np.isfinite(vif).all()

    def test_legacy_run_separated_names_keep_single_mechanism_behavior(self, tmp_path):
        """Legacy `0_poly_0`-style run columns translate to `.nl_r0_poly_0` (A-2).

        Untranslated, add_poly() silently adds a global drift beside per-run
        drift, and a later append restarts run numbering at `.nl_r0_`.
        """
        pytest.importorskip("h5py")
        path = tmp_path / "legacy_multi.h5"
        rng = np.random.default_rng(1)
        n = 12
        run = np.repeat([0, 1], n // 2)
        values = np.column_stack(
            [
                rng.standard_normal(n),
                (run == 0).astype(float),
                (run == 1).astype(float),
            ]
        )
        self._write_legacy_h5(
            path,
            values,
            ["stim", "0_poly_0", "1_poly_0"],
            confounds=["0_poly_0", "1_poly_0"],
            multi=True,
        )

        back = DesignMatrix(path)

        assert back.columns == ["stim", ".nl_r0_poly_0", ".nl_r1_poly_0"]
        assert back.confounds == [".nl_r0_poly_0", ".nl_r1_poly_0"]

        # Adding global drift beside per-run drift is ambiguous — must raise.
        with pytest.raises(ValueError):
            back.add_poly(1)
        with pytest.raises(ValueError):
            back.add_dct_basis()

        # Appending a new run continues numbering at r2 instead of restarting.
        new_run = DesignMatrix(
            {"stim": rng.standard_normal(6)}, sampling_freq=0.5
        ).add_poly(0)
        combined = back.append(new_run, keep_separate=True)
        assert ".nl_r2_poly_0" in combined.columns
        assert combined.columns.count(".nl_r0_poly_0") == 1
