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
        keeps them readable.
        """
        path = tmp_path / "legacy.csv"
        dm.write(path, sep="\t")

        back = DesignMatrix(path, sampling_freq=0.5, run_length="infer")

        assert back.columns == dm.columns
        np.testing.assert_allclose(back.to_numpy(), dm.to_numpy())


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
