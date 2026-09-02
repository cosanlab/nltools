"""Read and write DesignMatrix objects.

Loads BIDS events and tabular confound files into the frame a `DesignMatrix`
wraps, converts to pandas/NumPy, and round-trips through TSV/CSV or HDF5
(which also preserves the metadata).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nltools.utils import find_stack_level

if TYPE_CHECKING:
    import pandas as pd

    from nltools.data.designmatrix import DesignMatrix


def events_to_dm(
    events: pl.DataFrame | pd.DataFrame,
    *,
    run_length: int,
    sampling_freq: float,
) -> pl.DataFrame:
    """Convert a BIDS events table to boxcar regressors aligned to TRs.

    Uses `nilearn.glm.first_level.make_first_level_design_matrix` with
    `hrf_model=None` to sample events onto the TR grid without HRF
    convolution — the caller is expected to call `DesignMatrix.convolve()`
    explicitly when convolution is desired. Drops nilearn's auto-added
    `constant` column; users add the intercept via `add_poly(0)`.

    Args:
        events (pl.DataFrame | pd.DataFrame): Events table with BIDS columns
            `onset`, `duration`, `trial_type` (required); `modulation` is
            passed through if present.
        run_length (int): Number of TRs the run contains.
        sampling_freq (float): Sampling frequency in Hz (= 1/TR).

    Returns:
        pl.DataFrame: One column per unique `trial_type`, values in
            {0, modulation} indicating where each condition is active.
    """
    import pandas as pd
    from nilearn.glm.first_level import make_first_level_design_matrix

    if isinstance(events, pl.DataFrame):
        events = pd.DataFrame(events.to_dict(as_series=False))

    tr = 1.0 / sampling_freq
    frame_times = np.arange(run_length) * tr
    dm = make_first_level_design_matrix(
        frame_times,
        events=events,
        hrf_model=None,
        drift_model=None,
    )
    if "constant" in dm.columns:
        dm = dm.drop(columns=["constant"])
    # Avoid pyarrow dep on the pandas → polars hop (matches `to_pandas` below).
    return pl.DataFrame({str(c): dm[c].to_numpy() for c in dm.columns})


def separator_for_path(path: str | Path) -> str:
    """Return the delimiter a text DesignMatrix file uses, from its extension.

    The single source of truth for both `write` and `load_from_file`, so a
    file nltools writes is always a file nltools can read back. ``.csv`` means
    comma; every other extension means tab, matching the BIDS convention for
    ``.tsv`` and keeping the historical default for ``.txt`` and friends.

    Args:
        path (str | Path): File path whose extension decides the delimiter.

    Returns:
        str: ``','`` for `.csv`, ``'\\t'`` otherwise.
    """
    return "," if Path(path).suffix.lower() == ".csv" else "\t"


def _read_delimited(path: Path, sep: str) -> pl.DataFrame:
    """Read a delimited text file, recovering from a mismatched separator.

    nltools <= 0.6.0 wrote tab-separated data into whatever extension it was
    handed, so a ``.csv`` on disk may really be a TSV. Parsing it with the
    wrong delimiter yields a single column whose *name* still contains the
    real one. That is only a hint, not proof — ``onset,ms`` is a valid
    single-column TSV header — so the re-parse is accepted only when it
    actually produces multiple fully-populated columns (a header that merely
    *contains* the alternate delimiter splits into all-null columns instead),
    and it warns, since it reinterprets the file against its extension.
    """
    import warnings

    def read(delimiter: str) -> pl.DataFrame:
        return pl.read_csv(
            path,
            separator=delimiter,
            null_values=["n/a", "N/A", "NA", ""],
            infer_schema_length=10_000,
        )

    raw = read(sep)
    if raw.width == 1:
        alternate = "," if sep == "\t" else "\t"
        if alternate in raw.columns[0]:
            reparsed = read(alternate)
            plausible = reparsed.width > 1 and (
                reparsed.height == 0
                or all(
                    reparsed[c].null_count() < reparsed.height for c in reparsed.columns
                )
            )
            if plausible:
                shown = {",": "','", "\t": "tab"}
                warnings.warn(
                    f"{path.name} parsed as a single column with the "
                    f"{shown[sep]} separator its extension implies, but "
                    f"re-parsing with {shown[alternate]} produced "
                    f"{reparsed.width} columns — using the re-parse. The "
                    f"file's separator does not match its extension "
                    f"(nltools <= 0.6.0 wrote such files); rewrite it to "
                    f"silence this warning.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                return reparsed
    return raw


def load_from_file(
    path: str | Path,
    *,
    run_length: int | str,
    sampling_freq: float,
) -> tuple[pl.DataFrame, bool]:
    """Read a TSV/CSV into the frame a DesignMatrix wraps.

    Dispatches on column inspection: when `onset` and `duration` are both
    present the file is a BIDS events table and becomes a boxcar design via
    `events_to_dm` (unconvolved; the caller convolves later); otherwise it is
    a tabular file (confounds / nuisance regressors) read as-is.

    ``run_length='infer'`` is accepted only for the tabular path; events
    files must provide an explicit integer (they have a variable row count
    per run, unlike confounds which are 1 row per TR).

    Args:
        path (str | Path): Path to a `.tsv` or `.csv` file.
        run_length (int | str): Number of TRs, or ``'infer'`` for tabular inputs.
        sampling_freq (float): Sampling frequency in Hz (= 1/TR).

    Returns:
        tuple[pl.DataFrame, bool]: `(frame, is_events)` — `is_events` signals to
            the caller that the columns are experimental regressors rather than
            nuisance.
    """
    p = Path(path)
    raw = _read_delimited(p, separator_for_path(p))

    is_events = "onset" in raw.columns and "duration" in raw.columns

    if is_events:
        if run_length == "infer":
            raise ValueError(
                "run_length='infer' is not valid for BIDS events files "
                "(the row count is the number of events, not the number "
                "of TRs). Pass an explicit integer run_length."
            )
        data_df = events_to_dm(
            raw,
            run_length=int(run_length),
            sampling_freq=sampling_freq,
        )
        return data_df, True

    if run_length != "infer":
        rl = int(run_length)
        if raw.height != rl:
            raise ValueError(
                f"Tabular file {p.name} has {raw.height} rows but "
                f"run_length={rl}. Pass run_length='infer' to accept "
                f"whatever the file contains."
            )
    return raw, False


def to_pandas(dm: DesignMatrix):
    """Convert DesignMatrix to pandas DataFrame.

    Uses dict-based conversion to avoid pyarrow dependency. This is slightly
    slower (~10-20%) than pyarrow-based conversion but removes the dependency.

    Args:
        dm (DesignMatrix): DesignMatrix instance.

    Returns:
        pd.DataFrame: pandas DataFrame with the same data and column names.

    Examples:
        ```python
        dm = DesignMatrix(np.random.randn(100, 3))
        pd_df = to_pandas(dm)
        type(pd_df)  # → <class 'pandas.core.frame.DataFrame'>
        ```
    """
    import pandas as pd

    return pd.DataFrame(dm.data.to_dict(as_series=False))


def to_numpy(dm: DesignMatrix) -> np.ndarray:
    """Convert a DesignMatrix to a NumPy array.

    Returns the data columns as a 2D array (rows x columns), preserving the
    DataFrame's column order.

    Args:
        dm (DesignMatrix): DesignMatrix instance.

    Returns:
        np.ndarray: 2D array with shape ``(n_samples, n_columns)``.

    Examples:
        ```python
        dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)
        arr = to_numpy(dm)
        arr.shape  # → (3, 2)
        ```
    """
    # np.asarray(dm) routes through DesignMatrix.__array__, which knows how to
    # honor the recorded length of a column-less matrix (polars itself would
    # report (0, 0)).
    return np.asarray(dm)


def write(dm: DesignMatrix, file_name: str, sep: str | None = None) -> None:
    """Write DesignMatrix to file.

    Supports TSV, CSV, and HDF5 formats. The format is automatically
    determined by file extension.

    Args:
        dm (DesignMatrix): DesignMatrix instance.
        file_name (str): Output file path with a `.tsv`, `.csv`, `.h5`, or
            `.hdf5` extension.
        sep (str | None): Column separator for text files. Defaults to the
            delimiter the extension implies (comma for `.csv`, tab otherwise),
            so the file reads back correctly; pass a value to override.
            Ignored for HDF5.

    Examples:
        ```python
        dm = DesignMatrix(np.random.randn(100, 3), sampling_freq=1)
        write(dm, "design_matrix.tsv")  # tab separated (BIDS compatible)
        write(dm, "design_matrix.csv")  # comma separated
        write(dm, "design_matrix.h5")   # HDF5, metadata preserved
        ```

    Note:
        TSV format is recommended for BIDS compatibility. Text formats carry
        the data only — HDF5 additionally preserves ``sampling_freq``,
        ``.convolved``, ``.confounds``, ``.multi``, and the row count of a
        column-less matrix, so ``DesignMatrix(path)`` restores the object.
    """
    from pathlib import Path

    from nltools.io import is_h5_path

    if isinstance(file_name, Path):
        file_name = str(file_name)

    if is_h5_path(file_name):
        write_h5(dm, file_name)
    else:
        # Write as delimited text file. The separator follows the extension by
        # default so `write` and the file constructor cannot disagree.
        dm.data.write_csv(
            file_name, separator=separator_for_path(file_name) if sep is None else sep
        )


def write_h5(dm: DesignMatrix, file_name: str) -> None:
    """Write DesignMatrix to HDF5 file with metadata.

    The frame is stored as Arrow IPC bytes (via the shared
    `nltools.io.h5` helpers) so every dtype round-trips exactly — an integer
    spike indicator comes back an integer rather than being floated by a
    detour through a homogeneous numpy array.

    Args:
        dm (DesignMatrix): DesignMatrix instance.
        file_name (str): Output HDF5 file path.
    """
    import h5py

    from nltools.io.h5 import _write_polars_frame

    with h5py.File(file_name, "w") as f:
        _write_polars_frame(f, "data", dm.data, "gzip")

        meta = f.create_group("metadata")
        if dm.sampling_freq is not None:
            meta.attrs["sampling_freq"] = dm.sampling_freq
        meta.attrs["convolved"] = np.array(dm.convolved, dtype="S")
        meta.attrs["confounds"] = np.array(dm.confounds, dtype="S")
        meta.attrs["multi"] = dm.multi
        # A column-less matrix still describes a specific number of
        # timepoints, and polars cannot carry that in the frame itself.
        if dm._n_rows is not None:
            meta.attrs["n_rows"] = dm._n_rows
        meta.attrs["obj_type"] = "design_matrix"


# Pre-`.nl_` spellings of the column names nltools generated before commit
# 604073fb reserved the namespace: `poly_0` / `cosine_1` (add_poly /
# add_dct_basis), `global_spike1` / `diff_spike1` (find_spikes), and the
# run-separated `{run}_{base}` variants a multi-run append produced. Only these
# exact shapes translate — everything else in a legacy file is a user column.
_LEGACY_GENERATED_RE = re.compile(
    r"(?:(?P<run>\d+)_)?(?P<base>(?:poly|cosine)_\d+|(?:global|diff)_spike\d+)"
)


def _legacy_generated_to_reserved(name: str) -> str:
    """Translate a pre-0.6 generated column name into the `.nl_` namespace.

    Applied ONLY to the legacy h5 layout in `read_h5`, where the names are
    provably machine-generated (the old writer produced them). Every other
    ingestion path deliberately treats `poly_0` as a user column (see
    `nltools.utils.RESERVED_PREFIX`), so recognition stays keyed on the
    reserved prefix alone.
    """
    from nltools.utils import is_reserved_name, reserved_name, run_separated_name

    if is_reserved_name(name):
        return name
    match = _LEGACY_GENERATED_RE.fullmatch(name)
    if match is None:
        return name
    if match["run"] is not None:
        return run_separated_name(int(match["run"]), match["base"])
    return reserved_name(match["base"])


def read_h5(file_name: str | Path) -> tuple[pl.DataFrame, dict]:
    """Read a DesignMatrix HDF5 file written by `write_h5`.

    Handles both on-disk layouts: the current one (frame as Arrow IPC bytes)
    and the pre-reader one written by nltools <= 0.6.0 (a plain float matrix
    in ``data`` beside an ``S``-typed ``columns`` dataset). Legacy files may
    also carry pre-`.nl_` generated column names (``poly_0``, ``0_poly_0``,
    ``cosine_1``); those are translated into the reserved namespace at load
    time so downstream recognition stays keyed on the prefix alone.

    Args:
        file_name (str | Path): Path to the HDF5 file.

    Returns:
        tuple[pl.DataFrame, dict]: `(frame, metadata)`, where metadata holds
            ``sampling_freq``, ``convolved``, ``confounds``, ``multi``, and
            ``n_rows`` — absent keys meaning the file didn't record them.
    """
    import h5py

    from nltools.io.h5 import _read_polars_frame

    def _decode(values) -> list[str]:
        return [v.decode() if isinstance(v, bytes) else str(v) for v in values]

    with h5py.File(file_name, "r") as f:
        legacy = "columns" in f
        if legacy:
            # Legacy layout: homogeneous matrix + separate column names.
            values = np.asarray(f["data"])
            columns = [
                _legacy_generated_to_reserved(c)
                for c in _decode(np.asarray(f["columns"]))
            ]
            data = pl.DataFrame(values, schema=columns)
        else:
            data = _read_polars_frame(f, "data")

        metadata: dict = {}
        if "metadata" in f:
            attrs = f["metadata"].attrs
            if "sampling_freq" in attrs:
                metadata["sampling_freq"] = float(attrs["sampling_freq"])
            if "convolved" in attrs:
                metadata["convolved"] = _decode(attrs["convolved"])
            if "confounds" in attrs:
                metadata["confounds"] = _decode(attrs["confounds"])
            if "multi" in attrs:
                metadata["multi"] = bool(attrs["multi"])
            if "n_rows" in attrs:
                metadata["n_rows"] = int(attrs["n_rows"])

        if legacy:
            # The legacy metadata lists name the same old spellings.
            for key in ("convolved", "confounds"):
                if key in metadata:
                    metadata[key] = [
                        _legacy_generated_to_reserved(c) for c in metadata[key]
                    ]

    return data, metadata
