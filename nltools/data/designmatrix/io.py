"""Provide DesignMatrix I/O and visualization functions.

Standalone functions extracted from DesignMatrix methods.
Each takes a DesignMatrix instance (`dm`) as its first argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

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
        events: pandas or polars DataFrame with BIDS columns `onset`,
            `duration`, `trial_type` (required); `modulation` is passed
            through if present.
        run_length: Number of TRs the run contains.
        sampling_freq: Sampling frequency in Hz (= 1/TR).

    Returns:
        pl.DataFrame with one column per unique `trial_type`, values in
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


def load_from_file(
    path: str | Path,
    *,
    run_length: int | str,
    sampling_freq: float,
) -> tuple[pl.DataFrame, bool]:
    """Read a TSV/CSV into the frame a DesignMatrix wraps.

    Dispatches on column inspection:

    - `onset` and `duration` both present → BIDS events → boxcar DM via
      `events_to_dm` (unconvolved; caller convolves later).
    - otherwise → tabular file (confounds / nuisance regressors) read as-is.

    `run_length='infer'` is accepted only for the tabular path; events
    files must provide an explicit integer (they have a variable row count
    per run, unlike confounds which are 1 row per TR).

    Args:
        path: Path to a `.tsv` or `.csv` file.
        run_length: Number of TRs, or `'infer'` for tabular inputs.
        sampling_freq: Sampling frequency in Hz (= 1/TR).

    Returns:
        Tuple of (data frame, is_events) — `is_events` signals to the
        caller that the columns are experimental regressors rather than
        nuisance.
    """
    p = Path(path)
    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    raw = pl.read_csv(
        p,
        separator=sep,
        null_values=["n/a", "N/A", "NA", ""],
        infer_schema_length=10_000,
    )

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
        dm: DesignMatrix instance.

    Returns:
        pd.DataFrame: Pandas DataFrame with same data and column names.

    Examples:
        >>> dm = DesignMatrix(np.random.randn(100, 3))
        >>> pd_df = to_pandas(dm)
        >>> type(pd_df)
        <class 'pandas.core.frame.DataFrame'>
    """
    import pandas as pd

    return pd.DataFrame(dm.data.to_dict(as_series=False))


def to_numpy(dm: DesignMatrix) -> np.ndarray:
    """Convert a DesignMatrix to a NumPy array.

    Returns data columns as 2D numpy array (rows x columns).
    Column order is preserved from DataFrame.

    Args:
        dm: DesignMatrix instance.

    Returns:
        np.ndarray: 2D array with shape (n_samples, n_columns)

    Examples:
        >>> dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)
        >>> arr = to_numpy(dm)
        >>> arr.shape
        (3, 2)
    """
    return dm.data.to_numpy()


def write(dm: DesignMatrix, file_name: str, sep: str = "\t") -> None:
    """Write DesignMatrix to file.

    Supports TSV (default), CSV, and HDF5 formats. The format is
    automatically determined by file extension.

    Args:
        dm: DesignMatrix instance.
        file_name: Output file path. Use .tsv, .csv, or .h5/.hdf5 extension.
        sep: Column separator for text files (default: tab for TSV).
             Ignored for HDF5 files.

    Returns:
        None

    Examples:
        >>> dm = DesignMatrix(np.random.randn(100, 3), sampling_freq=1)
        >>> write(dm, "design_matrix.tsv")  # TSV format (BIDS compatible)
        >>> write(dm, "design_matrix.csv", sep=",")  # CSV format
        >>> write(dm, "design_matrix.h5")  # HDF5 format

    Notes:
        TSV format is recommended for BIDS compatibility.
        HDF5 format preserves metadata (sampling_freq, convolved, confounds).
    """
    from pathlib import Path

    from nltools.io import is_h5_path

    if isinstance(file_name, Path):
        file_name = str(file_name)

    if is_h5_path(file_name):
        write_h5(dm, file_name)
    else:
        # Write as delimited text file (TSV or CSV)
        dm.data.write_csv(file_name, separator=sep)


def write_h5(dm: DesignMatrix, file_name: str) -> None:
    """Write DesignMatrix to HDF5 file with metadata.

    Args:
        dm: DesignMatrix instance.
        file_name (str): Output HDF5 file path.

    Returns:
        None
    """
    import h5py

    with h5py.File(file_name, "w") as f:
        # Store data
        f.create_dataset("data", data=dm.data.to_numpy(), compression="gzip")

        # Store column names
        f.create_dataset(
            "columns",
            data=np.array(dm.columns, dtype="S"),
            compression="gzip",
        )

        # Store metadata
        meta = f.create_group("metadata")
        if dm.sampling_freq is not None:
            meta.attrs["sampling_freq"] = dm.sampling_freq
        meta.attrs["convolved"] = np.array(dm.convolved, dtype="S")
        meta.attrs["confounds"] = np.array(dm.confounds, dtype="S")
        meta.attrs["multi"] = dm.multi
        meta.attrs["obj_type"] = "design_matrix"
