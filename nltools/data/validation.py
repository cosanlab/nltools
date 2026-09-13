"""Frame validation shared by the data classes.

`BrainData.X`/`.Y` and `Adjacency.Y` accept the same range of tabular inputs
and store the same polars frame, so the ingress check lives here rather than
inside either class.
"""

from pathlib import Path

import numpy as np
import polars as pl


def _validate_frame(frame, data_shape=None, frame_type="DataFrame"):
    """Validate and process an X or Y frame for a data class.

    Accepts pandas DataFrames for user convenience but always returns a
    polars DataFrame. Internal data-class state is polars-only.

    Args:
        frame (pl.DataFrame | pd.DataFrame | dict | np.ndarray | str | Path | None):
            Input to validate: ``None``, a path to a CSV, a polars or pandas
            DataFrame, a dict of columns, or a 1D/2D numpy array.
        data_shape (tuple | None): Data shape to validate the row count against.
        frame_type (str): Name of the frame for error messages (e.g. ``"X"``,
            ``"Y"``).

    Returns:
        pl.DataFrame: Validated frame as polars. Empty ``pl.DataFrame()`` when
            ``frame`` is ``None``.

    Raises:
        TypeError: If frame is not a supported type.
        ValueError: If frame rows do not match ``data_shape[0]`` or CSV read fails.
    """
    if frame is None:
        return pl.DataFrame()

    # Unwrap DesignMatrix to its underlying polars DataFrame — DM-specific
    # metadata (sampling_freq, convolved, confounds) isn't preserved on
    # BrainData, but users should be able to hand a DM in directly.
    from nltools.data.designmatrix import DesignMatrix

    if isinstance(frame, DesignMatrix):
        frame = frame.data

    if isinstance(frame, pl.DataFrame):
        out = frame
    elif isinstance(frame, (str, Path)):
        try:
            out = pl.read_csv(frame, has_header=False)
        except Exception as e:
            raise ValueError(
                f"Could not read {frame_type} from file '{frame}'. "
                f"Make sure the file exists and is a valid CSV. Error: {e}"
            )
    elif isinstance(frame, dict):
        out = pl.DataFrame(frame)
    elif isinstance(frame, np.ndarray):
        arr = frame if frame.ndim == 2 else frame.reshape(-1, 1)
        out = pl.DataFrame(arr)
    else:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(frame, pd.DataFrame):
            out = pl.DataFrame({str(c): frame[c].to_numpy() for c in frame.columns})
        else:
            raise TypeError(
                f"{frame_type} must be a filepath (str/Path), numpy array, dict, or "
                f"polars/pandas DataFrame. Received {type(frame).__name__}"
            )

    if not out.is_empty() and data_shape is not None:
        if out.shape[0] != data_shape[0]:
            raise ValueError(
                f"{frame_type} rows ({out.shape[0]}) do not match "
                f"data rows ({data_shape[0]}). Each row in {frame_type} should "
                f"correspond to an image in the data."
            )

    return out
