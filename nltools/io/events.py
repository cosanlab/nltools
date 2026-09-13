"""Convert BIDS events tables into design-matrix regressors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import pandas as pd


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

    Examples:
        ```python
        import polars as pl
        from nltools.io import events_to_dm

        events = pl.DataFrame(
            {"onset": [0.0, 10.0], "duration": [5.0, 5.0], "trial_type": ["a", "b"]}
        )
        regressors = events_to_dm(events, run_length=20, sampling_freq=0.5)
        ```
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
    # Avoid pyarrow dep on the pandas → polars hop.
    return pl.DataFrame({str(c): dm[c].to_numpy() for c in dm.columns})
