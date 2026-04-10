"""
NeuroLearn File Reading Tools
=============================

"""

from __future__ import annotations

__all__ = ["onsets_to_dm"]

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix as make_dm

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

from nltools.algorithms import glover_hrf
from nltools.data import DesignMatrix


def onsets_to_dm(
    timings: str
    | Path
    | pd.DataFrame
    | pl.DataFrame
    | list[str | Path | pd.DataFrame | pl.DataFrame],
    run_length: int | list[int],
    TR: float,
    hrf_model: str | Callable | None = "glover",
    drift_model: str | None = None,
    high_pass: float = 0.01,
    drift_order: int = 0,
    fill_na: Any = None,
    **kwargs: Any,
) -> DesignMatrix | list[DesignMatrix]:
    """Read 1 or more file paths and return 1 or more design matrices.

    Your timing file needs have the following column names:

    - 'onset': required
    - 'duration': required
    - 'trial_type': optional
    - 'modulation': optional

    This function is a wrapper around [`nilearn.glm.first_level.make_first_level_design_matrix`](https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.make_first_level_design_matrix.html#nilearn.glm.first_level.make_first_level_design_matrix) which is more robust that older implementations.

    However, the default options are **different** and create a design matrix with minimal additional modifications. You can use kwargs to control settings to also convolve predictors with a variety of HRF functions, add nuisance parameters, drift and cosine functions, etc.

    Args:
        timings (str, Path, pd.DataFrame, list): file(s) or dataframe(s) containing stimulus timing
        run_length (int, list): number or list of numbers for run lengths in TRs
        TR (float, optional): repetition time in seconds. Defaults to None.
        hrf_model (str, optional): convolve each column of the design matrix (e.g. 'glover'). Defaults to None.
        drift_model (str, optional): how to add drift ('cosine' or 'polynomial'). Defaults to None.
        high_pass (float, optional): high-pass frequency if drift_model='cosine'. Defaults to 0.01
        drift_order (int, optional): what order if drift_model='polynomial'. Defaults to 0.
        fill_na (Any, optional): value to fill NaN entries with. Defaults to None.

    Returns:
        DesignMatrix | list[DesignMatrix]: Single DesignMatrix if one timing file provided,
            or list of DesignMatrices if multiple timing files provided.
    """
    if not isinstance(timings, list):
        timings = [timings]
    if not isinstance(run_length, list):
        run_length = [run_length]
    if len(timings) != len(run_length):
        raise ValueError("timings and run_length must have the same length")

    # Nilearn auto-calculates approximate TR from diff-ing timings
    # when passing a string name to hrf_model
    # This approach gives us more control using the TR kwarg
    if TR is not None:
        if hrf_model == "glover":
            hrf_model = lambda arg1, oversampling: glover_hrf(TR, oversampling)

    import polars as pl

    out = []
    for file, run in zip(timings, run_length):
        if isinstance(file, pl.DataFrame):
            import pandas as pd

            file = pd.DataFrame(file.to_dict(as_series=False))
        frame_times = np.arange(run) * TR
        dm = make_dm(
            frame_times,
            events=file,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass,
            drift_order=drift_order,
            **kwargs,
        )
        dm = dm.fill_na(fill_na) if fill_na is not None else dm
        if isinstance(hrf_model, Callable):
            dm.columns = [c.rstrip("_<lambda>") for c in dm.columns]
        if hrf_model is not None:
            convolved = [
                c for c in dm.columns if "drift" not in c and "constant" not in c
            ]
            polys = [c for c in dm.columns if "drift" in c or "constant" in c]
        else:
            convolved, polys = [], []
        dm = DesignMatrix(
            dm, convolved=convolved, sampling_freq=1 / TR, polys=polys
        ).reset_index(drop=True)
        out.append(dm)

    return out if len(out) > 1 else out[0]
