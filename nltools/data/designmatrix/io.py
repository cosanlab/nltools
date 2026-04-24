"""
DesignMatrix I/O and visualization functions.

Standalone functions extracted from DesignMatrix methods.
Each takes a DesignMatrix instance (`dm`) as its first argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nltools.data.designmatrix import DesignMatrix


def plot_designmatrix(
    dm: DesignMatrix,
    figsize: tuple = (8, 6),
    *,
    rescale: bool = True,
    **kwargs,
):
    """
    Visualize design matrix as heatmap (SPM-style).

    Creates a heatmap visualization of the design matrix columns.
    Uses seaborn + matplotlib under the hood.

    Args:
        dm: DesignMatrix instance.
        figsize (tuple): Figure size (width, height) in inches. Default: (8, 6).
        rescale (bool): If True, rescale each column by its L2 norm so columns
            with different native magnitudes are visually comparable (matches
            SPM/nilearn convention). Default: True.
        **kwargs: Additional keyword arguments passed to seaborn.heatmap().

    Returns:
        matplotlib.axes.Axes: The axes object containing the heatmap.

    Examples:
        >>> dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        >>> plot_designmatrix(dm)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert to pandas for seaborn (which expects pandas DataFrames)
    df_for_plot = to_pandas(dm)

    if rescale:
        import pandas as pd

        X = df_for_plot.to_numpy(dtype=float)
        X = X / np.maximum(1.0e-12, np.sqrt(np.sum(X**2, 0)))
        df_for_plot = pd.DataFrame(X, columns=df_for_plot.columns)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set default heatmap parameters
    heatmap_kwargs = {
        "cmap": "gray",
        "cbar": False,
        "cbar_kws": {"label": "Value"},
        "yticklabels": False,  # Too many rows for labels typically
    }
    heatmap_kwargs.update(kwargs)

    # Create heatmap
    sns.heatmap(df_for_plot, ax=ax, **heatmap_kwargs)

    # Set labels
    ax.set_xlabel("Regressors")
    ax.set_ylabel("Time (TRs)")

    return ax


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
    """
    Convert DesignMatrix to numpy array.

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
        HDF5 format preserves metadata (sampling_freq, convolved, polys).
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
        meta.attrs["polys"] = np.array(dm.polys, dtype="S")
        meta.attrs["multi"] = dm.multi
        meta.attrs["obj_type"] = "design_matrix"
