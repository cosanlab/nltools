"""
DesignMatrix - Polars-based design matrix for neuroimaging analysis

Efficient design matrix implementation using Polars for fast DataFrame operations.
Provides HRF convolution, resampling, polynomial regressors, and diagnostic tools.

Uses composition pattern (wrapping pl.DataFrame) for clean metadata preservation.
"""

from __future__ import annotations

__all__ = ["DesignMatrix"]

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .utils import copy_with, df_passthrough

if TYPE_CHECKING:
    import pandas as pd


def _is_pandas_dataframe(obj) -> bool:
    """Duck-type check for pandas DataFrame without importing pandas."""
    cls = type(obj)
    module = cls.__module__
    return cls.__name__ == "DataFrame" and (
        module == "pandas" or module.startswith("pandas.")
    )


class DesignMatrix:
    """
    Polars-based design matrix for experimental designs in neuroimaging.

    Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
    Uses composition pattern (not subclassing) for clean metadata preservation.

    Args:
        data (DataFrame, ndarray, dict, str/Path, or None): Input data. Accepts:
            - Polars DataFrame (zero-copy)
            - pandas DataFrame (converted to Polars)
            - numpy ndarray
            - dict (keys=columns, values=data)
            - str or Path to a `.tsv`/`.csv` file. BIDS events files
              (containing `onset` and `duration` columns) are converted to
              boxcar regressors — call ``convolve()`` afterwards if you want
              HRF convolution. Any other tabular file is read as-is and is
              typically used for confounds.
            - None (empty initialization)
        sampling_freq (float, optional): Sampling frequency in Hz (1/TR for fMRI data).
            Mutually exclusive with ``TR``.
        TR (float, optional): Repetition time in seconds. Convenience for
            ``sampling_freq = 1/TR``. Mutually exclusive with ``sampling_freq``.
        run_length (int or 'infer', optional): Required when ``data`` is a
            file path. Number of TRs in the run. Pass ``'infer'`` for
            tabular/confounds files to accept whatever row count the file
            has (not valid for events files).
        columns (list of str, optional): Column names (used with ndarray input)
        convolved (list of str, optional): Names of convolved columns (tracked internally)
        confounds (list of str, optional): Names of nuisance/confound columns
            (intercept, polynomial drift, DCT cosines, motion, …) tracked internally

    Attributes:
        sampling_freq (float or None): Sampling frequency in Hz
        convolved (list of str): Columns that have been convolved
        confounds (list of str): Nuisance/confound columns (intercept, polynomial
            trends, DCT bases, motion, physio, …) — these are skipped by
            ``.convolve()`` and kept separate per run on multi-run vertical append.
        multi (bool): True if created from multi-run concatenation

    Examples:
        >>> # Create from numpy array
        >>> dm = DesignMatrix(np.zeros((100, 2)), sampling_freq=0.5, columns=['a', 'b'])

        >>> # Add columns
        >>> dm['stim'] = [0, 1, 1, 0] * 25

        >>> # Convolve with HRF — convolved columns get a `_c0` suffix
        >>> dm_conv = dm.convolve()  # 'stim' → 'stim_c0'

        >>> # Add polynomial drift terms
        >>> dm_conv = dm_conv.add_poly(order=2)

        >>> # Multi-run concatenation (auto-separates polynomials)
        >>> dm_run1 = DesignMatrix(...).add_poly(0)
        >>> dm_run2 = DesignMatrix(...).add_poly(0)
        >>> dm_multi = dm_run1.append(dm_run2, axis=0)  # Creates 0_poly_0, 1_poly_0
    """

    _metadata = ["sampling_freq", "convolved", "confounds", "multi"]

    def __init__(
        self,
        data: pl.DataFrame
        | pd.DataFrame
        | np.ndarray
        | dict
        | str
        | Path
        | None = None,
        *,
        sampling_freq: float | None = None,
        TR: float | None = None,
        run_length: int | str | None = None,
        columns: list[str] | None = None,
        convolved: list[str] | None = None,
        confounds: list[str] | None = None,
    ):
        """Initialize DesignMatrix from various input types."""
        if TR is not None and sampling_freq is not None:
            raise ValueError("Pass exactly one of `TR` or `sampling_freq`, not both.")
        if TR is not None:
            sampling_freq = 1.0 / TR

        # Initialize metadata
        self.sampling_freq = sampling_freq
        self.convolved = convolved if convolved is not None else []
        self.confounds = confounds if confounds is not None else []
        self.multi = False

        # Create internal Polars DataFrame based on input type
        if data is None:
            # Empty initialization
            self.data = pl.DataFrame()

        elif isinstance(data, (str, Path)):
            if run_length is None:
                raise ValueError(
                    "Loading DesignMatrix from a file requires `run_length`."
                )
            if sampling_freq is None:
                raise ValueError(
                    "Loading DesignMatrix from a file requires `TR` or `sampling_freq`."
                )
            from .io import load_from_file

            self.data, _is_events = load_from_file(
                data,
                run_length=run_length,
                sampling_freq=sampling_freq,
            )

        elif isinstance(data, pl.DataFrame):
            # Polars DataFrame - zero copy, just ensure string column names
            self.data = data.rename({col: str(col) for col in data.columns})

        elif isinstance(data, dict):
            # Dictionary - let Polars handle it, ensure string column names
            self.data = pl.DataFrame(data)
            self.data = self.data.rename({col: str(col) for col in self.data.columns})

        elif isinstance(data, np.ndarray):
            # Numpy array - handle column names
            if columns is not None:
                # Use provided column names
                self.data = pl.DataFrame(data, schema=[str(c) for c in columns])
            else:
                # Auto-generate column names as strings: '0', '1', '2', ...
                n_cols = data.shape[1] if data.ndim > 1 else 1
                auto_columns = [str(i) for i in range(n_cols)]
                self.data = pl.DataFrame(data, schema=auto_columns)

        elif _is_pandas_dataframe(data):
            # pandas DataFrame - convert to Polars, ensure string column names
            self.data = pl.from_pandas(data)
            self.data = self.data.rename({col: str(col) for col in self.data.columns})

        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Expected Polars/pandas DataFrame, numpy array, dict, or None."
            )

    # ── Dunders (alphabetical) ──────────────────────────────────────────

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Numpy array interface - enables np.array(design_matrix) and np.asarray().

        Args:
            dtype (numpy dtype, optional): Desired data type for the array

        Returns:
            np.ndarray: 2D numpy array representation
        """
        arr = self.data.to_numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __dir__(self):
        """Include polars DataFrame attrs for REPL/IDE autocomplete."""
        return sorted(set(super().__dir__()) | set(dir(self.data)))

    def __eq__(self, other) -> bool:
        """
        Check equality with another DesignMatrix.

        Compares data frames only (ignores metadata like sampling_freq, convolved,
        confounds, and multi).

        Args:
            other (DesignMatrix): Design matrix to compare with

        Returns:
            bool: True if data frames are equal (same shape, column names, and values)
        """
        if not isinstance(other, DesignMatrix):
            return NotImplemented
        return self.data.equals(other.data)

    def __getattr__(self, name: str):
        """Forward unknown attrs to the underlying polars DataFrame.

        Allowlisted row-preserving methods (see ``WRAP_AS_DESIGNMATRIX`` in utils)
        return a new ``DesignMatrix`` with metadata preserved; everything else
        returns the raw polars attribute. The ``data`` guard avoids recursion
        during construction before ``data`` is assigned.
        """
        if name.startswith("_") or "data" not in self.__dict__:
            raise AttributeError(name)
        try:
            return df_passthrough(self, name)
        except AttributeError:
            raise AttributeError(
                f"'DesignMatrix' object has no attribute {name!r}"
            ) from None

    def __getitem__(self, key: str | list[str]) -> pl.Series | DesignMatrix:
        """
        Access columns.

        dm['col'] returns Series
        dm[['col1', 'col2']] returns DesignMatrix
        """
        if isinstance(key, str):
            # Single column - return Series
            return self.data[key]
        if isinstance(key, list):
            # Multiple columns - return DesignMatrix with metadata
            subset_df = self.data.select(key)
            return copy_with(self, subset_df)
        raise TypeError(f"Column key must be str or list of str, got {type(key)}")

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.data)

    def __repr__(self) -> str:
        """Human-readable metadata summary."""
        lines = [
            f"DesignMatrix(sampling_freq={self.sampling_freq}, shape={self.shape})"
        ]
        if self.convolved:
            lines.append(f"  convolved ({len(self.convolved)}): {self.convolved}")
        if self.confounds:
            lines.append(f"  confounds ({len(self.confounds)}): {self.confounds}")
        return "\n".join(lines)

    def __setitem__(self, key: str, value: int | float | list | np.ndarray | pl.Series):
        """
        Set column values.

        dm['col'] = 0  # Broadcast scalar
        dm['col'] = [1, 2, 3]  # Array assignment
        """
        if isinstance(value, (int, float)):
            self.data = self.data.with_columns(pl.lit(value).alias(key))
        elif isinstance(value, (list, np.ndarray)):
            self.data = self.data.with_columns(pl.Series(key, value))
        elif isinstance(value, pl.Series):
            self.data = self.data.with_columns(value.alias(key))
        else:
            raise TypeError(f"Cannot set column from type {type(value)}")

    # ── Properties (alphabetical) ───────────────────────────────────────

    @property
    def columns(self) -> list[str]:
        """Column names of the design matrix as a list of strings."""
        return self.data.columns

    @columns.setter
    def columns(self, new_names: list[str]):
        """Set column names."""
        str_names = [str(name) for name in new_names]
        self.data = self.data.rename(dict(zip(self.data.columns, str_names)))

    @property
    def is_empty(self) -> bool:
        """Check if DesignMatrix has no data.

        Returns:
            bool: True if the design matrix is empty, False otherwise.
        """
        return self.data.is_empty()

    @property
    def shape(self) -> tuple:
        """Return (n_rows, n_cols) tuple."""
        return self.data.shape

    # ── Public methods (alphabetical) ───────────────────────────────────

    def add_dct_basis(
        self,
        duration: float = 180,
        drop: int = 0,
        *,
        include_constant: bool = True,
    ) -> DesignMatrix:
        """
        Add discrete cosine transform basis functions (high-pass filter).

        Args:
            duration (float): Filter duration in seconds. Default: 180.
            drop (int): Number of low-frequency bases to drop. Default: 0.
            include_constant (bool): If True, also add a constant/intercept
                column named ``cosine_0`` (analogous to ``poly_0`` in
                :meth:`add_poly`). The underlying DCT basis drops the constant
                per SPM convention; set False to match SPM behavior.
                Default: True.

        Returns:
            DesignMatrix: New DesignMatrix with DCT basis columns appended.
        """
        from .regressors import add_dct_basis

        return add_dct_basis(self, duration, drop, include_constant=include_constant)

    def add_poly(self, order: int = 0, include_lower: bool = True) -> DesignMatrix:
        """
        Add Legendre polynomial drift terms.

        Args:
            order (int): Polynomial order (0=intercept, 1=linear, 2=quadratic, ...).
                Default: 0.
            include_lower (bool): If True, include all orders from 0 to order.
                Default: True.

        Returns:
            DesignMatrix: New DesignMatrix with polynomial columns appended.
        """
        from .regressors import add_poly

        return add_poly(self, order, include_lower)

    def append(
        self,
        dm: DesignMatrix | list[DesignMatrix],
        *,
        axis: int = 0,
        keep_separate: bool = True,
        unique_cols: list[str] | None = None,
        fill_na: int | float = 0,
        as_confounds: bool = False,
        verbose: bool = False,
    ) -> DesignMatrix:
        """
        Concatenate design matrices.

        Args:
            dm (DesignMatrix or list of DesignMatrix): Design matrix/matrices to append.
            axis (int): 0 for row-wise (vertical), 1 for column-wise (horizontal).
                Default: 0.
            keep_separate (bool): Whether to separate confound columns across runs
                (only applies when axis=0). Default: True.
            unique_cols (list of str, optional): Additional columns to keep separated
                (supports wildcards).
            fill_na (int or float): Value to fill NaN values during vertical
                concatenation. Default: 0.
            as_confounds (bool): Only applies when ``axis=1``. If True, mark all
                columns from ``dm`` as nuisance/confounds in the result — they
                get skipped by ``.convolve()`` and separated across runs on
                later vertical appends. Default: False.
            verbose (bool): Print messages about confound separation. Default: False.

        Returns:
            DesignMatrix: Concatenated design matrix.
        """
        from .append import append

        return append(
            self,
            dm,
            axis=axis,
            keep_separate=keep_separate,
            unique_cols=unique_cols,
            fill_na=fill_na,
            as_confounds=as_confounds,
            verbose=verbose,
        )

    def clean(
        self,
        fill_na: int | float | None = 0,
        exclude_confounds: bool = False,
        thresh: float = 0.95,
        verbose: bool = True,
    ) -> DesignMatrix:
        """
        Remove highly correlated columns.

        Args:
            fill_na (int, float, or None): Fill NaN values before checking correlations (default 0)
            exclude_confounds (bool): Skip confound/nuisance columns from correlation check
            thresh (float): Correlation threshold (drop if abs(r) >= thresh, default 0.95)
            verbose (bool): Print dropped column names

        Returns:
            DesignMatrix: Cleaned matrix with highly correlated columns removed
        """
        from .diagnostics import clean

        return clean(self, fill_na, exclude_confounds, thresh, verbose)

    def convolve(
        self,
        conv_func: str | np.ndarray = "hrf",
        columns: list[str] | None = None,
    ) -> DesignMatrix:
        """
        Convolve columns with HRF or custom kernel.

        Convolved columns are always renamed to ``<col>_c{i}`` (where ``i`` is
        the kernel index, ``0`` for a single 1-D kernel). The source columns
        are dropped, and ``self.convolved`` lists the post-suffix names so
        downstream metadata stays in sync with the dataframe.

        Args:
            conv_func (str or ndarray): 'hrf' for canonical Glover HRF, or custom kernel(s).
                Can be 1D array (single kernel) or 2D (samples x kernels).
            columns (list of str, optional): Columns to convolve (default: all non-confound columns).

        Returns:
            DesignMatrix: New DesignMatrix with convolved columns renamed.
        """
        from .regressors import convolve

        return convolve(self, conv_func, columns)

    def copy(self) -> DesignMatrix:
        """
        Create a deep copy of the DesignMatrix.

        Returns:
            DesignMatrix: Copy of the current DesignMatrix
        """
        cloned_df = self.data.clone()
        return copy_with(self, cloned_df)

    def downsample(self, target: float, method: str = "mean", **kwargs) -> DesignMatrix:
        """
        Reduce temporal resolution to target frequency using Polars-native operations.

        Args:
            target (float): Target sampling frequency in Hz (must be < current sampling_freq)
            method (str): Aggregation method - 'mean' or 'median' (default: 'mean')

        Returns:
            DesignMatrix: Downsampled DesignMatrix with updated sampling_freq
        """
        from .transforms import downsample

        return downsample(self, target, method=method, **kwargs)

    def drop(self, columns: list[str]) -> DesignMatrix:
        """Drop specified columns.

        Args:
            columns (list of str): Column names to remove.

        Returns:
            DesignMatrix: New DesignMatrix without the specified columns.
        """
        dropped_df = self.data.drop(columns)
        return copy_with(self, dropped_df)

    def fillna(self, value: int | float) -> DesignMatrix:
        """Fill NaN/null values with specified value.

        Args:
            value (int or float): Value to replace NaN/null entries with.

        Returns:
            DesignMatrix: New DesignMatrix with NaN/null values replaced.
        """
        filled_df = self.data.fill_null(value).fill_nan(value)
        return copy_with(self, filled_df)

    def plot(self, figsize: tuple = (4, 6), *, rescale: bool = True, **kwargs):
        """
        Visualize design matrix as heatmap (SPM-style).

        Args:
            figsize (tuple, default=(8, 6)): Figure size (width, height) in inches
            rescale (bool, default=True): If True, rescale each column by its
                L2 norm so columns with different native magnitudes are visually
                comparable (matches SPM/nilearn convention). Set False to plot
                raw values.
            **kwargs: Additional keyword arguments passed to seaborn.heatmap()

        Returns:
            matplotlib.figure.Figure: The figure containing the heatmap.
        """
        from .io import plot_designmatrix

        return plot_designmatrix(self, figsize, rescale=rescale, **kwargs)

    def replace_data(
        self,
        data: np.ndarray,
        column_names: list[str] | None = None,
    ) -> DesignMatrix:
        """
        Replace data columns while preserving confounds and metadata.

        Args:
            data (ndarray): New data array (must match number of rows in current DesignMatrix)
            column_names (list of str, optional): Names for new data columns.

        Returns:
            DesignMatrix: New DesignMatrix with replaced data columns, preserved confounds

        Raises:
            ValueError: If row count doesn't match existing data
        """
        if data.shape[0] != self.shape[0]:
            raise ValueError(
                f"Row count mismatch: new data has {data.shape[0]} rows, "
                f"but DesignMatrix has {self.shape[0]} rows"
            )

        if column_names is None:
            n_cols = data.shape[1] if data.ndim > 1 else 1
            column_names = [f"col_{i}" for i in range(n_cols)]

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        new_data_df = pl.DataFrame(data, schema=column_names, orient="row")

        confound_df = (
            self.data.select(self.confounds) if self.confounds else pl.DataFrame()
        )

        if confound_df.shape[1] > 0:
            combined_df = pl.concat([new_data_df, confound_df], how="horizontal")
        else:
            combined_df = new_data_df

        return copy_with(self, combined_df)

    def standardize(
        self, method: str = "zscore", columns: list[str] | None = None
    ) -> DesignMatrix:
        """Standardize columns using the specified method.

        Args:
            method: Standardization method ('zscore' or 'center'). Default: 'zscore'.
            columns: Columns to standardize. If None, standardize all
                non-polynomial columns.

        Returns:
            DesignMatrix: New DesignMatrix with standardized columns.
        """
        from .transforms import standardize

        return standardize(self, columns, method)

    def sum(self, axis: int = 0) -> pl.Series:
        """
        Compute sum along axis.

        Args:
            axis (int, default=0): 0: sum down columns, 1: sum across rows.

        Returns:
            pl.Series: Sums along specified axis.
        """
        if axis == 0:
            sums = [self.data[col].sum() for col in self.data.columns]
            return pl.Series(values=sums, name="")
        if axis == 1:
            return self.data.select(pl.sum_horizontal(pl.all())).to_series()
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    def to_numpy(self) -> np.ndarray:
        """
        Convert DesignMatrix to numpy array.

        Returns:
            np.ndarray: 2D array with shape (n_samples, n_columns)
        """
        from .io import to_numpy

        return to_numpy(self)

    def to_pandas(self) -> pd.DataFrame:
        """Convert DesignMatrix to pandas DataFrame.

        Returns:
            pd.DataFrame: Pandas DataFrame with same data and column names.
        """
        from .io import to_pandas

        return to_pandas(self)

    def upsample(self, target: float, method: str = "linear", **kwargs) -> DesignMatrix:
        """
        Increase temporal resolution to target frequency.

        Args:
            target (float): Target sampling frequency in Hz (must be > current sampling_freq)
            method (str): Interpolation method - 'linear' or 'nearest' (default: 'linear')

        Returns:
            DesignMatrix: Upsampled DesignMatrix with updated sampling_freq
        """
        from .transforms import upsample

        return upsample(self, target, method, **kwargs)

    def vif(self, exclude_confounds: bool = True) -> np.ndarray | None:
        """
        Compute variance inflation factor for each column.

        Args:
            exclude_confounds (bool): Skip confound/nuisance columns. Default: True.

        Returns:
            np.ndarray: VIF values for each included column. Returns None if the
                correlation matrix is singular.
        """
        from .diagnostics import vif

        return vif(self, exclude_confounds)

    def write(self, file_name: str, sep: str = "\t") -> None:
        """Write DesignMatrix to file.

        Supports TSV (default), CSV, and HDF5 formats. Format is
        auto-detected from file extension.

        Args:
            file_name: Output file path. Use .tsv, .csv, or .h5/.hdf5 extension.
            sep: Column separator for text files (default: tab).
        """
        from .io import write

        return write(self, file_name, sep)

    def zscore(self, columns: list[str] | None = None) -> DesignMatrix:
        """
        Z-score standardize columns (mean=0, std=1).

        Args:
            columns (list of str, optional): Columns to standardize. If None,
                standardize all non-polynomial columns.

        Returns:
            DesignMatrix: New DesignMatrix with standardized columns
        """
        from .transforms import zscore

        return zscore(self, columns)
