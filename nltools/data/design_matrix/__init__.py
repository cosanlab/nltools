"""
DesignMatrix - Polars-based design matrix for neuroimaging analysis

Efficient design matrix implementation using Polars for fast DataFrame operations.
Provides HRF convolution, resampling, polynomial regressors, and diagnostic tools.

Uses composition pattern (wrapping pl.DataFrame) for clean metadata preservation.
"""

__all__ = ["DesignMatrix"]

import numpy as np
import pandas as pd
import polars as pl
from typing import Union, List, Optional


class DesignMatrix:
    """
    Polars-based design matrix for experimental designs in neuroimaging.

    Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
    Uses composition pattern (not subclassing) for clean metadata preservation.

    Args:
        data (DataFrame, ndarray, dict, or None): Input data. Accepts:
            - Polars DataFrame (zero-copy)
            - pandas DataFrame (converted to Polars)
            - numpy ndarray
            - dict (keys=columns, values=data)
            - None (empty initialization)
        sampling_freq (float, optional): Sampling frequency in Hz (1/TR for fMRI data)
        columns (list of str, optional): Column names (used with ndarray input)
        convolved (list of str, optional): Names of convolved columns (tracked internally)
        polys (list of str, optional): Names of polynomial columns (tracked internally)

    Attributes:
        sampling_freq (float or None): Sampling frequency in Hz
        convolved (list of str): Columns that have been convolved
        polys (list of str): Polynomial/nuisance columns (intercept, trends, DCT bases)
        multi (bool): True if created from multi-run concatenation

    Examples:
        >>> # Create from numpy array
        >>> dm = DesignMatrix(np.zeros((100, 2)), sampling_freq=0.5, columns=['a', 'b'])

        >>> # Add columns
        >>> dm['stim'] = [0, 1, 1, 0] * 25

        >>> # Convolve with HRF
        >>> dm_conv = dm.convolve()

        >>> # Add polynomial drift terms
        >>> dm_conv = dm_conv.add_poly(order=2)

        >>> # Multi-run concatenation (auto-separates polynomials)
        >>> dm_run1 = DesignMatrix(...).add_poly(0)
        >>> dm_run2 = DesignMatrix(...).add_poly(0)
        >>> dm_multi = dm_run1.append(dm_run2, axis=0)  # Creates 0_poly_0, 1_poly_0
    """

    _metadata = ["sampling_freq", "convolved", "polys", "multi"]

    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, np.ndarray, dict, None] = None,
        *,
        sampling_freq: Optional[float] = None,
        columns: Optional[List[str]] = None,
        convolved: Optional[List[str]] = None,
        polys: Optional[List[str]] = None,
    ):
        """Initialize DesignMatrix from various input types."""
        # Initialize metadata
        self.sampling_freq = sampling_freq
        self.convolved = convolved if convolved is not None else []
        self.polys = polys if polys is not None else []
        self.multi = False

        # Create internal Polars DataFrame based on input type
        if data is None:
            # Empty initialization
            self._df = pl.DataFrame()

        elif isinstance(data, pl.DataFrame):
            # Polars DataFrame - zero copy, just ensure string column names
            self._df = data.rename({col: str(col) for col in data.columns})

        elif isinstance(data, pd.DataFrame):
            # pandas DataFrame - convert to Polars, ensure string column names
            self._df = pl.from_pandas(data)
            self._df = self._df.rename({col: str(col) for col in self._df.columns})

        elif isinstance(data, dict):
            # Dictionary - let Polars handle it, ensure string column names
            self._df = pl.DataFrame(data)
            self._df = self._df.rename({col: str(col) for col in self._df.columns})

        elif isinstance(data, np.ndarray):
            # Numpy array - handle column names
            if columns is not None:
                # Use provided column names
                self._df = pl.DataFrame(data, schema=[str(c) for c in columns])
            else:
                # Auto-generate column names as strings: '0', '1', '2', ...
                n_cols = data.shape[1] if data.ndim > 1 else 1
                auto_columns = [str(i) for i in range(n_cols)]
                self._df = pl.DataFrame(data, schema=auto_columns)

        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Expected Polars/pandas DataFrame, numpy array, dict, or None."
            )

    # ==================== Properties ====================

    @property
    def shape(self) -> tuple:
        """Return (n_rows, n_cols) tuple."""
        return self._df.shape

    @property
    def columns(self) -> List[str]:
        """Column names of the design matrix as a list of strings."""
        return self._df.columns

    @columns.setter
    def columns(self, new_names: List[str]):
        """Set column names."""
        # Ensure all names are strings
        str_names = [str(name) for name in new_names]
        self._df = self._df.rename(dict(zip(self._df.columns, str_names)))

    @property
    def is_empty(self) -> bool:
        """Check if DesignMatrix has no data.

        Returns:
            bool: True if the design matrix is empty, False otherwise.
        """
        return self._df.is_empty()

    @property
    def empty(self) -> bool:
        """Return True if DesignMatrix has no data.

        .. deprecated:: 0.6.0
            Use :attr:`is_empty` instead.
        """
        import warnings

        warnings.warn(
            "empty is deprecated and will be removed in a future version. "
            "Use is_empty instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_empty

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self._df)

    # ==================== Data Access ====================

    def __getitem__(
        self, key: Union[str, List[str]]
    ) -> Union[pl.Series, "DesignMatrix"]:
        """
        Access columns.

        dm['col'] returns Series
        dm[['col1', 'col2']] returns DesignMatrix
        """
        if isinstance(key, str):
            # Single column - return Series
            return self._df[key]
        elif isinstance(key, list):
            # Multiple columns - return DesignMatrix with metadata
            subset_df = self._df.select(key)
            return self._copy_with(subset_df)
        else:
            raise TypeError(f"Column key must be str or list of str, got {type(key)}")

    def __setitem__(
        self, key: str, value: Union[int, float, list, np.ndarray, pl.Series]
    ):
        """
        Set column values.

        dm['col'] = 0  # Broadcast scalar
        dm['col'] = [1, 2, 3]  # Array assignment
        """
        # Polars with_columns handles broadcasting and type conversion automatically
        # Use .with_columns() for immutable pattern, then reassign
        if isinstance(value, (int, float)):
            # Scalar - Polars will broadcast
            self._df = self._df.with_columns(pl.lit(value).alias(key))
        elif isinstance(value, (list, np.ndarray)):
            # Array-like - convert to Series
            self._df = self._df.with_columns(pl.Series(key, value))
        elif isinstance(value, pl.Series):
            # Polars Series - rename and add
            self._df = self._df.with_columns(value.alias(key))
        else:
            raise TypeError(f"Cannot set column from type {type(value)}")

    # ==================== Simple Transformations ====================

    def fillna(self, value: Union[int, float]) -> "DesignMatrix":
        """Fill NaN/null values with specified value."""
        # Polars distinguishes between null (None) and NaN
        # Fill both to match pandas/user expectations
        filled_df = self._df.fill_null(value).fill_nan(value)
        return self._copy_with(filled_df)

    def drop(self, columns: List[str]) -> "DesignMatrix":
        """Drop specified columns."""
        dropped_df = self._df.drop(columns)
        return self._copy_with(dropped_df)

    def copy(self) -> "DesignMatrix":
        """
        Create a deep copy of the DesignMatrix.

        Returns a new DesignMatrix with cloned data and metadata.

        Returns:
            DesignMatrix: Copy of the current DesignMatrix

        Examples:
            >>> dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)
            >>> dm_copy = dm.copy()
            >>> dm_copy["a"] = [4, 5, 6]  # Modifying copy doesn't affect original
            >>> dm["a"].to_list()
            [1, 2, 3]
        """
        # Clone the Polars DataFrame (creates deep copy)
        cloned_df = self._df.clone()
        return self._copy_with(cloned_df)

    # ==================== Statistical Operations ====================

    def zscore(self, columns: Optional[List[str]] = None) -> "DesignMatrix":
        """
        Z-score standardize columns (mean=0, std=1).

        Args:
            columns (list of str, optional): Columns to standardize. If None, standardize all non-polynomial columns.

        Returns:
            DesignMatrix: New DesignMatrix with standardized columns
        """
        from .transforms import zscore

        return zscore(self, columns)

    def standardize(
        self, method: str = "zscore", columns: Optional[List[str]] = None
    ) -> "DesignMatrix":
        """Standardize columns using the specified method.

        This method provides a consistent API with BrainData and Collection
        for data normalization.

        Args:
            method: Standardization method. Options are:
                - 'zscore': Z-score standardization (mean=0, std=1) [default]
                - 'center': Mean centering only (mean=0)
            columns: Columns to standardize. If None, standardize all
                non-polynomial columns.

        Returns:
            DesignMatrix: New DesignMatrix with standardized columns.

        Raises:
            ValueError: If an invalid method is specified.

        Examples:
            >>> dm = DesignMatrix(np.random.randn(100, 3))
            >>> dm_z = dm.standardize(method='zscore')  # z-score all columns
            >>> dm_c = dm.standardize(method='center')  # center only
        """
        from .transforms import standardize

        return standardize(self, columns, method)

    def downsample(
        self, target: float, method: str = "mean", **kwargs
    ) -> "DesignMatrix":
        """
        Reduce temporal resolution to target frequency using Polars-native operations.

        Args:
            target (float): Target sampling frequency in Hz (must be < current sampling_freq)
            method (str): Aggregation method - 'mean' or 'median' (default: 'mean')
            **kwargs: Reserved for future extensions

        Returns:
            DesignMatrix: Downsampled DesignMatrix with updated sampling_freq

        Examples:
            >>> dm = DesignMatrix({"a": list(range(100))}, sampling_freq=1.0)
            >>> dm_down = dm.downsample(target=0.5)  # 1 Hz → 0.5 Hz (100 → 50 samples)
        """
        from .transforms import downsample

        return downsample(self, target, method=method, **kwargs)

    def upsample(
        self, target: float, method: str = "linear", **kwargs
    ) -> "DesignMatrix":
        """
        Increase temporal resolution to target frequency using Polars-native interpolation.

        Args:
            target (float): Target sampling frequency in Hz (must be > current sampling_freq)
            method (str): Interpolation method - 'linear' or 'nearest' (default: 'linear')
            **kwargs: Reserved for future extensions

        Returns:
            DesignMatrix: Upsampled DesignMatrix with updated sampling_freq

        Examples:
            >>> dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)
            >>> dm_up = dm.upsample(target=2.0)  # 1 Hz → 2 Hz (10 → 19 samples)
        """
        from .transforms import upsample

        return upsample(self, target, method, **kwargs)

    # ==================== Convolution ====================

    def convolve(
        self,
        conv_func: Union[str, np.ndarray] = "hrf",
        columns: Optional[List[str]] = None,
    ) -> "DesignMatrix":
        """
        Convolve columns with HRF or custom kernel.

        Args:
            conv_func (str or ndarray): 'hrf' for canonical Glover HRF, or custom kernel(s).
                Can be 1D array (single kernel) or 2D (samples x kernels)
            columns (list of str, optional): Columns to convolve (default: all non-polynomial columns)

        Returns:
            DesignMatrix: New DesignMatrix with convolved columns

        Examples:
            >>> # Default HRF convolution
            >>> dm_conv = dm.convolve()

            >>> # Custom kernel
            >>> kernel = np.array([0.5, 1.0, 0.5])
            >>> dm_conv = dm.convolve(conv_func=kernel)

            >>> # Multiple kernels (FIR model)
            >>> kernels = np.array([[1.0, 0.5], [0.5, 1.0]]).T  # 2 kernels
            >>> dm_conv = dm.convolve(conv_func=kernels)  # Creates col_c0, col_c1
        """
        from .regressors import convolve

        return convolve(self, conv_func, columns)

    # ==================== Polynomial/Basis Functions ====================

    def add_poly(self, order: int = 0, include_lower: bool = True) -> "DesignMatrix":
        """
        Add Legendre polynomial drift terms.

        Args:
            order (int): Polynomial order (0=intercept, 1=linear, 2=quadratic, ...)
            include_lower (bool): If True, include all orders from 0 to order
        """
        from .regressors import add_poly

        return add_poly(self, order, include_lower)

    def add_dct_basis(self, duration: float = 180, drop: int = 0) -> "DesignMatrix":
        """
        Add discrete cosine transform basis functions (high-pass filter).

        Args:
            duration (float): Filter duration in seconds
            drop (int): Number of low-frequency bases to drop
        """
        from .regressors import add_dct_basis

        return add_dct_basis(self, duration, drop)

    # ==================== Concatenation ====================

    def append(
        self,
        dm: Union["DesignMatrix", List["DesignMatrix"]],
        axis: int = 0,
        keep_separate: bool = True,
        unique_cols: Optional[List[str]] = None,
        fill_na: Union[int, float] = 0,
        verbose: bool = False,
    ) -> "DesignMatrix":
        """
        Concatenate design matrices.

        Args:
            dm (DesignMatrix or list of DesignMatrix): Design matrix/matrices to append
            axis (int): 0 for row-wise (vertical), 1 for column-wise (horizontal)
            keep_separate (bool): Whether to separate polynomial columns across runs (only axis=0)
            unique_cols (list of str, optional): Additional columns to keep separated (supports wildcards)
            fill_na (int or float): Value to fill NaN values during vertical concatenation
            verbose (bool): Print messages about polynomial separation
        """
        from .append import append

        return append(self, dm, axis, keep_separate, unique_cols, fill_na, verbose)

    def _append_horizontal(
        self,
        to_append: List["DesignMatrix"],
        fill_na: Union[int, float],
    ) -> "DesignMatrix":
        """
        Horizontal concatenation (axis=1) - add columns from other matrices.
        """
        from .append import append_horizontal

        return append_horizontal(self, to_append, fill_na)

    def _append_vertical(
        self,
        to_append: List["DesignMatrix"],
        keep_separate: bool,
        unique_cols: Optional[List[str]],
        fill_na: Union[int, float],
        verbose: bool,
    ) -> "DesignMatrix":
        """
        Vertical concatenation (axis=0) - stack rows, with optional polynomial separation.
        """
        from .append import append_vertical

        return append_vertical(
            self, to_append, keep_separate, unique_cols, fill_na, verbose
        )

    def _match_column_pattern(self, columns: List[str], pattern: str) -> List[str]:
        """
        Match columns against pattern with wildcard support.

        Args:
            columns (list of str): Column names to search
            pattern (str): Pattern to match (supports '*' as wildcard)
                - 'motion*' matches motion_x, motion_y
                - '*_motion' matches x_motion, y_motion
                - 'exact' matches only 'exact'
        """
        from .append import match_column_pattern

        return match_column_pattern(columns, pattern)

    def _get_starting_run_idx(self) -> int:
        """
        Determine next run index for multi-run appending.

        Returns:
            int: Next run index (0 if not multi-run, max_existing_idx + 1 otherwise)
        """
        from .append import get_starting_run_idx

        return get_starting_run_idx(self)

    def _identify_columns_to_separate(
        self, all_dms: List["DesignMatrix"], unique_cols: Optional[List[str]]
    ) -> set:
        """
        Identify which columns need run-specific separation.

        Args:
            all_dms (list of DesignMatrix): All matrices being concatenated
            unique_cols (list of str, optional): User-specified columns to separate (supports wildcards)

        Returns:
            set: Column names that should be separated with run prefixes
        """
        from .append import identify_columns_to_separate

        return identify_columns_to_separate(self, all_dms, unique_cols)

    def _append_vertical_with_separation(
        self,
        to_append: List["DesignMatrix"],
        unique_cols: Optional[List[str]],
        fill_na: Union[int, float],
        verbose: bool,
    ) -> "DesignMatrix":
        """
        Vertical concatenation with automatic polynomial separation.

        This creates run-specific columns (e.g., 0_poly_0, 1_poly_0) that are
        active only in their respective runs (sparse representation).
        """
        from .append import append_vertical_with_separation

        return append_vertical_with_separation(
            self, to_append, unique_cols, fill_na, verbose
        )

    # ==================== Diagnostics ====================

    def vif(self, exclude_polys: bool = True) -> np.ndarray:
        """
        Compute variance inflation factor for each column.

        Uses diagonal elements of inverted correlation matrix
        (same method as Matlab and R).

        Args:
            exclude_polys (bool): Skip polynomial columns (default True)

        Returns:
            np.ndarray: VIF values for each non-polynomial column
        """
        from .diagnostics import vif

        return vif(self, exclude_polys)

    def clean(
        self,
        fill_na: Union[int, float, None] = 0,
        exclude_polys: bool = False,
        thresh: float = 0.95,
        verbose: bool = True,
    ) -> "DesignMatrix":
        """
        Remove highly correlated columns.

        Removes columns with correlation >= threshold. Keeps first instance
        of correlated pair, drops duplicates.

        Args:
            fill_na (int, float, or None): Fill NaN values before checking correlations (default 0)
            exclude_polys (bool): Skip polynomial columns from correlation check
            thresh (float): Correlation threshold (drop if abs(r) >= thresh, default 0.95)
            verbose (bool): Print dropped column names

        Returns:
            DesignMatrix: Cleaned matrix with highly correlated columns removed
        """
        from .diagnostics import clean

        return clean(self, fill_na, exclude_polys, thresh, verbose)

    # ==================== Utilities ====================

    def details(self) -> str:
        """
        Return human-readable metadata summary.

        Returns:
            str: Formatted string showing sampling_freq, shape, convolved columns,
                and polynomial columns
        """
        from .diagnostics import details

        return details(self)

    def replace_data(
        self,
        data: np.ndarray,
        column_names: Optional[List[str]] = None,
    ) -> "DesignMatrix":
        """
        Replace data columns while preserving polynomials and metadata.

        This is useful when you want to substitute stimulus regressors
        while keeping nuisance regressors (polynomials, drift terms) intact.

        Args:
            data (ndarray): New data array (must match number of rows in current DesignMatrix)
            column_names (list of str, optional): Names for new data columns. If not provided, uses numeric names.

        Returns:
            DesignMatrix: New DesignMatrix with replaced data columns, preserved polynomials

        Raises:
            ValueError: If row count doesn't match existing data
        """
        # Validate row count
        if data.shape[0] != self.shape[0]:
            raise ValueError(
                f"Row count mismatch: new data has {data.shape[0]} rows, "
                f"but DesignMatrix has {self.shape[0]} rows"
            )

        # Generate column names if not provided
        if column_names is None:
            n_cols = data.shape[1] if data.ndim > 1 else 1
            column_names = [f"col_{i}" for i in range(n_cols)]

        # Create Polars DataFrame from new data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        new_data_df = pl.DataFrame(data, schema=column_names, orient="row")

        # Extract polynomial columns from current data
        poly_df = self._df.select(self.polys) if self.polys else pl.DataFrame()

        # Combine: new data + existing polynomials
        if poly_df.shape[1] > 0:
            combined_df = pl.concat([new_data_df, poly_df], how="horizontal")
        else:
            combined_df = new_data_df

        # Return new DesignMatrix with preserved metadata
        return self._copy_with(combined_df)

    def heatmap(self, figsize: tuple = (8, 6), **kwargs):
        """
        Visualize design matrix as heatmap (SPM-style).

        Creates a heatmap visualization of the design matrix columns.
        Uses seaborn + matplotlib under the hood.

        Args:
            figsize (tuple, default=(8, 6)): Figure size (width, height) in inches
            **kwargs: Additional keyword arguments passed to seaborn.heatmap()

        Returns:
            matplotlib.axes.Axes: The axes object containing the heatmap

        Examples:
            >>> dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
            >>> dm.heatmap()
        """
        from .io import heatmap

        return heatmap(self, figsize, **kwargs)

    # ==================== Internal Helpers ====================

    def _copy_with(
        self,
        new_df: pl.DataFrame,
        **metadata_updates,
    ) -> "DesignMatrix":
        """
        Create new DesignMatrix with updated data/metadata.

        This is the core pattern for immutable transformations.
        All methods that transform data should use this helper.

        Args:
            new_df (pl.DataFrame): New underlying data
            **metadata_updates: Metadata attributes to override (e.g., convolved=['stim'])
        """
        # Start with current metadata
        metadata = self._get_metadata()

        # Apply updates
        metadata.update(metadata_updates)

        # Create new DesignMatrix
        new_dm = DesignMatrix(
            new_df,
            sampling_freq=metadata["sampling_freq"],
            convolved=metadata["convolved"],
            polys=metadata["polys"],
        )
        new_dm.multi = metadata["multi"]

        return new_dm

    def _get_metadata(self) -> dict:
        """Extract metadata as dict (for copying)."""
        return {
            "sampling_freq": self.sampling_freq,
            "convolved": self.convolved.copy(),
            "polys": self.polys.copy(),
            "multi": self.multi,
        }

    def _get_data_columns(self, exclude_polys: bool = True) -> List[str]:
        """
        Get column names, optionally excluding polynomials.

        This helper reduces code duplication across methods that need to
        distinguish between data columns and polynomial/nuisance columns.

        Args:
            exclude_polys (bool, default=True): If True, exclude polynomial/nuisance columns from the result

        Returns:
            list of str: Column names (excluding polys if requested)

        Examples:
            >>> dm = DesignMatrix(...)
            >>> dm.add_poly(2)
            >>> dm._get_data_columns(exclude_polys=True)
            ['stim_1', 'stim_2']  # poly_0, poly_1, poly_2 excluded
            >>> dm._get_data_columns(exclude_polys=False)
            ['stim_1', 'stim_2', 'poly_0', 'poly_1', 'poly_2']
        """
        if exclude_polys and self.polys:
            return [col for col in self.columns if col not in self.polys]
        return list(self.columns)

    def to_pandas(self) -> pd.DataFrame:
        """Convert DesignMatrix to pandas DataFrame.

        Uses dict-based conversion to avoid pyarrow dependency. This is slightly
        slower (~10-20%) than pyarrow-based conversion but removes the dependency.

        Returns:
            pd.DataFrame: Pandas DataFrame with same data and column names.

        Examples:
            >>> dm = DesignMatrix(np.random.randn(100, 3))
            >>> pd_df = dm.to_pandas()
            >>> type(pd_df)
            <class 'pandas.core.frame.DataFrame'>
        """
        from .io import to_pandas

        return to_pandas(self)

    def _to_pandas(self) -> pd.DataFrame:
        """Internal method for pandas conversion at library boundaries.

        .. deprecated:: 0.6.0
            Use :meth:`to_pandas` instead.

        Returns:
            pd.DataFrame: Pandas DataFrame with same data and column names.
        """
        from .io import _to_pandas

        return _to_pandas(self)

    def to_numpy(self) -> np.ndarray:
        """
        Convert DesignMatrix to numpy array.

        Returns data columns as 2D numpy array (rows × columns).
        Column order is preserved from DataFrame.

        Returns:
            np.ndarray: 2D array with shape (n_samples, n_columns)

        Examples:
            >>> dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)
            >>> arr = dm.to_numpy()
            >>> arr.shape
            (3, 2)
        """
        from .io import to_numpy

        return to_numpy(self)

    def write(self, file_name: str, sep: str = "\t") -> None:
        """Write DesignMatrix to file.

        Supports TSV (default), CSV, and HDF5 formats. The format is
        automatically determined by file extension.

        Args:
            file_name: Output file path. Use .tsv, .csv, or .h5/.hdf5 extension.
            sep: Column separator for text files (default: tab for TSV).
                 Ignored for HDF5 files.

        Returns:
            None

        Examples:
            >>> dm = DesignMatrix(np.random.randn(100, 3), sampling_freq=1)
            >>> dm.write("design_matrix.tsv")  # TSV format (BIDS compatible)
            >>> dm.write("design_matrix.csv", sep=",")  # CSV format
            >>> dm.write("design_matrix.h5")  # HDF5 format

        Notes:
            TSV format is recommended for BIDS compatibility.
            HDF5 format preserves metadata (sampling_freq, convolved, polys).
        """
        from .io import write

        return write(self, file_name, sep)

    def _write_h5(self, file_name: str) -> None:
        """Write DesignMatrix to HDF5 file with metadata.

        Args:
            file_name: Output HDF5 file path.
        """
        from .io import write_h5

        return write_h5(self, file_name)

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Numpy array interface - enables np.array(design_matrix) and np.asarray().

        This is the standard numpy protocol for converting objects to arrays.
        It's what numpy.asarray() calls internally.

        Args:
            dtype (numpy dtype, optional): Desired data type for the array

        Returns:
            np.ndarray: 2D numpy array representation

        Examples:
            >>> dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)
            >>> np.array(dm)  # Uses __array__()
            array([[1], [2], [3]])
            >>> np.asarray(dm)  # Also uses __array__()
            array([[1], [2], [3]])
        """
        arr = self._df.to_numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def sum(self, axis: int = 0) -> pl.Series:
        """
        Compute sum along axis.

        For design matrices, typically used to count onsets (sum down columns).
        This is useful for validating that the number of events in each column
        matches expected onset counts.

        Args:
            axis (int, default=0): 0: sum down columns (returns Series with column sums)
                1: sum across rows (returns Series with row sums)

        Returns:
            pl.Series: Sums along specified axis with appropriate names

        Examples:
            >>> dm = DesignMatrix({"stim_a": [1, 0, 1, 0], "stim_b": [0, 1, 0, 1]})
            >>> dm.sum()  # Count total events per condition
            shape: (2,)
            Series: '' [i64]
            [
                2
                2
            ]
            >>> dm.sum(axis=1)  # Sum across conditions per timepoint
            shape: (4,)
            Series: '' [i64]
            [
                1
                1
                1
                1
            ]
        """
        if axis == 0:
            # Sum down columns: collect sums into a Series
            # Polars df.sum() returns a DataFrame with 1 row, we need a Series
            sums = [self._df[col].sum() for col in self._df.columns]
            return pl.Series(values=sums, name="")
        elif axis == 1:
            # Sum across columns for each row
            return self._df.select(pl.sum_horizontal(pl.all())).to_series()
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    def __eq__(self, other) -> bool:
        """
        Check equality with another DesignMatrix.

        Compares data frames only (ignores metadata like sampling_freq, convolved,
        polys, and multi). Uses Polars' native equals() for fast comparison.

        Args:
            other (DesignMatrix): Design matrix to compare with

        Returns:
            bool: True if data frames are equal (same shape, column names, and values)

        Examples:
            >>> dm1 = DesignMatrix({"a": [1, 2, 3]})
            >>> dm2 = DesignMatrix({"a": [1, 2, 3]})
            >>> dm1 == dm2
            True
            >>> dm3 = DesignMatrix({"a": [1, 2, 4]})
            >>> dm1 == dm3
            False

        Notes:
            This implements Python's equality protocol. It only compares data,
            not metadata. Use this for verifying that two design matrices have
            identical structure and values.
        """
        if not isinstance(other, DesignMatrix):
            return NotImplemented
        return self._df.equals(other._df)

    def reset_index(self, drop: bool = True) -> "DesignMatrix":
        """
        Reset index (pandas compatibility method).

        Polars DataFrames don't have row indexes like pandas, so this is a no-op
        that returns self. Included for backward compatibility with pandas-based
        code (e.g., file_reader.py).

        Args:
            drop (bool, default=True): Ignored (Polars has no index to drop). Kept for API compatibility.

        Returns:
            DesignMatrix: Returns self unchanged

        Examples:
            >>> dm = DesignMatrix({"a": [1, 2, 3]})
            >>> dm_reset = dm.reset_index(drop=True)
            >>> dm_reset is dm  # Same object
            True

        Notes:
            This method exists solely for compatibility with pandas-based code.
            In pandas, reset_index() resets row indexes to default (0, 1, 2, ...).
            In Polars, there are no row indexes, so this is unnecessary.
        """
        return self
