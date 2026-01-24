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
from polars import selectors as cs
from typing import Union, List, Optional
from scipy.special import legendre
from ..stats import make_cosine_basis


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
        shape (tuple): (n_rows, n_cols)
        columns (list of str): Column names
        empty (bool): True if no data

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
        """Return list of column names."""
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
        # Determine which columns to z-score
        if columns is None:
            # Default: all columns except polynomials
            columns_to_zscore = self._get_data_columns(exclude_polys=True)
        else:
            columns_to_zscore = columns

        # Build Polars expressions for z-scoring
        # For each column: (col - mean) / std
        zscore_exprs = [
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
            for col in columns_to_zscore
        ]

        # Use .with_columns() to replace only the zscored columns
        # (automatically preserves untouched columns - idiomatic Polars pattern)
        zscored_df = self._df.with_columns(zscore_exprs)

        return self._copy_with(zscored_df)

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
        if method == "zscore":
            return self.zscore(columns=columns)
        elif method == "center":
            # Determine which columns to center
            if columns is None:
                columns_to_center = self._get_data_columns(exclude_polys=True)
            else:
                columns_to_center = columns

            # Build Polars expressions for centering: (col - mean)
            center_exprs = [
                (pl.col(col) - pl.col(col).mean()).alias(col)
                for col in columns_to_center
            ]

            centered_df = self._df.with_columns(center_exprs)
            return self._copy_with(centered_df)
        else:
            raise ValueError(
                f"Invalid method '{method}'. Must be 'zscore' or 'center'."
            )

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
        if self.sampling_freq is None:
            raise ValueError(
                "DesignMatrix must have sampling_freq set for downsampling. "
                "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
            )

        if target >= self.sampling_freq:
            raise ValueError(
                f"Downsampling target ({target} Hz) must be less than current sampling_freq "
                f"({self.sampling_freq} Hz). For upsampling, use .upsample() instead."
            )

        if method not in ("mean", "median"):
            raise ValueError("method must be 'mean' or 'median'")

        # Calculate n_samples (number of original samples per downsampled sample)
        # This replicates stats.downsample() logic: n_samples = sampling_freq / target
        n_samples = self.sampling_freq / target

        # Create grouping indices: [0,0,..., 1,1,..., 2,2,...] with n_samples repetitions
        # This replicates: np.repeat(np.arange(1, data.shape[0] / n_samples, 1), n_samples)
        # Note: stats.downsample starts at 1, but we start at 0 (doesn't affect grouping)
        n_groups = int(self.shape[0] / n_samples)
        idx = pl.Series(np.repeat(np.arange(n_groups), int(n_samples)))

        # Handle remainder samples (last incomplete group)
        if self.shape[0] > len(idx):
            remainder = pl.Series(np.repeat(idx[-1] + 1, self.shape[0] - len(idx)))
            idx = pl.concat([idx, remainder])

        # Add grouping index to dataframe
        df_with_idx = self._df.with_columns(idx.alias("_group_idx"))

        # Get all data columns
        data_cols = self._get_data_columns(exclude_polys=False)

        # Group by index and aggregate
        if method == "mean":
            downsampled_df = (
                df_with_idx.group_by("_group_idx", maintain_order=True)
                .agg([pl.col(col).mean() for col in data_cols])
                .drop("_group_idx")
            )
        else:  # median
            downsampled_df = (
                df_with_idx.group_by("_group_idx", maintain_order=True)
                .agg([pl.col(col).median() for col in data_cols])
                .drop("_group_idx")
            )

        return self._copy_with(downsampled_df, sampling_freq=target)

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
        from scipy.interpolate import interp1d

        if self.sampling_freq is None:
            raise ValueError(
                "DesignMatrix must have sampling_freq set for upsampling. "
                "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
            )

        if target <= self.sampling_freq:
            raise ValueError(
                f"Upsampling target ({target} Hz) must be greater than current sampling_freq "
                f"({self.sampling_freq} Hz). For downsampling, use .downsample() instead."
            )

        if method not in ("linear", "nearest"):
            raise ValueError("method must be 'linear' or 'nearest'")

        # Calculate step size (this matches stats.upsample logic)
        # For hz target_type: n_samples = sampling_freq / target
        step_size = self.sampling_freq / target

        # Create original and new index arrays (matches stats.upsample)
        orig_indices = np.arange(0, self.shape[0], 1)
        new_indices = np.arange(0, self.shape[0] - 1, step_size)

        # Get all data columns (including polynomials - upsample everything)
        data_cols = self._get_data_columns(exclude_polys=False)

        # Interpolate each column using scipy (matches stats.upsample)
        upsampled_data = {}
        for col in data_cols:
            col_data = self._df[col].to_numpy()

            # Create interpolation function
            interpolate = interp1d(orig_indices, col_data, kind=method)

            # Interpolate to new indices
            upsampled_data[col] = interpolate(new_indices)

        # Create new Polars DataFrame
        upsampled_df = pl.DataFrame(upsampled_data)

        return self._copy_with(upsampled_df, sampling_freq=target)

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
        from ..algorithms.hrf import glover_hrf

        if self.sampling_freq is None:
            raise ValueError(
                "DesignMatrix must have sampling_freq set for convolution. "
                "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
            )

        # Determine which columns to convolve
        if columns is None:
            # Default: all columns except polynomials
            columns_to_convolve = self._get_data_columns(exclude_polys=True)
        else:
            columns_to_convolve = columns

        # Get the convolution kernel
        if isinstance(conv_func, str):
            if conv_func != "hrf":
                raise ValueError(
                    f"String conv_func must be 'hrf', got '{conv_func}'. "
                    "Use conv_func='hrf' or provide a numpy array. "
                    "Tip: Use nltools.utils.glover_hrf() to generate custom HRFs."
                )
            # Generate Glover HRF at this sampling frequency
            # TR = 1 / sampling_freq
            conv_func = glover_hrf(1.0 / self.sampling_freq, oversampling=1.0)
        elif isinstance(conv_func, np.ndarray):
            if len(conv_func.shape) > 2:
                raise ValueError(
                    f"HRF function must be 1D (shape: (samples,)) or 2D (shape: (samples, n_kernels)). "
                    f"Got shape: {conv_func.shape}. "
                    "Tip: Use nltools.utils.glover_hrf() to generate HRFs."
                )
        else:
            raise TypeError(
                f"conv_func must be 'hrf' (str) or numpy array, got {type(conv_func).__name__}. "
                "Tip: Use conv_func='hrf' for canonical HRF."
            )

        # Perform convolution
        n_rows = self.shape[0]

        if len(conv_func.shape) == 1:
            # Single kernel: keep original column names (replace in-place)
            convolved_series = []
            for col in columns_to_convolve:
                # Convert to numpy for convolution operation
                # NECESSARY: np.convolve requires numpy arrays (no Polars equivalent)
                col_data = self._df[col].to_numpy()
                convolved = np.convolve(col_data, conv_func)[:n_rows]
                convolved_series.append(pl.Series(col, convolved))

            # Use .with_columns() to replace only convolved columns
            # (automatically preserves non-convolved columns and column order)
            new_df = self._df.with_columns(convolved_series)

        else:
            # Multiple kernels: shape is (samples, n_kernels)
            n_kernels = conv_func.shape[1]
            convolved_series = []

            for col in columns_to_convolve:
                col_data = self._df[col].to_numpy()
                for k_idx in range(n_kernels):
                    kernel = conv_func[:, k_idx]
                    convolved = np.convolve(col_data, kernel)[:n_rows]
                    convolved_series.append(pl.Series(f"{col}_c{k_idx}", convolved))

            # Drop original columns, add convolved variants (idiomatic Polars pattern)
            new_df = self._df.drop(columns_to_convolve).with_columns(convolved_series)

        # Update metadata
        return self._copy_with(new_df, convolved=columns_to_convolve)

    # ==================== Polynomial/Basis Functions ====================

    def add_poly(self, order: int = 0, include_lower: bool = True) -> "DesignMatrix":
        """
        Add Legendre polynomial drift terms.

        Args:
            order (int): Polynomial order (0=intercept, 1=linear, 2=quadratic, ...)
            include_lower (bool): If True, include all orders from 0 to order
        """
        if order < 0:
            raise ValueError(
                f"Polynomial order must be >= 0, got {order}. "
                "Common orders: 0 (intercept only), 1 (linear trend), 2 (quadratic), 3 (cubic)."
            )

        # Check for ambiguous polynomials from previous append operations
        if self.polys and any(elem.count("_") == 2 for elem in self.polys):
            raise ValueError(
                "This Design Matrix contains polynomial terms that were kept "
                "separate from a previous append operation. This makes it ambiguous "
                "for adding polynomial terms. Try calling .add_poly() on each "
                "separate Design Matrix before appending them instead."
            )

        # Determine which polynomials to add
        if include_lower:
            orders_to_add = range(order + 1)
        else:
            orders_to_add = [order]

        # Check if we already have these polynomials (idempotent)
        new_poly_cols = {}
        for i in orders_to_add:
            poly_name = f"poly_{i}"
            if poly_name in self.polys:
                print(f"Design Matrix already has {i}th order polynomial...skipping")
            else:
                # Create normalized Legendre polynomial over [-1, 1]
                norm_order = np.linspace(-1, 1, self.shape[0])
                poly_values = legendre(i)(norm_order)
                new_poly_cols[poly_name] = poly_values

        # If no new polynomials to add, return self unchanged
        if not new_poly_cols:
            return self

        # Add new polynomial columns using Polars .with_columns()
        new_df = self._df.with_columns(
            [pl.Series(name, values) for name, values in new_poly_cols.items()]
        )

        # Update polys metadata
        new_polys = self.polys.copy() if self.polys else []
        new_polys.extend(new_poly_cols.keys())

        # Return new DesignMatrix with updated data and metadata
        return self._copy_with(new_df, polys=new_polys)

    def add_dct_basis(self, duration: float = 180, drop: int = 0) -> "DesignMatrix":
        """
        Add discrete cosine transform basis functions (high-pass filter).

        Args:
            duration (float): Filter duration in seconds
            drop (int): Number of low-frequency bases to drop
        """
        if self.sampling_freq is None:
            raise ValueError(
                "DesignMatrix must have sampling_freq set for DCT basis functions. "
                "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
            )

        # Check for ambiguous cosine bases from previous append operations
        if self.polys and any(
            elem.count("_") == 2 and "cosine" in elem for elem in self.polys
        ):
            raise ValueError(
                "This Design Matrix contains cosine bases that were kept "
                "separate from a previous append operation. This makes it ambiguous "
                "for adding polynomial terms. Try calling .add_dct_basis() on each "
                "separate Design Matrix before appending them instead."
            )

        # Create DCT basis matrix using stats function
        basis_mat = make_cosine_basis(
            self.shape[0], 1.0 / self.sampling_freq, duration, drop=drop
        )

        # Generate column names (cosine_1, cosine_2, ...)
        # Note: If drop > 0, numbering starts from drop+1 to reflect original indices
        # e.g., drop=2 -> cosine_3, cosine_4, ... (skipped cosine_1, cosine_2)
        basis_col_names = [f"cosine_{drop + i + 1}" for i in range(basis_mat.shape[1])]

        # Check which bases we don't already have (idempotent)
        if self.polys:
            basis_to_add = [name for name in basis_col_names if name not in self.polys]
        else:
            basis_to_add = basis_col_names

        # If no new bases to add, return self unchanged
        if not basis_to_add:
            print("All basis functions already exist...skipping")
            return self

        # Print message if only adding some bases
        if len(basis_to_add) < len(basis_col_names):
            print("Some basis functions already exist...skipping")

        # Add new cosine basis columns
        # Only add the columns we don't already have
        new_basis_cols = {}
        for i, name in enumerate(basis_col_names):
            if name in basis_to_add:
                new_basis_cols[name] = basis_mat[:, i]

        new_df = self._df.with_columns(
            [pl.Series(name, values) for name, values in new_basis_cols.items()]
        )

        # Update polys metadata
        new_polys = self.polys.copy() if self.polys else []
        new_polys.extend(new_basis_cols.keys())

        # Return new DesignMatrix with updated data and metadata
        return self._copy_with(new_df, polys=new_polys)

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
        # Normalize to list
        to_append = [dm] if not isinstance(dm, list) else dm

        # Validate all are DesignMatrix with same sampling_freq
        if not all(isinstance(elem, DesignMatrix) for elem in to_append):
            raise TypeError("All items to append must be DesignMatrix objects")
        if not all(elem.sampling_freq == self.sampling_freq for elem in to_append):
            raise ValueError(
                "All Design Matrices must have the same sampling frequency!"
            )

        if axis == 1:
            return self._append_horizontal(to_append, fill_na)
        elif axis == 0:
            return self._append_vertical(
                to_append, keep_separate, unique_cols, fill_na, verbose
            )
        else:
            raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")

    def _append_horizontal(
        self,
        to_append: List["DesignMatrix"],
        fill_na: Union[int, float],
    ) -> "DesignMatrix":
        """
        Horizontal concatenation (axis=1) - add columns from other matrices.
        """
        # Check all have same number of rows
        if not all(elem.shape[0] == self.shape[0] for elem in to_append):
            raise ValueError("All Design Matrices must have the same number of rows!")

        # Warn about duplicate column names
        all_columns = set(self.columns)
        for elem in to_append:
            if not all_columns.isdisjoint(elem.columns):
                print("Duplicate column names detected. Will be repeated.")
            all_columns.update(elem.columns)

        # Use Polars hstack to concatenate DataFrames horizontally
        dfs_to_stack = [self._df] + [elem._df for elem in to_append]
        new_df = pl.concat(dfs_to_stack, how="horizontal")

        # Fill NaN if requested
        if fill_na is not None:
            new_df = new_df.fill_null(fill_na)

        # Combine polys metadata from all matrices
        all_polys = self.polys.copy() if self.polys else []
        for elem in to_append:
            if elem.polys:
                all_polys.extend(elem.polys)

        return self._copy_with(new_df, polys=all_polys)

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
        # Simple case: keep_separate=False - just stack rows
        if not keep_separate:
            dfs_to_stack = [self._df] + [elem._df for elem in to_append]
            new_df = pl.concat(dfs_to_stack, how="vertical")

            # Fill NaN if requested
            if fill_na is not None:
                new_df = new_df.fill_null(fill_na)

            # Combine polys metadata (no separation)
            all_polys = self.polys.copy() if self.polys else []
            for elem in to_append:
                if elem.polys:
                    # Add new polys we don't already have
                    for p in elem.polys:
                        if p not in all_polys:
                            all_polys.append(p)

            return self._copy_with(new_df, polys=all_polys)

        # Complex case: keep_separate=True - separate polynomial columns across runs
        return self._append_vertical_with_separation(
            to_append, unique_cols, fill_na, verbose
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
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [c for c in columns if c.startswith(prefix)]
        elif pattern.startswith("*"):
            suffix = pattern[1:]
            return [c for c in columns if c.endswith(suffix)]
        else:
            return [c for c in columns if c == pattern]

    def _get_starting_run_idx(self) -> int:
        """
        Determine next run index for multi-run appending.

        Returns:
            int: Next run index (0 if not multi-run, max_existing_idx + 1 otherwise)
        """
        if not self.multi:
            return 0

        # Find max run index from column names like "0_poly_0", "1_motion_x"
        max_idx = -1
        for col in self.columns:
            if "_" in col:
                first_part = col.split("_")[0]
                if first_part.isdigit():
                    idx = int(first_part)
                    max_idx = max(max_idx, idx)

        return max_idx + 1 if max_idx >= 0 else 0

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
        cols_to_sep = set()

        # Add polynomial columns from non-multi DMs only
        # (Multi-run DMs already have separated polynomials)
        for dm in all_dms:
            if dm.polys and not dm.multi:
                cols_to_sep.update(dm.polys)

        # Add unique_cols with wildcard matching
        if unique_cols:
            # Collect all column names across all DMs
            all_column_names = set()
            for dm in all_dms:
                all_column_names.update(dm.columns)

            # Match each pattern
            for pattern in unique_cols:
                matched = self._match_column_pattern(list(all_column_names), pattern)
                cols_to_sep.update(matched)

        return cols_to_sep

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
        # Handle two cases differently:
        # 1. Self is NOT multi: process all DMs with sequential numbering
        # 2. Self IS multi: keep self unchanged, only process to_append DMs

        if not self.multi:
            # Case 1: Standard multi-run creation
            all_dms = [self] + to_append
            cols_to_sep = self._identify_columns_to_separate(all_dms, unique_cols)

            if verbose and cols_to_sep:
                print(f"Separating columns across runs: {sorted(cols_to_sep)}")

            processed_dfs = []
            all_new_polys = []

            for i, dm in enumerate(all_dms):
                # Build rename mapping for separated columns
                rename_map = {
                    col: f"{i}_{col}" for col in dm.columns if col in cols_to_sep
                }

                # Rename and collect
                processed_df = dm._df.rename(rename_map) if rename_map else dm._df
                processed_dfs.append(processed_df)

                # Track renamed polys
                for poly in dm.polys:
                    if poly in rename_map:
                        all_new_polys.append(rename_map[poly])

        else:
            # Case 2: Appending to existing multi-run DM
            start_idx = self._get_starting_run_idx()
            cols_to_sep = self._identify_columns_to_separate(to_append, unique_cols)

            if verbose and cols_to_sep:
                print(f"Separating columns across runs: {sorted(cols_to_sep)}")

            # Keep self's DataFrame unchanged
            processed_dfs = [self._df]
            all_new_polys = self.polys.copy() if self.polys else []

            # Process only the DMs being appended
            for i, dm in enumerate(to_append):
                run_idx = start_idx + i

                # Build rename mapping for separated columns
                rename_map = {
                    col: f"{run_idx}_{col}" for col in dm.columns if col in cols_to_sep
                }

                # Rename and collect
                processed_df = dm._df.rename(rename_map) if rename_map else dm._df
                processed_dfs.append(processed_df)

                # Track renamed polys
                for poly in dm.polys:
                    if poly in rename_map:
                        all_new_polys.append(rename_map[poly])

        # Concatenate with diagonal (auto-fills missing columns with null)
        result_df = pl.concat(processed_dfs, how="diagonal")

        # Fill nulls with fill_na value (creates sparse separation)
        result_df = result_df.fill_null(fill_na)

        # Return with updated metadata
        return self._copy_with(result_df, polys=all_new_polys, multi=True)

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
        if self.shape[1] <= 1:
            raise ValueError(
                "Can't compute VIF with only 1 column! "
                "VIF measures multicollinearity and requires at least 2 columns. "
                f"Your DesignMatrix has shape {self.shape}."
            )

        # Determine which columns to include (using polars selectors for declarative filtering)
        if exclude_polys and self.polys:
            # Use polars selector: "select all columns except polynomial terms"
            subset_df = self._df.select(cs.exclude(self.polys))
        elif exclude_polys:
            # No polys to exclude, use all columns
            subset_df = self._df
        else:
            # Always exclude intercept (poly_0) columns even when exclude_polys=False
            cols_to_use = [c for c in self.columns if "poly_0" not in c]
            subset_df = self._df.select(cols_to_use)

        # Edge case: single column has VIF = 1 (no multicollinearity)
        if subset_df.shape[1] == 1:
            return np.array([1.0])

        # Convert to numpy for correlation matrix and linear algebra
        # NECESSARY: Polars doesn't have correlation matrix or matrix inversion
        data_array = subset_df.to_numpy()

        # Compute correlation matrix
        corr_matrix = np.corrcoef(data_array, rowvar=False)

        # Compute VIF = diagonal of inverse correlation matrix
        try:
            inv_corr = np.linalg.inv(corr_matrix)
            return np.diag(inv_corr)
        except np.linalg.LinAlgError:
            # Matrix is singular - perfect collinearity detected
            # Return None and warn user (matches old behavior)
            print(
                "ERROR: Cannot compute VIF! Design Matrix is singular because it has "
                "some perfectly correlated or duplicated columns. Using .clean() may help."
            )
            return None

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
        # Check for duplicate column names
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                "Duplicate column names detected. Using .clean() with duplicate "
                "columns is not supported as it can produce unexpected results."
            )

        # Start with a copy
        result = self

        # Fill NaN if requested
        if fill_na is not None:
            result = result.fillna(fill_na)

        # Determine which columns to check for correlation
        if exclude_polys:
            cols_to_check = result._get_data_columns(exclude_polys=True)
        else:
            cols_to_check = list(result.columns)

        if len(cols_to_check) <= 1:
            if verbose:
                print("Only 1 column to check...skipping")
            return result

        # Compute pairwise correlations and identify columns to drop
        keep = []
        remove = []

        # Convert to numpy for pairwise correlation computation
        # NECESSARY: More efficient than Polars for this operation
        subset_df = result._df.select(cols_to_check)
        data_array = subset_df.to_numpy()

        # Check each pair of columns
        for i in range(len(cols_to_check)):
            col_i = cols_to_check[i]
            col_i_data = data_array[:, i]

            for j in range(i + 1, len(cols_to_check)):
                col_j = cols_to_check[j]
                col_j_data = data_array[:, j]

                # Skip if already marked for removal or keeping
                if col_j in keep or col_j in remove:
                    continue

                # Check for constant arrays (avoid correlation warnings)
                if np.var(col_i_data) == 0 or np.var(col_j_data) == 0:
                    r = 0.0
                else:
                    # Compute correlation
                    r = np.abs(np.corrcoef(col_i_data, col_j_data)[0, 1])

                # Mark for removal if correlation exceeds threshold
                if r >= thresh and col_i not in keep and col_i not in remove:
                    if verbose:
                        print(
                            f"{col_i} and {col_j} correlated at {r:.2f} which is >= "
                            f"threshold of {thresh}. Dropping {col_j}"
                        )
                    keep.append(col_i)
                    remove.append(col_j)

        # Drop correlated columns
        if remove:
            # Drop from DataFrame
            new_df = result._df.drop(remove)

            # Update polys metadata
            new_polys = [p for p in result.polys if p not in remove]

            # Return cleaned matrix
            return result._copy_with(new_df, polys=new_polys)
        else:
            if verbose:
                print("Dropping columns not needed...skipping")
            return result

    # ==================== Utilities ====================

    def details(self) -> str:
        """
        Return human-readable metadata summary.

        Returns:
            str: Formatted string showing sampling_freq, shape, convolved columns,
                and polynomial columns
        """
        lines = [
            f"DesignMatrix(sampling_freq={self.sampling_freq}, shape={self.shape})",
        ]

        if self.convolved:
            lines.append(f"  convolved: {self.convolved}")

        if self.polys:
            lines.append(f"  polys: {self.polys}")

        return "\n".join(lines)

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
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Convert to pandas for seaborn (which expects pandas DataFrames)
        df_for_plot = self._to_pandas()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Set default heatmap parameters
        heatmap_kwargs = {
            "cmap": "RdBu_r",
            "center": 0,
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
        return pd.DataFrame(self._df.to_dict(as_series=False))

    def _to_pandas(self) -> pd.DataFrame:
        """Internal method for pandas conversion at library boundaries.

        .. deprecated:: 0.6.0
            Use :meth:`to_pandas` instead.

        Returns:
            pd.DataFrame: Pandas DataFrame with same data and column names.
        """
        return self.to_pandas()

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
        return self._df.to_numpy()

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
        from pathlib import Path
        from ..utils import is_h5_path

        if isinstance(file_name, Path):
            file_name = str(file_name)

        if is_h5_path(file_name):
            self._write_h5(file_name)
        else:
            # Write as delimited text file (TSV or CSV)
            self._df.write_csv(file_name, separator=sep)

    def _write_h5(self, file_name: str) -> None:
        """Write DesignMatrix to HDF5 file with metadata.

        Args:
            file_name: Output HDF5 file path.
        """
        import h5py

        with h5py.File(file_name, "w") as f:
            # Store data
            f.create_dataset("data", data=self._df.to_numpy(), compression="gzip")

            # Store column names
            f.create_dataset(
                "columns",
                data=np.array(self.columns, dtype="S"),
                compression="gzip",
            )

            # Store metadata
            meta = f.create_group("metadata")
            if self.sampling_freq is not None:
                meta.attrs["sampling_freq"] = self.sampling_freq
            meta.attrs["convolved"] = np.array(self.convolved, dtype="S")
            meta.attrs["polys"] = np.array(self.polys, dtype="S")
            meta.attrs["multi"] = self.multi
            meta.attrs["obj_type"] = "design_matrix"

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
