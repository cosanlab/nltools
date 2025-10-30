"""
DesignMatrix - Polars-based design matrix for neuroimaging analysis

This is a clean reimplementation using Polars instead of pandas.
Focus: Idiomatic Polars patterns, efficient vectorization, clean composition.
"""

import warnings
import numpy as np
import pandas as pd
import polars as pl
from typing import Union, List, Optional, Any


class DesignMatrix:
    """
    Polars-based design matrix for experimental designs in neuroimaging.

    Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
    Uses composition pattern (not subclassing) for clean metadata preservation.

    Parameters
    ----------
    data : DataFrame, ndarray, dict, or None
        Input data. Accepts:
        - Polars DataFrame (zero-copy)
        - pandas DataFrame (converted to Polars)
        - numpy ndarray
        - dict (keys=columns, values=data)
        - None (empty initialization)
    sampling_freq : float, optional
        Sampling frequency in Hz (1/TR for fMRI data)
    columns : list of str, optional
        Column names (used with ndarray input)
    convolved : list of str, optional
        Names of convolved columns (tracked internally)
    polys : list of str, optional
        Names of polynomial columns (tracked internally)

    Attributes
    ----------
    sampling_freq : float or None
        Sampling frequency in Hz
    convolved : list of str
        Columns that have been convolved
    polys : list of str
        Polynomial/nuisance columns (intercept, trends, DCT bases)
    multi : bool
        True if created from multi-run concatenation
    shape : tuple
        (n_rows, n_cols)
    columns : list of str
        Column names
    empty : bool
        True if no data

    Examples
    --------
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

        # Create internal Polars DataFrame
        # TODO: Implement construction logic
        self._df = pl.DataFrame()

    # ==================== Properties ====================

    @property
    def shape(self) -> tuple:
        """Return (n_rows, n_cols) tuple."""
        # TODO: Implement
        return (0, 0)

    @property
    def columns(self) -> List[str]:
        """Return list of column names."""
        # TODO: Implement
        return []

    @columns.setter
    def columns(self, new_names: List[str]):
        """Set column names."""
        # TODO: Implement
        pass

    @property
    def empty(self) -> bool:
        """Return True if DesignMatrix has no data."""
        # TODO: Implement
        return True

    def __len__(self) -> int:
        """Return number of rows."""
        # TODO: Implement
        return 0

    # ==================== Data Access ====================

    def __getitem__(self, key: Union[str, List[str]]) -> Union[pl.Series, "DesignMatrix"]:
        """
        Access columns.

        dm['col'] returns Series
        dm[['col1', 'col2']] returns DesignMatrix
        """
        # TODO: Implement
        raise NotImplementedError("__getitem__ not yet implemented")

    def __setitem__(self, key: str, value: Union[int, float, list, np.ndarray, pl.Series]):
        """
        Set column values.

        dm['col'] = 0  # Broadcast scalar
        dm['col'] = [1, 2, 3]  # Array assignment
        """
        # TODO: Implement
        raise NotImplementedError("__setitem__ not yet implemented")

    # ==================== Simple Transformations ====================

    def fillna(self, value: Union[int, float]) -> "DesignMatrix":
        """Fill NaN/null values with specified value."""
        # TODO: Implement using .fill_null() and .fill_nan()
        raise NotImplementedError("fillna not yet implemented")

    def drop(self, columns: List[str]) -> "DesignMatrix":
        """Drop specified columns."""
        # TODO: Implement
        raise NotImplementedError("drop not yet implemented")

    # ==================== Statistical Operations ====================

    def zscore(self, columns: Optional[List[str]] = None) -> "DesignMatrix":
        """
        Z-score standardize columns (mean=0, std=1).

        Excludes polynomial columns by default.
        """
        # TODO: Implement using Polars expressions
        raise NotImplementedError("zscore not yet implemented")

    def downsample(self, target: float, **kwargs) -> "DesignMatrix":
        """Reduce temporal resolution to target frequency."""
        # TODO: Implement
        raise NotImplementedError("downsample not yet implemented")

    def upsample(self, target: float, **kwargs) -> "DesignMatrix":
        """Increase temporal resolution to target frequency."""
        # TODO: Implement
        raise NotImplementedError("upsample not yet implemented")

    # ==================== Convolution ====================

    def convolve(
        self,
        conv_func: Union[str, np.ndarray] = "hrf",
        columns: Optional[List[str]] = None,
    ) -> "DesignMatrix":
        """
        Convolve columns with HRF or custom kernel.

        Parameters
        ----------
        conv_func : str or ndarray
            'hrf' for canonical Glover HRF, or custom kernel(s)
            Can be 1D array (single kernel) or 2D (samples x kernels)
        columns : list of str, optional
            Columns to convolve (default: all non-polynomial columns)
        """
        # TODO: Implement
        raise NotImplementedError("convolve not yet implemented")

    # ==================== Polynomial/Basis Functions ====================

    def add_poly(self, order: int = 0, include_lower: bool = True) -> "DesignMatrix":
        """
        Add Legendre polynomial drift terms.

        Parameters
        ----------
        order : int
            Polynomial order (0=intercept, 1=linear, 2=quadratic, ...)
        include_lower : bool
            If True, include all orders from 0 to order
        """
        # TODO: Implement
        raise NotImplementedError("add_poly not yet implemented")

    def add_dct_basis(self, duration: float = 180, drop: int = 0) -> "DesignMatrix":
        """
        Add discrete cosine transform basis functions (high-pass filter).

        Parameters
        ----------
        duration : float
            Filter duration in seconds
        drop : int
            Number of low-frequency bases to drop
        """
        # TODO: Implement
        raise NotImplementedError("add_dct_basis not yet implemented")

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

        Parameters
        ----------
        dm : DesignMatrix or list of DesignMatrix
            Design matrix/matrices to append
        axis : int
            0 for vertical (row-wise), 1 for horizontal (column-wise)
        keep_separate : bool
            Auto-separate polynomial columns across runs (axis=0 only)
        unique_cols : list of str, optional
            Additional columns to separate (supports wildcards: 'house*', '*_motion')
        fill_na : int or float
            Fill value for missing columns
        verbose : bool
            Print separation details
        """
        # TODO: Implement complex append logic
        raise NotImplementedError("append not yet implemented")

    # ==================== Diagnostics ====================

    def vif(self, exclude_polys: bool = True) -> np.ndarray:
        """
        Compute variance inflation factor for each column.

        Returns array of VIF values. VIF > 10 indicates problematic collinearity.
        """
        # TODO: Implement using correlation matrix
        raise NotImplementedError("vif not yet implemented")

    def clean(
        self,
        fill_na: Union[int, float, None] = 0,
        exclude_polys: bool = False,
        thresh: float = 0.95,
        verbose: bool = True,
    ) -> "DesignMatrix":
        """
        Remove highly correlated columns.

        Parameters
        ----------
        fill_na : int, float, or None
            Fill NaNs before checking correlation
        exclude_polys : bool
            Skip polynomials in collinearity check
        thresh : float
            Correlation threshold (drop if r >= thresh)
        verbose : bool
            Print dropped columns
        """
        # TODO: Implement
        raise NotImplementedError("clean not yet implemented")

    # ==================== Utilities ====================

    def details(self) -> str:
        """Return human-readable metadata summary."""
        # TODO: Implement
        return f"DesignMatrix(sampling_freq={self.sampling_freq}, shape={self.shape})"

    def replace_data(
        self,
        data: np.ndarray,
        column_names: Optional[List[str]] = None,
    ) -> "DesignMatrix":
        """
        Replace data columns while preserving polynomials and metadata.

        Parameters
        ----------
        data : ndarray
            New data (must match number of rows)
        column_names : list of str, optional
            Names for new columns
        """
        # TODO: Implement
        raise NotImplementedError("replace_data not yet implemented")

    def heatmap(self, figsize: tuple = (8, 6), **kwargs):
        """
        Visualize design matrix as heatmap (SPM-style).

        Uses seaborn + matplotlib under the hood.
        """
        # TODO: Implement using seaborn.heatmap
        raise NotImplementedError("heatmap not yet implemented")

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

        Parameters
        ----------
        new_df : pl.DataFrame
            New underlying data
        **metadata_updates
            Metadata attributes to override (e.g., convolved=['stim'])
        """
        # TODO: Implement
        raise NotImplementedError("_copy_with not yet implemented")

    def _get_metadata(self) -> dict:
        """Extract metadata as dict (for copying)."""
        return {
            "sampling_freq": self.sampling_freq,
            "convolved": self.convolved.copy(),
            "polys": self.polys.copy(),
            "multi": self.multi,
        }

    @classmethod
    def _from_polars(cls, df: pl.DataFrame, metadata: Optional[dict] = None) -> "DesignMatrix":
        """
        Create DesignMatrix from Polars DataFrame with optional metadata.

        Used internally for constructing results of operations.
        """
        # TODO: Implement
        raise NotImplementedError("_from_polars not yet implemented")
