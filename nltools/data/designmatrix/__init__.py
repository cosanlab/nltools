"""Polars-based design matrix for neuroimaging analysis.

`DesignMatrix` wraps a Polars DataFrame with neuroimaging metadata (sampling
frequency, which columns are HRF-convolved, which are confounds) and offers
HRF convolution, resampling, polynomial and cosine drift regressors, multi-run
concatenation, and collinearity diagnostics.
"""

from __future__ import annotations

__all__ = ["DesignMatrix"]

from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .utils import (
    copy_frame,
    copy_with,
    df_passthrough,
    effective_frame,
    replacement_names,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def _is_pandas_dataframe(obj) -> bool:
    """Duck-type check for pandas DataFrame without importing pandas."""
    cls = type(obj)
    module = cls.__module__
    return cls.__name__ == "DataFrame" and (
        module == "pandas" or module.startswith("pandas.")
    )


class DesignMatrix:
    """Represent an experimental design for neuroimaging as a Polars-backed matrix.

    Wraps a Polars DataFrame (one row per timepoint, one column per regressor)
    together with the metadata a GLM needs: the sampling frequency, which
    columns have been HRF-convolved, and which columns are nuisance/confound
    regressors. Transformations return new instances with that metadata
    preserved; `DesignMatrix` is composed over the DataFrame rather than
    subclassing it. Unknown attributes are forwarded to the underlying
    DataFrame, so the Polars API is available directly (``dm.select(...)``,
    ``dm.filter(...)``, ``dm.slice(...)`` return a `DesignMatrix`). Every eager DataFrame result becomes a new
    `DesignMatrix`; Series and builder objects remain native Polars values.
    Metadata is retained only when the operation establishes its validity.

    `data` accepts a Polars DataFrame (copied), a pandas DataFrame
    (converted), a NumPy array (named via `columns`), a dict of columns,
    another `DesignMatrix` (copied), ``None`` (empty), or a file path.
    A `.tsv`/`.csv` path is read as a BIDS events file when it has `onset`
    and `duration` columns — each `trial_type` becomes an HRF-convolved
    regressor, or a raw boxcar under ``hrf_model=None`` — and as a plain table
    otherwise (typically confounds). A `.h5`/`.hdf5` path written by `write` restores
    the data and the metadata (`sampling_freq`, `convolved`, `confounds`,
    `multi`), so neither `run_length` nor `sampling_freq` is required;
    passing either overrides what the file recorded.

    Args:
        data (DesignMatrix | pl.DataFrame | pd.DataFrame | np.ndarray | dict | str | Path | None):
            Input data; see above for how each type is interpreted.
        sampling_freq (float | None): Sampling frequency in Hz (1/TR for fMRI
            data). Mutually exclusive with `TR`.
        TR (float | None): Repetition time in seconds, a convenience for
            ``sampling_freq = 1/TR``. Mutually exclusive with `sampling_freq`.
        run_length (int | str | None): Number of TRs in the run. Required when
            `data` is a path to a text file. Pass ``'infer'`` for tabular
            (confounds) files to accept whatever row count the file has; not
            valid for events files. Not used for `.h5` inputs, which carry
            their own length.
        columns (list[str] | None): Column names, used with NumPy input.
        convolved (list[str] | None): Names of columns that are already
            HRF-convolved.
        confounds (list[str] | None): Names of nuisance/confound columns
            (intercept, polynomial drift, DCT cosines, motion, …).
        hrf_model (str | None): HRF model used to convolve regressors loaded
            from a BIDS events file — ``'glover'`` (the default),
            ``'glover_time'``, ``'glover_dispersion'``, ``'spm'``,
            ``'spm_time'``, ``'spm_dispersion'``, or ``None`` to keep raw
            boxcar regressors. A model name hands the events straight to
            nilearn's ``make_first_level_design_matrix``, so the regressors are
            the ones a nilearn `FirstLevelModel` would build from the same
            file. Ignored for every other kind of `data`.
        n_rows (int | None): Number of timepoints for a matrix with no columns
            (Polars cannot represent "n rows, 0 columns"). Rarely needed
            directly; set by `find_spikes` and by `append`.

    Attributes:
        data (pl.DataFrame): The underlying Polars DataFrame.
        sampling_freq (float | None): Sampling frequency in Hz.
        convolved (list[str]): Names of HRF-convolved columns (read-only;
            managed by `convolve` and `append`).
        confounds (list[str]): Names of nuisance/confound columns (read-only;
            managed by `add_poly`, `add_dct_basis`, `append`, and the
            constructor). Skipped by `convolve` and kept separate per run on
            multi-run vertical `append`.
        multi (bool): True if the matrix was created by a multi-run
            vertical `append`.
        columns (list[str]): Column names.
        shape (tuple[int, int]): ``(n_rows, n_cols)``.
        is_empty (bool): True if the matrix holds no data.

    Examples:
        ```python
        # Create from a NumPy array
        dm = DesignMatrix(np.zeros((100, 2)), sampling_freq=0.5, columns=["a", "b"])

        # Add a column
        dm["stim"] = [0, 1, 1, 0] * 25

        # Convolve with the HRF — convolved columns get a `_c0` suffix
        dm_conv = dm.convolve()  # 'stim' → 'stim_c0'

        # Add polynomial drift terms
        dm_conv = dm_conv.add_poly(order=2)

        # Multi-run concatenation separates drift terms per run
        dm_run1 = DesignMatrix(run1_events, sampling_freq=0.5, run_length=100).add_poly(0)
        dm_run2 = DesignMatrix(run2_events, sampling_freq=0.5, run_length=100).add_poly(0)
        dm_multi = dm_run1.append(dm_run2, axis=0)  # → .nl_r0_poly_0, .nl_r1_poly_0
        ```
    """

    _metadata = ["sampling_freq", "convolved", "confounds", "multi"]

    def __init__(
        self,
        data: DesignMatrix
        | pl.DataFrame
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
        hrf_model: str | None = "glover",
        n_rows: int | None = None,
    ):
        """Initialize a DesignMatrix from any supported input type.

        Passing another `DesignMatrix` returns a copy: `data`, `sampling_freq`,
        `convolved`, `confounds`, and `multi` are carried over, and any explicit
        kwarg overrides the inherited value.

        When `data` is a path to a BIDS events file, the events go to nilearn's
        `make_first_level_design_matrix` with the named `hrf_model`
        (``'glover'`` by default): output columns are suffixed ``_c0`` and
        `convolved` is populated. Pass ``hrf_model=None`` to load raw boxcar
        regressors instead — useful for FIR designs, PPI flows that build
        interaction terms before convolution, or teaching material that
        introduces convolution as a separate step. Those boxcars are sampled
        onto the TR grid, so convolving them afterwards with `convolve` is not
        the same as letting the constructor convolve the events: onsets that
        fall between TRs have already been quantized.
        """
        if TR is not None and sampling_freq is not None:
            raise ValueError("Pass exactly one of `TR` or `sampling_freq`, not both.")
        for name, value in (("TR", TR), ("sampling_freq", sampling_freq)):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        if n_rows is not None and (
            isinstance(n_rows, bool) or not isinstance(n_rows, Integral) or n_rows < 0
        ):
            raise ValueError("n_rows must be a nonnegative integer.")
        if TR is not None:
            sampling_freq = 1.0 / TR

        from .regressors import _KERNELS, _kernel_names

        if hrf_model is not None and hrf_model not in _KERNELS:
            raise ValueError(
                f"Unknown hrf_model={hrf_model!r}. Accepted HRF model names "
                f"are {_kernel_names()}, or hrf_model=None (boxcar — caller "
                "convolves explicitly with .convolve())."
            )

        self.multi = False
        _is_events = False  # set True only by the events-file branch below

        # Create internal Polars DataFrame based on input type
        if isinstance(data, DesignMatrix):
            # Copy-constructor: inherit data + metadata; explicit kwargs override.
            self.data = copy_frame(data.data, {id(data): self})
            if sampling_freq is None:
                sampling_freq = data.sampling_freq
            if convolved is None:
                convolved = list(data.convolved)
            if confounds is None:
                confounds = list(data.confounds)
            if n_rows is None:
                n_rows = data._n_rows
            self.multi = data.multi
            self._run_count = data._run_count

        elif data is None:
            # Empty initialization
            self.data = pl.DataFrame()

        elif isinstance(data, (str, Path)):
            from nltools.io import is_h5_path

            if is_h5_path(data):
                # A .h5 is a serialized DesignMatrix rather than a table
                # awaiting interpretation: it carries its own sampling_freq
                # and row count, so neither has to be supplied (and
                # `run_length` has nothing to describe). Explicit kwargs
                # still win over what the file recorded.
                from .io import read_h5

                self.data, stored = read_h5(data)
                if sampling_freq is None:
                    sampling_freq = stored.get("sampling_freq")
                if convolved is None:
                    convolved = stored.get("convolved")
                if confounds is None:
                    confounds = stored.get("confounds")
                if n_rows is None:
                    n_rows = stored.get("n_rows")
                self.multi = stored.get("multi", False)
                if "run_count" in stored:
                    self._run_count = stored["run_count"]
            else:
                if run_length is None:
                    raise ValueError(
                        "Loading DesignMatrix from a file requires `run_length`."
                    )
                if sampling_freq is None:
                    raise ValueError(
                        "Loading DesignMatrix from a file requires `TR` or `sampling_freq`."
                    )
                from .io import load_from_file

                # _is_events mirrors to the outer scope so the post-dispatch
                # auto-convolve block (below) can pick it up.
                self.data, _is_events = load_from_file(
                    data,
                    run_length=run_length,
                    sampling_freq=sampling_freq,
                    hrf_model=hrf_model,
                )

        elif isinstance(data, pl.DataFrame):
            self.data = data

        elif isinstance(data, dict):
            # Dictionary - let Polars handle it, ensure string column names
            names = [str(c) for c in data]
            if len(set(names)) != len(names):
                raise ValueError(
                    "Column names must be unique after conversion to strings."
                )
            self.data = pl.DataFrame(dict(zip(names, data.values())))

        elif isinstance(data, np.ndarray):
            if data.ndim not in (1, 2):
                raise ValueError("NumPy input must have one or two dimensions.")
            if data.ndim == 2 and data.shape[1] == 0:
                if n_rows is not None and n_rows != data.shape[0]:
                    raise ValueError("n_rows conflicts with array observations.")
                n_rows = data.shape[0]
            data = data.copy()
            # Numpy array - handle column names
            if columns is not None:
                # Use provided column names
                self.data = pl.DataFrame(
                    data, schema=[str(c) for c in columns], orient="row"
                )
            else:
                # Auto-generate column names as strings: '0', '1', '2', ...
                n_cols = data.shape[1] if data.ndim > 1 else 1
                auto_columns = [str(i) for i in range(n_cols)]
                self.data = pl.DataFrame(data, schema=auto_columns, orient="row")

        elif _is_pandas_dataframe(data):
            # pandas DataFrame - convert to Polars, ensure string column names
            self.data = pl.from_pandas(data)
            self.data = self.data.rename({col: str(col) for col in self.data.columns})

        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Expected DesignMatrix, Polars/pandas DataFrame, numpy array, "
                f"dict, str/Path, or None."
            )

        if not isinstance(data, DesignMatrix):
            self.data = copy_frame(self.data)
        for annotation in (convolved, confounds):
            if annotation is not None and any(
                c not in self.data.columns for c in annotation
            ):
                raise ValueError("Annotation names must refer to existing columns.")

        # Initialize metadata (after data dispatch so copy-constructor can
        # populate inherited values). Stored on private attrs so the public
        # ``.convolved`` / ``.confounds`` are read-only properties.
        self.sampling_freq = sampling_freq
        self._convolved = list(convolved) if convolved is not None else []
        self._confounds = list(confounds) if confounds is not None else []

        # Polars derives height from its columns, so a frame with no columns
        # always reports 0 rows. A design matrix with no regressors still
        # describes a specific number of timepoints (e.g. find_spikes() on a
        # subject with no spikes), and that length is needed for .append() to
        # line it up against other runs. Remember it explicitly — and refuse
        # a value the data contradicts rather than silently ignoring it.
        if n_rows is not None:
            if n_rows < 0:
                raise ValueError(f"n_rows must be non-negative, got {n_rows}.")
            if self.data.width > 0 and n_rows != self.data.height:
                raise ValueError(
                    f"n_rows={n_rows} conflicts with the data's "
                    f"{self.data.height} rows. Omit n_rows when the frame has "
                    f"columns — it is only needed to give a column-less "
                    f"DesignMatrix a length."
                )
        self._n_rows = n_rows if self.data.width == 0 else None
        if "_run_count" not in self.__dict__:
            self._run_count = 1 if self.shape[0] > 0 else 0
            if self.multi:
                # Files predating explicit run counts encode identities in names.
                from nltools.utils import parse_run_separated

                runs = [parse_run_separated(c) for c in self.columns]
                self._run_count = max(
                    (run[0] + 1 for run in runs if run is not None),
                    default=self._run_count,
                )

        # An events file loaded with an `hrf_model` came back already convolved
        # by nilearn (`load_from_file` → `_events_to_convolved_dm`), suffixed
        # `_c0`. Record that rather than convolving a second time; with
        # ``hrf_model=None`` the frame is raw boxcars and stays unannotated.
        if _is_events and hrf_model is not None:
            self._convolved = list(self.data.columns)

    # ── Dunders (alphabetical) ──────────────────────────────────────────

    def __array__(self, dtype=None) -> np.ndarray:
        """Provide the NumPy array interface.

        This enables ``np.array(design_matrix)`` and ``np.asarray()``.

        Args:
            dtype (np.dtype | None): Desired data type for the array.

        Returns:
            np.ndarray: 2D array representation.
        """
        if self.data.width == 0 and self._n_rows is not None:
            return np.empty((self._n_rows, 0), dtype=dtype or np.float64)
        arr = deepcopy(self.data.to_numpy().copy())
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __dir__(self):
        """Include polars DataFrame attrs for REPL/IDE autocomplete."""
        return sorted(set(super().__dir__()) | set(dir(self.data)))

    def __eq__(self, other) -> bool:
        """Check equality with another DesignMatrix.

        Compares data frames only (ignores metadata like `sampling_freq`,
        `convolved`, `confounds`, and `multi`).

        Args:
            other (DesignMatrix): Design matrix to compare with.

        Returns:
            bool: True if the data frames are equal (same shape, column names, and values).
        """
        if not isinstance(other, DesignMatrix):
            return NotImplemented
        return self.shape == other.shape and self.data.equals(other.data)

    def __getattr__(self, name: str):
        """Forward unknown attrs to the underlying polars DataFrame.

        Eager frame results use operation-aware metadata policies; native
        Series, scalar and builder results retain Polars return types. The ``data`` guard avoids recursion
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
        """Access one column as a Series or several as a new DesignMatrix.

        Args:
            key (str | list[str]): A single column name or a list of names.

        Returns:
            pl.Series | DesignMatrix: ``dm['col']`` returns a Polars Series;
                ``dm[['col1', 'col2']]`` returns a `DesignMatrix` with metadata preserved.
        """
        if isinstance(key, str):
            # Single column - return Series
            return copy_frame(self.data.select(key)).to_series()
        if isinstance(key, list) and all(isinstance(c, str) for c in key):
            # Multiple columns - return DesignMatrix with metadata
            subset_df = self.data.select(key)
            return copy_with(self, subset_df)
        raise TypeError(f"Column key must be str or list of str, got {type(key)}")

    def __len__(self) -> int:
        """Return number of rows."""
        return self.shape[0]

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

    def __setitem__(
        self,
        key: str,
        value: int | float | list | np.ndarray | pl.Series | pl.Expr,
    ):
        """Set or add a column in place.

        Args:
            key (str): Column name.
            value (int | float | list | np.ndarray | pl.Series | pl.Expr): A
                scalar is broadcast; a list, array, or Series is assigned as-is;
                a Polars expression is evaluated against the current columns.

        Examples:
            ```python
            dm["col"] = 0                          # broadcast scalar
            dm["col"] = [1, 2, 3]                  # array assignment
            dm["col"] = pl.col("a") + pl.col("b")  # Polars expression
            ```
        """
        result = self.with_columns(**{key: value})
        self.__dict__.update(result.__dict__)

    # ── Properties (alphabetical) ───────────────────────────────────────

    @property
    def columns(self) -> list[str]:
        """Column names of the design matrix as a list of strings."""
        return self.data.columns

    @columns.setter
    def columns(self, new_names: list[str]):
        """Set column names."""
        str_names = [str(name) for name in new_names]
        if len(str_names) != len(self.columns):
            raise ValueError("Column names must match the number of columns.")
        result = self.rename(dict(zip(self.data.columns, str_names)))
        self.__dict__.update(result.__dict__)

    @property
    def confounds(self) -> list[str]:
        """Names of nuisance/confound columns (read-only).

        Managed by `convolve`, `append`, `add_poly`, `add_dct_basis`, and the
        ``confounds=`` constructor kwarg. Direct assignment raises
        ``AttributeError`` — pass via the constructor or use
        ``.append(other, axis=1)`` (which auto-tracks confounds when `other`
        is a raw Polars DataFrame).
        """
        return list(self._confounds)

    @confounds.setter
    def confounds(self, value):
        raise AttributeError(
            "DesignMatrix.confounds is read-only. Pass `confounds=...` to the "
            "constructor, or use `.append(other_dm, axis=1, as_confounds=True)` "
            "/ `.append(raw_df, axis=1)` (raw frames are auto-marked) to "
            "register confound regressors."
        )

    @property
    def convolved(self) -> list[str]:
        """Names of HRF-convolved columns (read-only).

        Managed by `convolve` and `append` (which merges across inputs).
        Direct assignment raises ``AttributeError`` — pass via the
        ``convolved=`` constructor kwarg if you need to set initial state.
        """
        return list(self._convolved)

    @convolved.setter
    def convolved(self, value):
        raise AttributeError(
            "DesignMatrix.convolved is read-only. Pass `convolved=...` to the "
            "constructor, or use `.convolve()` / `.append()` which manage this "
            "metadata automatically."
        )

    @property
    def is_empty(self) -> bool:
        """True if the design matrix holds no data."""
        return self.data.is_empty()

    @property
    def shape(self) -> tuple:
        """The ``(n_rows, n_cols)`` shape of the matrix.

        For a matrix with no regressors, ``n_rows`` comes from the height
        recorded at construction (Polars cannot represent "n rows, 0 columns").
        """
        if self.data.width == 0 and self._n_rows is not None:
            return (self._n_rows, 0)
        return self.data.shape

    # ── Public methods (alphabetical) ───────────────────────────────────

    def add_dct_basis(
        self,
        duration: float = 180,
        drop: int = 0,
        *,
        include_constant: bool = True,
    ) -> DesignMatrix:
        """Add discrete cosine transform basis functions for high-pass filtering.

        Args:
            duration (float): Filter duration in seconds. Default: 180.
            drop (int): Number of low-frequency bases to drop. Default: 0.
            include_constant (bool): If True, also add a constant/intercept
                column named ``.nl_cosine_0`` (analogous to ``.nl_poly_0`` in
                `add_poly`). The underlying DCT basis drops the constant
                per SPM convention; set False to match SPM behavior.
                Default: True.

        Returns:
            DesignMatrix: New DesignMatrix with DCT basis columns appended.
        """
        from .regressors import add_dct_basis

        return add_dct_basis(
            self, duration=duration, drop=drop, include_constant=include_constant
        )

    def add_poly(self, order: int = 0, include_lower: bool = True) -> DesignMatrix:
        """Add Legendre polynomial drift terms.

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
        data: DesignMatrix | list[DesignMatrix],
        *,
        axis: int = 0,
        keep_separate: bool = True,
        unique_cols: list[str] | None = None,
        fill_na: int | float | None = 0,
        as_confounds: bool = False,
        progress_bar: bool = False,
    ) -> DesignMatrix:
        """Concatenate design matrices.

        Args:
            data (DesignMatrix or list of DesignMatrix): Design matrix/matrices to append.
            axis (int): 0 for row-wise (vertical), 1 for column-wise (horizontal).
                Default: 0.
            keep_separate (bool): Whether to separate confound columns across runs
                (only applies when axis=0). Default: True.
            unique_cols (list of str, optional): Additional columns to keep separated
                (supports wildcards).
            fill_na (int, float, or None): Value to fill NaN values during
                vertical concatenation, or None to preserve nulls. Default: 0.
            as_confounds (bool): Only applies when ``axis=1``. If True, mark all
                columns from ``data`` as nuisance/confounds in the result — they
                get skipped by ``.convolve()`` and separated across runs on
                later vertical appends. Default: False.
            progress_bar (bool): Print messages about confound separation. Default: False.

        Returns:
            DesignMatrix: Concatenated design matrix.
        """
        from .append import append

        return append(
            self,
            data,
            axis=axis,
            keep_separate=keep_separate,
            unique_cols=unique_cols,
            fill_na=fill_na,
            as_confounds=as_confounds,
            progress_bar=progress_bar,
        )

    def clean(
        self,
        *,
        fill_na: int | float | None = 0,
        exclude_confounds: bool = False,
        thresh: float = 0.95,
        progress_bar: bool = False,
    ) -> DesignMatrix:
        """Remove highly correlated columns.

        Args:
            fill_na (int, float, or None): Fill NaN values before checking correlations (default 0)
            exclude_confounds (bool): Skip confound/nuisance columns from correlation check
            thresh (float): Correlation threshold (drop if abs(r) >= thresh, default 0.95)
            progress_bar (bool): Print dropped column names. Default: False

        Returns:
            DesignMatrix: Cleaned matrix with highly correlated columns removed
        """
        from .diagnostics import clean

        return clean(
            self,
            fill_na=fill_na,
            exclude_confounds=exclude_confounds,
            thresh=thresh,
            progress_bar=progress_bar,
        )

    def convolve(
        self,
        kernel: str | np.ndarray = "glover",
        columns: list[str] | None = None,
    ) -> DesignMatrix:
        """Convolve columns with an HRF model or custom kernel.

        Convolved columns are always renamed to ``<col>_c{i}`` (where ``i`` is
        the kernel index, ``0`` for a single 1-D kernel). The source columns
        are dropped, and ``self.convolved`` lists the post-suffix names so
        downstream metadata stays in sync with the dataframe.

        A kernel name selects one of nilearn's HRF models: each column goes to
        `nilearn.glm.first_level.compute_regressor` as a condition, convolved
        at an oversampling factor of 50 and resampled onto the frame times.
        That is exactly what `FirstLevelModel` computes, so a column whose
        samples sit on the TR grid gives the regressor nilearn would build from
        the same events; sub-TR timing a column cannot represent is lost before
        convolution, so pass an events table to the constructor for that.

        Args:
            kernel (str or ndarray): An HRF model name — `'glover'` (default),
                `'glover_time'`, `'glover_dispersion'`, `'spm'`, `'spm_time'`
                or `'spm_dispersion'` — or custom kernel(s) as a 1D array
                (single kernel) or 2D array (samples x kernels).
            columns (list of str, optional): Columns to convolve (default: all non-confound columns).

        Returns:
            DesignMatrix: New DesignMatrix with convolved columns renamed.
        """
        from .regressors import convolve

        return convolve(self, kernel, columns)

    def copy(self) -> DesignMatrix:
        """Create a deep copy of the DesignMatrix.

        Returns:
            DesignMatrix: Copy of the current DesignMatrix
        """
        return deepcopy(self)

    def __copy__(self):
        """Return an independently owned copy."""
        return deepcopy(self)

    def __deepcopy__(self, memo):
        """Copy the retained graph while preserving aliases and cycles."""
        if id(self) in memo:
            return memo[id(self)]
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.data = copy_frame(self.data, memo)
        for key, value in self.__dict__.items():
            if key != "data":
                setattr(result, key, deepcopy(value, memo))
        return result

    def downsample(self, target: float, method: str = "mean") -> DesignMatrix:
        """Reduce temporal resolution using Polars-native operations.

        Args:
            target (float): Target sampling frequency in Hz (must be < current sampling_freq)
            method (str): Aggregation method - 'mean' or 'median' (default: 'mean')

        Returns:
            DesignMatrix: Downsampled DesignMatrix with updated sampling_freq
        """
        from .transforms import downsample

        return downsample(self, target, method=method)

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

    def plot(  # nosemgrep: kwargs-internal-forwarding  # forwards to matplotlib via plot_designmatrix
        self,
        method: str = "matrix",
        *,
        columns: list[str] | None = None,
        rescale: bool = True,
        metric: str = "pearson",
        ax=None,
        figsize: tuple | None = None,
        title: str | None = None,
        cmap: str | None = None,
        save: str | None = None,
        **kwargs,
    ) -> Figure:
        """Visualize the design matrix.

        Dispatches over `method` (mirroring `BrainData.plot`):

        - ``'matrix'`` (default): SPM-style heatmap (rows = TRs, columns = regressors).
        - ``'timeseries'``: overlaid line plot of regressor time courses. Pass
          the same `ax` across calls to overlay multiple DesignMatrices
          (e.g. original vs. convolved).
        - ``'corr'``: labeled correlation heatmap of the columns (reuses
          `corr`; diagonal restored to 1.0 for display).

        Args:
            method (str): One of ``'matrix'``, ``'timeseries'``, or ``'corr'``.
                Default: ``'matrix'``.
            columns (list of str, optional): Subset of columns to plot.
                Defaults to all columns.
            rescale (bool): ``'matrix'`` only. Rescale each column by its L2
                norm so columns with different native magnitudes are visually
                comparable (SPM/nilearn convention). Default: True.
            metric (str): ``'corr'`` only. ``'pearson'`` (default) or
                ``'spearman'``.
            ax (matplotlib.axes.Axes, optional): Existing axis to draw on; a new
                figure is created if omitted.
            figsize (tuple, optional): Figure size; sensible per-method default
                when omitted.
            title (str, optional): Axis title.
            cmap (str, optional): Colormap (``'matrix'`` / ``'corr'``).
            save (str, optional): Path to save the figure.
            **kwargs (dict): Forwarded to the underlying plotter
                (``seaborn.heatmap`` for ``'matrix'`` / ``'corr'``;
                ``matplotlib.axes.Axes.plot`` for ``'timeseries'``).

        Returns:
            matplotlib.figure.Figure: The figure containing the plot.
        """
        from .plotting import plot_designmatrix

        return plot_designmatrix(
            self,
            method,
            columns=columns,
            rescale=rescale,
            metric=metric,
            ax=ax,
            figsize=figsize,
            title=title,
            cmap=cmap,
            save=save,
            **kwargs,
        )

    def replace_data(
        self,
        data: np.ndarray,
        column_names: list[str] | None = None,
    ) -> DesignMatrix:
        """Replace data columns while preserving confounds and metadata.

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
            combined_df = pl.concat([new_data_df, confound_df], how="horizontal_extend")
        else:
            combined_df = new_data_df

        return copy_with(self, combined_df, operation="replace", replaced=column_names)

    def standardize(
        self, *, method: str = "center", columns: list[str] | None = None
    ) -> DesignMatrix:
        """Standardize columns by centering them, optionally scaling to unit variance.

        Args:
            method (str): ``'center'`` subtracts the mean (default);
                ``'zscore'`` subtracts the mean and divides by the standard
                deviation.
            columns (list[str] | None): Columns to standardize. If None,
                standardize all non-confound columns.

        Returns:
            DesignMatrix: New DesignMatrix with standardized columns.

        Raises:
            ValueError: If `method` is neither ``'center'`` nor ``'zscore'``.
        """
        from .transforms import standardize

        return standardize(self, method=method, columns=columns)

    def sum(self, axis: int = 0) -> pl.Series:
        """Compute the sum along an axis.

        Args:
            axis (int): 0 to sum down each column, 1 to sum across each row.
                Default: 0.

        Returns:
            pl.Series: Sums along the specified axis.
        """
        if axis == 0:
            sums = [self.data[col].sum() for col in self.data.columns]
            return pl.Series(values=sums, name="")
        if axis == 1:
            return self.data.select(pl.sum_horizontal(pl.all())).to_series()
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    def to_numpy(self) -> np.ndarray:
        """Convert a DesignMatrix to a NumPy array.

        Returns:
            np.ndarray: 2D array with shape (n_samples, n_columns)
        """
        from .io import to_numpy

        return to_numpy(self)

    def upsample(self, target: float, method: str = "linear") -> DesignMatrix:
        """Increase temporal resolution to a target frequency.

        Args:
            target (float): Target sampling frequency in Hz (must be > current sampling_freq)
            method (str): Interpolation method - 'linear' or 'nearest' (default: 'linear')

        Returns:
            DesignMatrix: Upsampled DesignMatrix with updated sampling_freq
        """
        from .transforms import upsample

        return upsample(self, target, method)

    def corr(
        self,
        *,
        metric: str = "pearson",
        columns: list[str] | None = None,
    ):
        """Calculate column correlations as a similarity ``Adjacency``.

        Args:
            metric (str): ``'pearson'`` (default) or ``'spearman'``.
            columns (list of str, optional): Subset of columns to correlate.
                Defaults to all columns.

        Returns:
            Adjacency: Similarity matrix whose ``labels`` are the column names.
                The unit diagonal is dropped (self-correlation isn't an edge);
                use ``.plot(method='corr')`` for a heatmap with the diagonal
                restored.
        """
        from .diagnostics import corr

        return corr(self, metric=metric, columns=columns)

    def vif(self, exclude_confounds: bool = True) -> np.ndarray | None:
        """Compute the variance inflation factor for each column.

        Args:
            exclude_confounds (bool): Skip confound/nuisance columns. Default: True.

        Returns:
            np.ndarray: VIF values for each included column. Returns None if the
                correlation matrix is singular.
        """
        from .diagnostics import vif

        return vif(self, exclude_confounds)

    def with_columns(self, *exprs, **named_exprs) -> DesignMatrix:
        """Add or replace columns via Polars expressions.

        Mirrors ``pl.DataFrame.with_columns``. Named kwargs become named
        columns; positional ``pl.Expr`` arguments are accepted as-is
        (including ``pl.Expr.alias("name")``). Returns a new `DesignMatrix`
        preserving annotations on untouched columns. Replacing a column clears
        its convolution annotation and retains its confound role; new columns
        are untagged.

        For convenience, named-kwarg values that aren't ``pl.Expr`` /
        ``pl.Series`` are coerced: an ``int``/``float`` is broadcast as a
        scalar via ``pl.lit``, and a ``list`` / ``np.ndarray`` is wrapped as a
        ``pl.Series``.

        Args:
            *exprs (pl.Expr): Positional Polars expressions, passed through.
            **named_exprs (pl.Expr | pl.Series | np.ndarray | list | int | float):
                New columns keyed by name.

        Returns:
            DesignMatrix: New DesignMatrix with the columns added or replaced.

        Examples:
            ```python
            dm = dm.with_columns(motor=pl.sum_horizontal(motor_cols)).drop(motor_cols)
            dm = dm.with_columns(
                vmpfc=seed_signal,
                vmpfc_motor=pl.col("vmpfc") * pl.col("motor_c0"),
            )
            ```
        """
        from .utils import copy_with

        coerced = {}
        for name, value in named_exprs.items():
            if isinstance(value, (pl.Expr, pl.Series)):
                coerced[name] = value
            elif isinstance(value, (list, np.ndarray)):
                coerced[name] = pl.Series(name, value)
            elif isinstance(value, (int, float)):
                coerced[name] = pl.lit(value).alias(name)
            else:
                raise TypeError(
                    f"with_columns: kwarg {name!r} has unsupported type "
                    f"{type(value).__name__}. Pass a polars Expr/Series, "
                    "numpy array, list, or scalar."
                )
        frame = effective_frame(self)
        replaced = replacement_names(frame, exprs, coerced)
        new_data = frame.with_columns(*exprs, **coerced)
        if self.data.width == 0 and self._n_rows is not None and "" not in replaced:
            new_data = new_data.drop("")
        return copy_with(self, new_data, operation="replace", replaced=replaced)

    def write(self, file_name: str, sep: str | None = None) -> None:
        """Write DesignMatrix to file.

        Supports TSV, CSV, and HDF5 formats. Format is auto-detected from the
        file extension. Text formats carry the data only; ``.h5`` also
        preserves ``sampling_freq``, ``.convolved``, ``.confounds``, and
        ``.multi``, so ``DesignMatrix(path)`` restores the whole object.

        Args:
            file_name (str): Output file path with a `.tsv`, `.csv`, `.h5`, or
                `.hdf5` extension.
            sep (str | None): Column separator for text files. Defaults to the
                delimiter the extension implies (comma for `.csv`, tab
                otherwise); pass a value to override.
        """
        from .io import write

        return write(self, file_name, sep)
