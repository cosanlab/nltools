"""
This data class is for working with similarity/dissimilarity matrices
"""

import os
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import MDS
from sklearn.utils import check_random_state
from scipy.spatial.distance import squareform
from scipy.stats import ttest_1samp, t as t_dist
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from nltools.stats import (
    correlation_permutation,
    one_sample_permutation,
    two_sample_permutation,
    matrix_permutation,
    fisher_r_to_z,
    fisher_z_to_r,
)
from nltools.plotting import plot_stacked_adjacency, plot_silhouette
from nltools.utils import (
    all_same,
    attempt_to_import,
    concatenate,
    is_h5_path,
    to_h5,
)
from ..design_matrix import DesignMatrix
from pathlib import Path
from h5py import File as h5File
import warnings

# Optional dependencies
nx = attempt_to_import("networkx", "nx")
tables = attempt_to_import("tables")

MAX_INT = np.iinfo(np.int32).max


class Adjacency(object):
    """
    Adjacency is a class to represent Adjacency matrices as a vector rather
    than a 2-dimensional matrix. This makes it easier to perform data
    manipulation and analyses.

    Args:
        data: pandas data instance or list of files
        matrix_type: (str) type of matrix.  Possible values include:
                    ['distance','similarity','directed','distance_flat',
                    'similarity_flat','directed_flat']
        Y: Pandas DataFrame of training labels
        **kwargs: Additional keyword arguments

    """

    def __init__(self, data=None, Y=None, matrix_type=None, labels=None, **kwargs):
        if matrix_type is not None and matrix_type.lower() not in [
            "distance",
            "similarity",
            "directed",
            "distance_flat",
            "similarity_flat",
            "directed_flat",
        ]:
            raise ValueError(
                "matrix_type must be [None,'distance', "
                "'similarity','directed','distance_flat', "
                "'similarity_flat','directed_flat']"
            )

        verbose = kwargs.pop("verbose", False)

        # Setup data
        if data is None:
            self.data = np.array([])
            self.matrix_type = "empty"
            self.is_single_matrix = np.nan
            self.issymmetric = np.nan

        # List of Adjacency or filepaths to h5s or csvs
        elif isinstance(data, list):
            if isinstance(data[0], Adjacency):
                tmp = concatenate(data)
                for item in ["data", "matrix_type", "Y", "issymmetric"]:
                    setattr(self, item, getattr(tmp, item))

            # File paths or array/dataframes
            # NOTE: We don't support list of hdf5 filepaths! Only .csvs
            else:
                d_all = []
                symmetric_all = []
                matrix_type_all = []
                for d in data:
                    # CSV or array/dataframe
                    (
                        data_tmp,
                        issymmetric_tmp,
                        matrix_type_tmp,
                        _,
                    ) = self._import_single_data(d, matrix_type=matrix_type)
                    d_all.append(data_tmp)
                    symmetric_all.append(issymmetric_tmp)
                    matrix_type_all.append(matrix_type_tmp)

                if not all_same(symmetric_all):
                    raise ValueError("Not all matrices are of the same symmetric type.")
                if not all_same(matrix_type_all):
                    raise ValueError("Not all matrices are of the same matrix type.")

                self.data = np.array(d_all)
                self.issymmetric = symmetric_all[0]
                self.matrix_type = matrix_type_all[0]
            self.is_single_matrix = False

        # File path
        elif isinstance(data, (str, Path)):
            to_load = str(data)

            # HDF5
            if is_h5_path(to_load):
                try:
                    # Load X and Y attributes
                    with pd.HDFStore(to_load, "r") as f:
                        self.Y = f["Y"]

                    # Load other attributes
                    with h5File(to_load, "r") as f:
                        self.data = np.array(f["data"])
                        self.matrix_type = f["matrix_type"][()].decode()
                        self.is_single_matrix = f["is_single_matrix"][()]
                        self.issymmetric = f["issymmetric"][()]
                        # Deepdish saved empty label lists as np arrays of length 1
                        if len(f["labels"]) == 1:
                            self.labels = list(f["labels"])
                        elif len(f["labels"]) > 1:
                            self.labels = list(f["labels"].asstr())
                        else:
                            self.labels = []

                    # Done initializing
                    return
                except Exception as e:
                    if verbose:
                        warnings.warn(
                            f"Falling back to legacy h5 loading due to error: {e}"
                        )

                    with tables.open_file(to_load, mode="r") as f:
                        # Setup data
                        self.data = np.array(f.root["data"])

                        # Setup Y
                        if len(list(f.root["Y_columns"])):
                            self.Y = pd.DataFrame(
                                np.array(f.root["Y"]).squeeze(),
                                columns=[
                                    e.decode("utf-8") if isinstance(e, bytes) else e
                                    for e in np.array(f.root["Y_columns"])
                                ],
                                index=[
                                    e.decode("utf-8") if isinstance(e, bytes) else e
                                    for e in np.array(f.root["Y_index"])
                                ],
                            )
                        else:
                            self.Y = pd.DataFrame()

                        # Setup other attributes
                        if "matrix_type" in f.root:
                            self.matrix_type = list(f.root["matrix_type"])[0]
                        else:
                            warnings.warn(
                                "Loading legacy h5 file: matrix_type field missing, assuming 'distance'. "
                                "Consider re-saving the file to update to current format.",
                                UserWarning,
                            )
                            self.matrix_type = "distance_flat"

                        if "labels" in f.root:
                            self.labels = list(f.root["labels"])
                        else:
                            self.labels = None

                        # Compute other properties from data and matrix type
                        (
                            self.data,
                            self.issymmetric,
                            self.matrix_type,
                            self.is_single_matrix,
                        ) = self._import_single_data(
                            self.data, matrix_type=self.matrix_type
                        )

                        return

            # CSV or array/dateframe
            else:
                (
                    self.data,
                    self.issymmetric,
                    self.matrix_type,
                    self.is_single_matrix,
                ) = self._import_single_data(data, matrix_type=matrix_type)

        # CSV or array/dataframe
        else:
            (
                self.data,
                self.issymmetric,
                self.matrix_type,
                self.is_single_matrix,
            ) = self._import_single_data(data, matrix_type=matrix_type)

        # Setup Y dataframe
        if Y is None:
            self.Y = pd.DataFrame()

        elif isinstance(Y, (str, Path)):
            self.Y = pd.read_csv(Y, header=None, index_col=None)

        elif isinstance(Y, pd.DataFrame):
            self.Y = Y
        else:
            raise TypeError("Make sure Y filepath or pandas data frame.")

        # Ensure consistency
        if not self.Y.empty and self.data.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"Y rows ({self.Y.shape[0]}) do not match data rows ({self.data.shape[0]})"
            )

        if labels is None:
            self.labels = []

        elif isinstance(labels, (list, np.ndarray)):
            if self.is_single_matrix:
                if len(labels) != self.n_nodes:
                    raise ValueError(
                        "Make sure the length of labels matches the shape of data."
                    )
                self.labels = deepcopy(labels)
            else:
                if len(labels) != len(self):
                    if len(labels) != self.n_nodes:
                        raise ValueError(
                            "Make sure length of labels either "
                            "matches the number of Adjacency "
                            "matrices or the size of a single "
                            "matrix."
                        )
                    else:
                        self.labels = list(labels) * len(self)
                else:
                    if np.all(np.array([len(x) for x in labels]) != self.n_nodes):
                        raise ValueError(
                            "All lists of labels must be same length as shape of data."
                        )
                    self.labels = deepcopy(labels)
        else:
            raise TypeError("Make sure labels is a list or numpy array.")

    def __repr__(self):
        return "%s.%s(shape=%s, Y=%s, is_symmetric=%s, matrix_type=%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape,
            self.Y.shape,
            self.issymmetric,
            self.matrix_type,
        )

    def __getitem__(self, index):
        new = self.copy()
        if isinstance(index, (int, np.integer)):
            new.data = np.array(self.data[index, :]).squeeze()
            new.is_single_matrix = True
        else:
            new.data = np.array(self.data[index, :]).squeeze()
            new.is_single_matrix = self._test_is_single_matrix(new.data)
        if not self.Y.empty:
            new.Y = self.Y.iloc[index]
        return new

    def __len__(self):
        if self.is_single_matrix:
            return 1
        else:
            return self.data.shape[0]

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]

    def _perform_arithmetic(self, y, op, op_name, reverse=False):
        """Perform arithmetic operation with validation.

        Args:
            y: Operand (scalar or Adjacency).
            op: Callable that performs the operation on arrays.
            op_name: Name of operation for error messages.
            reverse: If True, reverse operand order (y op self).

        Returns:
            New Adjacency with result.
        """
        new = deepcopy(self)
        if isinstance(y, (int, np.integer, float, np.floating)):
            if reverse:
                new.data = op(y, new.data)
            else:
                new.data = op(new.data, y)
        elif isinstance(y, Adjacency):
            if self.shape != y.shape:
                raise ValueError(
                    "Both Adjacency() instances need to be the same shape."
                )
            if reverse:
                new.data = op(y.data, new.data)
            else:
                new.data = op(new.data, y.data)
        else:
            raise ValueError(f"Can only {op_name} int, float, or Adjacency")
        return new

    def __add__(self, y):
        return self._perform_arithmetic(y, np.add, "add")

    def __radd__(self, y):
        return self._perform_arithmetic(y, np.add, "add", reverse=True)

    def __sub__(self, y):
        return self._perform_arithmetic(y, np.subtract, "subtract")

    def __rsub__(self, y):
        return self._perform_arithmetic(y, np.subtract, "subtract", reverse=True)

    def __mul__(self, y):
        return self._perform_arithmetic(y, np.multiply, "multiply")

    def __rmul__(self, y):
        return self._perform_arithmetic(y, np.multiply, "multiply", reverse=True)

    def __truediv__(self, y):
        return self._perform_arithmetic(y, np.divide, "divide")

    @staticmethod
    def _test_is_single_matrix(data):
        """Static method because it belongs to the class, ie is only invoked via self.test_single_matrix or Adjacency.test_single_matrix and requires no self argument."""
        return len(data.shape) == 1

    def _import_single_data(self, data, matrix_type=None):
        """Helper function to import single data matrix."""

        if isinstance(data, str) or isinstance(data, Path):
            if os.path.isfile(data):
                data = pd.read_csv(data)
            else:
                raise ValueError("Make sure you have specified a valid file path.")

        if matrix_type is not None:
            if matrix_type.lower() == "distance_flat":
                matrix_type = "distance"
                data = np.array(data)
                issymmetric = True
                is_single_matrix = self._test_is_single_matrix(data)
            elif matrix_type.lower() == "similarity_flat":
                matrix_type = "similarity"
                data = np.array(data)
                issymmetric = True
                is_single_matrix = self._test_is_single_matrix(data)
            elif matrix_type.lower() == "directed_flat":
                matrix_type = "directed"
                data = np.array(data).flatten()
                issymmetric = False
                is_single_matrix = self._test_is_single_matrix(data)
            elif matrix_type.lower() in ["distance", "similarity", "directed"]:
                if data.shape[0] != data.shape[1]:
                    raise ValueError("Data matrix must be square")
                data = np.array(data)
                matrix_type = matrix_type.lower()
                if matrix_type in ["distance", "similarity"]:
                    issymmetric = True
                    data = data[np.triu_indices(data.shape[0], k=1)]
                else:
                    issymmetric = False
                    if isinstance(data, pd.DataFrame):
                        data = data.values.flatten()
                    elif isinstance(data, np.ndarray):
                        data = data.flatten()
                is_single_matrix = True
        else:
            if len(data.shape) == 1:  # Single Vector
                try:
                    data = squareform(data)
                except ValueError:
                    print(
                        "Data is not flattened upper triangle from "
                        "similarity/distance matrix or flattened directed "
                        "matrix."
                    )
                is_single_matrix = True
            elif data.shape[0] == data.shape[1]:  # Square Matrix
                is_single_matrix = True
            else:  # Rectangular Matrix
                data_all = deepcopy(data)
                try:
                    data = squareform(data_all[0, :])
                except ValueError:
                    print(
                        "Data is not flattened upper triangle from multiple "
                        "similarity/distance matrices or flattened directed "
                        "matrices."
                    )
                is_single_matrix = False

            # Test if matrix is symmetrical
            if np.all(
                data[np.triu_indices(data.shape[0], k=1)]
                == data.T[np.triu_indices(data.shape[0], k=1)]
            ):
                issymmetric = True
            else:
                issymmetric = False

            # Determine matrix type
            if issymmetric:
                if np.sum(np.diag(data)) == 0:
                    matrix_type = "distance"
                elif np.sum(np.diag(data)) == data.shape[0]:
                    matrix_type = "similarity"
                data = data[np.triu_indices(data.shape[0], k=1)]
            else:
                matrix_type = "directed"
                data = data.flatten()

            if not is_single_matrix:
                data = data_all

        return (data, issymmetric, matrix_type, is_single_matrix)

    @property
    def is_empty(self) -> bool:
        """Check if Adjacency object is empty.

        Returns:
            bool: True if the adjacency matrix is empty, False otherwise.
        """
        return self.matrix_type == "empty"

    @property
    def isempty(self) -> bool:
        """Check if Adjacency object is empty.

        .. deprecated:: 0.6.0
            Use :attr:`is_empty` instead.
        """
        import warnings

        warnings.warn(
            "isempty is deprecated and will be removed in a future version. "
            "Use is_empty instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_empty

    def to_square(self):
        """Convert adjacency back to square matrix format.

        This is an alias for :meth:`squareform`.

        Returns:
            np.ndarray or list: Square matrix representation. Returns a list
            of matrices if this object contains multiple adjacency matrices.
        """
        return self.squareform()

    def squareform(self):
        """Convert adjacency back to squareform"""
        if self.issymmetric:
            if self.is_single_matrix:
                return squareform(self.data)
            else:
                return [squareform(x.data) for x in self]
        else:
            if self.is_single_matrix:
                return self.data.reshape(
                    int(np.sqrt(self.data.shape[0])), int(np.sqrt(self.data.shape[0]))
                )
            else:
                return [
                    x.data.reshape(
                        int(np.sqrt(x.data.shape[0])), int(np.sqrt(x.data.shape[0]))
                    )
                    for x in self
                ]

    def plot(self, limit=3, axes=None, *args, **kwargs):
        """Create Heatmap of Adjacency Matrix

        Can pass in any sns.heatmap argument

        Args:
            limit: (int) number of heatmaps to plot if object contains multiple adjacencies (default: 3)
            axes: matplotlib axis handle
        """

        if self.is_single_matrix:
            if axes is None:
                _, axes = plt.subplots(nrows=1, figsize=(7, 5))
            if self.labels:
                sns.heatmap(
                    self.squareform(),
                    square=True,
                    ax=axes,
                    xticklabels=self.labels,
                    yticklabels=self.labels,
                    *args,
                    **kwargs,
                )
            else:
                sns.heatmap(self.squareform(), square=True, ax=axes, *args, **kwargs)
        else:
            if axes is not None:
                print("axes is ignored when plotting multiple images")
            n_subs = np.minimum(len(self), limit)
            _, a = plt.subplots(nrows=n_subs, figsize=(7, len(self) * 5))
            for i in range(n_subs):
                if self.labels:
                    sns.heatmap(
                        self[i].squareform(),
                        square=True,
                        xticklabels=self.labels[i],
                        yticklabels=self.labels[i],
                        ax=a[i],
                        *args,
                        **kwargs,
                    )
                else:
                    sns.heatmap(
                        self[i].squareform(), square=True, ax=a[i], *args, **kwargs
                    )
        return

    def _apply_stat(self, func, axis=0):
        """Apply a statistical function along an axis.

        Args:
            func: Numpy function to apply (e.g., np.nanmean).
            axis: Axis along which to apply function. 0 for across matrices,
                  1 for across upper triangle elements.

        Returns:
            float if single matrix, Adjacency if axis=0 with multiple matrices,
            np.array if axis=1 with multiple matrices.
        """
        if self.is_single_matrix:
            return func(self.data)
        else:
            if axis == 0:
                return Adjacency(
                    data=func(self.data, axis=axis),
                    matrix_type=self.matrix_type + "_flat",
                )
            elif axis == 1:
                return func(self.data, axis=axis)

    def mean(self, axis=0):
        """Calculate mean of Adjacency.

        Args:
            axis: Calculate mean over matrices (0) or upper triangle (1).

        Returns:
            float if single matrix, Adjacency if axis=0, np.array if axis=1.
        """
        return self._apply_stat(np.nanmean, axis)

    def sum(self, axis=0):
        """Calculate sum of Adjacency.

        Args:
            axis: Calculate sum over matrices (0) or upper triangle (1).

        Returns:
            float if single matrix, Adjacency if axis=0, np.array if axis=1.
        """
        return self._apply_stat(np.nansum, axis)

    def std(self, axis=0):
        """Calculate standard deviation of Adjacency.

        Args:
            axis: Calculate std over matrices (0) or upper triangle (1).

        Returns:
            float if single matrix, Adjacency if axis=0, np.array if axis=1.
        """
        return self._apply_stat(np.nanstd, axis)

    def median(self, axis=0):
        """Calculate median of Adjacency.

        Args:
            axis: Calculate median over matrices (0) or upper triangle (1).

        Returns:
            float if single matrix, Adjacency if axis=0, np.array if axis=1.
        """
        return self._apply_stat(np.nanmedian, axis)

    @property
    def shape(self):
        """Return the logical shape of the adjacency matrix.

        Returns:
            tuple: For single matrix: (n_nodes, n_nodes)
                   For stacked matrices: (n_matrices, n_nodes, n_nodes)
                   For empty: (0, 0)

        Note:
            Use `.vector_shape` to get the internal vectorized representation shape.
        """
        if self.matrix_type == "empty":
            return (0, 0)

        # Compute n_nodes from vector length
        if self.is_single_matrix:
            vector_len = self.data.shape[0]
        else:
            vector_len = self.data.shape[1]

        if self.issymmetric:
            # For symmetric: vector_len = n*(n-1)/2, solve for n
            n_nodes = int((1 + np.sqrt(1 + 8 * vector_len)) / 2)
        else:
            # For directed: vector_len = n*n
            n_nodes = int(np.sqrt(vector_len))

        if self.is_single_matrix:
            return (n_nodes, n_nodes)
        else:
            return (len(self), n_nodes, n_nodes)

    @property
    def vector_shape(self):
        """Return shape of internal vectorized representation.

        Returns:
            tuple: For single matrix: (vector_length,)
                   For stacked matrices: (n_matrices, vector_length)

        Note:
            This is the raw shape of the internal data storage.
            Use `.shape` for the logical (n_nodes, n_nodes) shape.
        """
        return self.data.shape

    @property
    def n_nodes(self):
        """Return the number of nodes in the adjacency matrix.

        Returns:
            int: Number of nodes (n) for an (n, n) matrix.
        """
        return self.shape[-1]

    def square_shape(self):
        """Calculate shape of squareform data.

        .. deprecated::
            Use `.shape` instead. This method will be removed in v0.7.0.
        """
        warnings.warn(
            "square_shape() is deprecated. Use .shape instead, which now returns "
            "(n_nodes, n_nodes) for single matrices or (n_matrices, n_nodes, n_nodes) "
            "for stacked matrices.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.matrix_type == "empty":
            return np.array([])
        else:
            if self.is_single_matrix:
                return self.squareform().shape
            else:
                return self[0].squareform().shape

    def copy(self):
        """Create a copy of Adjacency object."""
        return deepcopy(self)

    def append(self, data):
        """Append data to Adjacency instance

        Args:
            data:  (Adjacency) Adjacency instance to append

        Returns:
            out: (Adjacency) new appended Adjacency instance

        """

        if not isinstance(data, Adjacency):
            raise ValueError("Make sure data is a Adjacency instance.")

        if self.is_empty:
            out = data.copy()
        else:
            out = self.copy()
            if self.n_nodes != data.n_nodes:
                raise ValueError("Data is not the same shape as Adjacency instance.")

            out.data = np.vstack([self.data, data.data])
            out.is_single_matrix = False
            if out.Y.size:
                out.Y = pd.concat([self.Y, data.Y], ignore_index=True)

        return out

    def write(self, file_name, method="long"):
        """Write out Adjacency object to csv file.

        Args:
            file_name (str):  name of file name to write
            method (str):     method to write out data ['long','square']

        """
        if method not in ["long", "square"]:
            raise ValueError('Make sure method is ["long","square"].')

        if isinstance(file_name, Path):
            file_name = str(file_name)

        if is_h5_path(file_name):
            if method == "square":
                raise NotImplementedError(
                    'Saving as hdf5 does not support method="square"'
                )
            to_h5(self, file_name, obj_type="adjacency")
        else:
            if method == "long":
                pd.DataFrame(self.data).to_csv(file_name, index=None)
            elif self.is_single_matrix and method == "square":
                pd.DataFrame(self.squareform()).to_csv(file_name, index=None)
            elif not self.is_single_matrix and method == "square":
                raise NotImplementedError(
                    "Need to decide how we should write out multiple matrices. As separate files?"
                )

    def similarity(
        self,
        data,
        plot=False,
        perm_type="2d",
        n_permute=5000,
        metric="spearman",
        ignore_diagonal=False,
        nan_policy="omit",
        **kwargs,
    ):
        """
        Calculate similarity between two Adjacency matrices. Default is to use spearman
        correlation and permutation test.

        Args:
            data (Adjacency or array): Adjacency data, or 1-d array same size as self.data
            perm_type: (str) '1d','2d', or None
            metric: (str) 'spearman','pearson','kendall'
            ignore_diagonal: (bool) only applies to 'directed' Adjacency types using
                perm_type=None or perm_type='1d'
            nan_policy: (str) How to handle NaN values. Options:
                - 'omit': Remove NaN values pairwise before computing correlation (default)
                - 'propagate': Allow NaN to propagate through calculations
                - 'raise': Raise an error if NaN values are present

        """
        if nan_policy not in ("omit", "propagate", "raise"):
            raise ValueError(
                f"nan_policy must be 'omit', 'propagate', or 'raise', got {nan_policy!r}"
            )

        def _handle_nans(arr1, arr2, nan_policy):
            """Handle NaN values according to policy.

            For 1D arrays: masks out positions where either array has NaN.
            For 2D arrays: flattens and masks, then reshapes (for matrix perm).
            """
            arr1 = np.asarray(arr1)
            arr2 = np.asarray(arr2)

            # Check for NaN presence
            has_nan1 = np.any(np.isnan(arr1))
            has_nan2 = np.any(np.isnan(arr2))

            if not has_nan1 and not has_nan2:
                return arr1, arr2

            if nan_policy == "raise":
                if has_nan1 or has_nan2:
                    raise ValueError(
                        "Input contains NaN values. Use nan_policy='omit' to ignore them "
                        "or nan_policy='propagate' to allow NaN in results."
                    )
            elif nan_policy == "propagate":
                return arr1, arr2
            elif nan_policy == "omit":
                # For 2D matrix permutation, we can't easily mask individual elements
                # because the permutation test permutes rows/columns together.
                # Instead, we warn and use propagate for 2D.
                if arr1.ndim == 2:
                    import warnings

                    warnings.warn(
                        "NaN values detected in 2D matrix data. For perm_type='2d', "
                        "NaN handling is limited. Consider using perm_type='1d' or None, "
                        "or removing NaN values before calling similarity().",
                        UserWarning,
                    )
                    return arr1, arr2

                # For 1D: mask out NaN positions from both arrays
                mask = ~(np.isnan(arr1) | np.isnan(arr2))
                if not np.any(mask):
                    raise ValueError(
                        "All values are NaN after pairwise removal. Cannot compute similarity."
                    )
                return arr1[mask], arr2[mask]

            return arr1, arr2

        data1 = self.copy()
        if not isinstance(data, Adjacency):
            data2 = Adjacency(data)
        else:
            data2 = data.copy()

        if perm_type is None:
            n_permute = 0
            similarity_func = correlation_permutation
        elif perm_type == "1d":
            similarity_func = correlation_permutation
        elif perm_type == "2d":
            similarity_func = matrix_permutation
        else:
            raise ValueError("perm_type must be ['1d','2d', or None']")

        def _convert_data_similarity(
            data, perm_type=None, ignore_diagonal=ignore_diagonal
        ):
            """Helper function to convert data correctly"""
            if (perm_type is None) or (perm_type == "1d"):
                if ignore_diagonal and (not data.issymmetric):
                    d = data.squareform()
                    data = d[~np.eye(d.shape[0]).astype(bool)]
                else:
                    data = data.data
            elif perm_type == "2d":
                if not data.issymmetric:
                    raise TypeError(
                        f"data must be symmetric to do {perm_type} permutation"
                    )
                else:
                    data = data.squareform()
            else:
                raise ValueError("perm_type must be ['1d','2d', or None']")
            return data

        if self.is_single_matrix:
            if plot:
                plot_stacked_adjacency(self, data)
            arr1 = _convert_data_similarity(data1, perm_type=perm_type)
            arr2 = _convert_data_similarity(data2, perm_type=perm_type)
            arr1, arr2 = _handle_nans(arr1, arr2, nan_policy)
            return similarity_func(
                arr1,
                arr2,
                metric=metric,
                n_permute=n_permute,
                **kwargs,
            )
        else:
            if plot:
                _, a = plt.subplots(len(self))
                for i in a:
                    plot_stacked_adjacency(self, data, ax=i)
            results = []
            arr2_base = _convert_data_similarity(data2, perm_type=perm_type)
            for x in self:
                arr1 = _convert_data_similarity(x, perm_type=perm_type)
                arr1_clean, arr2_clean = _handle_nans(arr1, arr2_base, nan_policy)
                results.append(
                    similarity_func(
                        arr1_clean,
                        arr2_clean,
                        metric=metric,
                        n_permute=n_permute,
                        **kwargs,
                    )
                )
            return results

    def distance(self, metric="correlation", include_diag=False, **kwargs):
        """Calculate distance between images within an Adjacency() instance.

        Args:
            metric: (str) type of distance metric (can use any scikit learn or
                    scipy metric)
            include_diag: (bool) whether to include the main diagonal when
                    computing distances between adjacency matrices. Only applies
                    to symmetric matrices. Default False (consistent with how
                    symmetric matrices are stored without diagonal).

        Returns:
            dist: (Adjacency) Outputs a 2D distance matrix.

        """
        if include_diag and self.issymmetric:
            # Get square form and extract upper triangle WITH diagonal
            squares = self.squareform()
            if self.is_single_matrix:
                squares = [squares]
            # Extract upper triangle including diagonal for each matrix
            data_with_diag = []
            for sq in squares:
                mask = np.triu(
                    np.ones_like(sq, dtype=bool), k=0
                )  # k=0 includes diagonal
                data_with_diag.append(sq[mask])
            data = np.array(data_with_diag)
        else:
            data = self.data

        return Adjacency(
            pairwise_distances(data, metric=metric, **kwargs),
            matrix_type="distance",
        )

    def r_to_z(self):
        """Apply Fisher's r to z transformation to each element of the data
        object."""

        out = self.copy()
        out.data = fisher_r_to_z(out.data)
        return out

    def z_to_r(self):
        """Convert z score back into r value for each element of data object"""

        out = self.copy()
        out.data = fisher_z_to_r(out.data)
        return out

    def threshold(self, upper=None, lower=None, binarize=False):
        """Threshold Adjacency instance. Provide upper and lower values or
           percentages to perform two-sided thresholding. Binarize will return
           a mask image respecting thresholds if provided, otherwise respecting
           every non-zero value.

        Args:
            upper: (float or str) Upper cutoff for thresholding. If string
                    will interpret as percentile; can be None for one-sided
                    thresholding.
            lower: (float or str) Lower cutoff for thresholding. If string
                    will interpret as percentile; can be None for one-sided
                    thresholding.
            binarize (bool): return binarized image respecting thresholds if
                    provided, otherwise binarize on every non-zero value;
                    default False

        Returns:
            Adjacency: thresholded Adjacency instance

        """

        b = self.copy()
        if isinstance(upper, str) and upper[-1] == "%":
            upper = np.percentile(b.data, float(upper[:-1]))
        if isinstance(lower, str) and lower[-1] == "%":
            lower = np.percentile(b.data, float(lower[:-1]))

        if upper and lower:
            b.data[(b.data < upper) & (b.data > lower)] = 0
        elif upper:
            b.data[b.data < upper] = 0
        elif lower:
            b.data[b.data > lower] = 0
        if binarize:
            b.data[b.data != 0] = 1
        return b

    def to_graph(self):
        """Convert Adjacency into networkx graph.  only works on
        single_matrix for now."""

        if self.is_single_matrix:
            if self.matrix_type == "directed":
                G = nx.DiGraph(self.squareform())
            else:
                G = nx.Graph(self.squareform())
            if self.labels:
                labels = {x: y for x, y in zip(G.nodes, self.labels)}
                nx.relabel_nodes(G, labels, copy=False)
            return G
        else:
            raise NotImplementedError(
                "This function currently only works on single matrices."
            )

    def ttest(self, permutation=False, **kwargs):
        """Calculate ttest across samples.

        Args:
            permutation: (bool) Run ttest as permutation. Note this can be very slow.

        Returns:
            out: (dict) contains Adjacency instances of t values (or mean if
                 running permutation) and Adjacency instance of p values.

        """
        if self.is_single_matrix:
            raise ValueError("t-test cannot be run on single matrices.")

        if permutation:
            t = []
            p = []
            for i in range(self.data.shape[1]):
                stats = one_sample_permutation(self.data[:, i], **kwargs)
                t.append(stats["mean"])
                p.append(stats["p"])
            t = Adjacency(np.array(t))
            p = Adjacency(np.array(p))
        else:
            t = self.mean().copy()
            p = deepcopy(t)
            t.data, p.data = ttest_1samp(self.data, 0, 0)

        return {"t": t, "p": p}

    def plot_label_distance(self, labels=None, ax=None):
        """Create a violin plot indicating within and between label distance

        Args:
            labels (np.array):  numpy array of labels to plot

        Returns:
            f: violin plot handles

        """

        if not self.is_single_matrix:
            raise ValueError("This function only works on single adjacency matrices.")

        distance = pd.DataFrame(self.squareform())

        if labels is None:
            labels = np.array(deepcopy(self.labels))
        else:
            if len(labels) != distance.shape[0]:
                raise ValueError("Labels must be same length as distance matrix")

        frames = []
        for i in np.unique(labels):
            # Within-group distances (upper triangle of group i x group i)
            within_vals = distance.loc[labels == i, labels == i].values[
                np.triu_indices(sum(labels == i), k=1)
            ]
            tmp_w = pd.DataFrame(
                {"Distance": within_vals, "Type": "Within", "Group": i}
            )
            # Between-group distances (group i to all other groups)
            between_vals = distance.loc[labels == i, labels != i].values.flatten()
            tmp_b = pd.DataFrame(
                {"Distance": between_vals, "Type": "Between", "Group": i}
            )
            frames.extend([tmp_w, tmp_b])
        out = pd.concat(frames, ignore_index=True)
        f = sns.violinplot(
            x="Group",
            y="Distance",
            hue="Type",
            data=out,
            split=True,
            inner="quartile",
            palette={"Within": "lightskyblue", "Between": "red"},
            ax=ax,
        )
        f.set_ylabel("Average Distance")
        f.set_title("Average Group Distance")
        return

    def stats_label_distance(self, labels=None, n_permute=5000, n_jobs=-1):
        """Calculate permutation tests on within and between label distance.

        Args:
            labels (np.array):  numpy array of labels to plot
            n_permute (int): number of permutations to run (default=5000)

        Returns:
            dict:  dictionary of within and between group differences
                    and p-values

        """

        if not self.is_single_matrix:
            raise ValueError("This function only works on single adjacency matrices.")

        distance = pd.DataFrame(self.squareform())

        if labels is None:
            labels = deepcopy(self.labels)
        else:
            if len(labels) != distance.shape[0]:
                raise ValueError("Labels must be same length as distance matrix")

        frames = []
        for i in np.unique(labels):
            # Within-group distances (upper triangle of group i x group i)
            within_vals = distance.loc[labels == i, labels == i].values[
                np.triu_indices(sum(labels == i), k=1)
            ]
            tmp_w = pd.DataFrame(
                {"Distance": within_vals, "Type": "Within", "Group": i}
            )
            # Between-group distances (group i to all other groups)
            between_vals = distance.loc[labels == i, labels != i].values.flatten()
            tmp_b = pd.DataFrame(
                {"Distance": between_vals, "Type": "Between", "Group": i}
            )
            frames.extend([tmp_w, tmp_b])
        out = pd.concat(frames, ignore_index=True)
        stats = {}
        for i in np.unique(labels):
            # Within group test
            tmp1 = out.loc[(out["Group"] == i) & (out["Type"] == "Within"), "Distance"]
            tmp2 = out.loc[(out["Group"] == i) & (out["Type"] == "Between"), "Distance"]
            stats[str(i)] = two_sample_permutation(
                tmp1, tmp2, n_permute=n_permute, n_jobs=n_jobs
            )
        return stats

    def plot_silhouette(
        self, labels=None, ax=None, permutation_test=True, n_permute=5000, **kwargs
    ):
        """Create a silhouette plot"""
        distance = pd.DataFrame(self.squareform())

        if labels is None:
            labels = np.array(deepcopy(self.labels))
        else:
            if len(labels) != distance.shape[0]:
                raise ValueError("Labels must be same length as distance matrix")

        return plot_silhouette(
            distance,
            pd.Series(labels),
            ax=None,
            permutation_test=True,
            n_permute=5000,
            **kwargs,
        )

    def bootstrap(
        self,
        stat,
        n_samples=5000,
        save_boots=False,
        n_jobs=-1,
        random_state=None,
        percentiles=(2.5, 97.5),
    ):
        """Bootstrap statistics using efficient online algorithms.

        Uses memory-efficient bootstrap infrastructure with CPU parallelization.
        Supports simple aggregation statistics (mean, std, median, sum, min, max).

        Args:
            stat: (str) Statistic to bootstrap. Options:
                - Simple stats: 'mean', 'median', 'std', 'sum', 'min', 'max'
            n_samples: (int) Number of bootstrap iterations. Default: 5000
            save_boots: (bool) If True, store all bootstrap samples (memory intensive).
                       Default: False
            n_jobs: (int) Number of CPU cores for parallelization. -1 means all CPUs.
            random_state: (int, optional) Random seed for reproducibility
            percentiles: (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5)

        Returns:
            dict: Dictionary with keys: 'Z', 'p', 'mean', 'std', 'ci_lower', 'ci_upper'
                  (all Adjacency objects). If save_boots=True, also includes 'samples'.

        Examples:
            >>> # Simple aggregation
            >>> boot = adj.bootstrap(stat='mean', n_samples=1000)
            >>> assert 'mean' in boot
            >>> assert isinstance(boot['mean'], Adjacency)
        """
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_simple_cpu_parallel,
        )

        # Validate stat parameter
        SIMPLE_STATS = ["mean", "median", "std", "sum", "min", "max"]
        if stat not in SIMPLE_STATS:
            raise ValueError(
                f"Unsupported stat '{stat}'. Supported simple stats: {SIMPLE_STATS}."
            )

        # Get data as numpy array
        # Adjacency.data shape: (n_matrices, n_features)
        data = self.data  # Shape: (n_samples, n_features)

        # Route to bootstrap function
        result = _bootstrap_simple_cpu_parallel(
            data,
            method=stat,
            n_samples=n_samples,
            save_boots=save_boots,
            n_jobs=n_jobs,
            random_state=random_state,
            percentiles=percentiles,
        )

        # Convert result to Adjacency format
        return self._convert_bootstrap_results_to_adjacency(
            result, save_boots=save_boots
        )

    def _convert_bootstrap_results_to_adjacency(self, result, save_boots=False):
        """Convert bootstrap results dictionary to Adjacency format.

        Helper method to convert numpy arrays from bootstrap functions into
        Adjacency objects.

        Args:
            result: (dict) Result dictionary from bootstrap function with keys:
                    'mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', and optionally 'samples'
            save_boots: (bool) If True, include 'samples' key in output

        Returns:
            dict: Dictionary with Adjacency objects for each statistic
        """
        out = {}
        for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
            if key in result:
                # Convert numpy array to Adjacency
                # Result shape: (n_features,) for aggregated stats
                adj_data = result[key]
                if adj_data.ndim == 0:
                    # Scalar - convert to 1D array
                    adj_data = np.array([adj_data])
                elif adj_data.ndim == 1:
                    # Already 1D - reshape to (1, n_features) for Adjacency
                    adj_data = adj_data.reshape(1, -1)
                # adj_data is now (1, n_features)

                out[key] = Adjacency(
                    data=adj_data,
                    matrix_type=self.matrix_type + "_flat",
                )

        if save_boots and "samples" in result:
            # Samples shape: (n_samples, n_features)
            out["samples"] = result["samples"]

        return out

    def plot_mds(
        self,
        n_components=2,
        metric=True,
        labels=None,
        labels_color=None,
        cmap=plt.cm.hot_r,
        n_jobs=-1,
        view=(30, 20),
        figsize=[12, 8],
        ax=None,
        *args,
        **kwargs,
    ):
        """Plot Multidimensional Scaling

        Args:
            n_components: (int) Number of dimensions to project (can be 2 or 3)
            metric: (bool) Perform metric or non-metric dimensional scaling; default
            labels: (list) Can override labels stored in Adjacency Class
            labels_color: (str) list of colors for labels, if len(1) then make all same color
            n_jobs: (int) Number of parallel jobs
            view: (tuple) view for 3-Dimensional plot; default (30,20)

        """

        if self.matrix_type != "distance":
            raise ValueError("MDS only works on distance matrices.")
        if not self.is_single_matrix:
            raise ValueError("MDS only works on single matrices.")
        if n_components not in [2, 3]:
            raise ValueError("Cannot plot {0}-d image".format(n_components))
        if labels is not None:
            if len(labels) != self.n_nodes:
                raise ValueError(
                    "Make sure labels matches the same shape as Adjacency data"
                )
        else:
            labels = self.labels
        if labels_color is not None:
            if len(labels) == 0:
                raise ValueError(
                    "Make sure that Adjacency object has labels specified."
                )
            if len(labels) != len(labels_color):
                raise ValueError("Length of labels_color must match self.labels.")

        # Run MDS
        mds = MDS(
            *args,
            n_components=n_components,
            metric=metric,
            n_jobs=n_jobs,
            dissimilarity="precomputed",
            **kwargs,
        )
        proj = mds.fit_transform(self.squareform())

        # Create Plot
        if ax is None:  # Create axis
            fig = plt.figure(figsize=figsize)
            if n_components == 3:
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(*view)
            elif n_components == 2:
                ax = fig.add_subplot(111)

        # Plot dots
        if n_components == 3:
            ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], s=1, c="k")
        elif n_components == 2:
            ax.scatter(proj[:, 0], proj[:, 1], s=1, c="k")

        # Plot labels
        if labels_color is None:
            labels_color = ["black"] * len(labels)
        if n_components == 3:
            for (x, y, z), label, color in zip(proj, labels, labels_color):
                ax.text(
                    x,
                    y,
                    z,
                    label,
                    color="white",
                    bbox=dict(facecolor=color, alpha=1, boxstyle="round,pad=0.3"),
                )
        else:
            for (x, y), label, color in zip(proj, labels, labels_color):
                ax.text(
                    x,
                    y,
                    label,
                    color="white",  # color,
                    bbox=dict(facecolor=color, alpha=1, boxstyle="round,pad=0.3"),
                )

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    def distance_to_similarity(self, metric="correlation", beta=1):
        """Convert distance matrix to similarity matrix.

        Note: currently only implemented for correlation and euclidean.

        Args:
            metric: (str) Can only be correlation or euclidean
            beta: (float) parameter to scale exponential function (default: 1) for euclidean

        Returns:
            out: (Adjacency) Adjacency object

        """
        if self.matrix_type == "distance":
            if metric == "correlation":
                return Adjacency(1 - self.squareform(), matrix_type="similarity")
            elif metric == "euclidean":
                return Adjacency(
                    np.exp(-beta * self.squareform() / self.squareform().std()),
                    labels=self.labels,
                    matrix_type="similarity",
                )
            else:
                raise ValueError('metric can only be ["correlation","euclidean"]')
        else:
            raise ValueError("Matrix is not a distance matrix.")

    def cluster_summary(self, clusters=None, metric="mean", summary="within"):
        """This function provides summaries of clusters within Adjacency matrices.

        It can compute mean/median of within and between cluster values. Requires a
        list of cluster ids indicating the row/column of each cluster.

        Args:
            clusters: (list) list of cluster labels
            metric: (str) method to summarize mean or median. If 'None" then return all r values
            summary: (str) summarize within cluster or between clusters

        Returns:
            dict: (dict) within cluster means

        """
        if metric not in ["mean", "median", None]:
            raise ValueError("metric must be ['mean','median', None]")

        distance = pd.DataFrame(self.squareform())
        clusters = np.array(clusters)

        if len(clusters) != distance.shape[0]:
            raise ValueError("Cluster labels must be same length as distance matrix")

        out = {}
        for i in list(set(clusters)):
            if summary == "within":
                if metric == "mean":
                    out[i] = np.mean(
                        distance.loc[clusters == i, clusters == i].values[
                            np.triu_indices(sum(clusters == i), k=1)
                        ]
                    )
                elif metric == "median":
                    out[i] = np.median(
                        distance.loc[clusters == i, clusters == i].values[
                            np.triu_indices(sum(clusters == i), k=1)
                        ]
                    )
                elif metric is None:
                    out[i] = distance.loc[clusters == i, clusters == i].values[
                        np.triu_indices(sum(clusters == i), k=1)
                    ]
            elif summary == "between":
                if metric == "mean":
                    out[i] = distance.loc[clusters == i, clusters != i].mean().mean()
                elif metric == "median":
                    out[i] = (
                        distance.loc[clusters == i, clusters != i].median().median()
                    )
                elif metric is None:
                    out[i] = distance.loc[clusters == i, clusters != i]
        return out

    def regress(self, X, mode="ols", **kwargs):
        """Run a regression on an adjacency instance.
        You can decompose an adjacency instance with another adjacency instance.
        You can also decompose each pixel by passing a design_matrix instance.

        Args:
            X: Design matrix can be an Adjacency or DesignMatrix instance
            mode: type of regression (default: ols) - only 'ols' is currently supported

        Returns:
            stats: (dict) dictionary of stats outputs.
        """
        if mode != "ols":
            raise ValueError(
                "Only 'ols' mode is currently supported for Adjacency.regress()"
            )

        stats = {}
        if isinstance(X, Adjacency):
            if X.n_nodes != self.n_nodes:
                raise ValueError("Adjacency instances must be the same size.")
            # Convert to numpy arrays for regression
            X_data = X.data.T
            Y_data = self.data

            # Ensure Y is 2D
            if len(Y_data.shape) == 1:
                Y_data = Y_data[:, np.newaxis]

            # OLS regression: b = (X'X)^-1 X'Y
            # X_data shape: (n_features, n_regressors)
            # Y_data shape: (n_features, 1)
            # b shape: (n_regressors, 1)
            b = np.dot(np.linalg.pinv(X_data), Y_data)
            res = Y_data - np.dot(X_data, b)

            # Unbiased estimator of residual standard error: sqrt(RSS / df)
            # This is correct for both intercept and intercept-free models
            # See GH #287 for details on why np.std(res, ddof=p) is biased
            n, p = X_data.shape
            sigma = np.sqrt(np.sum(res**2, axis=0) / (n - p))
            if sigma.ndim == 0:
                sigma = sigma[np.newaxis]

            stderr = (
                np.sqrt(np.diag(np.linalg.pinv(np.dot(X_data.T, X_data))))[
                    :, np.newaxis
                ]
                * sigma[np.newaxis, :]
            )

            # t-statistics
            t = np.zeros_like(b)
            t[stderr > 1.0e-6] = b[stderr > 1.0e-6] / stderr[stderr > 1.0e-6]

            # p-values
            df = np.array([X_data.shape[0] - X_data.shape[1]] * t.shape[1])
            p = 2 * (1 - t_dist.cdf(np.abs(t), df))

            # Create Adjacency objects for each stat
            # For Adjacency X, b has shape (n_regressors, 1), so we need to reshape
            # to match Adjacency data format which expects (n_matrices, n_features)
            stats["beta"] = self.copy()
            stats["sigma"] = self.copy()
            stats["t"] = self.copy()
            stats["p"] = self.copy()
            stats["df"] = self.copy()
            stats["residual"] = self.copy()

            # Assign data - ensure 2D shape for Adjacency compatibility
            b_flat = b.squeeze()
            if b_flat.ndim == 0:
                b_flat = np.array([b_flat])
            stats["beta"].data = b_flat
            stats["sigma"].data = (
                stderr.squeeze().T if stderr.shape[0] > 1 else stderr.squeeze()
            )
            stats["t"].data = t.squeeze().T if t.shape[0] > 1 else t.squeeze()
            stats["p"].data = p.squeeze().T if p.shape[0] > 1 else p.squeeze()
            stats["df"].data = df.squeeze()
            stats["residual"].data = (
                res.squeeze().T if res.shape[1] == 1 else res.squeeze()
            )

        elif isinstance(X, DesignMatrix):
            if X.shape[0] != len(self):
                raise ValueError(
                    "Design matrix must have same number of observations as Adjacency"
                )
            # Convert Polars DesignMatrix to numpy
            X_data = X.to_numpy()
            Y_data = self.data

            # Ensure Y is 2D
            if len(Y_data.shape) == 1:
                Y_data = Y_data[:, np.newaxis]

            # OLS regression: b = (X'X)^-1 X'Y
            b = np.dot(np.linalg.pinv(X_data), Y_data)
            res = Y_data - np.dot(X_data, b)

            # Unbiased estimator of residual standard error: sqrt(RSS / df)
            # This is correct for both intercept and intercept-free models
            # See GH #287 for details on why np.std(res, ddof=p) is biased
            n, p = X_data.shape
            sigma = np.sqrt(np.sum(res**2, axis=0) / (n - p))
            if sigma.ndim == 0:
                sigma = sigma[np.newaxis]

            stderr = (
                np.sqrt(np.diag(np.linalg.pinv(np.dot(X_data.T, X_data))))[
                    :, np.newaxis
                ]
                * sigma[np.newaxis, :]
            )

            # t-statistics
            t = np.zeros_like(b)
            t[stderr > 1.0e-6] = b[stderr > 1.0e-6] / stderr[stderr > 1.0e-6]

            # p-values
            df = np.array([X_data.shape[0] - X_data.shape[1]] * t.shape[1])
            p = 2 * (1 - t_dist.cdf(np.abs(t), df))

            stats["beta"], stats["sigma"], stats["t"] = [self.copy() for _ in range(3)]
            stats["p"], stats["df"], stats["residual"] = [self.copy() for _ in range(3)]

            # Assign data - ensure proper shape for DesignMatrix case
            # For DesignMatrix, b has shape (n_regressors, n_features)
            # We need to reshape to (n_features, n_regressors) to match Adjacency format
            # where each row is a matrix (feature) and columns are regressors
            # But since we only have one regressor, we need (n_features,) shape
            # to match the original Adjacency data format
            if b.shape[0] == 1:
                # Single regressor case: b is (1, n_features), transpose to (n_features,)
                # Result is a single matrix of coefficients
                for key in ["beta", "sigma", "t", "p", "df", "residual"]:
                    stats[key].is_single_matrix = True
                stats["beta"].data = b.squeeze()
                stats["sigma"].data = stderr.squeeze()
                stats["t"].data = t.squeeze()
                stats["p"].data = p.squeeze()
                stats["df"].data = df.squeeze() if df.ndim > 0 else df
                stats["residual"].data = res.squeeze()
            else:
                # Multiple regressors: b is (n_regressors, n_features), transpose to (n_features, n_regressors)
                stats["beta"].data = b.T
                stats["sigma"].data = stderr.T
                stats["t"].data = t.T
                stats["p"].data = p.T
                stats["df"].data = df
                stats["residual"].data = res.T
        else:
            raise ValueError("X must be a DesignMatrix or Adjacency Instance.")

        return stats

    def social_relations_model(self, summarize_results=True, nan_replace=True):
        """Estimate the social relations model from a matrix for a round-robin design.

        X_{ij} = m + \\alpha_i + \\beta_j + g_{ij} + \\epsilon_{ijl}

        where X_{ij} is the score for person i rating person j, m is the group mean,
        \\alpha_i  is person i's actor effect, \\beta_j is person j's partner effect, g_{ij}
        is the relationship  effect and \\epsilon_{ijl} is the error in measure l  for actor i and partner j.

        This model is primarily concerned with partioning the variance of the various effects.

        Code is based on implementation presented in Chapter 8 of Kenny, Kashy, & Cook (2006).
        Tests replicate examples  presented in the book. Note, that this method assumes that
        actor scores are rows (lower triangle), while partner scores are columnns (upper triangle).
        The minimal sample size to estimate these effects is 4.

        Model Assumptions:
         - Social interactions are exclusively dyadic
         - People are randomly sampled from population
         - No order effects
         - The effects combine additively and relationships are linear

        In the future we might update the formulas and standard errors based on
        Bond and Lashley, 1996

        Args:
            self: (adjacency) can be a single matrix or many matrices for each group
            summarize_results: (bool) will provide a formatted summary of model results
            nan_replace: (bool) will replace nan values with row and column means

        Returns:
            estimated effects: (pd.Series/pd.DataFrame) All of the effects estimated using SRM
        """

        def mean_square_between(x1, x2=None, df="standard"):
            """Calculate between dyad variance"""

            if df == "standard":
                n = len(x1)
                df = n - 1
            elif df == "relationship":
                n = len(squareform(x1))
                df = ((n - 1) * (n - 2) / 2) - 1
            else:
                raise ValueError("df can only be ['standard', 'relationship']")
            if x2 is not None:
                return (
                    2
                    * np.nansum((((x1 + x2) / 2) - np.nanmean((x1 + x2) / 2)) ** 2)
                    / df
                )
            else:
                return np.nansum((x1 - np.nanmean(x1)) ** 2) / df

        def mean_square_within(x1, x2, df="standard"):
            """Calculate within dyad variance"""

            if df == "standard":
                n = len(x1)
                df = n
            elif df == "relationship":
                n = len(squareform(x1))
                df = (n - 1) * (n - 2) / 2
            else:
                raise ValueError("df can only be ['standard', 'relationship']")
            return np.nansum((x1 - x2) ** 2) / (2 * df)

        def estimate_person_effect(n, x1_mean, x2_mean, grand_mean):
            """Calculate effect for actor, partner, and relationship"""
            return (
                ((n - 1) ** 2 / (n * (n - 2))) * x1_mean
                + ((n - 1) / (n * (n - 2))) * x2_mean
                - ((n - 1) / (n - 2)) * grand_mean
            )

        def estimate_person_variance(x, ms_b, ms_w):
            """Calculate variance of a specific dyad member (e.g., actor, partner)"""
            n = len(x)
            return mean_square_between(x) - (ms_b / (2 * (n - 2))) - (ms_w / (2 * n))

        def estimate_srm(data):
            """Estimate Social Relations Model from a Single Matrix"""

            if not data.is_single_matrix:
                raise ValueError(
                    "This function only operates on single matrix Adjacency instances."
                )

            n = data.n_nodes
            if n < 4:
                raise ValueError(
                    "The Social Relations Model cannot be estimated when sample size is less than 4."
                )
            grand_mean = data.mean()
            dat = data.squareform().copy()
            np.fill_diagonal(dat, np.nan)
            actor_mean = np.nanmean(dat, axis=1)
            partner_mean = np.nanmean(dat, axis=0)

            a = estimate_person_effect(
                n, actor_mean, partner_mean, grand_mean
            )  # Actor effects
            b = estimate_person_effect(
                n, partner_mean, actor_mean, grand_mean
            )  # Partner effects

            # Relationship effects
            g = np.ones(dat.shape) * np.nan
            for i in range(n):
                for j in range(n):
                    if i != j:
                        g[i, j] = dat[i, j] - a[i] - b[j] - grand_mean

            # Estimate Variance
            x1 = g[np.tril_indices(n, k=-1)]
            x2 = g[np.triu_indices(n, k=1)]
            ms_b = mean_square_between(x1, x2, df="relationship")
            ms_w = mean_square_within(x1, x2, df="relationship")
            actor_variance = estimate_person_variance(a, ms_b, ms_w)
            partner_variance = estimate_person_variance(b, ms_b, ms_w)
            relationship_variance = (ms_b + ms_w) / 2
            dyadic_reciprocity_covariance = (ms_b - ms_w) / 2
            dyadic_reciprocity_correlation = (ms_b - ms_w) / (ms_b + ms_w)
            actor_partner_covariance = (
                (np.sum(a * b) / (n - 1)) - (ms_b / (2 * (n - 2))) + (ms_w / (2 * n))
            )
            actor_partner_correlation = actor_partner_covariance / (
                np.sqrt(actor_variance * partner_variance)
            )
            actor_reliability = actor_variance / (
                actor_variance
                + (relationship_variance / (n - 1))
                - (dyadic_reciprocity_covariance / ((n - 1) ** 2))
            )
            partner_reliability = partner_variance / (
                partner_variance
                + (relationship_variance / (n - 1))
                - (dyadic_reciprocity_covariance / ((n - 1) ** 2))
            )
            adjusted_dyadic_reciprocity_correlation = (
                actor_partner_correlation
                * np.sqrt(actor_reliability * partner_reliability)
            )
            total_variance = actor_variance + partner_variance + relationship_variance

            return pd.Series(
                {
                    "grand_mean": grand_mean,
                    "actor_effect": a,
                    "partner_effect": b,
                    "relationship_effect": g,
                    "actor_variance": actor_variance,
                    "partner_variance": partner_variance,
                    "relationship_variance": relationship_variance,
                    "actor_partner_covariance": actor_partner_covariance,
                    "actor_partner_correlation": actor_partner_correlation,
                    "dyadic_reciprocity_covariance": dyadic_reciprocity_covariance,
                    "dyadic_reciprocity_correlation": dyadic_reciprocity_correlation,
                    "adjusted_dyadic_reciprocity_correlation": adjusted_dyadic_reciprocity_correlation,
                    "actor_reliability": actor_reliability,
                    "partner_reliability": partner_reliability,
                    "total_variance": total_variance,
                }
            )

        def summarize_srm_results(results):
            """Summarize results of SRM"""

            def estimate_srm_stats(results, var_name, tailed=1):
                estimate = results[var_name].mean()
                standardized = (results[var_name] / results["total_variance"]).mean()
                se = results[var_name].std() / np.sqrt(len(results[var_name]))
                with np.errstate(invalid="ignore", divide="ignore"):
                    t = estimate / se
                if tailed == 1:
                    p = 1 - stats.t.cdf(t, len(results[var_name]) - 1)
                elif tailed == 2:
                    p = 2 * (1 - stats.t.cdf(t, len(results[var_name]) - 1))
                else:
                    raise ValueError("tailed can only be [1,2]")
                return (estimate, standardized, se, t, p)

            def print_srm_stats(results, var_name, tailed=1):
                estimate, standardized, se, t, p = estimate_srm_stats(
                    results, var_name, tailed
                )
                print(
                    f"{var_name:<40} {estimate:^10.2f}{standardized:^10.2f} {se:^10.2f} {t:^10.2f} {p:^10.4f}"
                )

            def print_single_group_srm_stats(results, var_name):
                estimate = results[var_name].mean()
                standardized = (results[var_name] / results["total_variance"]).mean()
                print(
                    f"{var_name:<40} {estimate:^10.2f}{standardized:^10.2f} {np.nan:^10.2f} {np.nan:^10.2f} {np.nan:^10.4f}"
                )

            def print_srm_covariances(results, var_name):
                estimate, _, se, t, p = estimate_srm_stats(
                    results, f"{var_name}_covariance", tailed=2
                )
                standardized = results[f"{var_name}_correlation"].mean()
                print(
                    f"{var_name:<40} {estimate:^10.2f}{standardized:^10.2f} {se:^10.2f} {t:^10.2f} {p:^10.4f}"
                )

            def print_single_srm_covariances(results, var_name):
                estimate = results[f"{var_name}_covariance"].mean()
                standardized = results[f"{var_name}_correlation"].mean()
                print(
                    f"{var_name:<40} {estimate:^10.2f}{standardized:^10.2f} {np.nan:^10.2f} {np.nan:^10.2f} {np.nan:^10.4f}"
                )

            if isinstance(results, pd.Series):
                n_groups = 1
                group_size = results["actor_effect"].shape[0]
            elif isinstance(results, pd.DataFrame):
                n_groups = len(results)
                group_size = np.mean([x.shape for x in results["actor_effect"]])

            print("Social Relations Model: Results")
            print("\n")
            print(f"Number of Groups: {n_groups:<20}")
            print(f"Average Group Size: {group_size:<20}")
            print("\n")
            print(
                f"{'':<40} {'Estimate':<10} {'Standardized':<10} {'se':<10} {'t':<10} {'p':<10}"
            )
            if isinstance(results, pd.Series):
                print_single_group_srm_stats(results, "actor_variance")
                print_single_group_srm_stats(results, "partner_variance")
                print_single_group_srm_stats(results, "relationship_variance")
                print_single_srm_covariances(results, "actor_partner")
                print_single_srm_covariances(results, "dyadic_reciprocity")
            elif isinstance(results, pd.DataFrame):
                print_srm_stats(results, "actor_variance")
                print_srm_stats(results, "partner_variance")
                print_srm_stats(results, "relationship_variance")
                print_srm_covariances(results, "actor_partner")
                print_srm_covariances(results, "dyadic_reciprocity")
            print("\n")
            print(
                f"{'Actor Reliability':<20} {results['actor_reliability'].mean():^20.2f}"
            )
            print(
                f"{'Partner Reliability':<20} {results['partner_reliability'].mean():^20.2f}"
            )
            print("\n")

        def replace_missing(data):
            """Replace missing data with row/column means and return new data and missing coordinates"""

            def fix_missing(data):
                X = data.squareform().copy()
                x, y = np.where(np.isnan(X))
                for i, j in zip(x, y):
                    if i != j:
                        X[i, j] = (np.nanmean(X[i, :]) + np.nanmean(X[:, j])) / 2
                X = Adjacency(X, matrix_type=data.matrix_type)
                return (X, (x, y))

            if data.is_single_matrix:
                X, coord = fix_missing(data)
            else:
                X = []
                coord = []
                for d in data:
                    m, c = fix_missing(d)
                    X.append(m)
                    coord.append(c)
                X = Adjacency(X)
            return (X, coord)

        if nan_replace:
            data, _ = replace_missing(self)
        else:
            data = self.copy()

        if self.is_single_matrix:
            results = estimate_srm(data)
        else:
            results = pd.DataFrame([estimate_srm(x) for x in data])

        if summarize_results:
            summarize_srm_results(results)

        return results

    def generate_permutations(self, n_perm, random_state=None):
        """
        Generate n_perm permutated versions of Adjacency in a lazy fashion. Useful for iterating against.


        Args:
            n_perm (int): number of permutations
            random_state (int, np.random.seed, optional): random seed for reproducibility. Defaults to None.

        Examples:
            >>> for perm in adj.generate_permutations(1000):
            >>>     out = neural_distance_mat.similarity(perm)
            >>>     ...

        Yields:
            Adjacency: permuted version of self
        """

        random_state = check_random_state(random_state)

        for _ in range(n_perm):
            # Get squareform as numpy array (no pandas conversion needed)
            dat = self.squareform()
            # Generate random permutation indices
            permuted_idx = random_state.choice(
                dat.shape[0], size=dat.shape[0], replace=False
            )
            # Permute rows and columns using numpy advanced indexing (faster than pandas)
            dat = dat[np.ix_(permuted_idx, permuted_idx)]
            yield Adjacency(dat)
