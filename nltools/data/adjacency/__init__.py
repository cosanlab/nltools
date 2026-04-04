"""
This data class is for working with similarity/dissimilarity matrices
"""

import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from h5py import File as h5File
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances

from nltools.io import is_h5_path
from nltools.utils import (
    all_same,
    attempt_to_import,
    concatenate,
)

from .utils import (
    apply_stat,
    import_single_data,
    perform_arithmetic,
    test_is_single_matrix,
)

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
                    (
                        data_tmp,
                        issymmetric_tmp,
                        matrix_type_tmp,
                        _,
                    ) = import_single_data(d, matrix_type=matrix_type)
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
                        ) = import_single_data(self.data, matrix_type=self.matrix_type)

                        return

            # CSV or array/dataframe
            else:
                (
                    self.data,
                    self.issymmetric,
                    self.matrix_type,
                    self.is_single_matrix,
                ) = import_single_data(data, matrix_type=matrix_type)

        # CSV or array/dataframe
        else:
            (
                self.data,
                self.issymmetric,
                self.matrix_type,
                self.is_single_matrix,
            ) = import_single_data(data, matrix_type=matrix_type)

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

    # ── Dunders (alphabetical) ──────────────────────────────────────────

    def __add__(self, y):
        return perform_arithmetic(self, y, np.add, "add")

    def __getitem__(self, index):
        new = self.copy()
        if isinstance(index, (int, np.integer)):
            new.data = np.array(self.data[index, :]).squeeze()
            new.is_single_matrix = True
        else:
            new.data = np.array(self.data[index, :]).squeeze()
            new.is_single_matrix = test_is_single_matrix(new.data)
        if not self.Y.empty:
            new.Y = self.Y.iloc[index]
        return new

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]

    def __len__(self):
        if self.is_single_matrix:
            return 1
        else:
            return self.data.shape[0]

    def __mul__(self, y):
        return perform_arithmetic(self, y, np.multiply, "multiply")

    def __radd__(self, y):
        return perform_arithmetic(self, y, np.add, "add", reverse=True)

    def __repr__(self):
        return "%s.%s(shape=%s, Y=%s, is_symmetric=%s, matrix_type=%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape,
            self.Y.shape,
            self.issymmetric,
            self.matrix_type,
        )

    def __rmul__(self, y):
        return perform_arithmetic(self, y, np.multiply, "multiply", reverse=True)

    def __rsub__(self, y):
        return perform_arithmetic(self, y, np.subtract, "subtract", reverse=True)

    def __sub__(self, y):
        return perform_arithmetic(self, y, np.subtract, "subtract")

    def __truediv__(self, y):
        return perform_arithmetic(self, y, np.divide, "divide")

    # ── Properties (alphabetical) ───────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """Check if Adjacency object is empty.

        Returns:
            bool: True if the adjacency matrix is empty, False otherwise.
        """
        return self.matrix_type == "empty"

    @property
    def n_nodes(self):
        """Return the number of nodes in the adjacency matrix.

        Returns:
            int: Number of nodes (n) for an (n, n) matrix.
        """
        return self.shape[-1]

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

    # ── Public methods (alphabetical) ───────────────────────────────────

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
        from .modeling import bootstrap

        return bootstrap(
            self, stat, n_samples, save_boots, n_jobs, random_state, percentiles
        )

    def cluster_summary(self, clusters=None, metric="mean", summary="within"):
        """Provide summaries of clusters within Adjacency matrices.

        Computes mean/median of within and between cluster values. Requires a
        list of cluster ids indicating the row/column of each cluster.

        Args:
            clusters: (list) list of cluster labels
            metric: (str) method to summarize mean or median. If 'None" then return all r values
            summary: (str) summarize within cluster or between clusters

        Returns:
            dict: within cluster means

        """
        from .stats import cluster_summary

        return cluster_summary(self, clusters, metric, summary)

    def copy(self):
        """Create a copy of Adjacency object."""
        return deepcopy(self)

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

    def generate_permutations(self, n_perm, random_state=None):
        """
        Generate n_perm permutated versions of Adjacency in a lazy fashion.

        Args:
            n_perm (int): number of permutations
            random_state (int, np.random.seed, optional): random seed for reproducibility.

        Examples:
            >>> for perm in adj.generate_permutations(1000):
            >>>     out = neural_distance_mat.similarity(perm)
            >>>     ...

        Yields:
            Adjacency: permuted version of self
        """
        from .modeling import generate_permutations

        return generate_permutations(self, n_perm, random_state)

    def mean(self, axis=0):
        """Calculate mean of Adjacency.

        Args:
            axis: Calculate mean over matrices (0) or upper triangle (1).

        Returns:
            float if single matrix, Adjacency if axis=0, np.array if axis=1.
        """
        return apply_stat(self, np.nanmean, axis)

    def median(self, axis=0):
        """Calculate median of Adjacency.

        Args:
            axis: Calculate median over matrices (0) or upper triangle (1).

        Returns:
            float if single matrix, Adjacency if axis=0, np.array if axis=1.
        """
        return apply_stat(self, np.nanmedian, axis)

    def plot(self, limit=3, axes=None, *args, **kwargs):
        """Create Heatmap of Adjacency Matrix

        Can pass in any sns.heatmap argument

        Args:
            limit: (int) number of heatmaps to plot if object contains multiple adjacencies (default: 3)
            axes: matplotlib axis handle
        """
        from .plotting import plot

        return plot(self, limit, axes, *args, **kwargs)

    def plot_label_distance(self, labels=None, ax=None):
        """Create a violin plot indicating within and between label distance

        Args:
            labels (np.array):  numpy array of labels to plot

        Returns:
            f: violin plot handles

        """
        from .stats import plot_label_distance

        return plot_label_distance(self, labels, ax)

    def plot_mds(
        self,
        n_components=2,
        metric=True,
        labels=None,
        labels_color=None,
        cmap=None,
        n_jobs=-1,
        view=(30, 20),
        figsize=None,
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
            cmap: colormap instance (default: plt.cm.hot_r)
            n_jobs: (int) Number of parallel jobs
            view: (tuple) view for 3-Dimensional plot; default (30,20)
            figsize: (list) figure size; default [12, 8]
            ax: matplotlib axis handle

        """
        from .plotting import plot_mds

        return plot_mds(
            self,
            n_components,
            metric,
            labels,
            labels_color,
            cmap,
            n_jobs,
            view,
            figsize,
            ax,
            *args,
            **kwargs,
        )

    def plot_silhouette(
        self, labels=None, ax=None, permutation_test=True, n_permute=5000, **kwargs
    ):
        """Create a silhouette plot"""
        from .stats import plot_silhouette

        return plot_silhouette(self, labels, ax, permutation_test, n_permute, **kwargs)

    def r_to_z(self):
        """Apply Fisher's r to z transformation to each element of the data
        object."""
        from .stats import r_to_z

        return r_to_z(self)

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
        from .modeling import regress

        return regress(self, X, mode, **kwargs)

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
        from .stats import similarity

        return similarity(
            self,
            data,
            plot=plot,
            perm_type=perm_type,
            n_permute=n_permute,
            metric=metric,
            ignore_diagonal=ignore_diagonal,
            nan_policy=nan_policy,
            **kwargs,
        )

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
        from .modeling import social_relations_model

        return social_relations_model(self, summarize_results, nan_replace)

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

    def stats_label_distance(self, labels=None, n_permute=5000, n_jobs=-1):
        """Calculate permutation tests on within and between label distance.

        Args:
            labels (np.array):  numpy array of labels to plot
            n_permute (int): number of permutations to run (default=5000)

        Returns:
            dict:  dictionary of within and between group differences
                    and p-values

        """
        from .stats import stats_label_distance

        return stats_label_distance(self, labels, n_permute, n_jobs)

    def std(self, axis=0):
        """Calculate standard deviation of Adjacency.

        Args:
            axis: Calculate std over matrices (0) or upper triangle (1).

        Returns:
            float if single matrix, Adjacency if axis=0, np.array if axis=1.
        """
        return apply_stat(self, np.nanstd, axis)

    def sum(self, axis=0):
        """Calculate sum of Adjacency.

        Args:
            axis: Calculate sum over matrices (0) or upper triangle (1).

        Returns:
            float if single matrix, Adjacency if axis=0, np.array if axis=1.
        """
        return apply_stat(self, np.nansum, axis)

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
        from .stats import threshold

        return threshold(self, upper, lower, binarize)

    def to_graph(self):
        """Convert Adjacency into networkx graph.  only works on
        single_matrix for now."""
        from .io import to_graph

        return to_graph(self)

    def to_square(self):
        """Convert adjacency back to square matrix format.

        This is an alias for :meth:`squareform`.

        Returns:
            np.ndarray or list: Square matrix representation. Returns a list
            of matrices if this object contains multiple adjacency matrices.
        """
        return self.squareform()

    def ttest(self, permutation=False, **kwargs):
        """Calculate ttest across samples.

        Args:
            permutation: (bool) Run ttest as permutation. Note this can be very slow.

        Returns:
            out: (dict) contains Adjacency instances of t values (or mean if
                 running permutation) and Adjacency instance of p values.

        """
        from .stats import ttest

        return ttest(self, permutation, **kwargs)

    def write(self, file_name, method="long"):
        """Write out Adjacency object to csv file.

        Args:
            file_name (str):  name of file name to write
            method (str):     method to write out data ['long','square']

        """
        from .io import write

        return write(self, file_name, method)

    def z_to_r(self):
        """Convert z score back into r value for each element of data object"""
        from .stats import z_to_r

        return z_to_r(self)
