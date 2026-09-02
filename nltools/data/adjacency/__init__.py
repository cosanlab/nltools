"""Provide data structures for working with similarity and dissimilarity matrices."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances

from nltools.data.braindata.utils import _polars_row_select
from nltools.data.braindata.validation import validate_frame
from nltools.io import is_h5_path
from nltools.utils import (
    all_same,
    attempt_to_import,
    concatenate,
)

from .spatial import SpatialScale
from .utils import (
    apply_stat,
    import_single_data,
    perform_arithmetic,
    test_is_single_matrix,
)

# Optional dependencies
nx = attempt_to_import("networkx", "nx")

MAX_INT = np.iinfo(np.int32).max


class Adjacency:
    """Represent adjacency matrices in vectorized form.

    Adjacency is a class to represent Adjacency matrices as a vector rather
    than a 2-dimensional matrix. This makes it easier to perform data
    manipulation and analyses.

    Args:
        data (np.ndarray | pd.DataFrame | pl.DataFrame | str | Path | list): A square
            matrix, a flattened vector, a `.csv`/`.h5` path, or a list of
            matrices/`Adjacency` instances/`.csv` paths to stack.
        Y (pd.DataFrame | pl.DataFrame, optional): Training labels, one row per matrix.
        matrix_type (str, optional): Type of matrix. One of `'distance'`, `'similarity'`,
            `'directed'`, `'distance_flat'`, `'similarity_flat'`, `'directed_flat'`.
        labels (list, optional): Node labels, one per row/column.
        spatial_scale (SpatialScale, optional): Spatial-scale metadata linking rows/
            columns to a brain parcellation, enabling projection back into brain space.

    Attributes:
        data (np.ndarray): Vectorized matrix values. Shape `(vector_length,)` for a
            single matrix or `(n_matrices, vector_length)` for a stack; symmetric
            matrices store only the upper triangle without the diagonal.
        matrix_type (str): One of `'distance'`, `'similarity'`, `'directed'`, or
            `'empty'` (the `'_flat'` input variants are normalized to their base type).
        is_single_matrix (bool): True when the instance holds exactly one matrix.
        issymmetric (bool): True for distance/similarity matrices, False for directed.
        labels (list): Node labels (empty list when none were given).
        spatial_scale (SpatialScale | None): Parcellation provenance for a stack
            produced by `BrainData.distance`; None otherwise.
        Y (pl.DataFrame): Training labels as a polars DataFrame (possibly empty).
        is_empty (bool): True if the instance holds no data.
        n_nodes (int): Number of nodes `n` for an `(n, n)` matrix.
        shape (tuple): Logical shape — `(n_nodes, n_nodes)` for a single matrix,
            `(n_matrices, n_nodes, n_nodes)` for a stack, `(0, 0)` when empty.
        vector_shape (tuple): Shape of the internal vectorized storage (`data.shape`).
    """

    def __init__(
        self,
        data=None,
        *,
        Y=None,
        matrix_type=None,
        labels=None,
        spatial_scale: SpatialScale | None = None,
    ):
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
                self.is_single_matrix = False
                # Return early (mirroring the h5/legacy branches) so the
                # concatenated Y and labels are not clobbered by the None
                # constructor params below.
                self.labels = deepcopy(tmp.labels)
                self.spatial_scale = spatial_scale
                return

            # File paths or array/dataframes
            # NOTE: We don't support list of hdf5 filepaths! Only .csvs
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
                from nltools.io.h5 import (
                    _read_polars_frame,
                    _require_h5,
                    is_legacy_adjacency_h5,
                    load_legacy_adjacency_h5,
                )

                _require_h5()
                import h5py
                from h5py import File as h5File

                if is_legacy_adjacency_h5(to_load):
                    legacy = load_legacy_adjacency_h5(to_load, matrix_type=matrix_type)
                    (
                        self.data,
                        self.issymmetric,
                        self.matrix_type,
                        self.is_single_matrix,
                    ) = import_single_data(
                        legacy["data"], matrix_type=legacy["matrix_type"]
                    )
                    self.Y = legacy["Y"]
                    self.labels = legacy["labels"]
                    return

                with h5File(to_load, "r") as f:
                    self.data = np.array(f["data"])
                    self.matrix_type = f["matrix_type"][()].decode()
                    self.is_single_matrix = f["is_single_matrix"][()]
                    self.issymmetric = f["issymmetric"][()]
                    self.Y = _read_polars_frame(f, "Y")
                    labels_ds = f["labels"]
                    if len(labels_ds) == 0:
                        self.labels = []
                    elif h5py.check_string_dtype(labels_ds.dtype) is not None:
                        self.labels = list(labels_ds.asstr())
                    else:
                        self.labels = list(labels_ds)

                return

            # CSV or array/dataframe
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

        # Setup Y dataframe — setter validates + converts to polars
        self.Y = Y

        # Ensure consistency
        if (
            not self.Y.is_empty()
            and not self.is_single_matrix
            and self.data.shape[0] != self.Y.shape[0]
        ):
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
                    self.labels = list(labels) * len(self)
                else:
                    if np.all(np.array([len(x) for x in labels]) != self.n_nodes):
                        raise ValueError(
                            "All lists of labels must be same length as shape of data."
                        )
                    self.labels = deepcopy(labels)
        else:
            raise TypeError("Make sure labels is a list or numpy array.")

        # Optional spatial-scale provenance. Only valid on a stack where the
        # number of matrices equals the number of roi_labels.
        if spatial_scale is not None:
            if self.is_empty or self.is_single_matrix:
                raise ValueError(
                    "spatial_scale requires a stack of Adjacency matrices "
                    "(one per parcel/searchlight); got a single matrix."
                )
            if len(spatial_scale.roi_labels) != len(self):
                raise ValueError(
                    f"spatial_scale.roi_labels length "
                    f"({len(spatial_scale.roi_labels)}) does not match the "
                    f"number of matrices in the stack ({len(self)})."
                )
        self.spatial_scale: SpatialScale | None = spatial_scale

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
        if not self.Y.is_empty():
            new.Y = _polars_row_select(self.Y, index)
        # Spatial-scale provenance: preserve when the result is still a stack
        # (subset the roi_labels to match); drop when collapsed to a single
        # matrix — a single RDM has no per-parcel structure to back-project.
        if self.spatial_scale is not None:
            if new.is_single_matrix:
                new.spatial_scale = None
            else:
                ss = self.spatial_scale
                new.spatial_scale = SpatialScale(
                    atlas=ss.atlas,
                    roi_labels=ss.roi_labels[index],
                    source_mask=ss.source_mask,
                    kind=ss.kind,
                )
        return new

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]

    def __len__(self):
        if self.is_single_matrix:
            return 1
        return self.data.shape[0]

    def __mul__(self, y):
        return perform_arithmetic(self, y, np.multiply, "multiply")

    def __radd__(self, y):
        return perform_arithmetic(self, y, np.add, "add", reverse=True)

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}(shape={self.shape}, Y={self.Y.shape}, is_symmetric={self.issymmetric}, matrix_type={self.matrix_type})"

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
    def Y(self) -> pl.DataFrame:
        """Training labels as a polars DataFrame (possibly empty)."""
        return self._Y

    @Y.setter
    def Y(self, value) -> None:
        self._Y = validate_frame(value, frame_type="Y")

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
            tuple: `(n_nodes, n_nodes)` for a single matrix, `(n_matrices, n_nodes,
                n_nodes)` for stacked matrices, `(0, 0)` when empty.

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
        return (len(self), n_nodes, n_nodes)

    @property
    def vector_shape(self):
        """Return shape of internal vectorized representation.

        Returns:
            tuple: `(vector_length,)` for a single matrix, `(n_matrices,
                vector_length)` for stacked matrices.

        Note:
            This is the raw shape of the internal data storage.
            Use `.shape` for the logical (n_nodes, n_nodes) shape.
        """
        return self.data.shape

    # ── Public methods (alphabetical) ───────────────────────────────────

    def append(self, data):
        """Append data to an Adjacency instance.

        Args:
            data (Adjacency): Adjacency instance to append.

        Returns:
            Adjacency: New appended Adjacency instance.
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
            if not out.Y.is_empty():
                out.Y = pl.concat([self.Y, data.Y], how="vertical_relaxed")

        return out

    def bootstrap(
        self,
        stat,
        *,
        n_samples=5000,
        save_boots=False,
        percentiles=(2.5, 97.5),
        tail=2,
        n_jobs=-1,
        random_state=None,
        progress_bar: bool = False,
    ):
        """Bootstrap statistics using efficient online algorithms.

        Uses memory-efficient bootstrap infrastructure with CPU parallelization.
        Supports simple aggregation statistics (mean, std, median, sum, min, max).

        Args:
            stat (str): Statistic to bootstrap: `'mean'`, `'median'`, `'std'`, `'sum'`,
                `'min'`, or `'max'`.
            n_samples (int): Number of bootstrap iterations. Default 5000.
            save_boots (bool): If True, store all bootstrap samples (memory intensive).
                Default False.
            percentiles (tuple): Percentiles for confidence intervals. Default (2.5, 97.5).
            tail (int | str): `2`/`'two'` for two-tailed (default); `1`/`'one'` for
                one-tailed (statistic > 0; negate the data for the other direction).
            n_jobs (int): Number of CPU cores for parallelization. -1 means all CPUs.
            random_state (int, optional): Random seed for reproducibility.
            progress_bar (bool): If True, show a progress bar. Default False.

        Returns:
            dict: Dictionary with keys `'Z'`, `'p'`, `'mean'`, `'std'`, `'ci_lower'`,
                `'ci_upper'` (all Adjacency objects). If `save_boots=True`, also
                includes `'samples'`.

        Examples:
            ```python
            boot = adj.bootstrap(stat="mean", n_samples=1000)
            boot["mean"]  # → Adjacency
            ```
        """
        from .modeling import bootstrap

        return bootstrap(
            self,
            stat,
            n_samples=n_samples,
            save_boots=save_boots,
            percentiles=percentiles,
            tail=tail,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )

    def cluster_summary(self, *, clusters=None, summary="mean", scope="within"):
        """Provide summaries of clusters within Adjacency matrices.

        Computes mean/median of within and between cluster values. Requires a
        list of cluster ids indicating the row/column of each cluster.

        Args:
            clusters (list): Cluster label for each row/column.
            summary (str | None): Central tendency, `'mean'` or `'median'`. If None,
                return all values instead of a summary.
            scope (str): Summarize `'within'` cluster or `'between'` clusters.

        Returns:
            dict: Per-cluster summaries keyed by cluster label.
        """
        from .stats import cluster_summary

        return cluster_summary(self, clusters=clusters, summary=summary, scope=scope)

    def copy(self):
        """Create a copy of Adjacency object."""
        return deepcopy(self)

    def distance(  # nosemgrep: kwargs-internal-forwarding  # forwards to sklearn.metrics.pairwise_distances
        self, metric="correlation", include_diag=False, **kwargs
    ):
        """Calculate distance between images within an Adjacency() instance.

        Args:
            metric (str): Distance metric; any metric accepted by
                `sklearn.metrics.pairwise_distances` (scikit-learn or scipy).
            include_diag (bool): Whether to include the main diagonal when
                computing distances between adjacency matrices. Only applies
                to symmetric matrices. Default False (consistent with how
                symmetric matrices are stored without the diagonal).
            **kwargs (dict): Forwarded to `sklearn.metrics.pairwise_distances`.

        Returns:
            Adjacency: A 2D distance matrix.
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

        Currently only implemented for the 'correlation' and 'euclidean' metrics.

        Args:
            metric (str): Either 'correlation' or 'euclidean'.
            beta (float): Scale parameter of the exponential used for 'euclidean' (default: 1).

        Returns:
            Adjacency: The converted similarity matrix.
        """
        if self.matrix_type == "distance":
            if metric == "correlation":
                return Adjacency(1 - self.squareform(), matrix_type="similarity")
            if metric == "euclidean":
                return Adjacency(
                    np.exp(-beta * self.squareform() / self.squareform().std()),
                    labels=self.labels,
                    matrix_type="similarity",
                )
            raise ValueError('metric can only be ["correlation","euclidean"]')
        raise ValueError("Matrix is not a distance matrix.")

    def generate_permutations(self, n_permute, random_state=None):
        """Generate permuted versions of an Adjacency instance lazily.

        Args:
            n_permute (int): Number of permutations.
            random_state (int | np.random.RandomState, optional): Random seed for
                reproducibility.

        Yields:
            Adjacency: Permuted version of self.

        Examples:
            ```python
            for perm in adj.generate_permutations(1000):
                out = neural_distance_mat.similarity(perm)
            ```
        """
        from .modeling import generate_permutations

        return generate_permutations(self, n_permute, random_state)

    def mean(self, axis=0):
        """Calculate mean of Adjacency.

        Args:
            axis (int): Calculate mean over matrices (0) or upper triangle (1).

        Returns:
            float | Adjacency | np.ndarray: A float for a single matrix; an
                Adjacency when `axis=0`; an array when `axis=1`.
        """
        return apply_stat(self, np.nanmean, axis)

    def median(self, axis=0):
        """Calculate median of Adjacency.

        Args:
            axis (int): Calculate median over matrices (0) or upper triangle (1).

        Returns:
            float | Adjacency | np.ndarray: A float for a single matrix; an
                Adjacency when `axis=0`; an array when `axis=1`.
        """
        return apply_stat(self, np.nanmedian, axis)

    def plot(  # nosemgrep: kwargs-internal-forwarding  # forwards to matplotlib via plot_adjacency
        self, limit=3, axes=None, *args, **kwargs
    ):
        """Create a heatmap of an Adjacency matrix.

        Args:
            limit (int): Number of heatmaps to plot if the object contains multiple
                matrices. Default 3.
            axes (matplotlib.axes.Axes, optional): Axis to draw on (single matrix only).
            *args (tuple): Forwarded positionally to `seaborn.heatmap`.
            **kwargs (dict): Forwarded to `seaborn.heatmap`.
        """
        from .plotting import plot_adjacency

        return plot_adjacency(self, limit, axes, *args, **kwargs)

    def plot_label_distance(self, labels=None, ax=None):
        """Create a violin plot of within- and between-label distances.

        Args:
            labels (np.ndarray, optional): Group label per node; defaults to the
                stored labels.
            ax (matplotlib.axes.Axes, optional): Axis to draw on.
        """
        from .stats import plot_label_distance

        return plot_label_distance(self, labels, ax)

    def plot_mds(  # nosemgrep: kwargs-internal-forwarding  # forwards to matplotlib via plotting.plot_mds
        self,
        *,
        n_components=2,
        metric_mds=True,
        labels=None,
        labels_color=None,
        cmap=None,
        view=(30, 20),
        figsize=None,
        ax=None,
        n_jobs=-1,
        **kwargs,
    ):
        """Plot multidimensional scaling.

        Args:
            n_components (int): Number of dimensions to project (2 or 3).
            metric_mds (bool): Perform metric (True) or non-metric (False) scaling.
                Default True.
            labels (list, optional): Overrides the labels stored on the instance.
            labels_color (list, optional): One color per label.
            cmap (matplotlib.colors.Colormap, optional): Colormap. Default `plt.cm.hot_r`.
            view (tuple): Elevation/azimuth for a 3-D plot. Default (30, 20).
            figsize (list): Figure size. Default [12, 8].
            ax (matplotlib.axes.Axes, optional): Axis to draw on.
            n_jobs (int): Number of parallel jobs.
            **kwargs (dict): Forwarded to `sklearn.manifold.MDS`.
        """
        from .plotting import plot_mds

        return plot_mds(
            self,
            n_components=n_components,
            metric_mds=metric_mds,
            labels=labels,
            labels_color=labels_color,
            cmap=cmap,
            view=view,
            figsize=figsize,
            ax=ax,
            n_jobs=n_jobs,
            **kwargs,
        )

    def plot_silhouette(
        self,
        *,
        labels=None,
        ax=None,
        permutation_test=True,
        n_permute=5000,
        colors=None,
        figsize=(6, 4),
    ):
        """Create a silhouette plot.

        Args:
            labels (np.ndarray, optional): Cluster/group label per node (overrides
                stored labels).
            ax (matplotlib.axes.Axes, optional): Axis to draw on.
            permutation_test (bool): Whether to run a permutation test. Default True.
            n_permute (int): Number of permutations for the test. Default 5000.
            colors (list, optional): RGB triplets, one per cluster. Default: seaborn
                `'hls'` palette.
            figsize (tuple): Figure size. Default (6, 4).

        Returns:
            pl.DataFrame: Columns `label` and `mean_silhouette`, plus `p` when
                `permutation_test=True`.
        """
        from .stats import plot_silhouette

        return plot_silhouette(
            self,
            labels=labels,
            ax=ax,
            permutation_test=permutation_test,
            n_permute=n_permute,
            colors=colors,
            figsize=figsize,
        )

    def r_to_z(self):
        """Apply Fisher's r-to-z transformation to each data element."""
        from .stats import r_to_z

        return r_to_z(self)

    def regress(self, X, method="ols", tail=2):
        """Run a regression on an adjacency instance.

        Pass an `Adjacency` as `X` to decompose this matrix with other matrices, or a
        `DesignMatrix` to regress each cell across a stack of matrices.

        Args:
            X (Adjacency | DesignMatrix): Design matrix.
            method (str): Type of regression; only `'ols'` is currently supported.
            tail (int | str): `2`/`'two'` (two-tailed, default) or `1`/`'one'`
                (one-tailed: beta > 0; negate a regressor for the other direction).

        Returns:
            dict: Adjacency instances keyed `'beta'`, `'sigma'`, `'t'`, `'p'`, `'df'`,
                `'residual'`.
        """
        from .modeling import regress

        return regress(self, X, method, tail=tail)

    def similarity(
        self,
        data,
        *,
        plot=False,
        method="2d",
        n_permute=5000,
        metric="spearman",
        include_diag=False,
        nan_policy="omit",
        tail=2,
        return_null=False,
        n_jobs=-1,
        random_state=None,
        progress_bar: bool = False,
        project: bool = False,
    ):
        """Calculate similarity between two Adjacency matrices.

        The default uses Spearman correlation and a permutation test.

        Args:
            data (Adjacency | np.ndarray): Adjacency to compare against, or a 1-D array
                the same size as `self.data`.
            plot (bool): Plot the two stacked adjacency matrices being compared.
                Default False.
            method (str | None): Permutation scheme, `'1d'`, `'2d'`, or None (no
                permutation test).
            n_permute (int): Number of permutations for the p-value. Default 5000.
            metric (str): `'spearman'`, `'pearson'`, or `'kendall'`.
            include_diag (bool): Only applies to `'directed'` matrices with
                `method=None` or `method='1d'`. Default False (self-similarity is
                uninformative). Symmetric matrices never store the diagonal, so this
                flag is a no-op for them.
            nan_policy (str): How to handle NaN values: `'omit'` removes NaN pairwise
                before computing the correlation (default), `'propagate'` lets NaN
                flow through, `'raise'` errors if any NaN is present.
            tail (int | str): `2`/`'two'` (two-tailed, default) or `1`/`'one'`
                (one-tailed, positive direction).
            return_null (bool): If True, also return the null distribution. Default False.
            n_jobs (int): Number of parallel jobs. Default -1 (all cores).
            random_state (int, optional): Random seed for reproducibility.
            progress_bar (bool): If True, show a progress bar. Default False.
            project (bool): If True and this Adjacency has a `spatial_scale`, project
                the per-matrix correlations back into brain space. Default False.

        Returns:
            dict | list[dict] | BrainData: A correlation result dict with keys
                'correlation', 'p', and 'device' for a single matrix, a list of
                such dicts when this Adjacency holds multiple matrices, or a
                `BrainData` when `project=True` (per-matrix correlations
                projected via spatial_scale).
        """
        from .stats import similarity

        return similarity(
            self,
            data,
            plot=plot,
            method=method,
            n_permute=n_permute,
            metric=metric,
            include_diag=include_diag,
            nan_policy=nan_policy,
            tail=tail,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
            project=project,
        )

    def social_relations_model(self, summarize_results=True, nan_replace=True):
        """Estimate the social relations model from a matrix for a round-robin design.

        $$X_{ij} = m + \\alpha_i + \\beta_j + g_{ij} + \\epsilon_{ijl}$$

        where $X_{ij}$ is the score for person i rating person j, $m$ is the group mean,
        $\\alpha_i$ is person i's actor effect, $\\beta_j$ is person j's partner effect, $g_{ij}$
        is the relationship effect and $\\epsilon_{ijl}$ is the error in measure l for actor i and partner j.

        This model is primarily concerned with partitioning the variance of the various
        effects. The implementation follows Chapter 8 of Kenny, Kashy, & Cook (2006) and
        the tests replicate the book's examples. Actor scores are rows (lower triangle)
        and partner scores are columns (upper triangle). The minimal sample size to
        estimate these effects is 4.

        **Model assumptions:** social interactions are exclusively dyadic; people are
        randomly sampled from the population; there are no order effects; the effects
        combine additively and relationships are linear.

        Args:
            summarize_results (bool): If True, print a formatted summary of model results.
            nan_replace (bool): If True, replace NaN values with row and column means.

        Returns:
            pd.Series | pd.DataFrame: All of the effects estimated using SRM, as a
                Series (single matrix) or DataFrame (one row per matrix).

        References:
            Kenny, D. A., Kashy, D. A., & Cook, W. L. (2006). *Dyadic data analysis*.
            Guilford Press.
        """
        from .modeling import social_relations_model

        return social_relations_model(self, summarize_results, nan_replace)

    def squareform(self):
        """Convert adjacency data back to square form.

        Returns:
            np.ndarray | list[np.ndarray]: A square matrix, or a list of them for a
                stack.
        """
        if self.issymmetric:
            if self.is_single_matrix:
                return squareform(self.data)
            return [squareform(x.data) for x in self]
        if self.is_single_matrix:
            return self.data.reshape(
                int(np.sqrt(self.data.shape[0])), int(np.sqrt(self.data.shape[0]))
            )
        return [
            x.data.reshape(int(np.sqrt(x.data.shape[0])), int(np.sqrt(x.data.shape[0])))
            for x in self
        ]

    def stats_label_distance(self, *, labels=None, n_permute=5000, n_jobs=-1):
        """Calculate permutation tests on within and between label distance.

        Args:
            labels (np.ndarray, optional): Group label per node; defaults to the
                stored labels.
            n_permute (int): Number of permutations to run. Default 5000.
            n_jobs (int): Number of parallel jobs. Default -1 (all cores).

        Returns:
            dict: Per-group within-vs-between distance differences and p-values, keyed
                by group label.
        """
        from .stats import stats_label_distance

        return stats_label_distance(
            self, labels=labels, n_permute=n_permute, n_jobs=n_jobs
        )

    def std(self, axis=0):
        """Calculate standard deviation of Adjacency.

        Args:
            axis (int): Calculate std over matrices (0) or upper triangle (1).

        Returns:
            float | Adjacency | np.ndarray: A float for a single matrix; an
                Adjacency when `axis=0`; an array when `axis=1`.
        """
        return apply_stat(self, np.nanstd, axis)

    def sum(self, axis=0):
        """Calculate sum of Adjacency.

        Args:
            axis (int): Calculate sum over matrices (0) or upper triangle (1).

        Returns:
            float | Adjacency | np.ndarray: A float for a single matrix; an
                Adjacency when `axis=0`; an array when `axis=1`.
        """
        return apply_stat(self, np.nansum, axis)

    def threshold(self, *, upper=None, lower=None, binarize=False):
        """Threshold an Adjacency instance.

        Provide upper and lower values or percentages to perform two-sided
        thresholding. Binarize will return a mask image respecting thresholds
        if provided, otherwise respecting every non-zero value.

        Args:
            upper (float | str, optional): Upper cutoff. A string such as `'95%'` is
                interpreted as a percentile; None for one-sided thresholding.
            lower (float | str, optional): Lower cutoff. A string such as `'5%'` is
                interpreted as a percentile; None for one-sided thresholding.
            binarize (bool): Return a binarized matrix respecting the thresholds if
                provided, otherwise binarize on every non-zero value. Default False.

        Returns:
            Adjacency: Thresholded Adjacency instance.
        """
        from .stats import threshold

        return threshold(self, upper=upper, lower=lower, binarize=binarize)

    def to_brain(self, values, *, fill: float = np.nan):
        """Project per-matrix scalars back to voxel-space `BrainData`.

        Requires `spatial_scale` to be set (i.e. this stack came from
        `BrainData.distance` or another spatial-scale-aware producer).
        Each entry of `values` is painted onto the voxels assigned to its
        corresponding parcel by `spatial_scale.atlas` /
        `spatial_scale.roi_labels`. Voxels outside the atlas receive
        `fill`.

        Args:
            values (np.ndarray): 1-D array of length `len(self)` — one scalar per
                matrix in the stack.
            fill (float): Value for voxels not covered by any provided ROI label.
                Default `np.nan`.

        Returns:
            BrainData: Single image masked to `spatial_scale.source_mask`.

        Raises:
            ValueError: If `spatial_scale` is None, or `values` has the
                wrong length.

        Examples:
            ```python
            rdms = brain.distance(metric="correlation", spatial_scale="roi", roi_mask=atlas)
            sims = [r["correlation"] for r in rdms.similarity(model_rdm)]
            brain_map = rdms.to_brain(sims)
            ```
        """
        from nltools.mask import roi_to_brain_from_atlas

        if self.spatial_scale is None:
            raise ValueError(
                "to_brain() requires spatial_scale to be set on the "
                "Adjacency. Produce this stack via a spatial-scale-aware "
                "operation (e.g. BrainData.distance(spatial_scale='roi', "
                "roi_mask=atlas))."
            )
        arr = np.asarray(values)
        if arr.shape != (len(self),):
            raise ValueError(
                f"values must be 1-D with length {len(self)} (one per "
                f"stacked matrix); got shape {arr.shape}"
            )
        ss = self.spatial_scale
        return roi_to_brain_from_atlas(
            arr,
            atlas=ss.atlas,
            source_mask=ss.source_mask,
            roi_labels=ss.roi_labels,
            fill=fill,
        )

    def to_graph(self):
        """Convert a single Adjacency matrix into a NetworkX graph.

        This currently works only when `is_single_matrix` is True.

        Returns:
            networkx.Graph | networkx.DiGraph: `DiGraph` for directed matrices,
                `Graph` otherwise; nodes are relabeled with `labels` when set.
        """
        from .io import to_graph

        return to_graph(self)

    def to_square(self):
        """Convert adjacency back to square matrix format.

        This is an alias for `squareform`.

        Returns:
            np.ndarray | list[np.ndarray]: Square matrix representation, or a list
                of them if this object contains multiple adjacency matrices.
        """
        return self.squareform()

    def ttest(
        self,
        *,
        permutation=False,
        n_permute=5000,
        tail=2,
        return_null=False,
        n_jobs=-1,
        random_state=None,
        progress_bar: bool = False,
    ):
        """Calculate a one-sample t-test across stacked matrices.

        Args:
            permutation (bool): Run the test as a permutation test. Note this can be
                very slow.
            n_permute (int): Number of permutations (used only when
                `permutation=True`). Default 5000.
            tail (int | str): `2`/`'two'` (two-tailed, default) or `1`/`'one'`
                (one-tailed: mean > 0; negate the data for the other direction).
                Applies to both the parametric and permutation paths.
            return_null (bool): If True, also return the null distribution. Default False.
            n_jobs (int): Number of parallel jobs. Default -1 (all cores).
            random_state (int, optional): Random seed for reproducibility.
            progress_bar (bool): If True, show a progress bar. Default False.

        Returns:
            dict: `'t'` — Adjacency of t values (or means when `permutation=True`) —
                and `'p'` — Adjacency of p values.
        """
        from .stats import ttest

        return ttest(
            self,
            permutation=permutation,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )

    def write(self, file_name, method="long"):
        """Write the Adjacency to a `.csv` or `.h5` file.

        Args:
            file_name (str | Path): Output path; an `.h5`/`.hdf5` suffix writes HDF5.
            method (str): Layout for CSV output, `'long'` (vectorized rows) or
                `'square'` (single matrix only).
        """
        from .io import write

        return write(self, file_name, method)

    def z_to_r(self):
        """Convert each z score back into an r value."""
        from .stats import z_to_r

        return z_to_r(self)
