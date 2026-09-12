"""Provide data structures for working with similarity and dissimilarity matrices."""

from copy import deepcopy
import numpy as np
import polars as pl
from sklearn.metrics.pairwise import pairwise_distances

from nltools.utils import attempt_to_import
from .utils import apply_stat, perform_arithmetic

__all__ = ["Adjacency"]

# Optional dependencies
nx = attempt_to_import("networkx")

MAX_INT = np.iinfo(np.int32).max


class Adjacency:
    """Represent adjacency matrices in vectorized form.

    Store distance/similarity matrices as strict upper triangles and directed
    matrices as full row-major vectors. Symmetric reconstruction always has a
    zero diagonal; input diagonals are discarded. Flat rectangular stacks require
    an explicit `*_flat` matrix type. A list or 2-D flat array retains stack rank,
    including one matrix. A zero-length symmetric vector represents one node.
    Construction and result methods return independently owned mutable state.

    Args:
        data (Adjacency | np.ndarray | pd.DataFrame | pl.DataFrame | str | Path | list): A square
            matrix, a flattened vector, a `.csv`/`.h5` path, or a list of
            matrices/`Adjacency` instances/`.csv` paths to stack.
        Y (pd.DataFrame | pl.DataFrame, optional): Matrix metadata, one row per matrix.
            None inherits metadata during copy construction.
        matrix_type (str, optional): Type of matrix. One of `'distance'`, `'similarity'`,
            `'directed'`, `'distance_flat'`, `'similarity_flat'`, `'directed_flat'`.
            For copy construction, this confirms the existing kind without reinterpreting it.
        labels (list, optional): Shared node labels, or a nested matrix-by-node
            label grid for a stack. None inherits labels during copy construction.

    Attributes:
        data (np.ndarray): Vectorized matrix values. Shape `(vector_length,)` for a
            single matrix or `(n_matrices, vector_length)` for a stack; symmetric
            matrices store only the upper triangle without the diagonal.
        matrix_type (str): One of `'distance'`, `'similarity'`, `'directed'`, or
            `'empty'` (the `'_flat'` input variants are normalized to their base type).
        is_single_matrix (bool): True for single storage; a one-row stack is False.
        issymmetric (bool): True for distance/similarity matrices, False for directed.
        labels (list): Node labels (empty list when none were given).
        Y (pl.DataFrame): Training labels as a polars DataFrame (possibly empty).
        is_empty (bool): True if the instance holds no matrices.
        n_nodes (int): Number of nodes `n` for an `(n, n)` matrix.
        shape (tuple): Logical shape — `(n_nodes, n_nodes)` for a single matrix,
            `(n_matrices, n_nodes, n_nodes)` for a stack, including typed empty stacks;
            `(0, 0)` for an untyped empty constructor.
        vector_shape (tuple): Shape of the internal vectorized storage (`data.shape`).
    """

    def __init__(self, data=None, *, Y=None, matrix_type=None, labels=None):
        from .state import initialize

        initialize(self, data, matrix_type=matrix_type, labels=labels, Y=Y)

    # ── Dunders (alphabetical) ──────────────────────────────────────────

    def __add__(self, y):
        return perform_arithmetic(self, y, np.add, "add")

    def __copy__(self):
        from nltools.data.braindata.utils import _copy_complete

        return _copy_complete(self)

    def __deepcopy__(self, memo):
        from nltools.data.braindata.utils import _copy_complete

        return _copy_complete(self, memo)

    def __getitem__(self, index):
        from .state import select

        return select(self, index)

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
        from .state import owned_frame

        self._Y = owned_frame(value, len(self))

    @property
    def is_empty(self) -> bool:
        """Check if Adjacency object is empty.

        Returns:
            bool: True if the adjacency matrix is empty, False otherwise.
        """
        return len(self) == 0

    @property
    def n_nodes(self):
        """Return the number of nodes in the adjacency matrix.

        Returns:
            int: Number of nodes (n) for an (n, n) matrix.
        """
        return self._n_nodes

    @property
    def shape(self):
        """Return the logical shape of the adjacency matrix.

        Returns:
            tuple: `(n_nodes, n_nodes)` for a single matrix, `(n_matrices, n_nodes,
                n_nodes)` for stacked matrices, including typed empty stacks; `(0, 0)`
                for an untyped empty constructor.

        Note:
            Use `.vector_shape` to get the internal vectorized representation shape.
        """
        if self.matrix_type == "empty":
            return (0, 0)

        if self.is_single_matrix:
            return (self.n_nodes, self.n_nodes)
        return (len(self), self.n_nodes, self.n_nodes)

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
        from .state import append

        return append(self, data)

    def bootstrap(
        self,
        statistic,
        *,
        n_samples=5000,
        confidence_level=0.95,
        return_samples=False,
        n_jobs=-1,
        random_state=None,
        progress_bar: bool = False,
    ):
        """Bootstrap an aggregate statistic across a stack of matrices.

        Resamples matrices with replacement and aggregates the replicates as
        they complete, so what the run holds is the retained tail — about
        ``(1 - confidence_level)`` of the replicates per edge — plus one
        dispatch window, rather than all ``n_samples`` matrices.

        Args:
            statistic (str): Statistic to bootstrap: `'mean'`, `'median'`,
                `'std'`, `'sum'`, `'min'`, or `'max'` — each the corresponding
                NumPy reduction over matrices, with `'std'` at ``ddof=0``.
            n_samples (int): Number of bootstrap replicates, at least two.
                Default 5000.
            confidence_level (float): Confidence level of the reported
                interval, strictly between zero and one. Default 0.95. The
                bounds are the central percentile interval, elementwise
                marginal per edge.
            return_samples (bool): Retain and return every replicate. Default
                False.
            n_jobs (int): CPU worker ceiling. -1 (default) means all cores.
            random_state (int | None): Random seed for reproducibility.
            progress_bar (bool): If True, show a progress bar. Default False.

        Returns:
            BootstrapResult: ``estimate`` (the statistic on the unresampled
                stack), ``standard_error``, ``ci_lower`` and ``ci_upper`` as
                single-matrix `Adjacency` objects, plus ``samples`` as a NumPy
                array with the bootstrap axis first when
                ``return_samples=True``.

        Examples:
            ```python
            boot = adj.bootstrap("mean", n_samples=1000)
            boot.estimate  # → Adjacency
            ```
        """
        from .modeling import bootstrap

        return bootstrap(
            self,
            statistic,
            n_samples=n_samples,
            confidence_level=confidence_level,
            return_samples=return_samples,
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
        """Return an independently owned copy, preserving internal aliases and cycles."""
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
            data = np.atleast_2d(self.data)

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
        from .state import distance_to_similarity

        return distance_to_similarity(self, metric, beta)

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
            dict: Keys `beta`, `sigma` (coefficient standard error), `t`, `p`,
                `df`, and `residual`. With DesignMatrix predictors, coefficient
                fields are Adjacency maps per predictor (single for one predictor).
                With Adjacency predictors, a single response is required and
                coefficient fields are native predictor arrays or scalars.
                `df` is a scalar; `residual` retains response shape and metadata.
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

        Returns:
            dict | list[dict]: A correlation result dict with keys
                'correlation' and 'p' for a single matrix, or a list of these
                dicts for a stack.
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
            np.ndarray | list[np.ndarray]: Detached square matrix, or a list of
                detached matrices for a stack. Symmetric diagonals are zero.
        """
        from .state import to_square

        return to_square(self)

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
        popmean=0.0,
        permutation=False,
        n_permute=5000,
        tail=2,
        return_null=False,
        n_jobs=-1,
        random_state=None,
        progress_bar: bool = False,
    ):
        """Run a one-sample t-test across stacked matrices.

        Tests every stored edge against `popmean` across the matrices in the
        stack.

        Args:
            popmean (float): Population mean to test against. Default 0.0.
            permutation (bool): If True, take p from a sign-flip permutation
                test on `matrices - popmean`. The reported `t` stays the
                observed parametric statistic. Default False.
            n_permute (int): Number of permutations, used only when
                `permutation=True`. Default 5000.
            tail (int | str): `2`/`'two'` (two-tailed, default) or `1`/`'one'`
                (one-tailed: mean > `popmean`). Applies to both paths.
            return_null (bool): If True, also return the permutation null. Has
                no effect on the parametric path, which computes no null.
                Default False.
            n_jobs (int): Number of parallel jobs. Default -1 (all cores).
            random_state (int, optional): Random seed for reproducibility.
            progress_bar (bool): If True, show a progress bar. Default False.

        Returns:
            dict: `'mean'`, `'t'`, `'z'` and `'p'` as independent single-matrix
                `Adjacency` results that retain the node count, storage kind
                (including directed) and shared node labels, with matrix
                metadata cleared. `'mean'` is the edgewise mean minus `popmean`;
                `'t'` is the observed one-sample t-statistic on both paths;
                `'p'` is parametric, or the empirical sign-flip p-value when
                `permutation=True`; `'z'` is the tail-aware normal score of `p`.
                With `permutation=True` and `return_null=True` the dict also
                holds `'null_dist'`, an owned `(n_permute, n_edges)` array of
                centered means in flat storage order and in the units of
                `'mean'`. Maps are unthresholded. Apply a cutoff or a
                multiple-comparison correction afterwards.

        Raises:
            ValueError: If this Adjacency holds fewer than two matrices.

        Examples:
            ```python
            result = stacked.ttest()
            result["mean"]  # effect size per edge
            result["t"].squareform()  # back to a node-by-node matrix

            # Threshold after testing, never inside it
            import numpy as np

            significant = result["t"].copy()
            significant.data = np.where(result["p"].data < 0.05, result["t"].data, 0.0)
            ```
        """
        from .stats import ttest

        return ttest(
            self,
            popmean=popmean,
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

        HDF5 is the round-trip format: values, matrix kind, node labels, and
        `Y`. CSV stores values only, so a CSV read back loses the labels, `Y`,
        and the matrix kind, and needs an explicit `matrix_type` wherever the
        flat layout is ambiguous. Square CSV output is single-matrix only.

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
