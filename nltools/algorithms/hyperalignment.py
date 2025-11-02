"""
HyperAlignment: Multi-subject cortical surface alignment using iterative Procrustes refinement.

Hyperalignment finds a common representational space across subjects by iteratively
refining pairwise Procrustes transformations. Unlike simple alignment, hyperalignment
preserves both spatial structure and representational similarity.

Algorithm overview:
    1. Initialize template (first subject or group average)
    2. For each iteration:
       - Align each subject to template (Procrustes transformation)
       - Update template (average in aligned space)
    3. Converge when transformations stabilize or max iterations reached
    4. Final alignment: Apply learned transformations to all subjects

Performance:
    - Time complexity: O(n_iter × n_subjects² × n_voxels × n_samples)
    - Memory complexity: O(n_subjects × n_voxels × n_features)
    - Parallelization: ~4-8× speedup with CPU-parallel (parallel="cpu")
    - Most beneficial when subjects have many voxels (>10K) and multiple iterations

When to use hyperalignment:
    - Multi-subject alignment preserving spatial structure
    - Alternative to SRM when spatial structure is important
    - See `nltools.algorithms.srm.SRM` for dimension-reduction approach
    - See `nltools.stats.procrustes()` for single-subject alignment

This module implements the hyperalignment technique described in:

Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
A common, high-dimensional model of the representational space in
human ventral temporal cortex. Neuron, 72(2), 404-416.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import orthogonal_procrustes

__all__ = ["HyperAlignment"]


def _procrustes_pairwise(
    data1: np.ndarray, data2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Pairwise Procrustes alignment between two matrices.

    Procrustes alignment finds the optimal orthogonal transformation (rotation + reflection)
    and scaling to align data2 to data1, minimizing the sum of squared differences.

    Algorithm:
        - Centers both matrices (removes translation)
        - Normalizes to unit Frobenius norm (removes scale)
        - Finds optimal rotation via SVD: R = U V^T where U Σ V^T = data1^T @ data2
        - Applies transformation: data2_aligned = data2 @ R^T * scale

    Performance:
        - Time complexity: O(min(n_samples, n_features)^3) for SVD
        - Memory complexity: O(n_samples × n_features)
        - Used in pairwise fashion during hyperalignment iterations

    Internal helper function that performs pairwise Procrustes alignment.
    This is adapted from nltools.stats.procrustes() for internal use.

    Args:
        data1: Reference matrix (target for alignment), shape (n_samples, n_features).
        data2: Matrix to be aligned to data1, shape (n_samples, n_features).

    Returns:
        tuple: (mtx1, mtx2, disparity, R, scale) where:
            - mtx1: Standardized version of data1 (centered and normalized).
            - mtx2: Aligned version of data2 (transformed to match mtx1).
            - disparity: Sum of squared differences between aligned matrices.
            - R: Orthogonal transformation matrix (rotation + reflection).
            - scale: Scale factor from singular values.

    Raises:
        ValueError: If input matrices have incompatible shapes or are empty.

    Examples:
        >>> import numpy as np
        >>> data1 = np.random.randn(100, 50)
        >>> data2 = np.random.randn(100, 50)
        >>> mtx1, mtx2, disparity, R, scale = _procrustes_pairwise(data1, data2)
        >>> disparity  # Should be small after alignment
        0.023

    Notes:
        - Handles different column sizes by zero-padding the smaller matrix.
        - Centers and normalizes inputs before alignment.
        - Uses singular value decomposition to find optimal transformation.
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape[0] != mtx2.shape[0]:
        raise ValueError("Input matrices must have same number of rows.")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # Handle different column sizes by padding
    if mtx1.shape[1] != mtx2.shape[1]:
        if mtx1.shape[1] > mtx2.shape[1]:
            mtx2 = np.append(
                mtx2, np.zeros((mtx1.shape[0], mtx1.shape[1] - mtx2.shape[1])), axis=1
            )
        else:
            mtx1 = np.append(
                mtx1, np.zeros((mtx1.shape[0], mtx2.shape[1] - mtx1.shape[1])), axis=1
            )

    # Center data
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    # Normalize
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    mtx1 /= norm1
    mtx2 /= norm2

    # Find optimal transformation
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # Calculate disparity
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, s


class HyperAlignment(BaseEstimator, TransformerMixin):
    """Hyperalignment using iterative Procrustes alignment.

    Three-stage iterative process for aligning multi-subject data:
    1. Create initial average template
    2. Refine template through n_iter iterations
    3. Final alignment of all subjects to refined template

    This implements the Procrustes-based hyperalignment method commonly
    used in multi-subject neuroimaging analysis.

    Args:
        n_iter (int, default=2): Number of template refinement iterations
            (stages 1-2).
        auto_pad (bool, default=True): If True, automatically zero-pad matrices
            to standardize sizes. If False, caller must ensure all matrices have
            same dimensions.

    Attributes:
        w_ (list of ndarray, element i has shape=[features_i, features]): The
            transformation matrices (rotation + reflection) for each subject.
        s_ (ndarray, shape=[features, samples]): The aligned common template
            (shared response).
        disparity_ (list of float): Disparity (sum of squared differences) for
            each subject.
        scale_ (list of float): Scale factors for each subject.

    Note:
        ``common_model_`` property provides alias for ``s_`` (backward compatibility).

    Examples:
        Basic multi-subject alignment:

        >>> from nltools.algorithms import HyperAlignment
        >>> import numpy as np
        >>>
        >>> # Create sample data (3 subjects)
        >>> data = [np.random.randn(100, 50) for _ in range(3)]
        >>>
        >>> # Fit hyperalignment with CPU parallelization (default)
        >>> hyper = HyperAlignment(n_iter=2)
        >>> hyper.fit(data, parallel="cpu", n_jobs=-1)
        >>>
        >>> # Transform to common space
        >>> aligned = hyper.transform(data)
        >>>
        >>> # Access common template
        >>> template = hyper.s_  # or hyper.common_model_
        >>>
        >>> # Align a new subject
        >>> new_subject = np.random.randn(100, 50)
        >>> new_transform = hyper.transform_subject(new_subject)

        When to use parallel processing:
        - Use ``parallel="cpu"`` (default) for datasets with 3+ subjects to speed up
          pairwise Procrustes operations during template refinement.
        - Use ``parallel=None`` for debugging or small datasets (<3 subjects) where
          parallelization overhead isn't beneficial.
        - Parallel processing is most beneficial when subjects have many voxels
          (>10K) and template refinement requires multiple iterations.

    References:
        Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
        Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
        A common, high-dimensional model of the representational space in
        human ventral temporal cortex. Neuron, 72(2), 404-416.
    """

    def __init__(self, n_iter: int = 2, auto_pad: bool = True) -> None:
        """Initialize HyperAlignment.

        Args:
            n_iter (int, default=2): Number of template refinement iterations
            auto_pad (bool, default=True): Whether to automatically pad matrices
                to same size
        """
        self.n_iter = n_iter
        self.auto_pad = auto_pad

    def fit(
        self,
        data: List[np.ndarray],
        parallel: Optional[str] = "cpu",
        n_jobs: int = -1,
    ) -> "HyperAlignment":
        """Fit hyperalignment model to data.

        Args:
            data (list of ndarray): List of data matrices, each with shape
                (n_features, n_samples). Different subjects can have different
                numbers of features if auto_pad=True.
            parallel (str, optional): Execution backend.
                - None: Single-threaded NumPy (debugging/small problems)
                - "cpu": CPU parallelization via joblib (default, multi-subject processing)
            n_jobs (int): Number of CPU cores for parallelization (-1 = auto-detect based on memory).
                Only used when parallel="cpu". Defaults to -1.

        Returns:
            self (HyperAlignment): Fitted model
        """
        # Validate parallel parameter
        if parallel not in [None, "cpu"]:
            raise ValueError(f"parallel must be None or 'cpu', got {parallel}")

        # Store parallel settings
        self._parallel = parallel
        self._n_jobs = n_jobs
        if not isinstance(data, list):
            raise TypeError("Data must be a list of arrays")
        if len(data) == 0:
            raise ValueError("Data list cannot be empty")

        # Validate input
        for i, x in enumerate(data):
            if not isinstance(x, np.ndarray):
                raise TypeError(f"Element {i} is not a numpy array")
            if x.ndim != 2:
                raise ValueError(f"Element {i} must be 2-dimensional")

        # Check that all have same number of samples
        n_samples = data[0].shape[1]
        for i, x in enumerate(data):
            if x.shape[1] != n_samples:
                raise ValueError(
                    f"All matrices must have same number of samples (columns). "
                    f"Element 0 has {n_samples}, element {i} has {x.shape[1]}"
                )

        ## STAGE 0: STANDARDIZE SIZE AND SHAPE ##
        if self.auto_pad:
            # Find dimensions
            sizes_0 = [x.shape[0] for x in data]
            sizes_1 = [x.shape[1] for x in data]

            # Use minimum rows (features) and maximum cols (samples)
            R = min(sizes_0)
            C = max(sizes_1)

            # Pad to standardized size
            m = []
            for x in data:
                y = x[0:R, :]  # Truncate to min features
                missing = C - y.shape[1]
                if missing > 0:
                    add = np.zeros((y.shape[0], missing))
                    y = np.append(y, add, axis=1)
                m.append(y)
        else:
            # Verify all same size
            shapes = [x.shape for x in data]
            if not all(s == shapes[0] for s in shapes):
                raise ValueError(
                    "With auto_pad=False, all matrices must have same shape. "
                    f"Got shapes: {shapes}"
                )
            m = [x.copy() for x in data]

        ## STAGE 1: CREATE INITIAL AVERAGE TEMPLATE ##
        # Start with first subject as template, then incrementally align others
        # This provides a better starting point than random initialization
        template = None
        for i, x in enumerate(m):
            if i == 0:
                # Use first data as initial template
                template = np.copy(x.T)
            else:
                # Align to evolving template and accumulate
                # Incremental averaging reduces bias from order-dependence
                _, trans, _, _, _ = _procrustes_pairwise(template / i, x.T)
                template += trans
        template /= len(m)

        ## STAGE 2: REFINE TEMPLATE (n_iter iterations) ##
        # Iteratively refine template by aligning all subjects and averaging
        # Each iteration improves the common space representation
        for iteration in range(self.n_iter):
            # Align each subject to current template and create refined template
            # Use CPU parallelization for pairwise Procrustes operations if requested
            if self._parallel == "cpu" and len(m) > 2:
                from joblib import Parallel, delayed

                def _align_to_template(subj_idx):
                    """Align one subject to template."""
                    _, trans, _, _, _ = _procrustes_pairwise(template, m[subj_idx].T)
                    return trans

                # Auto-detect n_jobs if needed
                n_jobs_to_use = self._n_jobs
                if n_jobs_to_use == -1:
                    try:
                        from nltools.algorithms.inference.utils import (
                            _auto_n_jobs_cpu,
                            _estimate_data_size_mb,
                        )

                        # Estimate memory for largest subject
                        max_size_mb = max(_estimate_data_size_mb(x) for x in m)
                        n_jobs_to_use = _auto_n_jobs_cpu(
                            data_size_mb=max_size_mb,
                            n_permute=len(m),
                            max_memory_gb=8.0,
                            min_jobs=1,
                        )
                    except ImportError:
                        n_jobs_to_use = 1

                # Parallel alignment
                aligned_subjects = Parallel(n_jobs=n_jobs_to_use)(
                    delayed(_align_to_template)(i) for i in range(len(m))
                )
                common = np.zeros(template.shape)
                for trans in aligned_subjects:
                    common += trans
            else:
                # Single-threaded alignment
                common = np.zeros(template.shape)
                for x in m:
                    _, trans, _, _, _ = _procrustes_pairwise(template, x.T)
                    common += trans
            common /= len(m)
            template = common

        ## STAGE 3: FINAL ALIGNMENT TO REFINED TEMPLATE ##
        aligned = []
        transformation_matrix = []
        disparity = []
        scale = []

        for x in m:
            _, transformed, d, t, s = _procrustes_pairwise(template, x.T)
            aligned.append(transformed.T)
            transformation_matrix.append(t)
            disparity.append(d)
            scale.append(s)

        # Store fitted attributes
        # Note: template is in [samples, features], transpose to [features, samples]
        self.w_ = transformation_matrix
        self.s_ = template.T
        self.disparity_ = disparity
        self.scale_ = scale

        return self

    @property
    def common_model_(self):
        """Alias for ``s_`` (common template)."""
        return self.s_

    def transform(
        self,
        data: List[np.ndarray],
        parallel: Optional[str] = "cpu",
        n_jobs: int = -1,
    ) -> List[np.ndarray]:
        """Transform data to common space using fitted transformations.

        Args:
            data (list of ndarray): List of data matrices to transform. Should be
                the same data used for fitting (or have compatible dimensions).
            parallel (str, optional): Execution backend.
                - None: Single-threaded NumPy (debugging/small problems)
                - "cpu": CPU parallelization via joblib (default, multi-subject processing)
            n_jobs (int): Number of CPU cores for parallelization (-1 = auto-detect based on memory).
                Only used when parallel="cpu". Defaults to -1.

        Returns:
            transformed (list of ndarray): List of transformed data matrices in
                common space
        """
        # Validate parallel parameter
        if parallel not in [None, "cpu"]:
            raise ValueError(f"parallel must be None or 'cpu', got {parallel}")

        # Use stored parallel settings if not provided
        parallel_to_use = (
            parallel if parallel is not None else getattr(self, "_parallel", None)
        )
        n_jobs_to_use = n_jobs if n_jobs != -1 else getattr(self, "_n_jobs", -1)
        if not hasattr(self, "w_"):
            raise ValueError("Model must be fit before transform")

        # Apply stored transformations
        # Use CPU parallelization if requested
        if parallel_to_use == "cpu" and len(data) > 1:
            from joblib import Parallel, delayed

            def _transform_one_subject(subj_idx):
                """Transform one subject."""
                x = data[subj_idx]
                # Apply the Procrustes transformation
                # Standardize first (center and normalize)
                centered = x.T - np.mean(x.T, 0)
                norm = np.linalg.norm(centered)
                if norm > 0:
                    standardized = centered / norm
                else:
                    standardized = centered

                # Apply transformation and scale
                aligned = (
                    np.dot(standardized, self.w_[subj_idx].T) * self.scale_[subj_idx]
                )
                return aligned.T

            # Auto-detect n_jobs if needed
            if n_jobs_to_use == -1:
                try:
                    from nltools.algorithms.inference.utils import (
                        _auto_n_jobs_cpu,
                        _estimate_data_size_mb,
                    )

                    # Estimate memory for largest subject
                    max_size_mb = max(_estimate_data_size_mb(x) for x in data)
                    n_jobs_to_use = _auto_n_jobs_cpu(
                        data_size_mb=max_size_mb,
                        n_permute=len(data),
                        max_memory_gb=8.0,
                        min_jobs=1,
                    )
                except ImportError:
                    n_jobs_to_use = 1

            transformed = Parallel(n_jobs=n_jobs_to_use)(
                delayed(_transform_one_subject)(i) for i in range(len(data))
            )
        else:
            # Single-threaded transform
            transformed = []
            for i, x in enumerate(data):
                # Apply the Procrustes transformation
                # Standardize first (center and normalize)
                centered = x.T - np.mean(x.T, 0)
                norm = np.linalg.norm(centered)
                if norm > 0:
                    standardized = centered / norm
                else:
                    standardized = centered

                # Apply transformation and scale
                aligned = np.dot(standardized, self.w_[i].T) * self.scale_[i]
                transformed.append(aligned.T)

        return transformed

    def transform_subject(
        self, subject_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Align a new subject to the common space.

        Args:
            subject_data (ndarray, shape (n_features, n_samples)): Data from a new
                subject to align to the common template

        Returns:
            transformed (ndarray): Aligned data in common space
            R (ndarray): Transformation matrix used
            disparity (float): Alignment quality (sum of squared differences)
            scale (float): Scale factor used
        """
        if not hasattr(self, "s_"):
            raise ValueError("Model must be fit before transform_subject")

        # Align new subject to common template
        # s_ is [features, samples], transpose to [samples, features] for procrustes
        _, transformed, disparity, R, scale = _procrustes_pairwise(
            self.s_.T, subject_data.T
        )

        return transformed.T, R, disparity, scale
