"""
HyperAlignment using Procrustes Analysis
=========================================

Procrustes-based hyperalignment for aligning multi-subject neuroimaging data
to a common representational space through iterative template refinement.

This module implements the hyperalignment technique described in:

Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
A common, high-dimensional model of the representational space in
human ventral temporal cortex. Neuron, 72(2), 404-416.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import orthogonal_procrustes

__all__ = ["HyperAlignment"]


def _procrustes_pairwise(data1, data2):
    """Pairwise Procrustes alignment between two matrices.

    Internal helper function that performs pairwise Procrustes alignment.
    This is adapted from nltools.stats.procrustes() for internal use.

    Args:
        data1 (array_like, shape (n_samples, n_features)): Reference matrix
            (target for alignment)
        data2 (array_like, shape (n_samples, n_features)): Matrix to be aligned
            to data1

    Returns:
        mtx1 (ndarray): Standardized version of data1
        mtx2 (ndarray): Aligned version of data2
        disparity (float): Sum of squared differences between aligned matrices
        R (ndarray): Orthogonal transformation matrix
        scale (float): Scale factor from singular values
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
        >>> # Fit hyperalignment
        >>> hyper = HyperAlignment(n_iter=2)
        >>> hyper.fit(data)
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

    References:
        Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
        Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
        A common, high-dimensional model of the representational space in
        human ventral temporal cortex. Neuron, 72(2), 404-416.
    """

    def __init__(self, n_iter=2, auto_pad=True):
        """Initialize HyperAlignment.

        Args:
            n_iter (int, default=2): Number of template refinement iterations
            auto_pad (bool, default=True): Whether to automatically pad matrices
                to same size
        """
        self.n_iter = n_iter
        self.auto_pad = auto_pad

    def fit(self, data):
        """Fit hyperalignment model to data.

        Args:
            data (list of ndarray): List of data matrices, each with shape
                (n_features, n_samples). Different subjects can have different
                numbers of features if auto_pad=True.

        Returns:
            self (HyperAlignment): Fitted model
        """
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
        template = None
        for i, x in enumerate(m):
            if i == 0:
                # Use first data as initial template
                template = np.copy(x.T)
            else:
                # Align to evolving template and accumulate
                _, trans, _, _, _ = _procrustes_pairwise(template / i, x.T)
                template += trans
        template /= len(m)

        ## STAGE 2: REFINE TEMPLATE (n_iter iterations) ##
        for iteration in range(self.n_iter):
            # Align each subject to current template and create refined template
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

    def transform(self, data):
        """Transform data to common space using fitted transformations.

        Args:
            data (list of ndarray): List of data matrices to transform. Should be
                the same data used for fitting (or have compatible dimensions).

        Returns:
            transformed (list of ndarray): List of transformed data matrices in
                common space
        """
        if not hasattr(self, "w_"):
            raise ValueError("Model must be fit before transform")

        # Apply stored transformations
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

    def transform_subject(self, subject_data):
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
