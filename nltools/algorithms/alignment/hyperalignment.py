"""HyperAlignment: multi-subject alignment by iterative Procrustes refinement.

Hyperalignment finds a common representational space across subjects by
iteratively refining pairwise Procrustes transformations (Haxby et al., 2011).
Every subject keeps its full feature dimensionality — unlike `SRM`, which
projects into a lower-dimensional shared space.

**Algorithm.** Initialize the template from the first subject, incrementally
aligning and averaging the rest; then for `n_iter` iterations align each
subject to the template with a Procrustes transform and re-average in the
aligned space; finally align every subject to the refined template.

**Performance.** Time is O(n_iter × n_subjects × n_voxels × n_samples) with an
SVD per subject per iteration; memory is O(n_subjects × n_voxels × n_samples).
`parallel='cpu'` runs the per-subject Procrustes fits with joblib, which pays
off with 3+ subjects, many voxels (>10K), or several iterations.

**When to use.** Multi-subject alignment that must preserve spatial structure;
an alternative to `SRM` when dimension reduction is not wanted. For aligning a
single pair of matrices use `procrustes`.

**Reference.** Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko,
Y. O., Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011). A common,
high-dimensional model of the representational space in human ventral temporal
cortex. *Neuron*, 72(2), 404-416.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import orthogonal_procrustes

__all__ = ["HyperAlignment"]


def _procrustes_pairwise(
    data1: np.ndarray, data2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Pairwise Procrustes alignment between two matrices.

    Finds the orthogonal transformation (rotation + reflection) and scaling that
    aligns `data2` to `data1` with minimum sum of squared differences. Both
    matrices are centered and normalized to unit Frobenius norm, the rotation
    `R = U V^T` is taken from the SVD `U Σ V^T = data1^T @ data2`, and the
    result is `data2 @ R^T * scale`. Matrices with different column counts are
    zero-padded to match. Same computation as `procrustes`; used pairwise inside
    the hyperalignment iterations (one SVD, O(min(n_samples, n_features)^3)).

    Args:
        data1 (np.ndarray): Reference matrix, shape (n_samples, n_features).
        data2 (np.ndarray): Matrix to align to `data1`, shape (n_samples, n_features).

    Returns:
        tuple[np.ndarray, np.ndarray, float, np.ndarray, float]: `(mtx1, mtx2,
            disparity, R, scale)` — the standardized `data1`, the aligned `data2`,
            the sum of squared differences between them, the orthogonal
            transformation matrix, and the scale factor from the singular values.

    Raises:
        ValueError: If input matrices have incompatible shapes or are empty.

    Examples:
        ```python
        import numpy as np

        data1 = np.random.randn(100, 50)
        data2 = np.random.randn(100, 50)
        mtx1, mtx2, disparity, R, scale = _procrustes_pairwise(data1, data2)
        disparity  # → small after alignment, e.g. 0.023
        ```
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
    """Hyperalignment using iterative Procrustes alignment (Haxby et al., 2011).

    Aligns multi-subject data in three stages: build an initial average
    template, refine it over `n_iter` rounds of align-and-average, then align
    every subject to the refined template. Each subject's data is a
    (n_features, n_samples) matrix; subjects may differ in `n_features` when
    `auto_pad=True`.

    Args:
        n_iter (int): Number of template refinement iterations. Defaults to 2.
        auto_pad (bool): If True, zero-pad each subject's feature axis up to the
            largest feature count. If False, all matrices must already have the
            same shape. Defaults to True.

    Attributes:
        w_ (list[np.ndarray]): Per-subject transformation matrices (rotation +
            reflection), each of shape (n_features, n_features).
        s_ (np.ndarray): The common template, shape (n_features, n_samples).
        common_model_ (np.ndarray): Alias for `s_`.
        disparity_ (list[float]): Per-subject sum of squared differences from the
            template after alignment.
        scale_ (list[float]): Per-subject scale factors.

    Note:
        `parallel='cpu'` (the default of `fit` and `transform`) runs the
        per-subject Procrustes fits with joblib; it pays off with 3+ subjects,
        many voxels (>10K), or several iterations. Use `parallel=None` for
        debugging or small problems.

    Examples:
        ```python
        import numpy as np
        from nltools.algorithms import HyperAlignment

        data = [np.random.randn(100, 50) for _ in range(3)]  # 3 subjects

        hyper = HyperAlignment(n_iter=2)
        hyper.fit(data, parallel="cpu", n_jobs=-1)
        aligned = hyper.transform(data)  # list of arrays in the common space
        template = hyper.s_  # or hyper.common_model_

        # Align a new subject to the fitted template
        new_subject = np.random.randn(100, 50)
        transformed, R, disparity, scale = hyper.transform_subject(new_subject)
        ```
    """

    def __init__(self, n_iter: int = 2, auto_pad: bool = True) -> None:
        """Initialize HyperAlignment.

        Args:
            n_iter (int): Number of template refinement iterations. Defaults to 2.
            auto_pad (bool): Whether to zero-pad matrices to the same size.
                Defaults to True.
        """
        self.n_iter = n_iter
        self.auto_pad = auto_pad

    def fit(
        self,
        data: list[np.ndarray],
        *,
        parallel: str | None = "cpu",
        n_jobs: int = -1,
    ) -> "HyperAlignment":
        """Fit hyperalignment model to data.

        Args:
            data (list[np.ndarray]): Data matrices, each of shape
                (n_features, n_samples). Subjects may differ in `n_features`
                when `auto_pad=True`.
            parallel (str | None): `'cpu'` (default) aligns subjects in parallel
                with joblib; None runs single-threaded NumPy.
            n_jobs (int): Number of CPU workers when `parallel='cpu'`; -1
                (default) picks a count from available memory.

        Returns:
            HyperAlignment: Fitted model (`self`).
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
            # Zero-pad every subject's feature axis (rows) up to the LARGEST
            # feature count, as documented, so no subject's features are dropped.
            # (The sample axis / columns is already validated equal above.)
            R = max(x.shape[0] for x in data)

            m = []
            for x in data:
                missing = R - x.shape[0]
                if missing > 0:
                    add = np.zeros((missing, x.shape[1]), dtype=x.dtype)
                    x = np.vstack([x, add])
                m.append(np.asarray(x, dtype=float).copy())
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
                    from nltools.algorithms.backends import auto_n_jobs_for_arrays

                    n_jobs_to_use = auto_n_jobs_for_arrays(m)

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
        """Alias for `s_` (the common template)."""
        return self.s_

    def transform(
        self,
        data: list[np.ndarray],
        *,
        parallel: str | None = "cpu",
        n_jobs: int = -1,
    ) -> list[np.ndarray]:
        """Transform data to the common space using the fitted transformations.

        Args:
            data (list[np.ndarray]): Data matrices to transform, one per fitted
                subject in the same order as `fit` (the same data or data of
                compatible shape).
            parallel (str | None): `'cpu'` (default) transforms subjects in
                parallel with joblib; None falls back to the setting used in `fit`.
            n_jobs (int): Number of CPU workers when parallel; -1 (default) reuses
                the value from `fit`, itself resolved from available memory.

        Returns:
            list[np.ndarray]: Transformed data matrices in the common space.
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
                from nltools.algorithms.backends import auto_n_jobs_for_arrays

                n_jobs_to_use = auto_n_jobs_for_arrays(data)

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
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Align a new subject to the common space.

        Args:
            subject_data (np.ndarray): Data from a new subject, shape
                (n_features, n_samples), to align to the common template.

        Returns:
            tuple[np.ndarray, np.ndarray, float, float]: `(transformed, R, disparity,
                scale)` — aligned data in common space, the transformation matrix
                used, the alignment quality (sum of squared differences), and the
                scale factor used.
        """
        if not hasattr(self, "s_"):
            raise ValueError("Model must be fit before transform_subject")

        # Align new subject to common template
        # s_ is [features, samples], transpose to [samples, features] for procrustes
        _, transformed, disparity, R, scale = _procrustes_pairwise(
            self.s_.T, subject_data.T
        )

        return transformed.T, R, disparity, scale
