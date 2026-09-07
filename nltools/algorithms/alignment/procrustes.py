"""Data alignment — SRM, Procrustes, and state alignment."""

__all__ = ["align", "align_states", "procrustes", "procrustes_distance"]

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes as procrust
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from ..inference.utils import _compute_pvalue
from ..inference.validation import validate_tail_parameter
from .srm import SRM, DetSRM


def align(  # nosemgrep: kwargs-internal-forwarding  # forwards to the SRM/DetSRM algorithm constructors
    data, method="deterministic_srm", n_features=None, axis=0, *args, **kwargs
):
    """Align subject data into a common response model.

    A convenience wrapper around the `HyperAlignment` and `SRM`/`DetSRM` classes.
    Aligns a group of subjects either by Procrustes-based hyperalignment
    (Haxby et al., 2011) or by the Shared Response Model (Chen et al., 2015).
    The common model is the shared response (SRM) or the centered group template
    (Procrustes). Transformed data can be projected back into each subject's
    original space with its transformation matrix. To align a single `BrainData`
    to another, use `BrainData.align` instead.

    Args:
        data (list[BrainData] | list[np.ndarray]): Subjects to align; all elements
            must be the same type. Arrays are observations x features.
        method (str): One of `'probabilistic_srm'`, `'deterministic_srm'`, or
            `'procrustes'`. Defaults to `'deterministic_srm'`.
        n_features (int | None): Number of features in the common space (SRM only).
            None uses the number of voxels. Must be None for `'procrustes'`.
        axis (int): Axis to align on: 0 aligns timepoints (ISC computed per voxel),
            1 aligns voxels (ISC computed per timepoint). Defaults to 0.
        *args (Any): Positional arguments forwarded to the `SRM`/`DetSRM`
            constructor.
        **kwargs (Any): Keyword arguments forwarded to the `SRM`/`DetSRM`
            constructor.

    Returns:
        dict: Keys `'transformed'` (list of aligned subject data, same type as the
            input), `'transformation_matrix'` (per-subject transforms),
            `'common_model'` (shared response or group template), and `'isc'`
            (dict mapping each aligned unit to its mean intersubject correlation).
            With `method='procrustes'` also `'disparity'` and `'scale'`.

    Examples:
        ```python
        # Hyperalign using procrustes transform
        out = align(data, method='procrustes')

        # Align using shared response model
        out = align(data, method='probabilistic_srm', n_features=None)

        # Project aligned data back into original data space
        original_data = [
            np.dot(t.data, tm.T)
            for t, tm in zip(out['transformed'], out['transformation_matrix'])
        ]
        ```
    """

    from nltools.data import BrainData, Adjacency

    if not isinstance(data, list):
        raise ValueError("Make sure you are inputting data is a list.")
    if len({type(x) for x in data}) > 1:
        raise ValueError("Make sure all objects in the list are the same type.")
    if method not in ["probabilistic_srm", "deterministic_srm", "procrustes"]:
        raise ValueError(
            "Method must be ['probabilistic_srm','deterministic_srm','procrustes']"
        )

    if isinstance(data[0], BrainData):
        from nltools.data.braindata.utils import _result_from_array

        data_type = "BrainData"
        sources = data.copy()
        data = [np.array(x.data.T, copy=True) for x in data]
    elif isinstance(data[0], np.ndarray):
        data_type = "numpy"
        data = [np.array(x.T, copy=True) for x in data]
    else:
        raise ValueError(f"Type {type(data[0])} is not implemented yet.")

    # Align over time or voxels
    if axis == 1:
        data = [x.T for x in data]
    elif axis != 0:
        raise ValueError("axis must be 0 or 1.")

    out = {}
    if method in ["deterministic_srm", "probabilistic_srm"]:
        if n_features is None:
            n_features = int(data[0].shape[0])
        if method == "deterministic_srm":
            srm = DetSRM(features=n_features, *args, **kwargs)
        elif method == "probabilistic_srm":
            srm = SRM(features=n_features, *args, **kwargs)
        srm.fit(data)
        out["transformed"] = list(srm.transform(data))
        out["common_model"] = srm.s_.T
        out["transformation_matrix"] = srm.w_

    elif method == "procrustes":
        from nltools.algorithms import HyperAlignment

        if n_features is not None:
            raise NotImplementedError(
                "Currently must use all voxels."
                "Eventually will add a PCA reduction,"
                "must do this manually for now."
            )

        # Use HyperAlignment class for procrustes-based hyperalignment
        # Note: data is already transposed to [features, samples] format above (see the .T applied when loading each input)
        # n_iter=1 maintains backward compatibility with original implementation
        hyper = HyperAlignment(n_iter=1, auto_pad=True)
        hyper.fit(data)

        # Transform data to common space
        aligned = hyper.transform(data)

        # Extract attributes for output
        # Note: align() returns common_model in [samples, features] format (transposed)
        # but transformed in [features, samples] format (not transposed)
        out["transformed"] = aligned
        out["common_model"] = hyper.s_.T  # Transpose to [samples, features]
        out["transformation_matrix"] = hyper.w_
        out["disparity"] = hyper.disparity_
        out["scale"] = hyper.scale_

    if axis == 1:
        out["transformed"] = [x.T for x in out["transformed"]]
        out["common_model"] = out["common_model"].T

        if data_type == "BrainData":
            out["transformation_matrix"] = [x.T for x in out["transformation_matrix"]]

    if data_type == "BrainData":
        if method == "procrustes":
            out["transformed"] = [
                _result_from_array(source, values.T, rows="preserve")
                for source, values in zip(sources, out["transformed"])
            ]
            out["common_model"] = _result_from_array(
                sources[0], out["common_model"], rows="clear"
            )
        else:
            out["transformed"] = [x.T for x in out["transformed"]]

        out["transformation_matrix"] = [
            _result_from_array(source, values.T, rows="clear")
            for source, values in zip(sources, out["transformation_matrix"])
        ]

    # Calculate Intersubject Correlation (ISC) on final transformed data
    # ISC measures correlation along the aligned dimension:
    #   axis=0 (align timepoints): ISC per voxel (temporal correlation)
    #   axis=1 (align voxels): ISC per timepoint (spatial correlation)
    #
    # Final shapes after all formatting:
    #   BrainData: (timepoints, voxels)
    #   numpy: (voxels, timepoints)

    a = Adjacency()

    if data_type == "BrainData":
        # BrainData transformed shape: (timepoints, voxels)
        # For procrustes, transformed contains BrainData objects; extract .data
        # For SRM methods, transformed contains numpy arrays after the .T
        transformed_arrays = [
            x.data if isinstance(x, BrainData) else x for x in out["transformed"]
        ]
        if axis == 0:
            # Aligned timepoints → ISC per voxel (correlation over time)
            n_isc = transformed_arrays[0].shape[1]  # n_voxels
            for v in range(n_isc):
                # Extract timecourse for voxel v from each subject
                isc_data = np.array([x[:, v] for x in transformed_arrays])
                a = a.append(
                    Adjacency(
                        1 - pairwise_distances(isc_data, metric="correlation"),
                        matrix_type="similarity",
                    )
                )
        else:  # axis == 1
            # Aligned voxels → ISC per timepoint (spatial correlation)
            n_isc = transformed_arrays[0].shape[0]  # n_timepoints
            for t in range(n_isc):
                # Extract spatial pattern at timepoint t from each subject
                isc_data = np.array([x[t, :] for x in transformed_arrays])
                a = a.append(
                    Adjacency(
                        1 - pairwise_distances(isc_data, metric="correlation"),
                        matrix_type="similarity",
                    )
                )
    else:  # numpy
        # numpy transformed shape: (voxels, timepoints)
        if axis == 0:
            # Aligned timepoints → ISC per voxel (correlation over time)
            n_isc = out["transformed"][0].shape[0]  # n_voxels
            for v in range(n_isc):
                # Extract timecourse for voxel v from each subject
                isc_data = np.array([x[v, :] for x in out["transformed"]])
                a = a.append(
                    Adjacency(
                        1 - pairwise_distances(isc_data, metric="correlation"),
                        matrix_type="similarity",
                    )
                )
        else:  # axis == 1
            # Aligned voxels → ISC per timepoint (spatial correlation)
            n_isc = out["transformed"][0].shape[1]  # n_timepoints
            for t in range(n_isc):
                # Extract spatial pattern at timepoint t from each subject
                isc_data = np.array([x[:, t] for x in out["transformed"]])
                a = a.append(
                    Adjacency(
                        1 - pairwise_distances(isc_data, metric="correlation"),
                        matrix_type="similarity",
                    )
                )

    out["isc"] = dict(zip(np.arange(n_isc), a.mean(axis=1)))

    return out


def procrustes(data1, data2):
    """Perform a Procrustes similarity analysis on two data sets.

    For more comprehensive Procrustes-based alignment tasks, use
    `HyperAlignment` and `align()` instead.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both so that
    $tr(AA^{T}) = 1$ and both sets of points are centered around the origin.
    It then applies the optimal transform to the second matrix (including
    scaling/dilation, rotations, and reflections) to minimize
    $M^{2}=\\sum(data1-data2)^{2}$, the sum of squared pointwise differences
    between the two datasets. Both inputs must have the same number of rows;
    if they differ in the number of columns, the narrower one is padded with
    columns of zeros.

    Args:
        data1 (np.ndarray): Matrix whose n rows represent points in k (columns)
            space. `data1` is the reference data; after it is standardized, the
            data from `data2` will be transformed to fit the pattern in `data1`
            (must have >1 unique points).
        data2 (np.ndarray): n rows of data in k space to be fit to `data1`. Must
            have the same number of rows as `data1` (must have >1 unique points).

    Returns:
        tuple[np.ndarray, np.ndarray, float, np.ndarray, float]: `(mtx1, mtx2,
            disparity, R, scale)` — `mtx1` is a standardized version of `data1`;
            `mtx2` is the orientation of `data2` that best fits `data1` (centered,
            but not necessarily $tr(AA^{T}) = 1$); `disparity` is $M^{2}$ as defined
            above; `R` is the `(N, N)` matrix solution of the orthogonal Procrustes
            problem, minimizing the Frobenius norm of `dot(data1, R) - data2` subject
            to `dot(R.T, R) == I`; `scale` is the sum of the singular values of
            `dot(data1.T, data2)`.
    """

    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape[0] != mtx2.shape[0]:
        raise ValueError("Input matrices must have same number of rows.")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")
    if mtx1.shape[1] != mtx2.shape[1]:
        # Pad with zeros
        if mtx1.shape[1] > mtx2.shape[1]:
            mtx2 = np.append(
                mtx2, np.zeros((mtx1.shape[0], mtx1.shape[1] - mtx2.shape[1])), axis=1
            )
        else:
            mtx1 = np.append(
                mtx1, np.zeros((mtx1.shape[0], mtx2.shape[1] - mtx1.shape[1])), axis=1
            )

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, s


def procrustes_distance(
    mat1, mat2, *, n_permute=5000, tail=2, n_jobs=-1, random_state=None
):
    """Test matrix similarity using Procrustes superposition.

    Matrices need to match in size on their first dimension only, as the smaller
    matrix on the second dimension will be padded with zeros. After aligning two
    matrices using the Procrustes transformation, use the computed disparity
    between them (sum of squared error of elements) as a similarity metric.
    Shuffle the rows of one of the matrices and recompute the disparity to perform
    inference (Peres-Neto & Jackson, 2001).

    Args:
        mat1 (np.ndarray): 1d or 2d array; must have the same number of rows as
            `mat2`.
        mat2 (np.ndarray): 1d or 2d array; must have the same number of rows as
            `mat1`.
        n_permute (int): Number of permutation iterations. Defaults to 5000.
        tail (int | str): `2` or `'two'` for a two-tailed test (default); `1` or
            `'one'` for one-tailed (similarity greater than chance).
        n_jobs (int): Number of CPUs for the permutations; -1 (default) uses all.
        random_state (int | np.random.RandomState | None): Seed or generator for
            the row shuffling. Defaults to None.

    Returns:
        dict: Keys `'similarity'` (float in [0, 1], one minus the Procrustes
            disparity) and `'p'` (permutation p-value).
    """

    # raise NotImplementedError("procrustes distance is not currently implemented")
    if mat1.shape[0] != mat2.shape[0]:
        raise ValueError("Both arrays must match on their first dimension")

    random_state = check_random_state(random_state)

    # Make sure both matrices are 2d and the same dimension via padding
    validate_tail_parameter(tail)
    if len(mat1.shape) < 2:
        mat1 = mat1[:, np.newaxis]
    if len(mat2.shape) < 2:
        mat2 = mat2[:, np.newaxis]
    if mat1.shape[1] > mat2.shape[1]:
        mat2 = np.pad(mat2, ((0, 0), (0, mat1.shape[1] - mat2.shape[1])), "constant")
    elif mat2.shape[1] > mat1.shape[1]:
        mat1 = np.pad(mat1, ((0, 0), (0, mat2.shape[1] - mat1.shape[1])), "constant")

    # `procrust` (scipy.spatial.procrustes) returns a disparity in [0, 1] where
    # LOWER means more similar. Convert to a similarity (higher = more similar)
    # so the reported statistic matches the documented "similarity between 0 and
    # 1" and, critically, so the observed value and the permutation null live on
    # the SAME scale. Previously the observed disparity was compared against a
    # null of similarities, inverting the scales and yielding p ~ 1 for
    # near-identical matrices.
    _, _, disparity = procrust(mat1, mat2)
    observed_similarity = 1 - disparity

    null_disparities = Parallel(n_jobs=n_jobs)(
        delayed(procrust)(random_state.permutation(mat1), mat2)
        for _ in range(n_permute)
    )
    null_similarity = [1 - x[2] for x in null_disparities]

    # Use _compute_pvalue from inference module (signature: obs_stat, null_dist, tail)
    stats = {"similarity": float(observed_similarity)}
    stats["p"] = float(
        _compute_pvalue(
            np.array(observed_similarity), np.array(null_similarity), tail=tail
        )[0]
    )

    return stats


def align_states(
    reference,
    target,
    *,
    metric="correlation",
    return_index=False,
    replace_zero_variance=False,
):
    """Align state weight maps by minimizing pairwise distance between group states.

    This function uses the Hungarian algorithm for state alignment, which is
    different from aligning multiple subjects' data.

    Args:
        reference (np.ndarray): Reference pattern x state matrix.
        target (np.ndarray): Target pattern x state matrix to align to `reference`;
            must have the same shape.
        metric (str): Distance metric passed to `sklearn.metrics.pairwise_distances`.
            Defaults to `'correlation'`.
        return_index (bool): If True return the remapping index instead of the
            reordered data. Defaults to False.
        replace_zero_variance (bool): Replace zero-variance columns with uniform
            random numbers before computing distances; avoids NaNs with the
            correlation metric. Defaults to False.

    Returns:
        np.ndarray: If `return_index=False` (default), `target[:, remapping]` — the
            target's columns reordered to match the reference, oriented pattern x
            state (same shape as `target`). If `return_index=True`, the remapping
            index array that reorders the target's state columns.
    """
    if reference.shape != target.shape:
        raise ValueError("reference and target must be the same size")

    reference = np.array(reference)
    target = np.array(target)

    def replace_zero_variance_columns(data):
        """Replace zero-variance columns with random uniform noise.

        Prevents NaN values when correlation-based distance metrics encounter
        constant columns.

        Args:
            data (np.ndarray): 2-D array whose columns are checked for zero variance.

        Returns:
            np.ndarray: Array with zero-variance columns replaced by U(0, 1) values.
        """
        if np.any(data.std(axis=0) == 0):
            for i in np.where(data.std(axis=0) == 0)[0]:
                data[:, i] = np.random.uniform(low=0, high=1, size=data.shape[0])
        return data

    if replace_zero_variance:
        reference = replace_zero_variance_columns(reference)
        target = replace_zero_variance_columns(target)

    remapping = linear_sum_assignment(
        pairwise_distances(reference.T, target.T, metric=metric)
    )[1]

    if return_index:
        return remapping
    return target[:, remapping]
