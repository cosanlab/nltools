"""Data alignment — SRM, Procrustes, and state alignment."""

__all__ = ["align", "align_states", "procrustes", "procrustes_distance"]

import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes as procrust
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from ..algorithms.inference.utils import _compute_pvalue
from ..algorithms.alignment import SRM, DetSRM


def align(  # nosemgrep: kwargs-internal-forwarding  # forwards to the SRM/DetSRM algorithm constructors
    data, method="deterministic_srm", n_features=None, axis=0, *args, **kwargs
):
    """Align subject data into a common response model.

    This function is a convenience wrapper around `HyperAlignment` and `SRM` classes.

    Can be used to hyperalign source data to target data using
    Hyperalignment from Dartmouth (i.e., procrustes transformation; see
    nltools.stats.procrustes) or Shared Response Model from Princeton (see
    nltools.algorithms.srm). (see nltools.data.BrainData.align for aligning
    a single Brain object to another). Common Model is shared response
    model or centered target data. Transformed data can be back projected to
    original data using Tranformation matrix. Inputs must be a list of BrainData
    instances or numpy arrays (observations by features).


    Args:
        data: (list) A list of BrainData objects
        method: (str) alignment method to use
            ['probabilistic_srm','deterministic_srm','procrustes']
        n_features: (int) number of features to align to common space.
            If None then will select number of voxels
        axis: (int) axis to align on

    Returns:
        out: (dict) a dictionary containing a list of transformed subject
            matrices, a list of transformation matrices, the shared
            response matrix, and the intersubject correlation of the shared responses

    Examples:
        - Hyperalign using procrustes transform:
            >>> out = align(data, method='procrustes')
        - Align using shared response model:
            >>> out = align(data, method='probabilistic_srm', n_features=None)
        - Project aligned data into original data:
            >>> original_data = [np.dot(t.data,tm.T) for t,tm in zip(out['transformed'], out['transformation_matrix'])]
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

    data = deepcopy(data)

    if isinstance(data[0], BrainData):
        data_type = "BrainData"
        data_out = [x.copy() for x in data]
        transformation_out = [x.copy() for x in data]
        data = [x.data.T for x in data]
    elif isinstance(data[0], np.ndarray):
        data_type = "numpy"
        data = [x.T for x in data]
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
        # Note: data is already transposed to [features, samples] format by line 1330/1327
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
            for i, x in enumerate(out["transformed"]):
                data_out[i].data = x.T
                out["transformed"] = data_out
            common = data_out[0].copy()
            common.data = out["common_model"]
            out["common_model"] = common
        else:
            out["transformed"] = [x.T for x in out["transformed"]]

        for i, x in enumerate(out["transformation_matrix"]):
            transformation_out[i].data = x.T
        out["transformation_matrix"] = transformation_out

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
    two identically sized matrices, procrustes standardizes both such that:
    - $tr(AA^{T}) = 1$.
    - Both sets of points are centered around the origin.
    Procrustes then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    $M^{2}=\\sum(data1-data2)^{2}$, or the sum of the squares of the
    pointwise differences between the two input datasets.
    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), this function will add columns of zeros to
    the smaller of the two.

    Args:
        data1: Matrix whose n rows represent points in k (columns) space.
            `data1` is the reference data; after it is standardized, the data
            from `data2` will be transformed to fit the pattern in `data1`
            (must have >1 unique points).
        data2: n rows of data in k space to be fit to `data1`. Must be the same
            shape `(numrows, numcols)` as `data1` (must have >1 unique points).

    Returns:
        mtx1: A standardized version of `data1`.
        mtx2: The orientation of `data2` that best fits `data1`. Centered, but
            not necessarily $tr(AA^{T}) = 1$.
        disparity: $M^{2}$ as defined above.
        R: The `(N, N)` matrix solution of the orthogonal Procrustes problem.
            Minimizes the Frobenius norm of `dot(data1, R) - data2`, subject to
            `dot(R.T, R) == I`.
        scale: Sum of the singular values of `dot(data1.T, data2)`.
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
        mat1 (ndarray): 2d numpy array; must have same number of rows as mat2
        mat2 (ndarray): 1d or 2d numpy array; must have same number of rows as mat1
        n_permute (int): number of permutation iterations to perform
        tail (int): either 1 for one-tailed or 2 for two-tailed test; default 2
        n_jobs (int): The number of CPUs to use to do permutation; default -1 (all)
        random_state (int, np.random.RandomState, or None): seed or generator for
            the permutation shuffling; default None

    Returns:
        dict: results with keys `similarity` (float in [0, 1]) and `p` (permuted p-value)

    """

    # raise NotImplementedError("procrustes distance is not currently implemented")
    if mat1.shape[0] != mat2.shape[0]:
        raise ValueError("Both arrays must match on their first dimension")

    random_state = check_random_state(random_state)

    # Make sure both matrices are 2d and the same dimension via padding
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
        reference: (np.array) reference pattern x state matrix
        target: (np.array) target pattern x state matrix to align to reference
        metric: (str) distance metric to use
        return_index: (bool) return index if True, return remapped data if False
        replace_zero_variance: (bool) transform a vector with zero variance to random numbers from a uniform distribution.
                                Useful for when using correlation as a distance metric to avoid NaNs.
    Returns:
        ordered_weights: (list) a list of reordered state X pattern matrices

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
            data: 2-D array whose columns are checked for zero variance.

        Returns:
            Array with zero-variance columns replaced by U(0, 1) random values.
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
