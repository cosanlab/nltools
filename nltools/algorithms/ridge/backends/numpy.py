"""The "numpy" CPU backend, based on NumPy.

To use this backend, call ``nltools.algorithms.ridge.backends.set_backend("numpy")``.

This is the default backend and is always available.
"""

import numpy as np

try:
    import scipy.linalg as linalg

    use_scipy = True
except ImportError:
    import numpy.linalg as linalg

    use_scipy = False

###############################################################################
# Backend name
###############################################################################

name = "numpy"

###############################################################################
# Basic operations - direct assignments from numpy
###############################################################################

argmax = np.argmax
max = np.max
min = np.min
abs = np.abs
randn = np.random.randn
rand = np.random.rand
matmul = np.matmul
transpose = np.transpose
stack = np.stack
concatenate = np.concatenate
sum = np.sum
sqrt = np.sqrt
any = np.any
all = np.all
nan = np.nan
inf = np.inf
isnan = np.isnan
isinf = np.isinf
logspace = np.logspace
copy = np.copy
bool = np.bool_
float32 = np.float32
float64 = np.float64
int32 = np.int32
eigh = linalg.eigh
norm = linalg.norm
log = np.log
exp = np.exp
arange = np.arange
flatnonzero = np.flatnonzero
isin = np.isin
searchsorted = np.searchsorted
unique = np.unique
einsum = np.einsum
tanh = np.tanh
power = np.power
prod = np.prod
zeros = np.zeros
clip = np.clip
sign = np.sign
sort = np.sort
flip = np.flip
atleast_1d = np.atleast_1d
finfo = np.finfo
eye = np.eye
mean = np.mean
std = np.std
expand_dims = np.expand_dims
full = np.full

###############################################################################
# Custom functions
###############################################################################


def to_numpy(array):
    """Convert array to numpy (no-op for numpy arrays).

    This function is a no-op for numpy arrays, but provides a consistent interface
    for backend-agnostic code. Other backends (torch, torch_cuda) override this
    to convert their arrays to numpy.

    Args:
        array: Input array (numpy ndarray).

    Returns:
        ndarray: The input array unchanged (no conversion needed).

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> result = to_numpy(arr)
        >>> result is arr  # Same object (no copy)
        True
    """
    return array


def zeros_like(array, shape=None, dtype=None, device=None):
    """Create array of zeros with same shape and dtype as reference array.

    Extended version of numpy.zeros_like with additional shape parameter.
    Allows creating arrays with different shape but same dtype as reference.

    Args:
        array: Reference array for dtype inference.
        shape: Shape of output array. If None, uses array.shape.
        dtype: Data type of output. If None, uses array.dtype.
        device: Ignored parameter (for compatibility with torch backends).

    Returns:
        ndarray: Array of zeros with specified shape and dtype.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> zeros_like(arr, shape=(5, 4))
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
    """
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.zeros(shape, dtype=dtype)


def ones_like(array, shape=None, dtype=None, device=None):
    """Create array of ones with same shape and dtype as reference array.

    Extended version of numpy.ones_like with additional shape parameter.
    Allows creating arrays with different shape but same dtype as reference.

    Args:
        array: Reference array for dtype inference.
        shape: Shape of output array. If None, uses array.shape.
        dtype: Data type of output. If None, uses array.dtype.
        device: Ignored parameter (for compatibility with torch backends).

    Returns:
        ndarray: Array of ones with specified shape and dtype.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> ones_like(arr, shape=(2, 3))
        array([[1., 1., 1.],
               [1., 1., 1.]])
    """
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.ones(shape, dtype=dtype)


def full_like(array, fill_value, shape=None, dtype=None, device=None):
    """Create array filled with value, with same shape and dtype as reference array.

    Extended version of numpy.full_like with additional shape parameter.
    Allows creating arrays with different shape but same dtype as reference.

    Args:
        array: Reference array for dtype inference.
        fill_value: Scalar value to fill array with.
        shape: Shape of output array. If None, uses array.shape.
        dtype: Data type of output. If None, uses array.dtype.
        device: Ignored parameter (for compatibility with torch backends).

    Returns:
        ndarray: Array filled with fill_value, with specified shape and dtype.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> full_like(arr, 42, shape=(2, 2))
        array([[42., 42.],
               [42., 42.]])
    """
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.full(shape, fill_value, dtype=dtype)


def to_cpu(array):
    """Transfer array to CPU (no-op for numpy arrays).

    Provides consistent interface for backend-agnostic code. Since numpy arrays
    are always on CPU, this is a no-op.

    Args:
        array: Input numpy array.

    Returns:
        ndarray: The input array unchanged (already on CPU).

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> result = to_cpu(arr)
        >>> result is arr  # Same object
        True
    """
    return array


def to_gpu(array, device=None):
    """Transfer array to GPU (no-op for numpy backend).

    Provides consistent interface for backend-agnostic code. Since numpy backend
    doesn't support GPU, this is a no-op.

    Args:
        array: Input numpy array.
        device: Ignored parameter (for compatibility with torch backends).

    Returns:
        ndarray: The input array unchanged (numpy doesn't support GPU).

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> result = to_gpu(arr)
        >>> result is arr  # Same object
        True
    """
    return array


def is_in_gpu(array):
    """Check if array is in GPU (always False for numpy backend).

    Provides consistent interface for backend-agnostic code. Since numpy arrays
    are always on CPU, this always returns False.

    Args:
        array: Input numpy array.

    Returns:
        bool: Always False for numpy backend.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> is_in_gpu(arr)
        False
    """
    return False


def asarray_like(x, ref):
    """Convert x to array with same dtype as ref.

    Convenience function to ensure dtype consistency with a reference array.

    Args:
        x: Input data (array-like).
        ref: Reference array for dtype inference.

    Returns:
        ndarray: Array with same dtype as ref.

    Examples:
        >>> import numpy as np
        >>> ref = np.array([1, 2, 3], dtype=np.float32)
        >>> asarray_like([4, 5, 6], ref)
        array([4., 5., 6.], dtype=float32)
    """
    return np.asarray(x, dtype=ref.dtype)


def check_arrays(*all_inputs):
    """Convert all inputs to arrays with consistent dtype.

    Changes all inputs into arrays (or list of arrays) using the same precision
    as the first input. Some arrays can be None. Useful for ensuring dtype
    consistency across multiple inputs.

    Args:
        *all_inputs: Input arrays or lists of arrays. Can include None values.

    Returns:
        list: List of arrays with consistent dtype (matching first input).

    Examples:
        >>> import numpy as np
        >>> arr1 = np.array([1, 2], dtype=np.float32)
        >>> arr2 = np.array([3, 4], dtype=np.float64)
        >>> result = check_arrays(arr1, arr2)
        >>> result[1].dtype == np.float32  # Converted to match first
        True
    """
    all_arrays = []
    all_arrays.append(asarray(all_inputs[0]))
    dtype = all_arrays[0].dtype
    for tensor in all_inputs[1:]:
        if tensor is None:
            pass
        elif isinstance(tensor, list):
            tensor = [asarray(tt, dtype=dtype) for tt in tensor]
        else:
            tensor = asarray(tensor, dtype=dtype)
        all_arrays.append(tensor)
    return all_arrays


def asarray(a, dtype=None, order=None, device=None):
    """Convert input to numpy array.

    Universal converter that can handle numpy arrays, lists, torch tensors,
    cupy arrays, and other array types. Attempts multiple conversion strategies.

    Args:
        a: Input data (array-like). Can be numpy array, list, torch tensor,
            cupy array, or other array-like object.
        dtype: Desired data type. If None, inferred from input.
        order: Memory layout ('C' for C-order, 'F' for Fortran-order).
        device: Ignored parameter (for compatibility with torch backends).

    Returns:
        ndarray: NumPy array representation of input.

    Examples:
        >>> import numpy as np
        >>> asarray([1, 2, 3])
        array([1, 2, 3])
        >>> asarray([1, 2, 3], dtype=np.float32)
        array([1., 2., 3.], dtype=float32)
    """
    # works from numpy, lists, torch, and others
    try:
        return np.asarray(a, dtype=dtype, order=order)
    except Exception:
        pass
    # works from cupy
    try:
        import cupy

        return np.asarray(cupy.asnumpy(a), dtype=dtype, order=order)
    except Exception:
        pass
    # works from torch_cuda
    try:
        return np.asarray(a.cpu(), dtype=dtype, order=order)
    except Exception:
        pass

    return np.asarray(a, dtype=dtype, order=order)


def svd(X, full_matrices=True):
    """Compute singular value decomposition.

    Computes SVD for 2D or 3D arrays. For 3D arrays, computes SVD for each
    matrix along the first dimension.

    Args:
        X: Input matrix, shape (n_samples, n_features) for 2D, or
            (n_matrices, n_samples, n_features) for 3D.
        full_matrices: Whether to compute full U and V matrices. If False,
            computes only the necessary columns.

    Returns:
        tuple: (U, s, Vt) where:
            - U: Left singular vectors, shape (n_samples, n_samples) or (n_samples, min(n_samples, n_features)).
            - s: Singular values, shape (min(n_samples, n_features),).
            - Vt: Right singular vectors (transposed), shape (min(n_samples, n_features), n_features) or (n_features, n_features).

        For 3D input, returns stacked arrays with shape (n_matrices, ...).

    Raises:
        NotImplementedError: If input has more than 3 dimensions.

    Examples:
        >>> import numpy as np
        >>> X = np.random.randn(10, 5)
        >>> U, s, Vt = svd(X, full_matrices=False)
        >>> U.shape, s.shape, Vt.shape
        ((10, 5), (5,), (5, 5))
    """
    if X.ndim == 2 or not use_scipy:
        return linalg.svd(X, full_matrices=full_matrices)

    elif X.ndim == 3:
        UsV_list = [linalg.svd(Xi, full_matrices=full_matrices) for Xi in X]
        return map(np.stack, zip(*UsV_list))
    else:
        raise NotImplementedError("SVD only supports 2D and 3D arrays")
