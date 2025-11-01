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
    """Convert array to numpy (no-op for numpy arrays)."""
    return array


def zeros_like(array, shape=None, dtype=None, device=None):
    """Add a shape parameter in zeros_like.

    Parameters
    ----------
    array : ndarray
        Reference array.
    shape : tuple, optional
        Shape of output array. If None, uses array.shape.
    dtype : dtype, optional
        Data type of output. If None, uses array.dtype.
    device : ignored
        Device parameter (ignored for numpy backend).

    Returns
    -------
    ndarray
        Array of zeros.
    """
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.zeros(shape, dtype=dtype)


def ones_like(array, shape=None, dtype=None, device=None):
    """Add a shape parameter in ones_like.

    Parameters
    ----------
    array : ndarray
        Reference array.
    shape : tuple, optional
        Shape of output array. If None, uses array.shape.
    dtype : dtype, optional
        Data type of output. If None, uses array.dtype.
    device : ignored
        Device parameter (ignored for numpy backend).

    Returns
    -------
    ndarray
        Array of ones.
    """
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.ones(shape, dtype=dtype)


def full_like(array, fill_value, shape=None, dtype=None, device=None):
    """Add a shape parameter in full_like.

    Parameters
    ----------
    array : ndarray
        Reference array.
    fill_value : scalar
        Fill value.
    shape : tuple, optional
        Shape of output array. If None, uses array.shape.
    dtype : dtype, optional
        Data type of output. If None, uses array.dtype.
    device : ignored
        Device parameter (ignored for numpy backend).

    Returns
    -------
    ndarray
        Array filled with fill_value.
    """
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.full(shape, fill_value, dtype=dtype)


def to_cpu(array):
    """Transfer array to CPU (no-op for numpy)."""
    return array


def to_gpu(array, device=None):
    """Transfer array to GPU (no-op for numpy)."""
    return array


def is_in_gpu(array):
    """Check if array is in GPU (always False for numpy)."""
    return False


def asarray_like(x, ref):
    """Convert x to array with same dtype as ref.

    Parameters
    ----------
    x : array-like
        Input data.
    ref : ndarray
        Reference array for dtype.

    Returns
    -------
    ndarray
        Array with same dtype as ref.
    """
    return np.asarray(x, dtype=ref.dtype)


def check_arrays(*all_inputs):
    """Change all inputs into arrays (or list of arrays) using the same
    precision as the first one. Some arrays can be None.

    Parameters
    ----------
    *all_inputs : array-like or None
        Input arrays or lists of arrays.

    Returns
    -------
    list
        List of arrays with consistent dtype.
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
    """Convert input to array.

    This function can convert from numpy, lists, torch, and other array types.

    Parameters
    ----------
    a : array-like
        Input data.
    dtype : dtype, optional
        Desired data type.
    order : {'C', 'F'}, optional
        Memory layout.
    device : ignored
        Device parameter (ignored for numpy backend).

    Returns
    -------
    ndarray
        Array representation of a.
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

    Parameters
    ----------
    X : ndarray
        Input matrix (2D or 3D).
    full_matrices : bool
        Whether to compute full U and V matrices.

    Returns
    -------
    U : ndarray
        Left singular vectors.
    s : ndarray
        Singular values.
    Vt : ndarray
        Right singular vectors (transposed).
    """
    if X.ndim == 2 or not use_scipy:
        return linalg.svd(X, full_matrices=full_matrices)

    elif X.ndim == 3:
        UsV_list = [linalg.svd(Xi, full_matrices=full_matrices) for Xi in X]
        return map(np.stack, zip(*UsV_list))
    else:
        raise NotImplementedError("SVD only supports 2D and 3D arrays")
