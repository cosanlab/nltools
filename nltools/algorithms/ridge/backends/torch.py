"""The "torch" CPU backend, based on PyTorch.

To use this backend, call ``nltools.algorithms.ridge.backends.set_backend("torch")``.

Requires PyTorch to be installed.
"""

try:
    import torch
except ImportError as error:
    import sys

    if "pytest" in sys.modules:  # if run through pytest
        import pytest

        pytest.skip("PyTorch not installed.")
    raise ImportError("PyTorch not installed.") from error

from ._utils import _dtype_to_str

###############################################################################
# Backend name
###############################################################################

name = "torch"

###############################################################################
# Basic operations - direct assignments from torch
###############################################################################

argmax = torch.argmax
randn = torch.randn
rand = torch.rand
matmul = torch.matmul
stack = torch.stack
abs = torch.abs
sum = torch.sum
sqrt = torch.sqrt
any = torch.any
all = torch.all
nan = torch.tensor(float("nan"))
inf = torch.tensor(float("inf"))
isnan = torch.isnan
isinf = torch.isinf
logspace = torch.logspace
concatenate = torch.cat
bool = torch.bool
int32 = torch.int32
float32 = torch.float32
float64 = torch.float64
log = torch.log
exp = torch.exp
arange = torch.arange
unique = torch.unique
einsum = torch.einsum
tanh = torch.tanh
power = torch.pow
prod = torch.prod
sign = torch.sign
clip = torch.clamp
finfo = torch.finfo
eye = torch.eye

###############################################################################
# Custom functions
###############################################################################


def atleast_1d(array):
    """Ensure array is at least 1D."""
    array = asarray(array)
    if array.ndim == 0:
        array = array[None]
    return array


def flip(array, axis=0):
    """Flip array along axis."""
    return torch.flip(array, dims=[axis])


def sort(array, axis=-1):
    """Sort array along axis."""
    return torch.sort(array, dim=axis).values


def to_numpy(array):
    """Convert tensor to numpy array.

    Parameters
    ----------
    array : Tensor
        Input tensor.

    Returns
    -------
    ndarray
        NumPy array.
    """
    try:
        return array.cpu().numpy()
    except AttributeError:
        return array


def to_cpu(array):
    """Transfer tensor to CPU.

    Parameters
    ----------
    array : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor on CPU.
    """
    return array.cpu()


def to_gpu(array, device=None):
    """Transfer tensor to GPU (no-op for CPU backend).

    For CPU backend, this is a no-op and returns the input unchanged.

    Parameters
    ----------
    array : Tensor
        Input tensor.
    device : ignored
        Device parameter (ignored for CPU backend).

    Returns
    -------
    Tensor
        Input tensor (unchanged).
    """
    return array


def is_in_gpu(array):
    """Check if tensor is in GPU.

    Parameters
    ----------
    array : Tensor
        Input tensor.

    Returns
    -------
    bool
        True if tensor is on CUDA device.
    """
    return array.device.type == "cuda"


def isin(x, y):
    """Element-wise test for membership."""
    import numpy as np  # XXX

    np_result = np.isin(x.cpu().numpy(), y.cpu().numpy())
    return asarray(np_result, dtype=torch.bool, device=x.device)


def searchsorted(x, y):
    """Find indices where elements should be inserted to maintain order."""
    import numpy as np  # XXX

    np_result = np.searchsorted(x.cpu().numpy(), y.cpu().numpy())
    return asarray(np_result, dtype=torch.int64, device=x.device)


def flatnonzero(x):
    """Return indices of non-zero elements in flattened array."""
    return torch.nonzero(torch.flatten(x), as_tuple=True)[0]


def asarray(x, dtype=None, device="cpu"):
    """Convert input to tensor.

    Parameters
    ----------
    x : array-like
        Input data.
    dtype : dtype, optional
        Desired data type.
    device : str or torch.device
        Device to place tensor on.

    Returns
    -------
    Tensor
        Tensor representation of x.
    """
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        if hasattr(x, "dtype") and hasattr(x.dtype, "name"):
            dtype = x.dtype.name
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = getattr(torch, dtype)
    if device is None and isinstance(x, torch.Tensor):
        device = x.device

    try:
        tensor = torch.as_tensor(x, dtype=dtype, device=device)
    except Exception:
        import numpy as np

        array = np.asarray(x, dtype=_dtype_to_str(dtype))
        tensor = torch.as_tensor(array, dtype=dtype, device=device)
    return tensor


def asarray_like(x, ref):
    """Convert x to tensor with same dtype and device as ref.

    Parameters
    ----------
    x : array-like
        Input data.
    ref : Tensor
        Reference tensor for dtype and device.

    Returns
    -------
    Tensor
        Tensor with same dtype and device as ref.
    """
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def norm(x, ord="fro", axis=None, keepdims=False):
    """Compute matrix or vector norm.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    ord : str or int
        Order of norm.
    axis : int or tuple, optional
        Axis along which to compute norm.
    keepdims : bool
        Whether to keep dimensions.

    Returns
    -------
    Tensor
        Norm of x.
    """
    return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)


def copy(x):
    """Create a copy of tensor."""
    return x.clone()


def transpose(a, axes=None):
    """Transpose tensor.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    axes : tuple, optional
        Permutation of axes.

    Returns
    -------
    Tensor
        Transposed tensor.
    """
    if axes is None:
        return a.t()
    else:
        return a.permute(*axes)


def max(*args, **kwargs):
    """Compute maximum.

    Returns values only (not indices).
    """
    res = torch.max(*args, **kwargs)
    if isinstance(res, torch.Tensor):
        return res
    else:
        return res.values


def min(*args, **kwargs):
    """Compute minimum.

    Returns values only (not indices).
    """
    res = torch.min(*args, **kwargs)
    if isinstance(res, torch.Tensor):
        return res
    else:
        return res.values


def zeros(shape, dtype="float32", device="cpu"):
    """Create tensor of zeros.

    Parameters
    ----------
    shape : int or tuple
        Shape of output tensor.
    dtype : str or dtype
        Data type.
    device : str or torch.device
        Device to place tensor on.

    Returns
    -------
    Tensor
        Tensor of zeros.
    """
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def zeros_like(array, shape=None, dtype=None, device=None):
    """Add a shape parameter in zeros_like.

    Parameters
    ----------
    array : Tensor
        Reference tensor.
    shape : int or tuple, optional
        Shape of output. If None, uses array.shape.
    dtype : dtype, optional
        Data type. If None, uses array.dtype.
    device : str or torch.device, optional
        Device. If None, uses array.device.

    Returns
    -------
    Tensor
        Tensor of zeros.
    """
    if shape is None:
        shape = array.shape
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype is None:
        dtype = array.dtype
    if device is None:
        device = array.device
    return torch.zeros(shape, dtype=dtype, device=device, layout=array.layout)


def ones_like(array, shape=None, dtype=None, device=None):
    """Add a shape parameter in ones_like.

    Parameters
    ----------
    array : Tensor
        Reference tensor.
    shape : int or tuple, optional
        Shape of output. If None, uses array.shape.
    dtype : dtype, optional
        Data type. If None, uses array.dtype.
    device : str or torch.device, optional
        Device. If None, uses array.device.

    Returns
    -------
    Tensor
        Tensor of ones.
    """
    if shape is None:
        shape = array.shape
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype is None:
        dtype = array.dtype
    if device is None:
        device = array.device
    return torch.ones(shape, dtype=dtype, device=device, layout=array.layout)


def full_like(array, fill_value, shape=None, dtype=None, device=None):
    """Add a shape parameter in full_like.

    Parameters
    ----------
    array : Tensor
        Reference tensor.
    fill_value : scalar
        Fill value.
    shape : int or tuple, optional
        Shape of output. If None, uses array.shape.
    dtype : dtype, optional
        Data type. If None, uses array.dtype.
    device : str or torch.device, optional
        Device. If None, uses array.device.

    Returns
    -------
    Tensor
        Tensor filled with fill_value.
    """
    if shape is None:
        shape = array.shape
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype is None:
        dtype = array.dtype
    if device is None:
        device = array.device
    return torch.full(
        shape, fill_value, dtype=dtype, device=device, layout=array.layout
    )


def check_arrays(*all_inputs):
    """Change all inputs into Tensors (or list of Tensors) using the same
    precision and device as the first one. Some tensors can be None.

    Parameters
    ----------
    *all_inputs : array-like or None
        Input arrays or lists of arrays.

    Returns
    -------
    list
        List of tensors with consistent dtype and device.
    """
    all_tensors = []
    all_tensors.append(asarray(all_inputs[0]))
    dtype = all_tensors[0].dtype
    device = all_tensors[0].device
    for tensor in all_inputs[1:]:
        if tensor is None:
            pass
        elif isinstance(tensor, list):
            tensor = [asarray(tt, dtype=dtype, device=device) for tt in tensor]
        else:
            tensor = asarray(tensor, dtype=dtype, device=device)
        all_tensors.append(tensor)
    return all_tensors


# SVD with version compatibility
try:
    svd = torch.linalg.svd
except AttributeError:
    # torch.__version__ < 1.8

    def svd(X, full_matrices=True):
        """Compute SVD (backward compatible)."""
        U, s, V = torch.svd(X, some=not full_matrices)
        Vh = V.transpose(-2, -1)
        return U, s, Vh


# eigh (eigenvalue decomposition)
try:
    eigh = torch.linalg.eigh
except AttributeError:
    # torch.__version__ < 1.8
    from functools import partial

    eigh = partial(torch.symeig, eigenvectors=True)
