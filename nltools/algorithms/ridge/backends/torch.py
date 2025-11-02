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
mean = torch.mean
std = torch.std
expand_dims = torch.unsqueeze
full = torch.full

###############################################################################
# Custom functions
###############################################################################


def atleast_1d(array):
    """Ensure array is at least 1D.

    Converts 0D scalars to 1D arrays. Higher-dimensional arrays are unchanged.

    Args:
        array: Input tensor (can be 0D scalar or higher).

    Returns:
        Tensor: Tensor with at least 1 dimension.

    Examples:
        >>> import torch
        >>> atleast_1d(torch.tensor(5.0)).shape
        torch.Size([1])
    """
    array = asarray(array)
    if array.ndim == 0:
        array = array[None]
    return array


def flip(array, axis=0):
    """Flip array along specified axis.

    Reverses the order of elements along the given axis.

    Args:
        array: Input tensor.
        axis: Axis along which to flip (default: 0).

    Returns:
        Tensor: Flipped tensor.

    Examples:
        >>> import torch
        >>> arr = torch.tensor([[1, 2], [3, 4]])
        >>> flip(arr, axis=0)
        tensor([[3, 4],
                [1, 2]])
    """
    return torch.flip(array, dims=[axis])


def sort(array, axis=-1):
    """Sort array along specified axis.

    Sorts elements along the given axis in ascending order.

    Args:
        array: Input tensor.
        axis: Axis along which to sort (default: -1, last axis).

    Returns:
        Tensor: Sorted tensor (values only, not indices).

    Examples:
        >>> import torch
        >>> arr = torch.tensor([3, 1, 2])
        >>> sort(arr)
        tensor([1, 2, 3])
    """
    return torch.sort(array, dim=axis).values


def to_numpy(array):
    """Convert tensor to numpy array.

    Transfers tensor to CPU and converts to NumPy array. If input is already
    a NumPy array, returns it unchanged.

    Args:
        array: Input tensor (PyTorch Tensor).

    Returns:
        ndarray: NumPy array representation of tensor.

    Examples:
        >>> import torch
        >>> tensor = torch.tensor([1, 2, 3])
        >>> arr = to_numpy(tensor)
        >>> type(arr)
        <class 'numpy.ndarray'>
    """
    try:
        return array.cpu().numpy()
    except AttributeError:
        return array


def to_cpu(array):
    """Transfer tensor to CPU.

    Moves tensor from GPU to CPU if necessary. If already on CPU, returns
    unchanged.

    Args:
        array: Input tensor (PyTorch Tensor).

    Returns:
        Tensor: Tensor on CPU device.

    Examples:
        >>> import torch
        >>> tensor = torch.tensor([1, 2, 3])
        >>> cpu_tensor = to_cpu(tensor)
        >>> cpu_tensor.device.type
        'cpu'
    """
    return array.cpu()


def to_gpu(array, device=None):
    """Transfer tensor to GPU (no-op for CPU backend).

    For CPU backend, this is a no-op and returns the input unchanged.

    Args:
        array: Input tensor (PyTorch Tensor).
        device: Ignored parameter (for compatibility with torch_cuda backend).

    Returns:
        Tensor: Input tensor (unchanged for CPU backend).

    Examples:
        >>> import torch
        >>> tensor = torch.tensor([1, 2, 3])
        >>> result = to_gpu(tensor)
        >>> result is tensor  # Same object
        True
    """
    return array


def is_in_gpu(array):
    """Check if tensor is on GPU.

    Determines whether tensor is located on a CUDA device.

    Args:
        array: Input tensor (PyTorch Tensor).

    Returns:
        bool: True if tensor is on CUDA device, False otherwise.

    Examples:
        >>> import torch
        >>> cpu_tensor = torch.tensor([1, 2, 3])
        >>> is_in_gpu(cpu_tensor)
        False
    """
    return array.device.type == "cuda"


def isin(x, y):
    """Element-wise test for membership.

    Tests whether each element of x is in y. Returns boolean tensor.

    Args:
        x: Input tensor to test.
        y: Set of values to test membership against.

    Returns:
        Tensor: Boolean tensor indicating membership.

    Notes:
        Currently uses NumPy for computation (may be optimized in future).
    """
    import numpy as np  # XXX

    np_result = np.isin(x.cpu().numpy(), y.cpu().numpy())
    return asarray(np_result, dtype=torch.bool, device=x.device)


def searchsorted(x, y):
    """Find indices where elements should be inserted to maintain order.

    Finds insertion points for elements of y into sorted array x.

    Args:
        x: Sorted 1D tensor.
        y: Values to find insertion points for.

    Returns:
        Tensor: Indices where elements should be inserted.

    Notes:
        Currently uses NumPy for computation (may be optimized in future).
    """
    import numpy as np  # XXX

    np_result = np.searchsorted(x.cpu().numpy(), y.cpu().numpy())
    return asarray(np_result, dtype=torch.int64, device=x.device)


def flatnonzero(x):
    """Return indices of non-zero elements in flattened array.

    Finds all non-zero elements in the flattened version of the tensor.

    Args:
        x: Input tensor.

    Returns:
        Tensor: 1D tensor containing indices of non-zero elements.

    Examples:
        >>> import torch
        >>> arr = torch.tensor([[0, 1, 0], [2, 0, 3]])
        >>> flatnonzero(arr)
        tensor([1, 3, 5])
    """
    return torch.nonzero(torch.flatten(x), as_tuple=True)[0]


def asarray(x, dtype=None, device="cpu"):
    """Convert input to tensor.

    Universal converter that handles numpy arrays, lists, torch tensors, and
    other array types. Attempts to preserve dtype and device information.

    Args:
        x: Input data (array-like). Can be numpy array, list, torch tensor,
            or other array-like object.
        dtype: Desired data type. If None, inferred from input.
        device: Device to place tensor on (default: "cpu").

    Returns:
        Tensor: PyTorch tensor representation of input.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> tensor = asarray(arr, dtype=torch.float32)
        >>> tensor.dtype
        torch.float32
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

    Convenience function to ensure dtype and device consistency with a
    reference tensor.

    Args:
        x: Input data (array-like).
        ref: Reference tensor for dtype and device inference.

    Returns:
        Tensor: Tensor with same dtype and device as ref.

    Examples:
        >>> import torch
        >>> ref = torch.tensor([1, 2, 3], dtype=torch.float32)
        >>> asarray_like([4, 5, 6], ref).dtype
        torch.float32
    """
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def norm(x, ord="fro", axis=None, keepdims=False):
    """Compute matrix or vector norm.

    Computes various types of norms (Frobenius, vector norms, etc.) along
    specified axes.

    Args:
        x: Input tensor.
        ord: Order of norm. Can be "fro" (Frobenius), int (vector norm), etc.
        axis: Axis or axes along which to compute norm. If None, computes
            overall norm.
        keepdims: Whether to keep dimensions in result.

    Returns:
        Tensor: Norm of x.

    Examples:
        >>> import torch
        >>> arr = torch.tensor([[1, 2], [3, 4]])
        >>> norm(arr, ord="fro")
        tensor(5.4772)
    """
    return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)


def copy(x):
    """Create a copy of tensor.

    Creates a deep copy of the tensor with independent memory.

    Args:
        x: Input tensor.

    Returns:
        Tensor: Copy of input tensor.

    Examples:
        >>> import torch
        >>> arr = torch.tensor([1, 2, 3])
        >>> arr_copy = copy(arr)
        >>> arr_copy[0] = 99
        >>> arr[0]  # Original unchanged
        tensor(1)
    """
    return x.clone()


def transpose(a, axes=None):
    """Transpose tensor.

    Transposes tensor along specified axes or swaps last two dimensions if
    axes is None.

    Args:
        a: Input tensor.
        axes: Permutation of axes. If None, swaps last two dimensions.

    Returns:
        Tensor: Transposed tensor.

    Examples:
        >>> import torch
        >>> arr = torch.tensor([[1, 2], [3, 4]])
        >>> transpose(arr)
        tensor([[1, 3],
                [2, 4]])
    """
    if axes is None:
        return a.t()
    else:
        return a.permute(*axes)


def max(*args, **kwargs):
    """Compute maximum.

    Computes maximum value, returning only the values (not indices).

    Args:
        *args: Positional arguments passed to torch.max.
        **kwargs: Keyword arguments passed to torch.max.

    Returns:
        Tensor: Maximum values (not indices).

    Examples:
        >>> import torch
        >>> arr = torch.tensor([[1, 5], [3, 2]])
        >>> max(arr, dim=0)
        tensor([3, 5])
    """
    res = torch.max(*args, **kwargs)
    if isinstance(res, torch.Tensor):
        return res
    else:
        return res.values


def min(*args, **kwargs):
    """Compute minimum.

    Computes minimum value, returning only the values (not indices).

    Args:
        *args: Positional arguments passed to torch.min.
        **kwargs: Keyword arguments passed to torch.min.

    Returns:
        Tensor: Minimum values (not indices).

    Examples:
        >>> import torch
        >>> arr = torch.tensor([[1, 5], [3, 2]])
        >>> min(arr, dim=0)
        tensor([1, 2])
    """
    res = torch.min(*args, **kwargs)
    if isinstance(res, torch.Tensor):
        return res
    else:
        return res.values


def zeros(shape, dtype="float32", device="cpu"):
    """Create tensor of zeros.

    Creates a new tensor filled with zeros on specified device.

    Args:
        shape: Shape of output tensor. Can be int or tuple.
        dtype: Data type (default: "float32").
        device: Device to place tensor on (default: "cpu").

    Returns:
        Tensor: Tensor filled with zeros.

    Examples:
        >>> import torch
        >>> zeros((2, 3))
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
    """
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def zeros_like(array, shape=None, dtype=None, device=None):
    """Create tensor of zeros with same properties as reference.

    Extended version of torch.zeros_like with additional shape parameter.
    Allows creating tensors with different shape but same dtype and device.

    Args:
        array: Reference tensor for dtype and device inference.
        shape: Shape of output. If None, uses array.shape.
        dtype: Data type. If None, uses array.dtype.
        device: Device. If None, uses array.device.

    Returns:
        Tensor: Tensor of zeros with specified shape, dtype, and device.

    Examples:
        >>> import torch
        >>> ref = torch.tensor([1, 2, 3])
        >>> zeros_like(ref, shape=(2, 3))
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
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
    """Create tensor of ones with same properties as reference.

    Extended version of torch.ones_like with additional shape parameter.
    Allows creating tensors with different shape but same dtype and device.

    Args:
        array: Reference tensor for dtype and device inference.
        shape: Shape of output. If None, uses array.shape.
        dtype: Data type. If None, uses array.dtype.
        device: Device. If None, uses array.device.

    Returns:
        Tensor: Tensor of ones with specified shape, dtype, and device.

    Examples:
        >>> import torch
        >>> ref = torch.tensor([1, 2, 3])
        >>> ones_like(ref, shape=(2, 3))
        tensor([[1., 1., 1.],
                [1., 1., 1.]])
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
    """Create tensor filled with value, with same properties as reference.

    Extended version of torch.full_like with additional shape parameter.
    Allows creating tensors with different shape but same dtype and device.

    Args:
        array: Reference tensor for dtype and device inference.
        fill_value: Scalar value to fill tensor with.
        shape: Shape of output. If None, uses array.shape.
        dtype: Data type. If None, uses array.dtype.
        device: Device. If None, uses array.device.

    Returns:
        Tensor: Tensor filled with fill_value, with specified shape, dtype, and device.

    Examples:
        >>> import torch
        >>> ref = torch.tensor([1, 2, 3])
        >>> full_like(ref, 42, shape=(2, 2))
        tensor([[42., 42.],
                [42., 42.]])
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
    """Convert all inputs to tensors with consistent dtype and device.

    Changes all inputs into Tensors (or list of Tensors) using the same
    precision and device as the first input. Some tensors can be None.

    Args:
        *all_inputs: Input arrays or lists of arrays. Can include None values.

    Returns:
        list: List of tensors with consistent dtype and device (matching first input).

    Examples:
        >>> import torch
        >>> arr1 = torch.tensor([1, 2], dtype=torch.float32)
        >>> arr2 = torch.tensor([3, 4], dtype=torch.float64)
        >>> result = check_arrays(arr1, arr2)
        >>> result[1].dtype == torch.float32  # Converted to match first
        True
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
        """Compute SVD (backward compatible).

        Computes singular value decomposition for PyTorch versions < 1.8.
        Uses torch.svd instead of torch.linalg.svd.

        Args:
            X: Input tensor (2D).
            full_matrices: Whether to compute full U and V matrices.

        Returns:
            tuple: (U, s, Vh) where:
                - U: Left singular vectors.
                - s: Singular values.
                - Vh: Right singular vectors (transposed).
        """
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
