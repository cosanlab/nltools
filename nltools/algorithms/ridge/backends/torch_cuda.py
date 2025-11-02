"""The "torch_cuda" GPU backend, based on PyTorch.

To use this backend, call ``nltools.algorithms.ridge.backends.set_backend("torch_cuda")``.

Requires PyTorch with CUDA to be installed.
"""

from .torch import *  # noqa
import torch

if not torch.cuda.is_available():
    import sys

    if "pytest" in sys.modules:  # if run through pytest
        import pytest

        pytest.skip("PyTorch with CUDA is not available.")
    raise RuntimeError("PyTorch with CUDA is not available.")

from ._utils import _dtype_to_str, warn_if_not_float32

###############################################################################
# Backend name
###############################################################################

name = "torch_cuda"

###############################################################################
# Override functions to default to CUDA device
###############################################################################


def randn(*args, **kwargs):
    """Create random tensor on CUDA.

    Creates tensor filled with random values from standard normal distribution
    and places it on CUDA device.

    Args:
        *args: Positional arguments passed to torch.randn (shape).
        **kwargs: Keyword arguments passed to torch.randn.

    Returns:
        Tensor: CUDA tensor filled with random values.

    Examples:
        >>> randn(2, 3).device.type
        'cuda'
    """
    return torch.randn(*args, **kwargs).cuda()


def rand(*args, **kwargs):
    """Create random tensor on CUDA.

    Creates tensor filled with random values from uniform distribution [0, 1)
    and places it on CUDA device.

    Args:
        *args: Positional arguments passed to torch.rand (shape).
        **kwargs: Keyword arguments passed to torch.rand.

    Returns:
        Tensor: CUDA tensor filled with random values.

    Examples:
        >>> rand(2, 3).device.type
        'cuda'
    """
    return torch.rand(*args, **kwargs).cuda()


def asarray(x, dtype=None, device="cuda"):
    """Convert input to CUDA tensor.

    Universal converter that handles numpy arrays, lists, torch tensors, and
    other array types. By default, places tensors on CUDA device.

    Args:
        x: Input data (array-like). Can be numpy array, list, torch tensor,
            or other array-like object.
        dtype: Desired data type. If None, inferred from input.
        device: Device to place tensor on (default: "cuda").

    Returns:
        Tensor: CUDA tensor representation of input.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> tensor = asarray(arr)
        >>> tensor.device.type
        'cuda'
    """
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        if hasattr(x, "dtype") and hasattr(x.dtype, "name"):
            dtype = x.dtype.name
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = getattr(torch, dtype)
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = "cuda"
    try:
        tensor = torch.as_tensor(x, dtype=dtype, device=device)
    except Exception:
        import numpy as np

        array = np.asarray(x, dtype=_dtype_to_str(dtype))
        tensor = torch.as_tensor(array, dtype=dtype, device=device)
    return tensor


def check_arrays(*all_inputs):
    """Convert all inputs to CUDA tensors with consistent dtype and device.

    Changes all inputs into CUDA Tensors (or list of Tensors) using the same
    precision and device as the first input. Some tensors can be None.

    Warns if inputs are not float32 (GPU is faster with float32).

    Args:
        *all_inputs: Input arrays or lists of arrays. Can include None values.

    Returns:
        list: List of CUDA tensors with consistent dtype and device.

    Examples:
        >>> import torch
        >>> arr1 = torch.tensor([1, 2], dtype=torch.float32)
        >>> arr2 = torch.tensor([3, 4], dtype=torch.float64)
        >>> result = check_arrays(arr1, arr2)
        >>> result[0].device.type  # All on CUDA
        'cuda'
    """
    all_tensors = []
    all_tensors.append(asarray(all_inputs[0]))
    dtype = all_tensors[0].dtype
    warn_if_not_float32(dtype)
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


def zeros(shape, dtype="float32", device="cuda"):
    """Create tensor of zeros on CUDA.

    Creates a new tensor filled with zeros on CUDA device.

    Args:
        shape: Shape of output tensor. Can be int or tuple.
        dtype: Data type (default: "float32").
        device: Device to place tensor on (default: "cuda").

    Returns:
        Tensor: CUDA tensor filled with zeros.

    Examples:
        >>> zeros((2, 3)).device.type
        'cuda'
    """
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def to_cpu(array):
    """Transfer tensor from GPU to CPU.

    Moves CUDA tensor to CPU. If already on CPU, returns unchanged.

    Args:
        array: Input CUDA tensor (PyTorch Tensor).

    Returns:
        Tensor: Tensor on CPU device.

    Examples:
        >>> import torch
        >>> tensor = torch.tensor([1, 2, 3], device="cuda")
        >>> cpu_tensor = to_cpu(tensor)
        >>> cpu_tensor.device.type
        'cpu'
    """
    return array.cpu()


def to_gpu(array, device="cuda"):
    """Transfer tensor to GPU.

    Moves tensor or numpy array to CUDA device. If already on GPU, returns
    unchanged.

    Args:
        array: Input tensor or numpy array.
        device: CUDA device (default: "cuda").

    Returns:
        Tensor: Tensor on CUDA device.

    Examples:
        >>> import torch
        >>> tensor = torch.tensor([1, 2, 3])
        >>> gpu_tensor = to_gpu(tensor)
        >>> gpu_tensor.device.type
        'cuda'
    """
    return asarray(array, device=device)
