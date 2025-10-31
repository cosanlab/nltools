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
    """Create random tensor on CUDA."""
    return torch.randn(*args, **kwargs).cuda()


def rand(*args, **kwargs):
    """Create random tensor on CUDA."""
    return torch.rand(*args, **kwargs).cuda()


def asarray(x, dtype=None, device="cuda"):
    """Convert input to CUDA tensor.

    Parameters
    ----------
    x : array-like
        Input data.
    dtype : dtype, optional
        Desired data type.
    device : str or torch.device
        Device to place tensor on (default: "cuda").

    Returns
    -------
    Tensor
        CUDA tensor representation of x.
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
    """Change all inputs into CUDA Tensors (or list of Tensors) using the same
    precision and device as the first one. Some tensors can be None.

    Warns if inputs are not float32 (GPU is faster with float32).

    Parameters
    ----------
    *all_inputs : array-like or None
        Input arrays or lists of arrays.

    Returns
    -------
    list
        List of CUDA tensors with consistent dtype and device.
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

    Parameters
    ----------
    shape : int or tuple
        Shape of output tensor.
    dtype : str or dtype
        Data type (default: "float32").
    device : str or torch.device
        Device to place tensor on (default: "cuda").

    Returns
    -------
    Tensor
        CUDA tensor of zeros.
    """
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def to_cpu(array):
    """Transfer tensor from GPU to CPU.

    Parameters
    ----------
    array : Tensor
        Input CUDA tensor.

    Returns
    -------
    Tensor
        Tensor on CPU.
    """
    return array.cpu()


def to_gpu(array, device="cuda"):
    """Transfer tensor to GPU.

    Parameters
    ----------
    array : Tensor or ndarray
        Input tensor or array.
    device : str or torch.device
        CUDA device (default: "cuda").

    Returns
    -------
    Tensor
        Tensor on CUDA device.
    """
    return asarray(array, device=device)
