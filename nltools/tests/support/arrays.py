"""Array assertions for tests that run across compute backends."""

import numpy as np

from nltools.utils import _find_stack_level


def assert_array_almost_equal(x, y, decimal=6, err_msg="", verbose=True, backend=None):
    """Assert two arrays are almost equal, relaxing precision for the MPS backend.

    A test helper: on `torch-mps` (float32 only) `decimal` is capped at 2 with a
    warning, so the same assertion holds across backends. Torch tensors are
    moved to the CPU and converted before comparison.

    Args:
        x (np.ndarray | torch.Tensor): First array to compare.
        y (np.ndarray | torch.Tensor): Second array to compare.
        decimal (int): Desired decimal precision. Defaults to 6.
        err_msg (str): Error message prefix. Defaults to `""`.
        verbose (bool): Whether to include the mismatching values in the error.
            Defaults to True.
        backend (Backend | None): Backend the arrays came from. If None, an MPS
            tensor is detected from `x`.

    Raises:
        AssertionError: If the arrays don't match.
    """
    # Auto-detect backend from x if possible
    if backend is None:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                if x.device.type == "mps":
                    backend_name = "torch-mps"
                else:
                    backend_name = None
            else:
                backend_name = None
        except (ImportError, AttributeError):
            backend_name = None
    else:
        backend_name = getattr(backend, "name", None)

    # Auto-adjust precision for torch_mps backend
    if backend_name == "torch-mps":
        if decimal > 2:
            import warnings

            warnings.warn(
                f"Reducing precision from decimal={decimal} to decimal=2 for "
                "torch-mps backend due to float32 conversion limitations",
                UserWarning,
                stacklevel=_find_stack_level(),
            )
            decimal = 2

    # Convert to numpy if needed
    if backend is not None:
        x = backend.to_numpy(x) if hasattr(backend, "to_numpy") else x
        y = backend.to_numpy(y) if hasattr(backend, "to_numpy") else y
    else:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
        except (ImportError, AttributeError):
            pass

    return np.testing.assert_array_almost_equal(
        x, y, decimal=decimal, err_msg=err_msg, verbose=verbose
    )
