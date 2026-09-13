"""Input validation for `BrainData`.

Helpers that validate constructor inputs, array shapes, and operand
compatibility between `BrainData` objects and other data types.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def validate_brain_data_shapes(brain1, brain2, operation="operation"):
    """Validate shape compatibility between two BrainData objects.

    Args:
        brain1 (BrainData): First operand.
        brain2 (BrainData): Second operand.
        operation (str): Name of the operation for error messages.

    Returns:
        tuple[bool, bool]: ``(brain1_is_single, brain2_is_single)``.

    Raises:
        ValueError: If shapes are incompatible for the operation.
    """
    shape1, shape2 = brain1.shape, brain2.shape
    brain1_is_single = len(shape1) == 1
    brain2_is_single = len(shape2) == 1

    if brain1_is_single and brain2_is_single:
        if shape1[0] != shape2[0]:
            raise ValueError(
                f"Cannot {operation}: both images must have the same number of voxels. "
                f"Image 1 has {shape1[0]} voxels, Image 2 has {shape2[0]} voxels."
            )
    elif brain1_is_single and not brain2_is_single:
        raise ValueError(
            f"Cannot {operation} multiple images to a single image. "
            f"Image 1 is single, Image 2 has {shape2[0]} images."
        )
    elif not brain1_is_single and brain2_is_single:
        if shape1[1] != shape2[0]:
            raise ValueError(
                f"Cannot {operation}: number of voxels must match. "
                f"Image 1 has {shape1[1]} voxels, Image 2 has {shape2[0]} voxels."
            )
    elif not brain1_is_single and not brain2_is_single:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            raise ValueError(
                f"Cannot {operation} multiple images of different shapes. "
                f"Image 1 shape: {shape1}, Image 2 shape: {shape2}"
            )

    return brain1_is_single, brain2_is_single


def validate_arithmetic_operand(other, operation_name):
    """Validate operand type for arithmetic operations.

    Args:
        other (object): The operand to validate.
        operation_name (str): Name of the operation (e.g. ``'add'``,
            ``'multiply'``).

    Returns:
        str: Type of operand ('scalar', 'brain_data', or 'array').

    Raises:
        ValueError: If operand type is not supported.
    """
    # Import here to avoid circular imports
    from nltools.data import BrainData

    if isinstance(other, (int, np.integer, float, np.floating)):
        return "scalar"
    if isinstance(other, BrainData):
        return "brain_data"
    if isinstance(other, (list, np.ndarray)) and operation_name == "multiply":
        return "array"
    valid_types = "int, float, or BrainData"
    if operation_name == "multiply":
        valid_types = "int, float, list, np.ndarray, or BrainData"
    raise ValueError(
        f"Cannot {operation_name} with type {type(other).__name__}. "
        f"Operand must be {valid_types}."
    )


def validate_data_type(data):
    """Validate input data type for BrainData initialization.

    Args:
        data (object): Constructor input to classify.

    Returns:
        str: One of ``'brain_data'``, ``'list'``, ``'h5'``, ``'url'``,
            ``'file'``, ``'nibabel'``, ``'array'``, or ``'none'``.

    Raises:
        TypeError: If data type is not supported.
    """
    # Import here to avoid circular imports
    from nltools.data import BrainData

    if data is None:
        return "none"
    if isinstance(data, BrainData):
        return "brain_data"
    if isinstance(data, list):
        return "list"
    if isinstance(data, (str, Path)):
        from nltools.io.h5 import is_h5_path

        data_str = str(data)
        if is_h5_path(data_str):
            return "h5"
        if "://" in data_str:
            return "url"
        return "file"
    if isinstance(data, nib.Nifti1Image):
        return "nibabel"
    if isinstance(data, np.ndarray):
        return "array"
    raise TypeError(
        f"Data must be a BrainData, filepath (str/Path), nibabel image, "
        f"numpy array, or list of these types. Received {type(data).__name__}"
    )


def validate_list_data(data_list):
    """Validate that all items in a list are the same type.

    Args:
        data_list (list): Items to validate.

    Returns:
        str: ``'brain_data'`` or ``'file'``.

    Raises:
        ValueError: If list contains mixed types or unsupported types.
    """
    if not data_list:
        raise ValueError("List is empty")

    # Import here to avoid circular imports
    from nltools.data import BrainData

    first_type = type(data_list[0])

    # Check if all items are the same type
    if not all(isinstance(x, first_type) for x in data_list):
        raise ValueError(
            "All items in the list must be the same type. "
            "Found mixed types in the list."
        )

    # Determine what type we're dealing with
    if isinstance(data_list[0], BrainData):
        return "brain_data"
    if isinstance(data_list[0], (str, Path, nib.Nifti1Image)):
        return "file"
    raise ValueError(
        f"List items must be BrainData objects, file paths, or nibabel images. "
        f"Found {first_type.__name__}"
    )


def validate_append_shapes(data1_shape, data2_shape):
    """Validate shape compatibility for appending BrainData objects.

    Args:
        data1_shape (tuple[int, ...]): Shape of the first BrainData.
        data2_shape (tuple[int, ...]): Shape of the BrainData being appended.

    Raises:
        ValueError: If shapes are incompatible for appending.
    """
    data1_is_single = len(data1_shape) == 1
    data2_is_single = len(data2_shape) == 1

    error_msg = (
        f"Cannot append: incompatible number of voxels. "
        f"Data 1 shape: {data1_shape}, Data 2 shape: {data2_shape}"
    )

    if data1_is_single and data2_is_single:
        if data1_shape[0] != data2_shape[0]:
            raise ValueError(error_msg)
    elif data1_is_single and not data2_is_single:
        if data1_shape[0] != data2_shape[1]:
            raise ValueError(error_msg)
    elif not data1_is_single and data2_is_single:
        if data1_shape[1] != data2_shape[0]:
            raise ValueError(error_msg)
    elif data1_shape[1] != data2_shape[1]:
        raise ValueError(error_msg)
