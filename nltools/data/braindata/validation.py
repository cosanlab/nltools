"""
Validation utilities for BrainData class.

This module contains helper functions for validating inputs, shapes, and
compatibility between BrainData objects and other data types.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl


def validate_frame(frame, data_shape=None, frame_type="DataFrame"):
    """Validate and process X or Y dataframes for BrainData.

    Accepts pandas DataFrames for user convenience but always returns a
    polars DataFrame. Internal BrainData state should be polars-only.

    Args:
        frame: Input to validate. Can be ``None``, a ``str``/``Path`` pointing
            to a CSV, a polars or pandas DataFrame, or a 1D/2D numpy array.
        data_shape: Optional tuple of data shape to validate row count against.
        frame_type: Type of frame for error messages (e.g., "X", "Y").

    Returns:
        pl.DataFrame: Validated frame as polars. Empty ``pl.DataFrame()`` when
        ``frame`` is ``None``.

    Raises:
        TypeError: If frame is not a supported type.
        ValueError: If frame rows do not match ``data_shape[0]`` or CSV read fails.
    """
    if frame is None:
        return pl.DataFrame()

    if isinstance(frame, pl.DataFrame):
        out = frame
    elif isinstance(frame, (str, Path)):
        try:
            out = pl.read_csv(frame, has_header=False)
        except Exception as e:
            raise ValueError(
                f"Could not read {frame_type} from file '{frame}'. "
                f"Make sure the file exists and is a valid CSV. Error: {e}"
            )
    elif isinstance(frame, np.ndarray):
        arr = frame if frame.ndim == 2 else frame.reshape(-1, 1)
        out = pl.DataFrame(arr)
    else:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(frame, pd.DataFrame):
            out = pl.DataFrame({str(c): frame[c].to_numpy() for c in frame.columns})
        else:
            raise TypeError(
                f"{frame_type} must be a filepath (str/Path), numpy array, or "
                f"polars/pandas DataFrame. Received {type(frame).__name__}"
            )

    if not out.is_empty() and data_shape is not None:
        if out.shape[0] != data_shape[0]:
            raise ValueError(
                f"{frame_type} rows ({out.shape[0]}) do not match "
                f"data rows ({data_shape[0]}). Each row in {frame_type} should "
                f"correspond to an image in the data."
            )

    return out


def validate_brain_data_shapes(brain1, brain2, operation="operation"):
    """Validate shape compatibility between two BrainData objects.

    Args:
        brain1: First BrainData object.
        brain2: Second BrainData object.
        operation: Name of operation for error messages.

    Returns:
        tuple: (brain1_is_single, brain2_is_single) booleans.

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
        other: The operand to validate.
        operation_name: Name of operation (e.g., 'add', 'multiply').

    Returns:
        str: Type of operand ('scalar', 'brain_data', or 'array').

    Raises:
        ValueError: If operand type is not supported.
    """
    # Import here to avoid circular imports
    from nltools.data import BrainData

    if isinstance(other, (int, np.integer, float, np.floating)):
        return "scalar"
    elif isinstance(other, BrainData):
        return "brain_data"
    elif isinstance(other, (list, np.ndarray)) and operation_name == "multiply":
        return "array"
    else:
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
        data: Input data to validate.

    Returns:
        str: Type of data ('brain_data', 'list', 'h5', 'url', 'file', 'nibabel', 'none').

    Raises:
        TypeError: If data type is not supported.
    """
    # Import here to avoid circular imports
    from nltools.data import BrainData

    if data is None:
        return "none"
    elif isinstance(data, BrainData):
        return "brain_data"
    elif isinstance(data, list):
        return "list"
    elif isinstance(data, (str, Path)):
        from nltools.io import is_h5_path

        data_str = str(data)
        if is_h5_path(data_str):
            return "h5"
        elif "://" in data_str:
            return "url"
        else:
            return "file"
    elif isinstance(data, nib.Nifti1Image):
        return "nibabel"
    else:
        raise TypeError(
            f"Data must be a BrainData, filepath (str/Path), nibabel image, "
            f"or list of these types. Received {type(data).__name__}"
        )


def validate_list_data(data_list):
    """Validate that all items in a list are the same type.

    Args:
        data_list: List to validate.

    Returns:
        str: Type of items ('brain_data' or 'file').

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
    elif isinstance(data_list[0], (str, Path, nib.Nifti1Image)):
        return "file"
    else:
        raise ValueError(
            f"List items must be BrainData objects, file paths, or nibabel images. "
            f"Found {first_type.__name__}"
        )


def validate_index_operations(data_shape, index):
    """Validate indexing operations for BrainData.

    Args:
        data_shape: Shape of the data array.
        index: Index to validate.

    Returns:
        str: Type of indexing ('single', 'slice', 'array').

    Raises:
        IndexError: If index is out of bounds.
    """
    n_images = data_shape[0] if len(data_shape) > 1 else 1

    if isinstance(index, (int, np.integer)):
        if index < -n_images or index >= n_images:
            raise IndexError(
                f"Index {index} is out of bounds for data with {n_images} images"
            )
        return "single"
    elif isinstance(index, slice):
        return "slice"
    else:
        # Convert to array for validation
        try:
            index_array = np.array(index).flatten()
            if np.any((index_array < -n_images) | (index_array >= n_images)):
                raise IndexError(
                    f"Some indices are out of bounds for data with {n_images} images"
                )
            return "array"
        except (ValueError, TypeError):
            raise TypeError(
                f"Index must be int, slice, or array-like. "
                f"Received {type(index).__name__}"
            )


def validate_append_shapes(data1_shape, data2_shape):
    """Validate shape compatibility for appending BrainData objects.

    Args:
        data1_shape: Shape of first BrainData.
        data2_shape: Shape of second BrainData to append.

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
