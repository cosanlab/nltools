"""HDF5 I/O utilities for nltools data types.

Shared serialization logic for BrainData and Adjacency objects.
"""

import io
from pathlib import Path, PureWindowsPath

import nibabel as nib
import numpy as np
import polars as pl

try:
    import h5py
    from h5py import File as h5File
except ImportError as _h5_import_error:
    h5py = None  # type: ignore[assignment]
    h5File = None  # type: ignore[assignment]
    _H5_IMPORT_ERROR: ImportError | None = _h5_import_error
else:
    _H5_IMPORT_ERROR = None

#: The compression filters `to_h5` accepts — h5py's own, needing no plugin.
_SUPPORTED_COMPRESSION = ("gzip", "lzf")


def _require_h5():
    """Raise a friendly error if h5py isn't installed."""
    if _H5_IMPORT_ERROR is not None:
        raise ImportError(
            "HDF5 I/O requires h5py. Install with: pip install 'nltools[h5]'"
        ) from _H5_IMPORT_ERROR


def _reject_legacy_h5(source, legacy_marker):
    """Raise when an open HDF5 file carries the nltools 0.5.1 deepdish layout.

    0.5.1 stored each frame as a flat dataset beside a sibling
    `<name>_columns` node; v0.6.0 reads only its own layout.

    Args:
        source (h5py.File): Open HDF5 file to inspect.
        legacy_marker (str): Top-level node that only the 0.5.1 layout has.

    Raises:
        ValueError: If the file was written by nltools 0.5.1 or earlier.
    """
    if legacy_marker in source:
        raise ValueError(
            "This HDF5 file was written by nltools 0.5.1 or earlier, a layout "
            "v0.6.0 no longer reads. Open it under 0.5.1 and export the data "
            "first — BrainData.write('x.nii.gz') for images, "
            "Adjacency.write('x.csv') for matrices — then load the export."
        )


def _validate_compression(compression):
    """Reject a compression filter h5py does not provide on its own.

    Args:
        compression (str): Value passed as `h5_compression`.

    Raises:
        ValueError: If `compression` is not one of `_SUPPORTED_COMPRESSION`.
    """
    if compression not in _SUPPORTED_COMPRESSION:
        raise ValueError(
            f"h5_compression must be one of {_SUPPORTED_COMPRESSION}; "
            f"got {compression!r}."
        )


def is_h5_path(file_name) -> bool:
    """Check if a file path indicates an HDF5 file.

    Args:
        file_name (str | Path): Path to check.

    Returns:
        bool: True if the file has an HDF5 extension (`.h5` or `.hdf5`).

    Examples:
        ```python
        is_h5_path("data.h5")  # → True
        is_h5_path("data.csv")  # → False
        is_h5_path(Path("results.hdf5"))  # → True
        ```
    """
    if isinstance(file_name, Path):
        file_name = str(file_name)
    return file_name.lower().endswith((".h5", ".hdf5"))


def _mask_basename(stored_name):
    """Reduce a stored mask filename to its basename.

    Files written before the basename rule — and files written on Windows,
    whose separators mean nothing to `pathlib` on POSIX — carry a full path, so
    both separators are stripped here.
    """
    if "\\" in stored_name:
        return PureWindowsPath(stored_name).name
    return Path(stored_name).name


def _write_polars_frame(h5_file, name, df, compression):
    """Write a polars DataFrame to an h5 dataset as raw IPC bytes.

    Uses polars' Arrow IPC format so every dtype — strings, booleans, nulls,
    and mixed-type frames — round-trips exactly. h5py handles the resulting
    byte buffer directly.
    """
    buf = io.BytesIO()
    df.write_ipc(buf)
    h5_file.create_dataset(
        name,
        data=np.frombuffer(buf.getvalue(), dtype=np.uint8),
        compression=compression,
    )


def _read_polars_frame(h5_file, name):
    """Read a polars DataFrame from an h5 dataset written by _write_polars_frame."""
    if name not in h5_file:
        return pl.DataFrame()
    return pl.read_ipc(io.BytesIO(np.asarray(h5_file[name]).tobytes()))


def to_h5(obj, file_name, obj_type="brain_data", h5_compression="gzip"):
    """Save BrainData or Adjacency objects to HDF5 files.

    Uses h5py for both types; the `X`/`Y` frames (BrainData) and `Y` (Adjacency)
    are stored as Arrow IPC byte datasets so every polars dtype round-trips
    exactly. A BrainData mask is always stored by value (data + affine
    datasets); the basename of its filename is stored alongside only when the
    mask is file-backed, so in-memory masks serialize without one and round-trip
    by value.

    Args:
        obj (BrainData | Adjacency): Object to save.
        file_name (str | Path): Path to save the file to.
        obj_type (str): `'brain_data'` or `'adjacency'`.
        h5_compression (str): Compression filter for h5py datasets, `'gzip'`
            (default) or `'lzf'`.

    Raises:
        ValueError: If `obj_type` or `h5_compression` is not one of the
            supported values.
    """
    _require_h5()
    _validate_compression(h5_compression)
    if obj_type not in ["brain_data", "adjacency"]:
        raise ValueError("obj_type must be one of 'brain_data' or 'adjacency'")

    if obj_type == "brain_data":
        with h5File(file_name, "w") as f:
            f.create_dataset("data", data=obj.data, compression=h5_compression)
            f.create_dataset(
                "mask_affine", data=obj.mask.affine, compression=h5_compression
            )
            f.create_dataset(
                "mask_data", data=obj.mask.get_fdata(), compression=h5_compression
            )
            mask_file_name = obj.mask.get_filename()
            if mask_file_name is not None:
                # In-memory masks have no filename; the mask still round-trips
                # by value via the mask_data + mask_affine datasets above. Only
                # the basename is stored: the writer's directory layout means
                # nothing on another machine, and nothing reopens the name.
                f.create_dataset("mask_file_name", data=_mask_basename(mask_file_name))
            _write_polars_frame(f, "X", obj.X, h5_compression)
            _write_polars_frame(f, "Y", obj.Y, h5_compression)
    else:
        with h5File(file_name, "w") as f:
            f.create_dataset("data", data=obj.data, compression=h5_compression)
            f.create_dataset("matrix_type", data=obj.matrix_type)
            f.create_dataset("issymmetric", data=obj.issymmetric)
            f.create_dataset("is_single_matrix", data=obj.is_single_matrix)
            if obj.labels:
                f.create_dataset(
                    "labels",
                    data=np.asarray(obj.labels, dtype=object)
                    if np.asarray(obj.labels).dtype.kind in "US"
                    else np.asarray(obj.labels),
                    dtype=h5py.string_dtype(encoding="utf-8")
                    if np.asarray(obj.labels).dtype.kind in "US"
                    else None,
                )
            else:
                f.create_dataset("labels", data=np.array([], dtype="float64"))
            _write_polars_frame(f, "Y", obj.Y, h5_compression)


def load_brain_data_h5(file_path, mask=None):
    """Load BrainData contents from an HDF5 file.

    Reads the v0.6 layout only (`X`/`Y` as Arrow IPC byte datasets); a file
    written by nltools 0.5.1 or earlier raises. A stored mask filename is
    reduced to its basename — the embedded mask data and affine are
    authoritative and the name is never reopened.

    Args:
        file_path (str | Path): Path to the HDF5 file.
        mask (nibabel.Nifti1Image, optional): Mask to use. If None, the mask stored
            in the file is loaded when present.

    Returns:
        dict: Keys `'data'` (np.ndarray), `'X'` and `'Y'` (pl.DataFrame),
            `'load_mask'` (bool), and `'mask'` (nibabel.Nifti1Image) when a mask was
            loaded from the file.

    Raises:
        ValueError: If the file was written by nltools 0.5.1 or earlier.
    """
    _require_h5()
    with h5File(file_path, "r") as f:
        _reject_legacy_h5(f, "X_columns")

        result = {}
        result["data"] = np.array(f["data"])
        result["X"] = _read_polars_frame(f, "X")
        result["Y"] = _read_polars_frame(f, "Y")

        if mask is None and "mask_data" in f:
            if "mask_file_name" in f:
                # Mask originally file-backed: keep the filename association,
                # reduced to a basename so a file written before that rule
                # stops reporting the writer's parent directory.
                file_map = {
                    "image": nib.FileHolder(
                        filename=_mask_basename(f["mask_file_name"][()].decode())
                    )
                }
            else:
                # Mask was in-memory at write time: reconstruct by value.
                file_map = None
            result["mask"] = nib.Nifti1Image(
                np.array(f["mask_data"]),
                affine=np.array(f["mask_affine"]),
                file_map=file_map,
            )
            result["load_mask"] = True
        else:
            result["load_mask"] = False

    return result
