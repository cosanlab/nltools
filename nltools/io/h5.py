"""
HDF5 I/O utilities for nltools data types.

Shared serialization logic for BrainData and Adjacency objects.
"""

__all__ = ["is_h5_path", "to_h5", "load_brain_data_h5"]

import h5py
import nibabel as nib
import numpy as np
import polars as pl
from h5py import File as h5File


def is_h5_path(file_name) -> bool:
    """Check if a file path indicates an HDF5 file.

    Args:
        file_name: Path to check (str or Path object).

    Returns:
        bool: True if the file has an HDF5 extension (.h5 or .hdf5).

    Examples:
        >>> is_h5_path("data.h5")
        True
        >>> is_h5_path("data.csv")
        False
        >>> is_h5_path(Path("results.hdf5"))
        True
    """
    from pathlib import Path

    if isinstance(file_name, Path):
        file_name = str(file_name)
    return ".h5" in file_name or ".hdf5" in file_name


def _write_polars_frame(h5_file, name, df, compression):
    """Write a polars DataFrame to an h5 group as columns + values datasets."""
    group = h5_file.create_group(name)
    columns = list(df.columns)
    group.create_dataset(
        "columns",
        data=np.array(columns, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    if df.is_empty() or len(columns) == 0:
        group.create_dataset("values", data=np.zeros((0, len(columns))))
    else:
        group.create_dataset("values", data=df.to_numpy(), compression=compression)


def _read_polars_frame(h5_file, name):
    """Read a polars DataFrame from an h5 group written by _write_polars_frame."""
    if name not in h5_file:
        return pl.DataFrame()
    group = h5_file[name]
    columns = [
        c.decode("utf-8") if isinstance(c, bytes) else str(c)
        for c in np.array(group["columns"])
    ]
    values = np.array(group["values"])
    if values.size == 0 or len(columns) == 0:
        return pl.DataFrame({c: [] for c in columns}) if columns else pl.DataFrame()
    return pl.DataFrame(values, schema=columns)


def to_h5(obj, file_name, obj_type="brain_data", h5_compression="gzip"):
    """Save BrainData or Adjacency objects to HDF5 files.

    Uses h5py for brain_data (X/Y stored as polars-compatible groups) and
    pandas HDFStore for adjacency (pending Chunk B migration).

    Args:
        obj: Object to save (BrainData or Adjacency).
        file_name: Path to save file to.
        obj_type: Type of object ('brain_data' or 'adjacency').
        h5_compression: Compression type for h5py datasets.
    """
    if obj_type not in ["brain_data", "adjacency"]:
        raise TypeError("obj_type must be one of 'brain_data' or 'adjacency'")

    if obj_type == "brain_data":
        with h5File(file_name, "w") as f:
            f.create_dataset("data", data=obj.data, compression=h5_compression)
            f.create_dataset(
                "mask_affine", data=obj.mask.affine, compression=h5_compression
            )
            f.create_dataset(
                "mask_data", data=obj.mask.get_fdata(), compression=h5_compression
            )
            f.create_dataset("mask_file_name", data=obj.mask.get_filename())
            _write_polars_frame(f, "X", obj.X, h5_compression)
            _write_polars_frame(f, "Y", obj.Y, h5_compression)
    else:
        import pandas as pd
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tables")
            warnings.filterwarnings(
                "ignore", message=".*performance.*", module="tables"
            )
            with pd.HDFStore(file_name, "w") as f:
                f["Y"] = obj.Y

        with h5File(file_name, "a") as f:
            f.create_dataset("data", data=obj.data, compression=h5_compression)
            f.create_dataset("matrix_type", data=obj.matrix_type)
            f.create_dataset("issymmetric", data=obj.issymmetric)
            f.create_dataset("labels", data=obj.labels)
            f.create_dataset("is_single_matrix", data=obj.is_single_matrix)


def load_brain_data_h5(file_path, mask=None):
    """Load BrainData from HDF5 file.

    Handles the v0.6 h5py layout (X/Y as groups with columns + values) and
    falls back to the pre-0.4.8 PyTables layout if the modern groups are
    missing.

    Args:
        file_path: Path to HDF5 file.
        mask: Optional mask to use. If None, loads mask from file if available.

    Returns:
        dict: Dictionary containing loaded data, X, Y, and optionally mask info.
    """
    result = {}

    try:
        with h5File(file_path, "r") as f:
            if "X" not in f or "Y" not in f or "data" not in f:
                raise KeyError("missing expected groups")
            result["data"] = np.array(f["data"])
            result["X"] = _read_polars_frame(f, "X")
            result["Y"] = _read_polars_frame(f, "Y")

            if mask is None and "mask_data" in f:
                result["mask"] = nib.Nifti1Image(
                    np.array(f["mask_data"]),
                    affine=np.array(f["mask_affine"]),
                    file_map={
                        "image": nib.FileHolder(
                            filename=f["mask_file_name"][()].decode()
                        )
                    },
                )
                result["load_mask"] = True
            else:
                result["load_mask"] = False

    except (OSError, KeyError):
        result = _load_legacy_brain_data_h5(file_path, mask)
        result["legacy_format"] = True

    return result


def _load_legacy_brain_data_h5(file_path, mask=None):
    """Load BrainData from legacy HDF5 format (pre-0.4.8)."""
    from nltools.utils import attempt_to_import

    tables_mod = attempt_to_import("tables")
    if tables_mod is None:
        raise ImportError("tables package required for legacy h5 format")

    result = {}

    with tables_mod.open_file(file_path, mode="r") as f:
        result["data"] = np.array(f.root["data"])

        if len(list(f.root["X_columns"])):
            result["X"] = pl.DataFrame(
                np.array(f.root["X"]).squeeze(),
                schema=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["X_columns"])
                ],
            )
        else:
            result["X"] = pl.DataFrame()

        if len(list(f.root["Y_columns"])):
            result["Y"] = pl.DataFrame(
                np.array(f.root["Y"]).squeeze(),
                schema=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["Y_columns"])
                ],
            )
        else:
            result["Y"] = pl.DataFrame()

        if mask is None and "mask_data" in f.root:
            filename = (
                f.root["mask_file_name"]
                if "mask_file_name" in f.root
                else "mask.nii.gz"
            )
            result["mask"] = nib.Nifti1Image(
                np.array(f.root["mask_data"]),
                affine=np.array(f.root["mask_affine"]),
                file_map={"image": nib.FileHolder(filename=filename)},
            )
            result["load_mask"] = True
        else:
            result["load_mask"] = False

    return result
