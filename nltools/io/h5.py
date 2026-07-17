"""HDF5 I/O utilities for nltools data types.

Shared serialization logic for BrainData and Adjacency objects.
"""

__all__ = ["is_h5_path", "load_brain_data_h5", "to_h5"]

import io
import warnings

import nibabel as nib
import numpy as np
import polars as pl

try:
    import h5py
    import hdf5plugin  # noqa: F401  -- registers blosc/zstd/lz4 filters with h5py
    from h5py import File as h5File
except ImportError as _h5_import_error:
    h5py = None  # type: ignore[assignment]
    h5File = None  # type: ignore[assignment]
    _H5_IMPORT_ERROR: ImportError | None = _h5_import_error
else:
    _H5_IMPORT_ERROR = None


def _require_h5():
    """Raise a friendly error if h5py/hdf5plugin aren't installed."""
    if _H5_IMPORT_ERROR is not None:
        raise ImportError(
            "HDF5 I/O requires h5py and hdf5plugin. "
            "Install with: pip install 'nltools[h5]'"
        ) from _H5_IMPORT_ERROR


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
    return file_name.lower().endswith((".h5", ".hdf5"))


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

    Uses h5py for both types; X/Y (BrainData) and Y (Adjacency) are stored
    as polars-compatible groups with ``columns`` and ``values`` datasets.

    Args:
        obj: Object to save (BrainData or Adjacency).
        file_name: Path to save file to.
        obj_type: Type of object ('brain_data' or 'adjacency').
        h5_compression: Compression type for h5py datasets.
    """
    _require_h5()
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
        with h5File(file_name, "w") as f:
            f.create_dataset("data", data=obj.data, compression=h5_compression)
            f.create_dataset("matrix_type", data=obj.matrix_type)
            f.create_dataset("issymmetric", data=obj.issymmetric)
            f.create_dataset("is_single_matrix", data=obj.is_single_matrix)
            if obj.labels:
                f.create_dataset(
                    "labels",
                    data=np.array(obj.labels, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            else:
                f.create_dataset("labels", data=np.array([], dtype="float64"))
            _write_polars_frame(f, "Y", obj.Y, h5_compression)


def load_brain_data_h5(file_path, mask=None):
    """Load BrainData from HDF5 file.

    Supports the v0.6 layout (X/Y as h5py groups with ``columns`` + ``values``)
    and the legacy deepdish/PyTables layout written by nltools <= 0.5.1
    (X/Y as flat datasets with sibling ``X_columns``/``X_index`` nodes).

    Args:
        file_path: Path to HDF5 file.
        mask: Optional mask to use. If None, loads mask from file if available.

    Returns:
        dict: Dictionary containing loaded data, X, Y, and optionally mask info.
    """
    _require_h5()
    with h5File(file_path, "r") as f:
        if _is_legacy_brain_data_layout(f):
            return _load_legacy_brain_data_h5(f, mask)

        result = {}
        result["data"] = np.array(f["data"])
        result["X"] = _read_polars_frame(f, "X")
        result["Y"] = _read_polars_frame(f, "Y")

        if mask is None and "mask_data" in f:
            result["mask"] = nib.Nifti1Image(
                np.array(f["mask_data"]),
                affine=np.array(f["mask_affine"]),
                file_map={
                    "image": nib.FileHolder(filename=f["mask_file_name"][()].decode())
                },
            )
            result["load_mask"] = True
        else:
            result["load_mask"] = False

    return result


def _is_legacy_brain_data_layout(f) -> bool:
    """Detect pre-0.6 deepdish/PyTables layout.

    Modern files have X as a Group containing ``columns``/``values`` children.
    Legacy files have X as a flat Dataset with a sibling ``X_columns`` node.
    """
    return "X_columns" in f


def _read_legacy_pytables_list(node):
    """Read a deepdish-encoded sequence from a legacy h5 node.

    deepdish wraps numpy arrays as Datasets and Python lists as PyTables Groups
    (with ``TITLE='list:N'``). When the original list was empty, the group has
    zero children — return ``[]``. Otherwise return a list of decoded values.
    """
    if isinstance(node, h5py.Group):
        if len(node) == 0:
            return []
        # Non-empty list-of-objects (rare for nltools-written files; columns
        # were always numpy arrays via _df_meta_to_arr). Read children in order.
        items = []
        for key in sorted(node.keys()):
            child = node[key]
            if isinstance(child, h5py.Dataset):
                val = child[()]
                items.append(val.decode("utf-8") if isinstance(val, bytes) else val)
        return items

    arr = np.asarray(node)
    return [v.decode("utf-8") if isinstance(v, bytes) else v for v in arr]


def _read_legacy_frame(f, name):
    """Reconstruct a polars DataFrame from legacy ``<name>``/``<name>_columns`` nodes."""
    cols_key = f"{name}_columns"
    if cols_key not in f:
        return pl.DataFrame()

    columns = _read_legacy_pytables_list(f[cols_key])
    if not columns:
        return pl.DataFrame()

    values = np.asarray(f[name])
    if values.ndim == 1:
        values = values.reshape(-1, len(columns))
    return pl.DataFrame(values, schema=[str(c) for c in columns])


def _decode_legacy_scalar(node):
    """Decode a deepdish scalar — either a Dataset or a list-wrapped Group."""
    if isinstance(node, h5py.Group):
        items = _read_legacy_pytables_list(node)
        return items[0] if items else None
    val = node[()]
    return val.decode("utf-8") if isinstance(val, bytes) else val


def _load_legacy_brain_data_h5(f, mask=None):
    """Load a pre-0.6 (deepdish/PyTables) BrainData h5 file using only h5py."""
    result = {
        "data": np.asarray(f["data"]),
        "X": _read_legacy_frame(f, "X"),
        "Y": _read_legacy_frame(f, "Y"),
    }

    if mask is None and "mask_data" in f:
        affine = np.asarray(f["mask_affine"])
        data = np.asarray(f["mask_data"])
        if "mask_file_name" in f:
            file_name = _decode_legacy_scalar(f["mask_file_name"])
            if file_name:
                file_map = {"image": nib.FileHolder(filename=file_name)}
                result["mask"] = nib.Nifti1Image(data, affine=affine, file_map=file_map)
            else:
                result["mask"] = nib.Nifti1Image(data, affine=affine)
        else:
            result["mask"] = nib.Nifti1Image(data, affine=affine)
        result["load_mask"] = True
    else:
        result["load_mask"] = False

    return result


def load_legacy_adjacency_h5(file_path, mask=None, matrix_type=None):
    """Load a pre-0.6 (deepdish/PyTables) Adjacency h5 file using only h5py.

    Returns a dict with ``data``, ``Y``, ``matrix_type``, ``labels``. Optional
    structural fields (``is_single_matrix``, ``issymmetric``) are derived by
    the caller via ``import_single_data`` since older files predate them.

    Args:
        file_path: Path to HDF5 file.
        mask: Unused; accepted for API parity with load_brain_data_h5.
        matrix_type: Optional override when the legacy file lacks ``matrix_type``.
            If None and the file is missing the field, defaults to
            ``"distance_flat"`` and emits a UserWarning.
    """
    _require_h5()
    with h5File(file_path, "r") as f:
        result = {
            "data": np.asarray(f["data"]),
            "Y": _read_legacy_frame(f, "Y"),
        }

        if "matrix_type" in f:
            mt = _decode_legacy_scalar(f["matrix_type"])
        elif matrix_type is not None:
            mt = matrix_type
        else:
            warnings.warn(
                "Loading legacy h5 file: matrix_type field missing, assuming "
                "'distance_flat'. Pass matrix_type= to override, or re-save "
                "the file to update to the current format.",
                UserWarning,
                stacklevel=2,
            )
            mt = "distance_flat"

        # Legacy files always stored long-form vectors; normalize to *_flat
        # so import_single_data treats the data as already-flattened.
        if mt and not mt.endswith("_flat"):
            mt = f"{mt}_flat"
        result["matrix_type"] = mt

        if "labels" in f:
            labels_node = f["labels"]
            if isinstance(labels_node, h5py.Group):
                result["labels"] = _read_legacy_pytables_list(labels_node)
            elif len(labels_node) == 0:
                result["labels"] = []
            elif h5py.check_string_dtype(labels_node.dtype) is not None:
                result["labels"] = list(labels_node.asstr())
            else:
                result["labels"] = [
                    v.decode("utf-8") if isinstance(v, bytes) else v
                    for v in np.asarray(labels_node)
                ]
        else:
            result["labels"] = []

    return result


def is_legacy_adjacency_h5(file_path) -> bool:
    """Detect pre-0.6 deepdish/PyTables Adjacency layout.

    Modern files have Y as a Group containing ``columns``/``values`` children.
    Legacy files have Y as a flat Dataset with a sibling ``Y_columns`` node.
    """
    _require_h5()
    with h5File(file_path, "r") as f:
        return "Y_columns" in f
