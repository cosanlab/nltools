"""
NeuroLearn Utilities
====================

Cross-cutting utilities used across the nltools package.

"""

__all__ = [
    "get_resource_path",
    "attempt_to_import",
    "all_same",
    "concatenate",
    "get_mni_from_img_resolution",
    "detect_best_matching_template",
    "is_h5_path",
    "to_h5",
    "load_brain_data_h5",
]

import collections
import os
from os.path import dirname, join, sep as pathsep

import nibabel as nib
import numpy as np
import pandas as pd
from h5py import File as h5File
from nltools.prefs import MNI_Template


# ---------------------------------------------------------------------------
# Cross-cutting helpers (used by multiple subsystems)
# ---------------------------------------------------------------------------


def get_resource_path():
    """Get path to nltools resource directory."""
    return join(dirname(__file__), "resources") + pathsep


module_names = {}
Dependency = collections.namedtuple("Dependency", "package value")


def attempt_to_import(dependency, name=None, fromlist=None):
    """Attempt to import an optional dependency, returning None if unavailable.

    This function is used to handle optional dependencies gracefully. If the
    import fails, the function returns None rather than raising an error,
    allowing the calling code to check and handle missing dependencies.

    Args:
        dependency: The module name to import (e.g., 'torch', 'cupy').
        name: Optional name to store the dependency under in module_names.
            Defaults to the dependency name.
        fromlist: Optional list of names to import from the module.

    Returns:
        The imported module, or None if the import failed.

    Examples:
        >>> torch = attempt_to_import('torch')
        >>> if torch is not None:
        ...     # Use torch
        ...     pass
    """
    if name is None:
        name = dependency
    try:
        mod = __import__(dependency, fromlist=fromlist)
    except ImportError:
        mod = None
    module_names[name] = Dependency(dependency, mod)
    return mod


def all_same(items):
    """Check if all items in a sequence are equal to the first item.

    Args:
        items: A sequence of items to compare.

    Returns:
        bool: True if all items equal the first item, False otherwise.

    Examples:
        >>> all_same([1, 1, 1])
        True
        >>> all_same([1, 2, 1])
        False
    """
    return np.all(x == items[0] for x in items)


def concatenate(data):
    """Concatenate a list of BrainData() or Adjacency() objects"""

    if not isinstance(data, list):
        raise ValueError("Make sure you are passing a list of objects.")

    if all([isinstance(x, data[0].__class__) for x in data]):
        out = data[0].__class__()
        for i in data:
            out = out.append(i)
    else:
        raise ValueError("Make sure all objects in the list are the same type.")
    return out


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------


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


def to_h5(obj, file_name, obj_type="brain_data", h5_compression="gzip"):
    """Save BrainData or Adjacency objects to HDF5 files.

    Uses a combination of pandas and h5py to save objects to h5 files.

    Args:
        obj: Object to save (BrainData or Adjacency).
        file_name: Path to save file to.
        obj_type: Type of object ('brain_data' or 'adjacency').
        h5_compression: Compression type for h5py datasets.
    """
    if obj_type not in ["brain_data", "adjacency"]:
        raise TypeError("obj_type must be one of 'brain_data' or 'adjacency'")

    if obj_type == "brain_data":
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tables")
            warnings.filterwarnings(
                "ignore", message=".*performance.*", module="tables"
            )
            with pd.HDFStore(file_name, "w") as f:
                if hasattr(obj, "X"):
                    f["X"] = obj.X
                else:
                    f["X"] = pd.DataFrame()
                if hasattr(obj, "Y"):
                    f["Y"] = obj.Y
                else:
                    f["Y"] = pd.DataFrame()

        with h5File(file_name, "a") as f:
            f.create_dataset("data", data=obj.data, compression=h5_compression)
            f.create_dataset(
                "mask_affine", data=obj.mask.affine, compression=h5_compression
            )
            f.create_dataset(
                "mask_data", data=obj.mask.get_fdata(), compression=h5_compression
            )
            f.create_dataset("mask_file_name", data=obj.mask.get_filename())
    else:
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

    Handles both modern and legacy (pre-0.4.8) HDF5 formats.

    Args:
        file_path: Path to HDF5 file.
        mask: Optional mask to use. If None, loads mask from file if available.

    Returns:
        dict: Dictionary containing loaded data, X, Y, and optionally mask info.
    """

    result = {}

    try:
        # Try modern format first
        with pd.HDFStore(file_path, "r") as f:
            result["X"] = f["X"]
            result["Y"] = f["Y"]

        with h5File(file_path, "r") as f:
            result["data"] = np.array(f["data"])

            # Handle mask loading
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

    except Exception:
        # Fall back to legacy format
        result = _load_legacy_brain_data_h5(file_path, mask)
        result["legacy_format"] = True

    return result


def _load_legacy_brain_data_h5(file_path, mask=None):
    """Load BrainData from legacy HDF5 format (pre-0.4.8).

    Args:
        file_path: Path to HDF5 file.
        mask: Optional mask to use.

    Returns:
        dict: Dictionary containing loaded data, X, Y, and optionally mask info.
    """
    tables_mod = attempt_to_import("tables")
    if tables_mod is None:
        raise ImportError("tables package required for legacy h5 format")

    result = {}

    with tables_mod.open_file(file_path, mode="r") as f:
        result["data"] = np.array(f.root["data"])

        if len(list(f.root["X_columns"])):
            result["X"] = pd.DataFrame(
                np.array(f.root["X"]).squeeze(),
                columns=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["X_columns"])
                ],
                index=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["X_index"])
                ],
            )
        else:
            result["X"] = pd.DataFrame()

        if len(list(f.root["Y_columns"])):
            result["Y"] = pd.DataFrame(
                np.array(f.root["Y"]).squeeze(),
                columns=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["Y_columns"])
                ],
                index=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["Y_index"])
                ],
            )
        else:
            result["Y"] = pd.DataFrame()

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


# ---------------------------------------------------------------------------
# MNI template resolution helpers
# ---------------------------------------------------------------------------


def get_mni_from_img_resolution(brain, img_type="plot"):
    """
    Get the path to the MNI anatomical image that matches the resolution of a BrainData instance.

    This function determines the resolution of the input BrainData and returns the appropriate
    MNI template image path from the current MNI_Template settings, adjusting only the resolution
    while keeping the same template variant.

    Args:
        brain: BrainData instance
        img_type: 'plot' for T1 image or 'brain' for brain-extracted image

    Returns:
        file_path: path to MNI image with matching resolution
    """

    if img_type not in ["plot", "brain"]:
        raise ValueError("img_type must be 'plot' or 'brain' ")

    # Get resolution from the brain data
    res_array = np.abs(np.diag(brain.nifti_masker.affine_)[:3])
    voxel_dims = np.unique(abs(res_array))
    if len(voxel_dims) != 1:
        raise ValueError(
            "Voxels are not isometric and cannot be visualized in standard space"
        )

    # Determine resolution in mm
    resolution = int(voxel_dims[0])

    # Check if this resolution is supported for the current template
    if resolution not in MNI_Template._supported_combinations.get(
        MNI_Template.template, []
    ):
        if img_type == "brain":
            return MNI_Template.brain
        else:
            return MNI_Template.plot

    # Map template names to version codes
    version_map = {
        "default": "fsl",
        "nilearn": "a",
        "fmriprep": "c",
    }

    version = version_map.get(MNI_Template.template)
    if version is None:
        raise ValueError(f"Unknown template: {MNI_Template.template}")

    # Build path with matching resolution using new naming convention
    from os.path import join, dirname

    base_path = join(
        dirname(MNI_Template.mask).rsplit("/niftis/", 1)[0],
        "niftis",
        MNI_Template.template,
    )
    res_str = f"{resolution}mm"

    if img_type == "brain":
        new_path = join(base_path, f"{res_str}-MNI152-2009{version}-brain.nii.gz")
        old_path = join(base_path, f"MNI152_{res_str}_brain.nii.gz")
    else:
        new_path = join(base_path, f"{res_str}-MNI152-2009{version}-T1.nii.gz")
        old_path = join(base_path, f"MNI152_{res_str}_T1.nii.gz")

    # Use new naming if available, otherwise fall back to old naming
    if not os.path.exists(new_path) and os.path.exists(old_path):
        return old_path
    return new_path


def detect_best_matching_template(
    data_img, prefer_exact_match=True, resample_enabled=True
):
    """
    Detect the best matching MNI template from available resources based on data resolution.

    This function analyzes the affine matrix of a nibabel image to determine its resolution
    and finds the best matching template from available resources (default, nilearn, fmriprep).
    For non-isotropic voxels, it uses the average voxel size across all dimensions.

    Args:
        data_img: nibabel Nifti1Image object
        prefer_exact_match: If True, prefer exact resolution match. If False, use closest match.
        resample_enabled: If True, indicates that resampling is enabled in the calling context.
                         Used to conditionally suggest resampling in warnings. Default: True.

    Returns:
        dict: Dictionary with keys:
            - 'template': Template variant ('default', 'nilearn', or 'fmriprep')
            - 'resolution': Resolution in mm (1, 2, or 3)
            - 'mask_path': Path to mask file
            - 'brain_path': Path to brain-extracted image
            - 'plot_path': Path to T1 image
            - 'match_distance': Distance from detected resolution to matched resolution

    Raises:
        ValueError: If no matching template is found or voxel sizes are invalid

    Examples:
        >>> import nibabel as nib
        >>> data = nib.Nifti1Image(
        ...     np.random.randn(91, 109, 91, 10),
        ...     affine=np.eye(4) * 2  # 2mm resolution
        ... )
        >>> template_info = detect_best_matching_template(data)
        >>> print(template_info['template'])  # 'default'
        >>> print(template_info['resolution'])  # 2
    """
    # Extract resolution from affine matrix
    affine = data_img.affine
    res_array = np.abs(np.diag(affine[:3, :3]))

    # Check if isotropic
    voxel_dims = np.unique(res_array)
    is_isotropic = len(voxel_dims) == 1

    if is_isotropic:
        resolution_float = float(voxel_dims[0])
    else:
        resolution_float = float(np.mean(res_array))

    # Determine resolution in mm (round to nearest integer)
    resolution = int(np.round(resolution_float))

    if resolution < 1 or resolution > 10:
        raise ValueError(
            f"Detected resolution ({resolution_float}mm) is outside reasonable range (1-10mm). "
            f"Data may not be in standard MNI space."
        )

    # Define template priority order
    template_priority = ["default", "nilearn", "fmriprep"]

    supported_combinations = {
        "default": [2, 3],
        "nilearn": [1, 2, 3],
        "fmriprep": [1, 2],
    }

    # Find best matching template
    best_template = None
    best_resolution = None
    best_match_distance = float("inf")

    if prefer_exact_match:
        for template in template_priority:
            if resolution in supported_combinations[template]:
                best_template = template
                best_resolution = resolution
                best_match_distance = 0
                break

    if best_template is None:
        available_resolutions = set()
        for resolutions in supported_combinations.values():
            available_resolutions.update(resolutions)

        if not available_resolutions:
            raise ValueError(
                "No templates available. This suggests an incomplete installation."
            )

        closest_res = min(available_resolutions, key=lambda x: abs(x - resolution))
        best_match_distance = abs(closest_res - resolution)

        for template in template_priority:
            if closest_res in supported_combinations[template]:
                best_template = template
                best_resolution = closest_res
                break

        if best_template is None:
            raise ValueError(
                f"Could not find template for resolution {closest_res}mm. "
                f"This should not happen - please report as a bug."
            )

        if best_match_distance > 0:
            import warnings

            if resample_enabled:
                warnings.warn(
                    f"\nResampling data resolution ({resolution_float:.3f}mm) to closest matching template resolution: {best_resolution}mm.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"\nData resolution ({resolution_float:.3f}mm) doesn't match template for masking: {best_template} {best_resolution}mm.\nConsider resampling your data to {best_resolution}mm for best results.",
                    UserWarning,
                    stacklevel=2,
                )

    version_map = {
        "default": "fsl",
        "nilearn": "a",
        "fmriprep": "c",
    }

    version = version_map.get(best_template)
    if version is None:
        raise ValueError(f"Unknown template: {best_template}")

    base_path = join(dirname(__file__), "resources", "niftis", best_template)
    res_str = f"{best_resolution}mm"

    mask_path = join(base_path, f"{res_str}-MNI152-2009{version}-mask.nii.gz")
    brain_path = join(base_path, f"{res_str}-MNI152-2009{version}-brain.nii.gz")
    plot_path = join(base_path, f"{res_str}-MNI152-2009{version}-T1.nii.gz")

    old_mask_path = join(base_path, f"MNI152_{res_str}_mask.nii.gz")
    old_brain_path = join(base_path, f"MNI152_{res_str}_brain.nii.gz")
    old_plot_path = join(base_path, f"MNI152_{res_str}_T1.nii.gz")

    if not os.path.exists(mask_path) and os.path.exists(old_mask_path):
        mask_path = old_mask_path
    if not os.path.exists(brain_path) and os.path.exists(old_brain_path):
        brain_path = old_brain_path
    if not os.path.exists(plot_path) and os.path.exists(old_plot_path):
        plot_path = old_plot_path

    for path_name, path in [
        ("mask", mask_path),
        ("brain", brain_path),
        ("plot", plot_path),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Template file not found: {path}\n"
                f"This suggests an incomplete installation or missing template files."
            )

    return {
        "template": best_template,
        "resolution": best_resolution,
        "mask_path": mask_path,
        "brain_path": brain_path,
        "plot_path": plot_path,
        "match_distance": best_match_distance,
    }
