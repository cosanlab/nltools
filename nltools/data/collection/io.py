"""I/O and memory-management functions for BrainCollection.

Save path resolution, on-disk write, and lazy load/unload helpers extracted
from BrainCollection. All BrainCollection I/O methods converted to functions
taking ``bc`` as first argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nltools.data.collection import BrainCollection


def _resolve_save_path(
    template: str,
    metadata_row: dict,
    idx: int,
) -> Path:
    """Resolve a template path using metadata values.

    Replaces {column_name} placeholders with values from metadata.
    Falls back to {idx} for the subject index.

    Args:
        template: Path template with {placeholders}, e.g., 'output/{subject}_betas.nii.gz'
        metadata_row: Dict of metadata for this subject (e.g. from ``pl.DataFrame.row(i, named=True)``)
        idx: Subject index (used for {idx} placeholder)

    Returns:
        Resolved Path object

    Raises:
        KeyError: If placeholder not found in metadata and not 'idx'

    Example:
        >>> row = {'subject': 'sub-01', 'session': 'ses-01'}
        >>> _resolve_save_path('out/{subject}_{session}.nii.gz', row, 0)
        PosixPath('out/sub-01_ses-01.nii.gz')
    """
    import re
    from pathlib import Path

    result = template

    placeholders = re.findall(r"\{(\w+)\}", template)

    for placeholder in placeholders:
        if placeholder == "idx":
            value = str(idx)
        elif placeholder in metadata_row:
            value = str(metadata_row[placeholder])
        else:
            available = list(metadata_row.keys()) + ["idx"]
            raise KeyError(
                f"Placeholder '{{{placeholder}}}' not found in metadata. "
                f"Available: {available}"
            )
        result = result.replace(f"{{{placeholder}}}", value)

    path = Path(result)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def write(
    bc: BrainCollection,
    directory: str | Path,
    pattern: str = "image_{i:04d}.nii.gz",
    metadata_file: str | None = "metadata.csv",
) -> list[Path]:
    """Write all images in collection to files.

    Args:
        bc: BrainCollection to write.
        directory: Output directory path. Will be created if it doesn't exist.
        pattern: Filename pattern with {i} placeholder for image index.
            Default: "image_{i:04d}.nii.gz" produces image_0000.nii.gz, etc.
        metadata_file: Optional filename for metadata CSV. Set to None to skip.
            Default: "metadata.csv"

    Returns:
        List of paths to written files.

    Examples:
        >>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
        >>> paths = write(bc, "output/")
        >>> # Creates: output/image_0000.nii.gz, image_0001.nii.gz, etc.

        >>> # Custom pattern
        >>> write(bc, "output/", pattern="sub-{i:02d}_bold.nii.gz")
        >>> # Creates: output/sub-00_bold.nii.gz, sub-01_bold.nii.gz, etc.

        >>> # With BIDS-style naming using metadata
        >>> bc.metadata["filename"] = [f"sub-{s}_bold.nii.gz" for s in subjects]
        >>> for i, bd in enumerate(bc):
        ...     bd.write(f"output/{bc.metadata.loc[i, 'filename']}")
    """
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for i in range(len(bc)):
        bd = bc._load_item(i)
        file_name = pattern.format(i=i)
        file_path = directory / file_name
        bd.write(str(file_path))
        written_paths.append(file_path)

    # Write metadata if requested
    if metadata_file is not None and not bc._metadata.is_empty():
        import polars as pl

        metadata_out = bc._metadata.with_columns(
            pl.Series("file_path", [str(p) for p in written_paths])
        )
        metadata_out.write_csv(directory / metadata_file)

    return written_paths


def memory_estimate(bc: BrainCollection) -> str:
    """Estimate memory usage for loading all images.

    Returns a human-readable string like
    ``"12.4 GB total (1.2 GB per image avg)"``.
    """
    # If we know sample counts, use them; otherwise estimate from loaded
    known_counts = [c for c in bc._sample_counts if c is not None]
    if known_counts:
        avg_obs = np.mean(known_counts)
    else:
        # Load first item to estimate
        bc._load_item(0)
        avg_obs = bc._sample_counts[0]

    bytes_per_element = 8  # float64
    bytes_per_image = avg_obs * bc.n_voxels * bytes_per_element
    total_bytes = bytes_per_image * bc.n_images

    def format_bytes(b: float) -> str:
        if b >= 1e9:
            return f"{b / 1e9:.1f} GB"
        if b >= 1e6:
            return f"{b / 1e6:.1f} MB"
        return f"{b / 1e3:.1f} KB"

    return (
        f"{format_bytes(total_bytes)} total "
        f"({format_bytes(bytes_per_image)} per image avg)"
    )


def load(bc: BrainCollection, indices: list[int] | None = None) -> BrainCollection:
    """Load specified images into memory. Returns the collection for chaining."""
    idx_iter = range(len(bc._items)) if indices is None else indices

    for idx in idx_iter:
        bc._load_item(idx)

    return bc


def unload(bc: BrainCollection, indices: list[int] | None = None) -> BrainCollection:
    """Free memory for items originally loaded from paths.

    Replaces in-memory BrainData with its source path so it can be lazily
    reloaded on next access. Items without a known source path are skipped.
    Returns the collection for chaining.
    """
    from nltools.data.braindata import BrainData

    idx_iter = range(len(bc._items)) if indices is None else indices

    for idx in idx_iter:
        item = bc._items[idx]
        if isinstance(item, BrainData) and hasattr(item, "_source_path"):
            # Can only unload if we know the original path
            bc._items[idx] = item._source_path
            bc._is_loaded[idx] = False

    return bc
