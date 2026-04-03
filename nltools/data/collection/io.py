"""I/O functions for BrainCollection.

Provides save path resolution and write functionality extracted from BrainCollection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from nltools.data.collection import BrainCollection


def _resolve_save_path(
    template: str,
    metadata_row: "pd.Series",
    idx: int,
) -> "Path":
    """Resolve a template path using metadata values.

    Replaces {column_name} placeholders with values from metadata.
    Falls back to {idx} for the subject index.

    Args:
        template: Path template with {placeholders}, e.g., 'output/{subject}_betas.nii.gz'
        metadata_row: Series with metadata for this subject
        idx: Subject index (used for {idx} placeholder)

    Returns:
        Resolved Path object

    Raises:
        KeyError: If placeholder not found in metadata and not 'idx'

    Example:
        >>> row = pd.Series({'subject': 'sub-01', 'session': 'ses-01'})
        >>> _resolve_save_path('out/{subject}_{session}.nii.gz', row, 0)
        PosixPath('out/sub-01_ses-01.nii.gz')
    """
    import re
    from pathlib import Path

    result = template

    # Find all {placeholder} patterns
    placeholders = re.findall(r"\{(\w+)\}", template)

    for placeholder in placeholders:
        if placeholder == "idx":
            value = str(idx)
        elif placeholder in metadata_row.index:
            value = str(metadata_row[placeholder])
        else:
            available = list(metadata_row.index) + ["idx"]
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
    bc: "BrainCollection",
    directory: "str | Path",
    pattern: str = "image_{i:04d}.nii.gz",
    metadata_file: str | None = "metadata.csv",
) -> list["Path"]:
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
    if metadata_file is not None and not bc._metadata.empty:
        # Add file paths to metadata
        metadata_out = bc._metadata.copy()
        metadata_out["file_path"] = [str(p) for p in written_paths]
        metadata_out.to_csv(directory / metadata_file, index=False)

    return written_paths
