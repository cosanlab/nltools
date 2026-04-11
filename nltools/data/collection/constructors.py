"""Constructor functions for BrainCollection.

Standalone functions that create BrainCollection instances from various sources
(BIDS datasets, glob patterns, stacked BrainData).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib

    from nltools.data.braindata import BrainData
    from nltools.data.collection import BrainCollection


def from_bids(
    layout: Any,  # BIDSLayout or path
    mask: "nib.Nifti1Image | Path | str",
    *,
    task: str | None = None,
    subject: str | list[str] | None = None,
    session: str | list[str] | None = None,
    run: int | list[int] | None = None,
    space: str | None = None,
    suffix: str = "bold",
    extension: str = "nii.gz",
    **bids_filters,
) -> "BrainCollection":
    """
    Create BrainCollection from a BIDS dataset.

    Requires pybids to be installed: `pip install pybids`

    Args:
        layout: pybids BIDSLayout object or path to BIDS dataset.
        mask: Shared mask (required).
        task: BIDS task filter.
        subject: Subject ID(s) to include.
        session: Session ID(s) to include.
        run: Run number(s) to include.
        space: BIDS space filter (e.g., 'MNI152NLin2009cAsym').
        suffix: BIDS suffix (default 'bold').
        extension: File extension (default 'nii.gz').
        **bids_filters: Additional BIDS entity filters.

    Returns:
        BrainCollection with metadata extracted from BIDS entities.

    Examples:
        >>> bc = from_bids(
        ...     '/data/bids_dataset',
        ...     mask='2mm-MNI152-2009c',
        ...     task='rest',
        ...     space='MNI152NLin2009cAsym'
        ... )
    """
    import polars as pl
    from pathlib import Path

    from nltools.data.collection import BrainCollection
    from nltools.utils import attempt_to_import

    # Import pybids
    bids_module = attempt_to_import("bids", "bids")
    if bids_module is None:
        raise ImportError(
            "pybids required for BIDS loading. Install with: pip install pybids"
        )
    BIDSLayout = bids_module.BIDSLayout

    # Create layout if path provided
    if isinstance(layout, (str, Path)):
        layout = BIDSLayout(layout, validate=False)

    # Build filter dict
    filters = {"extension": extension, "suffix": suffix}
    if task is not None:
        filters["task"] = task
    if subject is not None:
        filters["subject"] = subject
    if session is not None:
        filters["session"] = session
    if run is not None:
        filters["run"] = run
    if space is not None:
        filters["space"] = space
    filters.update(bids_filters)

    # Get files
    files = layout.get(return_type="file", **filters)
    if not files:
        raise ValueError(f"No files found matching filters: {filters}")

    # Extract metadata
    metadata_rows = []
    for f in files:
        bf = layout.get_file(f)
        entities = bf.get_entities() if bf else {}
        metadata_rows.append(
            {
                "subject": entities.get("subject"),
                "session": entities.get("session"),
                "run": entities.get("run"),
                "task": entities.get("task"),
                "space": entities.get("space"),
            }
        )

    # Build per-column dict so polars handles None uniformly
    cols = ["subject", "session", "run", "task", "space"]
    metadata = pl.DataFrame({c: [row[c] for row in metadata_rows] for c in cols})
    return BrainCollection(files, mask=mask, metadata=metadata)


def from_glob(
    pattern: str,
    mask: "nib.Nifti1Image | Path | str",
    *,
    pattern_groups: "dict[str, int] | str | None" = None,
    sort: bool = True,
) -> "BrainCollection":
    """
    Create BrainCollection from glob pattern.

    Args:
        pattern: Glob pattern (e.g., ``'/data/*/func/*_bold.nii.gz'``).
        mask: Shared mask (required).
        pattern_groups: Regex pattern with named groups for metadata extraction.
            Example: ``r'sub-(?P<subject>\\w+)/.*run-(?P<run>\\d+)'``
        sort: Sort files alphabetically (default True).

    Returns:
        BrainCollection with optional metadata from pattern groups.

    Examples:
        >>> bc = from_glob(
        ...     '/data/sub-*/func/*_bold.nii.gz',
        ...     mask=mask,
        ...     pattern_groups=r'sub-(?P<subject>\\w+)'
        ... )
    """
    import glob
    import re

    import polars as pl

    from nltools.data.collection import BrainCollection

    files = glob.glob(pattern, recursive=True)
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    if sort:
        files = sorted(files)

    # Extract metadata from paths
    metadata = None
    if pattern_groups is not None:
        if isinstance(pattern_groups, str):
            regex = re.compile(pattern_groups)
            metadata_rows = []
            for f in files:
                match = regex.search(f)
                metadata_rows.append(match.groupdict() if match else {})
            # Union of keys across rows; missing values become None
            keys: list[str] = []
            for row in metadata_rows:
                for k in row:
                    if k not in keys:
                        keys.append(k)
            if keys:
                metadata = pl.DataFrame(
                    {k: [row.get(k) for row in metadata_rows] for k in keys}
                )

    return BrainCollection(files, mask=mask, metadata=metadata)


def from_stacked(
    brain_data: "BrainData",
    splits: list[int] | None = None,
    n_images: int | None = None,
) -> "BrainCollection":
    """
    Create BrainCollection by splitting a stacked BrainData.

    Args:
        brain_data: BrainData with shape (n_total_obs, n_voxels).
        splits: List of observation counts per image. Must sum to n_total_obs.
        n_images: Number of images (splits evenly). Mutually exclusive with splits.

    Returns:
        BrainCollection with data split according to specification.

    Examples:
        >>> # Split evenly into 3 images
        >>> bc = from_stacked(bd, n_images=3)

        >>> # Split with explicit counts
        >>> bc = from_stacked(bd, splits=[100, 100, 150])
    """
    from nltools.data.braindata import BrainData
    from nltools.data.collection import BrainCollection

    if splits is None and n_images is None:
        raise ValueError("Must provide either splits or n_images")
    if splits is not None and n_images is not None:
        raise ValueError("Cannot provide both splits and n_images")

    data = brain_data.data
    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_total = data.shape[0]

    if n_images is not None:
        if n_total % n_images != 0:
            raise ValueError(
                f"Cannot evenly split {n_total} observations into {n_images} images"
            )
        splits = [n_total // n_images] * n_images

    assert splits is not None  # guaranteed by the n_images branch above
    if sum(splits) != n_total:
        raise ValueError(
            f"splits sum ({sum(splits)}) must equal total observations ({n_total})"
        )

    # Split data
    items = []
    idx = 0
    for count in splits:
        bd = BrainData(mask=brain_data.mask)
        bd.data = data[idx : idx + count]
        items.append(bd)
        idx += count

    return BrainCollection(items, mask=brain_data.mask)
