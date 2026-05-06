"""IO and constructors for BrainCollection.

Constructors (``from_bids``, ``from_glob``, ``from_paths``, ``read``),
write, load/unload, cache plumbing, and ``memory_estimate``. Anything that
crosses the disk boundary lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import nibabel as nib
import polars as pl

if TYPE_CHECKING:
    import pandas as pd

    from . import BrainCollection


__all__ = [
    "discover_bids",
    "from_bids",
    "from_glob",
    "from_paths",
    "load",
    "memory_estimate",
    "read",
    "unload",
    "write",
]


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def from_bids(
    cls: type[BrainCollection],
    root: Path | str | Any,
    *,
    mask: nib.Nifti1Image | Path | str,
    task: str | None = None,
    space: str | None = None,
    sub_labels: list[str] | None = None,
    img_filters: list[tuple[str, str]] | None = None,
    derivatives_folder: str = "derivatives",
    pair_events: bool = True,
    confounds_strategy: str | tuple[str, ...] | None = None,
    confounds_kwargs: dict | None = None,
    TR: float | str = "infer",
    cache_dir: Path | str | None = "./.nltools_cache",
) -> BrainCollection:
    """Build a ``BrainCollection`` from a BIDS dataset.

    Delegates discovery to ``nilearn.glm.first_level.first_level_from_bids``
    (which wraps pybids), drops the returned ``models``, and keeps paths +
    events/confounds DataFrames. Per-item ``DesignMatrix`` is built from the
    events DataFrame; convolution / drift / confound merging is **not** done
    here — that's the user's ``transform_designs`` step.

    See SPEC §"``from_bids`` — concrete design" for edge cases.
    """
    raise NotImplementedError("scaffold")


def from_glob(
    cls: type[BrainCollection],
    pattern: str,
    *,
    mask: nib.Nifti1Image | Path | str,
    design_pattern: str | None = None,
    pattern_groups: dict[str, int] | str | None = None,
    sort: bool = True,
    cache_dir: Path | str | None = "./.nltools_cache",
) -> BrainCollection:
    """Build a collection by globbing for BOLD images (and optionally designs)."""
    raise NotImplementedError("scaffold")


def from_paths(
    cls: type[BrainCollection],
    brain_paths: list[Path | str],
    *,
    mask: nib.Nifti1Image | Path | str,
    design_paths: list[Path | str | None] | None = None,
    metadata: pl.DataFrame | pd.DataFrame | dict | None = None,
    cache_dir: Path | str | None = "./.nltools_cache",
) -> BrainCollection:
    """Build a collection from explicit lists of brain (and design) paths.

    Always lazy — items are stored as ``Path`` and loaded on demand.
    """
    return cls(
        brains=list(brain_paths),
        mask=mask,
        designs=design_paths,
        metadata=metadata,
        lazy=True,
        cache_dir=cache_dir,
    )


def read(
    cls: type[BrainCollection],
    directory: Path | str,
    *,
    mask: nib.Nifti1Image | Path | str,
    cache_dir: Path | str | None = "./.nltools_cache",
) -> BrainCollection:
    """Inverse of ``write()``: read images + ``metadata.csv`` from ``directory``.

    Discovers items by globbing ``image_*.nii*`` (matches the ``write()``
    default pattern) and pairs them with rows from ``metadata.csv`` if it
    exists. Does **not** recover from cache subdirs in v0.6.0.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"directory not found: {directory}")

    paths = sorted(directory.glob("image_*.nii*"))
    if not paths:
        raise ValueError(f"no image_*.nii* files found in {directory}")

    metadata_path = directory / "metadata.csv"
    metadata: pl.DataFrame | None = None
    if metadata_path.exists():
        metadata = pl.read_csv(metadata_path)

    return cls.from_paths(
        paths,
        mask=mask,
        metadata=metadata,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# BIDS discovery (split out so it's testable on its own)
# ---------------------------------------------------------------------------


def discover_bids(
    root: Path | str | Any,
    *,
    task: str | None,
    space: str | None,
    sub_labels: list[str] | None,
    img_filters: list[tuple[str, str]] | None,
    derivatives_folder: str,
    confounds_strategy: str | tuple[str, ...] | None,
    confounds_kwargs: dict | None,
    TR: float | str,
) -> dict[str, list]:
    """Walk the BIDS dataset and return aligned per-item lists.

    Returns a dict with keys: ``bold_paths``, ``events_dfs``, ``confounds_dfs``,
    ``sample_masks``, ``metadata_rows``, ``TRs``. Each list is the same length
    (one entry per BOLD file). Anything missing for an item is ``None``.

    Errors per SPEC §"Edge cases / errors":
      - Missing TR with ``TR='infer'``: raise.
      - ``task=None`` + ``pair_events=True``: caller silently downgrades.
      - fmriprep absent + ``confounds_strategy`` set: raise.
      - pybids not installed: raise ``ImportError``.
    """
    raise NotImplementedError("scaffold")


# ---------------------------------------------------------------------------
# Write / load / unload / memory
# ---------------------------------------------------------------------------


def write(
    bc: BrainCollection,
    directory: Path | str,
    *,
    pattern: str = "image_{i:04d}.nii.gz",
    metadata_file: str | None = "metadata.csv",
) -> list[Path]:
    """Write a clean, portable copy of ``bc`` outside the cache root.

    Inverse of ``BrainCollection.read()``. Writes one NIfTI per item under
    ``directory`` plus a metadata CSV. Skips the cache layout entirely so
    the result is shareable / archival.
    """
    from .execution import _atomic_write_nifti

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for i in range(len(bc._items)):
        bd = bc._load_item(i)
        out_path = directory / pattern.format(i=i)
        _atomic_write_nifti(out_path, bd)
        paths.append(out_path)

    if metadata_file is not None and bc._metadata is not None:
        bc._metadata.write_csv(directory / metadata_file)

    return paths


def load(
    bc: BrainCollection,
    indices: list[int] | None = None,
) -> BrainCollection:
    """Materialize path-backed items into ``BrainData``.

    Mutates ``bc`` in place. This is the only mutation method besides
    ``unload`` and does not allocate a step
    subdir, does not write to disk, does not produce a new identity.
    """
    from ..braindata import BrainData as _BrainData

    targets = range(len(bc._items)) if indices is None else indices
    for i in targets:
        item = bc._items[i]
        if not isinstance(item, _BrainData):
            bc._items[i] = _BrainData(item, mask=bc._mask)
    return bc


def unload(
    bc: BrainCollection,
    indices: list[int] | None = None,
) -> BrainCollection:
    """Drop in-memory data for items that have backing paths.

    Mutates in place. This is a no-op for items that don't have a backing path
    because dropping them would lose data.
    """
    from ..braindata import BrainData as _BrainData

    targets = range(len(bc._items)) if indices is None else indices
    for i in targets:
        if isinstance(bc._items[i], _BrainData) and bc._source_paths[i] is not None:
            bc._items[i] = bc._source_paths[i]
    return bc


def memory_estimate(bc: BrainCollection) -> str:
    """Human-readable RAM estimate if every item were loaded.

    Reports ``n_subjects``, the per-item shape (or "unknown" if path-backed
    and not yet loaded), and an estimated total in MB/GB based on float32.
    """
    from ..braindata import BrainData as _BrainData

    n = len(bc._items)
    if n == 0:
        return "BrainCollection(empty)"

    # Find a probe item to read shape from. Prefer in-memory; fall back to
    # loading the first path-backed item.
    probe = None
    for it in bc._items:
        if isinstance(it, _BrainData):
            probe = it
            break
    if probe is None:
        probe = bc._load_item(0)

    n_obs, n_vox = probe.shape
    # Assume float32 throughout — over-estimates float16 by 2x, fine.
    bytes_per_item = n_obs * n_vox * 4
    total = bytes_per_item * n

    def _human(b: float) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"

    return (
        f"BrainCollection(n_subjects={n}, per_item={n_obs}×{n_vox}, "
        f"estimated_total≈{_human(total)})"
    )
