"""Lazy loading of atlas NIfTI + label CSV files from the HF dataset."""

import functools
from dataclasses import dataclass

import nibabel as nb
import polars as pl

from nltools.templates.fetch import fetch_resource

from .registry import ATLASES, AtlasKind


@dataclass(frozen=True)
class Atlas:
    """A loaded atlas — image, labels, and metadata.

    Constructed by `load_atlas`; users normally don't instantiate
    directly.

    Attributes:
        name: Registry key (e.g. ``"harvard_oxford"``).
        image: NIfTI volume. 3D for deterministic atlases, 4D for
            probabilistic ones (last axis indexes regions).
        labels: Two-column ``index, name`` table. For deterministic
            atlases ``index`` is the integer voxel value; for
            probabilistic atlases ``index`` is the region index along
            the 4th dim of ``image``.
        kind: ``"deterministic"`` or ``"probabilistic"``.
        citation: Short citation for the original atlas.
    """

    name: str
    image: nb.Nifti1Image
    labels: pl.DataFrame
    kind: AtlasKind
    citation: str


@functools.cache
def load_atlas(name: str) -> Atlas:
    """Lazy-load an atlas by registry name.

    First call fetches the NIfTI + label CSV from
    ``huggingface.co/datasets/nltools/niftis`` (cached locally
    afterwards). Subsequent calls in the same process are memoized.

    Args:
        name: Atlas key from `list_atlases`.

    Returns:
        An `Atlas` with image, labels, and metadata loaded.

    Raises:
        ValueError: If ``name`` isn't a registered atlas.
    """
    if name not in ATLASES:
        known = ", ".join(sorted(ATLASES))
        raise ValueError(f"unknown atlas {name!r}; choose from: {known}")

    meta = ATLASES[name]
    img_path = fetch_resource(f"atlases/atlas_{name}.nii.gz")
    csv_path = fetch_resource(f"atlases/labels_{name}.csv")

    return Atlas(
        name=name,
        image=nb.load(img_path),
        labels=pl.read_csv(csv_path),
        kind=meta.kind,
        citation=meta.citation,
    )
