"""Spatial-scale provenance for stacked Adjacency matrices.

When a stack of Adjacency matrices comes from a per-parcel or per-searchlight
operation on a BrainData, attaching a :class:`SpatialScale` records the atlas,
the parcel labels in stack order, and the source mask — enough to project
per-matrix reductions back to a voxel-space :class:`BrainData` via
``Adjacency.to_brain()``.

See :class:`Adjacency` for the optional ``spatial_scale`` attribute, and
:meth:`BrainData.distance` (with ``spatial_scale='roi'|'searchlight'``) for
the canonical producer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from nibabel import Nifti1Image

    from nltools.data import BrainData


_VALID_KINDS: tuple[str, ...] = ("roi", "searchlight")


@dataclass(frozen=True)
class SpatialScale:
    """Provenance for a stacked Adjacency that came from a per-parcel or
    per-searchlight operation on a :class:`BrainData`.

    Attributes:
        atlas: Labeled volume indicating parcel membership (or searchlight
            centers). One matrix in the stack per unique label.
        roi_labels: Integer atlas IDs in stack order. ``len(roi_labels)``
            must equal the number of matrices in the stack.
        source_mask: The brain mask the atlas/values live in. Used as the
            target space for back-projection in ``Adjacency.to_brain()``.
        kind: Which spatial scale produced this stack — ``'roi'`` or
            ``'searchlight'``.
    """

    atlas: BrainData
    roi_labels: np.ndarray
    source_mask: Nifti1Image
    kind: Literal["roi", "searchlight"] = field(default="roi")

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {self.kind!r}")
        # Coerce roi_labels to an integer ndarray; keep it immutable-friendly.
        labels = np.asarray(self.roi_labels)
        if labels.dtype.kind not in ("i", "u"):
            labels = labels.astype(np.int64)
        # Frozen dataclass — bypass setattr to write the coerced value.
        object.__setattr__(self, "roi_labels", labels)

    def __len__(self) -> int:
        return int(self.roi_labels.shape[0])
