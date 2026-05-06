"""Group-level reductions and cross-subject ops for BrainCollection.

Module-level functions that the ``BrainCollection`` facade delegates to.
Reductions stream from path-backed inputs (Welford-style) and produce
in-memory ``BrainData`` (or dicts of them); they never path-back their
own output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from ..braindata import BrainData
    from . import BrainCollection


__all__ = [
    "align",
    "anova",
    "concat",
    "isc",
    "isc_test",
    "max_",
    "mean",
    "median",
    "min_",
    "permutation_test",
    "permutation_test2",
    "std",
    "sum_",
    "ttest",
    "ttest2",
    "var",
]


# ---------------------------------------------------------------------------
# Stream-friendly reductions (Welford one-pass)
# ---------------------------------------------------------------------------


def concat(bc: BrainCollection) -> BrainData:
    """Stack along axis 0 → ``BrainData`` of shape ``(n_total_obs, n_voxels)``.

    Not streamable — the operation *is* materialization. Items must share
    a voxel dimension; mismatched shapes raise.
    """
    raise NotImplementedError("scaffold")


def mean(bc: BrainCollection) -> BrainData:
    """Mean across subjects (leading axis). Streams from path-backed input."""
    raise NotImplementedError("scaffold")


def std(bc: BrainCollection) -> BrainData:
    """Std across subjects. Streams via Welford; ddof=1 by default."""
    raise NotImplementedError("scaffold")


def var(bc: BrainCollection) -> BrainData:
    """Variance across subjects. Streams via Welford; ddof=1 by default."""
    raise NotImplementedError("scaffold")


def median(bc: BrainCollection) -> BrainData:
    """Median across subjects. Not streaming-friendly — materializes."""
    raise NotImplementedError("scaffold")


def sum_(bc: BrainCollection) -> BrainData:
    """Sum across subjects. Streams."""
    raise NotImplementedError("scaffold")


def min_(bc: BrainCollection) -> BrainData:
    """Per-voxel min across subjects. Streams."""
    raise NotImplementedError("scaffold")


def max_(bc: BrainCollection) -> BrainData:
    """Per-voxel max across subjects. Streams."""
    raise NotImplementedError("scaffold")


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------


def ttest(
    bc: BrainCollection,
    *,
    popmean: float = 0.0,
) -> dict[str, BrainData]:
    """One-sample t-test across subjects.

    Returns ``{'mean', 't', 'z', 'p'}`` — same shape contract as
    ``BrainData.ttest``. Streams from path-backed input via Welford.
    """
    raise NotImplementedError("scaffold")


def ttest2(
    bc: BrainCollection,
    other: BrainCollection,
    *,
    equal_var: bool = True,
) -> dict[str, BrainData]:
    """Two-sample t-test between two collections."""
    raise NotImplementedError("scaffold")


def anova(
    bc: BrainCollection,
    groups: str | list | np.ndarray,
) -> dict[str, BrainData]:
    """One-way ANOVA across subjects.

    ``groups`` is a metadata column name, a list, or an ndarray of length
    ``n_subjects``.
    """
    raise NotImplementedError("scaffold")


def permutation_test(
    bc: BrainCollection,
    *,
    n_permute: int = 5000,
    tail: int = 2,
    device: str = "cpu",
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> dict:
    """Sign-flipping permutation test across subjects.

    Materializes all subjects (or memmaps); see SPEC streaming-algorithms
    table — sign-flipping needs the full set in memory by design.
    """
    raise NotImplementedError("scaffold")


def permutation_test2(
    bc: BrainCollection,
    other: BrainCollection,
    *,
    n_permute: int = 5000,
    tail: int = 2,
    device: str = "cpu",
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> dict:
    """Two-sample permutation test between two collections."""
    raise NotImplementedError("scaffold")


# ---------------------------------------------------------------------------
# Cross-subject ops (inherently multi-subject)
# ---------------------------------------------------------------------------


def isc(
    bc: BrainCollection,
    *,
    method: str = "loo",
    roi_mask: nib.Nifti1Image | Path | str | None = None,
    radius_mm: float | None = 6.0,
    metric: str = "median",
    device: str = "cpu",
    n_jobs: int = -1,
    progress_bar: bool = False,
) -> dict:
    """Inter-subject correlation.

    LOO method streams (two passes, sum-trick); pairwise streams two
    subjects at a time. Voxelwise/searchlight path goes through
    ``nltools.algorithms.inference.isc`` after the v0.6 streaming rewrite.
    """
    raise NotImplementedError("scaffold")


def isc_test(
    bc: BrainCollection,
    *,
    method: str = "loo",
    roi_mask: nib.Nifti1Image | Path | str | None = None,
    radius_mm: float | None = 6.0,
    n_permute: int = 5000,
    permutation_method: str = "bootstrap",
    metric: str = "median",
    device: str = "cpu",
    n_jobs: int = -1,
    progress_bar: bool = False,
    random_state: int | None = None,
) -> dict:
    """Permutation/bootstrap inference on ISC."""
    raise NotImplementedError("scaffold")


def align(
    bc: BrainCollection,
    *,
    method: str = "procrustes",
    scheme: str = "searchlight",
    radius_mm: float = 10.0,
    parcellation: nib.Nifti1Image | None = None,
    n_features: int | None = None,
    n_iter: int = 3,
    device: str = "cpu",
    return_model: bool = False,
    n_jobs: int = -1,
    progress_bar: bool = False,
    cache: Literal["auto", True, False] = "auto",
):
    """Functional alignment (Procrustes / SRM / hyperalignment).

    Returns ``BrainCollection`` (aligned data) or
    ``(BrainCollection, LocalAlignment)`` when ``return_model=True``.
    Materializes all subjects in v0.6 (algorithm constraint, see SPEC
    streaming-algorithms table).
    """
    raise NotImplementedError("scaffold")
