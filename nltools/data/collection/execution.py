"""Parallel execution machinery for BrainCollection.

Holds the worker-side dataclasses (``_ItemTask``, ``_DesignContext``), the
single parallel primitive (``_apply``), the worker-error type, and the
HDF5 fit-bundle IO. Every per-subject method on ``BrainCollection`` routes
through ``_apply`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
)
from collections.abc import Callable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from ..braindata import BrainData
    from ..designmatrix import DesignMatrix
    from . import BrainCollection


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BrainCollectionWorkerError",
    "_DesignContext",
    "_ItemTask",
    "_apply",
    "_materialize",
    "read_glm_bundle",
    "read_ridge_bundle",
    "tqdm_joblib",
    "write_glm_bundle",
    "write_ridge_bundle",
]


T = TypeVar("T")


# Bumped on any breaking change to the on-disk HDF5 fit-bundle layout.
BUNDLE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BrainCollectionWorkerError(RuntimeError):
    """Raised in the parent process when a worker fails inside ``_apply``.

    Wraps the original exception via ``raise ... from e`` so the full
    traceback is preserved. The message embeds subject/run context from
    ``_ItemTask.metadata_row`` so users can locate the offending item.
    """


# ---------------------------------------------------------------------------
# Worker-side dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ItemTask:
    """Pickle-friendly task spec sent to a worker.

    Workers receive paths or already-loaded objects — never the parent
    ``BrainCollection``. ``out_path`` is set when the op writes through
    cache; the worker writes via tmp+rename and returns the final path.
    """

    idx: int
    item: Any  # BrainData | Path
    design: Any  # DesignMatrix | Path | None
    confounds: Any  # pd.DataFrame | None
    sample_mask: np.ndarray | None
    metadata_row: dict
    mask_path: Path
    out_path: Path | None


@dataclass(frozen=True)
class _DesignContext:
    """Argument passed to user design-builder callables in ``fit(X=fn)``.

    Private — users never import this. They write ``def make_design(ctx): ...``
    and access fields by attribute. ``__getitem__`` falls back to the
    ``metadata`` row for user-added columns.
    """

    bd: BrainData
    dm: DesignMatrix | None
    confounds: pd.DataFrame | None
    sample_mask: np.ndarray | None

    metadata: dict

    subject: str | None
    session: str | None
    run: int | None
    task: str | None
    TR: float
    bold_path: Path
    events_path: Path | None
    confounds_path: Path | None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata[key]


# ---------------------------------------------------------------------------
# Worker materialization + parallel primitive
# ---------------------------------------------------------------------------


def _materialize(task: _ItemTask) -> tuple[BrainData, DesignMatrix | None]:
    """Load any path-backed fields on ``task`` into in-memory objects.

    Workers call this at the top of every job. The mask is cached at
    process scope (one load per worker) so repeated tasks don't re-parse
    the same NIfTI.
    """
    raise NotImplementedError("scaffold")


def _apply(
    bc: BrainCollection,
    fn: Callable[[_ItemTask], T],
    *,
    n_jobs: int = -1,
    progress_bar: bool = False,
    backend: str = "loky",
    cache: Literal["auto", True, False] = "auto",
    step_name: str,
    require_design: bool = False,
) -> list[T]:
    """Run ``fn`` over every item of ``bc`` in parallel.

    Resolves ``cache=`` (``'auto'`` → on if any source item is path-backed),
    allocates a step subdir under ``bc.cache_root`` when caching, builds
    ``_ItemTask``s, dispatches via joblib, fails fast on the first worker
    exception (wrapped in ``BrainCollectionWorkerError``), and returns the
    list of worker results in input order.

    Nested-parallelism guard: when ``n_jobs > 1`` we set
    ``inner_max_num_threads=1`` for the duration of the call.
    """
    raise NotImplementedError("scaffold")


# ---------------------------------------------------------------------------
# Progress-bar plumbing
# ---------------------------------------------------------------------------


class tqdm_joblib:
    """Context manager that updates a tqdm bar as joblib workers complete.

    Replaces today's submit-time wrapper, which advances on dispatch rather
    than completion. Single implementation; every collection method reuses it.
    """

    def __init__(self, total: int, desc: str = "", disable: bool = False) -> None:
        self.total = total
        self.desc = desc
        self.disable = disable

    def __enter__(self) -> tqdm_joblib:
        raise NotImplementedError("scaffold")

    def __exit__(self, exc_type, exc, tb) -> None:
        raise NotImplementedError("scaffold")


# ---------------------------------------------------------------------------
# HDF5 fit-bundle IO
# ---------------------------------------------------------------------------


def write_glm_bundle(
    out_path: Path,
    *,
    betas: np.ndarray,
    residuals: np.ndarray,
    sigma2: np.ndarray,
    r2: np.ndarray,
    X: np.ndarray,
    mask_bytes: bytes,
    affine: np.ndarray,
    regressor_names: list[str],
    scale: bool,
    scale_value: float,
    model_kwargs: dict,
    step_id: str,
    parent_step_id: str | None,
    op: str,
    op_kwargs: dict,
    nltools_version: str,
) -> Path:
    """Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).

    Layout (see SPEC §"HDF5 fit bundle"):
        /betas, /residuals, /sigma2, /r2, /X, /mask
        attrs: affine, regressor_names, scale, scale_value, model_kwargs,
               nltools_version, bundle_schema_version,
               step_id, parent_step_id, op, kwargs (JSON-encoded).

    Mask is embedded as a dataset (raw NIfTI bytes) so the bundle is
    portable across machines. Uses ``h5py.File(..., locking=False)``.
    """
    raise NotImplementedError("scaffold")


def read_glm_bundle(path: Path) -> dict[str, Any]:
    """Read and validate a GLM bundle.

    Validates ``bundle_schema_version``. A schema-version mismatch raises with
    a migration message; nltools-version
    mismatch logs a warning but does not refuse — bundles are usually
    forward-compatible within a minor version.
    """
    raise NotImplementedError("scaffold")


def write_ridge_bundle(
    out_path: Path,
    *,
    weights: np.ndarray,
    cv_scores: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    X: np.ndarray,
    mask_bytes: bytes,
    affine: np.ndarray,
    regressor_names: list[str],
    model_kwargs: dict,
    step_id: str,
    parent_step_id: str | None,
    op: str,
    op_kwargs: dict,
    nltools_version: str,
) -> Path:
    """Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

    Parallel layout to ``write_glm_bundle`` with ridge-specific datasets
    (``weights``, ``cv_scores``, ``predictions``, ``scores``).
    """
    raise NotImplementedError("scaffold")


def read_ridge_bundle(path: Path) -> dict[str, Any]:
    """Read a ridge bundle.

    Uses the same schema and version handling as ``read_glm_bundle``.
    """
    raise NotImplementedError("scaffold")


# ---------------------------------------------------------------------------
# Sidecar JSON for non-bundle outputs
# ---------------------------------------------------------------------------


def write_lineage_sidecar(
    nifti_path: Path,
    *,
    step_id: str,
    parent_step_id: str | None,
    op: str,
    op_kwargs: dict,
    nltools_version: str,
) -> Path:
    """Write ``{nifti_path.stem}.json`` with lineage attrs.

    Used for NIfTI outputs (``smooth``, ``standardize``, ``compute_contrasts``,
    ``predict(X_new)``, etc.) that don't embed lineage in their own format.
    """
    raise NotImplementedError("scaffold")
