"""Parallel execution machinery for BrainCollection.

Holds the worker-side dataclasses (``_ItemTask``, ``_DesignContext``), the
single parallel primitive (``_apply``), the worker-error type, and the
HDF5 fit-bundle IO. Every per-subject method on ``BrainCollection`` routes
through ``_apply`` here.
"""

from __future__ import annotations

import os
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

from . import core

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


# Process-scoped mask cache so repeated tasks in the same worker don't
# re-parse the same NIfTI. Keyed by absolute path string.
_MASK_CACHE: dict[str, Any] = {}


def _get_cached_mask(path: Path | str):
    import nibabel as nib

    key = str(path)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = nib.load(key)
    return _MASK_CACHE[key]


def _materialize(task: _ItemTask) -> tuple[BrainData, DesignMatrix | None]:
    """Load any path-backed fields on ``task`` into in-memory objects.

    Workers call this at the top of every job. The mask is cached at
    process scope (one load per worker) so repeated tasks don't re-parse
    the same NIfTI.
    """
    from ..braindata import BrainData as _BrainData
    from ..designmatrix import DesignMatrix as _DesignMatrix

    mask = _get_cached_mask(task.mask_path)

    item = task.item
    if isinstance(item, _BrainData):
        bd = item
    else:
        bd = _BrainData(item, mask=mask)

    design = task.design
    if design is None or isinstance(design, _DesignMatrix):
        dm = design
    else:
        # design is a path — DesignMatrix has no read() classmethod yet,
        # so callers that pass paths today are responsible for loading.
        dm = design
    return bd, dm


def _atomic_write_nifti(out_path: Path, bd: BrainData) -> Path:
    """Write ``bd`` to ``out_path`` via tmp+rename (crash-safe).

    Tmp file lives in the same directory with a ``.tmp_`` prefix so the
    original suffix (e.g. ``.nii.gz``) is preserved — nibabel uses the
    extension to pick a serializer.
    """
    tmp = out_path.parent / f".tmp_{out_path.name}"
    bd.write(tmp)
    os.rename(tmp, out_path)
    return out_path


def _wrap_worker(fn: Callable[[_ItemTask], T], task: _ItemTask) -> T:
    """Run ``fn(task)``, wrapping exceptions in ``BrainCollectionWorkerError``."""
    try:
        return fn(task)
    except BrainCollectionWorkerError:
        raise
    except Exception as e:
        ctx_parts = [f"idx={task.idx}"]
        for key in ("subject", "run"):
            v = task.metadata_row.get(key)
            if v is not None:
                ctx_parts.append(f"{key}={v}")
        raise BrainCollectionWorkerError(
            f"[{', '.join(ctx_parts)}] {type(e).__name__}: {e}"
        ) from e


def _resolve_cache_mode(
    items: list,
    cache: Literal["auto", True, False],
) -> bool:
    """Map ``cache=`` to a concrete True/False given the source state."""
    from ..braindata import BrainData as _BrainData

    if cache == "auto":
        return any(not isinstance(it, _BrainData) for it in items)
    return bool(cache)


def _stash_mask(bc: BrainCollection) -> Path:
    """Persist ``bc._mask`` to ``cache_root/_mask.nii.gz`` once.

    Workers load the mask via path so we don't pickle the Nifti1Image with
    every task.
    """
    import nibabel as nib

    if bc._cache_root is None:
        raise RuntimeError("cannot stash mask: cache_root is None")
    mask_path = bc._cache_root / "_mask.nii.gz"
    if not mask_path.exists():
        nib.save(bc._mask, mask_path)
    return mask_path


def _apply(
    bc: BrainCollection,
    fn: Callable[[_ItemTask], T],
    *,
    op: str,
    op_kwargs: dict | None = None,
    n_jobs: int = -1,
    progress_bar: bool = False,
    backend: str = "loky",
    cache: Literal["auto", True, False] = "auto",
    out_ext: str = ".nii.gz",
    require_design: bool = False,
) -> tuple[list, Path | None, str]:
    """Run ``fn`` over every item of ``bc`` in parallel.

    Returns ``(results, step_dir_or_None, step_id)``.

    Behavior:
      - Resolves ``cache=`` (``'auto'`` → on if any item is path-backed).
      - Allocates a step subdir under ``bc.cache_root`` when caching.
      - Persists the mask once into ``cache_root/_mask.nii.gz``.
      - Builds ``_ItemTask``s, dispatches via joblib with
        ``inner_max_num_threads=1`` to guard against nested parallelism.
      - Wraps worker exceptions in ``BrainCollectionWorkerError`` carrying
        subject/run context; the original is chained via ``from e``.

    The per-item ``out_path`` is set to ``step_dir / sub-{i+1:04d}{out_ext}``
    when caching, ``None`` otherwise. Workers decide whether to write to
    ``out_path`` and what to return (path vs in-memory).
    """
    from joblib import Parallel, delayed, parallel_backend

    op_kwargs = op_kwargs or {}
    if require_design and any(d is None for d in bc._designs):
        missing = [i for i, d in enumerate(bc._designs) if d is None]
        raise ValueError(f"items {missing} have no design but op {op!r} requires one")

    do_cache = _resolve_cache_mode(bc._items, cache)

    step_id = core.make_run_id()
    step_dir: Path | None = None
    if do_cache:
        step_dirname = core.make_step_dirname(op, op_kwargs)
        step_dir = bc.cache_root / step_dirname
        step_dir.mkdir(parents=True, exist_ok=True)

    mask_path = _stash_mask(bc)

    tasks: list[_ItemTask] = []
    for i, item in enumerate(bc._items):
        meta_row = (
            dict(bc._metadata.row(i, named=True)) if bc._metadata is not None else {}
        )
        out_path = step_dir / f"sub-{i + 1:04d}{out_ext}" if step_dir else None
        tasks.append(
            _ItemTask(
                idx=i,
                item=item,
                design=bc._designs[i],
                confounds=bc._confounds[i],
                sample_mask=bc._sample_masks[i],
                metadata_row=meta_row,
                mask_path=mask_path,
                out_path=out_path,
            )
        )

    # Single-job fast path — avoid joblib overhead and surface tracebacks
    # without the extra wrapping layer.
    if n_jobs == 1 or len(tasks) == 1:
        results = [_wrap_worker(fn, t) for t in tasks]
    else:
        with (
            parallel_backend(backend, inner_max_num_threads=1),
            tqdm_joblib(total=len(tasks), desc=op, disable=not progress_bar),
        ):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_wrap_worker)(fn, t) for t in tasks
            )

    return results, step_dir, step_id


# ---------------------------------------------------------------------------
# Progress-bar plumbing
# ---------------------------------------------------------------------------


class tqdm_joblib:
    """Context manager that updates a tqdm bar as joblib workers complete.

    Replaces today's submit-time wrapper, which advances on dispatch rather
    than completion. Monkey-patches ``BatchCompletionCallBack.__call__`` for
    the duration of the ``with`` block.
    """

    def __init__(self, total: int, desc: str = "", disable: bool = False) -> None:
        self.total = total
        self.desc = desc
        self.disable = disable
        self._tqdm = None
        self._old_call = None

    def __enter__(self) -> tqdm_joblib:
        if self.disable:
            return self
        from tqdm.auto import tqdm
        from joblib.parallel import BatchCompletionCallBack

        self._tqdm = tqdm(total=self.total, desc=self.desc)
        outer = self
        old = BatchCompletionCallBack.__call__

        def _new_call(self_cb, *args, **kwargs):
            outer._tqdm.update(n=self_cb.batch_size)
            return old(self_cb, *args, **kwargs)

        self._old_call = old
        BatchCompletionCallBack.__call__ = _new_call
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.disable:
            return
        from joblib.parallel import BatchCompletionCallBack

        BatchCompletionCallBack.__call__ = self._old_call
        if self._tqdm is not None:
            self._tqdm.close()


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
