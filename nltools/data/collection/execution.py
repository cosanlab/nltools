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

from nltools.utils import coalesced_gc

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
BUNDLE_SCHEMA_VERSION = 2


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
    """Run ``fn(task)``, wrapping exceptions in ``BrainCollectionWorkerError``.

    The per-subject chokepoint for both the serial fast path and the loky
    path, so ``coalesced_gc()`` here collapses nilearn's per-copy gc storm
    inside each worker process (one real collect per item, not dozens).
    """
    try:
        with coalesced_gc():
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


def _persist_or_keep(
    bc: BrainCollection,
    brains: list,
    *,
    op: str,
    op_kwargs: dict | None = None,
    cache: Literal["auto", True, False] = "auto",
    out_ext: str = ".nii.gz",
) -> tuple[list, list, Path | None]:
    """Persist joint-op output items to disk, or keep them in memory.

    The caching counterpart to `_apply` for ops that compute all their
    outputs jointly (every subject at once) rather than per-item — e.g.
    `align`, which must materialize all subjects to fit its aligner. Resolves
    ``cache=`` against the *source* items with the same ``'auto'`` rule as
    `_apply` (cache when the source is path-backed), and when caching writes
    each output ``BrainData`` to a fresh step subdir under ``bc.cache_root``
    via `_atomic_write_nifti`.

    Args:
        bc: The source collection (its ``_items`` decide ``'auto'``; its
            ``cache_root`` receives the step subdir).
        brains: The freshly computed output ``BrainData`` items, in order.
        op: Short op name for the step-subdir label (e.g. ``'align'``).
        op_kwargs: Scalar kwargs to stamp into the subdir name.
        cache: ``'auto'`` | ``True`` | ``False``.
        out_ext: File extension for the persisted items.

    Returns:
        ``(items, source_paths, step_dir)``. When caching, ``items`` are the
        written ``Path``s, ``source_paths`` mirror them, and ``step_dir`` is
        the new subdir. Otherwise ``items`` are the in-memory ``brains``,
        ``source_paths`` are all ``None``, and ``step_dir`` is ``None``.
    """
    do_cache = _resolve_cache_mode(bc._items, cache)
    if not do_cache:
        return list(brains), [None] * len(brains), None

    step_dir = bc.cache_root / core.make_step_dirname(op, op_kwargs or {})
    step_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _atomic_write_nifti(step_dir / f"sub-{i + 1:04d}{out_ext}", bd)
        for i, bd in enumerate(brains)
    ]
    return paths, list(paths), step_dir


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
    step_id: str | None = None,
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

    if step_id is None:
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


def _write_bundle(
    out_path: Path,
    *,
    datasets: dict[str, np.ndarray | bytes],
    attrs: dict[str, Any],
) -> Path:
    """Generic bundle writer (atomic tmp+rename, ``h5py.File(locking=False)``).

    Used by ``write_glm_bundle`` and ``write_ridge_bundle``. Embeds raw bytes
    via ``np.frombuffer(..., dtype=np.uint8)`` so HDF5 stores them as a
    fixed-shape uint8 dataset (rather than a variable-length blob).
    """
    import h5py

    tmp = out_path.parent / f".tmp_{out_path.name}"
    with h5py.File(tmp, "w", locking=False) as f:
        for name, value in datasets.items():
            if isinstance(value, bytes):
                f.create_dataset(name, data=np.frombuffer(value, dtype=np.uint8))
            else:
                f.create_dataset(name, data=value)
        for k, v in attrs.items():
            f.attrs[k] = v
    os.rename(tmp, out_path)
    return out_path


def _read_bundle_attrs_and_validate(path: Path) -> tuple[Any, dict[str, Any]]:
    """Open ``path``, validate ``bundle_schema_version``, warn on nltools mismatch.

    Returns the open ``h5py.File`` and a dict of decoded attrs.
    Caller is responsible for ``close()``.
    """
    import h5py

    f = h5py.File(path, "r", locking=False)
    try:
        version = int(f.attrs.get("bundle_schema_version", 0))
        if version != BUNDLE_SCHEMA_VERSION:
            f.close()
            raise ValueError(
                f"bundle schema mismatch: file is v{version}, current is "
                f"v{BUNDLE_SCHEMA_VERSION}. Re-run the upstream op to "
                f"regenerate {path.name}."
            )
        file_nltools_version = f.attrs.get("nltools_version", b"") or b""
        if isinstance(file_nltools_version, bytes):
            file_nltools_version = file_nltools_version.decode()
        try:
            from nltools import __version__ as current_version
        except Exception:
            current_version = ""
        if file_nltools_version and file_nltools_version != current_version:
            import warnings

            warnings.warn(
                f"bundle was written by nltools v{file_nltools_version}, "
                f"reading with v{current_version}. Usually fine within a "
                f"minor version.",
                stacklevel=3,
            )
        return f, dict(f.attrs)
    except Exception:
        f.close()
        raise


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
    standardize: str | None,
    model_kwargs: dict,
    step_id: str,
    parent_step_id: str | None,
    op: str,
    op_kwargs: dict,
    nltools_version: str,
) -> Path:
    """Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).

    Layout (see ``docs/development/execution-model.md``):
        /betas, /residuals, /sigma2, /r2, /X, /mask
        attrs: affine, regressor_names, scale, standardize, model_kwargs,
               nltools_version, bundle_schema_version,
               step_id, parent_step_id, op, kwargs (JSON-encoded).

    Mask is embedded as a dataset (raw NIfTI bytes) so the bundle is
    portable across machines. Uses ``h5py.File(..., locking=False)``.
    """
    import json

    return _write_bundle(
        out_path,
        datasets={
            "betas": np.asarray(betas),
            "residuals": np.asarray(residuals),
            "sigma2": np.asarray(sigma2),
            "r2": np.asarray(r2),
            "X": np.asarray(X),
            "mask": mask_bytes,
        },
        attrs={
            "affine": np.asarray(affine),
            "regressor_names": json.dumps(list(regressor_names)),
            "scale": bool(scale),
            "standardize": standardize or "",
            "model_kwargs": json.dumps(model_kwargs),
            "nltools_version": nltools_version,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "step_id": step_id,
            "parent_step_id": parent_step_id or "",
            "op": op,
            "kwargs": json.dumps(op_kwargs),
        },
    )


def read_glm_bundle(path: Path) -> dict[str, Any]:
    """Read a GLM bundle. Validates ``bundle_schema_version``.

    Schema-version mismatch raises with a migration message; nltools-version
    mismatch logs a warning but does not refuse — bundles are usually
    forward-compatible within a minor version.
    """
    import json

    f, attrs = _read_bundle_attrs_and_validate(Path(path))
    try:
        out: dict[str, Any] = {
            "betas": f["betas"][:],
            "residuals": f["residuals"][:],
            "sigma2": f["sigma2"][:],
            "r2": f["r2"][:],
            "X": f["X"][:],
            "mask_bytes": bytes(f["mask"][:]),
            "affine": np.asarray(attrs["affine"]),
            "regressor_names": json.loads(attrs["regressor_names"]),
            "scale": bool(attrs["scale"]),
            "standardize": _to_str(attrs["standardize"]) or None,
            "model_kwargs": json.loads(attrs["model_kwargs"]),
            "step_id": _to_str(attrs["step_id"]),
            "parent_step_id": _to_str(attrs.get("parent_step_id", "")) or None,
            "op": _to_str(attrs["op"]),
            "kwargs": json.loads(attrs["kwargs"]),
            "nltools_version": _to_str(attrs.get("nltools_version", "")),
            "bundle_schema_version": int(attrs["bundle_schema_version"]),
        }
        return out
    finally:
        f.close()


def write_ridge_bundle(
    out_path: Path,
    *,
    weights: np.ndarray,
    intercept: np.ndarray,
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
    (``weights``, ``intercept``, ``cv_scores``, ``predictions``, ``scores``).
    """
    import json

    return _write_bundle(
        out_path,
        datasets={
            "weights": np.asarray(weights),
            "intercept": np.asarray(intercept),
            "cv_scores": np.asarray(cv_scores),
            "predictions": np.asarray(predictions),
            "scores": np.asarray(scores),
            "X": np.asarray(X),
            "mask": mask_bytes,
        },
        attrs={
            "affine": np.asarray(affine),
            "regressor_names": json.dumps(list(regressor_names)),
            "model_kwargs": json.dumps(model_kwargs),
            "nltools_version": nltools_version,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "step_id": step_id,
            "parent_step_id": parent_step_id or "",
            "op": op,
            "kwargs": json.dumps(op_kwargs),
        },
    )


def read_ridge_bundle(path: Path) -> dict[str, Any]:
    """Read a ridge bundle. Same schema/version handling as ``read_glm_bundle``."""
    import json

    f, attrs = _read_bundle_attrs_and_validate(Path(path))
    try:
        out: dict[str, Any] = {
            "weights": f["weights"][:],
            "intercept": f["intercept"][:],
            "cv_scores": f["cv_scores"][:],
            "predictions": f["predictions"][:],
            "scores": f["scores"][:],
            "X": f["X"][:],
            "mask_bytes": bytes(f["mask"][:]),
            "affine": np.asarray(attrs["affine"]),
            "regressor_names": json.loads(attrs["regressor_names"]),
            "model_kwargs": json.loads(attrs["model_kwargs"]),
            "step_id": _to_str(attrs["step_id"]),
            "parent_step_id": _to_str(attrs.get("parent_step_id", "")) or None,
            "op": _to_str(attrs["op"]),
            "kwargs": json.loads(attrs["kwargs"]),
            "nltools_version": _to_str(attrs.get("nltools_version", "")),
            "bundle_schema_version": int(attrs["bundle_schema_version"]),
        }
        return out
    finally:
        f.close()


def _to_str(v: Any) -> str:
    """Decode HDF5 attr that might be bytes or str into a plain str."""
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


# ---------------------------------------------------------------------------
# Bundle extraction from a fitted BrainData
# ---------------------------------------------------------------------------


def _mask_to_bytes(mask) -> bytes:
    """Serialize a Nifti1Image mask to raw NIfTI bytes for embedding."""
    import nibabel as nib

    if not isinstance(mask, nib.Nifti1Image):
        mask = nib.Nifti1Image(np.asarray(mask.dataobj), mask.affine)
    return mask.to_bytes()


def _bytes_to_mask(b: bytes):
    """Inverse of ``_mask_to_bytes``."""
    import nibabel as nib

    return nib.Nifti1Image.from_bytes(b)


def _extract_ridge_bundle_data(bd) -> dict[str, Any]:
    """Pull ridge arrays + design info off a fitted ``BrainData``.

    Mirrors ``_extract_glm_bundle_data``. Pulls weights/scores/intercept
    off the fitted model, plus held-out CV ``predictions`` and per-fold
    ``cv_scores`` from ``bd.cv_results_`` when a CV ran.
    """
    import numpy as np

    weights = np.asarray(bd.ridge_weights.data)  # (n_features, n_voxels)
    scores = np.asarray(bd.ridge_scores.data).reshape(-1)  # (n_voxels,)
    # Ridge fits store the design under bd.X_ (ndarray); GLM also sets
    # bd.design_matrix (a DesignMatrix carrying column names).
    X_src = getattr(bd, "design_matrix", None)
    if X_src is None:
        X_src = bd.X_
    X = np.asarray(X_src)
    if hasattr(X_src, "columns"):
        regressor_names = list(X_src.columns)
    else:
        regressor_names = [f"x{i}" for i in range(X.shape[1])]
    n_voxels = weights.shape[1] if weights.ndim == 2 else weights.shape[0]

    intercept = np.asarray(getattr(bd.model_, "intercept_", 0.0))
    if intercept.ndim == 0:
        intercept = np.full(n_voxels, float(intercept), dtype=np.float32)

    cv_results = getattr(bd, "cv_results_", None)
    if cv_results is not None:
        cv_scores = np.asarray(cv_results.get("scores"))
        predictions = np.asarray(cv_results["predictions"].data)
    else:
        cv_scores = np.zeros((0, n_voxels), dtype=np.float32)
        predictions = np.zeros((0, n_voxels), dtype=np.float32)

    return {
        "weights": weights.astype(np.float32),
        "intercept": intercept.astype(np.float32),
        "cv_scores": cv_scores.astype(np.float32),
        "predictions": predictions.astype(np.float32),
        "scores": scores.astype(np.float32),
        "X": X.astype(np.float32),
        "mask_bytes": _mask_to_bytes(bd.mask),
        "affine": np.asarray(bd.mask.affine),
        "regressor_names": regressor_names,
    }


def _extract_glm_bundle_data(bd) -> dict[str, Any]:
    """Pull GLM arrays + design info off a fitted ``BrainData``."""
    import numpy as np

    betas = np.asarray(bd.glm_betas.data)
    residuals = np.asarray(bd.glm_residual.data)
    r2 = np.asarray(bd.glm_r2.data).reshape(-1)
    X = np.asarray(bd.design_matrix)
    n_obs = X.shape[0]
    rank = int(np.linalg.matrix_rank(X))
    df = max(n_obs - rank, 1)
    sigma2 = (residuals**2).sum(axis=0) / df
    return {
        "betas": betas.astype(np.float32),
        "residuals": residuals.astype(np.float32),
        "sigma2": sigma2.astype(np.float32),
        "r2": r2.astype(np.float32),
        "X": X.astype(np.float32),
        "mask_bytes": _mask_to_bytes(bd.mask),
        "affine": np.asarray(bd.mask.affine),
        "regressor_names": list(bd.design_matrix.columns),
    }


# ---------------------------------------------------------------------------
# Per-subject worker for BC.fit
# ---------------------------------------------------------------------------


def _build_design_context(task: _ItemTask, bd) -> _DesignContext:
    """Construct a ``_DesignContext`` from an ``_ItemTask`` for X=callable."""
    meta = task.metadata_row

    def _path_or_none(key: str) -> Path | None:
        v = meta.get(key)
        return Path(v) if v else None

    return _DesignContext(
        bd=bd,
        dm=task.design,
        confounds=task.confounds,
        sample_mask=task.sample_mask,
        metadata=meta,
        subject=meta.get("subject"),
        session=meta.get("session"),
        run=meta.get("run"),
        task=meta.get("task"),
        TR=float(meta.get("TR", 2.0)),
        bold_path=_path_or_none("bold_path") or Path(str(task.item)),
        events_path=_path_or_none("events_path"),
        confounds_path=_path_or_none("confounds_path"),
    )


def _resolve_x_for_item(
    task: _ItemTask,
    bd,
    *,
    x_mode: str,
    x_value: Any,
):
    """Per-item X resolution shared by ``fit`` and ``predict``."""
    if x_mode == "designs":
        return task.design
    if x_mode == "shared":
        return x_value
    if x_mode == "list":
        return x_value[task.idx]
    if x_mode == "callable":
        ctx = _build_design_context(task, bd)
        return x_value(ctx)
    raise ValueError(f"unknown x_mode {x_mode!r}")


# ---------------------------------------------------------------------------
# Contrast computation from a bundle
# ---------------------------------------------------------------------------


_CONTRAST_TERM_RE = None  # lazily compiled in _parse_contrast_string


def _parse_contrast_string(expr: str, regressor_names: list[str]) -> np.ndarray:
    """Parse ``"A - B"`` / ``"2*A - B"`` into a contrast vector over ``regressor_names``.

    Supports terms of the form ``[+-]? coef * name`` separated by ``+`` / ``-``.
    Coefficient is optional (defaults to 1). Names must exactly match
    ``regressor_names``.
    """
    import re

    global _CONTRAST_TERM_RE
    if _CONTRAST_TERM_RE is None:
        _CONTRAST_TERM_RE = re.compile(
            r"\s*([+-]?)\s*(?:([0-9.]+)\s*\*\s*)?([\w.]+)\s*"
        )

    name_idx = {n: i for i, n in enumerate(regressor_names)}
    c = np.zeros(len(regressor_names), dtype=np.float64)
    pos = 0
    matched_any = False
    for m in _CONTRAST_TERM_RE.finditer(expr.strip()):
        if m.start() != pos:
            raise ValueError(f"could not parse contrast {expr!r} at position {pos}")
        sign = -1.0 if m.group(1) == "-" else 1.0
        coef = float(m.group(2)) if m.group(2) else 1.0
        name = m.group(3)
        if name not in name_idx:
            raise ValueError(f"regressor {name!r} not in design ({regressor_names!r})")
        c[name_idx[name]] += sign * coef
        pos = m.end()
        matched_any = True
    if not matched_any or pos != len(expr.strip()):
        raise ValueError(f"could not parse contrast {expr!r}")
    return c


def _coerce_contrast(contrast, regressor_names: list[str]) -> np.ndarray:
    """Turn ``contrast`` (str/list/ndarray) into a contrast vector."""
    if isinstance(contrast, str):
        return _parse_contrast_string(contrast, regressor_names)
    arr = np.asarray(contrast, dtype=np.float64)
    if arr.shape != (len(regressor_names),):
        raise ValueError(
            f"contrast vector length {arr.shape} != n_regressors "
            f"({len(regressor_names)})"
        )
    return arr


_CONTRAST_TYPES = ("beta", "t", "z", "p", "se")


def _compute_contrast_from_bundle(
    bundle: dict[str, Any],
    contrast,
    contrast_type: str,
):
    """Compute one contrast statistic (or all) from a GLM bundle.

    Closed-form OLS contrast statistics:
      - effect = c'β
      - var(c'β) = sigma² · c' (X'X)⁻¹ c
      - t = effect / sqrt(var)
      - df = n_obs − rank(X)
      - p = 2 · sf(|t|, df), z = sign(t) · isf(p/2)

    Returns an ndarray (one stat per voxel) or, if ``contrast_type='all'``,
    a dict with ``{'beta', 't', 'z', 'p', 'se'}``.
    """
    if contrast_type not in (*_CONTRAST_TYPES, "all"):
        raise ValueError(
            f"contrast_type must be one of {_CONTRAST_TYPES + ('all',)}; "
            f"got {contrast_type!r}"
        )

    X = bundle["X"]
    betas = bundle["betas"]
    sigma2 = bundle["sigma2"]
    c = _coerce_contrast(contrast, bundle["regressor_names"])

    effect = c @ betas
    XtX_inv = np.linalg.pinv(X.T @ X)
    var_factor = float(c @ XtX_inv @ c)
    se = np.sqrt(var_factor * sigma2)
    # avoid divide-by-zero where sigma2 is exactly 0 (degenerate voxels)
    t_stat = np.divide(
        effect, se, out=np.zeros_like(effect, dtype=np.float64), where=se > 0
    )

    if contrast_type == "beta":
        return effect
    if contrast_type == "se":
        return se
    if contrast_type == "t":
        return t_stat

    from scipy.stats import norm, t as t_dist

    df = max(X.shape[0] - int(np.linalg.matrix_rank(X)), 1)
    p = 2.0 * t_dist.sf(np.abs(t_stat), df)

    if contrast_type == "p":
        return p

    z = np.sign(t_stat) * norm.isf(np.clip(p / 2.0, 1e-300, 1.0))

    if contrast_type == "z":
        return z

    return {"beta": effect, "se": se, "t": t_stat, "p": p, "z": z}


def _contrast_worker(
    task: _ItemTask,
    *,
    contrast,
    contrast_type: str,
    step_id: str,
    parent_step_id: str | None,
    op: str,
    op_kwargs: dict,
):
    """Worker for ``BC.compute_contrasts``: read bundle, compute, write NIfTI."""
    from ..braindata import BrainData as _BrainData

    if not isinstance(task.item, (str, Path)):
        raise ValueError(
            "compute_contrasts requires bundle-path items; got an in-memory "
            "object. Run .fit(model='glm', cache=True) first or use cache=True."
        )

    bundle = read_glm_bundle(Path(task.item))
    arr = _compute_contrast_from_bundle(bundle, contrast, contrast_type)
    if isinstance(arr, dict):
        # contrast_type='all' is dispatched at the BC layer; should not
        # reach the per-stat worker as 'all'.
        raise RuntimeError("worker received contrast_type='all' — bug")

    mask = _bytes_to_mask(bundle["mask_bytes"])
    bd = _BrainData(arr.reshape(1, -1), mask=mask)

    if task.out_path is None:
        return bd

    _atomic_write_nifti(task.out_path, bd)
    try:
        from nltools import __version__ as _v
    except Exception:
        _v = ""
    write_lineage_sidecar(
        task.out_path,
        step_id=step_id,
        parent_step_id=parent_step_id,
        op=op,
        op_kwargs=op_kwargs,
        nltools_version=_v,
    )
    return task.out_path


def _predict_after_fit_worker(
    task: _ItemTask,
    *,
    X_new: np.ndarray,
    step_id: str,
    parent_step_id: str | None,
    op_kwargs: dict,
):
    """Worker for ``BC.predict(X_new=...)``: read ridge bundle, apply weights.

    Each task's item is the path to a ridge fit bundle. We compute
    ``X_new @ weights + intercept`` directly (bypassing ``BD.predict``'s
    is_fitted check), wrap as a BrainData on the bundle's mask, and write
    the per-subject NIfTI plus a lineage sidecar.
    """
    from ..braindata import BrainData as _BrainData

    if not isinstance(task.item, (str, Path)) or Path(task.item).suffix not in (
        ".h5",
        ".hdf5",
    ):
        raise ValueError(
            "predict(X_new=...) requires a ridge bundle path; got an in-memory "
            "object. Run .fit(model='ridge', cache=True) first."
        )

    bundle = read_ridge_bundle(Path(task.item))
    weights = bundle["weights"]  # (n_features, n_voxels)
    intercept = bundle["intercept"]  # (n_voxels,)
    X_new_arr = np.asarray(X_new)
    if X_new_arr.ndim == 1:
        X_new_arr = X_new_arr.reshape(1, -1)
    if X_new_arr.shape[1] != weights.shape[0]:
        raise ValueError(
            f"X_new has {X_new_arr.shape[1]} features but ridge weights expect "
            f"{weights.shape[0]} (regressors: {bundle['regressor_names']})."
        )
    y_pred = X_new_arr @ weights + intercept  # (n_obs, n_voxels)

    mask = _bytes_to_mask(bundle["mask_bytes"])
    bd = _BrainData(np.asarray(y_pred, dtype=np.float32), mask=mask)

    if task.out_path is None:
        return bd

    _atomic_write_nifti(task.out_path, bd)
    try:
        from nltools import __version__ as _v
    except Exception:
        _v = ""
    write_lineage_sidecar(
        task.out_path,
        step_id=step_id,
        parent_step_id=parent_step_id,
        op="predict_x_new",
        op_kwargs=op_kwargs,
        nltools_version=_v,
    )
    return task.out_path


def _fit_worker(
    task: _ItemTask,
    *,
    model: str,
    x_mode: str,
    x_value: Any,
    scale: bool,
    standardize: str | None,
    model_kwargs: dict,
    step_id: str,
    parent_step_id: str | None,
    op_kwargs: dict,
):
    """Worker fn for ``BC.fit``: run BD.fit, write a bundle if caching."""
    bd, _ = _materialize(task)
    X = _resolve_x_for_item(task, bd, x_mode=x_mode, x_value=x_value)
    if X is None:
        raise ValueError(
            f"item {task.idx} has no design (X=None and self.designs[i] is None)"
        )

    bd.fit(model=model, X=X, scale=scale, standardize=standardize, **model_kwargs)

    if task.out_path is None:
        return bd

    try:
        from nltools import __version__ as _v
    except Exception:
        _v = ""

    if model == "glm":
        bundle_data = _extract_glm_bundle_data(bd)
        write_glm_bundle(
            task.out_path,
            **bundle_data,
            scale=scale,
            standardize=standardize,
            model_kwargs=model_kwargs,
            step_id=step_id,
            parent_step_id=parent_step_id,
            op="fit",
            op_kwargs=op_kwargs,
            nltools_version=_v,
        )
    elif model == "ridge":
        bundle_data = _extract_ridge_bundle_data(bd)
        write_ridge_bundle(
            task.out_path,
            **bundle_data,
            model_kwargs=model_kwargs,
            step_id=step_id,
            parent_step_id=parent_step_id,
            op="fit",
            op_kwargs=op_kwargs,
            nltools_version=_v,
        )
    else:
        raise ValueError(f"unknown model {model!r}")

    return task.out_path


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
    """Write ``{nifti_path.basename}.json`` with lineage attrs.

    For ``.nii.gz`` paths, both extensions are stripped (so
    ``sub-01.nii.gz`` → ``sub-01.json``). Used for NIfTI outputs
    (``smooth``, ``standardize``, ``compute_contrasts``, ``predict(X_new)``,
    etc.) that don't embed lineage in their own format.
    """
    import json

    name = nifti_path.name
    base = name[:-7] if name.endswith(".nii.gz") else nifti_path.stem
    sidecar = nifti_path.parent / f"{base}.json"
    sidecar.write_text(
        json.dumps(
            {
                "step_id": step_id,
                "parent_step_id": parent_step_id,
                "op": op,
                "kwargs": op_kwargs,
                "nltools_version": nltools_version,
            },
            indent=2,
        )
    )
    return sidecar
