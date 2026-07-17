"""BrainCollection — multi-subject brain-data container (v0.6.0).

Public class is a thin facade over module-level helpers:
  - core.py       — metadata coercion, mask resolution, run/step IDs
  - io.py         — constructors, write/read, load/unload
  - execution.py  — parallel ``_apply``, worker dataclasses, HDF5 bundles
  - inference.py  — group reductions, ISC, align, permutation tests
  - pipeline.py   — ``BrainCollectionPipeline`` (CV pipeline; legacy API)

See ``SPEC.md`` for the full design contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)
from collections.abc import Callable, Iterator

import nibabel as nib
import numpy as np
import polars as pl

from . import core, inference, io
from .execution import BUNDLE_SCHEMA_VERSION, BrainCollectionWorkerError
from .pipeline import BrainCollectionPipeline

if TYPE_CHECKING:
    import pandas as pd

    from ..braindata import BrainData
    from ..designmatrix import DesignMatrix


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BrainCollection",
    "BrainCollectionPipeline",
    "BrainCollectionWorkerError",
]


# ---------------------------------------------------------------------------
# BrainCollection — top-level dataclass (non-frozen; load/unload mutate)
# ---------------------------------------------------------------------------


class BrainCollection:
    """Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.

    Constructed via ``__init__`` (explicit lists) or one of the classmethod
    factories (``from_bids``, ``from_glob``, ``from_paths``, ``read``).

    See ``SPEC.md`` §"Public API" for the full contract; key invariants:
      - Per-subject ops route through ``execution._apply`` and return a
        lightweight clone via ``self._clone(...)`` over the same cache root.
      - Path-backed by default after parallel ops; ``cache='auto'`` follows
        source state. ``cache=`` is only accepted on collection-returning ops.
      - ``load`` / ``unload`` are the only methods that mutate ``self``.

    Internal state (mutable list at top level; per-item slots are parallel):

      _items          list[BrainData | Path]        per-item brain data
      _mask           nib.Nifti1Image               shared mask (by reference)
      _designs        list[DesignMatrix | Path | None]
      _confounds      list[pd.DataFrame | None]
      _sample_masks   list[np.ndarray | None]
      _metadata       pl.DataFrame                  simple-typed columns only
      _cache_root     Path | None                   shared by clones
      _step_id        str | None                    this collection's step id
      _parent_step_id str | None                    upstream step id (lineage)
      _step_dirs      list[Path]                    lineage of step subdirs
                                                    that produced these items
      _source_paths   list[Path | None]             per-item backing path
                                                    (None for in-memory only)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        brains: list,  # list[BrainData | Path | str]
        *,
        mask: nib.Nifti1Image | Path | str,
        designs: list | None = None,  # list[DesignMatrix | Path | str | None] | None
        metadata: pl.DataFrame | pd.DataFrame | dict | None = None,
        lazy: bool = True,
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> None:
        """Build a ``BrainCollection`` from explicit lists.

        ``cache_dir`` precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env →
        ``./.nltools_cache``. Pass ``None`` for an auto-cleaned tempdir.
        Resolved at construction and frozen on the instance.
        """
        from ..braindata import BrainData as _BrainData

        n = len(brains)
        if designs is not None and len(designs) != n:
            raise ValueError(
                f"designs length ({len(designs)}) does not match brains ({n})"
            )

        self._mask = core.resolve_mask(mask)
        self._metadata = core.coerce_metadata(metadata, n)

        items: list = []
        source_paths: list[Path | None] = []
        for b in brains:
            if isinstance(b, _BrainData):
                items.append(b)
                source_paths.append(None)
            elif isinstance(b, (str, Path)):
                p = Path(b)
                source_paths.append(p)
                items.append(p if lazy else _BrainData(p, mask=self._mask))
            else:
                raise TypeError(
                    f"brains[i] must be BrainData/Path/str, got {type(b).__name__}"
                )
        self._items = items
        self._source_paths = source_paths

        self._designs = list(designs) if designs is not None else [None] * n
        self._confounds = [None] * n
        self._sample_masks = [None] * n

        resolved = core.resolve_cache_dir(cache_dir)
        if resolved is None:
            import atexit
            import shutil
            import tempfile

            resolved = Path(tempfile.mkdtemp(prefix="nltools_cache_"))
            atexit.register(lambda p=resolved: shutil.rmtree(p, ignore_errors=True))
        self._cache_root = resolved / core.make_run_id()
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._step_id = None
        self._parent_step_id = None
        self._step_dirs: list[Path] = []

    # ------------------------------------------------------------------
    # Classmethod factories — delegate to io.py
    # ------------------------------------------------------------------

    @classmethod
    def from_bids(
        cls,
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
        """Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.

        Full design and edge cases: SPEC §"``from_bids`` — concrete design".
        """
        return io.from_bids(
            cls,
            root,
            mask=mask,
            task=task,
            space=space,
            sub_labels=sub_labels,
            img_filters=img_filters,
            derivatives_folder=derivatives_folder,
            pair_events=pair_events,
            confounds_strategy=confounds_strategy,
            confounds_kwargs=confounds_kwargs,
            TR=TR,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_glob(
        cls,
        pattern: str,
        *,
        mask: nib.Nifti1Image | Path | str,
        design_pattern: str | None = None,
        pattern_groups: dict[str, int] | str | None = None,
        sort: bool = True,
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> BrainCollection:
        """Build a collection by glob-matching brain images (and optional designs).

        Args:
            pattern: Glob pattern matching the per-subject brain image files.
            mask: Shared mask image, path, or nltools template name.
            design_pattern: Optional glob matching per-subject design files,
                paired positionally with the brain images.
            pattern_groups: Regex capture-group spec used to extract metadata
                (e.g. subject/run) from each matched path.
            sort: If True, sort matched paths before pairing (stable ordering).
            cache_dir: Cache-directory precedence: explicit arg →
                ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a
                temp dir.

        Returns:
            A lazy, path-backed `BrainCollection`.
        """
        return io.from_glob(
            cls,
            pattern,
            mask=mask,
            design_pattern=design_pattern,
            pattern_groups=pattern_groups,
            sort=sort,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_paths(
        cls,
        brain_paths: list,
        *,
        mask: nib.Nifti1Image | Path | str,
        design_paths: list | None = None,
        metadata: pl.DataFrame | pd.DataFrame | dict | None = None,
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> BrainCollection:
        """Build a collection from explicit lists of brain (and design) paths.

        Args:
            brain_paths: Per-subject brain image paths.
            mask: Shared mask image, path, or nltools template name.
            design_paths: Optional per-subject design paths, aligned positionally
                with ``brain_paths`` (length must match, ``None`` entries allowed).
            metadata: Optional per-subject metadata (polars/pandas DataFrame or
                dict-of-columns), one row per path.
            cache_dir: Cache-directory precedence: explicit arg →
                ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a
                temp dir.

        Returns:
            A lazy, path-backed `BrainCollection`.
        """
        return io.from_paths(
            cls,
            brain_paths,
            mask=mask,
            design_paths=design_paths,
            metadata=metadata,
            cache_dir=cache_dir,
        )

    @classmethod
    def read(
        cls,
        directory: Path | str,
        *,
        mask: nib.Nifti1Image | Path | str,
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> BrainCollection:
        """Read a collection previously saved by ``write()``.

        Note:
            Does not recover from cache subdirs in v0.6.0.
        """
        return io.read(cls, directory, mask=mask, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_subjects(self) -> int:
        """Number of subjects (items) in the collection."""
        return len(self._items)

    @property
    def n_voxels(self) -> int:
        """Voxel count from the mask. Raises if mask is unset."""
        if self._mask is None:
            raise ValueError("mask not set")
        return int(np.asarray(self._mask.dataobj).astype(bool).sum())

    @property
    def mask(self) -> nib.Nifti1Image:
        """Shared mask image for the collection. Raises if the mask is unset."""
        if self._mask is None:
            raise ValueError("mask not set")
        return self._mask

    @property
    def metadata(self) -> pl.DataFrame:
        """Per-subject metadata as a polars DataFrame (one row per item)."""
        if self._metadata is None:
            raise ValueError("metadata not set")
        return self._metadata

    @property
    def designs(self) -> list:  # list[DesignMatrix | None]
        """Per-subject paired designs (a copy of the list; ``None`` where unpaired)."""
        return list(self._designs)

    @property
    def is_loaded(self) -> list[bool]:
        """Per-item flag — True iff the slot holds a ``BrainData`` (not a path)."""
        from ..braindata import BrainData

        return [isinstance(item, BrainData) for item in self._items]

    @property
    def shape(self) -> tuple[int, int | None, int]:
        """Collection shape as ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

        ``n_obs`` is ``None`` when any item is path-backed (loading just to
        report shape would defeat the purpose) or when items are ragged.
        """
        n_sub = len(self._items)
        n_vox = self.n_voxels
        if not self._items or not all(self.is_loaded):
            return (n_sub, None, n_vox)
        n_obs_set = {bd.shape[0] for bd in self._items}
        n_obs = next(iter(n_obs_set)) if len(n_obs_set) == 1 else None
        return (n_sub, n_obs, n_vox)

    @property
    def cache_root(self) -> Path:
        """Run-scoped cache directory shared by clones. Raises if unset."""
        if self._cache_root is None:
            raise ValueError("cache_root not set (constructed with cache_dir=None?)")
        return self._cache_root

    def memory_estimate(self) -> str:
        """Human-readable RAM estimate if every item were loaded into memory.

        Returns:
            A string reporting ``n_subjects``, the per-item shape (or "unknown"
            for path-backed items not yet loaded), and an estimated float32
            total in MB/GB.
        """
        return io.memory_estimate(self)

    # ------------------------------------------------------------------
    # Indexing and iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator:  # Iterator[BrainData]
        """Yield each item as a ``BrainData``. Loads paths lazily."""
        for i in range(len(self._items)):
            yield self._load_item(i)

    def __getitem__(self, key) -> Any:  # BrainData | BrainCollection
        """See SPEC §"Indexing and iteration" for the full dispatch table.

        bc[i]                 → BrainData
        bc[i:j] / bc[list]    → BrainCollection
        bc[bool_mask]         → BrainCollection
        bc['sub-01']          → BrainData (metadata['subject'] lookup)
        bc[pl.col(...) == ..] → BrainCollection (polars expression)
        """
        if isinstance(key, (int, np.integer)):
            return self._load_item(int(key))

        if isinstance(key, slice):
            indices = list(range(*key.indices(len(self))))
            return self._subset(indices)

        if isinstance(key, str):
            if "subject" not in self._metadata.columns:
                raise KeyError("metadata has no 'subject' column for label lookup")
            subjects = self._metadata["subject"].to_list()
            if key not in subjects:
                raise KeyError(f"subject {key!r} not found")
            return self._load_item(subjects.index(key))

        if isinstance(key, pl.Expr):
            mask = self._metadata.with_columns(_mask=key)["_mask"].to_numpy()
            return self._subset(np.where(mask.astype(bool))[0].tolist())

        if isinstance(key, (list, np.ndarray, pl.Series)):
            arr = key.to_numpy() if isinstance(key, pl.Series) else np.asarray(key)
            if arr.dtype == bool:
                if len(arr) != len(self):
                    raise ValueError(
                        f"boolean mask length ({len(arr)}) != n_subjects ({len(self)})"
                    )
                return self._subset(np.where(arr)[0].tolist())
            return self._subset([int(i) for i in arr])

        raise TypeError(f"unsupported index type: {type(key).__name__}")

    def iter_pairs(self) -> Iterator[tuple]:  # tuple[BrainData, DesignMatrix | None]
        """Yield ``(BrainData, DesignMatrix | None)`` pairs."""
        for i in range(len(self._items)):
            yield self._load_item(i), self._designs[i]

    def filter(
        self,
        predicate: Callable[[Any], Any] | list | np.ndarray | pl.Series | pd.Series,
    ) -> BrainCollection:
        """Filter to a subset by predicate, polars expression, or boolean array."""
        if isinstance(predicate, pl.Expr):
            return self[predicate]

        if callable(predicate):
            fn: Callable[[Any], Any] = predicate
            bool_arr = np.array(
                [bool(fn(self._load_item(i))) for i in range(len(self))]
            )
            return self._subset(np.where(bool_arr)[0].tolist())

        if isinstance(predicate, pl.Series):
            arr = predicate.to_numpy().astype(bool)
        else:
            # numpy array, list, or pandas Series
            arr = np.asarray(predicate, dtype=bool)
        if len(arr) != len(self):
            raise ValueError(
                f"predicate length ({len(arr)}) != n_subjects ({len(self)})"
            )
        return self._subset(np.where(arr)[0].tolist())

    # ------------------------------------------------------------------
    # Per-subject ops — mirror BrainData, run in parallel
    # ------------------------------------------------------------------

    def smooth(
        self,
        fwhm: float,
        *,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> BrainCollection:
        """Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`)."""
        return self.apply(
            "smooth",
            fwhm=fwhm,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )

    def standardize(
        self,
        *,
        axis: int = 0,
        method: str = "center",
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> BrainCollection:
        """Standardize every subject's image in parallel (delegates to `BrainData.standardize`).

        Args:
            axis: Axis along which to standardize (0 = across observations).
            method: Standardization variant (e.g. ``'center'``, ``'zscore'``).
            n_jobs: Parallel worker count (``-1`` uses all cores).
            progress_bar: If True, show a progress bar.
            cache: Cache policy for the result (``'auto'`` follows source state).

        Returns:
            A new `BrainCollection` of standardized items.
        """
        return self.apply(
            "standardize",
            axis=axis,
            method=method,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )

    def detrend(
        self,
        *,
        method: str = "linear",
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> BrainCollection:
        """Detrend every subject's image in parallel (delegates to `BrainData.detrend`)."""
        return self.apply(
            "detrend",
            method=method,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )

    def threshold(
        self,
        *,
        lower: float | None = None,
        upper: float | None = None,
        binarize: bool = False,
        coerce_nan: bool = True,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> BrainCollection:
        """Threshold every subject's image in parallel (delegates to `BrainData.threshold`).

        Args:
            lower: Values below this are zeroed (or set NaN); ``None`` disables.
            upper: Values above this are zeroed (or set NaN); ``None`` disables.
            binarize: If True, set surviving voxels to 1.
            coerce_nan: If True, coerce thresholded-out voxels to NaN instead of 0.
            n_jobs: Parallel worker count (``-1`` uses all cores).
            progress_bar: If True, show a progress bar.
            cache: Cache policy for the result (``'auto'`` follows source state).

        Returns:
            A new `BrainCollection` of thresholded items.
        """
        return self.apply(
            "threshold",
            lower=lower,
            upper=upper,
            binarize=binarize,
            coerce_nan=coerce_nan,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )

    def resample(
        self,
        target,
        *,
        interpolation: str = "continuous",
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> BrainCollection:
        """Resample every subject's image to a target space in parallel.

        Delegates to `BrainData.resample`.

        Args:
            target: Resampling target (image, affine/shape spec, or template)
                passed through to `BrainData.resample`.
            interpolation: Interpolation method (``'continuous'``, ``'linear'``,
                ``'nearest'``).
            n_jobs: Parallel worker count (``-1`` uses all cores).
            progress_bar: If True, show a progress bar.
            cache: Cache policy for the result (``'auto'`` follows source state).

        Returns:
            A new `BrainCollection` of resampled items.
        """
        return self.apply(
            "resample",
            target=target,
            interpolation=interpolation,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )

    def transform_designs(
        self,
        fn: Callable,
        *,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> BrainCollection:
        """Map ``fn(dm) -> DesignMatrix`` over each paired design.

        Items with no paired design are skipped (kept as ``None``). Runs in
        the parent process — designs are small. ``n_jobs``/``progress_bar``/
        ``cache`` are accepted for surface consistency but ignored.
        """
        new_designs = [fn(dm) if dm is not None else None for dm in self._designs]
        return self._clone(_designs=new_designs, _step_id=self._next_step_id())

    # ------------------------------------------------------------------
    # Fit / contrasts / predict — mirror BrainData
    # ------------------------------------------------------------------

    def fit(
        self,
        model: str = "glm",
        X: DesignMatrix | list | Callable | None = None,
        *,
        scale: bool | str = "auto",
        standardize: str | None = "auto",
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
        **model_kwargs,
    ) -> BrainCollection:
        """Per-subject fit; returns a path-backed collection of HDF5 fit bundles.

        ``X`` resolution priority:
          - ``None``         → use ``self.designs`` (must be set per subject)
          - ``DesignMatrix`` → shared across all subjects
          - ``list``         → per-subject (len == n_subjects)
          - ``callable``     → ``fn(ctx: _DesignContext) -> DesignMatrix``
        """
        from . import execution

        if model not in ("glm", "ridge"):
            raise ValueError(f"unknown model {model!r}; expected 'glm' or 'ridge'")

        # AR noise models need per-voxel whitened covariance, which the
        # serializable OLS closed-form contrast path cannot represent. Refuse
        # rather than silently returning OLS-approximated stats; users who need
        # AR can loop BrainData.fit(model='glm', noise_model='ar1') per subject.
        if model == "glm" and model_kwargs.get("noise_model", "ols") != "ols":
            raise NotImplementedError(
                "BrainCollection.fit only supports noise_model='ols' for GLM; "
                f"got {model_kwargs.get('noise_model')!r}. AR noise models are "
                "available per-subject via BrainData.fit(model='glm', "
                "noise_model='ar1'); loop over subjects manually if you need them."
            )

        # Resolve 'auto' sentinels once here so the per-subject bundles persist
        # concrete scale/standardize values (shared logic with BrainData.fit).
        from ..braindata.modeling import resolve_preprocessing_defaults

        scale, standardize = resolve_preprocessing_defaults(model, scale, standardize)

        x_mode, x_value = self._resolve_x_arg(X)

        # Pre-generate step_id so the worker can stamp it into the bundle.
        step_id = self._next_step_id()
        parent_step_id = self._step_id
        op_kwargs_full = {"model": model, **model_kwargs}

        def worker(task):
            return execution._fit_worker(
                task,
                model=model,
                x_mode=x_mode,
                x_value=x_value,
                scale=scale,
                standardize=standardize,
                model_kwargs=model_kwargs,
                step_id=step_id,
                parent_step_id=parent_step_id,
                op_kwargs=op_kwargs_full,
            )

        results, step_dir, _ = execution._apply(
            self,
            worker,
            op=f"fit_{model}",
            op_kwargs={
                k: v
                for k, v in op_kwargs_full.items()
                if isinstance(v, (int, float, bool, str))
            },
            step_id=step_id,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
            out_ext="_fit.h5",
        )
        new_dirs = self._step_dirs + ([step_dir] if step_dir else [])
        new_sources = [r if isinstance(r, Path) else None for r in results]
        return self._clone(
            _items=results,
            _step_id=step_id,
            _step_dirs=new_dirs,
            _source_paths=new_sources,
        )

    def _resolve_x_arg(
        self,
        X: DesignMatrix | list | Callable | None,
    ) -> tuple[str, Any]:
        """Inspect ``X`` and return ``(mode, value)`` for worker dispatch.

        Modes:
          - ``"designs"``: ``value`` is ``None`` (worker reads ``task.design``)
          - ``"shared"``: ``value`` is a single ``DesignMatrix``
          - ``"list"``: ``value`` is a list of ``DesignMatrix`` per item
          - ``"callable"``: ``value`` is the callable
        """
        from ..designmatrix import DesignMatrix as _DesignMatrix

        if X is None:
            if any(d is None for d in self._designs):
                missing = [i for i, d in enumerate(self._designs) if d is None]
                raise ValueError(
                    f"items {missing} have no paired design; pass X= or use "
                    f"from_bids(pair_events=True)"
                )
            return "designs", None
        if isinstance(X, _DesignMatrix):
            return "shared", X
        if isinstance(X, list):
            if len(X) != len(self):
                raise ValueError(
                    f"X list length ({len(X)}) != n_subjects ({len(self)})"
                )
            return "list", list(X)
        if callable(X):
            return "callable", X
        raise TypeError(
            f"X must be None, DesignMatrix, list, or callable; got {type(X).__name__}"
        )

    def compute_contrasts(
        self,
        contrasts: str | list[str] | dict[str, np.ndarray],
        *,
        statistic: str = "beta",
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> (
        BrainCollection
        | dict[str, BrainCollection]
        | dict[str, dict[str, BrainCollection]]
    ):
        """Compute per-subject contrast maps from fit-bundle items.

        Returns:
          single contrast + single ``statistic`` → ``BrainCollection``
          multiple contrasts (single type)            → ``dict[str, BrainCollection]``
          ``statistic='all'`` (single contrast)   → ``dict['beta'|'t'|'z'|'p'|'se', BrainCollection]``
          multiple contrasts + ``statistic='all'`` → nested
                                                         ``dict[name, dict[stat, BrainCollection]]``

        Each per-subject NIfTI gets a JSON sidecar with lineage attrs
        (``step_id``, ``parent_step_id``, ``op``, ``kwargs``,
        ``nltools_version``).
        """
        from . import execution

        # Normalize contrasts → dict[name, contrast_def] + single flag
        if isinstance(contrasts, str):
            contrast_dict = {contrasts: contrasts}
            single_contrast = True
        elif isinstance(contrasts, list):
            contrast_dict = {
                c if isinstance(c, str) else f"c{i}": c for i, c in enumerate(contrasts)
            }
            single_contrast = False
        elif isinstance(contrasts, dict):
            contrast_dict = contrasts
            single_contrast = False
        else:
            raise TypeError(
                f"contrasts must be str/list/dict, got {type(contrasts).__name__}"
            )

        # Normalize stat types
        if statistic == "all":
            stat_types = list(execution._CONTRAST_TYPES)
        elif statistic in execution._CONTRAST_TYPES:
            stat_types = [statistic]
        else:
            raise ValueError(
                f"statistic must be one of "
                f"{execution._CONTRAST_TYPES + ('all',)}; got {statistic!r}"
            )

        per_pair: dict[tuple[str, str], BrainCollection] = {}
        for cname, cdef in contrast_dict.items():
            for stat in stat_types:
                step_id = self._next_step_id()
                parent_step_id = self._step_id
                op = f"contrast_{cname}_{stat}"
                op_kwargs = {"contrast": str(cdef), "contrast_type": stat}

                def worker(
                    task,
                    *,
                    _cdef=cdef,
                    _stat=stat,
                    _step=step_id,
                    _parent=parent_step_id,
                    _op=op,
                    _op_kwargs=op_kwargs,
                ):
                    return execution._contrast_worker(
                        task,
                        contrast=_cdef,
                        contrast_type=_stat,
                        step_id=_step,
                        parent_step_id=_parent,
                        op=_op,
                        op_kwargs=_op_kwargs,
                    )

                results, step_dir, _ = execution._apply(
                    self,
                    worker,
                    op=op,
                    op_kwargs={"contrast_type": stat},
                    step_id=step_id,
                    n_jobs=n_jobs,
                    progress_bar=progress_bar,
                    cache=cache,
                )
                new_dirs = self._step_dirs + ([step_dir] if step_dir else [])
                new_sources = [r if isinstance(r, Path) else None for r in results]
                per_pair[(cname, stat)] = self._clone(
                    _items=results,
                    _step_id=step_id,
                    _step_dirs=new_dirs,
                    _source_paths=new_sources,
                )

        # Reshape outputs by input shape
        if single_contrast and statistic != "all":
            return per_pair[(next(iter(contrast_dict)), statistic)]
        if single_contrast and statistic == "all":
            cname = next(iter(contrast_dict))
            return {stat: per_pair[(cname, stat)] for stat in stat_types}
        if not single_contrast and statistic != "all":
            return {cname: per_pair[(cname, statistic)] for cname in contrast_dict}
        return {
            cname: {stat: per_pair[(cname, stat)] for stat in stat_types}
            for cname in contrast_dict
        }

    def predict(
        self,
        y: str | list | np.ndarray | None = None,
        *,
        X_new: np.ndarray | None = None,
        spatial_scale: str = "whole_brain",
        model: str = "svm",
        cv: int | str = "loso",
        groups: str | np.ndarray | None = None,
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        radius_mm: float = 10.0,
        scoring: str = "auto",
        standardize: bool = True,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ):  # Predict | BrainCollection
        """Predict via one of two paths, dispatched by argument.

          ``y=`` only    → group MVPA (subjects as samples) → ``Predict``
          ``X_new=`` only → per-subject predict-after-fit  → ``BrainCollection``
          both / neither → raise

        ``predict(y=...)`` requires single-map-per-subject items (run
        ``compute_contrasts(...)`` first if you have GLM/ridge bundles).
        """
        if y is None and X_new is None:
            raise ValueError(
                "predict requires either y= (group MVPA) or X_new= "
                "(per-subject predict-after-fit). Got neither."
            )
        if y is not None and X_new is not None:
            raise ValueError(
                "predict accepts y= or X_new=, not both. They're different "
                "operations: y= runs group MVPA across subjects, X_new= runs "
                "per-subject predict-after-fit."
            )

        if y is not None:
            return self._predict_group(
                y,
                spatial_scale=spatial_scale,
                model=model,
                cv=cv,
                groups=groups,
                roi_mask=roi_mask,
                radius_mm=radius_mm,
                scoring=scoring,
                standardize=standardize,
                n_jobs=n_jobs,
                progress_bar=progress_bar,
            )

        return self._predict_per_subject(
            X_new,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )

    def _predict_group(
        self,
        y,
        *,
        spatial_scale: str,
        model: str,
        cv,
        groups,
        roi_mask,
        radius_mm: float,
        scoring: str,
        standardize: bool,
        n_jobs: int,
        progress_bar: bool,
    ):
        """Group MVPA: subjects as samples → ``Predict`` dataclass with CV attrs."""
        from ..braindata import BrainData

        # Items must be single-map-per-subject (1, n_voxels) shape.
        # GLM bundles (.h5) and multi-row BD must call compute_contrasts first.
        for i, item in enumerate(self._items):
            if isinstance(item, Path) and item.suffix in (".h5", ".hdf5"):
                raise ValueError(
                    f"item {i} is a fit bundle ({item.name}); call "
                    f"compute_contrasts(...) first to get a single map per subject."
                )

        # Stack subject maps as one BrainData (n_subjects, n_voxels).
        arrays = []
        for i in range(len(self._items)):
            x = np.asarray(self._load_item(i).data)
            if x.ndim > 1 and x.shape[0] != 1:
                raise ValueError(
                    f"item {i} has shape {x.shape}; predict(y=...) requires "
                    f"single-map-per-subject items. Call compute_contrasts(...) "
                    f"first."
                )
            arrays.append(x.reshape(-1))
        stacked = np.stack(arrays, axis=0).astype(np.float32)
        bd = BrainData(stacked, mask=self._mask)

        # Resolve y from metadata column name if needed.
        y_arr = (
            np.asarray(self._metadata[y].to_list())
            if isinstance(y, str)
            else np.asarray(y)
        )
        groups_arr = (
            np.asarray(self._metadata[groups].to_list())
            if isinstance(groups, str)
            else (np.asarray(groups) if groups is not None else None)
        )

        # Build a CV splitter from the spec
        from sklearn.model_selection import KFold, LeaveOneGroupOut

        cv_arg = cv
        if cv == "loso":
            if groups_arr is None:
                # Default: each subject is its own group
                groups_arr = np.arange(len(self))
            cv_arg = LeaveOneGroupOut()
        elif cv == "loro":
            if groups_arr is None:
                if "run" not in self._metadata.columns:
                    raise ValueError(
                        "cv='loro' requires 'run' metadata or explicit groups"
                    )
                groups_arr = np.asarray(self._metadata["run"].to_list())
            cv_arg = LeaveOneGroupOut()
        elif isinstance(cv, int):
            cv_arg = KFold(cv)

        # Forward to BD.predict; result is a BrainData with CV attrs attached.
        # BD.predict signature: (*, y, X, spatial_scale, model, cv, ...)
        return bd.predict(
            y=y_arr,
            spatial_scale=spatial_scale,
            model=model,
            cv=cv_arg,
            groups=groups_arr,
            roi_mask=roi_mask,
            radius_mm=radius_mm,
            scoring=scoring,
            standardize=standardize,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )

    def _predict_per_subject(
        self,
        X_new: np.ndarray,
        *,
        n_jobs: int,
        progress_bar: bool,
        cache: Literal["auto", True, False],
    ) -> BrainCollection:
        """Per-subject predict-after-fit; each item is a fitted ridge bundle."""
        from . import execution

        # Validate eagerly so the user gets a clean message before workers
        # spin up. Mirrors the bundle-check at _predict_group above.
        for i, item in enumerate(self._items):
            if not (isinstance(item, Path) and item.suffix in (".h5", ".hdf5")):
                raise ValueError(
                    f"item {i} is not a ridge bundle; predict(X_new=...) "
                    f"requires items produced by .fit(model='ridge', cache=True)."
                )

        X_new_arr = np.asarray(X_new)
        op_kwargs = {"X_new_shape": list(X_new_arr.shape)}
        step_id = core.make_run_id()

        def worker(task):
            return execution._predict_after_fit_worker(
                task,
                X_new=X_new_arr,
                step_id=step_id,
                parent_step_id=self._step_id,
                op_kwargs=op_kwargs,
            )

        results, step_dir, step_id = execution._apply(
            self,
            worker,
            op="predict_x_new",
            op_kwargs=op_kwargs,
            step_id=step_id,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )
        new_dirs = self._step_dirs + ([step_dir] if step_dir else [])
        new_sources = [r if isinstance(r, Path) else None for r in results]
        return self._clone(
            _items=results,
            _step_id=step_id,
            _step_dirs=new_dirs,
            _source_paths=new_sources,
        )

    # ------------------------------------------------------------------
    # Group reductions — delegate to inference.py
    # ------------------------------------------------------------------

    def concat(self) -> BrainData:
        """Stack all subject maps into a single `BrainData` (subjects as rows)."""
        return inference.concat(self)

    def mean(self) -> BrainData:
        """Voxelwise mean across subjects as a single `BrainData`."""
        return inference.mean(self)

    def std(self) -> BrainData:
        """Voxelwise standard deviation across subjects as a single `BrainData`."""
        return inference.std(self)

    def var(self) -> BrainData:
        """Voxelwise variance across subjects as a single `BrainData`."""
        return inference.var(self)

    def median(self) -> BrainData:
        """Voxelwise median across subjects as a single `BrainData`."""
        return inference.median(self)

    def sum(self) -> BrainData:
        """Voxelwise sum across subjects as a single `BrainData`."""
        return inference.sum_(self)

    def min(self) -> BrainData:
        """Voxelwise minimum across subjects as a single `BrainData`."""
        return inference.min_(self)

    def max(self) -> BrainData:
        """Voxelwise maximum across subjects as a single `BrainData`."""
        return inference.max_(self)

    def ttest(self, *, popmean: float = 0.0) -> dict:  # dict[str, BrainData]
        """One-sample t-test across subjects (delegates to `inference.ttest`).

        Args:
            popmean: Null-hypothesis population mean to test against.

        Returns:
            Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps.
        """
        return inference.ttest(self, popmean=popmean)

    def ttest2(
        self,
        other: BrainCollection,
        *,
        equal_var: bool = True,
    ) -> dict:  # dict[str, BrainData]
        """Two-sample t-test between this collection and ``other`` (subject-level).

        Args:
            other: The second collection to compare against.
            equal_var: If True, pooled-variance t-test; if False, Welch's test.

        Returns:
            Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps (``mean`` is the
            group difference).
        """
        return inference.ttest2(self, other, equal_var=equal_var)

    def anova(
        self,
        groups: str | list | np.ndarray,
    ) -> dict:  # dict[str, BrainData]
        """One-way ANOVA across subjects grouped by ``groups``.

        Args:
            groups: A metadata column name, or a list/ndarray of length
                ``n_subjects`` giving each subject's group label.

        Returns:
            Dict with ``{'F', 'p'}`` `BrainData` maps plus ``df_between`` and
            ``df_within`` degrees of freedom.
        """
        return inference.anova(self, groups)

    def permutation_test(
        self,
        *,
        n_permute: int = 5000,
        tail: int = 2,
        device: str = "cpu",
        return_null: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> dict:
        """One-sample sign-flipping permutation test across subjects.

        Args:
            n_permute: Number of sign-flip permutations.
            tail: 1 for one-tailed, 2 for two-tailed.
            device: Backend selector (currently informational).
            return_null: If True, include the null distribution in the result.
            n_jobs: Parallel worker count (``-1`` uses all cores).
            random_state: Seed for the sign-flip RNG.

        Returns:
            Dict ``{'mean', 'p'}`` of `BrainData` maps, plus
            ``'null_distribution'`` when ``return_null=True``.
        """
        return inference.permutation_test(
            self,
            n_permute=n_permute,
            tail=tail,
            device=device,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def permutation_test2(
        self,
        other: BrainCollection,
        *,
        n_permute: int = 5000,
        tail: int = 2,
        device: str = "cpu",
        return_null: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> dict:
        """Two-sample permutation test between this collection and ``other``.

        Uses random label shuffling of the pooled subjects.

        Args:
            other: The second collection to compare against.
            n_permute: Number of label-shuffle permutations.
            tail: 1 for one-tailed, 2 for two-tailed.
            device: Backend selector (currently informational).
            return_null: If True, include the null distribution in the result.
            n_jobs: Parallel worker count (``-1`` uses all cores).
            random_state: Seed for the shuffling RNG.

        Returns:
            Dict ``{'mean', 'p'}`` of `BrainData` maps (``mean`` is the group
            difference), plus ``'null_distribution'`` when ``return_null=True``.
        """
        return inference.permutation_test2(
            self,
            other,
            n_permute=n_permute,
            tail=tail,
            device=device,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Cross-subject ops
    # ------------------------------------------------------------------

    def isc(
        self,
        *,
        method: str = "loo",
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        metric: str = "median",
    ) -> dict:
        """Inter-subject correlation (ISC) across the time dimension.

        Args:
            method: ``'loo'`` (leave-one-out template) or ``'pairwise'`` (all
                subject pairs).
            roi_mask: Optional ROI/atlas mask restricting the computation to
                those voxels. The returned maps carry the ROI mask. If None,
                ISC is computed across the collection's whole-brain mask.
            metric: Aggregation across subjects/pairs (e.g. ``'median'``).

        Returns:
            Dict ``{'isc', 'per_subject'}`` for ``method='loo'`` or
            ``{'isc', 'pairs'}`` for ``method='pairwise'`` (``'isc'`` is a
            `BrainData` map).
        """
        return inference.isc(
            self,
            method=method,
            roi_mask=roi_mask,
            metric=metric,
        )

    def isc_test(
        self,
        *,
        method: str = "loo",
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        n_samples: int = 5000,
        metric: str = "median",
        random_state: int | None = None,
    ) -> dict:
        """Bootstrap inference on ISC (per-voxel p-values).

        Resamples subjects with replacement, recomputes ISC each draw, and
        derives a per-voxel two-tailed p-value from the null centered at 0.

        Args:
            method: ``'loo'`` or ``'pairwise'`` (matches `isc`).
            roi_mask: Optional ROI/atlas mask restricting the computation to
                those voxels. The returned maps carry the ROI mask. If None,
                ISC is computed across the collection's whole-brain mask.
            n_samples: Number of bootstrap resamples.
            metric: Aggregation across subjects/pairs (e.g. ``'median'``).
            random_state: Seed for the bootstrap RNG.

        Returns:
            Dict ``{'isc', 'p', 'null_distribution'}`` (``'isc'`` and ``'p'`` are
            `BrainData` maps).
        """
        return inference.isc_test(
            self,
            method=method,
            roi_mask=roi_mask,
            n_samples=n_samples,
            metric=metric,
            random_state=random_state,
        )

    def align(  # nosemgrep: banned-kwarg-permutation-count  # n_iter = LocalAlignment solver iterations, not a permutation count
        self,
        *,
        method: str = "procrustes",
        spatial_scale: str = "searchlight",
        radius_mm: float = 10.0,
        roi_mask: nib.Nifti1Image | None = None,
        n_features: int | None = None,
        n_iter: int = 3,
        device: str = "cpu",
        return_model: bool = False,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ):  # BrainCollection | tuple[BrainCollection, LocalAlignment]
        """Functionally align subjects into a common space via `LocalAlignment`.

        Materializes all subjects (algorithm constraint in v0.6.0).

        Args:
            method: Alignment solver (e.g. ``'procrustes'``).
            spatial_scale: Alignment spatial scale — ``'searchlight'`` (default,
                overlapping spheres) or ``'roi'`` (non-overlapping parcels).
                Whole-brain alignment is not supported at the collection level.
            radius_mm: Searchlight sphere radius in mm (``spatial_scale='searchlight'``).
            roi_mask: Parcellation/ROI mask (used when ``spatial_scale='roi'``).
            n_features: Optional target feature count for the common space.
            n_iter: LocalAlignment solver iteration count (not a permutation count).
            device: Backend selector (``'cpu'``/``'gpu'``).
            return_model: If True, also return the fitted `LocalAlignment`.
            n_jobs: Parallel worker count (``-1`` uses all cores).
            progress_bar: If True, show a progress bar.
            cache: Cache policy for the result (``'auto'`` follows source state).

        Returns:
            A new `BrainCollection` of aligned data, or a
            ``(BrainCollection, LocalAlignment)`` tuple when
            ``return_model=True``.
        """
        return inference.align(
            self,
            method=method,
            spatial_scale=spatial_scale,
            radius_mm=radius_mm,
            roi_mask=roi_mask,
            n_features=n_features,
            n_iter=n_iter,
            device=device,
            return_model=return_model,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )

    # ------------------------------------------------------------------
    # CV pipeline (legacy API, preserved for now)
    # ------------------------------------------------------------------

    def cv(
        self,
        *,
        k: int | None = None,
        method: str = "kfold",
        split_by: str | None = None,
        groups: np.ndarray | None = None,
        n: int = 1000,
        random_state: int | None = None,
    ) -> BrainCollectionPipeline:
        """Build a CV pipeline for cross-subject prediction.

        See ``pipeline.py`` for the builder API. The pipeline's ``predict``
        terminal returns a ``BrainData`` with CV attrs attached.
        """
        from .pipesteps.cv import CVScheme

        cv_scheme = CVScheme(
            scheme=method,
            k=k,
            split_by=split_by,
            n=n,
            random_state=random_state,
        )
        return BrainCollectionPipeline(self, cv=cv_scheme, groups=groups)

    # ------------------------------------------------------------------
    # Composition primitives
    # ------------------------------------------------------------------

    def map(
        self,
        fn: Callable,
        *,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> BrainCollection:
        """Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel."""
        from . import execution

        def worker(task):
            bd, _ = execution._materialize(task)
            result = fn(bd)
            if task.out_path is not None:
                execution._atomic_write_nifti(task.out_path, result)
                return task.out_path
            return result

        results, step_dir, step_id = execution._apply(
            self,
            worker,
            op="map",
            op_kwargs={},
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )
        new_dirs = self._step_dirs + ([step_dir] if step_dir else [])
        new_sources = [r if isinstance(r, Path) else None for r in results]
        return self._clone(
            _items=results,
            _step_id=step_id,
            _step_dirs=new_dirs,
            _source_paths=new_sources,
        )

    def apply(  # nosemgrep: kwargs-internal-forwarding  # generic dispatch to BrainData.<op>(*args, **kwargs)
        self,
        op: str,
        *args,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
        **kwargs,
    ) -> BrainCollection:
        """Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.

        All per-subject methods (``smooth``, ``standardize``, ...) reduce to
        this. Centralizes the ``_apply`` plumbing and the cache-knob handling.
        ``op`` is named ``op`` (not ``method``) to avoid colliding with
        ``BrainData`` methods that themselves take a ``method=`` kwarg
        (``standardize``, ``detrend``, ...).
        """
        from . import execution

        def worker(task):
            bd, _ = execution._materialize(task)
            result = getattr(bd, op)(*args, **kwargs)
            if task.out_path is not None:
                execution._atomic_write_nifti(task.out_path, result)
                return task.out_path
            return result

        results, step_dir, step_id = execution._apply(
            self,
            worker,
            op=op,
            op_kwargs=kwargs,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
        )
        new_dirs = self._step_dirs + ([step_dir] if step_dir else [])
        new_sources = [r if isinstance(r, Path) else None for r in results]
        return self._clone(
            _items=results,
            _step_id=step_id,
            _step_dirs=new_dirs,
            _source_paths=new_sources,
        )

    # ------------------------------------------------------------------
    # IO / cleanup — delegate to io.py
    # ------------------------------------------------------------------

    def load(self, indices: list[int] | None = None) -> BrainCollection:
        """Materialize path-backed items in place. Returns ``self`` for chaining."""
        return io.load(self, indices)

    def unload(self, indices: list[int] | None = None) -> BrainCollection:
        """Drop in-memory data for items with backing paths. Returns ``self``."""
        return io.unload(self, indices)

    def steps(self) -> list[Path]:
        """Step subdirs that produced this collection's items, oldest to newest.

        Lineage chain accumulated through clones (one entry per upstream
        cached op). Empty when the collection was constructed directly or
        no ancestor wrote to disk.
        """
        return list(getattr(self, "_step_dirs", []))

    def write(
        self,
        directory: Path | str,
        *,
        pattern: str = "image_{i:04d}.nii.gz",
        metadata_file: str | None = "metadata.csv",
    ) -> list[Path]:
        """Write a clean, portable copy of the collection outside the cache root.

        Inverse of `BrainCollection.read`. Writes one NIfTI per item plus an
        optional metadata CSV, skipping the internal cache layout so the result
        is shareable/archival.

        Args:
            directory: Output directory (created if missing).
            pattern: Filename template per item, formatted with ``i`` (item index).
            metadata_file: CSV filename for the metadata table, or ``None`` to skip.

        Returns:
            List of written NIfTI paths, in item order.
        """
        return io.write(self, directory, pattern=pattern, metadata_file=metadata_file)

    def cleanup(self) -> None:
        """Remove ``cache_root`` and invalidate every clone derived from ``self``.

        Idempotent — calling twice is a no-op. Path-backed items in any
        clone become unloadable after this; use ``bc.write(...)`` first to
        materialize a portable copy if needed.
        """
        import shutil

        if self._cache_root is not None and self._cache_root.exists():
            shutil.rmtree(self._cache_root)

    @classmethod
    def cleanup_all(cls, directory: Path | str = ".") -> None:
        """Remove every ``.nltools_cache/{run_id}/`` under ``directory``.

        Wide brush — can kill sibling sessions in the same cwd. Prefer
        ``bc.cleanup()`` for surgical removal.
        """
        import shutil

        directory = Path(directory)
        cache_parent = directory / ".nltools_cache"
        if not cache_parent.exists():
            return
        for run_dir in cache_parent.iterdir():
            if run_dir.is_dir() and core.is_run_id(run_dir.name):
                shutil.rmtree(run_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clone(self, **overrides) -> BrainCollection:
        """Lightweight shallow clone — overrides any subset of internal slots.

        Used by every parallel op to return a new collection while sharing
        ``_mask``, ``_metadata``, ``_designs``, and ``_cache_root`` by
        reference. Cost is ``O(n_subjects)`` paths.
        """
        new = self.__class__.__new__(self.__class__)
        new._items = overrides.get("_items", list(self._items))
        new._mask = overrides.get("_mask", self._mask)
        new._designs = overrides.get("_designs", self._designs)
        new._confounds = overrides.get("_confounds", self._confounds)
        new._sample_masks = overrides.get("_sample_masks", self._sample_masks)
        new._metadata = overrides.get("_metadata", self._metadata)
        new._cache_root = overrides.get("_cache_root", self._cache_root)
        new._step_id = overrides.get("_step_id", None)
        new._parent_step_id = overrides.get("_parent_step_id", self._step_id)
        new._step_dirs = overrides.get(
            "_step_dirs", list(getattr(self, "_step_dirs", []))
        )
        new._source_paths = overrides.get(
            "_source_paths", list(getattr(self, "_source_paths", []))
        )
        return new

    def _next_step_id(self) -> str:
        """Generate a fresh step id (run-id format: ``{timestamp}_{uuid8}``)."""
        return core.make_run_id()

    def _load_item(self, i: int) -> BrainData:
        """Return item ``i`` as a ``BrainData``, loading from path if needed."""
        from ..braindata import BrainData as _BrainData

        item = self._items[i]
        if isinstance(item, _BrainData):
            return item
        return _BrainData(item, mask=self._mask)

    def _subset(self, indices: list[int]) -> BrainCollection:
        """Return a clone restricted to ``indices`` (preserves cache + slot alignment)."""
        return self._clone(
            _items=[self._items[i] for i in indices],
            _designs=[self._designs[i] for i in indices],
            _confounds=[self._confounds[i] for i in indices],
            _sample_masks=[self._sample_masks[i] for i in indices],
            _source_paths=[self._source_paths[i] for i in indices],
            _metadata=self._metadata[indices] if self._metadata is not None else None,
        )

    # ------------------------------------------------------------------
    # Lifecycle / repr
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """No-op. Cache cleanup is always explicit (``bc.cleanup()``)."""
        return

    def __repr__(self) -> str:
        items = getattr(self, "_items", None)
        if items is None:
            return "BrainCollection(<uninitialized>)"
        n = len(items)
        loaded = sum(self.is_loaded) if items else 0
        return f"BrainCollection(n_subjects={n}, loaded={loaded}/{n})"
