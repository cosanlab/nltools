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
        raise NotImplementedError("scaffold")

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
        """Read a collection written by ``write()``.

        This does not recover from cache subdirectories in v0.6.0.
        """
        return io.read(cls, directory, mask=mask, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_subjects(self) -> int:
        return len(self._items)

    @property
    def n_voxels(self) -> int:
        """Return the voxel count from the mask.

        Raises if the mask is unset.
        """
        raise NotImplementedError("scaffold")

    @property
    def mask(self) -> nib.Nifti1Image:
        if self._mask is None:
            raise ValueError("mask not set")
        return self._mask

    @property
    def metadata(self) -> pl.DataFrame:
        if self._metadata is None:
            raise ValueError("metadata not set")
        return self._metadata

    @property
    def designs(self) -> list:  # list[DesignMatrix | None]
        return list(self._designs)

    @property
    def is_loaded(self) -> list[bool]:
        """Per-item flag — True iff the slot holds a ``BrainData`` (not a path)."""
        from ..braindata import BrainData

        return [isinstance(item, BrainData) for item in self._items]

    @property
    def shape(self) -> tuple[int, int | None, int]:
        """``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``."""
        raise NotImplementedError("scaffold")

    @property
    def cache_root(self) -> Path:
        if self._cache_root is None:
            raise ValueError("cache_root not set (constructed with cache_dir=None?)")
        return self._cache_root

    def memory_estimate(self) -> str:
        return io.memory_estimate(self)

    # ------------------------------------------------------------------
    # Indexing and iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator:  # Iterator[BrainData]
        """Yield each item as a ``BrainData``.

        Paths are loaded lazily.
        """
        raise NotImplementedError("scaffold")

    def __getitem__(self, key) -> Any:  # BrainData | BrainCollection
        """See SPEC §"Indexing and iteration" for the full dispatch table.

        bc[i]                 → BrainData
        bc[i:j] / bc[list]    → BrainCollection
        bc[bool_mask]         → BrainCollection
        bc['sub-01']          → BrainData (metadata['subject'] lookup)
        bc[pl.col(...) == ..] → BrainCollection (polars expression)
        """
        raise NotImplementedError("scaffold")

    def iter_pairs(self) -> Iterator[tuple]:  # tuple[BrainData, DesignMatrix | None]
        """Yield ``(BrainData, DesignMatrix | None)`` pairs."""
        raise NotImplementedError("scaffold")

    def filter(
        self,
        predicate: Callable | list | np.ndarray | pl.Series | pd.Series,
    ) -> BrainCollection:
        """Filter to a subset by predicate, mask, or boolean Series."""
        raise NotImplementedError("scaffold")

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
        return self.apply(
            "smooth", fwhm, n_jobs=n_jobs, progress_bar=progress_bar, cache=cache
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
        return self.apply(
            "resample",
            target,
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
        """Map a function over paired ``DesignMatrix``es.

        ``fn`` may take either a ``DesignMatrix`` or a ``DesignContext``;
        the wrapper inspects arity and dispatches.
        """
        raise NotImplementedError("scaffold")

    # ------------------------------------------------------------------
    # Fit / contrasts / predict — mirror BrainData
    # ------------------------------------------------------------------

    def fit(
        self,
        model: str = "glm",
        X: DesignMatrix | list | Callable | None = None,
        *,
        scale: bool = True,
        scale_value: float = 100.0,
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
        raise NotImplementedError("scaffold")

    def compute_contrasts(
        self,
        contrasts: str | list[str] | dict[str, np.ndarray],
        *,
        contrast_type: str = "beta",
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ) -> BrainCollection | dict[str, BrainCollection]:
        """Compute per-subject contrast maps from fit-bundle items.

        Returns:
          single contrast + single ``contrast_type`` → ``BrainCollection``
          multiple contrasts                          → ``dict[str, BrainCollection]``
          ``contrast_type='all'``                     → ``dict['beta'|'t'|'z'|'p'|'se', BrainCollection]``
        """
        raise NotImplementedError("scaffold")

    def predict(
        self,
        y: str | list | np.ndarray | None = None,
        *,
        X_new: np.ndarray | None = None,
        spatial_scale: str = "whole_brain",
        estimator: str = "svm",
        cv: int | str = "loso",
        groups: str | np.ndarray | None = None,
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        radius_mm: float = 10.0,
        scoring: str = "accuracy",
        standardize: bool = True,
        return_weights: bool = True,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
        **kwargs,
    ):  # BrainData | BrainCollection
        """Dispatch prediction according to the provided target argument.

        - ``y=`` only → group MVPA (subjects as samples) → ``BrainData``
        - ``X_new=`` only → per-subject predict-after-fit → ``BrainCollection``
        - both / neither → raise

        ``predict(y=...)`` requires single-map-per-subject items (run
        ``compute_contrasts(...)`` first if you have GLM/ridge bundles).
        """
        raise NotImplementedError("scaffold")

    # ------------------------------------------------------------------
    # Group reductions — delegate to inference.py
    # ------------------------------------------------------------------

    def concat(self) -> BrainData:
        return inference.concat(self)

    def mean(self) -> BrainData:
        return inference.mean(self)

    def std(self) -> BrainData:
        return inference.std(self)

    def var(self) -> BrainData:
        return inference.var(self)

    def median(self) -> BrainData:
        return inference.median(self)

    def sum(self) -> BrainData:
        return inference.sum_(self)

    def min(self) -> BrainData:
        return inference.min_(self)

    def max(self) -> BrainData:
        return inference.max_(self)

    def ttest(self, *, popmean: float = 0.0) -> dict:  # dict[str, BrainData]
        return inference.ttest(self, popmean=popmean)

    def ttest2(
        self,
        other: BrainCollection,
        *,
        equal_var: bool = True,
    ) -> dict:  # dict[str, BrainData]
        return inference.ttest2(self, other, equal_var=equal_var)

    def anova(
        self,
        groups: str | list | np.ndarray,
    ) -> dict:  # dict[str, BrainData]
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
        radius_mm: float | None = 6.0,
        metric: str = "median",
        device: str = "cpu",
        n_jobs: int = -1,
        progress_bar: bool = False,
    ) -> dict:
        return inference.isc(
            self,
            method=method,
            roi_mask=roi_mask,
            radius_mm=radius_mm,
            metric=metric,
            device=device,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )

    def isc_test(
        self,
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
        return inference.isc_test(
            self,
            method=method,
            roi_mask=roi_mask,
            radius_mm=radius_mm,
            n_permute=n_permute,
            permutation_method=permutation_method,
            metric=metric,
            device=device,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            random_state=random_state,
        )

    def align(
        self,
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
    ):  # BrainCollection | tuple[BrainCollection, LocalAlignment]
        return inference.align(
            self,
            method=method,
            scheme=scheme,
            radius_mm=radius_mm,
            parcellation=parcellation,
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
        """Build a cross-validation pipeline for cross-subject prediction.

        See ``pipeline.py`` for details.
        """
        raise NotImplementedError("scaffold")

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
        raise NotImplementedError("scaffold")

    def apply(
        self,
        method: str,
        *args,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
        **kwargs,
    ) -> BrainCollection:
        """Call ``BrainData.<method>(*args, **kwargs)`` on every item in parallel.

        All per-subject methods (``smooth``, ``standardize``, ...) reduce to
        this. Centralizes the ``_apply`` plumbing and the cache-knob handling.
        """
        raise NotImplementedError("scaffold")

    # ------------------------------------------------------------------
    # IO / cleanup — delegate to io.py
    # ------------------------------------------------------------------

    def load(self, indices: list[int] | None = None) -> BrainCollection:
        """Materialize path-backed items in place.

        Returns ``self`` for chaining.
        """
        return io.load(self, indices)

    def unload(self, indices: list[int] | None = None) -> BrainCollection:
        """Drop in-memory data for items with backing paths.

        Returns ``self``.
        """
        return io.unload(self, indices)

    def steps(self) -> list[Path]:
        """List step subdirs under ``cache_root``, oldest to newest (lex-sorted)."""
        raise NotImplementedError("scaffold")

    def write(
        self,
        directory: Path | str,
        *,
        pattern: str = "image_{i:04d}.nii.gz",
        metadata_file: str | None = "metadata.csv",
    ) -> list[Path]:
        return io.write(self, directory, pattern=pattern, metadata_file=metadata_file)

    def cleanup(self) -> None:
        """Remove ``cache_root`` and invalidate every clone derived from ``self``."""
        raise NotImplementedError("scaffold")

    @classmethod
    def cleanup_all(cls, directory: Path | str = ".") -> None:
        """Remove every ``.nltools_cache/{run_id}/`` under ``directory``.

        Wide brush — can kill sibling sessions in the same cwd. Prefer
        ``bc.cleanup()`` for surgical removal.
        """
        raise NotImplementedError("scaffold")

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
        return new

    def _next_step_id(self) -> str:
        """Generate a fresh step id (run-id format: ``{timestamp}_{uuid8}``)."""
        return core.make_run_id()

    # ------------------------------------------------------------------
    # Lifecycle / repr
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Perform no cleanup during finalization.

        Cache cleanup is always explicit via ``bc.cleanup()``.
        """
        return

    def __repr__(self) -> str:
        items = getattr(self, "_items", None)
        if items is None:
            return "BrainCollection(<uninitialized>)"
        n = len(items)
        loaded = sum(self.is_loaded) if items else 0
        return f"BrainCollection(n_subjects={n}, loaded={loaded}/{n})"
