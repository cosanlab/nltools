"""`BrainCollection` — a multi-subject container of `BrainData` with a parallel, mirrored API.

The class is a facade; the work is done by its submodules: `core` (metadata
coercion, mask resolution, run/step ids), `io` (constructors, `write`/`read`,
`load`/`unload`), `execution` (parallel per-subject dispatch, HDF5 fit
bundles), and `inference` (group reductions, ISC, alignment, permutation
tests). The execution model — caching, lineage, parallel write safety — is
documented in ``docs/development/execution-model.md``.
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

if TYPE_CHECKING:
    import pandas as pd

    from ..braindata import BrainData
    from ..designmatrix import DesignMatrix


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BrainCollection",
    "BrainCollectionWorkerError",
]


# ---------------------------------------------------------------------------
# BrainCollection — top-level plain class (non-frozen; load/unload mutate)
# ---------------------------------------------------------------------------


class BrainCollection:
    """A lazy, parallel collection of `BrainData` — one item per subject — whose API mirrors `BrainData`.

    Every item shares one mask and lines up with an optional paired
    `DesignMatrix` (`designs`) and one row of `metadata` (a polars
    DataFrame). Build one from explicit lists (`BrainCollection(...)`) or
    from disk with `from_bids`, `from_glob`, `from_paths`, or `read`.

    Items are **lazy** by default: each is held as a file path and loaded
    into a `BrainData` only when accessed (`bc[i]`, iteration) or inside a
    worker. `load` / `unload` switch items between the two states in place
    and are the only methods that mutate a collection.

    Per-subject methods (`smooth`, `fit`, `predict`, `map`, `apply`, ...)
    run over every item in parallel and return a **new** collection. With
    ``cache='auto'`` (the default) their outputs are written to the
    collection's `cache_root` when the inputs were path-backed and kept in
    memory otherwise; ``cache=True``/``False`` force either behavior. Every
    cached step lands in its own subdirectory and the chain of steps that
    produced a collection is available from `steps`; `cleanup` removes the
    whole cache root. Group reductions (`mean`, `ttest`, `isc`, ...)
    return in-memory `BrainData` (or dicts of them) and never cache.

    Indexing: ``bc[i]`` → `BrainData`; ``bc[i:j]``, ``bc[list]``,
    ``bc[bool_mask]``, ``bc[polars_expr]`` → `BrainCollection`;
    ``bc['sub-01']`` → `BrainData` looked up in ``metadata['subject']``.
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
        """Build a `BrainCollection` from explicit lists.

        Args:
            brains (list[BrainData | Path | str]): One brain image per subject —
                in-memory `BrainData` objects or paths to NIfTI/HDF5 files.
            mask (Nifti1Image | Path | str): Mask shared by every item — an
                image, a path, or an nltools template name (e.g.
                ``'3mm-MNI152-2009c'``).
            designs (list[DesignMatrix | Path | str | None] | None): Optional
                per-subject designs, aligned positionally with ``brains``
                (``None`` entries allowed; length must match).
            metadata (pl.DataFrame | pd.DataFrame | dict | None): Per-subject
                table (one row per item). ``None`` creates a default
                ``subject`` column (``sub-0001``, ...).
            lazy (bool): If True, path items stay as paths until accessed;
                if False, they are loaded into `BrainData` up front.
            cache_dir (Path | str | None): Where cached outputs go. Precedence:
                explicit arg → ``NLTOOLS_CACHE_DIR`` env var →
                ``./.nltools_cache``. ``None`` uses a temp dir that is removed
                at process exit. Resolved once, at construction.
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
        """Build a collection from a BIDS dataset, pairing each BOLD run with its events and confounds.

        Discovery goes through nilearn's ``first_level_from_bids``; each run's
        ``events.tsv`` becomes an unconvolved `DesignMatrix` (add HRF
        convolution, drift, and confound columns yourself with
        `transform_designs`). Metadata gets one row per run with ``subject``,
        ``session``, ``run``, ``task``, ``space``, ``bold_path``, and ``TR``.

        Args:
            root (Path | str): BIDS dataset root.
            mask (Nifti1Image | Path | str): Mask shared by every item.
            task (str | None): BIDS task label. ``None`` discovers BOLD files
                without pairing events (designs are all ``None``).
            space (str | None): Only keep images in this ``space-`` entity.
            sub_labels (list[str] | None): Restrict to these subject labels.
            img_filters (list[tuple[str, str]] | None): Extra BIDS
                ``(entity, value)`` filters on image filenames.
            derivatives_folder (str): Preprocessed-derivatives folder name
                under ``root``.
            pair_events (bool): If True (and ``task`` is set), build a
                `DesignMatrix` from each run's ``events.tsv``. Runs without
                one get ``None`` and a warning.
            confounds_strategy (str | tuple[str, ...] | None): fMRIPrep
                confounds strategy forwarded to nilearn's ``load_confounds``;
                requires fMRIPrep-style derivatives.
            confounds_kwargs (dict | None): Extra keyword arguments for
                ``load_confounds``.
            TR (float | str): Repetition time in seconds, or ``'infer'`` to read
                it from the BIDS sidecars.
            cache_dir (Path | str | None): Cache location; see the class
                constructor.

        Returns:
            BrainCollection: A lazy, path-backed collection.

        Raises:
            ValueError: If no BOLD files match, or ``TR='infer'`` finds no
                repetition time for a run.
            ImportError: If nilearn/pybids are unavailable, or
                ``confounds_strategy`` is set without fMRIPrep support.
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
        """Read a collection previously saved by `write`.

        Discovers ``image_*.nii*`` files in ``directory`` and pairs them with
        the rows of ``metadata.csv`` when present. Only the portable layout
        written by `write` is readable this way — not the cache directories.

        Args:
            directory (Path | str): Directory produced by `write`.
            mask (Nifti1Image | Path | str): Mask shared by every item.
            cache_dir (Path | str | None): Cache location; see the class
                constructor.

        Returns:
            BrainCollection: A lazy, path-backed collection.
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
        """Voxel count from the mask.

        Raises `ValueError` if the mask is unset.
        """
        if self._mask is None:
            raise ValueError("mask not set")
        return int(np.asarray(self._mask.dataobj).astype(bool).sum())

    @property
    def mask(self) -> nib.Nifti1Image:
        """Shared mask image for the collection.

        Raises `ValueError` if the mask is unset.
        """
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
        # A single-image BrainData is 1-D (n_voxels,): one observation, not n_voxels.
        n_obs_set = {1 if bd.data.ndim == 1 else bd.shape[0] for bd in self._items}
        n_obs = next(iter(n_obs_set)) if len(n_obs_set) == 1 else None
        return (n_sub, n_obs, n_vox)

    @property
    def cache_root(self) -> Path:
        """Run-scoped cache directory shared by clones.

        Raises `ValueError` if unset.
        """
        if self._cache_root is None:
            raise ValueError("cache_root not set (constructed with cache_dir=None?)")
        return self._cache_root

    def memory_estimate(self) -> str:
        """Human-readable RAM estimate if every item were loaded into memory.

        The per-item shape is read from the first in-memory item, or from the
        first item loaded on demand when none is in memory.

        Returns:
            str: ``n_subjects``, the per-item shape, and the estimated float32
                total in human-readable units.
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
        """Index by position, label, slice, index list, boolean mask, or polars expression.

        ``bc[i]`` and ``bc['sub-01']`` (a ``metadata['subject']`` lookup)
        return a single `BrainData`; ``bc[i:j]``, ``bc[list_of_int]``,
        ``bc[bool_array]``, and ``bc[pl.col(...) == ...]`` return a
        `BrainCollection` sharing this one's cache root.
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
        """Filter to a subset by predicate, polars expression, or boolean array.

        Args:
            predicate (Callable | pl.Expr | list | np.ndarray | pl.Series | pd.Series):
                A callable ``fn(BrainData) -> bool`` evaluated per item (loads
                each item), a polars expression over ``metadata``, or a
                boolean array-like of length ``n_subjects``.

        Returns:
            BrainCollection: The matching items, sharing this collection's
                cache root.
        """
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
        """Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).

        Args:
            fwhm (float): Gaussian kernel full-width at half-maximum, in mm.
            n_jobs (int): Parallel worker count (``-1`` uses all cores).
            progress_bar (bool): If True, show a progress bar.
            cache (Literal['auto', True, False]): Cache policy for the result
                (``'auto'`` follows source state).

        Returns:
            BrainCollection: A new collection of smoothed items.
        """
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
        """Detrend every subject's image in parallel (delegates to `BrainData.detrend`).

        Args:
            method (str): Detrending method (``'linear'`` or ``'constant'``).
            n_jobs (int): Parallel worker count (``-1`` uses all cores).
            progress_bar (bool): If True, show a progress bar.
            cache (Literal['auto', True, False]): Cache policy for the result
                (``'auto'`` follows source state).

        Returns:
            BrainCollection: A new collection of detrended items.
        """
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

        Args:
            target (Nifti1Image | Path | str): Target image whose grid and
                affine every item is resampled onto.
            interpolation (str): Interpolation method (``'continuous'``,
                ``'linear'``, ``'nearest'``).
            n_jobs (int): Parallel worker count (``-1`` uses all cores).
            progress_bar (bool): If True, show a progress bar.
            cache (Literal['auto', True, False]): Cache policy for the result
                (``'auto'`` follows source state).

        Returns:
            BrainCollection: A new collection of resampled items.
        """
        return self.apply(
            "resample_to",
            img=target,
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
        the parent process — designs are small — so ``n_jobs``,
        ``progress_bar``, and ``cache`` are accepted for consistency with the
        other per-subject methods but ignored.

        Args:
            fn (Callable): Function taking one `DesignMatrix` and returning the
                transformed `DesignMatrix`.
            n_jobs (int): Ignored.
            progress_bar (bool): Ignored.
            cache (Literal['auto', True, False]): Ignored.

        Returns:
            BrainCollection: A new collection with the same items and the
                transformed designs.
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
        """Fit a GLM or ridge model to every subject in parallel (delegates to `BrainData.fit`).

        Each item becomes an HDF5 *fit bundle* holding the fitted arrays, the
        design, and lineage attributes. Feed the result to `compute_contrasts`
        (GLM or ridge) or `predict(X_new=...)` (ridge).

        Args:
            model (str): ``'glm'`` or ``'ridge'``.
            X (DesignMatrix | list | Callable | None): The design. ``None`` uses
                each subject's paired design from `designs` (all must be set);
                a single `DesignMatrix` is shared across subjects; a list gives
                one design per subject (length ``n_subjects``); a callable is
                invoked per subject as ``fn(ctx) -> DesignMatrix``, where
                ``ctx`` exposes ``bd`` (the loaded `BrainData`), ``dm`` (the
                paired design or ``None``), ``confounds``, ``sample_mask``,
                ``metadata`` (that subject's row), and the BIDS entities
                ``subject``, ``session``, ``run``, ``task``, ``TR``,
                ``bold_path``, ``events_path``, ``confounds_path``.
            scale (bool | str): Percent-signal-change scaling before fitting;
                ``'auto'`` resolves to ``False`` for both models.
            standardize (str | None): Per-voxel standardization before fitting;
                ``'auto'`` resolves to ``'zscore'`` for ridge and ``None`` for
                GLM.
            n_jobs (int): Parallel worker count (``-1`` uses all cores).
            progress_bar (bool): If True, show a progress bar.
            cache (Literal['auto', True, False]): Cache policy for the result
                (``'auto'`` follows source state).
            **model_kwargs (dict): Forwarded to `BrainData.fit` (e.g. ``alpha``,
                ``cv`` for ridge).

        Returns:
            BrainCollection: A new collection whose items are per-subject fit
                bundles.

        Raises:
            ValueError: If ``model`` is unknown, or ``X`` is ``None`` while
                some items have no paired design.
            NotImplementedError: For ``model='glm'`` with a ``noise_model``
                other than ``'ols'`` — AR models are available per subject via
                `BrainData.fit`.
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
        """Compute per-subject contrast maps from fit-bundle items (the output of `fit`).

        Each per-subject NIfTI gets a JSON sidecar recording its lineage
        (``step_id``, ``parent_step_id``, ``op``, ``kwargs``,
        ``nltools_version``).

        Args:
            contrasts (str | list[str] | dict[str, np.ndarray]): A contrast
                expression over regressor names (``'A - B'``, ``'2*A - B'``),
                a list of them, or a dict mapping contrast names to expressions
                or weight vectors.
            statistic (str): Which map to return — ``'beta'``, ``'t'``, ``'z'``,
                ``'p'``, ``'se'``, or ``'all'`` for every one.
            n_jobs (int): Parallel worker count (``-1`` uses all cores).
            progress_bar (bool): If True, show a progress bar.
            cache (Literal['auto', True, False]): Cache policy for the result
                (``'auto'`` follows source state).

        Returns:
            BrainCollection | dict[str, BrainCollection] | dict[str, dict[str, BrainCollection]]:
                A `BrainCollection` for one contrast and one statistic; a dict
                keyed by contrast name for several contrasts and one statistic;
                a dict keyed by statistic for one contrast with
                ``statistic='all'``; and a nested ``{name: {stat: collection}}``
                dict for several contrasts with ``statistic='all'``.
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
        cv: int | str = 5,
        groups: str | list | np.ndarray | None = None,
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        radius_mm: float = 10.0,
        scoring: str = "auto",
        standardize: bool = True,
        n_jobs: int = -1,
        random_state: int | None = None,
        progress_bar: bool = False,
        cache: Literal["auto", True, False] = "auto",
    ):  # -> PredictCollection | BrainCollection
        """Per-subject decoding (``y``) or predict-after-fit (``X_new``).

        The per-subject counterpart to every other method on this class —
        one operation per subject, no cross-subject mixing. (For **group
        MVPA** — subjects as samples, one model across the collection — use
        `predict_group`.) Dispatched by which argument is provided:

        1. **Per-subject decoding** (``y``, or omitted with stored labels):
           maps `BrainData.predict` over subjects — one model per subject,
           cross-validated within that subject's own rows — and returns a
           `PredictCollection` carrying the collection's metadata. Stack the
           per-subject decoder maps for second-level inference via
           ``result.weight_maps``.
        2. **Predict-after-fit** (``X_new``): map each subject's fitted
           ridge model over a new design matrix, returning a
           ``BrainCollection`` of predicted maps. Requires ridge fit-bundle
           items (``.fit(model='ridge', cache=True)``).

        Labels travel with the data: with ``y`` omitted, each subject
        decodes its own single-column ``.Y``; ``y='name'`` picks a column of
        each subject's ``.Y``, and ``groups='name'`` does the same for a
        within-subject grouping variable (e.g. run). Alternatively pass one
        shared label array (applied to every subject) or a list of arrays
        (one per subject, in collection order).

        Args:
            y: Per-subject decoding targets — ``None`` (each subject's
                single-column ``.Y``), a ``.Y`` column name, one shared
                array, or a list of per-subject arrays.
            X_new: New design matrix for predict-after-fit (mode 2).
            spatial_scale: One of ``'whole_brain'``, ``'roi'``, or ``'searchlight'``.
            model: Model name or sklearn estimator (see ``BrainData.predict``).
            cv: Within-subject CV — an int fold count (default 5, honoring
                ``groups`` via the Group variants), ``'loo'``, ``'logo'``
                (with ``groups``, e.g. leave-one-run-out), or an sklearn
                splitter.
            groups: Within-subject grouping variable — a ``.Y`` column name,
                one shared array, or a list of per-subject arrays.
            roi_mask: Atlas image for ``spatial_scale='roi'``.
            radius_mm: Searchlight radius.
            scoring: ``'auto'`` → accuracy (classifier) / r2 (regressor).
            standardize: Standardize features within each CV fold.
            n_jobs: CPU workers (subject-level; each subject decodes with
                ``n_jobs=1`` to avoid nested parallelism).
            random_state: Seed for shuffled int-``cv`` folds.
            progress_bar: Whether to display a progress bar.
            cache: ``'auto'`` (cache when the source is path-backed), ``True``,
                or ``False``. Caching writes one predict bundle
                (``.h5``) per subject holding the result's ingredients —
                never a pickled estimator, so cached results have
                ``estimator=None``.

        Returns:
            PredictCollection | BrainCollection: `PredictCollection` (mode 1) or
                `BrainCollection` (mode 2).
        """
        if X_new is not None:
            if y is not None:
                raise ValueError(
                    "Cannot specify both y and X_new. Use y for per-subject "
                    "decoding or X_new for predict-after-fit."
                )
            return self._predict_per_subject(
                X_new,
                n_jobs=n_jobs,
                progress_bar=progress_bar,
                cache=cache,
            )

        return self._predict_mvpa_per_subject(
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
            random_state=random_state,
            progress_bar=progress_bar,
            cache=cache,
        )

    def predict_group(
        self,
        y: str | list | np.ndarray,
        *,
        spatial_scale: str = "whole_brain",
        model: str = "svm",
        cv: int | str = "logo",
        groups: str | np.ndarray | None = None,
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        radius_mm: float = 10.0,
        scoring: str = "auto",
        standardize: bool = True,
        n_permute: int = 0,
        n_jobs: int = -1,
        random_state: int | None = None,
        progress_bar: bool = False,
    ):  # -> Predict
        """Group MVPA: subjects as samples → one model → ``Predict``.

        Stacks the collection into a ``(n_subjects, n_voxels)`` matrix and
        trains a **single** model with subjects as samples (unlike the
        per-subject methods, this deliberately collapses across subjects).
        Requires single-map-per-subject items — run
        ``compute_contrasts(...)`` first for GLM/ridge bundles.

        Args:
            y: Labels/targets, one per subject — an array/list, or the name
                of a metadata column.
            spatial_scale: One of ``'whole_brain'``, ``'roi'``, or ``'searchlight'``.
            model: Model name (see ``BrainData.predict``).
            cv: ``'logo'`` (leave-one-group-out, default — with the default
                ``groups`` this is leave-one-subject-out), ``'loo'``
                (leave-one-out), an int fold count, or an sklearn splitter.
                An int spec **honors** ``groups``: it resolves to
                `StratifiedGroupKFold` (classifiers) / `GroupKFold`
                (regressors) so a group never straddles a train/test
                boundary.
            groups: Group labels, or a metadata column name. Defaults to one
                group per subject; pass ``groups='run'`` (or any metadata
                column) for e.g. leave-one-run-out under ``cv='logo'``.
            roi_mask: Restrict to an ROI.
            radius_mm: Searchlight radius.
            scoring: ``'auto'`` → accuracy (classifier) / r2 (regressor).
            standardize: Standardize features within each CV fold.
            n_permute: If ``> 0``, also build a label-permutation null of
                the CV score — shuffle ``y`` and re-score the identical CV
                (scoring only; no refit/weight-map work) — attached as
                ``permutation_scores`` and ``permutation_pvalue``
                (Phipson-Smyth upper-tail). Forms by ``spatial_scale``:
                whole_brain → null ``(n_permute,)``, p float; roi → null
                ``(n_permute, n_rois)``, p ``(n_rois,)``; searchlight →
                null ``(n_permute, n_voxels)``, p a `BrainData` map (NaN
                where the observed accuracy map is NaN). Default 0 (no
                null).
            n_jobs: CPU workers.
            random_state: Seed for the permutation-null label shuffling.
            progress_bar: Whether to display a progress bar.

        Returns:
            Predict: Result with CV attributes, plus the permutation-null fields
                when ``n_permute > 0``.
        """
        from ..braindata import BrainData
        from . import execution

        # Items must be single-map-per-subject (1, n_voxels) shape. Fit/predict
        # bundles must call compute_contrasts first; a user-saved BrainData
        # .h5 is a plain image and passes (structured bundle_kind check, not
        # bare suffix).
        for i, item in enumerate(self._items):
            if isinstance(item, Path) and (kind := execution.detect_bundle_kind(item)):
                raise ValueError(
                    f"item {i} is a {kind} bundle ({item.name}); call "
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

        # Default the group labels for the leave-one-group-out scheme:
        # each subject is its own group (leave-one-subject-out).
        if cv == "logo" and groups_arr is None:
            groups_arr = np.arange(len(self))

        # Resolve the spec into an sklearn splitter that honors groups
        from sklearn.base import is_classifier

        from ...cross_validation import resolve_cv
        from ..braindata.prediction import resolve_model

        cv_arg = resolve_cv(
            cv, groups=groups_arr, classifier=is_classifier(resolve_model(model))
        )

        def _run_cv(labels: np.ndarray):
            # Forward to BD.predict; returns a Predict dataclass.
            return bd.predict(
                y=labels,
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

        result = _run_cv(y_arr)
        if n_permute > 0:
            # Label-permutation null: shuffle y and re-score the identical CV.
            # An outer loop over the whole CV — not a train/test split — so
            # the null reflects only label exchange. Null iterations run the
            # scoring cores only (no all-data refit, no weight maps).
            result = _predict_group_null(
                result,
                bd=bd,
                y_arr=y_arr,
                groups_arr=groups_arr,
                cv_arg=cv_arg,
                spatial_scale=spatial_scale,
                model=model,
                scoring=scoring,
                standardize=standardize,
                roi_mask=roi_mask,
                radius_mm=radius_mm,
                n_permute=n_permute,
                n_jobs=n_jobs,
                random_state=random_state,
                progress_bar=progress_bar,
            )
        return result

    @staticmethod
    def _classify_spec(value, n_subjects: int, name: str) -> tuple[str, Any]:
        """Classify a per-subject argument into a pickle-friendly spec.

        ``None`` / column-name strings pass through to ``BrainData.predict``
        (resolved against each item's ``.Y``); a flat array/list of scalars
        is one shared vector; a list of array-likes is per-subject (length
        must match the collection).
        """
        if value is None or isinstance(value, str):
            return ("passthrough", value)
        if isinstance(value, (list, tuple)) and all(
            isinstance(el, (np.ndarray, list, tuple)) for el in value
        ):
            if len(value) != n_subjects:
                raise ValueError(
                    f"{name} has {len(value)} entries for {n_subjects} "
                    f"subjects — a per-subject list must match the "
                    f"collection length."
                )
            return ("list", [np.asarray(el) for el in value])
        return ("shared", np.asarray(value))

    def _predict_mvpa_per_subject(
        self,
        y,
        *,
        spatial_scale: str,
        model,
        cv,
        groups,
        roi_mask,
        radius_mm: float,
        scoring: str,
        standardize: bool,
        n_jobs: int,
        random_state: int | None,
        progress_bar: bool,
        cache: Literal["auto", True, False],
    ):  # -> PredictCollection
        """Map ``BrainData.predict(y=...)`` over subjects via ``_apply``."""
        from ..braindata import BrainData
        from ..results import PredictCollection
        from . import execution

        # Fit/predict bundles hold model arrays, not decodable images — refuse
        # eagerly with a pointer to the right paths. A user-saved BrainData
        # .h5 is a plain image and passes (structured bundle_kind check, not
        # bare suffix).
        for i, item in enumerate(self._items):
            if isinstance(item, Path) and (kind := execution.detect_bundle_kind(item)):
                raise ValueError(
                    f"item {i} is a {kind} bundle ({item.name}); predict(y=...) "
                    f"decodes image items. Use predict(X_new=...) for "
                    f"predict-after-fit, or compute_contrasts(...) + "
                    f"predict_group(...) for group MVPA."
                )

        y_spec = self._classify_spec(y, len(self), "y")
        groups_spec = self._classify_spec(groups, len(self), "groups")

        # Eager stored-Y validation for in-memory items — surface a clean
        # collection-level message before workers spin up. Path-backed items
        # resolve at load time inside the worker.
        if y_spec[0] == "passthrough":
            for i, item in enumerate(self._items):
                if not isinstance(item, BrainData):
                    continue
                if item.Y is None or item.Y.is_empty():
                    raise ValueError(
                        f"subject {i} has no stored .Y frame to decode "
                        f"against — set each item's .Y, or pass y= (a shared "
                        f"array or per-subject list)."
                    )
                if isinstance(y, str) and y not in item.Y.columns:
                    raise ValueError(
                        f"y={y!r} is not a column of subject {i}'s .Y "
                        f"(columns: {item.Y.columns})."
                    )

        predict_kwargs = {
            "spatial_scale": spatial_scale,
            "model": model,
            "cv": cv,
            "standardize": standardize,
            "scoring": scoring,
            "roi_mask": roi_mask,
            "radius_mm": radius_mm,
            "random_state": random_state,
        }
        # The bundle's model entry is a structured refit spec (class path +
        # params, or an explicit non-refittable marker) — never a bare repr,
        # which cannot be reconstructed. cv stays informational: a custom
        # splitter isn't needed to refit the final estimator.
        from ..braindata.prediction import _serialize_model_spec

        model_spec = {
            "model": _serialize_model_spec(model),
            "spatial_scale": spatial_scale,
            "cv": cv if isinstance(cv, (int, str)) else repr(cv),
            "scoring": scoring,
            "standardize": standardize,
            "radius_mm": radius_mm,
            "random_state": random_state,
        }
        op_kwargs = {
            "spatial_scale": spatial_scale,
            "model": model if isinstance(model, str) else type(model).__name__,
            "cv": str(cv),
        }
        step_id = core.make_run_id()
        # Hoist into a local so the closure never captures `self` — a closure
        # referencing the collection makes loky/cloudpickle serialize the
        # entire BrainCollection per dispatched task (see execution-model.md).
        parent_step_id = self._step_id

        def worker(task):
            return execution._predict_mvpa_worker(
                task,
                y_spec=y_spec,
                groups_spec=groups_spec,
                predict_kwargs=predict_kwargs,
                model_spec=model_spec,
                step_id=step_id,
                parent_step_id=parent_step_id,
                op_kwargs=op_kwargs,
            )

        raw, _step_dir, _step_id = execution._apply(
            self,
            worker,
            op="predict_mvpa",
            op_kwargs=op_kwargs,
            step_id=step_id,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            cache=cache,
            out_ext=".h5",
        )

        results = tuple(r for r, _ in raw)
        paths = tuple(p for _, p in raw)
        return PredictCollection(
            results=results,
            metadata=self._metadata,
            paths=paths if any(p is not None for p in paths) else None,
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
        # spin up (a structured bundle_kind check — a bare .h5 suffix could
        # be a user-saved BrainData image, which previously died inside
        # read_ridge_bundle with a misleading schema error).
        for i, item in enumerate(self._items):
            kind = (
                execution.detect_bundle_kind(item) if isinstance(item, Path) else None
            )
            if kind != "ridge":
                raise ValueError(
                    f"item {i} is not a ridge bundle; predict(X_new=...) "
                    f"requires items produced by .fit(model='ridge', cache=True)."
                )

        X_new_arr = np.asarray(X_new)
        op_kwargs = {"X_new_shape": list(X_new_arr.shape)}
        step_id = core.make_run_id()
        # Hoist into a local so the closure never captures `self` (see
        # _predict_mvpa_per_subject / execution-model.md).
        parent_step_id = self._step_id

        def worker(task):
            return execution._predict_after_fit_worker(
                task,
                X_new=X_new_arr,
                step_id=step_id,
                parent_step_id=parent_step_id,
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

    def ttest(
        self, *, popmean: float = 0.0, tail: int | str = 2
    ) -> dict:  # dict[str, BrainData]
        """One-sample t-test across subjects (delegates to `inference.ttest`).

        Args:
            popmean: Null-hypothesis population mean to test against.
            tail: `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed:
                mean > popmean; negate the data for the other direction).

        Returns:
            Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps.
        """
        return inference.ttest(self, popmean=popmean, tail=tail)

    def ttest2(
        self,
        other: BrainCollection,
        *,
        equal_var: bool = True,
        tail: int | str = 2,
    ) -> dict:  # dict[str, BrainData]
        """Two-sample t-test between this collection and ``other`` (subject-level).

        Args:
            other: The second collection to compare against.
            equal_var: If True, pooled-variance t-test; if False, Welch's test.
            tail: `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed:
                self > other; swap the operands for the other direction).

        Returns:
            Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps (``mean`` is the
                group difference).
        """
        return inference.ttest2(self, other, equal_var=equal_var, tail=tail)

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
        tail: int | str = 2,
        device: str = "cpu",
        return_null: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
        progress_bar: bool = False,
    ) -> dict:
        """One-sample sign-flipping permutation test across subjects.

        Delegates to the inference engine's `one_sample_permutation_test`
        over the stacked subject data.

        Args:
            n_permute: Number of sign-flip permutations.
            tail: 1 for one-tailed, 2 for two-tailed.
            device: Execution backend — ``None`` (single-threaded numpy),
                ``'cpu'`` (joblib parallel), or ``'gpu'`` (PyTorch).
            return_null: If True, include the null distribution in the result.
            n_jobs: CPU workers when ``device='cpu'`` (-1 = all cores).
            random_state: Seed for the sign-flip RNG.
            progress_bar: Whether to display a progress bar.

        Returns:
            Dict ``{'mean', 'p'}`` of `BrainData` maps, plus
                ``'null_dist'`` when ``return_null=True``.
        """
        return inference.permutation_test(
            self,
            n_permute=n_permute,
            tail=tail,
            device=device,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )

    def permutation_test2(
        self,
        other: BrainCollection,
        *,
        n_permute: int = 5000,
        tail: int | str = 2,
        device: str = "cpu",
        return_null: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
        progress_bar: bool = False,
    ) -> dict:
        """Two-sample permutation test between this collection and ``other``.

        Uses random label shuffling of the pooled subjects, delegating to the
        inference engine's `two_sample_permutation_test`.

        Args:
            other: The second collection to compare against.
            n_permute: Number of label-shuffle permutations.
            tail: 1 for one-tailed, 2 for two-tailed.
            device: Execution backend — ``None`` (single-threaded numpy),
                ``'cpu'`` (joblib parallel), or ``'gpu'`` (PyTorch).
            return_null: If True, include the null distribution in the result.
            n_jobs: CPU workers when ``device='cpu'`` (-1 = all cores).
            random_state: Seed for the shuffling RNG.
            progress_bar: Whether to display a progress bar.

        Returns:
            Dict ``{'mean', 'p'}`` of `BrainData` maps (``mean`` is the group
                difference), plus ``'null_dist'`` when ``return_null=True``.
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
            progress_bar=progress_bar,
        )

    # ------------------------------------------------------------------
    # Cross-subject ops
    # ------------------------------------------------------------------

    def isc(
        self,
        *,
        method: str = "loo",
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        summary: str = "median",
    ) -> dict:
        """Inter-subject correlation (ISC) across the time dimension.

        Args:
            method: ``'loo'`` (leave-one-out template) or ``'pairwise'`` (all
                subject pairs).
            roi_mask: Optional ROI/atlas mask restricting the computation to
                those voxels. The returned maps carry the ROI mask. If None,
                ISC is computed across the collection's whole-brain mask.
            summary: Aggregation across subjects/pairs (e.g. ``'median'``).

        Returns:
            Dict ``{'isc', 'per_subject'}`` for ``method='loo'`` or
                ``{'isc', 'pairs'}`` for ``method='pairwise'`` (``'isc'`` is a
                `BrainData` map).
        """
        return inference.isc(
            self,
            method=method,
            roi_mask=roi_mask,
            summary=summary,
        )

    def isc_test(
        self,
        *,
        method: str = "loo",
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        n_samples: int = 5000,
        summary: str = "median",
        tail: int | str = 2,
        random_state: int | None = None,
    ) -> dict:
        """Bootstrap inference on ISC (per-voxel p-values).

        Resamples subjects with replacement, recomputes ISC each draw, and
        derives a per-voxel p-value from the null centered at 0.

        Args:
            method: ``'loo'`` or ``'pairwise'`` (matches `isc`).
            roi_mask: Optional ROI/atlas mask restricting the computation to
                those voxels. The returned maps carry the ROI mask. If None,
                ISC is computed across the collection's whole-brain mask.
            n_samples: Number of bootstrap resamples.
            summary: Aggregation across subjects/pairs (e.g. ``'median'``).
            tail: `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: ISC > 0).
            random_state: Seed for the bootstrap RNG.

        Returns:
            Dict ``{'isc', 'p', 'null_dist'}`` (``'isc'`` and ``'p'`` are
                `BrainData` maps).
        """
        return inference.isc_test(
            self,
            method=method,
            roi_mask=roi_mask,
            n_samples=n_samples,
            summary=summary,
            tail=tail,
            random_state=random_state,
        )

    def align(  # n_iter exemption: solver iterations, not a permutation count (see api-vocabulary.yml)
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

        Loads every subject into memory — the aligner needs all of them at
        once.

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
            BrainCollection | tuple[BrainCollection, LocalAlignment]: A new
                collection of aligned data, or a ``(collection, model)`` tuple
                when ``return_model=True``.
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
        """Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.

        Args:
            fn (Callable): Function taking one loaded `BrainData` and returning
                a `BrainData`.
            n_jobs (int): Parallel worker count (``-1`` uses all cores).
            progress_bar (bool): If True, show a progress bar.
            cache (Literal['auto', True, False]): Cache policy for the result
                (``'auto'`` follows source state).

        Returns:
            BrainCollection: A new collection of the returned items.
        """
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

        The generic form of the per-subject methods (`smooth`, `standardize`,
        ...) — use it for any `BrainData` method that returns a `BrainData`
        and has no dedicated wrapper here. The method name is passed as ``op``
        rather than ``method`` because several `BrainData` methods take a
        ``method=`` keyword of their own (`standardize`, `detrend`, ...).

        Args:
            op (str): Name of the `BrainData` method to call.
            *args (tuple): Positional arguments forwarded to the method.
            n_jobs (int): Parallel worker count (``-1`` uses all cores).
            progress_bar (bool): If True, show a progress bar.
            cache (Literal['auto', True, False]): Cache policy for the result
                (``'auto'`` follows source state).
            **kwargs (dict): Keyword arguments forwarded to the method.

        Returns:
            BrainCollection: A new collection of the returned items.
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
        """Load path-backed items into memory, in place.

        Args:
            indices (list[int] | None): Items to load; ``None`` loads all.

        Returns:
            BrainCollection: ``self``, for chaining.
        """
        return io.load(self, indices)

    def unload(self, indices: list[int] | None = None) -> BrainCollection:
        """Drop in-memory data for items that have a backing path, in place.

        Items without a backing path (constructed from in-memory `BrainData`
        or produced with ``cache=False``) are left untouched, since dropping
        them would lose the data.

        Args:
            indices (list[int] | None): Items to unload; ``None`` unloads all.

        Returns:
            BrainCollection: ``self``, for chaining.
        """
        return io.unload(self, indices)

    def steps(self) -> list[Path]:
        """Cache subdirectories of the steps that produced this collection's items, oldest to newest.

        One entry per upstream cached operation. Empty when the collection was
        constructed directly or no ancestor wrote to disk.

        Returns:
            list[Path]: Step directories under `cache_root`.
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

        A wide brush — this also removes caches belonging to other live
        collections created in the same directory. Prefer `cleanup` on the
        collection you are done with.

        Args:
            directory (Path | str): Directory whose ``.nltools_cache`` to clear.
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


def _predict_group_null(
    result,
    *,
    bd,
    y_arr: np.ndarray,
    groups_arr: np.ndarray | None,
    cv_arg,
    spatial_scale: str,
    model,
    scoring: str,
    standardize: bool,
    roi_mask,
    radius_mm: float,
    n_permute: int,
    n_jobs: int,
    random_state: int | None,
    progress_bar: bool,
):
    """Attach a label-permutation null to a ``predict_group`` result.

    Each permutation shuffles ``y`` and re-scores the identical CV through
    the score-only cores in ``braindata.prediction`` — no all-data refit, no
    weight-map extraction — parallelized over permutations via joblib.
    The draw stream is one ``rng.permutation(y)`` per iteration in order,
    matching the pre-rework implementation, so whole-brain nulls for a given
    ``random_state`` are unchanged.

    Null / p-value forms by ``spatial_scale`` (p is Phipson-Smyth upper-tail
    via the shared engine helper ``_compute_pvalue``):

    - ``'whole_brain'``: null ``(n_permute,)``, p float.
    - ``'roi'``: null ``(n_permute, n_rois)``, p ``(n_rois,)`` ndarray
      (NaN where the observed per-ROI score is NaN).
    - ``'searchlight'``: null ``(n_permute, n_voxels)``, p a ``BrainData``
      ``(1, n_voxels)`` map (NaN where the observed accuracy map is NaN).
    """
    from dataclasses import replace as dataclass_replace

    from joblib import Parallel, delayed
    from sklearn.base import is_classifier

    from nltools.algorithms.inference.utils import _compute_pvalue

    from ..braindata import prediction as bdp
    from . import execution

    # Rebuild the exact pipeline/scoring the observed run used inside
    # BrainData.predict so null scores are commensurable with the observed.
    resolved_model = bdp.resolve_model(model)
    scoring_resolved = bdp.resolve_scoring(scoring, is_classifier(resolved_model))
    standardize_resolved = bdp._resolve_standardize_for_model(
        resolved_model, standardize
    )
    pipe = bdp.build_pipeline(resolved_model, standardize_resolved, None, None)
    X_data = bd.data

    rng = np.random.default_rng(random_state)
    permuted = [rng.permutation(y_arr) for _ in range(n_permute)]

    if spatial_scale == "whole_brain":
        jobs = (
            delayed(bdp._cv_mean_score)(
                X_data, labels, pipe, cv_arg, groups_arr, scoring_resolved
            )
            for labels in permuted
        )
    elif spatial_scale == "roi":
        label_vec, unique_labels = bdp._resolve_roi_labels(bd.mask, roi_mask)
        jobs = (
            delayed(bdp._cv_roi_mean_scores)(
                X_data,
                labels,
                pipe,
                cv_arg,
                groups_arr,
                scoring_resolved,
                label_vec,
                unique_labels,
            )
            for labels in permuted
        )
    else:  # spatial_scale == 'searchlight'
        from ..braindata.neighborhoods import compute_searchlight_neighborhoods

        neighborhoods = compute_searchlight_neighborhoods(
            bd.mask, radius_mm=radius_mm, use_cache=True
        )
        neighborhood_list = list(neighborhoods.iter_neighborhoods())
        jobs = (
            delayed(bdp._cv_searchlight_scores)(
                X_data,
                labels,
                pipe,
                cv_arg,
                groups_arr,
                scoring_resolved,
                neighborhood_list,
            )
            for labels in permuted
        )

    with execution.tqdm_joblib(
        total=n_permute, desc="Permutation null", disable=not progress_bar
    ):
        rows = Parallel(n_jobs=n_jobs)(jobs)

    null = np.asarray(rows, dtype=np.float64)

    if spatial_scale == "whole_brain":
        pvalue = float(
            np.squeeze(
                _compute_pvalue(np.asarray(result.mean_score), null, tail="upper")
            )
        )
    elif spatial_scale == "roi":
        obs = np.asarray(result.mean_score, dtype=np.float64)
        p = np.asarray(_compute_pvalue(obs, null, tail="upper")).reshape(-1)
        p[~np.isfinite(obs)] = np.nan
        pvalue = p
    else:
        from ..braindata import BrainData as _BrainData

        obs = np.asarray(result.accuracy_map.data, dtype=np.float64).reshape(-1)
        p = np.asarray(_compute_pvalue(obs, null, tail="upper")).reshape(-1)
        p[~np.isfinite(obs)] = np.nan
        pvalue = _BrainData(p.reshape(1, -1), mask=bd.mask)

    return dataclass_replace(result, permutation_scores=null, permutation_pvalue=pvalue)
