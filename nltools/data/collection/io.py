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

    See ``docs/development/execution-model.md`` for edge cases.
    """
    discovered = discover_bids(
        root,
        task=task,
        space=space,
        sub_labels=sub_labels,
        img_filters=img_filters,
        derivatives_folder=derivatives_folder,
        confounds_strategy=confounds_strategy,
        confounds_kwargs=confounds_kwargs,
        TR=TR,
    )

    bold_paths = discovered["bold_paths"]
    events_dfs = discovered["events_dfs"]
    confounds_dfs = discovered["confounds_dfs"]
    sample_masks = discovered["sample_masks"]
    metadata_rows = discovered["metadata_rows"]
    TRs = discovered["TRs"]

    if not bold_paths:
        raise ValueError("no BOLD files discovered with the given filters")

    # Build paired DesignMatrix from each events DF.
    designs: list = []
    if pair_events and task is not None:
        from ..designmatrix import DesignMatrix

        warned_missing: list[int] = []
        for i, ev_df in enumerate(events_dfs):
            if ev_df is None:
                designs.append(None)
                warned_missing.append(i)
                continue
            tr = TRs[i] if TRs[i] is not None else 2.0
            # Pre-convert via dict so DesignMatrix's polars conversion
            # doesn't trip on string columns without pyarrow installed.
            ev_pl = pl.DataFrame(ev_df.to_dict(orient="list"))
            designs.append(DesignMatrix(ev_pl, run_length="infer", TR=tr))
        if warned_missing:
            import warnings

            warnings.warn(
                f"{len(warned_missing)} BOLD files had no events.tsv; their "
                f"designs are None. Indices: {warned_missing[:5]}"
                f"{'...' if len(warned_missing) > 5 else ''}",
                stacklevel=2,
            )
    else:
        designs = [None] * len(bold_paths)

    # Build polars-friendly metadata table from the per-item dicts.
    cols: dict[str, list] = {}
    for row in metadata_rows:
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    # Pad missing keys per row with None.
    n = len(bold_paths)
    for k, vs in cols.items():
        if len(vs) < n:
            vs.extend([None] * (n - len(vs)))

    bc = cls.from_paths(
        bold_paths,
        mask=mask,
        design_paths=None,
        metadata=cols,
        cache_dir=cache_dir,
    )
    # Inject parallel slots that from_paths doesn't know about.
    bc._designs = list(designs)
    bc._confounds = list(confounds_dfs)
    bc._sample_masks = list(sample_masks)
    return bc


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
    """Build a collection by globbing for BOLD images (and optionally designs).

    ``pattern_groups`` extracts metadata from filename wildcards. Pass
    ``{column_name: wildcard_index}`` (0-based) to capture each ``*`` in
    ``pattern`` into a metadata column.
    """
    import glob as _glob
    import re

    paths = _glob.glob(pattern)
    if sort:
        paths = sorted(paths)
    if not paths:
        raise ValueError(f"no files matched {pattern!r}")

    metadata: dict | None = None
    if pattern_groups:
        # Convert glob pattern to a regex with capture groups for each '*'.
        re_pattern = re.escape(pattern).replace(r"\*", "(.*?)")
        cregex = re.compile(re_pattern + r"$")
        per_col: dict[str, list] = {col: [] for col in pattern_groups}
        for p in paths:
            m = cregex.match(p)
            if m is None:
                # Fallback: leave as None for this row
                for col in per_col:
                    per_col[col].append(None)
                continue
            for col, idx in pattern_groups.items():
                per_col[col].append(m.group(idx + 1))
        metadata = per_col

    designs = None
    if design_pattern is not None:
        design_paths = _glob.glob(design_pattern)
        if sort:
            design_paths = sorted(design_paths)
        if len(design_paths) != len(paths):
            raise ValueError(
                f"design glob matched {len(design_paths)} files but BOLD glob "
                f"matched {len(paths)}"
            )
        designs = design_paths

    return cls.from_paths(
        paths,
        mask=mask,
        design_paths=designs,
        metadata=metadata,
        cache_dir=cache_dir,
    )


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


def _discover_bids_no_task(
    root: Path | str | Any,
    *,
    space: str | None,
    sub_labels: list[str] | None,
    img_filters: list[tuple[str, str]] | None,
    derivatives_folder: str,
    TR: float | str,
) -> dict[str, list]:
    """Lightweight BIDS path discovery when ``task=None`` (no events/models).

    Walks the derivatives folder for ``*_bold.nii*`` files matching the given
    filters. Designs are unset; TR comes from a JSON sidecar lookup or the
    explicit ``TR=`` arg.
    """
    import json as _json
    import re

    root = Path(root)
    derivatives = root / derivatives_folder
    base = derivatives if derivatives.exists() else root

    candidates = sorted(base.rglob("*_bold.nii*"))

    def _match_filters(name: str) -> bool:
        if space is not None and f"space-{space}" not in name:
            return False
        for k, v in img_filters or []:
            if f"{k}-{v}" not in name:
                return False
        if sub_labels is not None:
            entities = dict(re.findall(r"([a-zA-Z]+)-([a-zA-Z0-9]+)", name))
            if entities.get("sub") not in sub_labels:
                return False
        return True

    matched = [p for p in candidates if _match_filters(p.name)]
    if not matched:
        raise ValueError(f"no BOLD files found under {base} matching the given filters")

    out_paths: list[Path] = []
    out_meta: list[dict] = []
    out_TRs: list = []

    for p in matched:
        entities = dict(re.findall(r"([a-zA-Z]+)-([a-zA-Z0-9]+)", p.name))

        tr_val: float | None = None
        if TR == "infer":
            # Try the sidecar adjacent to this BOLD first.
            sidecar = p.with_name(p.name.split(".")[0] + ".json")
            if not sidecar.exists():
                # Search the raw tree for a matching sub/ses/task/run sidecar
                # by stripping derivative-only entities (space-, desc-).
                raw_root = root
                pattern_parts = [
                    f"sub-{entities['sub']}" if entities.get("sub") else "*",
                    f"ses-{entities['ses']}" if entities.get("ses") else "",
                    f"task-{entities['task']}" if entities.get("task") else "*",
                    f"run-{entities['run']}" if entities.get("run") else "",
                ]
                # Build a glob that ignores order: just look for any json
                # whose name contains all required entities.
                for candidate in sorted(raw_root.rglob("*_bold.json")):
                    if all(
                        (part == "" or part == "*" or part in candidate.name)
                        for part in pattern_parts
                    ):
                        sidecar = candidate
                        break
            if sidecar and sidecar.exists():
                try:
                    tr_val = float(_json.loads(sidecar.read_text())["RepetitionTime"])
                except Exception:
                    tr_val = None
            if tr_val is None:
                raise ValueError(f"could not infer TR for {p.name!r}; pass TR=<float>")
        else:
            tr_val = float(TR)

        out_paths.append(p)
        out_meta.append(
            {
                "subject": entities.get("sub"),
                "session": entities.get("ses"),
                "run": int(entities["run"]) if entities.get("run") else None,
                "task": entities.get("task"),
                "space": entities.get("space") or space,
                "bold_path": str(p),
                "events_path": None,
                "confounds_path": None,
                "TR": tr_val,
            }
        )
        out_TRs.append(tr_val)

    n = len(out_paths)
    return {
        "bold_paths": out_paths,
        "events_dfs": [None] * n,
        "confounds_dfs": [None] * n,
        "sample_masks": [None] * n,
        "metadata_rows": out_meta,
        "TRs": out_TRs,
    }


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

    Errors (see ``docs/development/execution-model.md``):
      - Missing TR with ``TR='infer'``: raise.
      - ``task=None`` + ``pair_events=True``: caller silently downgrades.
      - fmriprep absent + ``confounds_strategy`` set: raise.
      - pybids not installed: raise ``ImportError``.
    """
    try:
        from nilearn.glm.first_level import first_level_from_bids
    except ImportError as e:
        raise ImportError(
            "from_bids requires nilearn (pip install nilearn). "
            "pybids is also needed for full BIDS support."
        ) from e

    if task is None:
        # Lightweight discovery — no events, no models. Per spec: when
        # task is None, we silently drop event/design pairing.
        return _discover_bids_no_task(
            root,
            space=space,
            sub_labels=sub_labels,
            img_filters=img_filters,
            derivatives_folder=derivatives_folder,
            TR=TR,
        )

    models, imgs_per_sub, events_per_sub, confounds_per_sub = first_level_from_bids(
        str(root),
        task,
        space_label=space,
        sub_labels=sub_labels,
        img_filters=img_filters or [],
        derivatives_folder=derivatives_folder,
    )

    # If user asked for fmriprep confounds via load_confounds, override
    # the raw DataFrames returned above.
    if confounds_strategy is not None:
        try:
            from nilearn.interfaces.fmriprep import load_confounds
        except ImportError as e:
            raise ImportError(
                "confounds_strategy= requires nilearn.interfaces.fmriprep "
                "(install fmriprep-compatible nilearn)."
            ) from e
        # load_confounds takes a list of BOLD paths and returns
        # (confounds_df, sample_mask) tuples.
        flat_imgs = [str(p) for sub_imgs in imgs_per_sub for p in sub_imgs]
        kwargs = {"strategy": confounds_strategy}
        if confounds_kwargs:
            kwargs.update(confounds_kwargs)
        cf_dfs, smask = load_confounds(flat_imgs, **kwargs)
        # Re-shape back to per-subject lists for alignment with the loop below.
        re_shaped_cf: list = []
        re_shaped_sm: list = []
        offset = 0
        for sub_imgs in imgs_per_sub:
            n = len(sub_imgs)
            re_shaped_cf.append(list(cf_dfs[offset : offset + n]))
            re_shaped_sm.append(list(smask[offset : offset + n]))
            offset += n
        confounds_per_sub = re_shaped_cf
        sample_masks_per_sub = re_shaped_sm
    else:
        # No strategy → no sample masks; keep the raw confounds DFs.
        sample_masks_per_sub = [[None] * len(sub) for sub in imgs_per_sub]

    out_paths: list[Path] = []
    out_events: list = []
    out_confounds: list = []
    out_sample_masks: list = []
    out_meta: list[dict] = []
    out_TRs: list = []

    import re

    for sub_idx, (model, sub_imgs, sub_events, sub_confs) in enumerate(
        zip(models, imgs_per_sub, events_per_sub, confounds_per_sub)
    ):
        sub_smasks = sample_masks_per_sub[sub_idx]
        for run_idx, img_path in enumerate(sub_imgs):
            ev = sub_events[run_idx] if run_idx < len(sub_events) else None
            cf = sub_confs[run_idx] if run_idx < len(sub_confs) else None
            sm = sub_smasks[run_idx] if run_idx < len(sub_smasks) else None

            name = Path(img_path).name
            entities = dict(re.findall(r"([a-zA-Z]+)-([a-zA-Z0-9]+)", name))

            if TR == "infer":
                tr_val = float(model.t_r) if model.t_r is not None else None
                if tr_val is None:
                    raise ValueError(
                        f"could not infer TR for {name!r}; pass TR=<float>"
                    )
            else:
                tr_val = float(TR)

            row = {
                "subject": entities.get("sub"),
                "session": entities.get("ses"),
                "run": int(entities["run"]) if entities.get("run") else None,
                "task": entities.get("task") or task,
                "space": entities.get("space") or space,
                "bold_path": str(img_path),
                "events_path": None,  # nilearn doesn't expose; left for from_bids
                "confounds_path": None,
                "TR": tr_val,
            }

            out_paths.append(Path(img_path))
            out_events.append(ev)
            out_confounds.append(cf)
            out_sample_masks.append(sm)
            out_meta.append(row)
            out_TRs.append(tr_val)

    return {
        "bold_paths": out_paths,
        "events_dfs": out_events,
        "confounds_dfs": out_confounds,
        "sample_masks": out_sample_masks,
        "metadata_rows": out_meta,
        "TRs": out_TRs,
    }


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
