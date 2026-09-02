"""Pure helpers behind `BrainCollection`: metadata coercion, mask and cache-dir resolution, run/step ids.

`coerce_metadata` and `resolve_mask` normalize constructor inputs;
`resolve_cache_dir` applies the cache-location precedence; `make_run_id`
and `make_step_dirname` name the cache root and its per-operation
subdirectories.
"""

from __future__ import annotations

import itertools
import os
import re
import secrets
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import nibabel as nib
import polars as pl

if TYPE_CHECKING:
    import pandas as pd


__all__ = [
    "coerce_metadata",
    "make_run_id",
    "make_step_dirname",
    "resolve_cache_dir",
    "resolve_mask",
]


_TEMPLATE_NAME_RE = re.compile(r"^\d+mm-MNI152-2009[acfsl]+$")


def coerce_metadata(
    metadata: pl.DataFrame | pd.DataFrame | dict | None,
    n_subjects: int,
) -> pl.DataFrame:
    """Coerce a metadata input into a polars DataFrame of length ``n_subjects``.

    Metadata holds simple per-subject values only; DataFrames and arrays
    (designs, confounds, sample masks) travel alongside it in their own
    per-item slots.

    Args:
        metadata (pl.DataFrame | pd.DataFrame | dict | None): A polars or
            pandas DataFrame, a dict of columns, or ``None`` for a default
            ``subject`` column (``sub-0001``, ...).
        n_subjects (int): Required number of rows.

    Returns:
        pl.DataFrame: One row per subject.

    Raises:
        ValueError: If the row count does not equal ``n_subjects``.
    """
    if metadata is None:
        return pl.DataFrame(
            {"subject": [f"sub-{i + 1:04d}" for i in range(n_subjects)]}
        )

    if isinstance(metadata, pl.DataFrame):
        df = metadata
    elif isinstance(metadata, dict):
        df = pl.DataFrame(metadata)
    else:
        # Convert via to_dict to avoid the pyarrow dependency that
        # pl.from_pandas requires for string columns.
        df = pl.DataFrame(metadata.to_dict(orient="list"))

    if df.height != n_subjects:
        raise ValueError(
            f"metadata length ({df.height}) does not match n_subjects ({n_subjects})"
        )
    return df


def resolve_mask(
    mask: nib.Nifti1Image | Path | str,
) -> nib.Nifti1Image:
    """Resolve a mask spec into a Nifti1Image.

    Args:
        mask (Nifti1Image | Path | str): An image, a path, or an nltools
            template name such as ``'3mm-MNI152-2009c'`` (resolved the same
            way `BrainData` resolves template masks).

    Returns:
        Nifti1Image: The loaded mask.

    Raises:
        TypeError: For any other input type.
    """
    if isinstance(mask, nib.Nifti1Image):
        return mask
    if isinstance(mask, str) and _TEMPLATE_NAME_RE.match(mask):
        from ...templates.paths import resolve_template_name

        return cast(
            "nib.Nifti1Image",
            nib.load(resolve_template_name(mask, file_type="mask")),
        )
    if isinstance(mask, (str, Path)):
        return cast("nib.Nifti1Image", nib.load(mask))
    raise TypeError(f"unsupported mask type: {type(mask).__name__}")


def resolve_cache_dir(cache_dir: Path | str | None) -> Path | None:
    """Resolve ``cache_dir`` in precedence order: explicit arg → ``NLTOOLS_CACHE_DIR`` → ``./.nltools_cache``.

    The environment variable is consulted only when the default
    ``'./.nltools_cache'`` was passed. The result is the cache *parent*; the
    collection appends its own ``run_id`` subdirectory at construction.

    Args:
        cache_dir (Path | str | None): Requested location, or ``None`` to ask
            for an auto-cleaned temp dir.

    Returns:
        Path | None: The resolved absolute path, or ``None`` when ``None`` was
            passed.
    """
    if cache_dir is None:
        # Sentinel: caller wants an auto-cleaned tempdir.
        return None
    if isinstance(cache_dir, (str, Path)) and str(cache_dir) == "./.nltools_cache":
        env = os.environ.get("NLTOOLS_CACHE_DIR")
        if env:
            return Path(env).expanduser().resolve()
    return Path(cache_dir).expanduser().resolve()


_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}_[0-9a-f]{8}$")


def make_run_id(now: datetime | None = None) -> str:
    """Build a fresh ``run_id`` of the form ``{timestamp}_{token}``.

    The timestamp is UTC ``YYYYMMDDTHHMMSS``; the token is 8 random hex
    characters, so ids sort lexicographically by time and do not collide
    across processes.

    Args:
        now (datetime | None): Timestamp to use; ``None`` means the current
            UTC time.

    Returns:
        str: The run id.
    """
    now = now or datetime.now(UTC)
    return f"{now.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(4)}"


def _slug_kwargs(kwargs: dict[str, Any]) -> str:
    """Turn ``{'fwhm': 6.0}`` into ``'fwhm-6.0'`` for step-dir naming.

    Sorted by key, simple types only. Skips ``None``. Lossy by design — the
    full kwargs are recorded in the bundle attrs / sidecar, not the dirname.
    """
    parts = [f"{k}-{v}" for k, v in sorted(kwargs.items()) if v is not None]
    return "_".join(parts)


# Process-monotonic counter: a deterministic tiebreak for step subdirs created
# within the same wall-clock second, so lex order tracks creation order even
# when the second-resolution timestamp ties (the uuid tail alone is random).
_STEP_SEQ = itertools.count()


def make_step_dirname(
    op: str,
    kwargs: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> str:
    """Name a cache step subdirectory: ``{timestamp}_{seq}_{token}_{op}_{key_kwargs}``.

    Each call yields a unique name (random token), so running the same op
    with the same parameters twice produces two subdirectories and never
    overwrites. The zero-padded ``seq`` is a process-monotonic counter placed
    after the second-resolution timestamp, so lexicographic order tracks
    creation order even for calls within the same second.

    Args:
        op (str): Short operation name (e.g. ``'smooth'``).
        kwargs (dict[str, Any] | None): Scalar kwargs to slug into the name
            (``{'fwhm': 6.0}`` → ``fwhm-6.0``); ``None`` values are skipped.
        now (datetime | None): Timestamp to use; ``None`` means the current
            UTC time.

    Returns:
        str: The directory name (no path components).
    """
    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%S")
    base = f"{stamp}_{next(_STEP_SEQ):09d}_{secrets.token_hex(4)}_{op}"
    slug = _slug_kwargs(kwargs or {})
    return f"{base}_{slug}" if slug else base


def is_run_id(name: str) -> bool:
    """True if ``name`` matches the run-id regex (``YYYYMMDDTHHMMSS_########``)."""
    return bool(_RUN_ID_RE.match(name))
