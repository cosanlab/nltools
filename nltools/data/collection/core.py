"""Module-level helpers for BrainCollection.

Pure functions: metadata coercion, mask resolution, run/step ID generation,
step-directory naming. No class state lives here.
"""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def coerce_metadata(
    metadata: pl.DataFrame | pd.DataFrame | dict | None,
    n_subjects: int,
) -> pl.DataFrame:
    """Coerce a metadata input into a polars DataFrame of length ``n_subjects``.

    Accepts polars/pandas DataFrames or a dict-of-columns. ``None`` yields a
    DataFrame with a default ``subject`` column (``sub-0001``, ...).

    Polars ``metadata`` cannot hold DataFrames or arrays — those belong in
    the parallel slots (``designs``, ``_confounds``, ``_sample_masks``).
    """
    raise NotImplementedError("scaffold")


def resolve_mask(
    mask: nib.Nifti1Image | Path | str,
) -> nib.Nifti1Image:
    """Resolve a mask spec into a Nifti1Image.

    Accepts a Nifti1Image, a path, or a known nltools template string
    (e.g. ``"3mm-MNI152-2009c"``). String templates dispatch to the same
    resolver used by ``BrainData``.
    """
    raise NotImplementedError("scaffold")


def resolve_cache_dir(cache_dir: Path | str | None) -> Path | None:
    """Resolve ``cache_dir`` per the spec's precedence rules.

    Order: explicit arg → ``NLTOOLS_CACHE_DIR`` env var → ``./.nltools_cache``.
    Returns ``None`` when the caller passes ``None`` (signaling tempdir mode).
    The returned path is *not* yet decorated with a ``run_id`` subdir; that
    happens at construction time on the instance.
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
    """Build a fresh ``run_id`` of the form ``{timestamp}_{uuid8}``.

    Timestamp is UTC ``YYYYMMDDTHHMMSS``; the uuid tail is 8 hex chars from
    ``secrets.token_hex(4)``. Lex-sortable, collision-free across processes.
    """
    now = now or datetime.now(UTC)
    return f"{now.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(4)}"


def _slug_kwargs(kwargs: dict[str, Any]) -> str:
    """Turn ``{'fwhm': 6.0}`` into ``'fwhm-6.0'`` for step-dir naming.

    Sorted by key, simple types only. Skips ``None``. Lossy by design — the
    full kwargs are recorded in the bundle attrs / sidecar, not the dirname.
    """
    raise NotImplementedError("scaffold")


def make_step_dirname(
    op: str,
    kwargs: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> str:
    """Name a step subdir: ``{timestamp}_{uuid8}_{op}_{key_kwargs}/``.

    Each call yields a unique name (UUID tail) — same op + same params
    twice produces two subdirs, never overwriting.
    """
    raise NotImplementedError("scaffold")


def is_run_id(name: str) -> bool:
    """True if ``name`` matches the run-id regex (``YYYYMMDDTHHMMSS_########``)."""
    return bool(_RUN_ID_RE.match(name))
