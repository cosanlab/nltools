"""Cross-cutting utilities used across the nltools package."""

__all__ = [
    "all_same",
    "attempt_to_import",
    "coalesced_gc",
    "concatenate",
    "get_resource_path",
    "make_progress_bar",
    "maybe_tqdm",
]

import collections
import gc
import os
from contextlib import contextmanager
from os.path import dirname, join, sep as pathsep

import numpy as np


# ---------------------------------------------------------------------------
# Cross-cutting helpers (used by multiple subsystems)
# ---------------------------------------------------------------------------


@contextmanager
def coalesced_gc():
    """Collapse nilearn's forced per-copy ``gc.collect()`` calls into ONE per operation.

    nilearn calls ``gc.collect()`` after every masked-array copy
    (``_utils/niimg.py:safe_get_data``); a masking-heavy op — a GLM fit that
    re-validates the same mask and builds several result maps — fires dozens.
    With torch/nilearn/sklearn resident each sweep costs ~0.1s, so the storm
    dominates the wall-clock of otherwise-trivial numerical work.

    This no-ops the interim collects and runs a single real collect on exit,
    so peak memory stays bounded to one operation's worth of cyclic garbage
    (the ``gc.collect()`` nilearn calls is a peak-memory optimization, not a
    correctness requirement — suppressing it only defers reclamation). Opt out
    with ``NLTOOLS_NO_GC_COALESCE=1``.

    Because ``@contextmanager`` results double as decorators, this can also be
    used as ``@coalesced_gc()`` on an operation-boundary method.

    Nesting is safe: each frame restores whatever it saved, so only the
    outermost frame restores the real ``gc.collect`` and runs the final sweep;
    inner frames' exit-time collect is a no-op.

    Caveat: this swaps a process-global builtin. It is safe under the default
    loky (process) worker backend — each worker has its own ``gc``. Under a
    *threading* backend there is a brief window where a concurrent thread sees
    the no-op collect; ``NLTOOLS_NO_GC_COALESCE=1`` is the escape hatch there.
    """
    if os.environ.get("NLTOOLS_NO_GC_COALESCE"):
        yield
        return
    saved = gc.collect  # may already be the no-op if we're nested
    gc.collect = lambda *a, **k: 0
    try:
        yield
    finally:
        gc.collect = saved  # only the outermost frame restores the real collect
        gc.collect()  # no-op if still nested; one real sweep at the top


def get_resource_path():
    """Get path to nltools resource directory."""
    return join(dirname(__file__), "resources") + pathsep


module_names = {}
Dependency = collections.namedtuple("Dependency", "package value")


def attempt_to_import(dependency, name=None, fromlist=None):
    """Attempt to import an optional dependency, returning None if unavailable.

    This function is used to handle optional dependencies gracefully. If the
    import fails, the function returns None rather than raising an error,
    allowing the calling code to check and handle missing dependencies.

    Args:
        dependency: The module name to import (e.g., 'torch', 'cupy').
        name: Optional name to store the dependency under in module_names.
            Defaults to the dependency name.
        fromlist: Optional list of names to import from the module.

    Returns:
        The imported module, or None if the import failed.

    Examples:
        >>> torch = attempt_to_import('torch')
        >>> if torch is not None:
        ...     # Use torch
        ...     pass
    """
    if name is None:
        name = dependency
    try:
        mod = __import__(dependency, fromlist=fromlist)
    except ImportError:
        mod = None
    module_names[name] = Dependency(dependency, mod)
    return mod


def all_same(items):
    """Check if all items in a sequence are equal to the first item.

    Args:
        items: A sequence of items to compare.

    Returns:
        bool: True if all items equal the first item, False otherwise.

    Examples:
        >>> all_same([1, 1, 1])
        True
        >>> all_same([1, 2, 1])
        False
    """
    return all(np.array_equal(x, items[0]) for x in items)


def concatenate(data):
    """Concatenate a list of BrainData() or Adjacency() objects."""

    if not isinstance(data, list):
        raise ValueError("Make sure you are passing a list of objects.")

    if all(isinstance(x, data[0].__class__) for x in data):
        out = data[0].__class__()
        for i in data:
            out = out.append(i)
    else:
        raise ValueError("Make sure all objects in the list are the same type.")
    return out


# ---------------------------------------------------------------------------
# Progress bars — the single library-wide mechanism
# ---------------------------------------------------------------------------


class _NullProgressBar:
    """No-op stand-in for `tqdm` used when `progress_bar=False`.

    Supports the subset of the tqdm interface nltools relies on, so call sites
    that drive a bar manually need no branching.
    """

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_postfix(
        self, *args, **kwargs
    ) -> None:  # nosemgrep: kwargs-internal-forwarding  # mirrors tqdm.set_postfix
        pass

    def set_description(
        self, *args, **kwargs
    ) -> None:  # nosemgrep: kwargs-internal-forwarding  # mirrors tqdm.set_description
        pass

    def __enter__(self) -> "_NullProgressBar":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def maybe_tqdm(iterable, *, progress_bar: bool, **tqdm_kwargs):
    """Wrap `iterable` in a tqdm progress bar only when `progress_bar` is True.

    tqdm writes to stderr, so an unconditional bar makes functions noisy when
    called in a loop (a 100-iteration calibration study would emit 100 bars).
    Importing tqdm lazily also keeps it off the import path when unused. Uses
    `tqdm.auto`, so notebooks get widget bars and terminals get text bars.

    Args:
        iterable: The iterable to wrap.
        progress_bar: Whether to display a progress bar.
        **tqdm_kwargs: Forwarded to `tqdm` (e.g. `desc`, `unit`, `total`).

    Returns:
        The original iterable, or a `tqdm`-wrapped version of it.

    Examples:
        ```python
        for i in maybe_tqdm(range(n_permute), progress_bar=progress_bar,
                            desc="CPU parallel perms", unit="perm"):
            ...
        ```
    """
    if not progress_bar:
        return iterable

    from tqdm.auto import tqdm

    return tqdm(iterable, **tqdm_kwargs)


def make_progress_bar(*, progress_bar: bool, **tqdm_kwargs):
    """Build a progress bar, or a no-op stand-in when `progress_bar` is False.

    Use this for call sites that drive the bar manually via `.update()` rather
    than by iteration. Uses `tqdm.auto`, so notebooks get widget bars and
    terminals get text bars.

    Args:
        progress_bar: Whether to display a progress bar.
        **tqdm_kwargs: Forwarded to `tqdm` (e.g. `total`, `desc`, `unit`).

    Returns:
        A `tqdm` instance, or a `_NullProgressBar` exposing the same subset of
        its interface (`update`, `close`, `set_postfix`, `set_description`, and
        the context-manager protocol).
    """
    if not progress_bar:
        return _NullProgressBar()

    from tqdm.auto import tqdm

    return tqdm(**tqdm_kwargs)
