"""Provide cross-cutting utilities for nltools.

Cross-cutting utilities used across the nltools package.

"""

__all__ = [
    "all_same",
    "attempt_to_import",
    "coalesced_gc",
    "concatenate",
    "get_resource_path",
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
    return np.all(x == items[0] for x in items)


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
