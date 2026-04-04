"""
NeuroLearn Utilities
====================

Cross-cutting utilities used across the nltools package.

"""

__all__ = [
    "get_resource_path",
    "attempt_to_import",
    "all_same",
    "concatenate",
]

import collections
from os.path import dirname, join, sep as pathsep

import numpy as np


# ---------------------------------------------------------------------------
# Cross-cutting helpers (used by multiple subsystems)
# ---------------------------------------------------------------------------


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
    """Concatenate a list of BrainData() or Adjacency() objects"""

    if not isinstance(data, list):
        raise ValueError("Make sure you are passing a list of objects.")

    if all([isinstance(x, data[0].__class__) for x in data]):
        out = data[0].__class__()
        for i in data:
            out = out.append(i)
    else:
        raise ValueError("Make sure all objects in the list are the same type.")
    return out
