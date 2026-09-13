"""Cross-cutting utilities used across the nltools package."""

__all__ = ["DesignMatrixWarning", "ResamplingWarning"]

import contextlib
import inspect
from os.path import dirname, join, sep as pathsep

import polars as pl


# ---------------------------------------------------------------------------
# polars compatibility
# ---------------------------------------------------------------------------

# polars 1.42.1 renamed the classic equal-height horizontal concat to
# ``horizontal_extend`` and deprecated the old ``horizontal`` spelling, which
# will start padding to the tallest frame in the next breaking release. Below
# 1.42.1 only ``horizontal`` exists. Every nltools call site concatenates
# frames of equal height, so the two names are interchangeable there, and
# picking by version lets the package run on the polars 1.33.1 that Pyodide
# bundles as well as on current releases.
_POLARS_VERSION = tuple(int(part) for part in pl.__version__.split(".")[:3])
_HORIZONTAL_CONCAT = (
    "horizontal_extend" if _POLARS_VERSION >= (1, 42, 1) else "horizontal"
)


# ---------------------------------------------------------------------------
# Warnings: attribution and library-wide categories
# ---------------------------------------------------------------------------

_PACKAGE_DIR = dirname(__file__) + pathsep
_TESTS_DIR = join(_PACKAGE_DIR, "tests") + pathsep
# ``@_coalesced_gc()`` (a contextmanager used as a decorator) wraps facade
# methods in a stdlib contextlib frame; it is nltools plumbing, not user code.
_PLUMBING_FILES = frozenset({contextlib.__file__})


def _is_library_frame(filename: str) -> bool:
    if filename in _PLUMBING_FILES:
        return True
    return filename.startswith(_PACKAGE_DIR) and not filename.startswith(_TESTS_DIR)


def _find_stack_level() -> int:
    """Return the ``stacklevel`` that attributes a warning to the caller's code.

    Walks up from the caller until the first frame outside the nltools package
    (``nltools/tests/`` counts as outside: tests are the library's users; the
    stdlib ``contextlib`` frame that ``@_coalesced_gc()`` inserts counts as
    inside), so a ``warnings.warn`` deep inside a facade lands on the user's
    line rather than on nltools internals — the same pattern nilearn and pandas
    use. Every ``warnings.warn`` in the library passes
    ``stacklevel=_find_stack_level()``; a source-scan test enforces it.

    Returns:
        int: Value for the ``stacklevel`` argument of ``warnings.warn``.

    Examples:
        ```python
        warnings.warn("message", UserWarning, stacklevel=_find_stack_level())
        ```
    """
    frame = inspect.currentframe()
    level = 0
    try:
        while frame is not None and _is_library_frame(inspect.getfile(frame)):
            frame = frame.f_back
            level += 1
    finally:
        del frame
    return level


class ResamplingWarning(UserWarning):
    """Data is (or will be) resampled to a different space than it arrived in.

    Raised when a data image does not match the mask/template it is loaded
    against — a detected template at another resolution, or a mask in a
    different space — and nltools resamples to reconcile them. Subclasses
    ``UserWarning`` so it participates in default filtering while staying
    individually silenceable:
    ``warnings.filterwarnings("ignore", category=ResamplingWarning)``.
    """


class DesignMatrixWarning(UserWarning):
    """A ``DesignMatrix`` operation was skipped, or the design itself is suspect.

    Raised by regressor builders (``add_poly``, ``add_dct_basis``,
    ``convolve``) when the requested columns already exist and are skipped,
    and by ``BrainData.fit()`` when the design it receives is rank deficient.
    Subclasses ``UserWarning`` so it participates in default filtering while
    staying individually silenceable:
    ``warnings.filterwarnings("ignore", category=DesignMatrixWarning)``.
    """


# ---------------------------------------------------------------------------
# Cross-cutting helpers (used by multiple subsystems)
# ---------------------------------------------------------------------------


def _attempt_to_import(dependency, fromlist=None):
    """Attempt to import an optional dependency, returning None if unavailable.

    This function is used to handle optional dependencies gracefully. If the
    import fails, the function returns None rather than raising an error,
    allowing the calling code to check and handle missing dependencies.

    Args:
        dependency (str): The module name to import (e.g. `'torch'`, `'cupy'`).
        fromlist (list[str], optional): Names to import from the module (passed to
            `__import__`).

    Returns:
        ModuleType | None: The imported module, or None if the import failed.

    Examples:
        ```python
        torch = _attempt_to_import("torch")
        if torch is not None:
            ...  # use torch
        ```
    """
    try:
        mod = __import__(dependency, fromlist=fromlist)
    except ImportError:
        mod = None
    return mod


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

    def __enter__(self) -> "_NullProgressBar":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def _maybe_tqdm(iterable, *, progress_bar: bool, **tqdm_kwargs):
    """Wrap `iterable` in a tqdm progress bar only when `progress_bar` is True.

    tqdm writes to stderr, so an unconditional bar makes functions noisy when
    called in a loop (a 100-iteration calibration study would emit 100 bars).
    Importing tqdm lazily also keeps it off the import path when unused. Uses
    `tqdm.auto`, so notebooks get widget bars and terminals get text bars.

    Args:
        iterable (Iterable): The iterable to wrap.
        progress_bar (bool): Whether to display a progress bar.
        **tqdm_kwargs (dict): Forwarded to `tqdm` (e.g. `desc`, `unit`, `total`).

    Returns:
        Iterable: The original iterable, or a `tqdm`-wrapped version of it.

    Examples:
        ```python
        for i in _maybe_tqdm(range(n_permute), progress_bar=progress_bar,
                            desc="CPU parallel perms", unit="perm"):
            ...
        ```
    """
    if not progress_bar:
        return iterable

    from tqdm.auto import tqdm

    return tqdm(iterable, **tqdm_kwargs)


def _make_progress_bar(*, progress_bar: bool, **tqdm_kwargs):
    """Build a progress bar, or a no-op stand-in when `progress_bar` is False.

    Use this for call sites that drive the bar manually via `.update()` rather
    than by iteration. Uses `tqdm.auto`, so notebooks get widget bars and
    terminals get text bars.

    Args:
        progress_bar (bool): Whether to display a progress bar.
        **tqdm_kwargs (dict): Forwarded to `tqdm` (e.g. `total`, `desc`, `unit`).

    Returns:
        tqdm | _NullProgressBar: A `tqdm` instance, or a `_NullProgressBar` exposing
            the same subset of its interface (`update`, `close`, and the
            context-manager protocol).
    """
    if not progress_bar:
        return _NullProgressBar()

    from tqdm.auto import tqdm

    return tqdm(**tqdm_kwargs)
