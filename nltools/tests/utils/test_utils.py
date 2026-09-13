"""Tests for cross-cutting helpers in ``nltools.utils``."""

import warnings

import numpy as np

from nltools.utils import DesignMatrixWarning, _find_stack_level


class TestAttemptToImport:
    """``_attempt_to_import`` returns the module or ``None`` — no side-registry."""

    def test_returns_module_on_success(self):
        from nltools.utils import _attempt_to_import

        mod = _attempt_to_import("numpy")
        assert mod is np

    def test_returns_none_on_missing_module(self):
        from nltools.utils import _attempt_to_import

        assert _attempt_to_import("no_such_module_nltools_test") is None


class TestFindStackLevel:
    """`_find_stack_level()` attributes a library warning to the user's line.

    A fixed `stacklevel` drifts as facades gain layers, so every
    `warnings.warn` in the library computes the level instead.
    """

    def test_test_files_count_as_outside_the_package(self):
        """nltools/tests/ lives under the package dir but is user code here."""
        # Only _find_stack_level's own frame is inside the library -> level 1,
        # so a warn() issued from this file is attributed to this file.
        assert _find_stack_level() == 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("probe", UserWarning, stacklevel=_find_stack_level())
        assert caught[0].filename == __file__

    def test_walks_past_library_frames(self):
        """A warning raised inside the library is attributed to the caller."""
        from nltools.data import DesignMatrix

        dm = DesignMatrix({"a": [1.0, 2.0, 3.0, 4.0]}, TR=1.0).add_poly(order=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dm.add_poly(order=0)  # "already has 0th order polynomial" notice
        assert caught, "expected the already-exists notice"
        assert caught[0].filename == __file__
        assert caught[0].category is DesignMatrixWarning
