"""Tests for cross-cutting helpers in ``nltools.utils``."""

import warnings

import numpy as np

from nltools.data import DesignMatrix
from nltools.utils import DesignMatrixWarning, find_stack_level


class TestCoalescedGC:
    """``coalesced_gc()`` collapses nilearn's per-copy gc storm into one sweep."""

    def test_wrapped_fit_numerically_identical(self, minimal_brain_data, monkeypatch):
        """A coalesced BrainData.fit matches the un-coalesced (passthrough) fit."""
        design = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.RandomState(0).randn(len(minimal_brain_data)),
            }
        )

        # Coalesced path (default — fit is decorated with @coalesced_gc()).
        coalesced = minimal_brain_data.copy()
        coalesced.fit(model="glm", X=design)

        # Passthrough path (decorator becomes a no-op wrapper).
        monkeypatch.setenv("NLTOOLS_NO_GC_COALESCE", "1")
        passthrough = minimal_brain_data.copy()
        passthrough.fit(model="glm", X=design)

        for attr in ("glm_betas", "glm_residual", "glm_predicted", "glm_r2"):
            a = getattr(coalesced, attr).data
            b = getattr(passthrough, attr).data
            assert np.array_equal(a, b), f"{attr} differs between coalesced/passthrough"


class TestAttemptToImport:
    """``attempt_to_import`` returns the module or ``None`` — no side-registry."""

    def test_returns_module_on_success(self):
        from nltools.utils import attempt_to_import

        mod = attempt_to_import("numpy")
        assert mod is np

    def test_returns_none_on_missing_module(self):
        from nltools.utils import attempt_to_import

        assert attempt_to_import("no_such_module_nltools_test") is None


class TestResolveThreshold:
    """Shared percentile-threshold resolution (#479)."""

    def test_numeric_and_none_pass_through(self):
        import numpy as np

        from nltools.utils import resolve_threshold

        data = np.arange(10.0)
        assert resolve_threshold(2.5, data) == 2.5
        assert resolve_threshold(None, data) is None

    def test_percentile_over_finite_nonzero(self):
        import numpy as np

        from nltools.utils import resolve_threshold

        # Zeros are absence-of-data (masked map) and must not skew the percentile.
        data = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
        expected = float(np.percentile([1.0, 2.0, 3.0, 4.0], 50))
        assert resolve_threshold("50%", data) == expected

    def test_all_zero_data_falls_back(self):
        import numpy as np

        from nltools.utils import resolve_threshold

        assert resolve_threshold("98%", np.zeros(10)) == 0.0

    def test_bad_string_raises(self):
        import numpy as np

        with __import__("pytest").raises(ValueError, match="threshold"):
            from nltools.utils import resolve_threshold

            resolve_threshold("high", np.arange(4.0))


class TestFindStackLevel:
    """`find_stack_level()` attributes a library warning to the user's line.

    A fixed `stacklevel` drifts as facades gain layers, so every
    `warnings.warn` in the library computes the level instead.
    """

    def test_test_files_count_as_outside_the_package(self):
        """nltools/tests/ lives under the package dir but is user code here."""
        # Only find_stack_level's own frame is inside the library -> level 1,
        # so a warn() issued from this file is attributed to this file.
        assert find_stack_level() == 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("probe", UserWarning, stacklevel=find_stack_level())
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

    def test_skips_contextlib_decorator_frames(self):
        """`@coalesced_gc()` facades put a stdlib contextlib frame between the
        user and nltools; the level must step over it too."""
        import nibabel as nib
        import numpy as np

        from nltools.data import BrainData, DesignMatrix
        from nltools.data.braindata.modeling import RankDeficientDesignWarning

        mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.int8), np.eye(4))
        rng = np.random.default_rng(0)
        bd = BrainData(
            nib.Nifti1Image(rng.standard_normal((3, 3, 3, 6)), np.eye(4)), mask=mask
        )
        design = DesignMatrix(np.column_stack([np.arange(6.0), 2 * np.arange(6.0)]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bd.fit(model="glm", X=design)  # BrainData.fit is @coalesced_gc()
        rank = [w for w in caught if w.category is RankDeficientDesignWarning]
        assert rank and rank[0].filename == __file__
