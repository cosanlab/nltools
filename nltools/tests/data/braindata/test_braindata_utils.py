"""Tests for the BrainData helpers in ``nltools.data.braindata.utils``."""

import numpy as np

from nltools.data import DesignMatrix


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


class TestResolveThreshold:
    """Shared percentile-threshold resolution (#479)."""

    def test_numeric_and_none_pass_through(self):
        import numpy as np

        from nltools.data.braindata.utils import resolve_threshold

        data = np.arange(10.0)
        assert resolve_threshold(2.5, data) == 2.5
        assert resolve_threshold(None, data) is None

    def test_percentile_over_finite_nonzero(self):
        import numpy as np

        from nltools.data.braindata.utils import resolve_threshold

        # Zeros are absence-of-data (masked map) and must not skew the percentile.
        data = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
        expected = float(np.percentile([1.0, 2.0, 3.0, 4.0], 50))
        assert resolve_threshold("50%", data) == expected

    def test_all_zero_data_falls_back(self):
        import numpy as np

        from nltools.data.braindata.utils import resolve_threshold

        assert resolve_threshold("98%", np.zeros(10)) == 0.0

    def test_bad_string_raises(self):
        import numpy as np

        with __import__("pytest").raises(ValueError, match="threshold"):
            from nltools.data.braindata.utils import resolve_threshold

            resolve_threshold("high", np.arange(4.0))
