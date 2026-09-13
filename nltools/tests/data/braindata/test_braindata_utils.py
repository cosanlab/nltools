"""Tests for the BrainData helpers in ``nltools.data.braindata.utils``."""

import numpy as np


class TestResolveThreshold:
    """Shared percentile-threshold resolution (#479)."""

    def test_numeric_and_none_pass_through(self):
        from nltools.data.braindata.utils import _resolve_threshold

        data = np.arange(10.0)
        assert _resolve_threshold(2.5, data) == 2.5
        assert _resolve_threshold(None, data) is None

    def test_percentile_over_finite_nonzero(self):
        from nltools.data.braindata.utils import _resolve_threshold

        # Zeros are absence-of-data (masked map) and must not skew the percentile.
        data = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
        expected = float(np.percentile([1.0, 2.0, 3.0, 4.0], 50))
        assert _resolve_threshold("50%", data) == expected

    def test_all_zero_data_falls_back(self):
        from nltools.data.braindata.utils import _resolve_threshold

        assert _resolve_threshold("98%", np.zeros(10)) == 0.0

    def test_bad_string_raises(self):
        with __import__("pytest").raises(ValueError, match="threshold"):
            from nltools.data.braindata.utils import _resolve_threshold

            _resolve_threshold("high", np.arange(4.0))
