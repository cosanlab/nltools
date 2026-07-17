"""Regression test for F047: filter_data double-passes detrend/standardize.

`filter_data` read ``detrend``/``standardize`` with ``kwargs.get()`` (leaving
them in ``kwargs``) then forwarded them to ``nilearn.signal.clean`` both
explicitly and again via ``**kwargs`` -> ``TypeError: got multiple values``.
The documented usage (pass ``detrend=True`` via kwargs) therefore crashed.
"""

import numpy as np


class TestFilterDetrendStandardize:
    def test_filter_with_detrend_via_kwargs(self, minimal_brain_data):
        """The documented `filter(..., detrend=True)` usage must not crash."""
        out = minimal_brain_data.filter(sampling_freq=2.0, high_pass=0.01, detrend=True)
        assert out.data.shape == minimal_brain_data.data.shape

    def test_filter_with_standardize_via_kwargs(self, minimal_brain_data):
        """`standardize` passed via kwargs must reach clean() exactly once."""
        out = minimal_brain_data.filter(
            sampling_freq=2.0, high_pass=0.01, standardize="zscore_sample"
        )
        # Standardized output should be roughly zero-mean per voxel.
        np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-6)

    def test_filter_default_no_detrend(self, minimal_brain_data):
        """Sanity: default path (no detrend/standardize) still works."""
        out = minimal_brain_data.filter(sampling_freq=2.0, high_pass=0.01)
        assert out.data.shape == minimal_brain_data.data.shape
