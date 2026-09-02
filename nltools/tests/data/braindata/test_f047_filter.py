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


class TestStandardizeIsNotABool:
    """nilearn 0.15 drops boolean ``standardize``; never hand it one."""

    def test_filter_default_emits_no_future_warning(self, minimal_brain_data):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            minimal_brain_data.filter(sampling_freq=2.0, high_pass=0.01)

    def test_filter_maps_true_to_zscore_sample(self, minimal_brain_data):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = minimal_brain_data.filter(
                sampling_freq=2.0, high_pass=0.01, standardize=True
            )
        np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-6)

    def test_filter_false_means_off(self, minimal_brain_data):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = minimal_brain_data.filter(
                sampling_freq=2.0, high_pass=0.01, standardize=False
            )
        default = minimal_brain_data.filter(sampling_freq=2.0, high_pass=0.01)
        np.testing.assert_array_equal(out.data, default.data)

    def test_extract_roi_labels_emits_no_future_warning(self):
        import warnings

        import nibabel as nib

        from nltools.data import BrainData

        shape, affine = (6, 6, 6), np.eye(4)
        mask = nib.Nifti1Image(np.ones(shape, dtype=np.int8), affine)
        labels = np.zeros(shape)  # 0 = background, as nilearn requires
        labels[:2] = 1
        labels[2:4] = 2
        atlas = BrainData(nib.Nifti1Image(labels, affine), mask=mask)
        rng = np.random.default_rng(0)
        brain = BrainData(
            nib.Nifti1Image(rng.standard_normal(shape + (4,)), affine), mask=mask
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = brain.extract_roi(atlas, method="mean")
        assert out.shape == (2, 4)
