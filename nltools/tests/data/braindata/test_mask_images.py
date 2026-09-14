"""Part B: mask-once dedup in list construction (``io.mask_images``).

``BrainData(list_of_niftis, mask=mask)`` routes through ``_load_from_list`` ->
``_mask_images``. These tests pin the byte-equivalence contract (must reproduce
the functional per-item ``apply_mask`` + ``vstack`` exactly) and the perf
contract (the mask is validated once per call, not once per image).
"""

from unittest import mock

import numpy as np
import nibabel as nib
import nilearn.masking as nm

from nltools.data import BrainData
from nltools.data.braindata import io as bd_io


def _make_mask_and_imgs(n=4, dtype=np.float64):
    shape = (4, 4, 3)
    mask_data = np.zeros(shape, dtype=np.float32)
    mask_data.flat[:10] = 1
    mask = nib.Nifti1Image(mask_data, np.eye(4))
    rng = np.random.RandomState(0)
    imgs = [
        nib.Nifti1Image(rng.randn(*shape).astype(dtype), np.eye(4)) for _ in range(n)
    ]
    return mask, imgs


class TestListConstruction:
    def test_construction_validates_mask_once(self):
        """Constructing from an N-item list validates the mask once, not N times."""
        mask, imgs = _make_mask_and_imgs(n=5)
        with mock.patch("nilearn.masking.load_mask_img", wraps=nm.load_mask_img) as spy:
            BrainData(imgs, mask=mask)
        assert spy.call_count == 1

    def test_list_construction_matches_per_image_apply_mask(self):
        """List construction is byte-identical to per-image ``apply_mask`` + vstack."""
        mask, imgs = _make_mask_and_imgs(n=4)
        expected = np.vstack([nm.apply_mask(im, mask) for im in imgs])
        assert np.array_equal(BrainData(imgs, mask=mask).data, expected)

    def test_list_construction_with_a_4d_item_matches_apply_mask(self):
        """A 4-D item in the list contributes one row per volume, as apply_mask does."""
        mask, imgs = _make_mask_and_imgs(n=2)
        rng = np.random.RandomState(2)
        four_d = nib.Nifti1Image(rng.randn(4, 4, 3, 3), np.eye(4))
        items = [imgs[0], four_d, imgs[1]]
        expected = np.vstack([nm.apply_mask(im, mask) for im in items])
        result = BrainData(items, mask=mask).data
        assert result.shape == (5, 10)
        assert np.array_equal(result, expected)

    def test_mask_images_fallback_matches_fast_path(self):
        """The functional fallback in ``_mask_images`` reproduces the fast path."""
        mask, imgs = _make_mask_and_imgs(n=3)
        fast = bd_io._mask_images_fast(mask, imgs)
        with mock.patch.object(
            bd_io, "_mask_images_fast", side_effect=RuntimeError("forced")
        ):
            fallback = bd_io._mask_images(mask, imgs)
        assert np.array_equal(fast, fallback)
