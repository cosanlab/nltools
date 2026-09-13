"""Part B: mask-once dedup in list construction (``io.mask_images``).

The GLM builds result maps as ``BrainData(list_of_niftis, mask=bd.mask)``,
routing through ``_load_from_list`` -> ``_mask_images``. These tests pin the
byte-equivalence contract (must reproduce the functional per-item
``apply_mask`` + ``vstack`` exactly) and the perf contract (the mask is
validated once per call, not once per image).
"""

from unittest import mock

import numpy as np
import nibabel as nib
import nilearn.masking as nm

from nltools.data import BrainData, DesignMatrix
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


class TestGLMFitMapsByteIdentical:
    def test_glm_maps_identical_across_mask_paths(
        self, minimal_brain_data, monkeypatch
    ):
        """The attached GLM maps are byte-identical fast-path vs functional-path."""
        design = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.RandomState(1).randn(len(minimal_brain_data)),
            }
        )

        # Fast path (default _mask_images dedup).
        fast = minimal_brain_data.copy()
        fast.fit(model="glm", X=design)

        # Reference: force list masking through the functional per-item path.
        def functional_mask_images(mask, imgs):
            return np.vstack([nm.apply_mask(im, mask) for im in imgs])

        monkeypatch.setattr(bd_io, "_mask_images", functional_mask_images)
        ref = minimal_brain_data.copy()
        ref.fit(model="glm", X=design)

        for attr in ("glm_betas", "glm_residual", "glm_predicted", "glm_r2"):
            assert np.array_equal(getattr(fast, attr).data, getattr(ref, attr).data), (
                f"{attr} differs between fast and functional mask paths"
            )
