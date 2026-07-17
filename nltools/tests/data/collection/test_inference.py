"""Group reductions, ttest/anova/perm tests, ISC, alignment."""

from __future__ import annotations

import numpy as np
import pytest

from nltools.data import BrainCollection, BrainData


XFAIL = pytest.mark.xfail(reason="not implemented", strict=True)


class TestReductionShape:
    """SPEC §"Group reductions": collapse subjects → BrainData (or dict)."""

    @pytest.mark.parametrize(
        "method",
        ["mean", "std", "var", "median", "sum", "min", "max"],
    )
    def test_simple_reductions_return_braindata(self, bc_inmem, method):
        out = getattr(bc_inmem, method)()
        assert isinstance(out, BrainData)

    def test_concat_returns_stacked_braindata(self, bc_inmem):
        out = bc_inmem.concat()
        assert isinstance(out, BrainData)
        # n_total_obs == sum of per-subject n_obs
        per_sub_obs = sum(bd.shape[0] for bd in bc_inmem)
        assert out.shape[0] == per_sub_obs

    def test_ttest_returns_dict_with_keys(self, bc_inmem):
        out = bc_inmem.ttest()
        for k in ("mean", "t", "z", "p"):
            assert k in out
            assert isinstance(out[k], BrainData)

    def test_ttest2_signature(self, bc_inmem):
        out = bc_inmem.ttest2(bc_inmem)
        for k in ("mean", "t", "z", "p"):
            assert k in out

    def test_anova_returns_dict(self, bc_inmem):
        out = bc_inmem.anova(np.array([0, 1, 0]))
        assert "f" in out or "F" in out


class TestStreaming:
    """SPEC §"Streaming algorithms": path-backed inputs stream when possible."""

    def test_mean_streams_from_path_backed(self, bc_pathbacked):
        out = bc_pathbacked.mean()
        assert isinstance(out, BrainData)
        # All items still path-backed afterward (not loaded as a side-effect).
        assert not any(bc_pathbacked.is_loaded)

    def test_ttest_streams_from_path_backed(self, bc_pathbacked):
        out = bc_pathbacked.ttest()
        assert "t" in out
        assert not any(bc_pathbacked.is_loaded)


class TestPermutationTest:
    def test_permutation_test_returns_dict(self, bc_inmem):
        out = bc_inmem.permutation_test(n_permute=20, random_state=0)
        assert isinstance(out, dict)

    def test_permutation_test_return_null(self, bc_inmem):
        out = bc_inmem.permutation_test(
            n_permute=10,
            return_null=True,
            random_state=0,
        )
        assert "null" in out or "null_distribution" in out


class TestISC:
    def test_isc_loo_returns_dict(self, bc_inmem):
        out = bc_inmem.isc(method="loo")
        assert isinstance(out, dict)

    def test_isc_test_returns_dict(self, bc_inmem):
        out = bc_inmem.isc_test(method="loo", n_samples=10)
        assert isinstance(out, dict)

    def test_isc_test_synchronized_voxels_are_significant(self, tiny_mask):
        """F066: strongly-synchronized voxels must yield a SMALL p-value.

        The bootstrap null must be centered at 0 (subtract the observed ISC)
        before the p-value comparison, matching the pre-0.6.0 implementation.
        Without centering, the null sits on top of the observed value and every
        voxel gets p~0.5 regardless of synchrony.
        """
        import nibabel as nib

        from nltools.data import BrainCollection, BrainData

        rng = np.random.default_rng(0)
        n_obs = 40
        common = rng.standard_normal((*tiny_mask.shape, n_obs)).astype(np.float32)
        brains = []
        for i in range(6):
            noise = rng.standard_normal(common.shape).astype(np.float32) * 0.05
            img = nib.Nifti1Image(common + noise, tiny_mask.affine)
            brains.append(BrainData(img, mask=tiny_mask))
        bc = BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=None)

        out = bc.isc_test(method="loo", n_samples=200, random_state=0)
        p = np.asarray(out["p"].data).reshape(-1)
        assert p.min() < 0.05, (
            f"synchronized voxels should be significant; min p was {p.min()}"
        )


class TestISCRoiMask:
    """F068: roi_mask must actually restrict the computation, not be ignored."""

    @staticmethod
    def _half_mask(tiny_mask):
        """An ROI covering a strict subset of tiny_mask's 27 voxels."""
        import nibabel as nib

        roi = np.zeros(tiny_mask.shape, dtype=np.int16)
        roi[:2, :2, :2] = 1  # 8 of 27 voxels
        return nib.Nifti1Image(roi, tiny_mask.affine)

    def test_isc_roi_mask_restricts_output_voxels(self, bc_inmem, tiny_mask):
        """Passing roi_mask must shrink the ISC map to the ROI's voxels."""
        roi = self._half_mask(tiny_mask)
        full = bc_inmem.isc(method="loo")
        scoped = bc_inmem.isc(method="loo", roi_mask=roi)

        n_full = np.asarray(full["isc"].data).reshape(-1).size
        n_scoped = np.asarray(scoped["isc"].data).reshape(-1).size
        assert n_full == 27
        assert n_scoped == 8, (
            f"roi_mask must restrict ISC to the ROI; got {n_scoped} voxels "
            f"(whole-brain is {n_full}) — roi_mask was silently ignored"
        )

    def test_isc_roi_mask_matches_premasked_collection(self, bc_inmem, tiny_mask):
        """isc(roi_mask=roi) == isc() on a collection already masked to roi."""
        from nltools.data.braindata.utils import check_brain_data

        roi = self._half_mask(tiny_mask)
        scoped = bc_inmem.isc(method="loo", roi_mask=roi)

        # The ROI must be coerced into the collection's space first: handing a
        # raw Niimg to apply_mask re-homes it onto the default MNI152 mask.
        roi_bd = check_brain_data(roi, mask=tiny_mask)
        premasked = BrainCollection(
            [bd.apply_mask(roi_bd) for bd in bc_inmem],
            mask=roi_bd.to_nifti(),
            lazy=False,
            cache_dir=None,
        )
        expected = premasked.isc(method="loo")

        np.testing.assert_allclose(
            np.asarray(scoped["isc"].data).reshape(-1),
            np.asarray(expected["isc"].data).reshape(-1),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_isc_roi_mask_pairwise_restricts_output(self, bc_inmem, tiny_mask):
        roi = self._half_mask(tiny_mask)
        out = bc_inmem.isc(method="pairwise", roi_mask=roi)
        assert np.asarray(out["isc"].data).reshape(-1).size == 8
        assert out["pairs"].shape[1] == 8

    def test_isc_test_roi_mask_restricts_output_voxels(self, bc_inmem, tiny_mask):
        roi = self._half_mask(tiny_mask)
        out = bc_inmem.isc_test(
            method="loo", roi_mask=roi, n_samples=10, random_state=0
        )
        assert np.asarray(out["isc"].data).reshape(-1).size == 8
        assert np.asarray(out["p"].data).reshape(-1).size == 8
        assert out["null_distribution"].shape[1] == 8

    @pytest.mark.parametrize("kwarg", ["radius_mm", "device", "n_jobs", "progress_bar"])
    def test_removed_never_implemented_kwargs_rejected(self, bc_inmem, kwarg):
        """F068: params that were accepted-but-ignored are gone, not silent."""
        with pytest.raises(TypeError):
            bc_inmem.isc(method="loo", **{kwarg: 1})
        with pytest.raises(TypeError):
            bc_inmem.isc_test(method="loo", n_samples=5, **{kwarg: 1})


class TestAlign:
    """Behavioral facade tests. Searchlight Procrustes needs more samples than
    the tiny fixture provides — algorithm correctness is covered separately
    in ``nltools.algorithms.alignment`` tests.
    """

    @pytest.mark.skip(
        reason="searchlight Procrustes needs >8 timepoints; algorithm "
        "correctness covered in nltools.algorithms tests"
    )
    def test_align_returns_collection(self, bc_inmem):
        out = bc_inmem.align(method="procrustes", spatial_scale="searchlight")
        assert isinstance(out, BrainCollection)

    @pytest.mark.skip(
        reason="searchlight Procrustes needs >8 timepoints; algorithm "
        "correctness covered in nltools.algorithms tests"
    )
    def test_align_with_return_model_returns_tuple(self, bc_inmem):
        out, model = bc_inmem.align(
            method="procrustes",
            spatial_scale="searchlight",
            return_model=True,
        )
        assert isinstance(out, BrainCollection)
        assert model is not None

    def test_align_facade_signature(self):
        import inspect

        sig = inspect.signature(BrainCollection.align)
        assert sig.parameters["return_model"].default is False
        assert sig.parameters["spatial_scale"].default == "searchlight"
        assert sig.parameters["method"].default == "procrustes"
