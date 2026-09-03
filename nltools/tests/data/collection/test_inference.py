"""Group reductions, ttest/anova/perm tests, ISC, alignment."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from nltools.algorithms.backends import check_gpu_available
from nltools.data import BrainCollection, BrainData


class TestReductionShape:
    """Group reductions: collapse subjects → BrainData (or dict)."""

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

    def test_ttest_tail_one_halves_positive_p(self, bc_inmem):
        two = bc_inmem.ttest(tail=2)
        one = bc_inmem.ttest(tail="one")
        t_arr = np.asarray(two["t"].data)
        pos = t_arr > 0
        np.testing.assert_allclose(
            np.asarray(one["p"].data)[pos], np.asarray(two["p"].data)[pos] / 2
        )

    def test_ttest_one_tailed_wrong_direction_z_is_finite(self, tiny_mask):
        """tail=1 on strongly negative data must not leak z = -inf (F3).

        ``t.sf(t, df)`` saturates to exactly 1.0 for large negative t; the
        clip must be symmetric so the z map stays finite in both directions.
        """
        rng = np.random.default_rng(0)
        # df must be high enough that t.sf(t, df) saturates to exactly 1.0
        # (heavy small-df t tails never quite reach it).
        brains = [
            BrainData(
                (rng.standard_normal((1, 27)) - 20.0).astype(np.float32),
                mask=tiny_mask,
            )
            for _ in range(31)
        ]
        bc = BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=None)
        out = bc.ttest(tail=1)
        z = np.asarray(out["z"].data)
        assert np.all(np.isfinite(z))
        assert np.all(z < 0)

    def test_ttest2_one_tailed_wrong_direction_z_is_finite(self, tiny_mask):
        """ttest2 tail=1 with bc << other must return finite z everywhere."""
        rng = np.random.default_rng(1)

        def _bc(shift):
            brains = [
                BrainData(
                    (rng.standard_normal((1, 27)) + shift).astype(np.float32),
                    mask=tiny_mask,
                )
                for _ in range(31)
            ]
            return BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=None)

        out = _bc(-20.0).ttest2(_bc(20.0), tail=1)
        z = np.asarray(out["z"].data)
        assert np.all(np.isfinite(z))
        assert np.all(z < 0)

    def test_ttest_z_matches_braindata_ttest_z(self, tiny_mask):
        """Collection ttest z must match stacking + BrainData.ttest (shared helper)."""
        rng = np.random.default_rng(3)
        maps = [rng.standard_normal((1, 27)).astype(np.float32) for _ in range(6)]
        bc = BrainCollection(
            [BrainData(m, mask=tiny_mask) for m in maps],
            mask=tiny_mask,
            lazy=False,
            cache_dir=None,
        )
        bd_stack = BrainData(np.concatenate(maps, axis=0), mask=tiny_mask)
        for tail in (2, 1):
            coll = bc.ttest(tail=tail)
            ref = bd_stack.ttest(tail=tail)
            np.testing.assert_allclose(
                np.asarray(coll["z"].data).reshape(-1),
                np.asarray(ref["z"].data).reshape(-1),
                rtol=1e-5,
                atol=1e-8,
            )

    def test_permutation_test_accepts_string_tail(self, bc_inmem):
        out = bc_inmem.permutation_test(n_permute=25, tail="one", random_state=0)
        assert 0 < float(np.asarray(out["p"].data).min()) <= 1
        with pytest.raises(ValueError, match="tail"):
            bc_inmem.permutation_test(n_permute=25, tail="upper")

    def test_isc_test_exposes_tail(self, bc_inmem):
        import inspect

        params = inspect.signature(type(bc_inmem).isc_test).parameters
        assert params["tail"].default == 2

    def test_anova_returns_dict(self, bc_inmem):
        out = bc_inmem.anova(np.array([0, 1, 0]))
        assert "f" in out or "F" in out


class TestStreaming:
    """Streaming: path-backed inputs stream when possible."""

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
        assert "null" in out or "null_dist" in out


class TestISC:
    def test_isc_loo_returns_dict(self, bc_inmem):
        out = bc_inmem.isc(method="loo")
        assert isinstance(out, dict)

    def test_isc_loo_streams_without_materializing_all_subjects(
        self, tiny_mask, tmp_path
    ):
        """loo ISC must stream (two-pass Welford), not stack all N subjects.

        The whole point of `BrainCollection` is that whole-brain reductions never
        hold every subject in RAM at once — `mean()`/`ttest()` stream one subject
        at a time. loo ISC is expressible the same way (pass 1: accumulate the
        sum; pass 2: template = (sum - subject)/(n-1), correlate), so peak
        concurrent subject arrays must stay ~1 regardless of N. The pre-fix
        implementation did `np.stack(list(_iter_arrays(...)))`, forcing all N
        resident — this instruments `_iter_arrays` to catch that.
        """
        import weakref

        from nltools.data import BrainCollection
        from nltools.data.collection import inference

        # A 6-subject path-backed collection so materialization (max_live=6) is
        # unambiguously distinct from streaming (max_live<=2).
        paths = []
        for i in range(6):
            rng = np.random.default_rng(i)
            vol = rng.standard_normal(tiny_mask.shape + (8,)).astype(np.float32)
            p = tmp_path / f"iscsub-{i:02d}.nii.gz"
            nib.save(nib.Nifti1Image(vol, tiny_mask.affine), p)
            paths.append(str(p))
        bc = BrainCollection(
            paths, mask=tiny_mask, lazy=True, cache_dir=str(tmp_path / ".c")
        )

        orig = inference._iter_arrays
        live = 0
        max_live = 0

        def _tracking(collection, roi=None):
            nonlocal live, max_live
            for arr in orig(collection, roi):
                live += 1
                max_live = max(max_live, live)

                def _dec(*_):
                    nonlocal live
                    live -= 1

                weakref.finalize(arr, _dec)
                yield arr

        inference._iter_arrays = _tracking
        try:
            bc.isc(method="loo", summary="median")
        finally:
            inference._iter_arrays = orig

        assert max_live <= 2, (
            f"isc(loo) held {max_live} subject arrays at once (N=6); it must "
            "stream (<=2). It is materializing all subjects — the streaming "
            "rewrite is not wired up."
        )

    def test_isc_loo_streaming_matches_reference(self, tiny_mask, tmp_path):
        """Streaming loo ISC must be numerically identical to the direct stack.

        Guards the refactor: the two-pass streaming result must match a
        straightforward materialize-then-compute reference to float tolerance.
        """
        from nltools.data import BrainCollection

        paths, arrays = [], []
        for i in range(5):
            rng = np.random.default_rng(100 + i)
            vol = rng.standard_normal(tiny_mask.shape + (12,)).astype(np.float32)
            p = tmp_path / f"refsub-{i:02d}.nii.gz"
            nib.save(nib.Nifti1Image(vol, tiny_mask.affine), p)
            paths.append(str(p))
            # In-mask (T, V) view matching what the collection yields.
            m = np.asarray(tiny_mask.dataobj) > 0
            arrays.append(vol[m].T.astype(np.float64))  # (12, n_vox)

        bc = BrainCollection(
            paths, mask=tiny_mask, lazy=True, cache_dir=str(tmp_path / ".c")
        )
        got = np.asarray(bc.isc(method="loo", summary="median")["isc"].data).reshape(-1)

        # Reference: leave-one-out template, per-voxel Pearson, median across subj.
        data = np.stack(arrays, axis=0)  # (n_subj, T, V)
        n = data.shape[0]
        total = data.sum(axis=0)
        corrs = np.empty((n, data.shape[2]))
        for i in range(n):
            tmpl = (total - data[i]) / (n - 1)
            a = data[i] - data[i].mean(0)
            b = tmpl - tmpl.mean(0)
            den = np.sqrt((a**2).sum(0)) * np.sqrt((b**2).sum(0))
            out = np.zeros(data.shape[2])
            np.divide((a * b).sum(0), den, out=out, where=den > 0)
            corrs[i] = out
        ref = np.median(corrs, axis=0)

        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)

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
        assert out["null_dist"].shape[1] == 8

    @pytest.mark.parametrize("kwarg", ["radius_mm", "device", "n_jobs", "progress_bar"])
    def test_removed_never_implemented_kwargs_rejected(self, bc_inmem, kwarg):
        """F068: params that were accepted-but-ignored are gone, not silent."""
        with pytest.raises(TypeError):
            bc_inmem.isc(method="loo", **{kwarg: 1})
        with pytest.raises(TypeError):
            bc_inmem.isc_test(method="loo", n_samples=5, **{kwarg: 1})


class TestAlign:
    """Behavioral facade tests for ``BrainCollection.align``.

    The facade feeds each subject to ``LocalAlignment`` as ``(n_voxels,
    n_samples)`` (transposing ``BrainData``'s ``(n_samples, n_voxels)``) and
    transposes the aligned result back; see ``nltools.algorithms.alignment``
    for the underlying-algorithm correctness tests.
    """

    def test_align_returns_collection(self, bc_inmem):
        # bc_inmem is 8 samples x 27 voxels: n_samples < n_voxels, which is the
        # exact case that crashed before the facade transposed its input.
        out = bc_inmem.align(method="procrustes", spatial_scale="searchlight", n_jobs=1)
        assert isinstance(out, BrainCollection)
        # Orientation preserved end-to-end: aligned items keep (n_samples, n_voxels).
        assert all(np.asarray(b.data).shape == (8, 27) for b in out)

    def test_align_with_return_model_returns_tuple(self, bc_inmem):
        out, model = bc_inmem.align(
            method="procrustes",
            spatial_scale="searchlight",
            return_model=True,
            n_jobs=1,
        )
        assert isinstance(out, BrainCollection)
        assert model is not None

    def test_align_identical_subjects_align_to_each_other(
        self, tiny_mask, tiny_brain_factory
    ):
        # Identical subjects must map to identical aligned outputs (symmetry);
        # a transposed feed would scramble which axis is aligned and break this.
        bd = tiny_brain_factory(n_obs=12, seed=7)
        bc = BrainCollection([bd, bd, bd], mask=tiny_mask, lazy=False, cache_dir=None)
        out = list(bc.align(spatial_scale="searchlight", radius_mm=10.0, n_jobs=1))
        first = np.asarray(out[0].data)
        for b in out[1:]:
            np.testing.assert_allclose(np.asarray(b.data), first, atol=1e-5)

    def test_align_roi_scale_works(self, bc_inmem, tiny_mask):
        # Regression: spatial_scale='roi' must flow through to LocalAlignment as
        # the 'roi' scale. The facade previously passed it straight through under
        # the old scheme='searchlight'|'piecewise' vocab, so the canonical 'roi'
        # raised "Unknown ...". Now roi/whole_brain vocab is honored end-to-end.
        parc = np.zeros(tiny_mask.shape, dtype=np.int16)
        parc[:2] = 1  # 18 voxels
        parc[2:] = 2  # 9 voxels
        roi_mask = nib.Nifti1Image(parc, tiny_mask.affine)
        out = bc_inmem.align(spatial_scale="roi", roi_mask=roi_mask, n_jobs=1)
        assert isinstance(out, BrainCollection)
        assert all(np.asarray(b.data).shape == (8, 27) for b in out)

    def test_align_whole_brain_not_implemented(self, bc_inmem):
        # Parity with BrainData.align: each facade validates the spatial_scale
        # vocab up front and raises a clear NotImplementedError for the one scale
        # it can't serve. Collection align is local-only (searchlight/roi).
        with pytest.raises(NotImplementedError, match="whole_brain"):
            bc_inmem.align(spatial_scale="whole_brain", n_jobs=1)

    def test_align_invalid_spatial_scale_raises(self, bc_inmem):
        with pytest.raises(ValueError, match="spatial_scale"):
            bc_inmem.align(spatial_scale="bogus", n_jobs=1)

    # --- cache= (F073): joint-op output persistence, uniform with other ops ---

    def test_align_cache_true_writes_through(self, bc_inmem):
        out = bc_inmem.align(spatial_scale="searchlight", cache=True, n_jobs=1)
        assert not any(out.is_loaded)  # path-backed

    def test_align_cache_false_stays_in_memory(self, bc_inmem):
        out = bc_inmem.align(spatial_scale="searchlight", cache=False, n_jobs=1)
        assert all(out.is_loaded)

    def test_align_cache_auto_follows_inmem_source(self, bc_inmem):
        out = bc_inmem.align(spatial_scale="searchlight", cache="auto", n_jobs=1)
        assert all(out.is_loaded)

    def test_align_cache_auto_follows_pathbacked_source(self, bc_pathbacked):
        out = bc_pathbacked.align(spatial_scale="searchlight", cache="auto", n_jobs=1)
        assert not any(out.is_loaded)

    def test_align_cache_roundtrip_is_lossless(self, bc_inmem):
        # Caching is a transparent memory/perf optimization: a path-backed result
        # must equal the in-memory one to float32 precision, not a quantized approx.
        cached = bc_inmem.align(spatial_scale="searchlight", cache=True, n_jobs=1)
        inmem = bc_inmem.align(spatial_scale="searchlight", cache=False, n_jobs=1)
        for a, b in zip(cached, inmem):
            np.testing.assert_allclose(
                np.asarray(a.data), np.asarray(b.data), rtol=0, atol=1e-6
            )

    def test_align_facade_signature(self):
        import inspect

        sig = inspect.signature(BrainCollection.align)
        assert sig.parameters["return_model"].default is False
        assert sig.parameters["spatial_scale"].default == "searchlight"
        assert sig.parameters["method"].default == "procrustes"


class TestPermutationEngineDelegation:
    """permutation_test/permutation_test2 route through the inference engine.

    The hand-rolled serial loops are gone: device= and n_jobs= are real, and
    results are identical to calling the engine on the stacked subject data.
    """

    def _stacked(self, bc):
        arrs = np.stack([np.asarray(bd.data) for bd in bc], axis=0)
        return arrs.reshape(len(bc), -1)

    def test_one_sample_matches_engine(self, bc_inmem):
        from nltools.algorithms import one_sample_permutation_test

        out = bc_inmem.permutation_test(n_permute=60, random_state=7, return_null=True)
        eng = one_sample_permutation_test(
            self._stacked(bc_inmem),
            n_permute=60,
            random_state=7,
            return_null=True,
            device="cpu",
        )
        np.testing.assert_allclose(
            np.asarray(out["p"].data).ravel(), np.asarray(eng["p"]).ravel()
        )
        np.testing.assert_allclose(
            np.asarray(out["mean"].data).ravel(), np.asarray(eng["mean"]).ravel()
        )
        np.testing.assert_allclose(out["null_dist"].reshape(60, -1), eng["null_dist"])

    def test_two_sample_matches_engine(self, bc_inmem, bc_pathbacked):
        from nltools.algorithms import two_sample_permutation_test

        out = bc_inmem.permutation_test2(
            bc_pathbacked, n_permute=60, random_state=3, return_null=True
        )
        eng = two_sample_permutation_test(
            self._stacked(bc_inmem),
            self._stacked(bc_pathbacked),
            n_permute=60,
            random_state=3,
            return_null=True,
            device="cpu",
        )
        np.testing.assert_allclose(
            np.asarray(out["p"].data).ravel(), np.asarray(eng["p"]).ravel()
        )
        np.testing.assert_allclose(
            np.asarray(out["mean"].data).ravel(), np.asarray(eng["mean_diff"]).ravel()
        )
        np.testing.assert_allclose(out["null_dist"].reshape(60, -1), eng["null_dist"])

    @pytest.mark.gpu
    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_device_gpu_is_real(self, bc_inmem):
        """device='gpu' actually dispatches to the engine's GPU path."""
        pytest.importorskip("torch")
        from nltools.algorithms import one_sample_permutation_test

        out = bc_inmem.permutation_test(
            n_permute=30, random_state=1, device="gpu", return_null=True
        )
        eng = one_sample_permutation_test(
            self._stacked(bc_inmem),
            n_permute=30,
            random_state=1,
            device="gpu",
            return_null=True,
        )
        np.testing.assert_allclose(
            out["null_dist"].reshape(30, -1), eng["null_dist"], atol=1e-5
        )

    def test_progress_bar_accepted(self, bc_inmem):
        out = bc_inmem.permutation_test(
            n_permute=10, random_state=0, progress_bar=False
        )
        assert "p" in out
