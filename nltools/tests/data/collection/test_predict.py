"""fit / compute_contrasts / predict / predict_group — the modeling surface.

predict (per-subject predict-after-fit) and predict_group (group MVPA, subjects
as samples) are two distinct prediction paths, each with its own contract.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from nltools.data import BrainCollection, BrainData


# ---------------------------------------------------------------------------
# Signatures (pass on scaffold)
# ---------------------------------------------------------------------------


class TestFitSignature:
    def test_fit_accepts_X_resolution_options(self):
        sig = inspect.signature(BrainCollection.fit)
        assert "X" in sig.parameters
        assert sig.parameters["X"].default is None

    def test_fit_default_scale(self):
        sig = inspect.signature(BrainCollection.fit)
        assert sig.parameters["scale"].default == "auto"
        assert sig.parameters["standardize"].default == "auto"
        assert "scale_value" not in sig.parameters

    def test_fit_glm_ar1_raises_not_implemented(self, tmp_path):
        """AR noise models are OLS-only in the collection closed form -> refuse
        rather than silently approximate; point users to per-subject BrainData."""
        import numpy as np
        import pytest
        from nibabel import Nifti1Image

        from nltools.data import BrainCollection

        rng = np.random.default_rng(0)
        img = Nifti1Image(
            (rng.standard_normal((4, 4, 4, 20))).astype("float32"), np.eye(4)
        )
        mask = Nifti1Image(np.ones((4, 4, 4), dtype="int8"), np.eye(4))
        bc = BrainCollection(
            [BrainData(img, mask=mask)], mask=mask, lazy=False, cache_dir=None
        )

        import pandas as pd

        design = pd.DataFrame({"Intercept": np.ones(20), "A": rng.standard_normal(20)})
        with pytest.raises(NotImplementedError, match="noise_model='ols'"):
            bc.fit(model="glm", X=design, noise_model="ar1")

    def test_fit_no_output_or_save_kwargs(self):
        """What's gone: ``output=`` / ``save=`` removed from fit."""
        sig = inspect.signature(BrainCollection.fit)
        assert "output" not in sig.parameters
        assert "save" not in sig.parameters


class TestComputeContrastsSignature:
    def test_default_contrast_type_is_beta(self):
        sig = inspect.signature(BrainCollection.compute_contrasts)
        assert sig.parameters["statistic"].default == "beta"

    def test_accepts_list_or_dict_contrasts(self):
        sig = inspect.signature(BrainCollection.compute_contrasts)
        assert "contrasts" in sig.parameters


class TestPredictSignature:
    def test_dispatch_args_default_none(self):
        """bc.predict — per-subject decoding (y) and predict-after-fit (X_new)."""
        sig = inspect.signature(BrainCollection.predict)
        assert sig.parameters["y"].default is None
        assert sig.parameters["X_new"].default is None

    def test_group_mvpa_kwargs_live_on_predict_group(self):
        sig = inspect.signature(BrainCollection.predict_group)
        assert sig.parameters["spatial_scale"].default == "whole_brain"
        assert sig.parameters["model"].default == "svm"
        assert sig.parameters["cv"].default == "logo"
        assert sig.parameters["n_permute"].default == 0

    def test_predict_carries_per_subject_decode_kwargs(self):
        """#478 phases 2-3: predict(y=) maps BrainData.predict over subjects."""
        sig = inspect.signature(BrainCollection.predict)
        assert sig.parameters["spatial_scale"].default == "whole_brain"
        assert sig.parameters["model"].default == "svm"
        # Per-subject decoding: an int KFold default — 'logo' would be
        # incoherent inside a single subject without a groups variable.
        assert sig.parameters["cv"].default == 5
        assert sig.parameters["groups"].default is None
        assert sig.parameters["cache"].default == "auto"


# ---------------------------------------------------------------------------
# Behavior — xfail until impl
# ---------------------------------------------------------------------------


class TestFitBehavior:
    def test_fit_glm_returns_collection(self, bc_with_designs):
        out = bc_with_designs.fit(model="glm", n_jobs=1)
        assert isinstance(out, BrainCollection)
        assert out.n_subjects == bc_with_designs.n_subjects
        # cache='auto' on a loaded source stays in memory
        assert all(out.is_loaded)

    def test_fit_glm_cache_true_writes_bundles(self, bc_with_designs):
        from pathlib import Path

        out = bc_with_designs.fit(model="glm", cache=True, n_jobs=1)
        assert not any(out.is_loaded)  # path-backed
        for item in out._items:
            assert isinstance(item, Path)
            assert item.suffix == ".h5"
            assert item.exists()

    def test_fit_glm_bundle_has_expected_arrays(self, bc_with_designs):
        from nltools.data.collection.execution import read_glm_bundle

        out = bc_with_designs.fit(model="glm", cache=True, n_jobs=1)
        bundle = read_glm_bundle(out._items[0])
        # 8 obs, 27 voxels, 2 regressors
        assert bundle["betas"].shape == (2, 27)
        assert bundle["residuals"].shape == (8, 27)
        assert bundle["sigma2"].shape == (27,)
        assert bundle["X"].shape == (8, 2)
        assert bundle["regressor_names"] == ["a", "b"]

    def test_fit_x_shared_designmatrix(self, bc_with_designs, tiny_design_factory):
        shared = tiny_design_factory(n_obs=8, seed=99)
        out = bc_with_designs.fit(model="glm", X=shared, n_jobs=1)
        assert isinstance(out, BrainCollection)

    def test_fit_x_callable_receives_design_context(self, bc_with_designs):
        seen = []

        def make_design(ctx):
            seen.append(ctx)
            return ctx.dm  # passthrough

        bc_with_designs.fit(model="glm", X=make_design, n_jobs=1)
        assert len(seen) == bc_with_designs.n_subjects
        assert hasattr(seen[0], "TR")
        assert hasattr(seen[0], "subject")
        assert hasattr(seen[0], "bd")

    def test_fit_x_none_requires_paired_designs(self, tiny_mask, tiny_brain_factory):
        brains = [tiny_brain_factory(seed=i) for i in range(2)]
        bc = BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=None)
        with pytest.raises(ValueError, match="no paired design"):
            bc.fit(model="glm", X=None)

    def test_fit_unknown_model_raises(self, bc_with_designs):
        with pytest.raises(ValueError, match="unknown model"):
            bc_with_designs.fit(model="bogus")


class TestComputeContrastsBehavior:
    @pytest.fixture(scope="class")
    def fitted_bc(self, tmp_path_factory):
        """One GLM fit shared across the contrast tests.

        Class-scoped so the ~8 contrast tests don't each re-run the fit in
        setup. Safe: ``compute_contrasts`` returns a copy and never mutates
        the collection. Built self-contained (not via the function-scoped
        ``bc_with_designs``) to avoid a fixture ScopeMismatch.
        """
        import nibabel as nib
        import numpy as np
        import pandas as pd

        from nltools.data import BrainCollection, DesignMatrix

        affine = np.eye(4) * 2
        affine[3, 3] = 1
        mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.int8), affine)

        n_obs = 8
        brains = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            vol = rng.standard_normal((3, 3, 3, n_obs)).astype(np.float32)
            brains.append(BrainData(nib.Nifti1Image(vol, affine), mask=mask))

        designs = []
        for seed in range(10, 13):
            rng = np.random.default_rng(seed)
            t = np.linspace(0, 2 * np.pi, n_obs)
            designs.append(
                DesignMatrix(
                    pd.DataFrame(
                        {
                            "a": np.sin(t) + 0.1 * rng.standard_normal(n_obs),
                            "b": np.cos(t) + 0.1 * rng.standard_normal(n_obs),
                        }
                    ),
                    TR=2.0,
                )
            )

        cache_dir = tmp_path_factory.mktemp("fitted_bc") / "cache"
        bc = BrainCollection(
            brains, mask=mask, designs=designs, lazy=False, cache_dir=cache_dir
        )
        return bc.fit(model="glm", cache=True, n_jobs=1)

    def test_single_contrast_returns_collection(self, fitted_bc):
        out = fitted_bc.compute_contrasts("a - b", statistic="beta", n_jobs=1)
        assert isinstance(out, BrainCollection)
        assert out.n_subjects == fitted_bc.n_subjects

    def test_single_regressor_identity_contrast(self, fitted_bc):
        out = fitted_bc.compute_contrasts("a", statistic="beta", n_jobs=1)
        assert isinstance(out, BrainCollection)

    def test_multiple_contrasts_returns_dict(self, fitted_bc):
        out = fitted_bc.compute_contrasts(
            {"main": "a - b", "avg": "a + b"},
            statistic="beta",
            n_jobs=1,
        )
        assert isinstance(out, dict)
        assert set(out.keys()) == {"main", "avg"}
        assert all(isinstance(v, BrainCollection) for v in out.values())

    def test_all_contrast_types_returns_dict_keyed_by_type(self, fitted_bc):
        out = fitted_bc.compute_contrasts("a", statistic="all", n_jobs=1)
        assert isinstance(out, dict)
        for k in ("beta", "t", "z", "p", "se"):
            assert k in out
            assert isinstance(out[k], BrainCollection)

    def test_contrast_writes_lineage_sidecar(self, fitted_bc):
        import json

        out = fitted_bc.compute_contrasts("a", statistic="beta", n_jobs=1)
        item = out._items[0]
        sidecar = item.parent / (item.name[:-7] + ".json")  # .nii.gz → .json
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["op"].startswith("contrast_")
        assert "step_id" in data

    def test_contrast_unknown_regressor_raises(self, fitted_bc):
        from nltools.data.collection import BrainCollectionWorkerError

        with pytest.raises(BrainCollectionWorkerError, match="not in design"):
            fitted_bc.compute_contrasts("nonexistent", n_jobs=1)

    def test_contrast_invalid_type_raises(self, fitted_bc):
        with pytest.raises(ValueError, match="statistic"):
            fitted_bc.compute_contrasts("a", statistic="bogus", n_jobs=1)


class TestPredictDispatch:
    """predict() is the per-subject path; group MVPA lives on predict_group()."""

    def test_y_and_x_new_together_raise(self, bc_inmem):
        with pytest.raises(ValueError, match="both"):
            bc_inmem.predict(y=[0, 1, 0], X_new=np.zeros((5, 3)))

    def test_no_args_without_stored_y_raises(self, bc_inmem):
        """bc_inmem items carry no .Y — nothing to decode against."""
        with pytest.raises(ValueError, match="Y"):
            bc_inmem.predict()

    def test_predict_group_requires_single_map_per_subject(self, bc_inmem):
        """Multi-row items raise. (bc_inmem items are (8, 27).)"""
        with pytest.raises(ValueError, match="single-map-per-subject"):
            bc_inmem.predict_group(np.array([0, 1, 0]))

    def test_predict_group_returns_predict_with_cv_attrs(
        self,
        tiny_mask,
        tiny_brain_factory,
    ):
        # Use single-map-per-subject items (1, 27).
        from nltools.data.fitresults import Predict

        np.random.seed(0)
        single_maps = [
            BrainData(np.random.randn(1, 27).astype(np.float32), mask=tiny_mask)
            for _ in range(6)
        ]
        bc = BrainCollection(single_maps, mask=tiny_mask, lazy=False, cache_dir=None)
        out = bc.predict_group(np.array([0, 1, 0, 1, 0, 1]))
        # Whole-brain group MVPA returns a Predict dataclass, not a BrainData.
        assert isinstance(out, Predict)
        assert out.predictions.shape == (6,)
        assert out.scores.shape == (6,)
        assert isinstance(out.mean_score, float)
        # Brain-space output is itself a BrainData; estimator is fitted on all data.
        assert isinstance(out.weight_map, BrainData)
        assert out.estimator is not None

    def test_x_new_only_returns_collection(self, bc_ridge_fitted):
        from pathlib import Path

        # bc_ridge_fitted designs are 2-column → X_new shape (5, 2).
        X_new = np.zeros((5, 2), dtype=np.float32)
        out = bc_ridge_fitted.predict(X_new=X_new, n_jobs=1)
        assert isinstance(out, BrainCollection)
        assert out.n_subjects == bc_ridge_fitted.n_subjects
        for item in out._items:
            assert isinstance(item, Path)
            assert item.suffix == ".gz" or item.suffix == ".nii"
            assert item.exists()
        # First subject's prediction shape should be (5, 27).
        bd0 = out[0]
        assert bd0.data.shape == (5, 27)
        # Sidecar carries the predict_x_new op tag.
        import json

        first = out._items[0]
        sidecar = first.parent / (first.name[:-7] + ".json")
        assert sidecar.exists()
        assert json.loads(sidecar.read_text())["op"] == "predict_x_new"

    def test_predict_x_new_requires_ridge_bundle(self, bc_inmem):
        with pytest.raises(ValueError, match="ridge bundle"):
            bc_inmem.predict(X_new=np.zeros((5, 2)))


class TestPredictGroupCarveOut:
    """#478 phase 1: group MVPA moved to an explicit name; predict(y=) directs there."""

    def _single_map_bc(self, tiny_mask, n=6, seed=0):
        rng = np.random.default_rng(seed)
        maps = [
            BrainData(rng.standard_normal((1, 27)).astype(np.float32), mask=tiny_mask)
            for _ in range(n)
        ]
        return BrainCollection(maps, mask=tiny_mask, lazy=False, cache_dir=None)

    def test_predict_group_returns_predict(self, tiny_mask):
        from nltools.data.fitresults import Predict

        bc = self._single_map_bc(tiny_mask)
        out = bc.predict_group(np.array([0, 1, 0, 1, 0, 1]))
        assert isinstance(out, Predict)
        assert out.predictions.shape == (6,)
        assert isinstance(out.mean_score, float)

    def test_predict_y_is_per_subject_not_group(self, tiny_mask):
        """predict(y=) maps over subjects — single-map items can't CV within."""
        bc = self._single_map_bc(tiny_mask)
        with pytest.raises(Exception, match="predict_group|splits|samples|fold"):
            # One map per subject → within-subject CV is impossible; the
            # group question belongs to predict_group.
            bc.predict(y=np.array([0]), n_jobs=1)

    def test_int_cv_honors_groups(self, tiny_mask):
        """cv=2 with groups= must keep each group intact within a fold.

        Previously an int cv resolved to plain KFold, which silently ignored
        groups= — the same subject could sit in train and test of every fold.
        """
        bc = self._single_map_bc(tiny_mask, n=8, seed=1)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        out = bc.predict_group(y, cv=2, groups=groups)
        # cv_folds records each sample's fold; both members of a group must share one.
        folds = np.asarray(out.cv_folds)
        for g in np.unique(groups):
            assert len(set(folds[groups == g])) == 1, (
                f"group {g} split across folds {folds[groups == g]}"
            )

    def test_permutation_null_ported(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        out = bc.predict_group(y, n_permute=10, random_state=0)
        assert out.permutation_scores.shape == (10,)
        assert 0.0 < out.permutation_pvalue <= 1.0

    def test_permutation_null_reproducible(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        a = bc.predict_group(y, n_permute=8, random_state=7)
        b = bc.predict_group(y, n_permute=8, random_state=7)
        np.testing.assert_array_equal(a.permutation_scores, b.permutation_scores)
        assert a.permutation_pvalue == b.permutation_pvalue

    def test_no_null_by_default(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        out = bc.predict_group(np.array([0, 1, 0, 1, 0, 1]))
        assert out.permutation_scores is None
        assert out.permutation_pvalue is None

    def test_legacy_cv_pipeline_removed(self, bc_inmem):
        assert not hasattr(bc_inmem, "cv")
        with pytest.raises(ImportError):
            from nltools.data.collection import BrainCollectionPipeline  # noqa: F401


class TestPredictGroupNull:
    """predict_group(n_permute>0) across all three spatial scales (F5/C-2).

    The null was whole-brain-only (scalar mean_score assumed) and re-ran the
    full CV — including a discarded all-data refit and weight-map stacking —
    per permutation. Now: per-ROI nulls for 'roi', per-voxel null + BrainData
    p-map for 'searchlight', and null iterations run the scoring core only.
    """

    def _single_map_bc(self, tiny_mask, n=6, seed=0):
        rng = np.random.default_rng(seed)
        maps = [
            BrainData(rng.standard_normal((1, 27)).astype(np.float32), mask=tiny_mask)
            for _ in range(n)
        ]
        return BrainCollection(maps, mask=tiny_mask, lazy=False, cache_dir=None)

    @staticmethod
    def _two_parcel_atlas(tiny_mask):
        import nibabel as nib

        labels = np.full((3, 3, 3), 2, dtype=np.int32)
        labels.reshape(-1)[:13] = 1
        return nib.Nifti1Image(labels, tiny_mask.affine)

    def test_roi_null_shapes_and_per_roi_pvalues(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        out = bc.predict_group(
            y,
            spatial_scale="roi",
            roi_mask=self._two_parcel_atlas(tiny_mask),
            n_permute=8,
            random_state=0,
            n_jobs=1,
        )
        assert out.permutation_scores.shape == (8, 2)
        p = np.asarray(out.permutation_pvalue)
        assert p.shape == (2,)
        assert np.all((p > 0.0) & (p <= 1.0))

    def test_roi_null_deterministic(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        atlas = self._two_parcel_atlas(tiny_mask)
        kw = {
            "spatial_scale": "roi",
            "roi_mask": atlas,
            "n_permute": 6,
            "random_state": 11,
            "n_jobs": 1,
        }
        a = bc.predict_group(y, **kw)
        b = bc.predict_group(y, **kw)
        np.testing.assert_array_equal(a.permutation_scores, b.permutation_scores)
        np.testing.assert_array_equal(
            np.asarray(a.permutation_pvalue), np.asarray(b.permutation_pvalue)
        )

    def test_searchlight_null_shapes_and_pvalue_map(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        out = bc.predict_group(
            y,
            spatial_scale="searchlight",
            radius_mm=3.0,
            n_permute=5,
            random_state=0,
            n_jobs=1,
        )
        assert out.permutation_scores.shape == (5, 27)
        assert isinstance(out.permutation_pvalue, BrainData)
        pv = np.asarray(out.permutation_pvalue.data).reshape(-1)
        acc = np.asarray(out.accuracy_map.data).reshape(-1)
        scored = np.isfinite(acc)
        assert np.all((pv[scored] > 0.0) & (pv[scored] <= 1.0))
        assert np.all(np.isnan(pv[~scored]))

    def test_searchlight_null_deterministic(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        kw = {
            "spatial_scale": "searchlight",
            "radius_mm": 3.0,
            "n_permute": 4,
            "random_state": 5,
            "n_jobs": 1,
        }
        a = bc.predict_group(y, **kw)
        b = bc.predict_group(y, **kw)
        np.testing.assert_array_equal(a.permutation_scores, b.permutation_scores)
        np.testing.assert_array_equal(
            np.asarray(a.permutation_pvalue.data), np.asarray(b.permutation_pvalue.data)
        )

    def test_whole_brain_null_matches_full_rerun_stream(self, tiny_mask):
        """The seeded draw stream and per-draw scores match the old
        run-the-full-CV implementation (null values unchanged for a seed)."""
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        out = bc.predict_group(y, n_permute=5, random_state=3, n_jobs=1)
        rng = np.random.default_rng(3)
        expected = [
            bc.predict_group(rng.permutation(y), n_jobs=1).mean_score for _ in range(5)
        ]
        np.testing.assert_allclose(out.permutation_scores, expected)

    def test_null_iterations_skip_refit_and_weight_maps(self, tiny_mask, monkeypatch):
        """Efficiency contract: only the observed run extracts weight maps —
        null iterations run the scoring core only."""
        from nltools.data.braindata import prediction as bd_prediction

        calls = {"n": 0}
        orig = bd_prediction._extract_weight_map

        def counting(*args, **kwargs):
            calls["n"] = calls["n"] + 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(bd_prediction, "_extract_weight_map", counting)
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        bc.predict_group(y, n_permute=4, random_state=0, n_jobs=1)
        # Observed run: one per fold (6 logo folds) + the all-data refit.
        assert calls["n"] == 7, (
            f"weight-map extraction ran {calls['n']} times — null iterations "
            f"must not refit or stack weight maps"
        )

    def test_whole_brain_pvalue_is_phipson_smyth_upper(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        y = np.array([0, 1, 0, 1, 0, 1])
        out = bc.predict_group(y, n_permute=10, random_state=2, n_jobs=1)
        expected = (1.0 + np.sum(out.permutation_scores >= out.mean_score)) / 11.0
        assert out.permutation_pvalue == pytest.approx(expected)


class TestPredictPerSubject:
    """#478 phases 2-3: predict(y=) maps BrainData.predict over subjects.

    One model per subject (CV within each subject's own rows), results in a
    `PredictCollection` carrying the collection's per-subject metadata.
    """

    N_OBS = 12

    def _labels(self, seed=0):
        return np.tile([0, 1], self.N_OBS // 2)

    def _bc(self, tiny_mask, tiny_brain_factory, *, with_y=False, metadata=None):
        from nltools.data import BrainCollection

        brains = [tiny_brain_factory(n_obs=self.N_OBS, seed=i) for i in range(3)]
        if with_y:
            for bd in brains:
                bd.Y = {
                    "condition": self._labels(),
                    "run": np.repeat([0, 1, 2], self.N_OBS // 3),
                }
        return BrainCollection(
            brains, mask=tiny_mask, metadata=metadata, lazy=False, cache_dir=None
        )

    def test_returns_predict_collection_one_result_per_subject(
        self, tiny_mask, tiny_brain_factory
    ):
        from nltools.data.fitresults import Predict, PredictCollection

        bc = self._bc(tiny_mask, tiny_brain_factory)
        pc = bc.predict(y=self._labels(), cv=3, random_state=0, n_jobs=1)
        assert isinstance(pc, PredictCollection)
        assert len(pc) == 3
        assert all(isinstance(r, Predict) for r in pc)
        # Whole-brain decoding fields per subject.
        assert pc[0].predictions.shape == (self.N_OBS,)
        assert isinstance(pc[0].mean_score, float)

    def test_models_are_per_subject(self, tiny_mask, tiny_brain_factory):
        """Different subject data → different decoder weight maps."""
        bc = self._bc(tiny_mask, tiny_brain_factory)
        pc = bc.predict(y=self._labels(), cv=3, random_state=0, n_jobs=1)
        w0 = pc[0].weight_map.data.reshape(-1)
        w1 = pc[1].weight_map.data.reshape(-1)
        assert not np.allclose(w0, w1)

    def test_stored_y_default(self, tiny_mask, tiny_brain_factory):
        """y=None decodes each subject's own .Y — labels travel with the data."""
        bc = self._bc(tiny_mask, tiny_brain_factory, with_y=True)
        pc = bc.predict(y="condition", cv=3, random_state=0, n_jobs=1)
        assert len(pc) == 3

    def test_y_list_per_subject(self, tiny_mask, tiny_brain_factory):
        bc = self._bc(tiny_mask, tiny_brain_factory)
        ys = [np.roll(self._labels(), i) for i in range(3)]
        pc = bc.predict(y=ys, cv=3, random_state=0, n_jobs=1)
        assert len(pc) == 3

    def test_y_list_wrong_length_raises(self, tiny_mask, tiny_brain_factory):
        bc = self._bc(tiny_mask, tiny_brain_factory)
        with pytest.raises(ValueError, match="3 subjects"):
            bc.predict(y=[self._labels()] * 2, n_jobs=1)

    def test_groups_column_logo_within_subject(self, tiny_mask, tiny_brain_factory):
        """cv='logo', groups='run' → leave-one-run-out within each subject."""
        bc = self._bc(tiny_mask, tiny_brain_factory, with_y=True)
        pc = bc.predict(
            y="condition", cv="logo", groups="run", random_state=0, n_jobs=1
        )
        # 3 runs per subject → 3 folds each.
        assert pc[0].scores.shape == (3,)

    def test_metadata_carried_into_scores(self, tiny_mask, tiny_brain_factory):
        bc = self._bc(
            tiny_mask,
            tiny_brain_factory,
            metadata={"subject": ["s1", "s2", "s3"]},
        )
        pc = bc.predict(y=self._labels(), cv=3, random_state=0, n_jobs=1)
        assert pc.scores["subject"].to_list() == ["s1", "s2", "s3"]

    def test_weight_maps_stack_for_second_level(self, tiny_mask, tiny_brain_factory):
        from nltools.data import BrainData

        bc = self._bc(tiny_mask, tiny_brain_factory)
        pc = bc.predict(y=self._labels(), cv=3, random_state=0, n_jobs=1)
        wm = pc.weight_maps
        assert isinstance(wm, BrainData)
        assert wm.data.shape == (3, 27)

    def test_parallel_matches_serial(self, tiny_mask, tiny_brain_factory):
        bc = self._bc(tiny_mask, tiny_brain_factory)
        serial = bc.predict(y=self._labels(), cv=3, random_state=0, n_jobs=1)
        par = bc.predict(y=self._labels(), cv=3, random_state=0, n_jobs=2)
        for s, p in zip(serial, par):
            np.testing.assert_array_equal(s.predictions, p.predictions)
            assert s.mean_score == p.mean_score

    def test_fit_bundle_items_raise_with_guidance(self, bc_ridge_fitted):
        with pytest.raises(ValueError, match="bundle"):
            bc_ridge_fitted.predict(y=np.tile([0, 1], 4), n_jobs=1)

    def test_inmem_items_without_y_raise_eagerly(self, tiny_mask, tiny_brain_factory):
        bc = self._bc(tiny_mask, tiny_brain_factory, with_y=False)
        with pytest.raises(ValueError, match="Y"):
            bc.predict(n_jobs=1)


class TestPredictPerSubjectCaching:
    """cache=True writes one predict bundle (.h5) per subject."""

    N_OBS = 12

    def _bc(self, tiny_mask, tiny_brain_factory, tmp_path):
        from nltools.data import BrainCollection

        brains = [tiny_brain_factory(n_obs=self.N_OBS, seed=i) for i in range(3)]
        return BrainCollection(
            brains, mask=tiny_mask, lazy=False, cache_dir=tmp_path / "cache"
        )

    def _labels(self):
        return np.tile([0, 1], self.N_OBS // 2)

    def test_cache_true_writes_bundles_and_records_paths(
        self, tiny_mask, tiny_brain_factory, tmp_path
    ):
        from pathlib import Path

        bc = self._bc(tiny_mask, tiny_brain_factory, tmp_path)
        pc = bc.predict(y=self._labels(), cv=3, random_state=0, cache=True, n_jobs=1)
        assert pc.paths is not None and len(pc.paths) == 3
        for p in pc.paths:
            assert isinstance(p, Path)
            assert p.suffix == ".h5"
            assert p.exists()

    def test_cached_bundle_round_trips_ingredients(
        self, tiny_mask, tiny_brain_factory, tmp_path
    ):
        from nltools.data.collection.execution import read_predict_bundle

        bc = self._bc(tiny_mask, tiny_brain_factory, tmp_path)
        pc = bc.predict(y=self._labels(), cv=3, random_state=0, cache=True, n_jobs=1)
        loaded = read_predict_bundle(pc.paths[0])
        np.testing.assert_array_equal(loaded.predictions, pc[0].predictions)
        np.testing.assert_allclose(
            loaded.weight_map.data, pc[0].weight_map.data, rtol=1e-6
        )
        assert loaded.mean_score == pc[0].mean_score

    def test_cached_results_carry_no_estimator(
        self, tiny_mask, tiny_brain_factory, tmp_path
    ):
        """Bundles store the ingredients, never a pickled estimator — and the
        in-memory results mirror the bundle so resumed sessions see the same
        fields."""
        bc = self._bc(tiny_mask, tiny_brain_factory, tmp_path)
        pc = bc.predict(y=self._labels(), cv=3, random_state=0, cache=True, n_jobs=1)
        assert all(r.estimator is None for r in pc)

    def test_uncached_results_keep_estimator(self, tiny_mask, tiny_brain_factory):
        from nltools.data import BrainCollection

        brains = [tiny_brain_factory(n_obs=self.N_OBS, seed=i) for i in range(2)]
        bc = BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=None)
        pc = bc.predict(y=self._labels(), cv=3, random_state=0, n_jobs=1)
        assert all(r.estimator is not None for r in pc)

    def test_pathbacked_nifti_items_decode_with_explicit_y(self, bc_pathbacked):
        """NIfTI items carry no .Y; explicit y decodes and cache='auto' persists."""
        y = np.tile([0, 1], 4)  # tiny_nifti_paths items have 8 obs
        pc = bc_pathbacked.predict(y=y, cv=2, random_state=0, n_jobs=1)
        assert len(pc) == 3
        assert pc.paths is not None  # path-backed source → 'auto' caches

    def test_bundle_model_spec_refits_customized_estimator(
        self, tiny_mask, tiny_brain_factory, tmp_path
    ):
        """The stored model_spec is a refit ingredient, not a repr string (C-15)."""
        import json

        import h5py
        from sklearn.svm import SVC

        from nltools.data.braindata.prediction import _model_from_spec

        bc = self._bc(tiny_mask, tiny_brain_factory, tmp_path)
        model = SVC(C=10.0, kernel="linear")
        pc = bc.predict(
            y=self._labels(), model=model, cv=3, random_state=0, cache=True, n_jobs=1
        )
        with h5py.File(pc.paths[0], "r") as f:
            spec = json.loads(f.attrs["model_spec"])
        rebuilt = _model_from_spec(spec["model"])
        assert isinstance(rebuilt, SVC)
        assert rebuilt.get_params()["C"] == 10.0
        # Refit on data of the decoded shape works — the docstring's
        # "refit from the stored spec on demand" claim holds.
        rng = np.random.default_rng(0)
        rebuilt.fit(rng.standard_normal((self.N_OBS, 27)), self._labels())


class TestWorkerClosureHygiene:
    """Workers must never capture the parent BrainCollection (F8).

    execution-model.md: "a closure that references self (the BrainCollection)
    ships every loaded BrainData to every worker." The fix pattern is hoisting
    `parent_step_id = self._step_id` into a local before defining the closure
    (as .fit() does). These tests spy on execution._apply and inspect the
    dispatched worker's closure cells.
    """

    N_OBS = 12

    @staticmethod
    def _spy_worker(monkeypatch):
        from nltools.data.collection import execution

        captured = {}
        orig = execution._apply

        def spy(bc_arg, fn, **kw):
            captured["worker"] = fn
            return orig(bc_arg, fn, **kw)

        monkeypatch.setattr(execution, "_apply", spy)
        return captured

    @staticmethod
    def _assert_no_collection_in_closure(fn):
        cells = fn.__closure__ or ()
        offenders = [
            c.cell_contents
            for c in cells
            if isinstance(c.cell_contents, BrainCollection)
        ]
        assert not offenders, (
            "worker closure captures the BrainCollection — loky/cloudpickle "
            "would serialize the whole collection per dispatched task (O(S^2))"
        )

    def test_predict_mvpa_worker_does_not_capture_collection(
        self, tiny_mask, tiny_brain_factory, monkeypatch
    ):
        captured = self._spy_worker(monkeypatch)
        brains = [tiny_brain_factory(n_obs=self.N_OBS, seed=i) for i in range(2)]
        bc = BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=None)
        bc.predict(
            y=np.tile([0, 1], self.N_OBS // 2),
            cv=3,
            random_state=0,
            n_jobs=1,
            cache=False,
        )
        self._assert_no_collection_in_closure(captured["worker"])

    def test_predict_x_new_worker_does_not_capture_collection(
        self, bc_ridge_fitted, monkeypatch
    ):
        captured = self._spy_worker(monkeypatch)
        rng = np.random.default_rng(0)
        bc_ridge_fitted.predict(X_new=rng.standard_normal((4, 2)), n_jobs=1)
        self._assert_no_collection_in_closure(captured["worker"])

    def test_fit_worker_does_not_capture_collection(self, bc_with_designs, monkeypatch):
        """Pin: the .fit() path already hoists parent_step_id — keep it so."""
        captured = self._spy_worker(monkeypatch)
        bc_with_designs.fit(model="glm", n_jobs=1, cache=False)
        self._assert_no_collection_in_closure(captured["worker"])


class TestStringLabelDecoding:
    """String class labels through the per-subject path and the cache (F4/F10)."""

    N_OBS = 12

    def _bc(self, tiny_mask, tiny_brain_factory, cache_dir=None):
        brains = [tiny_brain_factory(n_obs=self.N_OBS, seed=i) for i in range(3)]
        for bd in brains:
            bd.Y = {"condition": np.tile(["face", "house"], self.N_OBS // 2)}
        return BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=cache_dir)

    def test_predict_string_labels_per_subject(self, tiny_mask, tiny_brain_factory):
        """bd.Y stores string conditions; predict(y='condition') must decode."""
        bc = self._bc(tiny_mask, tiny_brain_factory)
        pc = bc.predict(y="condition", cv=3, random_state=0, n_jobs=1, cache=False)
        assert len(pc) == 3
        for r in pc:
            preds = np.asarray(r.predictions)
            assert preds.dtype.kind == "U"
            assert set(np.unique(preds)) <= {"face", "house"}

    def test_predict_bundle_round_trips_string_predictions(
        self, tiny_mask, tiny_brain_factory, tmp_path
    ):
        """cache=True must persist string predictions bit-perfectly (F10)."""
        from nltools.data.collection.execution import read_predict_bundle

        bc = self._bc(tiny_mask, tiny_brain_factory, cache_dir=tmp_path / "cache")
        pc = bc.predict(y="condition", cv=3, random_state=0, n_jobs=1, cache=True)
        assert pc.paths is not None
        loaded = read_predict_bundle(pc.paths[0])
        np.testing.assert_array_equal(loaded.predictions, pc[0].predictions)
        assert np.asarray(loaded.predictions).dtype.kind == "U"


class TestUserH5Items:
    """User-saved BrainData .h5 files are images, not fit bundles (F7).

    Bare-suffix detection misclassified them, so predict_group/predict(y=)
    refused valid image collections and predict(X_new=) died inside
    read_ridge_bundle with a misleading schema error.
    """

    def _mask(self, tmp_path, tiny_mask):
        import nibabel as nib

        mask_path = tmp_path / "mask.nii.gz"
        nib.save(tiny_mask, mask_path)
        return nib.load(mask_path)

    def _h5_paths(self, tmp_path, mask, *, n_subjects, n_obs, seed=0):
        rng = np.random.default_rng(seed)
        paths = []
        for i in range(n_subjects):
            bd = BrainData(
                rng.standard_normal((n_obs, 27)).astype(np.float32), mask=mask
            )
            p = tmp_path / f"sub-{i + 1:02d}.h5"
            bd.write(p)
            paths.append(p)
        return paths

    def test_predict_group_works_on_user_h5_items(self, tmp_path, tiny_mask):
        mask = self._mask(tmp_path, tiny_mask)
        paths = self._h5_paths(tmp_path, mask, n_subjects=6, n_obs=1)
        bc = BrainCollection.from_paths(paths, mask=mask, cache_dir=tmp_path / "cache")
        out = bc.predict_group(np.array([0, 1, 0, 1, 0, 1]))
        assert out.predictions.shape == (6,)

    def test_predict_y_works_on_user_h5_items(self, tmp_path, tiny_mask):
        mask = self._mask(tmp_path, tiny_mask)
        paths = self._h5_paths(tmp_path, mask, n_subjects=3, n_obs=12)
        bc = BrainCollection.from_paths(paths, mask=mask, cache_dir=tmp_path / "cache")
        pc = bc.predict(
            y=np.tile([0, 1], 6), cv=3, random_state=0, n_jobs=1, cache=False
        )
        assert len(pc) == 3

    def test_predict_x_new_on_user_h5_raises_clean_error(self, tmp_path, tiny_mask):
        mask = self._mask(tmp_path, tiny_mask)
        paths = self._h5_paths(tmp_path, mask, n_subjects=2, n_obs=1)
        bc = BrainCollection.from_paths(paths, mask=mask, cache_dir=tmp_path / "cache")
        with pytest.raises(ValueError, match="not a ridge bundle"):
            bc.predict(X_new=np.zeros((4, 2)), n_jobs=1)


class TestResolveCv:
    """The int/str cv spec -> sklearn splitter resolution is a pure function.

    String names follow sklearn's splitter classes: 'loo' (LeaveOneOut) and
    'logo' (LeaveOneGroupOut). The old domain names 'loso'/'loro' were removed
    in v0.6.0 — they were both LeaveOneGroupOut, differing only in the default
    groups, which is what ``groups=`` expresses directly.
    """

    def test_int_without_groups_regression_is_kfold(self):
        from sklearn.model_selection import KFold

        from nltools.cross_validation import resolve_cv

        assert isinstance(resolve_cv(3), KFold)

    def test_int_without_groups_classification_is_stratifiedkfold(self):
        from sklearn.model_selection import StratifiedKFold

        from nltools.cross_validation import resolve_cv

        assert isinstance(resolve_cv(3, classifier=True), StratifiedKFold)

    def test_int_shuffle_and_random_state_pass_through(self):
        from nltools.cross_validation import resolve_cv

        cv = resolve_cv(3, shuffle=True, random_state=7)
        assert cv.shuffle is True
        assert cv.random_state == 7

    def test_int_with_groups_regression_is_groupkfold(self):
        from sklearn.model_selection import GroupKFold

        from nltools.cross_validation import resolve_cv

        cv = resolve_cv(3, groups=np.array([0, 0, 1, 1, 2, 2]), classifier=False)
        assert isinstance(cv, GroupKFold)

    def test_int_with_groups_classification_is_stratifiedgroupkfold(self):
        from sklearn.model_selection import StratifiedGroupKFold

        from nltools.cross_validation import resolve_cv

        cv = resolve_cv(3, groups=np.array([0, 0, 1, 1, 2, 2]), classifier=True)
        assert isinstance(cv, StratifiedGroupKFold)

    def test_loo_is_leave_one_out(self):
        from sklearn.model_selection import LeaveOneOut

        from nltools.cross_validation import resolve_cv

        assert isinstance(resolve_cv("loo"), LeaveOneOut)

    def test_logo_is_leave_one_group_out(self):
        from sklearn.model_selection import LeaveOneGroupOut

        from nltools.cross_validation import resolve_cv

        assert isinstance(resolve_cv("logo"), LeaveOneGroupOut)

    def test_removed_names_raise_with_migration_guidance(self):
        from nltools.cross_validation import resolve_cv

        with pytest.raises(ValueError, match="logo"):
            resolve_cv("loso")
        with pytest.raises(ValueError, match="logo"):
            resolve_cv("loro")

    def test_unknown_string_raises(self):
        from nltools.cross_validation import resolve_cv

        with pytest.raises(ValueError, match="cv"):
            resolve_cv("bogus")

    def test_splitter_passes_through(self):
        from sklearn.model_selection import KFold

        from nltools.cross_validation import resolve_cv

        splitter = KFold(4)
        assert resolve_cv(splitter) is splitter
