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
        """bc.predict — per-subject path; y kept only to route to predict_group."""
        sig = inspect.signature(BrainCollection.predict)
        assert sig.parameters["y"].default is None
        assert sig.parameters["X_new"].default is None

    def test_group_mvpa_kwargs_live_on_predict_group(self):
        sig = inspect.signature(BrainCollection.predict_group)
        assert sig.parameters["spatial_scale"].default == "whole_brain"
        assert sig.parameters["model"].default == "svm"
        assert sig.parameters["cv"].default == "loso"
        assert sig.parameters["n_permute"].default == 0

    def test_predict_no_longer_carries_group_kwargs(self):
        sig = inspect.signature(BrainCollection.predict)
        for gone in ("spatial_scale", "model", "cv", "groups"):
            assert gone not in sig.parameters


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

    def test_y_raises_even_with_x_new(self, bc_inmem):
        with pytest.raises(ValueError, match="predict_group"):
            bc_inmem.predict(y=[0, 1, 0], X_new=np.zeros((5, 3)))

    def test_neither_arg_raises(self, bc_inmem):
        with pytest.raises(ValueError, match="X_new"):
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

    def test_predict_y_raises_with_guidance(self, tiny_mask):
        bc = self._single_map_bc(tiny_mask)
        with pytest.raises(ValueError, match="predict_group"):
            bc.predict(y=np.array([0, 1, 0, 1, 0, 1]))

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


class TestResolveGroupCv:
    """The int/str cv spec -> sklearn splitter resolution is a pure function."""

    def test_int_without_groups_is_kfold(self):
        from sklearn.model_selection import KFold

        from nltools.cross_validation import resolve_group_cv

        assert isinstance(resolve_group_cv(3), KFold)

    def test_int_with_groups_regression_is_groupkfold(self):
        from sklearn.model_selection import GroupKFold

        from nltools.cross_validation import resolve_group_cv

        cv = resolve_group_cv(3, groups=np.array([0, 0, 1, 1, 2, 2]), classifier=False)
        assert isinstance(cv, GroupKFold)

    def test_int_with_groups_classification_is_stratifiedgroupkfold(self):
        from sklearn.model_selection import StratifiedGroupKFold

        from nltools.cross_validation import resolve_group_cv

        cv = resolve_group_cv(3, groups=np.array([0, 0, 1, 1, 2, 2]), classifier=True)
        assert isinstance(cv, StratifiedGroupKFold)

    def test_loso_loro_are_leave_one_group_out(self):
        from sklearn.model_selection import LeaveOneGroupOut

        from nltools.cross_validation import resolve_group_cv

        assert isinstance(resolve_group_cv("loso"), LeaveOneGroupOut)
        assert isinstance(resolve_group_cv("loro"), LeaveOneGroupOut)

    def test_splitter_passes_through(self):
        from sklearn.model_selection import KFold

        from nltools.cross_validation import resolve_group_cv

        splitter = KFold(4)
        assert resolve_group_cv(splitter) is splitter
