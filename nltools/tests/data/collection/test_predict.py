"""fit / compute_contrasts / predict / cv() — the modeling surface.

SPEC §"predict and cv() — two distinct prediction paths" sets the contracts.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from nltools.data import BrainCollection, BrainData
from nltools.data.collection import BrainCollectionPipeline


XFAIL = pytest.mark.xfail(reason="not implemented", strict=True)


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
        assert sig.parameters["scale"].default is True
        assert sig.parameters["scale_value"].default == 100.0

    def test_fit_no_output_or_save_kwargs(self):
        """SPEC §"What's gone": ``output=`` / ``save=`` removed from fit."""
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
        """SPEC §"bc.predict — dispatch by argument": both default None."""
        sig = inspect.signature(BrainCollection.predict)
        assert sig.parameters["y"].default is None
        assert sig.parameters["X_new"].default is None

    def test_default_spatial_scale_is_whole_brain(self):
        sig = inspect.signature(BrainCollection.predict)
        assert sig.parameters["spatial_scale"].default == "whole_brain"

    def test_default_model_is_svm_default_cv_loso(self):
        sig = inspect.signature(BrainCollection.predict)
        assert sig.parameters["model"].default == "svm"
        assert sig.parameters["cv"].default == "loso"


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

        from nltools.data import BrainData, BrainCollection, DesignMatrix

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
    """SPEC §1021: dispatch is by argument, not by item state."""

    def test_both_args_raises(self, bc_inmem):
        with pytest.raises(ValueError):
            bc_inmem.predict(y=[0, 1, 0], X_new=np.zeros((5, 3)))

    def test_neither_arg_raises(self, bc_inmem):
        with pytest.raises(ValueError):
            bc_inmem.predict()

    def test_predict_y_requires_single_map_per_subject(self, bc_inmem):
        """Multi-row items raise. (bc_inmem items are (8, 27).)"""
        with pytest.raises(ValueError, match="single-map-per-subject"):
            bc_inmem.predict(y=np.array([0, 1, 0]))

    @pytest.mark.xfail(
        reason="BD.predict on (3, 27) data is too tiny for sklearn estimators",
        strict=False,
    )
    def test_y_only_returns_braindata_with_cv_attrs(
        self,
        tiny_mask,
        tiny_brain_factory,
    ):
        # Use single-map-per-subject items (1, 27).
        from nltools.data import BrainData

        np.random.seed(0)
        single_maps = [
            BrainData(np.random.randn(1, 27).astype(np.float32), mask=tiny_mask)
            for _ in range(6)
        ]
        bc = BrainCollection(single_maps, mask=tiny_mask, lazy=False, cache_dir=None)
        out = bc.predict(y=np.array([0, 1, 0, 1, 0, 1]))
        assert isinstance(out, BrainData)
        assert hasattr(out, "cv_scores")

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


class TestCVPipeline:
    def test_cv_returns_pipeline(self, bc_inmem):
        pipe = bc_inmem.cv(method="loso")
        assert isinstance(pipe, BrainCollectionPipeline)

    def test_pipeline_n_subjects_and_repr(self, bc_inmem):
        """F067: n_subjects referenced a nonexistent BrainCollection.n_images."""
        pipe = bc_inmem.cv(method="loso")
        assert pipe.n_subjects == bc_inmem.n_subjects
        assert repr(pipe)  # __repr__ interpolates n_subjects; must not raise

    def test_pipeline_standardize_step_added(self, bc_inmem):
        """SPEC §865: BrainCollectionPipeline.standardize() (renamed from normalize)."""
        assert bc_inmem.cv(method="loso").standardize().n_steps == 1
        # And confirm the old name is gone from the public surface.
        assert not hasattr(bc_inmem.cv(method="loso"), "normalize")

    def test_cv_predict_returns_braindata(self, bc_inmem):
        """SPEC §942: bc.cv(...).predict(...) returns the same BrainData type."""
        out = bc_inmem.cv(method="loso").predict(y=np.array([0, 1, 0]))
        assert isinstance(out, BrainData)
        assert hasattr(out, "cv_scores")
        assert hasattr(out, "mean_score")
        n_folds = bc_inmem.n_subjects
        assert out.cv_scores.shape == (n_folds,)
        # cv_predictions is (n_total_obs, n_voxels); n_voxels == 27 from tiny_mask.
        assert out.cv_predictions.ndim == 2
        assert out.cv_predictions.shape[1] == 27
        assert isinstance(out.mean_score, float)

    def test_pipeline_predict_braindata_carries_full_lineage(self, bc_inmem):
        """fold_results must preserve test_idx/train_idx/predictions for inspection."""
        out = bc_inmem.cv(method="loso").predict(y=np.array([0, 1, 0]))
        assert isinstance(out.fold_results, list)
        assert len(out.fold_results) > 0
        keys = out.fold_results[0]
        for required in ("test_idx", "train_idx", "predictions"):
            assert required in keys

    def test_predict_no_permutation_null_by_default(self, bc_inmem):
        """Without n_permute, no null is computed (default n_permute=0)."""
        out = bc_inmem.cv(method="loso").predict(y=np.array([0, 1, 0]))
        assert not hasattr(out, "permutation_scores")
        assert not hasattr(out, "permutation_pvalue")

    def test_predict_permutation_null_attached(self, bc_inmem):
        """Thread #87 (F112): predict(n_permute=) builds a label-permutation null.

        The dedicated outer loop shuffles y, re-runs the *normal* CV, and
        collects the mean score into a null distribution — the classic MVPA
        permutation test that the removed 'permutation' CV scheme mishandled.
        """
        out = bc_inmem.cv(method="loso").predict(
            y=np.array([0, 1, 0]), n_permute=20, random_state=0
        )
        # Observed CV result is still present and unchanged in shape.
        assert isinstance(out, BrainData)
        assert hasattr(out, "mean_score")
        # Null distribution + p-value attached.
        assert out.permutation_scores.shape == (20,)
        assert 0.0 <= out.permutation_pvalue <= 1.0

    def test_predict_permutation_null_reproducible(self, bc_inmem):
        """Same random_state → identical null and p-value."""
        a = bc_inmem.cv(method="loso").predict(
            y=np.array([0, 1, 0]), n_permute=15, random_state=7
        )
        b = bc_inmem.cv(method="loso").predict(
            y=np.array([0, 1, 0]), n_permute=15, random_state=7
        )
        np.testing.assert_array_equal(a.permutation_scores, b.permutation_scores)
        assert a.permutation_pvalue == b.permutation_pvalue
