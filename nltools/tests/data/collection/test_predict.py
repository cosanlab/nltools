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
        assert sig.parameters["contrast_type"].default == "beta"

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

    def test_default_estimator_is_svm_default_cv_loso(self):
        sig = inspect.signature(BrainCollection.predict)
        assert sig.parameters["estimator"].default == "svm"
        assert sig.parameters["cv"].default == "loso"

    def test_return_weights_default_true(self):
        sig = inspect.signature(BrainCollection.predict)
        assert sig.parameters["return_weights"].default is True


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
    @pytest.fixture(scope="function")
    def fitted_bc(self, bc_with_designs):
        return bc_with_designs.fit(model="glm", cache=True, n_jobs=1)

    def test_single_contrast_returns_collection(self, fitted_bc):
        out = fitted_bc.compute_contrasts("a - b", contrast_type="beta", n_jobs=1)
        assert isinstance(out, BrainCollection)
        assert out.n_subjects == fitted_bc.n_subjects

    def test_single_regressor_identity_contrast(self, fitted_bc):
        out = fitted_bc.compute_contrasts("a", contrast_type="beta", n_jobs=1)
        assert isinstance(out, BrainCollection)

    def test_multiple_contrasts_returns_dict(self, fitted_bc):
        out = fitted_bc.compute_contrasts(
            {"main": "a - b", "avg": "a + b"},
            contrast_type="beta",
            n_jobs=1,
        )
        assert isinstance(out, dict)
        assert set(out.keys()) == {"main", "avg"}
        assert all(isinstance(v, BrainCollection) for v in out.values())

    def test_all_contrast_types_returns_dict_keyed_by_type(self, fitted_bc):
        out = fitted_bc.compute_contrasts("a", contrast_type="all", n_jobs=1)
        assert isinstance(out, dict)
        for k in ("beta", "t", "z", "p", "se"):
            assert k in out
            assert isinstance(out[k], BrainCollection)

    def test_contrast_writes_lineage_sidecar(self, fitted_bc):
        import json

        out = fitted_bc.compute_contrasts("a", contrast_type="beta", n_jobs=1)
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
        with pytest.raises(ValueError, match="contrast_type"):
            fitted_bc.compute_contrasts("a", contrast_type="bogus", n_jobs=1)


class TestPredictDispatch:
    """SPEC §1021: dispatch is by argument, not by item state."""

    @XFAIL
    def test_y_only_returns_braindata_with_cv_attrs(self, bc_inmem):
        out = bc_inmem.predict(y=np.array([0, 1, 0]))
        assert isinstance(out, BrainData)
        assert hasattr(out, "cv_scores")
        assert hasattr(out, "cv_predictions")

    @XFAIL
    def test_x_new_only_returns_collection(self, bc_inmem):
        out = bc_inmem.predict(X_new=np.zeros((5, 3)))
        assert isinstance(out, BrainCollection)

    @XFAIL
    def test_both_args_raises(self, bc_inmem):
        with pytest.raises(ValueError):
            bc_inmem.predict(y=[0, 1, 0], X_new=np.zeros((5, 3)))

    @XFAIL
    def test_neither_arg_raises(self, bc_inmem):
        with pytest.raises(ValueError):
            bc_inmem.predict()

    @XFAIL
    def test_predict_y_requires_single_map_per_subject(self, bc_pathbacked):
        """SPEC §902: multi-row items must call compute_contrasts first."""
        with pytest.raises(ValueError, match="compute_contrasts"):
            bc_pathbacked.predict(y=np.array([0, 1, 0]))


class TestCVPipeline:
    @XFAIL
    def test_cv_returns_pipeline(self, bc_inmem):
        pipe = bc_inmem.cv(method="loso")
        assert isinstance(pipe, BrainCollectionPipeline)

    @XFAIL
    def test_cv_predict_returns_braindata(self, bc_inmem):
        """SPEC §942: bc.cv(...).predict(...) returns the same BrainData type."""
        out = bc_inmem.cv(method="loso").predict(y=np.array([0, 1, 0]))
        assert isinstance(out, BrainData)
        assert hasattr(out, "cv_scores")
