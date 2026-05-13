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
    @XFAIL
    def test_fit_glm_returns_pathbacked_collection(self, bc_pathbacked):
        # Each item needs a paired DesignMatrix; bc_pathbacked has none — this
        # path will need from_bids or explicit designs in the real test.
        # Placeholder: assert returned type only.
        out = bc_pathbacked.fit(model="glm", X=None)
        assert isinstance(out, BrainCollection)

    @XFAIL
    def test_fit_x_callable_receives_design_context(self, bc_pathbacked):
        seen: list = []

        def make_design(ctx):
            seen.append(ctx)
            from nltools.data import DesignMatrix

            return DesignMatrix(np.zeros((1, 1)))

        bc_pathbacked.fit(model="glm", X=make_design)
        assert len(seen) == bc_pathbacked.n_subjects
        # ctx exposes named attributes
        assert hasattr(seen[0], "TR")
        assert hasattr(seen[0], "subject")


class TestComputeContrastsBehavior:
    @XFAIL
    def test_single_contrast_returns_collection(self, bc_inmem):
        # Assumes bc_inmem holds fit bundles; placeholder for now.
        out = bc_inmem.compute_contrasts("a - b", contrast_type="beta")
        assert isinstance(out, BrainCollection)

    @XFAIL
    def test_multiple_contrasts_returns_dict(self, bc_inmem):
        out = bc_inmem.compute_contrasts(["a", "b"], contrast_type="beta")
        assert isinstance(out, dict)
        assert all(isinstance(v, BrainCollection) for v in out.values())

    @XFAIL
    def test_all_contrast_types_returns_dict_keyed_by_type(self, bc_inmem):
        out = bc_inmem.compute_contrasts("a", contrast_type="all")
        assert isinstance(out, dict)
        for k in ("beta", "t", "z", "p", "se"):
            assert k in out


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
