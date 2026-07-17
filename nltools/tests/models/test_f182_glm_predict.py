"""F182: Glm.predict(X=...) must not promise new-design prediction it lacks.

These are deliberately fast (no nilearn fit): they pin the *surface* — the
docstring must not advertise a working X path, and a non-None X must fail
loudly with an actionable message rather than returning a wrong map.

Unlike F068 (roi_mask silently ignored), this path already raised; the defect
was a docstring that read as though X were supported. Implementing real
new-design prediction is a deferred feature — see the raise site in
nltools/models/glm.py for why (no nilearn betas + ambiguous multi-run X).
"""

import pytest

from nltools.models import Glm


@pytest.fixture
def fake_fitted_glm():
    """A Glm marked fitted without a real fit — enough to reach the X branch."""
    model = Glm()
    model.is_fitted_ = True
    return model


class TestGlmPredictRejectsX:
    def test_non_none_x_raises_not_implemented(self, fake_fitted_glm):
        import pandas as pd

        X = pd.DataFrame({"cond": [1.0, 0.0, 1.0]})
        with pytest.raises(NotImplementedError, match="new design matrix"):
            fake_fitted_glm.predict(X)

    def test_raise_message_points_to_a_working_alternative(self, fake_fitted_glm):
        with pytest.raises(NotImplementedError) as exc:
            fake_fitted_glm.predict(X=[[1.0]])
        msg = str(exc.value)
        assert "predict() with no arguments" in msg
        assert "compute_contrast" in msg

    def test_docstring_does_not_advertise_new_design_prediction(self):
        doc = Glm.predict.__doc__ or ""
        assert "Raises:" in doc, "the NotImplementedError must be documented"
        assert "Not supported" in doc
        # The old docstring promised "generates predictions using new design
        # matrix", which is exactly the lie F182 flagged.
        assert "generates predictions using new design matrix" not in doc

    def test_x_still_accepted_for_base_contract(self):
        """BaseModel.predict(X) is abstract with X required — Glm must keep it."""
        import inspect

        assert "X" in inspect.signature(Glm.predict).parameters
