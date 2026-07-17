"""F182: Glm.predict(X) new-design prediction + coef_ parity with Ridge.

Betas are recovered from nilearn's run_glm results (``labels_``/``results_``
theta), cached as ``coef_``, and ``predict(X) == X @ coef_`` mirrors Ridge. The
old surface — a documented ``X`` that always raised NotImplementedError — is
gone.
"""

import inspect

import numpy as np
import pytest

from nltools.models import Glm

pytestmark = pytest.mark.slow


class TestGlmCoefAndPredict:
    def test_coef_shape_and_theta_parity(self, fitted_glm_single_run):
        """coef_ is (n_reg, n_vox) and each row is that regressor's beta map."""
        model, _ = fitted_glm_single_run
        n_reg = model.design_matrices_[0].shape[1]
        assert model.coef_.ndim == 2
        assert model.coef_.shape[0] == n_reg

        # coef row 0 == the identity-contrast effect size for regressor 0
        beta0_img = model.compute_contrast(np.eye(n_reg)[0], output_type="effect_size")
        remasked = model._glm.masker_.transform(beta0_img).ravel()
        np.testing.assert_allclose(model.coef_[0], remasked, atol=1e-4)

    def test_predict_new_X_equals_X_at_coef(self, fitted_glm_single_run):
        """predict(X) returns X @ coef_ as a 2-D ndarray (Ridge parity)."""
        model, _ = fitted_glm_single_run
        n_reg = model.coef_.shape[0]
        rng = np.random.RandomState(0)
        Xnew = rng.randn(8, n_reg)

        pred = model.predict(Xnew)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (8, model.coef_.shape[1])
        np.testing.assert_allclose(pred, Xnew @ model.coef_)

    def test_predict_none_returns_fitted_values(self, fitted_glm_single_run):
        """predict() with no args still returns training fitted values."""
        model, _ = fitted_glm_single_run
        assert isinstance(model.predict(), list)

    def test_predict_wrong_width_raises(self, fitted_glm_single_run):
        model, _ = fitted_glm_single_run
        with pytest.raises(ValueError, match="regressors"):
            model.predict(np.zeros((4, model.coef_.shape[0] + 1)))

    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError):
            Glm().predict(np.zeros((3, 2)))

    def test_x_still_accepted_for_base_contract(self):
        assert "X" in inspect.signature(Glm.predict).parameters

    def test_predict_docstring_no_longer_advertises_a_lie(self):
        doc = Glm.predict.__doc__ or ""
        assert "not supported" not in doc.lower()
        assert "X @ coef_" in doc


class TestGlmReport:
    def test_report_returns_html(self, fitted_glm_single_run):
        """Glm.report delegates to nilearn generate_report -> HTMLReport."""
        model, _ = fitted_glm_single_run
        rep = model.report(contrasts={"reg0": np.eye(model.coef_.shape[0])[0]})
        assert type(rep).__name__ == "HTMLReport"
