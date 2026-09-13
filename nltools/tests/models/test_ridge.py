"""Contract tests for `nltools.models.Ridge` and its Himalaya adapter.

Himalaya owns the numerics, so these tests pin the boundary: argument names and
validation, named feature-space alignment, device and memory policy, fitted
state, and parity with Himalaya 0.4.11's own selection rules.
"""

import importlib.util

import numpy as np
import pytest

from nltools.models import _Ridge
from nltools.models.ridge import (
    _prepare_feature_space_weights,
    _refit_fixed_hyperparameters,
)


def mps_available():
    """True when this machine can run the Himalaya `torch_mps` backend."""
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def gpu_available():
    """True when PyTorch can use CUDA or MPS."""
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available() or mps_available()


def torch_available():
    """True when PyTorch is importable, so a non-numpy Himalaya backend exists."""
    return importlib.util.find_spec("torch") is not None


def foreign_backend_name():
    """The strongest non-numpy Himalaya backend this host can activate."""
    if mps_available():
        return "torch_mps"
    import torch

    return "torch_cuda" if torch.cuda.is_available() else "torch"


def current_himalaya_backend():
    """Name of Himalaya's currently active process-global backend."""
    from himalaya.backend import get_backend

    return get_backend().name


requires_mps = pytest.mark.skipif(not mps_available(), reason="MPS not available")
requires_gpu = pytest.mark.skipif(not gpu_available(), reason="no GPU accelerator")
requires_torch = pytest.mark.skipif(not torch_available(), reason="torch not installed")

ALPHAS = [0.1, 1.0, 10.0, 100.0]


@pytest.fixture()
def foreign_ambient_backend():
    """Leave a non-numpy backend globally active for the body of a test.

    Ridge must neither depend on nor disturb whatever backend unrelated code
    left active, so the scoping tests have to start somewhere other than the
    `numpy` default.
    """
    from himalaya.backend import set_backend

    previous = current_himalaya_backend()
    name = foreign_backend_name()
    set_backend(name, on_error="raise")
    try:
        yield name
    finally:
        set_backend(previous, on_error="raise")


def make_data(n_samples=80, n_features=12, n_targets=4, seed=0, dtype=np.float64):
    """Return a well-conditioned `(X, Y)` pair with a known linear signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    beta = rng.standard_normal((n_features, n_targets))
    Y = X @ beta + 0.5 * rng.standard_normal((n_samples, n_targets))
    return X.astype(dtype), Y.astype(dtype)


def make_spaces(n_samples=80, sizes=(6, 9), n_targets=3, seed=1, dtype=np.float64):
    """Return a named banded design and its targets."""
    rng = np.random.default_rng(seed)
    spaces = {
        f"space{index}": rng.standard_normal((n_samples, size)).astype(dtype)
        for index, size in enumerate(sizes)
    }
    stacked = np.concatenate(list(spaces.values()), axis=1)
    beta = rng.standard_normal((stacked.shape[1], n_targets))
    Y = stacked @ beta + 0.5 * rng.standard_normal((n_samples, n_targets))
    return spaces, Y.astype(dtype)


def kfold(n_splits=4):
    """An unshuffled K-fold splitter, the shape Ridge builds for `cv=int`."""
    from sklearn.model_selection import KFold

    return KFold(n_splits=n_splits, shuffle=False)


# --------------------------------------------------------------------- signature


class TestPublicSignature:
    """Only the specified keywords exist, and removed names raise TypeError."""

    def test_every_argument_is_keyword_only(self):
        import inspect

        parameters = inspect.signature(_Ridge.__init__).parameters
        positional = [
            name
            for name, param in parameters.items()
            if name != "self" and param.kind is not param.KEYWORD_ONLY
        ]
        assert positional == []

    def test_keyword_names_and_defaults(self):
        import inspect

        parameters = inspect.signature(_Ridge.__init__).parameters
        defaults = {
            name: param.default for name, param in parameters.items() if name != "self"
        }
        assert defaults == {
            "alpha": 1.0,
            "cv": None,
            "search_iterations": 100,
            "dirichlet_concentration": (0.1, 1.0),
            "device": "cpu",
            "memory_budget_gb": None,
            "per_target_alpha": True,
            "prefer_conservative_alpha": False,
            "random_state": None,
            "progress_bar": False,
        }

    def test_fit_returns_self(self):
        X, Y = make_data()
        model = _Ridge(alpha=1.0)
        assert model.fit(X, Y) is model


# -------------------------------------------------------------------- validation


class TestValidation:
    """Every rejection names the conflicting argument or mismatched dimension."""

    def test_alpha_auto_is_invalid(self):
        with pytest.raises(ValueError, match="alpha='auto' is not supported"):
            _Ridge(alpha="auto", cv=5)

    def test_scalar_alpha_with_cv_raises(self):
        with pytest.raises(ValueError, match="requires cv=None"):
            _Ridge(alpha=1.0, cv=5)

    def test_sequence_alpha_without_cv_raises(self):
        with pytest.raises(ValueError, match="requires cv"):
            _Ridge(alpha=ALPHAS)

    def test_non_positive_alpha_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _Ridge(alpha=[0.0, 1.0], cv=5)

    def test_multidimensional_alpha_raises(self):
        with pytest.raises(ValueError, match="1D collection"):
            _Ridge(alpha=[[1.0, 2.0]], cv=5)

    def test_unsupported_device_raises(self):
        with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
            _Ridge(device="auto")

    def test_empty_banded_mapping_raises(self):
        with pytest.raises(ValueError, match="empty mapping"):
            _Ridge(alpha=ALPHAS, cv=3).fit({}, np.zeros(10))

    def test_banded_sample_count_mismatch_raises(self):
        spaces, Y = make_spaces()
        spaces = dict(spaces)
        spaces["space0"] = spaces["space0"][:-4]
        with pytest.raises(ValueError, match="same number of\n?\\s*samples"):
            _Ridge(alpha=ALPHAS, cv=3).fit(spaces, Y)

    def test_banded_with_scalar_alpha_raises(self):
        spaces, Y = make_spaces()
        with pytest.raises(ValueError, match="banded Ridge needs a sequence"):
            _Ridge(alpha=1.0).fit(spaces, Y)

    def test_predict_before_fit_raises(self):
        X, _ = make_data()
        with pytest.raises(ValueError, match="not fitted"):
            _Ridge(alpha=1.0).predict(X)


# ------------------------------------------------------- cross-validator handling


# ------------------------------------------------------------------ fitted state


class TestFittedState:
    """Attribute presence, shapes, and the attributes that must not exist."""

    def test_ordinary_fixed_alpha_state(self):
        X, Y = make_data()
        model = _Ridge(alpha=2.0).fit(X, Y)
        assert model.coef_.shape == (X.shape[1], Y.shape[1])
        assert model.alpha_ == 2.0
        assert model.cv_scores_ is None
        assert model.feature_space_weights_ is None
        assert model.feature_space_names_ is None
        assert model.feature_space_sizes_ is None
        assert model.n_samples_ == X.shape[0]
        assert model.n_features_in_ == X.shape[1]
        assert model.is_fitted_ is True
        assert model.backend_.name == "numpy"

    def test_ordinary_cv_state(self):
        X, Y = make_data()
        model = _Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        assert model.coef_.shape == (X.shape[1], Y.shape[1])
        assert model.alpha_.shape == (Y.shape[1],)
        assert set(np.unique(model.alpha_)) <= set(ALPHAS)
        assert model.cv_scores_.shape == (Y.shape[1],)
        assert model.feature_space_weights_ is None

    def test_shared_alpha_is_scalar(self):
        X, Y = make_data()
        model = _Ridge(alpha=ALPHAS, cv=kfold(), per_target_alpha=False).fit(X, Y)
        assert isinstance(model.alpha_, float)
        assert model.alpha_ in ALPHAS

    def test_one_dimensional_target_squeezes_every_attribute(self):
        X, Y = make_data()
        y = Y[:, 0]
        model = _Ridge(alpha=ALPHAS, cv=kfold()).fit(X, y)
        assert model.coef_.shape == (X.shape[1],)
        assert isinstance(model.alpha_, float)
        assert isinstance(model.cv_scores_, float)
        assert model.predict(X).shape == (X.shape[0],)
        assert isinstance(model.score(X, y), float)

    def test_banded_state(self):
        spaces, Y = make_spaces()
        model = _Ridge(
            alpha=ALPHAS, cv=kfold(), search_iterations=8, random_state=3
        ).fit(spaces, Y)
        n_features = sum(space.shape[1] for space in spaces.values())
        assert model.coef_.shape == (n_features, Y.shape[1])
        assert model.feature_space_names_ == tuple(spaces)
        assert model.feature_space_sizes_ == tuple(
            space.shape[1] for space in spaces.values()
        )
        assert model.feature_space_weights_.shape == (len(spaces), Y.shape[1])
        assert model.cv_scores_.shape == (8, Y.shape[1])
        assert model.alpha_.shape == (Y.shape[1],)
        assert model.n_features_in_ == n_features

    def test_fitted_arrays_are_cpu_numpy(self):
        X, Y = make_data()
        model = _Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        for value in (model.coef_, model.alpha_, model.cv_scores_):
            assert isinstance(value, np.ndarray)


# ---------------------------------------------------------------------- numerics


class TestNumericalBehavior:
    """`score` is a per-target R², with nltools' own constant-target rule."""

    def test_score_is_per_target_r2(self):
        X, Y = make_data()
        model = _Ridge(alpha=1.0).fit(X, Y)
        scores = model.score(X, Y)
        assert scores.shape == (Y.shape[1],)
        predictions = model.predict(X)
        expected = 1 - ((Y - predictions) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        np.testing.assert_allclose(scores, expected)


# ------------------------------------------------------------------ himalaya parity


# ---------------------------------------------------------- candidate preparation


class TestFeatureSpaceWeightCandidates:
    """Validation, dtype conversion, and the underflow floor."""

    def test_values_below_tiny_are_raised_to_tiny_float64(self):
        tiny = np.finfo(np.float64).tiny
        candidates = np.array([[1e-320, 1.0 - 1e-320]])
        prepared = _prepare_feature_space_weights(candidates, np.float64)
        assert prepared[0, 0] == tiny

    def test_zero_weight_is_rejected_not_floored(self):
        with pytest.raises(ValueError, match="strictly positive"):
            _prepare_feature_space_weights(np.array([[0.0, 1.0]]), np.float32)

    def test_negative_weight_is_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            _prepare_feature_space_weights(np.array([[-0.5, 1.5]]), np.float32)

    def test_rows_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to one"):
            _prepare_feature_space_weights(np.array([[0.5, 0.9]]), np.float32)


# ------------------------------------------------------------------- banded predict


class TestBandedPrediction:
    """Named spaces are aligned by name; anything else is an error."""

    @pytest.fixture()
    def fitted(self):
        spaces, Y = make_spaces()
        model = _Ridge(
            alpha=ALPHAS, cv=kfold(), search_iterations=6, random_state=4
        ).fit(spaces, Y)
        return model, spaces, Y

    def test_reordered_mapping_predicts_identically(self, fitted):
        model, spaces, _ = fitted
        reordered = {name: spaces[name] for name in reversed(list(spaces))}
        np.testing.assert_allclose(model.predict(reordered), model.predict(spaces))

    def test_missing_space_raises(self, fitted):
        model, spaces, _ = fitted
        partial = dict(list(spaces.items())[:1])
        with pytest.raises(ValueError, match="missing"):
            model.predict(partial)

    def test_prediction_equals_concatenated_design_times_coef(self, fitted):
        model, spaces, _ = fitted
        design = np.concatenate(
            [spaces[name] for name in model.feature_space_names_], axis=1
        )
        np.testing.assert_allclose(model.predict(spaces), design @ model.coef_)


# ---------------------------------------------------------------- fixed refitting


class TestFixedHyperparameterRefit:
    """One shared refit path, equal to the generalized ridge system it defines."""

    def _generalized_ridge(self, spaces, y, alpha, gamma):
        design = np.concatenate(spaces, axis=1)
        sizes = [space.shape[1] for space in spaces]
        penalty = np.concatenate(
            [np.full(size, alpha / g) for size, g in zip(sizes, gamma)]
        )
        return np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ y)

    def test_scalar_alpha_without_weights_matches_direct_solve(self):
        X, Y = make_data()
        coef = _refit_fixed_hyperparameters([X], Y, 2.0)
        expected = np.linalg.solve(X.T @ X + 2.0 * np.eye(X.shape[1]), X.T @ Y)
        np.testing.assert_allclose(coef, expected, rtol=1e-8, atol=1e-10)

    def test_per_target_alpha_matches_direct_solve(self):
        X, Y = make_data(n_targets=3)
        alphas = np.array([0.5, 5.0, 50.0])
        coef = _refit_fixed_hyperparameters([X], Y, alphas)
        for index, alpha in enumerate(alphas):
            expected = np.linalg.solve(
                X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y[:, index]
            )
            np.testing.assert_allclose(coef[:, index], expected, rtol=1e-8, atol=1e-10)

    def test_shared_gamma_equals_generalized_ridge(self):
        spaces, Y = make_spaces(sizes=(4, 6), n_targets=2)
        matrices = list(spaces.values())
        gamma = np.array([0.25, 0.75])
        coef = _refit_fixed_hyperparameters(
            matrices, Y, 3.0, feature_space_weights=gamma
        )
        expected = self._generalized_ridge(matrices, Y, 3.0, gamma)
        np.testing.assert_allclose(coef, expected, rtol=1e-6, atol=1e-8)

    def test_refit_does_not_mutate_the_feature_spaces(self):
        spaces, Y = make_spaces(sizes=(3, 5))
        matrices = list(spaces.values())
        snapshots = [matrix.copy() for matrix in matrices]
        _refit_fixed_hyperparameters(
            matrices, Y, 1.0, feature_space_weights=np.array([0.4, 0.6])
        )
        for matrix, snapshot in zip(matrices, snapshots):
            np.testing.assert_array_equal(matrix, snapshot)

    def test_banded_fit_coefficients_match_a_fixed_refit(self):
        spaces, Y = make_spaces(sizes=(4, 6))
        model = _Ridge(
            alpha=ALPHAS, cv=kfold(), search_iterations=8, random_state=13
        ).fit(spaces, Y)
        refit = _refit_fixed_hyperparameters(
            list(spaces.values()),
            Y,
            model.alpha_,
            feature_space_weights=model.feature_space_weights_,
        )
        np.testing.assert_allclose(model.coef_, refit, rtol=1e-5, atol=1e-7)


# -------------------------------------------------------------- device and memory


class TestSerialization:
    """A fitted model survives pickling, deepcopy, and process-based workers."""

    def test_fitted_model_round_trips_through_pickle(self):
        import pickle

        X, Y = make_data()
        model = _Ridge(alpha=1.0).fit(X, Y)

        restored = pickle.loads(pickle.dumps(model))

        assert restored.is_fitted_
        assert restored.backend_.name == model.backend_.name
        assert restored.backend_.device == model.backend_.device
        np.testing.assert_array_equal(restored.coef_, model.coef_)
        np.testing.assert_allclose(restored.predict(X), model.predict(X))


def _predict_in_worker(model, X):
    """Module-level so `loky` can pickle it alongside the fitted model."""
    return model.predict(X)


class TestBackendScoping:
    """Himalaya's process-global backend is set for the call and then restored."""

    @requires_torch
    def test_backend_restored_after_a_successful_fit(self, foreign_ambient_backend):
        X, Y = make_data()
        _Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        assert current_himalaya_backend() == foreign_ambient_backend

    @requires_torch
    def test_backend_restored_when_the_solver_raises(
        self, foreign_ambient_backend, monkeypatch
    ):
        import himalaya.ridge

        def explode(*args, **kwargs):
            raise RuntimeError("solver exploded")

        monkeypatch.setattr(himalaya.ridge, "solve_ridge_cv_svd", explode)
        X, Y = make_data()
        with pytest.raises(RuntimeError, match="solver exploded"):
            _Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        assert current_himalaya_backend() == foreign_ambient_backend


class TestDeviceAndMemory:
    """Explicit GPU runs or raises; the budget only sizes internal batches."""

    def test_explicit_gpu_raises_without_an_accelerator(self, monkeypatch):
        from nltools.algorithms import backends as backends_mod

        def _cpu_only(self):
            self.name = "torch-cpu"
            self.device = "cpu"
            self.xp = object()
            self._torch_device = "cpu"

        monkeypatch.setattr(backends_mod._Backend, "_init_torch", _cpu_only)
        X, Y = make_data()
        with pytest.raises(RuntimeError, match="no GPU accelerator"):
            _Ridge(alpha=1.0, device="gpu").fit(X, Y)

    def test_over_tight_budget_names_the_argument(self):
        X, Y = make_data()
        model = _Ridge(alpha=ALPHAS, cv=kfold(), memory_budget_gb=1e-9)
        with pytest.raises(ValueError, match="memory_budget_gb"):
            model.fit(X, Y)

    @requires_gpu
    def test_cpu_gpu_parity_cross_validated(self):
        X, Y = make_data(dtype=np.float32)
        cpu = _Ridge(alpha=ALPHAS, cv=kfold(), device="cpu").fit(X, Y)
        gpu = _Ridge(alpha=ALPHAS, cv=kfold(), device="gpu").fit(X, Y)
        np.testing.assert_array_equal(gpu.alpha_, cpu.alpha_)
        np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=1e-2, atol=1e-3)
