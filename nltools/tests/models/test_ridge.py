"""Contract tests for `nltools.models.Ridge` and its Himalaya adapter.

Himalaya owns the numerics, so these tests pin the boundary: argument names and
validation, named feature-space alignment, device and memory policy, fitted
state, and parity with Himalaya 0.4.11's own selection rules.
"""

import importlib.util

import numpy as np
import pytest

from nltools.models import Ridge
from nltools.models.ridge import (
    _himalaya_backend_name,
    _prepare_feature_space_weights,
    _refit_fixed_hyperparameters,
    _scoped_himalaya_backend,
    _working_dtype,
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

    @pytest.mark.parametrize(
        "removed",
        [
            "alphas",
            "n_iter",
            "concentration",
            "local_alpha",
            "conservative",
            "fit_intercept",
            "backend",
            "solver_params",
            "max_gpu_memory_gb",
            "parallel",
        ],
    )
    def test_removed_keyword_raises_type_error(self, removed):
        with pytest.raises(TypeError):
            Ridge(**{removed: 1})

    def test_every_argument_is_keyword_only(self):
        import inspect

        parameters = inspect.signature(Ridge.__init__).parameters
        positional = [
            name
            for name, param in parameters.items()
            if name != "self" and param.kind is not param.KEYWORD_ONLY
        ]
        assert positional == []

    def test_keyword_names_and_defaults(self):
        import inspect

        parameters = inspect.signature(Ridge.__init__).parameters
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
        model = Ridge(alpha=1.0)
        assert model.fit(X, Y) is model


# -------------------------------------------------------------------- validation


class TestValidation:
    """Every rejection names the conflicting argument or mismatched dimension."""

    def test_alpha_auto_is_invalid(self):
        with pytest.raises(ValueError, match="alpha='auto' is not supported"):
            Ridge(alpha="auto", cv=5)

    def test_scalar_alpha_with_cv_raises(self):
        with pytest.raises(ValueError, match="requires cv=None"):
            Ridge(alpha=1.0, cv=5)

    def test_sequence_alpha_without_cv_raises(self):
        with pytest.raises(ValueError, match="requires cv"):
            Ridge(alpha=ALPHAS)

    def test_empty_alpha_collection_raises(self):
        with pytest.raises(ValueError, match="empty collection"):
            Ridge(alpha=[], cv=5)

    def test_non_finite_alpha_raises(self):
        with pytest.raises(ValueError, match="finite"):
            Ridge(alpha=[1.0, np.inf], cv=5)

    def test_non_positive_alpha_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Ridge(alpha=[0.0, 1.0], cv=5)

    def test_multidimensional_alpha_raises(self):
        with pytest.raises(ValueError, match="1D collection"):
            Ridge(alpha=[[1.0, 2.0]], cv=5)

    def test_generator_cv_rejected(self):
        X, _ = make_data()
        with pytest.raises(TypeError, match="single-use generator"):
            Ridge(alpha=ALPHAS, cv=kfold().split(X))

    def test_unsupported_cv_type_raises(self):
        with pytest.raises(ValueError, match="cv must be None"):
            Ridge(alpha=ALPHAS, cv="loo")

    def test_unsupported_device_raises(self):
        with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
            Ridge(device="auto")

    def test_non_positive_memory_budget_raises(self):
        with pytest.raises(ValueError, match="memory_budget_gb must be positive"):
            Ridge(memory_budget_gb=0)

    def test_non_finite_memory_budget_raises(self):
        with pytest.raises(ValueError, match="memory_budget_gb must be positive"):
            Ridge(memory_budget_gb=float("nan"))

    def test_conservative_requires_per_target_alpha(self):
        with pytest.raises(ValueError, match="prefer_conservative_alpha=True"):
            Ridge(prefer_conservative_alpha=True, per_target_alpha=False)

    def test_three_dimensional_y_raises(self):
        X, _ = make_data()
        with pytest.raises(ValueError, match="y must be 1D or 2D"):
            Ridge(alpha=1.0).fit(X, np.zeros((X.shape[0], 2, 2)))

    def test_sample_count_mismatch_raises(self):
        X, Y = make_data()
        with pytest.raises(ValueError, match="inconsistent sample counts"):
            Ridge(alpha=1.0).fit(X, Y[:-3])

    def test_one_dimensional_X_raises(self):
        with pytest.raises(ValueError, match="2D feature matrix"):
            Ridge(alpha=1.0).fit(np.zeros(10), np.zeros(10))

    def test_empty_banded_mapping_raises(self):
        with pytest.raises(ValueError, match="empty mapping"):
            Ridge(alpha=ALPHAS, cv=3).fit({}, np.zeros(10))

    def test_non_string_feature_space_name_raises(self):
        _, Y = make_spaces()
        with pytest.raises(ValueError, match="names must be strings"):
            Ridge(alpha=ALPHAS, cv=3).fit({0: np.zeros((Y.shape[0], 2))}, Y)

    def test_banded_sample_count_mismatch_raises(self):
        spaces, Y = make_spaces()
        spaces = dict(spaces)
        spaces["space0"] = spaces["space0"][:-4]
        with pytest.raises(ValueError, match="same number of\n?\\s*samples"):
            Ridge(alpha=ALPHAS, cv=3).fit(spaces, Y)

    def test_banded_with_scalar_alpha_raises(self):
        spaces, Y = make_spaces()
        with pytest.raises(ValueError, match="banded Ridge needs a sequence"):
            Ridge(alpha=1.0).fit(spaces, Y)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"search_iterations": 7},
            {"dirichlet_concentration": 1.0},
        ],
    )
    def test_banded_only_arguments_rejected_for_ordinary_fit(self, kwargs):
        X, Y = make_data()
        model = Ridge(alpha=ALPHAS, cv=3, **kwargs)
        with pytest.raises(ValueError, match="only applies to banded Ridge"):
            model.fit(X, Y)

    def test_random_state_is_accepted_and_unused_for_ordinary_fits(self):
        """`BrainData.fit` forwards one shared `random_state` to both estimators.

        Ordinary Ridge has no randomness of its own, so it takes the keyword and
        ignores it rather than rejecting a universally accepted argument.
        """
        X, Y = make_data()
        seeded = Ridge(alpha=ALPHAS, cv=kfold(), random_state=0).fit(X, Y)
        unseeded = Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        np.testing.assert_array_equal(seeded.coef_, unseeded.coef_)
        np.testing.assert_array_equal(seeded.alpha_, unseeded.alpha_)

    def test_zero_target_y_raises(self):
        X, _ = make_data()
        with pytest.raises(ValueError, match="at least one target"):
            Ridge(alpha=1.0).fit(X, np.zeros((X.shape[0], 0)))

    def test_predict_before_fit_raises(self):
        X, _ = make_data()
        with pytest.raises(ValueError, match="not fitted"):
            Ridge(alpha=1.0).predict(X)


# ------------------------------------------------------- cross-validator handling


class TestCrossValidatorHandling:
    """`cv=int` means unshuffled K-fold; a splitter is used exactly as given."""

    def test_int_cv_builds_unshuffled_kfold(self):
        from sklearn.model_selection import KFold

        X, Y = make_data()
        by_int = Ridge(alpha=ALPHAS, cv=5).fit(X, Y)
        by_splitter = Ridge(alpha=ALPHAS, cv=KFold(5, shuffle=False)).fit(X, Y)
        np.testing.assert_array_equal(by_int.alpha_, by_splitter.alpha_)
        np.testing.assert_array_equal(by_int.cv_scores_, by_splitter.cv_scores_)
        np.testing.assert_array_equal(by_int.coef_, by_splitter.coef_)

    def test_supplied_splitter_is_used_as_given(self):
        from sklearn.model_selection import KFold

        X, Y = make_data()
        contiguous = Ridge(alpha=ALPHAS, cv=KFold(5, shuffle=False)).fit(X, Y)
        shuffled = Ridge(alpha=ALPHAS, cv=KFold(5, shuffle=True, random_state=0)).fit(
            X, Y
        )
        # Same data, different folds: the selection scores cannot coincide.
        assert not np.allclose(shuffled.cv_scores_, contiguous.cv_scores_)


# ------------------------------------------------------------------ fitted state


class TestFittedState:
    """Attribute presence, shapes, and the attributes that must not exist."""

    def test_ordinary_fixed_alpha_state(self):
        X, Y = make_data()
        model = Ridge(alpha=2.0).fit(X, Y)
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
        model = Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        assert model.coef_.shape == (X.shape[1], Y.shape[1])
        assert model.alpha_.shape == (Y.shape[1],)
        assert set(np.unique(model.alpha_)) <= set(ALPHAS)
        assert model.cv_scores_.shape == (Y.shape[1],)
        assert model.feature_space_weights_ is None

    def test_shared_alpha_is_scalar(self):
        X, Y = make_data()
        model = Ridge(alpha=ALPHAS, cv=kfold(), per_target_alpha=False).fit(X, Y)
        assert isinstance(model.alpha_, float)
        assert model.alpha_ in ALPHAS

    def test_one_dimensional_target_squeezes_every_attribute(self):
        X, Y = make_data()
        y = Y[:, 0]
        model = Ridge(alpha=ALPHAS, cv=kfold()).fit(X, y)
        assert model.coef_.shape == (X.shape[1],)
        assert isinstance(model.alpha_, float)
        assert isinstance(model.cv_scores_, float)
        assert model.predict(X).shape == (X.shape[0],)
        assert isinstance(model.score(X, y), float)

    def test_banded_state(self):
        spaces, Y = make_spaces()
        model = Ridge(
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

    def test_banded_one_dimensional_target_shapes(self):
        spaces, Y = make_spaces()
        model = Ridge(
            alpha=ALPHAS, cv=kfold(), search_iterations=6, random_state=3
        ).fit(spaces, Y[:, 0])
        assert model.feature_space_weights_.shape == (len(spaces),)
        assert model.cv_scores_.shape == (6,)
        assert isinstance(model.alpha_, float)

    def test_feature_space_weights_are_positive_and_sum_to_one(self):
        spaces, Y = make_spaces(sizes=(4, 5, 6))
        model = Ridge(
            alpha=ALPHAS, cv=kfold(), search_iterations=10, random_state=7
        ).fit(spaces, Y)
        assert np.all(model.feature_space_weights_ > 0)
        np.testing.assert_allclose(
            model.feature_space_weights_.sum(axis=0), 1.0, rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize("attribute", ["intercept_", "deltas_", "X_", "alphas"])
    def test_removed_attributes_absent(self, attribute):
        X, Y = make_data()
        model = Ridge(alpha=1.0).fit(X, Y)
        assert not hasattr(model, attribute)

    def test_fitted_arrays_are_cpu_numpy(self):
        X, Y = make_data()
        model = Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        for value in (model.coef_, model.alpha_, model.cv_scores_):
            assert isinstance(value, np.ndarray)


# ---------------------------------------------------------------------- numerics


class TestNumericalBehavior:
    """`score` is a per-target R², with nltools' own constant-target rule."""

    def test_score_is_per_target_r2(self):
        X, Y = make_data()
        model = Ridge(alpha=1.0).fit(X, Y)
        scores = model.score(X, Y)
        assert scores.shape == (Y.shape[1],)
        predictions = model.predict(X)
        expected = 1 - ((Y - predictions) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        np.testing.assert_allclose(scores, expected)

    def test_constant_target_scores_zero(self):
        X, Y = make_data()
        Y = Y.copy()
        Y[:, 1] = 4.0
        model = Ridge(alpha=1.0).fit(X, Y)
        assert model.score(X, Y)[1] == 0.0


# ------------------------------------------------------------------ himalaya parity


class TestHimalayaParity:
    """Selection details nltools pins on top of Himalaya: tie-breaks and seeding."""

    def test_ties_break_toward_the_larger_alpha(self):
        # A zero target scores exactly 0.0 for every alpha, so only Himalaya's
        # log-alpha tie-break slope can decide the selection.
        rng = np.random.default_rng(5)
        X = rng.standard_normal((60, 6))
        Y = np.zeros((60, 2))
        model = Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        assert np.all(model.alpha_ == max(ALPHAS))

    def test_banded_search_is_deterministic_under_a_seed(self):
        spaces, Y = make_spaces()
        first = Ridge(
            alpha=ALPHAS, cv=kfold(), search_iterations=8, random_state=99
        ).fit(spaces, Y)
        second = Ridge(
            alpha=ALPHAS, cv=kfold(), search_iterations=8, random_state=99
        ).fit(spaces, Y)
        np.testing.assert_array_equal(first.coef_, second.coef_)
        np.testing.assert_array_equal(
            first.feature_space_weights_, second.feature_space_weights_
        )
        np.testing.assert_array_equal(first.alpha_, second.alpha_)


# ---------------------------------------------------------- candidate preparation


class TestFeatureSpaceWeightCandidates:
    """Validation, dtype conversion, and the underflow floor."""

    def test_caller_array_is_not_mutated(self):
        candidates = np.array([[0.5, 0.5], [1e-45, 1.0 - 1e-45]], dtype=np.float64)
        original = candidates.copy()
        prepared = _prepare_feature_space_weights(candidates, np.float32)
        np.testing.assert_array_equal(candidates, original)
        assert prepared is not candidates

    def test_values_below_tiny_are_raised_to_tiny_float32(self):
        tiny = np.finfo(np.float32).tiny
        candidates = np.array([[1e-45, 1.0 - 1e-45]])
        prepared = _prepare_feature_space_weights(candidates, np.float32)
        assert prepared.dtype == np.float32
        assert prepared[0, 0] == tiny
        assert np.all(prepared > 0)

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

    def test_non_finite_weight_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            _prepare_feature_space_weights(np.array([[np.nan, 1.0]]), np.float32)

    def test_rows_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to one"):
            _prepare_feature_space_weights(np.array([[0.5, 0.9]]), np.float32)

    def test_one_dimensional_candidates_rejected(self):
        with pytest.raises(ValueError, match="must be 2D"):
            _prepare_feature_space_weights(np.array([0.5, 0.5]), np.float32)

    @requires_mps
    def test_tiny_floor_survives_the_mps_round_trip(self):
        import torch

        tiny = np.finfo(np.float32).tiny
        prepared = _prepare_feature_space_weights(
            np.array([[1e-45, 1.0 - 1e-45]]), np.float32
        )
        on_device = torch.as_tensor(prepared, device="mps")
        scaled = torch.sqrt(on_device)
        restored = scaled * scaled
        assert torch.all(scaled > 0).item()
        assert restored.cpu().numpy()[0, 0] >= tiny

    def test_fit_does_not_mutate_caller_arrays(self):
        spaces, Y = make_spaces()
        snapshots = {name: space.copy() for name, space in spaces.items()}
        targets = Y.copy()
        Ridge(alpha=ALPHAS, cv=kfold(), search_iterations=5, random_state=2).fit(
            spaces, Y
        )
        for name, space in spaces.items():
            np.testing.assert_array_equal(space, snapshots[name])
        np.testing.assert_array_equal(Y, targets)


# ------------------------------------------------------------------- banded predict


class TestBandedPrediction:
    """Named spaces are aligned by name; anything else is an error."""

    @pytest.fixture()
    def fitted(self):
        spaces, Y = make_spaces()
        model = Ridge(
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

    def test_additional_space_raises(self, fitted):
        model, spaces, Y = fitted
        extended = dict(spaces)
        extended["extra"] = np.zeros((Y.shape[0], 2))
        with pytest.raises(ValueError, match="unexpected"):
            model.predict(extended)

    def test_wrong_feature_count_raises(self, fitted):
        model, spaces, _ = fitted
        narrowed = dict(spaces)
        first = next(iter(spaces))
        narrowed[first] = spaces[first][:, :-1]
        with pytest.raises(ValueError, match="features, but"):
            model.predict(narrowed)

    def test_array_input_to_banded_model_raises(self, fitted):
        model, spaces, _ = fitted
        with pytest.raises(ValueError, match="must be a mapping"):
            model.predict(np.concatenate(list(spaces.values()), axis=1))

    def test_mapping_input_to_ordinary_model_raises(self):
        X, Y = make_data()
        model = Ridge(alpha=1.0).fit(X, Y)
        with pytest.raises(ValueError, match="single feature matrix"):
            model.predict({"a": X})

    def test_ordinary_feature_count_mismatch_raises(self):
        X, Y = make_data()
        model = Ridge(alpha=1.0).fit(X, Y)
        with pytest.raises(ValueError, match="fitted"):
            model.predict(X[:, :-1])

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

    def test_per_target_gamma_equals_generalized_ridge(self):
        spaces, Y = make_spaces(sizes=(4, 6), n_targets=3)
        matrices = list(spaces.values())
        gammas = np.array([[0.2, 0.5, 0.9], [0.8, 0.5, 0.1]])
        alphas = np.array([1.0, 10.0, 1.0])
        coef = _refit_fixed_hyperparameters(
            matrices, Y, alphas, feature_space_weights=gammas
        )
        for index in range(Y.shape[1]):
            expected = self._generalized_ridge(
                matrices, Y[:, index], alphas[index], gammas[:, index]
            )
            np.testing.assert_allclose(coef[:, index], expected, rtol=1e-6, atol=1e-8)

    def test_target_grouping_does_not_change_results(self):
        spaces, Y = make_spaces(sizes=(4, 6), n_targets=4)
        matrices = list(spaces.values())
        shared = np.array([[0.3] * 4, [0.7] * 4])
        grouped = _refit_fixed_hyperparameters(
            matrices, Y, 2.0, feature_space_weights=shared
        )
        one_at_a_time = np.column_stack(
            [
                _refit_fixed_hyperparameters(
                    matrices,
                    Y[:, [index]],
                    2.0,
                    feature_space_weights=np.array([0.3, 0.7]),
                )[:, 0]
                for index in range(Y.shape[1])
            ]
        )
        np.testing.assert_allclose(grouped, one_at_a_time, rtol=1e-8, atol=1e-10)

    def test_refit_does_not_mutate_the_feature_spaces(self):
        spaces, Y = make_spaces(sizes=(3, 5))
        matrices = list(spaces.values())
        snapshots = [matrix.copy() for matrix in matrices]
        _refit_fixed_hyperparameters(
            matrices, Y, 1.0, feature_space_weights=np.array([0.4, 0.6])
        )
        for matrix, snapshot in zip(matrices, snapshots):
            np.testing.assert_array_equal(matrix, snapshot)

    def test_integer_features_solve_like_float_features(self):
        """An integer design must not silently collapse to zero coefficients.

        Himalaya solves in the dtype it is handed, and an integer dtype makes the
        shrinkage arithmetic truncate to zero. The refit therefore promotes to a
        floating working dtype before it delegates.
        """
        rng = np.random.default_rng(3)
        X_int = rng.integers(0, 2, size=(40, 5))
        Y = rng.standard_normal((40, 3))
        from_int = _refit_fixed_hyperparameters([X_int], Y, 1.0)
        from_float = _refit_fixed_hyperparameters([X_int.astype(np.float64)], Y, 1.0)
        assert np.any(from_int != 0)
        np.testing.assert_allclose(from_int, from_float, rtol=1e-10, atol=1e-12)

    def test_integer_features_reach_the_bootstrap_refit_correctly(self):
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_design,
            _refit_resample,
        )

        rng = np.random.default_rng(4)
        X_int = rng.integers(0, 2, size=(40, 5))
        Y = rng.standard_normal((40, 3))
        indices = np.arange(40)
        weights = _refit_resample(_bootstrap_design([X_int], Y), indices, 1.0)
        expected = _refit_resample(
            _bootstrap_design([X_int.astype(np.float64)], Y), indices, 1.0
        )
        assert np.any(weights != 0)
        np.testing.assert_allclose(weights, expected, rtol=1e-10, atol=1e-12)

    def test_integer_targets_solve_like_float_targets(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((40, 5))
        Y_int = rng.integers(0, 4, size=(40, 2))
        from_int = _refit_fixed_hyperparameters([X], Y_int, 1.0)
        from_float = _refit_fixed_hyperparameters([X], Y_int.astype(np.float64), 1.0)
        np.testing.assert_allclose(from_int, from_float, rtol=1e-10, atol=1e-12)

    def test_banded_fit_coefficients_match_a_fixed_refit(self):
        spaces, Y = make_spaces(sizes=(4, 6))
        model = Ridge(
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
        model = Ridge(alpha=1.0).fit(X, Y)

        restored = pickle.loads(pickle.dumps(model))

        assert restored.is_fitted_
        assert restored.backend_.name == model.backend_.name
        assert restored.backend_.device == model.backend_.device
        np.testing.assert_array_equal(restored.coef_, model.coef_)
        np.testing.assert_allclose(restored.predict(X), model.predict(X))

    def test_fitted_model_deepcopies(self):
        import copy

        X, Y = make_data()
        model = Ridge(alpha=1.0).fit(X, Y)

        clone = copy.deepcopy(model)
        clone.coef_[0, 0] = 1234.0

        assert model.coef_[0, 0] != 1234.0
        assert clone.backend_.name == model.backend_.name

    def test_fitted_model_survives_a_process_based_worker(self):
        from joblib import Parallel, delayed

        X, Y = make_data()
        model = Ridge(alpha=1.0).fit(X, Y)

        (predicted,) = Parallel(n_jobs=2, backend="loky")(
            [delayed(_predict_in_worker)(model, X)]
        )

        np.testing.assert_allclose(predicted, model.predict(X))

    @requires_gpu
    def test_gpu_fitted_model_round_trips_through_pickle(self):
        """The pickled backend keeps the device it was fitted on."""
        import pickle

        X, Y = make_data()
        model = Ridge(alpha=1.0, device="gpu").fit(X, Y)

        restored = pickle.loads(pickle.dumps(model))

        assert restored.backend_.name == model.backend_.name
        assert restored.backend_.device == model.backend_.device
        assert restored.backend_.xp is model.backend_.xp
        np.testing.assert_allclose(restored.predict(X), model.predict(X))

    def test_unpickling_never_switches_device(self, monkeypatch):
        """A model fitted on an absent device keeps its descriptor and raises on use."""
        import pickle

        from nltools.algorithms.backends import Backend

        cuda_backend = Backend.__new__(Backend)
        cuda_backend.__dict__.update(
            {"name": "torch-cuda", "device": "cuda", "_torch_device": None}
        )
        payload = pickle.dumps(cuda_backend)

        import nltools.algorithms.backends as backends_module

        monkeypatch.setattr(backends_module, "_array_module_for", lambda name: None)
        restored = pickle.loads(payload)

        assert restored.name == "torch-cuda"
        assert restored.device == "cuda"
        with pytest.raises(RuntimeError, match="not available in this process"):
            restored.xp

    def test_unpickling_restores_the_torch_module_when_available(self):
        """The torch branch of `__setstate__` is exercised, not just numpy."""
        import pickle

        from nltools.algorithms.backends import Backend

        pytest.importorskip("torch")
        import torch

        cpu_backend = Backend.__new__(Backend)
        cpu_backend.__dict__.update(
            {"name": "torch-cpu", "device": "cpu", "_torch_device": None}
        )

        restored = pickle.loads(pickle.dumps(cpu_backend))

        assert restored.name == "torch-cpu"
        assert restored.xp is torch


def _predict_in_worker(model, X):
    """Module-level so `loky` can pickle it alongside the fitted model."""
    return model.predict(X)


class TestBackendScoping:
    """Himalaya's process-global backend is set for the call and then restored."""

    @requires_torch
    def test_backend_restored_after_a_successful_fit(self, foreign_ambient_backend):
        X, Y = make_data()
        Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        assert current_himalaya_backend() == foreign_ambient_backend

    @requires_torch
    def test_backend_restored_after_a_successful_banded_fit(
        self, foreign_ambient_backend
    ):
        spaces, Y = make_spaces()
        Ridge(alpha=ALPHAS, cv=kfold(), search_iterations=4, random_state=1).fit(
            spaces, Y
        )
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
            Ridge(alpha=ALPHAS, cv=kfold()).fit(X, Y)
        assert current_himalaya_backend() == foreign_ambient_backend

    @requires_torch
    def test_scope_restores_the_previous_backend_on_exception(
        self, foreign_ambient_backend
    ):
        with (
            pytest.raises(RuntimeError, match="boom"),
            _scoped_himalaya_backend("numpy"),
        ):
            assert current_himalaya_backend() == "numpy"
            raise RuntimeError("boom")
        assert current_himalaya_backend() == foreign_ambient_backend

    @requires_torch
    def test_candidate_sampling_runs_under_the_numpy_backend(
        self, foreign_ambient_backend, monkeypatch
    ):
        """Dirichlet candidates must not pick up the ambient backend's dtype."""
        import himalaya.kernel_ridge

        observed = []
        original = himalaya.kernel_ridge.generate_dirichlet_samples

        def record(*args, **kwargs):
            observed.append(current_himalaya_backend())
            return original(*args, **kwargs)

        monkeypatch.setattr(himalaya.kernel_ridge, "generate_dirichlet_samples", record)
        spaces, Y = make_spaces()
        Ridge(alpha=ALPHAS, cv=kfold(), search_iterations=4, random_state=1).fit(
            spaces, Y
        )
        assert observed == ["numpy"]
        assert current_himalaya_backend() == foreign_ambient_backend

    @requires_torch
    def test_banded_fit_is_independent_of_the_ambient_backend(self):
        from himalaya.backend import set_backend

        spaces, Y = make_spaces()
        kwargs = {
            "alpha": ALPHAS,
            "cv": kfold(),
            "search_iterations": 8,
            "random_state": 7,
        }
        previous = current_himalaya_backend()
        try:
            set_backend("numpy", on_error="raise")
            reference = Ridge(**kwargs).fit(spaces, Y)
            set_backend(foreign_backend_name(), on_error="raise")
            foreign = Ridge(**kwargs).fit(spaces, Y)
        finally:
            set_backend(previous, on_error="raise")
        np.testing.assert_array_equal(
            foreign.feature_space_weights_, reference.feature_space_weights_
        )
        np.testing.assert_array_equal(foreign.alpha_, reference.alpha_)
        np.testing.assert_array_equal(foreign.coef_, reference.coef_)


class TestDeviceAndMemory:
    """Explicit GPU runs or raises; the budget only sizes internal batches."""

    def test_explicit_gpu_raises_without_an_accelerator(self, monkeypatch):
        from nltools.algorithms import backends as backends_mod

        def _cpu_only(self):
            self.name = "torch-cpu"
            self.device = "cpu"
            self.xp = object()
            self._torch_device = "cpu"

        monkeypatch.setattr(backends_mod.Backend, "_init_torch", _cpu_only)
        X, Y = make_data()
        with pytest.raises(RuntimeError, match="no GPU accelerator"):
            Ridge(alpha=1.0, device="gpu").fit(X, Y)

    def test_cpu_uses_the_numpy_himalaya_backend(self):
        from nltools.models.ridge import _himalaya_backend_name

        X, Y = make_data()
        model = Ridge(alpha=1.0, device="cpu").fit(X, Y)
        assert model.backend_.device == "cpu"
        assert _himalaya_backend_name(model.backend_) == "numpy"

    def test_over_tight_budget_names_the_argument(self):
        X, Y = make_data()
        model = Ridge(alpha=ALPHAS, cv=kfold(), memory_budget_gb=1e-9)
        with pytest.raises(ValueError, match="memory_budget_gb"):
            model.fit(X, Y)

    def test_explicit_memory_budget_is_accepted(self):
        X, Y = make_data()
        model = Ridge(alpha=ALPHAS, cv=kfold(), memory_budget_gb=1.0).fit(X, Y)
        assert model.is_fitted_

    def test_tiny_budget_still_fits_by_shrinking_batches(self):
        X, Y = make_data()
        small = Ridge(alpha=ALPHAS, cv=kfold(), memory_budget_gb=1e-4).fit(X, Y)
        large = Ridge(alpha=ALPHAS, cv=kfold(), memory_budget_gb=8.0).fit(X, Y)
        np.testing.assert_allclose(small.coef_, large.coef_, rtol=1e-8, atol=1e-10)

    def test_per_target_alpha_refit_gets_a_smaller_batch(self):
        from nltools.algorithms.backends import Backend
        from nltools.models.ridge import _refit_targets_batch

        shape = {"n_samples": 200, "n_targets": 5000, "itemsize": 8, "n_features": 50}
        shared = _refit_targets_batch(
            Backend("numpy"), 0.5, per_target_alpha=False, **shape
        )
        per_target = _refit_targets_batch(
            Backend("numpy"), 0.5, per_target_alpha=True, **shape
        )
        assert per_target < shared

    def test_batch_sizes_shrink_with_the_budget(self):
        from nltools.algorithms.backends import Backend
        from nltools.models.ridge import _batch_sizes

        generous = _batch_sizes(Backend("numpy"), 8.0, 100, 500, 10000, 20, itemsize=4)
        stingy = _batch_sizes(Backend("numpy"), 0.01, 100, 500, 10000, 20, itemsize=4)
        assert stingy["n_targets_batch"] < generous["n_targets_batch"]
        assert stingy["n_alphas_batch"] <= generous["n_alphas_batch"]

    @requires_gpu
    def test_gpu_backend_is_a_real_accelerator(self):
        X, Y = make_data()
        model = Ridge(alpha=1.0, device="gpu").fit(X, Y)
        assert model.backend_.device in ("cuda", "mps")

    @requires_gpu
    def test_cpu_gpu_parity_fixed_alpha(self):
        X, Y = make_data(dtype=np.float32)
        cpu = Ridge(alpha=1.0, device="cpu").fit(X, Y)
        gpu = Ridge(alpha=1.0, device="gpu").fit(X, Y)
        np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=1e-3, atol=1e-4)

    @requires_gpu
    def test_cpu_gpu_parity_cross_validated(self):
        X, Y = make_data(dtype=np.float32)
        cpu = Ridge(alpha=ALPHAS, cv=kfold(), device="cpu").fit(X, Y)
        gpu = Ridge(alpha=ALPHAS, cv=kfold(), device="gpu").fit(X, Y)
        np.testing.assert_array_equal(gpu.alpha_, cpu.alpha_)
        np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=1e-2, atol=1e-3)

    @requires_gpu
    def test_cpu_gpu_parity_banded(self):
        spaces, Y = make_spaces(dtype=np.float32)
        kwargs = {
            "alpha": ALPHAS,
            "cv": kfold(),
            "search_iterations": 6,
            "random_state": 17,
        }
        cpu = Ridge(device="cpu", **kwargs).fit(spaces, Y)
        gpu = Ridge(device="gpu", **kwargs).fit(spaces, Y)
        np.testing.assert_allclose(
            gpu.feature_space_weights_,
            cpu.feature_space_weights_,
            rtol=1e-2,
            atol=1e-3,
        )
        np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=1e-2, atol=1e-2)

    @requires_mps
    def test_mps_fits_in_float32_and_normalizes_to_float64(self):
        X, Y = make_data()  # float64 inputs
        model = Ridge(alpha=1.0, device="gpu").fit(X, Y)
        if model.backend_.device != "mps":
            pytest.skip("resolved backend is not MPS")
        # The fit itself runs in float32 (MPS supports nothing else) ...
        assert _working_dtype([X], Y, model.backend_) == np.dtype(np.float32)
        assert _himalaya_backend_name(model.backend_) == "torch_mps"
        # ... and the fitted state comes back as CPU float64 NumPy.
        assert model.coef_.dtype == np.float64
        assert np.isfinite(model.coef_).all()
