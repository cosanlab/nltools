"""Tests for Fit dataclass (fit_results.py).

This module tests the Fit dataclass which is an immutable container for
model fitting results. Tests cover creation, immutability, introspection
methods, and common usage patterns.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from nltools.data.fit_results import Fit


class TestFitCreation:
    """Test creating Fit objects with different field combinations."""

    def test_minimal_creation(self):
        """Test creating Fit with only required field."""
        fitted_values = np.random.randn(100, 1000)
        fit = Fit(fitted_values=fitted_values)

        assert fit.fitted_values is fitted_values
        assert fit.weights is None
        assert fit.scores is None
        assert fit.betas is None
        assert fit.cv_scores is None

    def test_ridge_no_cv(self):
        """Test creating Fit for Ridge without CV."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        scores = np.random.randn(1000)

        fit = Fit(fitted_values=fitted_values, weights=weights, scores=scores)

        assert fit.fitted_values is fitted_values
        assert fit.weights is weights
        assert fit.scores is scores
        assert fit.cv_scores is None
        assert fit.betas is None

    def test_ridge_with_cv(self):
        """Test creating Fit for Ridge with CV."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        scores = np.random.randn(1000)
        cv_scores = np.random.randn(5, 1000)
        cv_mean_score = np.random.randn(1000)
        cv_predictions = np.random.randn(100, 1000)
        cv_folds = np.arange(100) % 5

        fit = Fit(
            fitted_values=fitted_values,
            weights=weights,
            scores=scores,
            cv_scores=cv_scores,
            cv_mean_score=cv_mean_score,
            cv_predictions=cv_predictions,
            cv_folds=cv_folds,
        )

        assert fit.fitted_values is fitted_values
        assert fit.weights is weights
        assert fit.scores is scores
        assert fit.cv_scores is cv_scores
        assert fit.cv_mean_score is cv_mean_score
        assert fit.cv_predictions is cv_predictions
        assert np.array_equal(fit.cv_folds, cv_folds)

    def test_ridge_with_alpha_selection(self):
        """Test creating Fit for Ridge with alpha='auto' CV."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        scores = np.random.randn(1000)
        cv_scores = np.random.randn(5, 1000)
        cv_mean_score = np.random.randn(1000)
        cv_predictions = np.random.randn(100, 1000)
        cv_folds = np.arange(100) % 5
        cv_best_alpha = 1.5
        cv_alpha_scores = np.random.randn(10)  # 10 alphas tested

        fit = Fit(
            fitted_values=fitted_values,
            weights=weights,
            scores=scores,
            cv_scores=cv_scores,
            cv_mean_score=cv_mean_score,
            cv_predictions=cv_predictions,
            cv_folds=cv_folds,
            cv_best_alpha=cv_best_alpha,
            cv_alpha_scores=cv_alpha_scores,
        )

        assert fit.cv_best_alpha == 1.5
        assert fit.cv_alpha_scores is cv_alpha_scores

    def test_glm_creation(self):
        """Test creating Fit for GLM results."""
        fitted_values = np.random.randn(100, 1000)
        betas = np.random.randn(5, 1000)
        t_stats = np.random.randn(5, 1000)
        p_values = np.random.randn(5, 1000)
        se = np.random.randn(5, 1000)
        residuals = np.random.randn(100, 1000)
        r2 = np.random.randn(1000)

        fit = Fit(
            fitted_values=fitted_values,
            betas=betas,
            t_stats=t_stats,
            p_values=p_values,
            se=se,
            residuals=residuals,
            r2=r2,
        )

        assert fit.fitted_values is fitted_values
        assert fit.betas is betas
        assert fit.t_stats is t_stats
        assert fit.p_values is p_values
        assert fit.se is se
        assert fit.residuals is residuals
        assert fit.r2 is r2
        # Ridge-specific fields should be None
        assert fit.weights is None
        assert fit.scores is None

    def test_mixed_ridge_glm_fields(self):
        """Test that Ridge and GLM fields can coexist (though unusual)."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        betas = np.random.randn(5, 1000)

        fit = Fit(fitted_values=fitted_values, weights=weights, betas=betas)

        assert fit.weights is weights
        assert fit.betas is betas


class TestFitImmutability:
    """Test that Fit is truly immutable (frozen dataclass)."""

    def test_cannot_modify_fitted_values(self):
        """Test that fitted_values cannot be reassigned."""
        fit = Fit(fitted_values=np.random.randn(100, 1000))

        with pytest.raises(FrozenInstanceError):
            fit.fitted_values = np.zeros((100, 1000))

    def test_cannot_modify_optional_field(self):
        """Test that optional fields cannot be reassigned."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000), weights=np.random.randn(5, 1000)
        )

        with pytest.raises(FrozenInstanceError):
            fit.weights = np.zeros((5, 1000))

    def test_cannot_add_new_attributes(self):
        """Test that new attributes cannot be added."""
        fit = Fit(fitted_values=np.random.randn(100, 1000))

        with pytest.raises(FrozenInstanceError):
            fit.new_field = np.zeros(1000)

    def test_cannot_delete_attributes(self):
        """Test that attributes cannot be deleted."""
        fit = Fit(fitted_values=np.random.randn(100, 1000))

        with pytest.raises(FrozenInstanceError):
            del fit.fitted_values

    def test_arrays_remain_mutable(self):
        """Test that numpy arrays inside Fit remain mutable (shallow freeze).

        Note: frozen=True prevents reassignment, but doesn't make arrays immutable.
        Users can still modify array contents in-place.
        """
        fitted_values = np.random.randn(100, 1000)
        fit = Fit(fitted_values=fitted_values)

        # This should work (modifying array contents)
        original_value = fit.fitted_values[0, 0]
        fit.fitted_values[0, 0] = 999.0
        assert fit.fitted_values[0, 0] == 999.0
        assert fit.fitted_values[0, 0] != original_value


class TestFitAvailable:
    """Test the available() method."""

    def test_available_minimal(self):
        """Test available() with only required field."""
        fit = Fit(fitted_values=np.random.randn(100, 1000))

        available = fit.available()
        assert available == ["fitted_values"]

    def test_available_ridge_no_cv(self):
        """Test available() for Ridge without CV."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000),
            weights=np.random.randn(5, 1000),
            scores=np.random.randn(1000),
        )

        available = fit.available()
        assert set(available) == {"fitted_values", "weights", "scores"}
        assert "cv_scores" not in available
        assert "betas" not in available

    def test_available_ridge_with_cv(self):
        """Test available() for Ridge with CV."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000),
            weights=np.random.randn(5, 1000),
            scores=np.random.randn(1000),
            cv_scores=np.random.randn(5, 1000),
            cv_mean_score=np.random.randn(1000),
            cv_predictions=np.random.randn(100, 1000),
            cv_folds=np.arange(100) % 5,
        )

        available = fit.available()
        assert set(available) == {
            "fitted_values",
            "weights",
            "scores",
            "cv_scores",
            "cv_mean_score",
            "cv_predictions",
            "cv_folds",
        }

    def test_available_glm(self):
        """Test available() for GLM results."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000),
            betas=np.random.randn(5, 1000),
            t_stats=np.random.randn(5, 1000),
            p_values=np.random.randn(5, 1000),
        )

        available = fit.available()
        assert set(available) == {"fitted_values", "betas", "t_stats", "p_values"}
        assert "weights" not in available
        assert "se" not in available

    def test_available_excludes_private_fields(self):
        """Test that available() excludes private fields starting with _."""
        # Create a Fit object normally
        fit = Fit(fitted_values=np.random.randn(100, 1000))

        # Add a private field using object.__setattr__ (bypass frozen)
        # Note: This is just for testing; users shouldn't do this
        object.__setattr__(fit, "_private_field", np.zeros(10))

        available = fit.available()
        assert "_private_field" not in available
        assert "fitted_values" in available


class TestFitAsDict:
    """Test the asdict() method."""

    def test_asdict_default_excludes_none(self):
        """Test that asdict() excludes None values by default."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000),
            weights=np.random.randn(5, 1000),
        )

        result = fit.asdict()
        assert set(result.keys()) == {"fitted_values", "weights"}
        assert "scores" not in result
        assert "betas" not in result

    def test_asdict_include_none(self):
        """Test that asdict(include_none=True) includes None values."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000),
            weights=np.random.randn(5, 1000),
        )

        result = fit.asdict(include_none=True)
        assert "fitted_values" in result
        assert "weights" in result
        assert "scores" in result
        assert result["scores"] is None
        assert "betas" in result
        assert result["betas"] is None

    def test_asdict_all_fields(self):
        """Test asdict() with all fields populated."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        scores = np.random.randn(1000)
        betas = np.random.randn(5, 1000)
        t_stats = np.random.randn(5, 1000)
        cv_scores = np.random.randn(5, 1000)

        fit = Fit(
            fitted_values=fitted_values,
            weights=weights,
            scores=scores,
            betas=betas,
            t_stats=t_stats,
            cv_scores=cv_scores,
        )

        result = fit.asdict()
        # Note: dataclass_asdict creates deep copies, so check equality not identity
        assert np.array_equal(result["fitted_values"], fitted_values)
        assert np.array_equal(result["weights"], weights)
        assert np.array_equal(result["scores"], scores)
        assert np.array_equal(result["betas"], betas)
        assert np.array_equal(result["t_stats"], t_stats)
        assert np.array_equal(result["cv_scores"], cv_scores)

    def test_asdict_excludes_private_fields(self):
        """Test that asdict() excludes private fields starting with _."""
        fit = Fit(fitted_values=np.random.randn(100, 1000))

        # Add a private field using object.__setattr__ (bypass frozen)
        object.__setattr__(fit, "_private_field", np.zeros(10))

        result = fit.asdict(include_none=True)
        assert "_private_field" not in result
        assert "fitted_values" in result

    def test_asdict_returns_dict_of_arrays(self):
        """Test that asdict() returns a dictionary suitable for np.savez."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        scores = np.random.randn(1000)

        fit = Fit(fitted_values=fitted_values, weights=weights, scores=scores)

        result = fit.asdict()
        assert isinstance(result, dict)
        assert isinstance(result["fitted_values"], np.ndarray)
        assert isinstance(result["weights"], np.ndarray)
        assert isinstance(result["scores"], np.ndarray)

    def test_asdict_with_scalar_cv_best_alpha(self):
        """Test asdict() handles scalar cv_best_alpha correctly."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000),
            cv_best_alpha=1.5,
        )

        result = fit.asdict()
        assert "cv_best_alpha" in result
        assert result["cv_best_alpha"] == 1.5
        assert isinstance(result["cv_best_alpha"], float)


class TestFitUsagePatterns:
    """Test common usage patterns (serialization, export, introspection)."""

    def test_serialization_npz_roundtrip(self, tmp_path):
        """Test saving to .npz and loading back."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        scores = np.random.randn(1000)

        fit = Fit(fitted_values=fitted_values, weights=weights, scores=scores)

        # Save to .npz
        save_path = tmp_path / "fit_results.npz"
        np.savez(save_path, **fit.asdict())

        # Load back
        loaded = np.load(save_path)
        fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})

        # Verify
        assert np.array_equal(fit_reloaded.fitted_values, fitted_values)
        assert np.array_equal(fit_reloaded.weights, weights)
        assert np.array_equal(fit_reloaded.scores, scores)

    def test_serialization_with_cv_fields(self, tmp_path):
        """Test serialization with CV fields."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        cv_scores = np.random.randn(5, 1000)
        cv_folds = np.arange(100) % 5

        fit = Fit(
            fitted_values=fitted_values,
            weights=weights,
            cv_scores=cv_scores,
            cv_folds=cv_folds,
        )

        # Save to .npz
        save_path = tmp_path / "fit_cv.npz"
        np.savez(save_path, **fit.asdict())

        # Load back
        loaded = np.load(save_path)
        fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})

        # Verify
        assert np.array_equal(fit_reloaded.cv_scores, cv_scores)
        assert np.array_equal(fit_reloaded.cv_folds, cv_folds)

    def test_serialization_with_alpha_selection(self, tmp_path):
        """Test serialization with alpha selection (scalar + array)."""
        fitted_values = np.random.randn(100, 1000)
        cv_best_alpha = 1.5
        cv_alpha_scores = np.random.randn(10)

        fit = Fit(
            fitted_values=fitted_values,
            cv_best_alpha=cv_best_alpha,
            cv_alpha_scores=cv_alpha_scores,
        )

        # Save to .npz
        save_path = tmp_path / "fit_alpha.npz"
        np.savez(save_path, **fit.asdict())

        # Load back
        loaded = np.load(save_path)

        # Note: np.load converts scalars to 0-d arrays, need to extract
        reloaded_alpha = float(loaded["cv_best_alpha"])
        fit_reloaded = Fit(
            fitted_values=loaded["fitted_values"],
            cv_best_alpha=reloaded_alpha,
            cv_alpha_scores=loaded["cv_alpha_scores"],
        )

        # Verify
        assert fit_reloaded.cv_best_alpha == cv_best_alpha
        assert np.array_equal(fit_reloaded.cv_alpha_scores, cv_alpha_scores)

    def test_partial_export(self, tmp_path):
        """Test exporting only specific fields."""
        fitted_values = np.random.randn(100, 1000)
        weights = np.random.randn(5, 1000)
        scores = np.random.randn(1000)
        cv_scores = np.random.randn(5, 1000)

        fit = Fit(
            fitted_values=fitted_values,
            weights=weights,
            scores=scores,
            cv_scores=cv_scores,
        )

        # Export only weights and scores
        save_path = tmp_path / "weights_scores.npz"
        np.savez(save_path, weights=fit.weights, scores=fit.scores)

        # Load and verify
        loaded = np.load(save_path)
        assert "weights" in loaded.files
        assert "scores" in loaded.files
        assert "fitted_values" not in loaded.files
        assert "cv_scores" not in loaded.files

    def test_introspection_workflow(self):
        """Test typical introspection workflow."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000),
            weights=np.random.randn(5, 1000),
            scores=np.random.randn(1000),
            cv_scores=np.random.randn(5, 1000),
            cv_mean_score=np.random.randn(1000),
        )

        # Check what's available
        available = fit.available()
        assert "cv_scores" in available

        # Conditional processing based on availability
        if "cv_mean_score" in available:
            mean_r2 = fit.cv_mean_score.mean()
            assert isinstance(mean_r2, (float, np.floating))

        # Get dict for further processing
        results = fit.asdict()
        assert set(results.keys()) == set(available)

    def test_combining_with_metadata(self):
        """Test combining Fit results with metadata dict."""
        fit = Fit(
            fitted_values=np.random.randn(100, 1000),
            weights=np.random.randn(5, 1000),
            scores=np.random.randn(1000),
        )

        # Add metadata
        metadata = {
            "model_type": "ridge",
            "alpha": 1.0,
            "n_samples": 100,
            "n_voxels": 1000,
        }

        # Combine for export
        combined = {**metadata, **fit.asdict()}
        assert "model_type" in combined
        assert "weights" in combined
        assert combined["model_type"] == "ridge"

    def test_shape_inspection(self):
        """Test inspecting shapes of results."""
        n_samples, n_voxels = 100, 1000
        n_features = 5
        n_folds = 5

        fit = Fit(
            fitted_values=np.random.randn(n_samples, n_voxels),
            weights=np.random.randn(n_features, n_voxels),
            scores=np.random.randn(n_voxels),
            cv_scores=np.random.randn(n_folds, n_voxels),
        )

        # Verify expected shapes
        assert fit.fitted_values.shape == (n_samples, n_voxels)
        assert fit.weights.shape == (n_features, n_voxels)
        assert fit.scores.shape == (n_voxels,)
        assert fit.cv_scores.shape == (n_folds, n_voxels)

    def test_extracting_voxel_wise_results(self):
        """Test extracting results for specific voxels."""
        n_samples, n_voxels = 100, 1000
        fit = Fit(
            fitted_values=np.random.randn(n_samples, n_voxels),
            scores=np.random.randn(n_voxels),
        )

        # Extract top 10 voxels by R²
        top_voxels = np.argsort(fit.scores)[-10:]
        top_scores = fit.scores[top_voxels]
        top_predictions = fit.fitted_values[:, top_voxels]

        assert len(top_scores) == 10
        assert top_predictions.shape == (n_samples, 10)
        # Verify these are the highest scores
        assert np.all(top_scores >= np.partition(fit.scores, -10)[-10])
