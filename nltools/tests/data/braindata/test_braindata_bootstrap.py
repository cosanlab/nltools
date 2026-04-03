import numpy as np
import pytest

from nltools.data import BrainData
from nltools.mask import create_sphere


class TestBrainDataBootstrap:
    def test_bootstrap(self, sim_brain_data):
        """Test bootstrap with mean/std."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # Test basic bootstrap with mean and std (should work)
        # Note: n_samples must be >= 10 for new implementation
        n_samples = 50
        b = masked.bootstrap(stat="mean", n_samples=n_samples)
        # New API returns BrainData directly
        assert isinstance(b, BrainData)
        assert b.shape == (1, masked.shape[1])  # (1, n_voxels)
        b = masked.bootstrap(stat="std", n_samples=n_samples)
        assert isinstance(b, BrainData)
        assert b.shape == (1, masked.shape[1])  # (1, n_voxels)

        # Bootstrap with "predict" requires fitted model (pass X_test to get past that check)
        X_test = np.random.randn(5, 10)  # Dummy test features
        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(stat="predict", n_samples=n_samples, X_test=X_test)

    def test_bootstrap_invalid_method_error(self, sim_brain_data):
        """Test error raised for unsupported method."""
        # New implementation validates stat names upfront
        with pytest.raises(
            ValueError,
            match="Unsupported stat.*Supported simple stats",
        ):
            sim_brain_data.bootstrap(stat="invalid_method_name", n_samples=10)

    # ==================== Phase 5: New Bootstrap Implementation ====================

    def test_bootstrap_new_stat_param(self, sim_brain_data):
        """Test new bootstrap with stat='mean' parameter."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # New API: stat parameter
        boot = masked.bootstrap(stat="mean", n_samples=100, random_state=42)

        # Should return BrainData with shape (1, n_voxels) for aggregated result
        assert isinstance(boot, BrainData)
        assert boot.shape == (
            1,
            masked.shape[1],
        )  # (1, n_voxels) - aggregated across samples

    def test_bootstrap_new_save_boots_param(self, sim_brain_data):
        """Test new bootstrap with save_boots parameter."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # New API: save_boots=True should return dict
        result = masked.bootstrap(
            stat="mean", n_samples=50, save_boots=True, random_state=42
        )

        # When save_boots=True, should return dict with samples
        assert isinstance(result, dict)
        assert "samples" in result
        assert result["samples"].shape[0] == 50  # n_samples

    def test_bootstrap_new_all_simple_stats(self, sim_brain_data):
        """Test all simple stats work with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        stats = ["mean", "median", "std", "sum", "min", "max"]
        for stat in stats:
            boot = masked.bootstrap(stat=stat, n_samples=50, random_state=42)
            assert isinstance(boot, BrainData)
            assert boot.shape == (1, masked.shape[1])  # (1, n_voxels) - aggregated

    def test_bootstrap_new_ridge_weights_requires_fit(self, sim_brain_data):
        """Test weights bootstrap requires fitted model."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(stat="weights", n_samples=10)

    def test_bootstrap_new_ridge_weights_basic(self, sim_brain_data):
        """Test Ridge weights bootstrap with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # Create design matrix
        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))

        # Fit model
        masked.fit(X=dm, model="ridge", alpha=1.0)

        # Bootstrap weights
        boot = masked.bootstrap(stat="weights", n_samples=100, random_state=42)

        # Should return dict with mean, std, Z, p, ci_lower, ci_upper
        assert isinstance(boot, dict)
        assert "mean" in boot
        assert "std" in boot
        assert "Z" in boot
        assert "p" in boot
        assert "ci_lower" in boot
        assert "ci_upper" in boot

        # Mean should be BrainData with shape (n_features, n_voxels)
        assert isinstance(boot["mean"], BrainData)
        assert boot["mean"].shape == (5, masked.shape[1])  # n_features × n_voxels

    def test_bootstrap_new_ridge_predict_requires_fit(self, sim_brain_data):
        """Test predict bootstrap requires fitted model."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        X_test = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(stat="predict", X_test=X_test, n_samples=10)

    def test_bootstrap_new_ridge_predict_requires_x_test(self, sim_brain_data):
        """Test predict bootstrap requires X_test."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))

        masked.fit(X=dm, model="ridge", alpha=1.0)

        with pytest.raises(ValueError, match="X_test.*required"):
            masked.bootstrap(stat="predict", n_samples=10)

    def test_bootstrap_new_ridge_predict_basic(self, sim_brain_data):
        """Test Ridge predict bootstrap with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))
        X_test = np.random.randn(10, 5)

        masked.fit(X=dm, model="ridge", alpha=1.0)

        boot = masked.bootstrap(
            stat="predict", X_test=X_test, n_samples=100, random_state=42
        )

        # Should return dict
        assert isinstance(boot, dict)
        assert "mean" in boot

        # Mean should be BrainData with shape (n_test_samples, n_voxels)
        assert isinstance(boot["mean"], BrainData)
        assert boot["mean"].shape == (10, masked.shape[1])  # n_test × n_voxels
