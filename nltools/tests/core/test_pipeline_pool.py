"""Tests for pool infrastructure (Phase 5).

Tests cover:
- PooledData: data pooling and properties
- StatResult: statistical test results and thresholding
- ResultDict: batch operations on multiple contrasts
- FittedBrainCollection.pool(): integration chain
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from nltools.pipelines.pool import PooledData, StatResult, ResultDict


class TestPooledData:
    """Tests for PooledData class."""

    @pytest.fixture
    def simple_pool(self):
        """Create simple 2D pooled data."""
        np.random.seed(42)
        data = np.random.randn(10, 100)  # 10 subjects, 100 voxels
        return PooledData(data=data, param="beta")

    @pytest.fixture
    def multicond_pool(self):
        """Create 3D multi-condition pooled data."""
        np.random.seed(42)
        data = np.random.randn(10, 3, 100)  # 10 subjects, 3 conditions, 100 voxels
        return PooledData(
            data=data, param="beta", condition_names=["face", "house", "object"]
        )

    def test_creation(self, simple_pool):
        """Test PooledData creation."""
        assert simple_pool.n_subjects == 10
        assert simple_pool.n_voxels == 100
        assert simple_pool.param == "beta"
        assert simple_pool.n_conditions is None

    def test_multicond_creation(self, multicond_pool):
        """Test multi-condition PooledData creation."""
        assert multicond_pool.n_subjects == 10
        assert multicond_pool.n_conditions == 3
        assert multicond_pool.n_voxels == 100
        assert multicond_pool.condition_names == ["face", "house", "object"]

    def test_shape_property(self, simple_pool, multicond_pool):
        """Test shape property."""
        assert simple_pool.shape == (10, 100)
        assert multicond_pool.shape == (10, 3, 100)

    def test_fit_ttest(self, simple_pool):
        """Test one-sample t-test."""
        result = simple_pool.fit(model="ttest")

        assert isinstance(result, StatResult)
        assert result.t_map is not None
        assert result.p_map is not None
        assert result.t_map.shape == (100,)

    def test_fit_ttest_two_sample(self, simple_pool):
        """Test two-sample t-test."""
        # Create group labels
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = simple_pool.fit(model="ttest", X=groups)

        assert isinstance(result, StatResult)
        assert result.t_map.shape == (100,)

    def test_fit_paired_ttest(self):
        """Test paired t-test with 2 conditions."""
        np.random.seed(42)
        data = np.random.randn(10, 2, 100)
        pool = PooledData(data=data, param="beta", condition_names=["A", "B"])

        result = pool.fit(model="paired_ttest")

        assert isinstance(result, StatResult)
        assert result.t_map.shape == (100,)

    def test_fit_anova(self, multicond_pool):
        """Test one-way ANOVA."""
        result = multicond_pool.fit(model="anova")

        assert isinstance(result, StatResult)
        assert result.f_map is not None
        assert result.f_map.shape == (100,)

    def test_fit_with_contrast(self, multicond_pool):
        """Test fitting with contrast."""
        result = multicond_pool.fit(model="ttest", contrast="face-house")

        assert isinstance(result, StatResult)
        assert result.contrast == "face-house"

    def test_fit_multiple_contrasts(self, multicond_pool):
        """Test fitting multiple contrasts returns ResultDict."""
        results = multicond_pool.fit(
            model="ttest", contrasts=["face-house", "face-object"]
        )

        assert isinstance(results, ResultDict)
        assert "face-house" in results
        assert "face-object" in results

    def test_contrast_parsing(self, multicond_pool):
        """Test contrast string parsing."""
        # Simple subtraction
        weights = multicond_pool._parse_contrast("face-house")
        np.testing.assert_array_equal(weights, [1, -1, 0])

        # Addition
        weights = multicond_pool._parse_contrast("face+house")
        np.testing.assert_array_equal(weights, [1, 1, 0])

        # Complex
        weights = multicond_pool._parse_contrast("face-house+object")
        np.testing.assert_array_equal(weights, [1, -1, 1])

    def test_contrast_unknown_condition_raises(self, multicond_pool):
        """Test unknown condition in contrast raises error."""
        with pytest.raises(ValueError, match="Unknown condition"):
            multicond_pool._parse_contrast("face-unknown")

    def test_repool_without_fitted_state_raises(self, simple_pool):
        """Test repool raises without fitted state."""
        with pytest.raises(ValueError, match="No fitted state"):
            simple_pool.repool("residual")

    def test_save_load_npz(self, simple_pool):
        """Test save/load with npz format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pooled.npz"
            simple_pool.save(str(path))

            loaded = PooledData.load(str(path))

            np.testing.assert_array_equal(loaded.data, simple_pool.data)
            assert loaded.param == simple_pool.param

    def test_save_load_directory(self, simple_pool):
        """Test save/load with directory format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pooled_dir"
            simple_pool.save(str(path))

            loaded = PooledData.load(str(path))

            np.testing.assert_array_equal(loaded.data, simple_pool.data)
            assert loaded.param == simple_pool.param

    def test_repr(self, simple_pool, multicond_pool):
        """Test string representation."""
        r1 = repr(simple_pool)
        assert "PooledData" in r1
        assert "beta" in r1
        assert "10" in r1

        r2 = repr(multicond_pool)
        assert "3" in r2  # n_conditions


class TestStatResult:
    """Tests for StatResult class."""

    @pytest.fixture
    def t_result(self):
        """Create t-test result."""
        np.random.seed(42)
        t_map = np.random.randn(100)
        p_map = np.random.rand(100)
        return StatResult(t_map=t_map, p_map=p_map, contrast="A-B")

    @pytest.fixture
    def f_result(self):
        """Create F-test result."""
        np.random.seed(42)
        f_map = np.abs(np.random.randn(100)) * 5
        p_map = np.random.rand(100)
        return StatResult(f_map=f_map, p_map=p_map)

    def test_creation(self, t_result):
        """Test StatResult creation."""
        assert t_result.t_map is not None
        assert t_result.p_map is not None
        assert t_result.contrast == "A-B"

    def test_threshold_fdr(self, t_result):
        """Test FDR thresholding."""
        thresholded = t_result.threshold(method="fdr", alpha=0.05)

        assert isinstance(thresholded, StatResult)
        # Thresholded map should have zeros where not significant
        assert thresholded.t_map is not None

    def test_threshold_bonferroni(self, t_result):
        """Test Bonferroni thresholding."""
        thresholded = t_result.threshold(method="bonferroni", alpha=0.05)

        assert isinstance(thresholded, StatResult)
        # Very few should survive Bonferroni
        n_sig = np.sum(thresholded.t_map != 0)
        assert n_sig <= t_result.t_map.size * 0.1  # At most 10%

    def test_threshold_uncorrected(self, t_result):
        """Test uncorrected thresholding."""
        thresholded = t_result.threshold(method="uncorrected", alpha=0.05)

        assert isinstance(thresholded, StatResult)
        # Should have roughly 5% surviving by chance
        n_sig = np.sum(thresholded.t_map != 0)
        assert n_sig > 0

    def test_threshold_unknown_method_raises(self, t_result):
        """Test unknown threshold method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            t_result.threshold(method="unknown")

    def test_threshold_without_pvalues_raises(self):
        """Test thresholding without p-values raises error."""
        result = StatResult(t_map=np.random.randn(100))
        with pytest.raises(ValueError, match="No p-values"):
            result.threshold()

    def test_to_nifti_not_implemented(self, t_result):
        """Test to_nifti raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            t_result.to_nifti("test.nii")

    def test_repr_ttest(self, t_result):
        """Test t-test repr."""
        r = repr(t_result)
        assert "t-test" in r
        assert "A-B" in r

    def test_repr_ftest(self, f_result):
        """Test F-test repr."""
        r = repr(f_result)
        assert "F-test" in r


class TestResultDict:
    """Tests for ResultDict class."""

    @pytest.fixture
    def result_dict(self):
        """Create ResultDict with multiple contrasts."""
        np.random.seed(42)
        return ResultDict(
            {
                "A-B": StatResult(
                    t_map=np.random.randn(100),
                    p_map=np.random.rand(100),
                    contrast="A-B",
                ),
                "A-C": StatResult(
                    t_map=np.random.randn(100),
                    p_map=np.random.rand(100),
                    contrast="A-C",
                ),
                "B-C": StatResult(
                    t_map=np.random.randn(100),
                    p_map=np.random.rand(100),
                    contrast="B-C",
                ),
            }
        )

    def test_access(self, result_dict):
        """Test accessing individual results."""
        assert "A-B" in result_dict
        assert isinstance(result_dict["A-B"], StatResult)

    def test_threshold_all(self, result_dict):
        """Test batch thresholding."""
        thresholded = result_dict.threshold_all(method="fdr", alpha=0.05)

        assert isinstance(thresholded, ResultDict)
        assert "A-B" in thresholded
        assert "A-C" in thresholded
        assert "B-C" in thresholded

    def test_repr(self, result_dict):
        """Test string representation."""
        r = repr(result_dict)
        assert "ResultDict" in r
        assert "A-B" in r or "A-C" in r or "B-C" in r


class TestPooledDataIntegration:
    """Integration tests for PooledData with statistical models."""

    def test_full_ttest_workflow(self):
        """Test complete t-test workflow."""
        np.random.seed(42)
        # Create data with signal in first 10 voxels
        data = np.random.randn(20, 100)
        data[:, :10] += 2  # Add signal

        pool = PooledData(data=data, param="beta")
        result = pool.fit(model="ttest")

        # First 10 voxels should have higher t-values
        assert np.mean(np.abs(result.t_map[:10])) > np.mean(np.abs(result.t_map[10:]))

    def test_full_anova_workflow(self):
        """Test complete ANOVA workflow."""
        np.random.seed(42)
        # Create 3-condition data
        data = np.random.randn(15, 3, 50)
        # Add condition effects to first 5 voxels
        data[:, 0, :5] += 1
        data[:, 1, :5] += 2
        data[:, 2, :5] += 3

        pool = PooledData(
            data=data, param="beta", condition_names=["cond1", "cond2", "cond3"]
        )
        result = pool.fit(model="anova")

        # First 5 voxels should have higher F-values
        assert np.mean(result.f_map[:5]) > np.mean(result.f_map[5:])

    def test_contrast_workflow(self):
        """Test contrast-based workflow."""
        np.random.seed(42)
        # Create data with condition difference
        data = np.zeros((20, 2, 100))
        data[:, 0, :] = np.random.randn(20, 100)  # Condition A
        data[:, 1, :] = np.random.randn(20, 100) + 1  # Condition B (higher)

        pool = PooledData(data=data, param="beta", condition_names=["A", "B"])
        result = pool.fit(model="ttest", contrast="A-B")

        # Should show significant negative t-values (A < B)
        assert np.mean(result.t_map) < 0

    def test_threshold_and_report(self):
        """Test thresholding and reporting significant voxels."""
        np.random.seed(42)
        data = np.random.randn(30, 100)
        data[:, :5] += 3  # Strong signal in 5 voxels

        pool = PooledData(data=data, param="beta")
        result = pool.fit(model="ttest")
        thresholded = result.threshold(method="fdr", alpha=0.05)

        # Should have some significant voxels in first 5
        sig_voxels = np.where(thresholded.t_map != 0)[0]
        assert len(sig_voxels) > 0
