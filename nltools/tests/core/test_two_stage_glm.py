"""Tests for two-stage GLM workflow (Phase 6).

Tests cover:
- bc.fit(model='glm') -> FittedBrainCollection
- FittedBrainCollection.pool() -> PooledData
- PooledData.fit(model='ttest') -> StatResult
- Complete two-stage chain integration
- Contrast parsing and application
- Multiple contrasts workflow
"""

import numpy as np
import pandas as pd
import pytest

from nltools.pipelines.pool import PooledData, ResultDict, StatResult


class TestFittedBrainCollectionPool:
    """Tests for FittedBrainCollection.pool() method."""

    @pytest.fixture
    def mock_fitted_bc(self):
        """Create a mock FittedBrainCollection for testing.

        Simulates output of bc.fit(model='glm', X=dm) without
        requiring actual BrainCollection/GLM infrastructure.
        """
        from nltools.data.collection import FittedBrainCollection

        # Create mock BrainData-like objects with .data attribute
        class MockBrainData:
            def __init__(self, data):
                self.data = data

        # Create mock BrainCollection-like object
        class MockBrainCollection:
            def __init__(self, items, design_columns=None):
                self._items = items
                self._design_columns = design_columns
                self.mask = None
                self.metadata = pd.DataFrame(
                    {"subject": [f"sub-{i:02d}" for i in range(len(items))]}
                )

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        # Create synthetic beta data: 5 subjects, 3 conditions, 100 voxels
        np.random.seed(42)
        n_subjects = 5
        n_conditions = 3
        n_voxels = 100

        # Add signal to first 10 voxels for condition 0
        betas = []
        for _ in range(n_subjects):
            beta_data = np.random.randn(n_conditions, n_voxels) * 0.5
            beta_data[0, :10] += 2.0  # Signal in condition 0
            beta_data[1, :10] += 1.0  # Weaker signal in condition 1
            betas.append(MockBrainData(beta_data))

        mock_bc = MockBrainCollection(betas, design_columns=["face", "house", "object"])
        fitted_results = MockBrainCollection(
            betas, design_columns=["face", "house", "object"]
        )

        return FittedBrainCollection(
            brain_collection=mock_bc,
            fitted_results=fitted_results,
            model="glm",
            condition_names=["face", "house", "object"],
        )

    def test_pool_returns_pooled_data(self, mock_fitted_bc):
        """Test pool() returns PooledData."""
        pool = mock_fitted_bc.pool(param="beta")

        assert isinstance(pool, PooledData)
        assert pool.n_subjects == 5
        assert pool.n_conditions == 3
        assert pool.n_voxels == 100
        assert pool.param == "beta"

    def test_pool_has_condition_names(self, mock_fitted_bc):
        """Test pool() preserves condition names."""
        pool = mock_fitted_bc.pool(param="beta")

        assert pool.condition_names == ["face", "house", "object"]

    def test_pool_has_subject_ids(self, mock_fitted_bc):
        """Test pool() preserves subject IDs."""
        pool = mock_fitted_bc.pool(param="beta")

        assert pool.subject_ids is not None
        assert len(pool.subject_ids) == 5

    def test_pool_data_shape(self, mock_fitted_bc):
        """Test pooled data has correct shape."""
        pool = mock_fitted_bc.pool(param="beta")

        assert pool.shape == (5, 3, 100)

    def test_pool_ttest_returns_stat_result(self, mock_fitted_bc):
        """Test pool.fit(ttest) returns StatResult."""
        pool = mock_fitted_bc.pool(param="beta")
        result = pool.fit(model="ttest", contrast="face-house")

        assert isinstance(result, StatResult)
        assert result.t_map is not None
        assert result.p_map is not None
        assert result.contrast == "face-house"

    def test_pool_ttest_detects_signal(self, mock_fitted_bc):
        """Test t-test detects planted signal."""
        pool = mock_fitted_bc.pool(param="beta")
        result = pool.fit(model="ttest", contrast="face-house")

        # First 10 voxels should have higher t-values (face > house)
        mean_sig = np.mean(np.abs(result.t_map[:10]))
        mean_noise = np.mean(np.abs(result.t_map[10:]))
        assert mean_sig > mean_noise

    def test_pool_multiple_contrasts(self, mock_fitted_bc):
        """Test multiple contrasts returns ResultDict."""
        pool = mock_fitted_bc.pool(param="beta")
        results = pool.fit(
            model="ttest", contrasts=["face-house", "face-object", "house-object"]
        )

        assert isinstance(results, ResultDict)
        assert "face-house" in results
        assert "face-object" in results
        assert "house-object" in results

    def test_pool_anova(self, mock_fitted_bc):
        """Test ANOVA on pooled data."""
        pool = mock_fitted_bc.pool(param="beta")
        result = pool.fit(model="anova")

        assert isinstance(result, StatResult)
        assert result.f_map is not None
        assert result.f_map.shape == (100,)


class TestTwoStageContrasts:
    """Tests for contrast specification and application."""

    @pytest.fixture
    def multicond_pool(self):
        """Create multi-condition pooled data."""
        np.random.seed(42)
        # 10 subjects, 4 conditions, 50 voxels
        data = np.random.randn(10, 4, 50)
        # Add condition effects
        data[:, 0, :10] += 2.0  # Face
        data[:, 1, :10] += 1.0  # House
        data[:, 2, :10] += 0.5  # Object
        data[:, 3, :10] += 0.0  # Scrambled (baseline)

        return PooledData(
            data=data,
            param="beta",
            condition_names=["face", "house", "object", "scrambled"],
        )

    def test_simple_subtraction(self, multicond_pool):
        """Test A-B contrast."""
        result = multicond_pool.fit(model="ttest", contrast="face-house")

        assert result.contrast == "face-house"
        # face > house in first 10 voxels
        assert np.mean(result.t_map[:10]) > 0

    def test_addition_contrast(self, multicond_pool):
        """Test A+B contrast (average)."""
        # face+house should be > scrambled
        result = multicond_pool.fit(
            model="ttest", contrast="face+house-scrambled-scrambled"
        )

        # This tests that both face and house contribute positively
        assert np.mean(result.t_map[:10]) > 0

    def test_complex_contrast(self, multicond_pool):
        """Test multi-term contrast."""
        # (face + house) vs (object + scrambled)
        result = multicond_pool.fit(
            model="ttest", contrast="face+house-object-scrambled"
        )

        assert result.t_map is not None

    def test_unknown_condition_raises(self, multicond_pool):
        """Test unknown condition in contrast raises error."""
        with pytest.raises(ValueError, match="Unknown condition"):
            multicond_pool.fit(model="ttest", contrast="face-unknown")


class TestStatResultThresholding:
    """Tests for StatResult thresholding in two-stage context."""

    @pytest.fixture
    def significant_result(self):
        """Create result with some significant voxels."""
        np.random.seed(42)
        n_voxels = 100

        # Create t-values with strong signal in first 5 voxels
        t_map = np.random.randn(n_voxels)
        t_map[:5] = np.abs(t_map[:5]) + 5  # Strong positive signal

        # Calculate p-values (approximate)
        from scipy import stats

        p_map = 2 * (1 - stats.t.cdf(np.abs(t_map), df=19))

        return StatResult(t_map=t_map, p_map=p_map, contrast="A-B", df=19)

    def test_fdr_threshold(self, significant_result):
        """Test FDR thresholding preserves signal."""
        thresholded = significant_result.threshold(method="fdr", alpha=0.05)

        # Should have significant voxels in first 5
        n_sig = np.sum(thresholded.t_map[:5] != 0)
        assert n_sig > 0

    def test_bonferroni_conservative(self, significant_result):
        """Test Bonferroni is more conservative than FDR."""
        fdr = significant_result.threshold(method="fdr", alpha=0.05)
        bonf = significant_result.threshold(method="bonferroni", alpha=0.05)

        n_fdr = np.sum(fdr.t_map != 0)
        n_bonf = np.sum(bonf.t_map != 0)

        assert n_bonf <= n_fdr

    def test_threshold_chain(self, significant_result):
        """Test thresholding in chained workflow."""
        # This mimics the full workflow ending
        thresholded = significant_result.threshold(method="fdr")

        # Should still be a StatResult
        assert isinstance(thresholded, StatResult)
        assert thresholded.contrast == "A-B"


class TestResultDictOperations:
    """Tests for ResultDict in multi-contrast workflows."""

    @pytest.fixture
    def multi_contrast_results(self):
        """Create ResultDict with multiple contrasts."""
        np.random.seed(42)
        n_voxels = 50

        results = {}
        for contrast in ["A-B", "A-C", "B-C"]:
            t_map = np.random.randn(n_voxels)
            p_map = np.random.rand(n_voxels)
            results[contrast] = StatResult(t_map=t_map, p_map=p_map, contrast=contrast)

        return ResultDict(results)

    def test_threshold_all(self, multi_contrast_results):
        """Test batch thresholding."""
        thresholded = multi_contrast_results.threshold_all(method="fdr")

        assert isinstance(thresholded, ResultDict)
        assert len(thresholded) == 3
        for key in ["A-B", "A-C", "B-C"]:
            assert key in thresholded
            assert isinstance(thresholded[key], StatResult)

    def test_iterate_results(self, multi_contrast_results):
        """Test iterating over results."""
        contrasts = list(multi_contrast_results.keys())
        assert len(contrasts) == 3

        for contrast, result in multi_contrast_results.items():
            assert isinstance(result, StatResult)
            assert result.contrast == contrast


class TestTwoStageIntegration:
    """End-to-end integration tests for two-stage GLM."""

    def test_synthetic_group_analysis(self):
        """Test complete two-stage workflow with synthetic data."""
        np.random.seed(42)

        # Simulate: 8 subjects, 2 conditions, 100 voxels
        # Condition A has signal in first 20 voxels
        n_subjects = 8
        n_voxels = 100

        betas = np.random.randn(n_subjects, 2, n_voxels) * 0.5
        betas[:, 0, :20] += 1.5  # Add signal to condition A
        betas[:, 1, :20] += 0.5  # Weaker signal in condition B

        # Create PooledData (simulating output of bc.fit().pool())
        pool = PooledData(data=betas, param="beta", condition_names=["A", "B"])

        # Run t-test on A-B contrast
        result = pool.fit(model="ttest", contrast="A-B")

        # Verify:
        # 1. Returns StatResult
        assert isinstance(result, StatResult)

        # 2. t-values exist
        assert result.t_map is not None
        assert result.t_map.shape == (n_voxels,)

        # 3. Signal detected in expected voxels
        mean_sig = np.mean(result.t_map[:20])
        mean_noise = np.mean(result.t_map[20:])
        assert mean_sig > mean_noise

        # 4. Thresholding works
        thresholded = result.threshold(method="fdr", alpha=0.05)
        assert isinstance(thresholded, StatResult)

    def test_multi_contrast_workflow(self):
        """Test multiple contrasts in single call."""
        np.random.seed(42)

        # 3 conditions
        betas = np.random.randn(10, 3, 50)
        pool = PooledData(
            data=betas, param="beta", condition_names=["face", "house", "object"]
        )

        # All pairwise contrasts
        results = pool.fit(
            model="ttest",
            contrasts=["face-house", "face-object", "house-object"],
        )

        # Verify ResultDict
        assert isinstance(results, ResultDict)
        assert len(results) == 3

        # Threshold all and verify
        thresholded = results.threshold_all(method="bonferroni")
        assert isinstance(thresholded, ResultDict)
        assert len(thresholded) == 3

    def test_anova_workflow(self):
        """Test ANOVA for 3+ conditions."""
        np.random.seed(42)

        # 4 conditions with different means in first 10 voxels
        betas = np.random.randn(12, 4, 80)
        for i in range(4):
            betas[:, i, :10] += i * 0.5  # Increasing means

        pool = PooledData(
            data=betas,
            param="beta",
            condition_names=["cond1", "cond2", "cond3", "cond4"],
        )

        result = pool.fit(model="anova")

        # Verify F-test
        assert isinstance(result, StatResult)
        assert result.f_map is not None
        assert result.f_map.shape == (80,)

        # First 10 voxels should have higher F-values
        mean_sig = np.mean(result.f_map[:10])
        mean_noise = np.mean(result.f_map[10:])
        assert mean_sig > mean_noise

    def test_paired_ttest_workflow(self):
        """Test paired t-test with 2 conditions."""
        np.random.seed(42)

        # 2 conditions - within-subject difference
        betas = np.random.randn(15, 2, 60)
        # Condition B consistently higher in first 15 voxels
        betas[:, 1, :15] += 1.0

        pool = PooledData(data=betas, param="beta", condition_names=["pre", "post"])

        result = pool.fit(model="paired_ttest")

        assert isinstance(result, StatResult)
        assert result.t_map is not None

        # Should detect the difference
        # (post > pre means negative t-values for paired test order)
        mean_sig = np.mean(result.t_map[:15])
        mean_noise = np.mean(result.t_map[15:])
        # Signal should be different from noise
        assert np.abs(mean_sig) > np.abs(mean_noise)

    def test_two_sample_ttest(self):
        """Test two-sample t-test with group design."""
        np.random.seed(42)

        # 20 subjects: 10 patients, 10 controls
        betas = np.random.randn(20, 1, 100)  # Single condition (e.g., task activation)

        # Patients (first 10) have stronger activation
        betas[:10, 0, :25] += 1.0

        pool = PooledData(data=betas.squeeze(), param="beta")  # (20, 100)

        # Group design: 0=patient, 1=control
        groups = np.array([0] * 10 + [1] * 10)

        result = pool.fit(model="ttest", X=groups)

        assert isinstance(result, StatResult)
        assert result.t_map is not None

        # Should detect group difference in first 25 voxels
        mean_sig = np.mean(np.abs(result.t_map[:25]))
        mean_noise = np.mean(np.abs(result.t_map[25:]))
        assert mean_sig > mean_noise
