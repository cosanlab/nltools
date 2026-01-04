"""Tests for alignment pipeline step (Phase 7).

Tests cover:
- AlignStep: SRM and HyperAlignment wrappers
- FittedAlign: transform and new subject handling
- MultiSubjectPipeline.align(): integration
"""

import numpy as np
import pytest

from nltools.pipelines.steps import AlignStep, FittedAlign


class TestAlignStep:
    """Tests for AlignStep class."""

    @pytest.fixture
    def multi_subject_data(self):
        """Create synthetic multi-subject data.

        Returns data in (samples, voxels) format as typically used in pipelines.
        """
        np.random.seed(42)
        n_subjects = 4
        n_samples = 50
        n_voxels = 100

        # Create correlated data across subjects (shared signal + noise)
        shared = np.random.randn(n_samples, 20)  # Shared signal in 20 dims

        data = []
        for _ in range(n_subjects):
            # Project shared signal to voxel space + add noise
            proj = np.random.randn(20, n_voxels)
            subj_data = shared @ proj + np.random.randn(n_samples, n_voxels) * 0.5
            data.append(subj_data)

        return data

    def test_creation_srm(self):
        """Test AlignStep creation with SRM."""
        step = AlignStep(method="srm", n_features=10)

        assert step.method == "srm"
        assert step.n_features == 10
        assert step.scheme == "global"

    def test_creation_hyperalignment(self):
        """Test AlignStep creation with HyperAlignment."""
        step = AlignStep(method="hyperalignment")

        assert step.method == "hyperalignment"
        assert step.invertible is True

    def test_unknown_method_raises(self):
        """Test unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            AlignStep(method="unknown")

    def test_searchlight_not_implemented(self):
        """Test searchlight scheme raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            AlignStep(method="srm", scheme="searchlight")

    def test_piecewise_not_implemented(self):
        """Test piecewise scheme raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            AlignStep(method="srm", scheme="piecewise")

    def test_invertible_property(self):
        """Test invertible property."""
        assert AlignStep(method="srm").invertible is False
        assert AlignStep(method="hyperalignment").invertible is True

    def test_fit_srm(self, multi_subject_data):
        """Test fitting SRM."""
        step = AlignStep(method="srm", n_features=10, n_iter=3)
        fitted = step.fit(multi_subject_data)

        assert isinstance(fitted, FittedAlign)
        assert fitted.method == "srm"
        assert hasattr(fitted.model, "w_")  # SRM fitted

    def test_fit_hyperalignment(self, multi_subject_data):
        """Test fitting HyperAlignment."""
        step = AlignStep(method="hyperalignment", n_iter=2)
        fitted = step.fit(multi_subject_data)

        assert isinstance(fitted, FittedAlign)
        assert fitted.method == "hyperalignment"
        assert hasattr(fitted.model, "s_")  # Template learned

    def test_transform_srm(self, multi_subject_data):
        """Test transforming with fitted SRM."""
        step = AlignStep(method="srm", n_features=10, n_iter=3)
        fitted = step.fit(multi_subject_data)

        # Transform uses pipeline convention: (samples, features)
        transformed = fitted.transform(multi_subject_data)

        assert len(transformed) == 4
        # SRM reduces to n_features dimensions: (samples, n_features)
        assert transformed[0].shape[1] == 10

    def test_transform_hyperalignment(self, multi_subject_data):
        """Test transforming with fitted HyperAlignment."""
        step = AlignStep(method="hyperalignment", n_iter=2)
        fitted = step.fit(multi_subject_data)

        # Transform uses pipeline convention: (samples, features)
        transformed = fitted.transform(multi_subject_data)

        assert len(transformed) == 4
        # HyperAlignment preserves dimensions: (samples, features)
        assert transformed[0].shape == multi_subject_data[0].shape


class TestFittedAlign:
    """Tests for FittedAlign class."""

    @pytest.fixture
    def fitted_srm(self):
        """Create a fitted SRM model."""
        np.random.seed(42)
        data = [np.random.randn(50, 100) for _ in range(3)]  # (samples, voxels)

        step = AlignStep(method="srm", n_features=10, n_iter=3)
        return step.fit(data)

    @pytest.fixture
    def fitted_hyperalign(self):
        """Create a fitted HyperAlignment model."""
        np.random.seed(42)
        data = [np.random.randn(50, 100) for _ in range(3)]

        step = AlignStep(method="hyperalignment", n_iter=2)
        return step.fit(data)

    def test_transform_new_subject_srm(self, fitted_srm):
        """Test transforming a new subject with SRM."""
        np.random.seed(123)
        # Pipeline convention: (samples, features)
        new_subject = np.random.randn(50, 100)  # (samples, voxels)

        aligned = fitted_srm.transform_new_subject(new_subject)

        # Should return aligned data: (samples, n_features)
        assert aligned.shape[0] == 50  # n_samples
        assert aligned.shape[1] == 10  # n_features

    def test_inverse_transform_hyperalignment(self, fitted_hyperalign):
        """Test inverse transform with HyperAlignment."""
        np.random.seed(42)
        # Pipeline convention: (samples, features)
        data = [np.random.randn(50, 100) for _ in range(3)]  # (samples, voxels)

        transformed = fitted_hyperalign.transform(data)
        reconstructed = fitted_hyperalign.inverse_transform(transformed)

        assert len(reconstructed) == 3
        assert reconstructed[0].shape == data[0].shape

    def test_inverse_transform_srm_raises(self, fitted_srm):
        """Test inverse transform with SRM raises error."""
        data = [np.random.randn(10, 50) for _ in range(3)]

        with pytest.raises(
            NotImplementedError, match="only supported for hyperalignment"
        ):
            fitted_srm.inverse_transform(data)


class TestMultiSubjectPipelineAlign:
    """Tests for MultiSubjectPipeline.align() integration."""

    @pytest.fixture
    def subject_data(self):
        """Create sample multi-subject data."""
        np.random.seed(42)
        return [np.random.randn(50, 100) for _ in range(4)]  # (samples, voxels)

    def test_align_method_exists(self, subject_data):
        """Test align() method exists on pipeline."""
        from nltools.pipelines.cv import CVScheme
        from nltools.pipelines.multi_subject import MultiSubjectPipeline

        cv = CVScheme(scheme="loso")
        pipeline = MultiSubjectPipeline(data=subject_data, cv=cv)

        assert hasattr(pipeline, "align")

    def test_align_returns_pipeline(self, subject_data):
        """Test align() returns new pipeline."""
        from nltools.pipelines.cv import CVScheme
        from nltools.pipelines.multi_subject import MultiSubjectPipeline

        cv = CVScheme(scheme="loso")
        p1 = MultiSubjectPipeline(data=subject_data, cv=cv)
        p2 = p1.align(method="srm", n_features=10)

        assert p1 is not p2
        assert p1.n_steps == 0
        assert p2.n_steps == 1

    def test_align_chaining(self, subject_data):
        """Test align() chains with other methods."""
        from nltools.pipelines.cv import CVScheme
        from nltools.pipelines.multi_subject import MultiSubjectPipeline

        cv = CVScheme(scheme="loso")
        pipeline = (
            MultiSubjectPipeline(data=subject_data, cv=cv)
            .normalize()
            .align(method="srm", n_features=10)
            .reduce(n_components=5)
        )

        assert pipeline.n_steps == 3

    def test_align_default_params(self, subject_data):
        """Test align() with default parameters."""
        from nltools.pipelines.cv import CVScheme
        from nltools.pipelines.multi_subject import MultiSubjectPipeline

        cv = CVScheme(scheme="loso")
        pipeline = MultiSubjectPipeline(data=subject_data, cv=cv).align()

        # Should have added an AlignStep with defaults
        assert pipeline.n_steps == 1


class TestAlignmentIntegration:
    """Integration tests for alignment in full workflows."""

    def test_srm_reduces_dimensions(self):
        """Test SRM actually reduces data dimensions."""
        np.random.seed(42)
        n_features = 20

        # Create data with more voxels than features
        # Pipeline convention: (samples, voxels)
        data = [np.random.randn(50, 100) for _ in range(3)]  # (samples, voxels)

        step = AlignStep(method="srm", n_features=n_features, n_iter=3)
        fitted = step.fit(data)

        # Transform uses pipeline convention: (samples, features)
        transformed = fitted.transform(data)

        # Should reduce to n_features: (samples, n_features)
        for t in transformed:
            assert t.shape[1] == n_features

    def test_hyperalignment_preserves_dimensions(self):
        """Test HyperAlignment preserves data dimensions."""
        np.random.seed(42)

        # Pipeline convention: (samples, features)
        data = [np.random.randn(50, 80) for _ in range(3)]

        step = AlignStep(method="hyperalignment", n_iter=2)
        fitted = step.fit(data)

        # Transform uses pipeline convention: (samples, features)
        transformed = fitted.transform(data)

        # Should preserve dimensions: (samples, features)
        for i, t in enumerate(transformed):
            assert t.shape == data[i].shape

    def test_alignment_improves_similarity(self):
        """Test that alignment increases inter-subject similarity."""
        np.random.seed(42)
        n_subjects = 4
        n_samples = 50
        n_voxels = 60

        # Create data with shared signal but different noise
        # Pipeline convention: (samples, features)
        shared = np.random.randn(n_samples, 20)
        data = []
        for _ in range(n_subjects):
            proj = np.random.randn(20, n_voxels)
            subj = shared @ proj + np.random.randn(n_samples, n_voxels) * 2
            data.append(subj)

        # Fit SRM
        step = AlignStep(method="srm", n_features=20, n_iter=5)
        fitted = step.fit(data)

        # Transform uses pipeline convention: (samples, features)
        transformed = fitted.transform(data)

        # Calculate mean pairwise correlation before and after
        def mean_pairwise_corr(arrays):
            corrs = []
            for i in range(len(arrays)):
                for j in range(i + 1, len(arrays)):
                    c = np.corrcoef(arrays[i].ravel(), arrays[j].ravel())[0, 1]
                    corrs.append(c)
            return np.mean(corrs)

        corr_before = mean_pairwise_corr(data)
        corr_after = mean_pairwise_corr(transformed)

        # Alignment should increase similarity
        assert corr_after > corr_before
