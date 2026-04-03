"""Tests for multi-subject pipeline (Phase 4).

Tests cover:
- MultiSubjectPipeline: LOSO and run-based CV
- BrainCollectionPipeline integration (requires test fixtures)
"""

import numpy as np
import pytest
from nltools.pipelines.cv import CVScheme
from nltools.pipelines.multi_subject import MultiSubjectPipeline

pytestmark = pytest.mark.slow


class TestMultiSubjectPipeline:
    """Tests for MultiSubjectPipeline class."""

    @pytest.fixture
    def subject_data(self):
        """Create sample multi-subject data."""
        np.random.seed(42)
        # 6 subjects, 30 observations each, 20 features
        return [np.random.randn(30, 20) for _ in range(6)]

    @pytest.fixture
    def balanced_labels(self):
        """Labels for 6 subjects with balanced classes."""
        return np.array([0, 1, 0, 1, 0, 1])

    def test_creation(self, subject_data):
        """Test pipeline creation."""
        cv = CVScheme(scheme="loso")
        pipeline = MultiSubjectPipeline(data=subject_data, cv=cv)

        assert pipeline.n_subjects == 6
        assert pipeline.n_steps == 0

    def test_add_step_immutable(self, subject_data):
        """Test _add_step returns new pipeline."""
        cv = CVScheme(scheme="loso")
        p1 = MultiSubjectPipeline(data=subject_data, cv=cv)
        p2 = p1.normalize()

        assert p1.n_steps == 0
        assert p2.n_steps == 1
        assert p1 is not p2

    def test_chaining(self, subject_data):
        """Test transform chaining."""
        cv = CVScheme(scheme="loso")
        pipeline = (
            MultiSubjectPipeline(data=subject_data, cv=cv)
            .normalize()
            .reduce(n_components=10)
            .pipe(None)  # Will fail if called, but tests chaining
        )

        assert pipeline.n_steps == 3

    def test_predict_requires_cv(self, subject_data, balanced_labels):
        """Test predict raises without CV."""
        pipeline = MultiSubjectPipeline(data=subject_data, cv=None)

        with pytest.raises(ValueError, match="requires CV context"):
            pipeline.predict(balanced_labels)

    def test_loso_basic(self, subject_data, balanced_labels):
        """Test basic LOSO CV."""
        cv = CVScheme(scheme="loso")
        result = MultiSubjectPipeline(data=subject_data, cv=cv).predict(
            balanced_labels, algorithm="svm"
        )

        assert result.n_folds == 6  # One per subject
        assert len(result.scores) == 6
        assert result.mean_score is not None

    def test_loso_with_transforms(self, subject_data, balanced_labels):
        """Test LOSO with preprocessing."""
        cv = CVScheme(scheme="loso")
        result = (
            MultiSubjectPipeline(data=subject_data, cv=cv)
            .normalize()
            .reduce(n_components=10)
            .predict(balanced_labels, algorithm="svm")
        )

        assert result.n_folds == 6
        # Should complete without error

    def test_run_cv_requires_groups(self, subject_data, balanced_labels):
        """Test run CV raises without groups."""
        cv = CVScheme(scheme="loro")
        pipeline = MultiSubjectPipeline(data=subject_data, cv=cv)

        with pytest.raises(ValueError, match="requires groups"):
            pipeline.predict(balanced_labels)

    def test_run_cv_with_groups(self, subject_data):
        """Test run-based CV with groups."""
        # Create run labels (3 runs per subject)
        runs_per_subject = 10
        groups = np.tile(np.arange(3), runs_per_subject)

        # Create labels matching pooled observations
        n_obs = 30 * 6  # 30 obs per subject, 6 subjects
        y = np.random.randint(0, 2, n_obs)

        cv = CVScheme(scheme="loro")
        result = MultiSubjectPipeline(data=subject_data, cv=cv, groups=groups).predict(
            y, algorithm="ridge"
        )

        assert result.n_folds == 3  # 3 runs
        assert len(result.scores) == 3

    def test_different_algorithms(self, subject_data, balanced_labels):
        """Test different prediction algorithms."""
        cv = CVScheme(scheme="loso")
        pipeline = MultiSubjectPipeline(data=subject_data, cv=cv)

        for algo in ["ridge", "svm", "logistic"]:
            result = pipeline.predict(balanced_labels, algorithm=algo)
            assert result.n_folds == 6

    def test_repr(self, subject_data):
        """Test string representation."""
        cv = CVScheme(scheme="loso")
        pipeline = MultiSubjectPipeline(data=subject_data, cv=cv)
        r = repr(pipeline)

        assert "MultiSubjectPipeline" in r
        assert "n_subjects=6" in r


class TestMultiSubjectCVResult:
    """Tests for multi-subject CV results."""

    @pytest.fixture
    def cv_result(self):
        """Run a simple LOSO CV and return result."""
        np.random.seed(42)
        data = [np.random.randn(20, 10) for _ in range(4)]
        y = np.array([0, 1, 0, 1])
        cv = CVScheme(scheme="loso")

        return MultiSubjectPipeline(data=data, cv=cv).predict(y, algorithm="svm")

    def test_scores_array(self, cv_result):
        """Test scores property returns array."""
        assert isinstance(cv_result.scores, np.ndarray)
        assert len(cv_result.scores) == 4

    def test_mean_std_score(self, cv_result):
        """Test mean and std score properties."""
        assert isinstance(cv_result.mean_score, float)
        assert isinstance(cv_result.std_score, float)
        assert cv_result.std_score >= 0

    def test_n_folds(self, cv_result):
        """Test n_folds property."""
        assert cv_result.n_folds == 4

    def test_repr(self, cv_result):
        """Test string representation."""
        r = repr(cv_result)
        assert "CVResult" in r
        assert "n_folds=4" in r
