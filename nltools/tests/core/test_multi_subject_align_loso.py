"""
Tests for MultiSubjectPipeline.align() with LOSO CV.

This file documents the bug found in nltools-7j3g:
The _execute_loso method passes concatenated data to step.fit(), but
AlignStep.fit() expects a list of subjects.

The fix should modify _execute_loso to handle alignment steps differently,
passing them the list of training subjects instead of concatenated data.
"""

import numpy as np
import pytest
from nltools.pipelines.cv import CVScheme
from nltools.pipelines.multi_subject import MultiSubjectPipeline

pytestmark = pytest.mark.slow


class TestMultiSubjectAlignLOSO:
    """Tests for alignment + LOSO CV integration (nltools-7j3g)."""

    @pytest.fixture
    def subject_data(self):
        """Create sample multi-subject data."""
        np.random.seed(42)
        # 6 subjects, 30 observations each, 50 features
        return [np.random.randn(30, 50) for _ in range(6)]

    @pytest.fixture
    def subject_labels(self):
        """Subject-level labels for LOSO classification."""
        # 3 subjects per group
        return np.array([0, 1, 0, 1, 0, 1])

    def test_align_srm_with_loso_predict(self, subject_data, subject_labels):
        """Test SRM alignment followed by LOSO prediction.

        This test currently fails with:
        IndexError: tuple index out of range

        The issue is in _execute_loso (multi_subject.py:339):
            fitted = step.fit(self._concat_subjects(train_subjects))

        AlignStep.fit() expects a list of subjects, not concatenated data.
        The _ensure_voxels_first method tries to iterate over the list,
        but receives a single 2D array instead.

        Expected behavior: LOSO should work with alignment by:
        1. Fitting alignment on training subjects (as a list)
        2. Transforming each training subject individually
        3. Transforming the held-out test subject
        4. Then concatenating for prediction
        """
        cv = CVScheme(scheme="loso")

        # This should work but currently fails
        result = (
            MultiSubjectPipeline(data=subject_data, cv=cv)
            .normalize()
            .align(method="srm", n_features=10)
            .predict(y=subject_labels, algorithm="svm")
        )

        # Verify result structure
        assert result.n_folds == 6
        assert len(result.scores) == 6
        assert result.mean_score is not None

    def test_align_hyperalignment_with_loso_predict(self, subject_data, subject_labels):
        """Test HyperAlignment followed by LOSO prediction.

        Same bug as SRM - alignment step receives concatenated data
        instead of list of subjects.
        """
        cv = CVScheme(scheme="loso")

        result = (
            MultiSubjectPipeline(data=subject_data, cv=cv)
            .normalize()
            .align(method="hyperalignment", n_iter=2)
            .predict(y=subject_labels, algorithm="svm")
        )

        assert result.n_folds == 6
        assert len(result.scores) == 6

    def test_normalize_reduce_align_loso_chain(self, subject_data, subject_labels):
        """Test full pipeline chain: normalize -> reduce -> align -> predict.

        This tests that the fix handles mixed step types correctly.
        Regular steps (normalize, reduce) should use concatenated data,
        while alignment steps should use the list of subjects.
        """
        cv = CVScheme(scheme="loso")

        result = (
            MultiSubjectPipeline(data=subject_data, cv=cv)
            .normalize()
            .reduce(n_components=20)
            .align(method="srm", n_features=10)
            .predict(y=subject_labels, algorithm="svm")
        )

        assert result.n_folds == 6

    def test_alignment_transforms_held_out_subject(self, subject_data, subject_labels):
        """Test that alignment correctly transforms the held-out subject.

        When using LOSO with alignment:
        1. Alignment model is fit on N-1 training subjects
        2. Training subjects are transformed to shared space
        3. Held-out subject should be projected into the same shared space
           using the trained alignment model

        This requires using the alignment's project_new_subject capability.
        """
        cv = CVScheme(scheme="loso")

        # The alignment step should use new_subject="project" or similar
        # to handle the held-out subject correctly
        result = (
            MultiSubjectPipeline(data=subject_data, cv=cv)
            .align(method="srm", n_features=10)
            .predict(y=subject_labels, algorithm="svm")
        )

        # Each fold should have a valid score
        for score in result.scores:
            assert 0 <= score <= 1

    def test_alignment_without_cv_still_works(self, subject_data):
        """Verify that alignment without CV (terminal execution) still works.

        This is the existing working case - alignment with a terminal
        method that doesn't use LOSO CV.
        """
        # This should still work (no CV, just ISC)
        pipeline = (
            MultiSubjectPipeline(data=subject_data, cv=None)
            .normalize()
            .align(method="srm", n_features=10)
        )

        # Pipeline should have the alignment step
        assert pipeline.n_steps == 2


class TestAlignStepLOSOFix:
    """Tests for the proposed fix to _execute_loso."""

    @pytest.fixture
    def subject_data(self):
        np.random.seed(42)
        return [np.random.randn(30, 50) for _ in range(4)]

    def test_align_step_receives_list_not_array(self, subject_data):
        """Verify AlignStep.fit() receives list of subjects.

        The fix should ensure that when an AlignStep is encountered
        in _execute_loso, it receives train_subjects as a list,
        not self._concat_subjects(train_subjects).

        Proposed fix in _execute_loso:

        ```python
        for step in self.steps:
            if isinstance(step, AlignStep):
                # Alignment needs list of subjects
                fitted = step.fit(train_subjects)
                train_subjects = [fitted.transform(s) for s in train_subjects]
                test_subject = fitted.transform(test_subject)
            else:
                # Other steps work on concatenated data
                fitted = step.fit(self._concat_subjects(train_subjects))
                fitted_stack.append(fitted)
                train_subjects = [fitted.transform(s) for s in train_subjects]
                test_subject = fitted.transform(test_subject)
        ```
        """
        from nltools.pipelines.steps import AlignStep

        # Create alignment step
        step = AlignStep(method="srm", n_features=10, n_iter=3)

        # This should work - pass list of subjects
        fitted = step.fit(subject_data)

        # Verify it returns a fitted alignment
        assert fitted is not None

        # Transform works on list of subjects (returns list)
        transformed = fitted.transform(subject_data)
        assert len(transformed) == len(subject_data)
        for arr in transformed:
            assert arr.shape[1] == 10  # n_features

        # Transform new subject (single array) via transform_new_subject
        new_subject = subject_data[0]
        transformed_new = fitted.transform_new_subject(new_subject)
        assert transformed_new.shape[1] == 10  # n_features

    def test_mixed_steps_handled_correctly(self, subject_data):
        """Test that regular and alignment steps are handled differently.

        Regular steps (NormalizeStep, ReduceStep) can work on concatenated
        data or individual subjects. AlignStep specifically needs the list.
        """
        from nltools.pipelines.steps import AlignStep, NormalizeStep

        # NormalizeStep works fine either way
        norm_step = NormalizeStep()
        concat_data = np.vstack(subject_data)
        fitted_norm = norm_step.fit(concat_data)
        assert fitted_norm is not None

        # AlignStep needs the list
        align_step = AlignStep(method="srm", n_features=10, n_iter=3)
        fitted_align = align_step.fit(subject_data)
        assert fitted_align is not None
