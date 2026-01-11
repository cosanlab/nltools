"""Tests for pipeline base infrastructure (Phase 1).

Tests cover:
- CVScheme: kfold, loso, loro, bootstrap, permutation splits
- NestedCVScheme: nested cross-validation
- Pipeline: creation, step accumulation, immutability
- FittedStack: collection and invertibility checks
- PermutationResult: permutation testing results
"""

import numpy as np
import pytest

from nltools.pipelines.base import FittedStack, Pipeline
from nltools.pipelines.cv import CVScheme, NestedCVScheme
from nltools.pipelines.results import CVResult, FoldResult, PermutationResult


class TestCVScheme:
    """Tests for CVScheme cross-validation configuration."""

    def test_kfold_default(self):
        """Test kfold defaults to 5 splits."""
        cv = CVScheme(scheme="kfold")
        assert cv.k == 5
        assert cv.n_splits() == 5

    def test_kfold_custom_k(self):
        """Test kfold with custom k."""
        cv = CVScheme(scheme="kfold", k=3)
        assert cv.k == 3
        assert cv.n_splits() == 3

    def test_kfold_split(self):
        """Test kfold generates correct splits."""
        cv = CVScheme(scheme="kfold", k=5, random_state=42)
        data = np.arange(100)

        splits = list(cv.split(data))
        assert len(splits) == 5

        # Each sample should appear in exactly one test set
        all_test_idx = np.concatenate([test for _, test in splits])
        assert len(all_test_idx) == 100
        assert len(np.unique(all_test_idx)) == 100

    def test_loso_requires_groups(self):
        """Test loso raises error without groups."""
        cv = CVScheme(scheme="loso")
        data = np.arange(10)

        with pytest.raises(ValueError, match="loso requires groups"):
            list(cv.split(data))

    def test_loso_split(self):
        """Test loso generates one split per group."""
        cv = CVScheme(scheme="loso")
        data = np.arange(30)
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 3 + [0, 1, 2])

        splits = list(cv.split(data, groups=groups))
        assert len(splits) == 3  # 3 unique groups

        for train_idx, test_idx in splits:
            # Test set contains all samples from one group
            test_groups = groups[test_idx]
            assert len(np.unique(test_groups)) == 1

    def test_loro_split(self):
        """Test loro (leave-one-run-out) generates correct splits."""
        cv = CVScheme(scheme="loro", split_by="runs")
        data = np.arange(40)
        runs = np.repeat([0, 1, 2, 3], 10)

        splits = list(cv.split(data, groups=runs))
        assert len(splits) == 4  # 4 runs

    def test_bootstrap_split(self):
        """Test bootstrap generates train/test with OOB."""
        cv = CVScheme(scheme="bootstrap", n=10, random_state=42)
        data = np.arange(50)

        splits = list(cv.split(data))
        # May be < n if some iterations have no OOB samples
        assert len(splits) <= 10
        assert len(splits) >= 5  # Should get most iterations

        for train_idx, test_idx in splits:
            # Train should have n_samples (with replacement)
            assert len(train_idx) == 50
            # Test (OOB) should have at least 1 sample
            assert len(test_idx) >= 1
            # Train may have duplicates (bootstrap)
            assert len(np.unique(train_idx)) < len(train_idx)

    def test_permutation_split(self):
        """Test permutation generates train indices and permuted indices."""
        cv = CVScheme(scheme="permutation", n=10, random_state=42)
        data = np.arange(50)

        splits = list(cv.split(data))
        assert len(splits) == 10  # Exactly n permutations

        for train_idx, perm_idx in splits:
            # Train indices should be original order
            np.testing.assert_array_equal(train_idx, np.arange(50))
            # Permuted indices should be a permutation
            assert len(perm_idx) == 50
            assert set(perm_idx) == set(range(50))
            # Should be shuffled (very unlikely to be in order)
            assert not np.array_equal(perm_idx, train_idx)

    def test_permutation_n_splits(self):
        """Test permutation n_splits returns n."""
        cv = CVScheme(scheme="permutation", n=100)
        assert cv.n_splits() == 100

    def test_permutation_repr(self):
        """Test permutation string representation."""
        cv = CVScheme(scheme="permutation", n=500)
        r = repr(cv)
        assert "permutation" in r
        assert "500" in r

    def test_permutation_reproducibility(self):
        """Test permutation splits are reproducible with same seed."""
        cv1 = CVScheme(scheme="permutation", n=5, random_state=42)
        cv2 = CVScheme(scheme="permutation", n=5, random_state=42)
        data = np.arange(30)

        splits1 = list(cv1.split(data))
        splits2 = list(cv2.split(data))

        for (t1, p1), (t2, p2) in zip(splits1, splits2):
            np.testing.assert_array_equal(p1, p2)

    def test_is_loso_property(self):
        """Test is_loso property."""
        assert CVScheme(scheme="loso").is_loso is True
        assert CVScheme(scheme="kfold").is_loso is False

    def test_is_loro_property(self):
        """Test is_loro property."""
        assert CVScheme(scheme="loro").is_loro is True
        assert CVScheme(scheme="kfold").is_loro is False

    def test_n_splits_with_groups(self):
        """Test n_splits returns correct count for group schemes."""
        cv = CVScheme(scheme="loso")
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        assert cv.n_splits(groups=groups) == 4

    def test_repr(self):
        """Test string representation."""
        cv = CVScheme(scheme="kfold", k=10)
        assert "kfold" in repr(cv)
        assert "10" in repr(cv)


class TestPipeline:
    """Tests for Pipeline base class."""

    def test_creation(self):
        """Test pipeline creation with data and CV."""
        data = np.random.randn(100, 50)
        cv = CVScheme(scheme="kfold", k=5)

        pipeline = Pipeline(data, cv=cv)

        assert pipeline.data is data
        assert pipeline.cv is cv
        assert pipeline.n_steps == 0
        assert pipeline._is_lazy is False

    def test_add_step_immutable(self):
        """Test _add_step returns new pipeline (immutable)."""

        class DummyStep:
            invertible = True

            def fit(self, data):
                return self

        data = np.random.randn(50, 20)
        cv = CVScheme(scheme="kfold", k=3)
        p1 = Pipeline(data, cv=cv)

        p2 = p1._add_step(DummyStep())

        # Original unchanged
        assert p1.n_steps == 0
        # New pipeline has step
        assert p2.n_steps == 1
        # Different objects
        assert p1 is not p2

    def test_step_accumulation(self):
        """Test steps accumulate correctly."""

        class DummyStep:
            invertible = True

            def fit(self, data):
                return self

        data = np.random.randn(50, 20)
        cv = CVScheme(scheme="kfold", k=3)

        p = Pipeline(data, cv=cv)
        p = p._add_step(DummyStep())
        p = p._add_step(DummyStep())
        p = p._add_step(DummyStep())

        assert p.n_steps == 3

    def test_predict_requires_cv(self):
        """Test predict raises error without CV."""
        data = np.random.randn(50, 20)
        pipeline = Pipeline(data, cv=None)

        with pytest.raises(ValueError, match="requires CV context"):
            pipeline.predict(np.arange(50))

    def test_split_data_numpy(self):
        """Test _split_data works for numpy arrays."""
        data = np.arange(100).reshape(10, 10)
        cv = CVScheme(scheme="kfold", k=2)
        pipeline = Pipeline(data, cv=cv)

        train_idx = np.array([0, 1, 2, 3, 4])
        test_idx = np.array([5, 6, 7, 8, 9])

        train, test = pipeline._split_data(train_idx, test_idx)

        assert train.shape == (5, 10)
        assert test.shape == (5, 10)
        np.testing.assert_array_equal(train, data[:5])
        np.testing.assert_array_equal(test, data[5:])

    def test_copy(self):
        """Test pipeline copy creates independent instance."""
        data = np.random.randn(50, 20)
        cv = CVScheme(scheme="kfold", k=3)
        p1 = Pipeline(data, cv=cv)

        p2 = p1.copy()

        assert p1 is not p2
        assert p1.data is p2.data  # Shallow copy shares data
        assert p1.cv is p2.cv

    def test_repr(self):
        """Test string representation."""
        data = np.random.randn(50, 20)
        cv = CVScheme(scheme="kfold", k=3)
        pipeline = Pipeline(data, cv=cv)

        r = repr(pipeline)
        assert "Pipeline" in r
        assert "steps=" in r


class TestFittedStack:
    """Tests for FittedStack collection."""

    def test_empty_stack(self):
        """Test empty stack properties."""
        stack = FittedStack()

        assert len(stack) == 0
        assert stack.is_fully_invertible is True  # vacuously true

    def test_append(self):
        """Test appending fitted transforms."""

        class DummyFitted:
            def transform(self, data):
                return data

            def inverse_transform(self, data):
                return data

        stack = FittedStack()
        stack.append(DummyFitted())
        stack.append(DummyFitted())

        assert len(stack) == 2

    def test_inverse_transform(self):
        """Test inverse transform applies in reverse order."""
        call_order = []

        class FittedA:
            def transform(self, data):
                return data

            def inverse_transform(self, data):
                call_order.append("A")
                return data + 1

        class FittedB:
            def transform(self, data):
                return data

            def inverse_transform(self, data):
                call_order.append("B")
                return data * 2

        stack = FittedStack()
        stack.append(FittedA())
        stack.append(FittedB())

        result = stack.inverse_transform(np.array([1.0]))

        # B should be called before A (reverse order)
        assert call_order == ["B", "A"]
        # Result: (1 * 2) + 1 = 3
        np.testing.assert_array_equal(result, [3.0])

    def test_is_fully_invertible_false(self):
        """Test invertibility check when step lacks inverse."""

        class Invertible:
            def transform(self, data):
                return data

            def inverse_transform(self, data):
                return data

        class NotInvertible:
            def transform(self, data):
                return data

        stack = FittedStack()
        stack.append(Invertible())
        assert stack.is_fully_invertible is True

        stack.append(NotInvertible())
        assert stack.is_fully_invertible is False

    def test_iteration(self):
        """Test iterating over stack."""

        class DummyFitted:
            def transform(self, data):
                return data

        stack = FittedStack()
        f1, f2, f3 = DummyFitted(), DummyFitted(), DummyFitted()
        stack.append(f1)
        stack.append(f2)
        stack.append(f3)

        items = list(stack)
        assert items == [f1, f2, f3]


class TestNestedCVScheme:
    """Tests for NestedCVScheme nested cross-validation."""

    def test_creation(self):
        """Test nested CV scheme creation."""
        outer = CVScheme(scheme="kfold", k=5)
        inner = CVScheme(scheme="kfold", k=3)
        nested = NestedCVScheme(outer=outer, inner=inner)

        assert nested.outer is outer
        assert nested.inner is inner

    def test_kfold_nested_split(self):
        """Test nested k-fold generates correct structure."""
        outer = CVScheme(scheme="kfold", k=3, random_state=42)
        inner = CVScheme(scheme="kfold", k=2, random_state=42)
        nested = NestedCVScheme(outer=outer, inner=inner)

        data = np.arange(60)

        outer_count = 0
        for outer_train, outer_test, inner_iter in nested.split(data):
            outer_count += 1

            # Outer split should partition data
            assert len(outer_train) + len(outer_test) == 60
            assert len(set(outer_train) & set(outer_test)) == 0

            # Inner iterator should generate splits within outer train
            inner_count = 0
            for inner_train, inner_val in inner_iter:
                inner_count += 1
                # Inner indices are relative to outer_train
                assert max(inner_train) < len(outer_train)
                assert max(inner_val) < len(outer_train)
                # Inner should partition outer training data
                assert len(inner_train) + len(inner_val) == len(outer_train)

            assert inner_count == 2  # 2-fold inner

        assert outer_count == 3  # 3-fold outer

    def test_loso_outer_kfold_inner(self):
        """Test LOSO outer with k-fold inner."""
        outer = CVScheme(scheme="loso")
        inner = CVScheme(scheme="kfold", k=3, random_state=42)
        nested = NestedCVScheme(outer=outer, inner=inner)

        data = np.arange(40)
        groups = np.repeat([0, 1, 2, 3], 10)  # 4 subjects, 10 samples each

        splits = list(nested.split(data, groups=groups))
        assert len(splits) == 4  # 4 subjects

        for outer_train, outer_test, inner_iter in splits:
            # Test set should have samples from one subject
            assert len(outer_test) == 10
            # Inner loop within outer training set
            inner_splits = list(inner_iter)
            assert len(inner_splits) == 3  # 3-fold inner

    def test_n_outer_splits(self):
        """Test n_outer_splits returns correct count."""
        outer = CVScheme(scheme="kfold", k=5)
        inner = CVScheme(scheme="kfold", k=3)
        nested = NestedCVScheme(outer=outer, inner=inner)

        assert nested.n_outer_splits() == 5

    def test_n_inner_splits(self):
        """Test n_inner_splits returns correct count."""
        outer = CVScheme(scheme="kfold", k=5)
        inner = CVScheme(scheme="kfold", k=3)
        nested = NestedCVScheme(outer=outer, inner=inner)

        assert nested.n_inner_splits() == 3

    def test_n_splits_with_groups(self):
        """Test n_splits with group-based schemes."""
        outer = CVScheme(scheme="loso")
        inner = CVScheme(scheme="kfold", k=3)
        nested = NestedCVScheme(outer=outer, inner=inner)

        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        assert nested.n_outer_splits(groups=groups) == 4

    def test_inner_groups(self):
        """Test inner groups are correctly indexed."""
        outer = CVScheme(scheme="kfold", k=2, random_state=42)
        inner = CVScheme(scheme="loso")
        nested = NestedCVScheme(outer=outer, inner=inner)

        data = np.arange(20)
        groups = np.repeat([0, 1, 2, 3], 5)  # 4 groups

        for outer_train, outer_test, inner_iter in nested.split(
            data, groups=groups, inner_groups=groups
        ):
            # Inner iterator should use groups for LOSO
            inner_splits = list(inner_iter)
            # Number of inner splits depends on groups in outer train
            assert len(inner_splits) >= 1

    def test_repr(self):
        """Test string representation."""
        outer = CVScheme(scheme="kfold", k=5)
        inner = CVScheme(scheme="kfold", k=3)
        nested = NestedCVScheme(outer=outer, inner=inner)

        r = repr(nested)
        assert "NestedCVScheme" in r
        assert "outer=" in r
        assert "inner=" in r


class TestPermutationResult:
    """Tests for PermutationResult permutation testing results."""

    def _make_dummy_cv_result(self, score: float) -> CVResult:
        """Create a dummy CVResult for testing."""
        fold = FoldResult(
            score=score,
            predictions=np.array([1.0, 2.0, 3.0]),
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([3, 4, 5]),
            fitted_stack=None,
        )
        return CVResult(fold_results=[fold], pipeline=None)

    def test_creation(self):
        """Test PermutationResult creation."""
        observed = self._make_dummy_cv_result(0.8)
        null_dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = PermutationResult(
            observed=observed,
            null_distribution=null_dist,
            p_value=0.05,
            n_permutations=5,
        )

        assert result.observed is observed
        np.testing.assert_array_equal(result.null_distribution, null_dist)
        assert result.p_value == 0.05
        assert result.n_permutations == 5

    def test_from_scores(self):
        """Test PermutationResult.from_scores factory method."""
        observed = self._make_dummy_cv_result(0.8)
        # Null scores: 0.1, 0.2, 0.3, 0.4, 0.5 - all below observed 0.8
        null_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = PermutationResult.from_scores(observed, null_scores)

        assert result.n_permutations == 5
        # p = (0 + 1) / (5 + 1) = 1/6
        assert result.p_value == pytest.approx(1 / 6)

    def test_from_scores_all_exceed(self):
        """Test p-value when all null scores exceed observed."""
        observed = self._make_dummy_cv_result(0.1)
        null_scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        result = PermutationResult.from_scores(observed, null_scores)

        # p = (5 + 1) / (5 + 1) = 1.0
        assert result.p_value == pytest.approx(1.0)

    def test_from_scores_half_exceed(self):
        """Test p-value when half of null scores exceed observed."""
        observed = self._make_dummy_cv_result(0.5)
        null_scores = np.array([0.3, 0.4, 0.6, 0.7])  # 2 >= 0.5

        result = PermutationResult.from_scores(observed, null_scores)

        # p = (2 + 1) / (4 + 1) = 3/5 = 0.6
        assert result.p_value == pytest.approx(0.6)

    def test_observed_score_property(self):
        """Test observed_score property."""
        observed = self._make_dummy_cv_result(0.75)
        result = PermutationResult(
            observed=observed,
            null_distribution=np.array([0.1, 0.2]),
            p_value=0.05,
            n_permutations=2,
        )

        assert result.observed_score == 0.75

    def test_null_mean(self):
        """Test null_mean property."""
        observed = self._make_dummy_cv_result(0.8)
        null_dist = np.array([0.2, 0.4, 0.6])

        result = PermutationResult(
            observed=observed,
            null_distribution=null_dist,
            p_value=0.05,
            n_permutations=3,
        )

        assert result.null_mean == pytest.approx(0.4)

    def test_null_std(self):
        """Test null_std property."""
        observed = self._make_dummy_cv_result(0.8)
        null_dist = np.array([0.2, 0.4, 0.6])

        result = PermutationResult(
            observed=observed,
            null_distribution=null_dist,
            p_value=0.05,
            n_permutations=3,
        )

        expected_std = np.std([0.2, 0.4, 0.6])
        assert result.null_std == pytest.approx(expected_std)

    def test_z_score(self):
        """Test z_score property."""
        observed = self._make_dummy_cv_result(0.8)
        null_dist = np.array([0.2, 0.4, 0.6])  # mean=0.4, std~0.163

        result = PermutationResult(
            observed=observed,
            null_distribution=null_dist,
            p_value=0.05,
            n_permutations=3,
        )

        expected_z = (0.8 - 0.4) / np.std(null_dist)
        assert result.z_score == pytest.approx(expected_z)

    def test_z_score_zero_std(self):
        """Test z_score when null distribution has zero variance."""
        observed = self._make_dummy_cv_result(0.8)
        null_dist = np.array([0.5, 0.5, 0.5])  # constant

        result = PermutationResult(
            observed=observed,
            null_distribution=null_dist,
            p_value=0.05,
            n_permutations=3,
        )

        assert result.z_score == float("inf")

    def test_summary(self):
        """Test summary string format."""
        observed = self._make_dummy_cv_result(0.8)
        null_dist = np.array([0.1, 0.2, 0.3])

        result = PermutationResult.from_scores(observed, null_dist)
        summary = result.summary()

        assert "PermutationResult" in summary
        assert "permutations" in summary
        assert "Observed score" in summary
        assert "Null mean" in summary
        assert "Z-score" in summary
        assert "p-value" in summary

    def test_repr(self):
        """Test string representation."""
        observed = self._make_dummy_cv_result(0.8)
        null_dist = np.array([0.1, 0.2, 0.3])

        result = PermutationResult.from_scores(observed, null_dist)
        r = repr(result)

        assert "PermutationResult" in r
        assert "observed_score" in r
        assert "p_value" in r
        assert "n_permutations" in r
