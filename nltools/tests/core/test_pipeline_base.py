"""Tests for pipeline base infrastructure.

Tests cover:
- CVScheme: kfold, loso, loro, bootstrap, permutation splits
- FittedStack: collection and invertibility checks
"""

import numpy as np
import pytest

from nltools.pipelines.base import FittedStack
from nltools.pipelines.cv import CVScheme


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
