"""Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.
"""

from dataclasses import dataclass, field
from typing import Any, Literal
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

CVSchemeType = Literal["kfold", "loso", "loro", "bootstrap", "permutation"]


@dataclass
class CVScheme:
    """Cross-validation scheme configuration.

    Supports multiple CV strategies:
    - kfold: k-fold cross-validation
    - loso: leave-one-subject-out (for multi-subject)
    - loro: leave-one-run-out
    - bootstrap: bootstrap resampling
    - permutation: permutation testing (shuffles targets)

    Args:
        k: Number of folds (for kfold scheme). Defaults to 5 if scheme is 'kfold'.
        scheme: CV scheme type. One of 'kfold', 'loso', 'loro', 'bootstrap', or 'permutation'.
        split_by: Attribute to split by ('runs', 'subjects', 'sessions').
            Used for documentation purposes with loso/loro schemes.
        n: Number of bootstrap iterations (for bootstrap scheme). Defaults to 1000.
        random_state: Random seed for reproducibility. If provided, sets the
            numpy random seed during initialization.

    Examples:
        >>> # 5-fold cross-validation
        >>> cv = CVScheme(scheme='kfold', k=5)
        >>> for train_idx, test_idx in cv.split(data):
        ...     # train and evaluate model
        ...     pass

        >>> # Leave-one-subject-out
        >>> cv = CVScheme(scheme='loso', split_by='subjects')
        >>> for train_idx, test_idx in cv.split(data, groups=subject_ids):
        ...     pass

        >>> # Bootstrap with 500 iterations
        >>> cv = CVScheme(scheme='bootstrap', n=500, random_state=42)

        >>> # Permutation testing with 1000 permutations
        >>> cv = CVScheme(scheme='permutation', n=1000, random_state=42)
    """

    k: int | None = None
    scheme: CVSchemeType = "kfold"
    split_by: str | None = None
    n: int = 1000  # For bootstrap
    random_state: int | None = None

    # Internal RNG state - initialized in __post_init__, excluded from init/repr
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize defaults and validate configuration."""
        if self.scheme == "kfold" and self.k is None:
            self.k = 5  # Default to 5-fold

        # Use a Generator for reproducible randomness instead of global state
        if self.random_state is not None:
            self._rng = np.random.default_rng(self.random_state)
        else:
            self._rng = np.random.default_rng()

    @property
    def is_loso(self) -> bool:
        """Check if this is leave-one-subject-out.

        Returns:
            True if scheme is 'loso', False otherwise.
        """
        return self.scheme == "loso"

    @property
    def is_loro(self) -> bool:
        """Check if this is leave-one-run-out.

        Returns:
            True if scheme is 'loro', False otherwise.
        """
        return self.scheme == "loro"

    def split(
        self, data: Any, groups: NDArray[np.intp] | None = None
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Generate train/test indices for each fold.

        Args:
            data: Data to split (used for length). Can be any object with
                __len__ or a numpy array with shape attribute.
            groups: Group labels for grouped CV (runs, subjects, etc.).
                Required for 'loso' and 'loro' schemes.

        Yields:
            Tuple of (train_indices, test_indices) for each fold.

        Raises:
            ValueError: If scheme is 'loso' or 'loro' and groups is None,
                or if scheme is unknown.
        """
        n_samples = len(data) if hasattr(data, "__len__") else data.shape[0]

        if self.scheme == "kfold":
            yield from self._kfold_split(n_samples)
        elif self.scheme == "loso" or self.scheme == "loro":
            if groups is None:
                raise ValueError(f"{self.scheme} requires groups parameter")
            yield from self._group_split(n_samples, groups)
        elif self.scheme == "bootstrap":
            yield from self._bootstrap_split(n_samples)
        elif self.scheme == "permutation":
            yield from self._permutation_split(n_samples)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    def _kfold_split(
        self, n_samples: int
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Generate k-fold splits.

        Args:
            n_samples: Total number of samples.

        Yields:
            Tuple of (train_indices, test_indices) for each fold.

        Raises:
            ValueError: If k is not set (should not happen if scheme is 'kfold').
        """
        if self.k is None:
            raise ValueError("k must be set for kfold scheme")

        k = self.k  # Local binding for type narrowing

        indices = np.arange(n_samples)
        if self.random_state is not None:
            self._rng.shuffle(indices)

        fold_sizes = np.full(k, n_samples // k, dtype=np.intp)
        fold_sizes[: n_samples % k] += 1

        current = 0
        for fold_size in fold_sizes:
            test_idx = indices[current : current + fold_size]
            train_idx = np.concatenate(
                [indices[:current], indices[current + fold_size :]]
            )
            yield train_idx, test_idx
            current += fold_size

    def _group_split(
        self, n_samples: int, groups: NDArray[np.intp]
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Generate leave-one-group-out splits.

        Args:
            n_samples: Total number of samples.
            groups: Array of group labels for each sample.

        Yields:
            Tuple of (train_indices, test_indices) for each unique group.
        """
        unique_groups = np.unique(groups)
        indices = np.arange(n_samples)

        for group in unique_groups:
            test_mask = groups == group
            train_mask = ~test_mask
            yield indices[train_mask], indices[test_mask]

    def _bootstrap_split(
        self, n_samples: int
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Generate bootstrap splits with out-of-bag as test.

        Uses bootstrap sampling with replacement for training, and
        out-of-bag (OOB) samples as the test set. Redraws (rather than
        skipping) any bootstrap sample whose OOB set is empty so that
        exactly ``n`` splits are always yielded, keeping ``split()`` and
        ``n_splits()`` in agreement.

        Args:
            n_samples: Total number of samples.

        Yields:
            Tuple of (train_indices, test_indices) for each bootstrap iteration.

        Raises:
            RuntimeError: If ``n`` non-empty-OOB draws cannot be produced
                (e.g. ``n_samples`` too small for reliable OOB testing).
        """
        indices = np.arange(n_samples)

        # Redraw empty-OOB samples so exactly self.n splits are produced. Cap
        # attempts to avoid an infinite loop for degenerate n_samples (e.g. 1),
        # where a non-empty OOB set is impossible.
        max_attempts = self.n * 100 + 100
        yielded = 0
        attempts = 0
        while yielded < self.n:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"bootstrap could not produce {self.n} splits with "
                    f"non-empty out-of-bag sets after {attempts} draws "
                    f"(n_samples={n_samples}); too few samples for reliable "
                    "out-of-bag testing."
                )
            attempts += 1
            train_idx = self._rng.choice(indices, size=n_samples, replace=True)
            # Out-of-bag samples as test
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[np.unique(train_idx)] = False
            test_idx = indices[oob_mask]

            # Redraw if no OOB samples (rare)
            if len(test_idx) == 0:
                continue

            yielded += 1
            yield train_idx, test_idx

    def _permutation_split(
        self, n_samples: int
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Generate permutation test splits.

        For each permutation, returns all sample indices for training and
        a permuted version of those indices for target shuffling.

        In permutation testing, the model is trained on all data but with
        shuffled target values. The train_idx represents the original order
        and perm_idx represents how to shuffle the target variable y.

        Args:
            n_samples: Total number of samples.

        Yields:
            Tuple of (train_indices, permuted_indices) for each permutation.
            train_indices: Original sample order (0, 1, 2, ..., n-1).
            permuted_indices: Shuffled indices to apply to target variable.

        Example:
            >>> cv = CVScheme(scheme='permutation', n=100, random_state=42)
            >>> for train_idx, perm_idx in cv.split(data):
            ...     y_shuffled = y[perm_idx]  # Shuffle targets
            ...     model.fit(X, y_shuffled)
            ...     null_score = model.score(X, y_shuffled)
        """
        indices = np.arange(n_samples, dtype=np.intp)

        for _ in range(self.n):
            # Permuted indices for shuffling targets
            perm_idx = self._rng.permutation(n_samples).astype(np.intp)
            yield indices.copy(), perm_idx

    def n_splits(self, data: Any = None, groups: NDArray[np.intp] | None = None) -> int:
        """Return number of splits.

        Args:
            data: Data to split (unused for most schemes, kept for API consistency).
            groups: Group labels for grouped CV. Required for 'loso' and 'loro'.

        Returns:
            Number of splits/folds that will be generated.

        Raises:
            ValueError: If scheme is 'loso' or 'loro' and groups is None,
                or if k is not set for kfold scheme.
        """
        if self.scheme == "kfold":
            if self.k is None:
                raise ValueError("k must be set for kfold scheme")
            return self.k
        if self.scheme in ("loso", "loro"):
            if groups is None:
                raise ValueError(f"{self.scheme} requires groups to count splits")
            return len(np.unique(groups))
        if self.scheme == "bootstrap":
            return self.n
        if self.scheme == "permutation":
            return self.n
        return 0

    def __repr__(self) -> str:
        """Return string representation of the CV scheme."""
        if self.scheme == "kfold":
            return f"CVScheme(scheme='kfold', k={self.k})"
        if self.scheme == "bootstrap":
            return f"CVScheme(scheme='bootstrap', n={self.n})"
        if self.scheme == "permutation":
            return f"CVScheme(scheme='permutation', n={self.n})"
        return f"CVScheme(scheme='{self.scheme}', split_by='{self.split_by}')"
