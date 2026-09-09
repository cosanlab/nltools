"""Bootstrap resampling for simple statistics and fitted-model outputs.

Resamples observations with replacement `n_samples` times and summarizes the
resulting distribution (mean, standard deviation, Z, p, and confidence bounds)
with a memory-efficient online aggregator, `OnlineBootstrapStats`. The engines
cover simple aggregations ('mean', 'median', 'std', ...), ridge weights, and
ridge predictions, each with a CPU-parallel path (`n_jobs` workers) and, for
ridge, a batched GPU path. `BrainData.bootstrap` and `Adjacency.bootstrap` are
the user-facing entry points that pick an engine.
"""

import numpy as np
import warnings
from scipy.stats import norm

from .validation import (
    validate_tail_parameter,
    validate_bootstrap_method,
    validate_bootstrap_data,
    validate_percentiles,
    validate_array_shape,
    validate_array_shape_range,
    validate_shape_compatibility,
)
from ..random import generate_bootstrap_indices
from .utils import maybe_tqdm, make_progress_bar
from nltools.utils import find_stack_level


# Constants for supported methods
SIMPLE_METHODS = ["mean", "median", "std", "sum", "min", "max"]
FITTED_METHODS = ["weights", "predict"]  # For future use


def _p_from_z(z: np.ndarray, tail_internal: str) -> np.ndarray:
    """P-values from bootstrap Z-scores: 'two' → 2·(1−Φ(|z|)), 'upper' → 1−Φ(z).

    The single home of the bootstrap p formula. `OnlineBootstrapStats.get_results`
    applies it for every engine, and the facades thread their `tail` kwarg down
    to that call, so tail handling has exactly one mechanism.

    Args:
        z (np.ndarray): Bootstrap Z-scores (mean / std).
        tail_internal (str): `'two'` or `'upper'`, as returned by
            `validate_tail_parameter`.

    Returns:
        np.ndarray: P-values, same shape as `z`.
    """
    if tail_internal == "upper":
        return 1 - norm.cdf(z)
    return 2 * (1 - norm.cdf(np.abs(z)))


def _validate_bootstrap_method(method: str) -> None:
    """Validate a bootstrap method name.

    Args:
        method (str): Method name to validate.

    Raises:
        ValueError: If the method is not supported.
    """
    validate_bootstrap_method(method, SIMPLE_METHODS, FITTED_METHODS)


def _validate_bootstrap_data(data: np.ndarray, method: str) -> None:
    """Validate input data for bootstrapping; warn when there are fewer than 10 samples.

    Args:
        data (np.ndarray): Data to validate.
        method (str): Bootstrap method.

    Raises:
        ValueError: If the data is invalid (wrong shape, too few samples, etc.).
    """
    validate_bootstrap_data(data, method)

    # Warn if very few samples
    n_samples = data.shape[0] if data.ndim == 2 else len(data)
    if n_samples < 10:
        warnings.warn(
            f"Only {n_samples} samples available. Bootstrap works best with n >= 30. "
            f"Results may be unreliable with very small sample sizes.",
            UserWarning,
            stacklevel=find_stack_level(),
        )


def _validate_n_samples(n_samples: int) -> None:
    """Validate the number of bootstrap iterations; warn when below 1000.

    Args:
        n_samples (int): Number of bootstrap iterations.

    Raises:
        TypeError: If `n_samples` is not an integer.
        ValueError: If `n_samples` is below 10.
    """
    if not isinstance(n_samples, (int, np.integer)):
        raise TypeError(f"n_samples must be an integer, got {type(n_samples).__name__}")

    if n_samples < 10:
        raise ValueError(
            f"n_samples must be at least 10, got {n_samples}. "
            f"Bootstrap requires many iterations for stable estimates. "
            f"Recommended: n_samples >= 1000 for confidence intervals."
        )

    # Warn if too few for reliable CIs
    if n_samples < 1000:
        warnings.warn(
            f"n_samples={n_samples} is low. For reliable confidence intervals, "
            f"use n_samples >= 1000. For hypothesis testing, use n_samples >= 5000.",
            UserWarning,
            stacklevel=find_stack_level(),
        )


def _validate_percentiles(percentiles: tuple) -> None:
    """Validate percentile values for confidence intervals.

    Args:
        percentiles (tuple[float, float]): Percentile values (lower, upper).

    Raises:
        ValueError: If the percentiles are invalid.
    """
    validate_percentiles(percentiles)


class OnlineBootstrapStats:
    """Memory-efficient online statistics aggregator for bootstrap samples.

    Accumulates the running mean and variance with Welford's algorithm, so the
    summary is numerically stable without holding every sample in memory.
    Optionally stores all samples for exact percentile confidence intervals.

    Args:
        shape (tuple[int, ...]): Shape of each bootstrap sample.
        save_samples (bool): If True, store all samples for exact percentile
            confidence intervals; if False, use the normal approximation (much
            more memory efficient). Defaults to False.
        percentiles (tuple[float, float]): Percentiles for confidence intervals,
            e.g. (2.5, 97.5) for a 95% CI. Defaults to (2.5, 97.5).

    Attributes:
        n (int): Number of samples seen so far.
        mean (np.ndarray): Running mean, shape `shape`.
        M2 (np.ndarray): Running sum of squared deviations from the mean.
        samples (list[np.ndarray] | None): Stored samples when `save_samples=True`,
            else None.

    Examples:
        ```python
        stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
        for _ in range(1000):
            stats.update(np.random.randn(100))
        results = stats.get_results()
        results.keys()  # → dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
        ```
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        save_samples: bool = False,
        percentiles: tuple[float, float] = (2.5, 97.5),
    ):
        self.shape = shape
        self.save_samples = save_samples
        self.percentiles = percentiles

        # Initialize Welford's algorithm variables
        self.n = 0  # Number of samples seen
        self.mean = np.zeros(shape, dtype=np.float64)  # Running mean
        self.M2 = np.zeros(shape, dtype=np.float64)  # Running sum of squared deviations

        # Optional sample storage
        self.samples = [] if save_samples else None

    def update(self, sample: np.ndarray) -> None:
        """Fold one bootstrap sample into the running statistics.

        Args:
            sample (np.ndarray): New bootstrap sample with shape matching `self.shape`.

        Raises:
            ValueError: If the sample's shape does not match `self.shape`.
        """
        # Ensure float64 for numerical precision
        sample = np.asarray(sample, dtype=np.float64)

        # Validate shape
        if sample.shape != self.shape:
            raise ValueError(
                f"Sample shape {sample.shape} does not match expected shape {self.shape}"
            )

        # Welford's algorithm for online mean and variance
        self.n += 1
        delta = sample - self.mean
        self.mean += delta / self.n
        delta2 = sample - self.mean
        self.M2 += delta * delta2

        # Store sample if requested
        if self.save_samples:
            self.samples.append(sample.copy())

    def get_results(self, tail: int | str = 2) -> dict[str, np.ndarray]:
        """Compute final bootstrap statistics.

        Args:
            tail (int | str): `2` or `'two'` for two-tailed (default); `1` or
                `'one'` for one-tailed (statistic > 0; negate the data for the
                other direction).

        Returns:
            dict[str, np.ndarray]: Keys 'mean' (bootstrap mean), 'std' (bootstrap
                standard deviation), 'Z' (z-scores, mean/std), 'p' (p-values per
                `tail`), 'ci_lower' and 'ci_upper' (confidence bounds; exact
                percentiles when samples were saved, else a normal approximation),
                and 'samples' (all samples, only when `save_samples=True`).

        Raises:
            ValueError: If fewer than 2 samples have been seen.

        Examples:
            ```python
            stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
            for _ in range(1000):
                stats.update(np.random.randn(100))
            results = stats.get_results()
            results.keys()  # → dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
            ```
        """
        if self.n < 2:
            raise ValueError(
                f"Need at least 2 bootstrap samples, got {self.n}. "
                "Cannot compute variance with fewer than 2 samples."
            )

        # Compute sample variance and standard deviation
        # Using n-1 for sample variance (Bessel's correction)
        variance = self.M2 / (self.n - 1)
        std = np.sqrt(variance)

        # Compute Z-scores (mean / std)
        # Use errstate to handle division by zero gracefully
        with np.errstate(invalid="ignore", divide="ignore"):
            z = self.mean / std

        p = _p_from_z(z, validate_tail_parameter(tail))

        # Build result dictionary
        result = {
            "mean": self.mean,
            "std": std,
            "Z": z,
            "p": p,
        }

        # Compute confidence intervals
        if self.save_samples and self.samples:
            # Exact percentile CIs from stored samples
            samples_array = np.array(self.samples)  # Shape: (n_samples, *shape)
            result["ci_lower"] = np.percentile(
                samples_array, self.percentiles[0], axis=0
            )
            result["ci_upper"] = np.percentile(
                samples_array, self.percentiles[1], axis=0
            )
            result["samples"] = samples_array
        else:
            # Normal approximation CIs
            # For (2.5, 97.5) percentiles → 95% CI → z_crit ≈ 1.96
            # General formula: z_crit = norm.ppf(1 - alpha/2)
            # where alpha = (100 - (upper - lower)) / 100
            alpha = (100 - (self.percentiles[1] - self.percentiles[0])) / 100
            z_crit = norm.ppf(1 - alpha / 2)

            result["ci_lower"] = self.mean - z_crit * std
            result["ci_upper"] = self.mean + z_crit * std

        return result


def _bootstrap_simple_method_worker(
    data: np.ndarray,
    method: str,
    indices: np.ndarray,
) -> np.ndarray:
    """Worker function for bootstrapping simple aggregation methods.

    Args:
        data (np.ndarray): Data to bootstrap, shape (n_samples, n_features).
        method (str): Aggregation method, one of 'mean', 'median', 'std', 'sum',
            'min', or 'max'.
        indices (np.ndarray): Bootstrap indices, shape (n_samples,).

    Returns:
        np.ndarray: Aggregated result, shape (n_features,).

    Raises:
        ValueError: If `method` is not one of the supported aggregations.
    """
    # Resample data
    data_boot = data[indices]

    # Apply aggregation method
    if method == "mean":
        return np.mean(data_boot, axis=0)
    if method == "median":
        return np.median(data_boot, axis=0)
    if method == "std":
        return np.std(data_boot, axis=0, ddof=1)
    if method == "sum":
        return np.sum(data_boot, axis=0)
    if method == "min":
        return np.min(data_boot, axis=0)
    if method == "max":
        return np.max(data_boot, axis=0)
    raise ValueError(f"Unsupported method: {method}")


def _bootstrap_simple_cpu_parallel(
    data: np.ndarray,
    method: str,
    n_samples: int = 5000,
    save_boots: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    percentiles: tuple[float, float] = (2.5, 97.5),
    tail: int | str = 2,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap a simple aggregation across CPU workers.

    Bootstrap indices are pre-generated from `random_state`, resamples run in
    parallel with joblib, and results are aggregated with `OnlineBootstrapStats`.

    Args:
        data (np.ndarray): Data to bootstrap, shape (n_samples, n_features) or
            (n_samples,).
        method (str): Aggregation method, one of 'mean', 'median', 'std', 'sum',
            'min', or 'max'.
        n_samples (int): Number of bootstrap iterations. Defaults to 5000.
        save_boots (bool): If True, store all bootstrap samples (memory
            intensive). Defaults to False.
        n_jobs (int): Number of CPU workers (-1 = all cores). Defaults to -1.
        random_state (int | None): Random seed for reproducibility.
        percentiles (tuple[float, float]): Percentiles for confidence intervals.
            Defaults to (2.5, 97.5).
        tail (int | str): `2` or `'two'` for two-tailed (default); `1` or `'one'`
            for one-tailed (statistic > 0).
        progress_bar (bool): Show a progress bar over iterations. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Results keyed by `'mean'` (bootstrap mean), `'std'`
            (bootstrap standard deviation), `'Z'` (z-scores, mean/std), `'p'`
            (p-values, per `tail`), `'ci_lower'` and `'ci_upper'` (lower and upper
            confidence bounds), `'samples'` (all samples; only if `save_boots=True`),
            and `'backend'` (backend used, e.g. `'cpu-parallel-8'`).

    Examples:
        ```python
        data = np.random.randn(100, 50)  # 100 samples, 50 features
        result = _bootstrap_simple_cpu_parallel(data, "mean", n_samples=1000)
        result["mean"].shape  # → (50,)
        result.keys()  # → dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', 'backend'])
        ```
    """
    from joblib import Parallel, delayed

    # Validate inputs
    _validate_bootstrap_method(method)
    _validate_n_samples(n_samples)
    _validate_percentiles(percentiles)
    validate_tail_parameter(tail)

    # Convert to array and validate
    data = np.asarray(data, dtype=np.float64)
    _validate_bootstrap_data(data, method)

    # Handle 1D input
    single_feature = data.ndim == 1
    if single_feature:
        data = data[:, np.newaxis]

    n_obs, n_features = data.shape
    output_shape = (n_features,) if not single_feature else ()

    # Pre-generate bootstrap indices (deterministic)
    all_indices = generate_bootstrap_indices(
        n_obs, n_samples, random_state=random_state
    )

    # Initialize online statistics aggregator
    stats = OnlineBootstrapStats(
        shape=output_shape if output_shape else (1,),
        save_samples=save_boots,
        percentiles=percentiles,
    )

    # Define worker function
    def _compute_one_bootstrap(idx):
        return _bootstrap_simple_method_worker(data, method, all_indices[idx])

    # Execute in parallel with progress bar
    bootstrap_samples = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_bootstrap)(i)
        for i in maybe_tqdm(
            range(n_samples),
            progress_bar=progress_bar,
            desc="Bootstrap iterations",
            unit="iter",
        )
    )

    # Aggregate results
    for sample in bootstrap_samples:
        if single_feature:
            sample = sample.flatten()
        stats.update(sample)

    # Get final results
    result = stats.get_results(tail)

    # Add backend info
    import multiprocessing

    actual_n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    result["backend"] = f"cpu-parallel-{actual_n_jobs}"

    # Remove samples if not requested
    if not save_boots:
        result.pop("samples", None)

    return result


def _refit_resample(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Refit ridge weights on one bootstrap resample with `alpha` held fixed.

    Both CPU ridge-bootstrap workers route through the package's single
    fixed-hyperparameter refit, so a resample cannot drift from the full-data
    fit numerically.

    Args:
        X (np.ndarray): Training features, shape (n_samples, n_features).
        y (np.ndarray): Training targets, shape (n_samples,) or
            (n_samples, n_voxels).
        indices (np.ndarray): Row indices of the resample.
        alpha (float | np.ndarray): Scalar or per-target regularization.

    Returns:
        np.ndarray: Weights, shape (n_features, n_voxels), or (n_features,)
            when `y` is one-dimensional.
    """
    from nltools.models.ridge import _refit_fixed_hyperparameters

    X_boot = np.asarray(X)[indices]
    y_boot = np.asarray(y)[indices]
    was_1d = y_boot.ndim == 1
    if was_1d:
        y_boot = y_boot[:, None]
    weights = _refit_fixed_hyperparameters([X_boot], y_boot, alpha)
    return weights[:, 0] if was_1d else weights


def _bootstrap_ridge_weights_worker(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Worker function for bootstrapping ridge weights.

    Calls the shared fixed-hyperparameter refit directly on numpy arrays, which
    is 10-100x faster than going through `BrainData`. The refit holds the
    selected `alpha` fixed: a bootstrap never reruns model selection.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features).
        y (np.ndarray): Target matrix, shape (n_samples, n_voxels).
        indices (np.ndarray): Bootstrap indices, shape (n_samples,).
        alpha (float | np.ndarray): Scalar or per-target regularization.

    Returns:
        np.ndarray: Ridge weights, shape (n_features, n_voxels).
    """
    return _refit_resample(X, y, indices, alpha)


def _bootstrap_ridge_weights_cpu_parallel(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_samples: int = 5000,
    save_boots: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    percentiles: tuple[float, float] = (2.5, 97.5),
    tail: int | str = 2,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap ridge weights across CPU workers.

    Each resample calls the shared fixed-hyperparameter refit directly on numpy
    arrays (no `BrainData` serialization), which is 10-100x faster than a naive
    implementation.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features).
        y (np.ndarray): Target matrix, shape (n_samples, n_voxels) or (n_samples,).
        alpha (float): Ridge regularization parameter.
        n_samples (int): Number of bootstrap iterations. Defaults to 5000.
        save_boots (bool): If True, store all bootstrap samples (memory
            intensive). Defaults to False.
        n_jobs (int): Number of CPU workers (-1 = all cores). Defaults to -1.
        random_state (int | None): Random seed for reproducibility.
        percentiles (tuple[float, float]): Percentiles for confidence intervals.
            Defaults to (2.5, 97.5).
        tail (int | str): `2` or `'two'` for two-tailed (default); `1` or `'one'`
            for one-tailed (statistic > 0).
        progress_bar (bool): Show a progress bar over iterations. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Results keyed by `'mean'` (bootstrap mean weights), `'std'`
            (bootstrap standard deviation), `'Z'` (z-scores, mean/std), `'p'`
            (p-values, per `tail`), `'ci_lower'` and `'ci_upper'` (lower and upper
            confidence bounds), `'samples'` (all samples; only if `save_boots=True`),
            and `'backend'` (backend used).

    Examples:
        ```python
        X = np.random.randn(100, 10)  # 100 samples, 10 features
        y = np.random.randn(100, 50)  # 100 samples, 50 voxels
        result = _bootstrap_ridge_weights_cpu_parallel(X, y, alpha=1.0)
        result["mean"].shape  # → (10, 50)
        ```
    """
    from joblib import Parallel, delayed
    from .validation import validate_array_shape, validate_array_shape_range

    # Input validation
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    validate_array_shape(X, 2, name="X")
    validate_array_shape_range(y, 1, 2, name="y")
    validate_shape_compatibility(X, y, X_name="X", y_name="y")
    validate_tail_parameter(tail)

    # Handle 1D y
    single_voxel = y.ndim == 1
    if single_voxel:
        y = y[:, np.newaxis]

    n_obs, n_features = X.shape
    n_voxels = y.shape[1]
    output_shape = (n_features, n_voxels)

    # Pre-generate bootstrap indices (deterministic)
    all_indices = generate_bootstrap_indices(
        n_obs, n_samples, random_state=random_state
    )

    # Initialize online statistics aggregator
    stats = OnlineBootstrapStats(
        shape=output_shape,
        save_samples=save_boots,
        percentiles=percentiles,
    )

    # Define worker function
    def _compute_one_bootstrap(idx):
        return _bootstrap_ridge_weights_worker(X, y, all_indices[idx], alpha)

    # Execute in parallel with progress bar
    bootstrap_samples = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_bootstrap)(i)
        for i in maybe_tqdm(
            range(n_samples),
            progress_bar=progress_bar,
            desc="Bootstrap Ridge weights",
            unit="iter",
        )
    )

    # Aggregate results
    for sample in bootstrap_samples:
        stats.update(sample)

    # Get final results
    result = stats.get_results(tail)

    # Add backend info
    import multiprocessing

    actual_n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    result["backend"] = f"cpu-parallel-{actual_n_jobs}"

    # Remove samples if not requested
    if not save_boots:
        result.pop("samples", None)

    return result


def _bootstrap_ridge_predict_worker(
    X: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    indices: np.ndarray,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Worker function for bootstrapping ridge predictions.

    Resamples the training data, fits a ridge model, and predicts the test data.

    Args:
        X (np.ndarray): Training feature matrix, shape (n_samples, n_features).
        y (np.ndarray): Training target matrix, shape (n_samples, n_voxels).
        X_pred (np.ndarray): Test feature matrix, shape (n_test_samples, n_features).
        indices (np.ndarray): Bootstrap indices into the training data, shape
            (n_samples,).
        alpha (float | np.ndarray): Scalar or per-target regularization.

    Returns:
        np.ndarray: Predictions, shape (n_test_samples, n_voxels).
    """
    weights = _refit_resample(X, y, indices, alpha)
    return X_pred @ weights


def _bootstrap_ridge_predict_cpu_parallel(
    X: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
    n_samples: int = 5000,
    save_boots: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    percentiles: tuple[float, float] = (2.5, 97.5),
    tail: int | str = 2,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap ridge predictions across CPU workers.

    Each resample refits the ridge model on the resampled training data and
    predicts `X_pred`; the predictions are aggregated with `OnlineBootstrapStats`.

    Args:
        X (np.ndarray): Training feature matrix, shape (n_samples, n_features).
        y (np.ndarray): Training target matrix, shape (n_samples, n_voxels) or
            (n_samples,).
        X_pred (np.ndarray): Test feature matrix, shape (n_test_samples, n_features).
        alpha (float): Ridge regularization parameter.
        n_samples (int): Number of bootstrap iterations. Defaults to 5000.
        save_boots (bool): If True, store all bootstrap predictions (memory
            intensive). Defaults to False.
        n_jobs (int): Number of CPU workers (-1 = all cores). Defaults to -1.
        random_state (int | None): Random seed for reproducibility.
        percentiles (tuple[float, float]): Percentiles for confidence intervals.
            Defaults to (2.5, 97.5).
        tail (int | str): `2` or `'two'` for two-tailed (default); `1` or `'one'`
            for one-tailed (statistic > 0).
        progress_bar (bool): Show a progress bar over iterations. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Results keyed by `'mean'` (bootstrap mean predictions), `'std'`
            (bootstrap standard deviation), `'Z'` (z-scores, mean/std), `'p'`
            (p-values, per `tail`), `'ci_lower'` and `'ci_upper'` (lower and upper
            confidence bounds), `'samples'` (all samples; only if `save_boots=True`),
            and `'backend'` (backend used).

    Examples:
        ```python
        X = np.random.randn(100, 10)  # training features
        y = np.random.randn(100, 50)  # training targets (50 voxels)
        X_test = np.random.randn(20, 10)  # test features
        result = _bootstrap_ridge_predict_cpu_parallel(X, y, X_test, alpha=1.0)
        result["mean"].shape  # → (20, 50): 20 test samples × 50 voxels
        ```
    """
    from joblib import Parallel, delayed
    from .validation import validate_shape_compatibility, validate_array_shape

    # Input validation
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X_pred = np.asarray(X_pred, dtype=np.float64)

    validate_array_shape(X, 2, name="X")
    validate_array_shape_range(y, 1, 2, name="y")
    validate_array_shape(X_pred, 2, name="X_pred")
    validate_shape_compatibility(X, y, X_name="X", y_name="y")
    if X.shape[1] != X_pred.shape[1]:
        raise ValueError(
            f"X and X_pred must have same n_features: {X.shape[1]} != {X_pred.shape[1]}"
        )
    validate_tail_parameter(tail)

    # Handle 1D y
    single_voxel = y.ndim == 1
    if single_voxel:
        y = y[:, np.newaxis]

    n_obs = X.shape[0]
    n_test_samples = X_pred.shape[0]
    n_voxels = y.shape[1]
    output_shape = (n_test_samples, n_voxels)

    # Pre-generate bootstrap indices (deterministic)
    all_indices = generate_bootstrap_indices(
        n_obs, n_samples, random_state=random_state
    )

    # Initialize online statistics aggregator
    stats = OnlineBootstrapStats(
        shape=output_shape,
        save_samples=save_boots,
        percentiles=percentiles,
    )

    # Define worker function
    def _compute_one_bootstrap(idx):
        return _bootstrap_ridge_predict_worker(X, y, X_pred, all_indices[idx], alpha)

    # Execute in parallel with progress bar
    bootstrap_samples = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_bootstrap)(i)
        for i in maybe_tqdm(
            range(n_samples),
            progress_bar=progress_bar,
            desc="Bootstrap Ridge predictions",
            unit="iter",
        )
    )

    # Aggregate results
    for sample in bootstrap_samples:
        stats.update(sample)

    # Get final results
    result = stats.get_results(tail)

    # Add backend info
    import multiprocessing

    actual_n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    result["backend"] = f"cpu-parallel-{actual_n_jobs}"

    # Remove samples if not requested
    if not save_boots:
        result.pop("samples", None)

    return result


def _auto_batch_size_ridge(
    n_bootstrap: int,
    n_samples: int,
    n_features: int,
    n_voxels: int,
    max_memory_gb: float | None = None,
    backend=None,
) -> tuple[int, int]:
    """Determine the Ridge-bootstrap GPU batch size for a memory budget.

    Thin adapter over the core layer in `nltools.algorithms.backends`: supplies
    the bootstrap working-set estimate — `X_boot` `(batch, n_samples, n_features)`
    plus `y_boot` `(batch, n_samples, n_voxels)` in float32, with a conservative
    3× overhead for SVD buffers.

    Args:
        n_bootstrap (int): Total number of bootstrap iterations.
        n_samples (int): Number of observations in the dataset.
        n_features (int): Number of features.
        n_voxels (int): Number of voxels/targets.
        max_memory_gb (float | None): Explicit memory budget in GB. None
            (default) measures the device via `device_memory_budget`.
        backend (Backend | None): Resolved backend the work runs on (used only
            to measure the budget when `max_memory_gb` is None).

    Returns:
        tuple[int, int]: `(batch_size, n_batches)`.
    """
    from nltools.algorithms.backends import auto_batch_size, device_memory_budget

    budget_gb = device_memory_budget(
        backend, max_gpu_memory_gb=max_memory_gb, cap_for_batching=True
    )
    bytes_per_boot = (n_samples * n_features + n_samples * n_voxels) * 4  # float32
    return auto_batch_size(
        n_bootstrap, bytes_per_boot, budget_gb=budget_gb, overhead=3.0
    )


def _validate_gpu_backend(backend) -> None:
    """Raise unless `backend` is a GPU device backend (torch-cuda or torch-mps).

    Guard for the GPU bootstrap engine: CPU backends ('numpy', 'torch-cpu')
    must be rejected — this engine assumes device compute.

    Args:
        backend (Backend): Resolved backend instance (only `.name` is inspected).

    Raises:
        ValueError: If `backend.name` is not 'torch-cuda' or 'torch-mps'.
    """
    if backend.name not in ("torch-cuda", "torch-mps"):
        raise ValueError(
            f"GPU backend requires 'torch-cuda' or 'torch-mps', got '{backend.name}'"
        )


def _bootstrap_ridge_gpu_batched(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    compute_sample,
    output_shape: tuple[int, ...],
    desc: str,
    n_samples: int = 5000,
    save_boots: bool = False,
    backend=None,
    max_gpu_memory_gb: float | None = None,
    random_state: int | None = None,
    percentiles: tuple[float, float] = (2.5, 97.5),
    tail: int | str = 2,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Shared GPU bootstrap driver for ridge statistics, with automatic batching.

    Owns everything the weights and predict bootstraps have in common — pre-drawn
    resample indices, batch sizing via `_auto_batch_size_ridge`, a one-time device
    transfer of X and y, the per-sample ridge-SVD solve inside an OOM-safe batch
    loop, `OnlineBootstrapStats` aggregation, the progress bar, and result
    formatting. The per-sample statistic is injected via `compute_sample`.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features), float32, 2-D.
        y (np.ndarray): Target matrix, shape (n_samples, n_voxels), float32, 2-D.
        alpha (float): Ridge regularization parameter.
        compute_sample (Callable): `(backend, coef_device) -> np.ndarray`, mapping
            one bootstrap sample's on-device ridge coefficients, shape
            (n_features, n_voxels), to the statistic aggregated on the CPU (the
            weights themselves, or predictions from them).
        output_shape (tuple[int, ...]): Shape of each `compute_sample` result.
        desc (str): Progress-bar description.
        n_samples (int): Number of bootstrap iterations. Defaults to 5000.
        save_boots (bool): If True, store all bootstrap samples (memory
            intensive). Defaults to False.
        backend (Backend | None): Backend instance (must be a GPU torch backend).
            None auto-selects.
        max_gpu_memory_gb (float | None): Explicit GPU memory budget in GB. None
            (default) measures the device's available memory.
        random_state (int | None): Random seed for reproducibility.
        percentiles (tuple[float, float]): Percentiles for confidence intervals.
            Defaults to (2.5, 97.5).
        tail (int | str): `2` or `'two'` for two-tailed (default); `1` or `'one'`
            for one-tailed (statistic > 0).
        progress_bar (bool): Show a progress bar over iterations. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Bootstrap statistics in the same format as the CPU
            engines, with `'backend'` set to `'gpu-<device>'`.
    """
    from nltools.algorithms.backends import auto_select_backend, compute_oom_safe

    # Handle backend
    if backend is None:
        backend = auto_select_backend(X.shape[0], X.shape[1])
    _validate_gpu_backend(backend)

    n_obs, n_features = X.shape
    n_voxels = y.shape[1]

    # Validate inputs
    _validate_n_samples(n_samples)
    _validate_percentiles(percentiles)
    validate_tail_parameter(tail)

    # Pre-generate bootstrap indices (deterministic)
    all_indices = generate_bootstrap_indices(
        n_obs, n_samples, random_state=random_state
    )

    # Determine batch size based on memory budget
    batch_size, n_batches = _auto_batch_size_ridge(
        n_samples,
        n_obs,
        n_features,
        n_voxels,
        max_memory_gb=max_gpu_memory_gb,
        backend=backend,
    )

    # Transfer X, y to GPU once (reused across batches)
    X_device = backend.to_device(X)
    y_device = backend.to_device(y)

    def _compute_batch(batch_indices: np.ndarray) -> np.ndarray:
        """GPU ridge statistics for one (sub-)batch of pre-drawn resample indices."""
        # Process each bootstrap sample in batch sequentially, with the ridge
        # computation inlined on the GPU to avoid CPU round-trips.
        batch_results = []
        for i in range(len(batch_indices)):
            # Resample data using advanced indexing
            indices_np = batch_indices[i].astype(np.int64)
            indices_device = backend.to_device(indices_np)
            # Ensure indices are int64 on GPU (MPS requires this)
            if hasattr(indices_device, "long"):
                indices_device = indices_device.long()
            elif hasattr(indices_device, "to"):
                import torch

                indices_device = indices_device.to(torch.int64)
            X_boot_device = X_device[indices_device]
            y_boot_device = y_device[indices_device]

            # Ridge solution: beta = V @ diag(s / (s² + alpha)) @ U.T @ y
            U, s, Vt = backend.svd(X_boot_device, full_matrices=False)
            shrinkage = s / (s**2 + alpha)
            Uty = backend.matmul(U.T, y_boot_device)
            coef_device = backend.matmul(Vt.T, shrinkage[:, None] * Uty)

            # Per-sample statistic, back on CPU for aggregation.
            batch_results.append(compute_sample(backend, coef_device))

        # Shape: (current_batch_size, *output_shape)
        return np.array(batch_results)

    # Initialize online statistics aggregator (on CPU)
    stats = OnlineBootstrapStats(
        shape=output_shape,
        save_samples=save_boots,
        percentiles=percentiles,
    )

    # Process bootstrap samples in batches with progress bar
    pbar = make_progress_bar(
        progress_bar=progress_bar,
        total=n_samples,
        desc=desc,
        unit="iter",
        disable=n_batches == 1,
    )

    for batch_idx in range(n_batches):
        # Determine current batch size
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        current_batch_size = end_idx - start_idx

        # Bootstrap indices for this batch were all pre-drawn (all_indices),
        # so OOM recovery reuses them exactly.
        batch_indices = all_indices[start_idx:end_idx]

        batch_results = compute_oom_safe(_compute_batch, batch_indices)
        for sample in batch_results:
            stats.update(sample)

        # Update progress bar
        pbar.update(current_batch_size)

    pbar.close()

    # Get final results
    result = stats.get_results(tail)

    # Add backend info
    result["backend"] = f"gpu-{backend.device}"

    # Remove samples if not requested
    if not save_boots:
        result.pop("samples", None)

    return result


def _bootstrap_ridge_weights_gpu_batched(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_samples: int = 5000,
    save_boots: bool = False,
    backend=None,
    max_gpu_memory_gb: float | None = None,
    random_state: int | None = None,
    percentiles: tuple[float, float] = (2.5, 97.5),
    tail: int | str = 2,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap ridge weights on the GPU with automatic batching.

    Thin wrapper over `_bootstrap_ridge_gpu_batched` whose per-sample statistic
    is the ridge coefficients themselves.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features).
        y (np.ndarray): Target matrix, shape (n_samples, n_voxels) or (n_samples,).
        alpha (float): Ridge regularization parameter.
        n_samples (int): Number of bootstrap iterations. Defaults to 5000.
        save_boots (bool): If True, store all bootstrap samples (memory
            intensive). Defaults to False.
        backend (Backend | None): Backend instance (must be a GPU torch backend).
            None auto-selects.
        max_gpu_memory_gb (float | None): Explicit GPU memory budget in GB. None
            (default) measures the device's available memory.
        random_state (int | None): Random seed for reproducibility.
        percentiles (tuple[float, float]): Percentiles for confidence intervals.
            Defaults to (2.5, 97.5).
        tail (int | str): `2` or `'two'` for two-tailed (default); `1` or `'one'`
            for one-tailed (statistic > 0).
        progress_bar (bool): Show a progress bar over iterations. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Bootstrap statistics in the same format as the CPU
            engine.
    """
    # Input validation
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    validate_array_shape(X, 2, name="X")
    validate_array_shape_range(y, 1, 2, name="y")
    validate_shape_compatibility(X, y, X_name="X", y_name="y")

    # Handle 1D y
    if y.ndim == 1:
        y = y[:, np.newaxis]

    def _compute_sample(backend, coef_device):
        return backend.to_numpy(coef_device)

    return _bootstrap_ridge_gpu_batched(
        X,
        y,
        alpha,
        compute_sample=_compute_sample,
        output_shape=(X.shape[1], y.shape[1]),
        desc="GPU bootstrap Ridge weights",
        n_samples=n_samples,
        save_boots=save_boots,
        backend=backend,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=random_state,
        percentiles=percentiles,
        tail=tail,
        progress_bar=progress_bar,
    )


def _bootstrap_ridge_predict_gpu_batched(
    X: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
    n_samples: int = 5000,
    save_boots: bool = False,
    backend=None,
    max_gpu_memory_gb: float | None = None,
    random_state: int | None = None,
    percentiles: tuple[float, float] = (2.5, 97.5),
    tail: int | str = 2,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap ridge predictions on the GPU with automatic batching.

    Thin wrapper over `_bootstrap_ridge_gpu_batched` whose per-sample statistic
    is `X_pred @ coef`, computed on the GPU (`X_pred` is transferred to the
    device once and reused across samples).

    Args:
        X (np.ndarray): Training feature matrix, shape (n_samples, n_features).
        y (np.ndarray): Training target matrix, shape (n_samples, n_voxels) or
            (n_samples,).
        X_pred (np.ndarray): Test feature matrix, shape (n_test_samples, n_features).
        alpha (float): Ridge regularization parameter.
        n_samples (int): Number of bootstrap iterations. Defaults to 5000.
        save_boots (bool): If True, store all bootstrap predictions (memory
            intensive). Defaults to False.
        backend (Backend | None): Backend instance (must be a GPU torch backend).
            None auto-selects.
        max_gpu_memory_gb (float | None): Explicit GPU memory budget in GB. None
            (default) measures the device's available memory.
        random_state (int | None): Random seed for reproducibility.
        percentiles (tuple[float, float]): Percentiles for confidence intervals.
            Defaults to (2.5, 97.5).
        tail (int | str): `2` or `'two'` for two-tailed (default); `1` or `'one'`
            for one-tailed (statistic > 0).
        progress_bar (bool): Show a progress bar over iterations. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Bootstrap statistics in the same format as the CPU
            engine.
    """
    # Input validation
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)

    validate_array_shape(X, 2, name="X")
    validate_array_shape_range(y, 1, 2, name="y")
    validate_array_shape(X_pred, 2, name="X_pred")
    validate_shape_compatibility(X, y, X_name="X", y_name="y")
    if X.shape[1] != X_pred.shape[1]:
        raise ValueError(
            f"X and X_pred must have same n_features: {X.shape[1]} != {X_pred.shape[1]}"
        )

    # Handle 1D y
    if y.ndim == 1:
        y = y[:, np.newaxis]

    # X_pred moves to the device once, lazily (the driver resolves the backend).
    device_cache: dict[str, object] = {}

    def _compute_sample(backend, coef_device):
        if "X_pred" not in device_cache:
            device_cache["X_pred"] = backend.to_device(X_pred)
        predictions_device = backend.matmul(device_cache["X_pred"], coef_device)
        return backend.to_numpy(predictions_device)

    return _bootstrap_ridge_gpu_batched(
        X,
        y,
        alpha,
        compute_sample=_compute_sample,
        output_shape=(X_pred.shape[0], y.shape[1]),
        desc="GPU bootstrap Ridge predictions",
        n_samples=n_samples,
        save_boots=save_boots,
        backend=backend,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=random_state,
        percentiles=percentiles,
        tail=tail,
        progress_bar=progress_bar,
    )
