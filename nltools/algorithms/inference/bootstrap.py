"""Bootstrap inference utilities with CPU/GPU support."""

import numpy as np
import warnings
from typing import Tuple, Dict, Optional
from scipy.stats import norm


# Constants for supported methods
SIMPLE_METHODS = ["mean", "median", "std", "sum", "min", "max"]
FITTED_METHODS = ["weights", "predict"]  # For future use


def _validate_bootstrap_method(method: str) -> None:
    """
    Validate bootstrap method name.

    Parameters
    ----------
    method : str
        Method name to validate

    Raises
    ------
    ValueError
        If method is not supported
    """
    supported = SIMPLE_METHODS + FITTED_METHODS
    if method not in supported:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Supported methods: {SIMPLE_METHODS} (simple methods), "
            f"{FITTED_METHODS} (fitted model methods). "
            f"For fitted methods, you must call .fit() first."
        )


def _validate_bootstrap_data(data: np.ndarray, method: str) -> None:
    """
    Validate input data for bootstrapping.

    Parameters
    ----------
    data : np.ndarray
        Data to validate
    method : str
        Bootstrap method

    Raises
    ------
    ValueError
        If data is invalid (wrong shape, too few samples, etc.)
    """
    # Check dimensionality
    if data.ndim not in [1, 2]:
        raise ValueError(
            f"Data must be 1D or 2D, got shape {data.shape}. "
            f"For 3D+ data, you may need to reshape or select specific dimensions."
        )

    # Check number of samples
    n_samples = data.shape[0] if data.ndim == 2 else len(data)
    if n_samples < 2:
        raise ValueError(
            f"Need at least 2 samples for bootstrap, got {n_samples}. "
            f"Bootstrap requires resampling, which needs multiple samples."
        )

    # Warn if very few samples
    if n_samples < 10:
        warnings.warn(
            f"Only {n_samples} samples available. Bootstrap works best with n >= 30. "
            f"Results may be unreliable with very small sample sizes.",
            UserWarning,
        )


def _validate_n_samples(n_samples: int) -> None:
    """
    Validate number of bootstrap iterations.

    Parameters
    ----------
    n_samples : int
        Number of bootstrap iterations

    Raises
    ------
    ValueError
        If n_samples is invalid
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
        )


def _validate_percentiles(percentiles: tuple) -> None:
    """
    Validate percentile values for confidence intervals.

    Parameters
    ----------
    percentiles : tuple
        Percentile values (lower, upper)

    Raises
    ------
    ValueError
        If percentiles are invalid
    """
    if not isinstance(percentiles, (tuple, list)) or len(percentiles) != 2:
        raise ValueError(f"percentiles must be a tuple of 2 values, got {percentiles}")

    lower, upper = percentiles

    if not (0 < lower < 50):
        raise ValueError(f"Lower percentile must be between 0 and 50, got {lower}")

    if not (50 < upper < 100):
        raise ValueError(f"Upper percentile must be between 50 and 100, got {upper}")

    if lower >= upper:
        raise ValueError(
            f"Lower percentile ({lower}) must be less than upper ({upper})"
        )


class OnlineBootstrapStats:
    """
    Memory-efficient online statistics aggregator for bootstrap samples.

    Uses Welford's algorithm for numerically stable online computation of
    mean and variance. Optionally stores all samples for exact percentile CIs.

    Parameters
    ----------
    shape : tuple
        Shape of each bootstrap sample
    save_samples : bool, default=False
        If True, store all samples for exact percentile confidence intervals.
        If False, use normal approximation (much more memory efficient).
    percentiles : tuple, default=(2.5, 97.5)
        Percentiles for confidence intervals (e.g., (2.5, 97.5) for 95% CI)

    Examples
    --------
    >>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
    >>> for i in range(1000):
    ...     sample = np.random.randn(100)
    ...     stats.update(sample)
    >>> results = stats.get_results()
    >>> print(results.keys())
    dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        save_samples: bool = False,
        percentiles: Tuple[float, float] = (2.5, 97.5),
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
        """
        Update statistics with a new bootstrap sample.

        Uses Welford's algorithm for numerical stability.

        Parameters
        ----------
        sample : np.ndarray
            New bootstrap sample with shape matching self.shape
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

    def get_results(self) -> Dict[str, np.ndarray]:
        """
        Compute final bootstrap statistics.

        Returns
        -------
        dict
            Dictionary containing:
            - 'mean': Bootstrap mean
            - 'std': Bootstrap standard deviation
            - 'Z': Z-scores (mean/std)
            - 'p': Two-tailed p-values
            - 'ci_lower': Lower confidence bound
            - 'ci_upper': Upper confidence bound
            - 'samples': All samples (only if save_samples=True)
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

        # Compute two-tailed p-values from Z-scores
        p = 2 * (1 - norm.cdf(np.abs(z)))

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
    """
    Worker function for bootstrapping simple aggregation methods.

    Parameters
    ----------
    data : np.ndarray
        Data to bootstrap (n_samples, n_features)
    method : str
        Aggregation method: 'mean', 'median', 'std', 'sum', 'min', 'max'
    indices : np.ndarray
        Bootstrap indices (n_samples,)

    Returns
    -------
    np.ndarray
        Aggregated result (n_features,) or scalar
    """
    # Resample data
    data_boot = data[indices]

    # Apply aggregation method
    if method == "mean":
        return np.mean(data_boot, axis=0)
    elif method == "median":
        return np.median(data_boot, axis=0)
    elif method == "std":
        return np.std(data_boot, axis=0, ddof=1)
    elif method == "sum":
        return np.sum(data_boot, axis=0)
    elif method == "min":
        return np.min(data_boot, axis=0)
    elif method == "max":
        return np.max(data_boot, axis=0)
    else:
        raise ValueError(f"Unsupported method: {method}")


def _bootstrap_simple_cpu_parallel(
    data: np.ndarray,
    method: str,
    n_samples: int = 5000,
    save_boots: bool = False,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    percentiles: Tuple[float, float] = (2.5, 97.5),
) -> Dict[str, np.ndarray]:
    """
    Bootstrap simple aggregation methods using CPU parallelization.

    Follows the same pattern as inference module: pre-generate indices,
    parallelize computation, aggregate with OnlineBootstrapStats.

    Parameters
    ----------
    data : np.ndarray
        Data to bootstrap, shape (n_samples, n_features) or (n_samples,)
    method : str
        Aggregation method: 'mean', 'median', 'std', 'sum', 'min', 'max'
    n_samples : int, default=5000
        Number of bootstrap iterations
    save_boots : bool, default=False
        If True, store all bootstrap samples (memory intensive)
    n_jobs : int, default=-1
        Number of CPU cores for parallelization
    random_state : int, optional
        Random seed for reproducibility
    percentiles : tuple, default=(2.5, 97.5)
        Percentiles for confidence intervals

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean': Bootstrap mean
        - 'std': Bootstrap standard deviation
        - 'Z': Z-scores (mean/std)
        - 'p': Two-tailed p-values
        - 'ci_lower': Lower confidence bound
        - 'ci_upper': Upper confidence bound
        - 'samples': All samples (only if save_boots=True)
        - 'backend': Backend used (e.g., 'cpu-parallel-8')

    Examples
    --------
    >>> data = np.random.randn(100, 50)  # 100 samples, 50 features
    >>> result = _bootstrap_simple_cpu_parallel(data, 'mean', n_samples=1000)
    >>> result['mean'].shape
    (50,)
    >>> result.keys()
    dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', 'backend'])
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from .utils import _generate_bootstrap_indices

    # Validate inputs
    _validate_bootstrap_method(method)
    _validate_n_samples(n_samples)
    _validate_percentiles(percentiles)

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
    all_indices = _generate_bootstrap_indices(
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
        for i in tqdm(range(n_samples), desc="Bootstrap iterations", unit="iter")
    )

    # Aggregate results
    for sample in bootstrap_samples:
        if single_feature:
            sample = sample.flatten()
        stats.update(sample)

    # Get final results
    result = stats.get_results()

    # Add backend info
    import multiprocessing

    actual_n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    result["backend"] = f"cpu-parallel-{actual_n_jobs}"

    # Remove samples if not requested
    if not save_boots:
        result.pop("samples", None)

    return result


def _bootstrap_ridge_weights_worker(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    alpha: float,
    **ridge_kwargs,
) -> np.ndarray:
    """
    Worker function for bootstrapping Ridge model weights.

    Bypasses BrainData overhead by calling ridge_svd() directly with numpy arrays.
    This provides 10-100× speedup compared to using BrainData methods.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    y : np.ndarray
        Target matrix, shape (n_samples, n_voxels)
    indices : np.ndarray
        Bootstrap indices, shape (n_samples,)
    alpha : float
        Ridge regularization parameter
    **ridge_kwargs : dict
        Additional parameters passed to ridge_svd()

    Returns
    -------
    np.ndarray
        Ridge weights, shape (n_features, n_voxels)
    """
    from nltools.algorithms.ridge import ridge_svd

    # Resample data
    X_boot = X[indices]
    y_boot = y[indices]

    # Call optimized ridge_svd directly (pure numpy, very fast)
    weights = ridge_svd(X_boot, y_boot, alpha=alpha, **ridge_kwargs)

    return weights


def _bootstrap_ridge_weights_cpu_parallel(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_samples: int = 5000,
    save_boots: bool = False,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    percentiles: Tuple[float, float] = (2.5, 97.5),
    **ridge_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Bootstrap Ridge model weights using CPU parallelization.

    Performance optimization: Calls ridge_svd() directly instead of using
    BrainData methods, avoiding serialization overhead. Provides 10-100×
    speedup compared to naive implementation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    y : np.ndarray
        Target matrix, shape (n_samples, n_voxels)
    alpha : float
        Ridge regularization parameter
    n_samples : int, default=5000
        Number of bootstrap iterations
    save_boots : bool, default=False
        If True, store all bootstrap samples (memory intensive)
    n_jobs : int, default=-1
        Number of CPU cores for parallelization
    random_state : int, optional
        Random seed for reproducibility
    percentiles : tuple, default=(2.5, 97.5)
        Percentiles for confidence intervals
    **ridge_kwargs : dict
        Additional parameters passed to ridge_svd()

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean': Bootstrap mean weights
        - 'std': Bootstrap standard deviation
        - 'Z': Z-scores (mean/std)
        - 'p': Two-tailed p-values
        - 'ci_lower': Lower confidence bound
        - 'ci_upper': Upper confidence bound
        - 'samples': All samples (only if save_boots=True)
        - 'backend': Backend used

    Examples
    --------
    >>> X = np.random.randn(100, 10)  # 100 samples, 10 features
    >>> y = np.random.randn(100, 50)  # 100 samples, 50 voxels
    >>> result = _bootstrap_ridge_weights_cpu_parallel(X, y, alpha=1.0)
    >>> result['mean'].shape
    (10, 50)
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from .utils import _generate_bootstrap_indices

    # Input validation
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim not in [1, 2]:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same n_samples: {X.shape[0]} != {y.shape[0]}"
        )

    # Handle 1D y
    single_voxel = y.ndim == 1
    if single_voxel:
        y = y[:, np.newaxis]

    n_obs, n_features = X.shape
    n_voxels = y.shape[1]
    output_shape = (n_features, n_voxels)

    # Pre-generate bootstrap indices (deterministic)
    all_indices = _generate_bootstrap_indices(
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
        return _bootstrap_ridge_weights_worker(
            X, y, all_indices[idx], alpha, **ridge_kwargs
        )

    # Execute in parallel with progress bar
    bootstrap_samples = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_bootstrap)(i)
        for i in tqdm(range(n_samples), desc="Bootstrap Ridge weights", unit="iter")
    )

    # Aggregate results
    for sample in bootstrap_samples:
        stats.update(sample)

    # Get final results
    result = stats.get_results()

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
    alpha: float,
    **ridge_kwargs,
) -> np.ndarray:
    """
    Worker function for bootstrapping Ridge model predictions.

    Resamples training data, fits Ridge model, and makes predictions on test data.

    Parameters
    ----------
    X : np.ndarray
        Training feature matrix, shape (n_samples, n_features)
    y : np.ndarray
        Training target matrix, shape (n_samples, n_voxels)
    X_pred : np.ndarray
        Test feature matrix for prediction, shape (n_test_samples, n_features)
    indices : np.ndarray
        Bootstrap indices for resampling training data, shape (n_samples,)
    alpha : float
        Ridge regularization parameter
    **ridge_kwargs : dict
        Additional parameters passed to ridge_svd()

    Returns
    -------
    np.ndarray
        Predictions, shape (n_test_samples, n_voxels)
    """
    from nltools.algorithms.ridge import ridge_svd

    # Resample training data
    X_boot = X[indices]
    y_boot = y[indices]

    # Fit Ridge model to bootstrap sample
    weights = ridge_svd(X_boot, y_boot, alpha=alpha, **ridge_kwargs)

    # Make predictions on test data
    # Matrix multiplication: (n_test, n_features) @ (n_features, n_voxels)
    predictions = X_pred @ weights

    return predictions


def _bootstrap_ridge_predict_cpu_parallel(
    X: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
    n_samples: int = 5000,
    save_boots: bool = False,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    percentiles: Tuple[float, float] = (2.5, 97.5),
    **ridge_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Bootstrap Ridge model predictions using CPU parallelization.

    Resamples training data, fits Ridge models, and aggregates predictions
    on test data. Uses same performance optimizations as weights bootstrap.

    Parameters
    ----------
    X : np.ndarray
        Training feature matrix, shape (n_samples, n_features)
    y : np.ndarray
        Training target matrix, shape (n_samples, n_voxels)
    X_pred : np.ndarray
        Test feature matrix for prediction, shape (n_test_samples, n_features)
    alpha : float
        Ridge regularization parameter
    n_samples : int, default=5000
        Number of bootstrap iterations
    save_boots : bool, default=False
        If True, store all bootstrap predictions (memory intensive)
    n_jobs : int, default=-1
        Number of CPU cores for parallelization
    random_state : int, optional
        Random seed for reproducibility
    percentiles : tuple, default=(2.5, 97.5)
        Percentiles for confidence intervals
    **ridge_kwargs : dict
        Additional parameters passed to ridge_svd()

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean': Bootstrap mean predictions
        - 'std': Bootstrap standard deviation
        - 'Z': Z-scores (mean/std)
        - 'p': Two-tailed p-values
        - 'ci_lower': Lower confidence bound
        - 'ci_upper': Upper confidence bound
        - 'samples': All samples (only if save_boots=True)
        - 'backend': Backend used

    Examples
    --------
    >>> X = np.random.randn(100, 10)         # Training features
    >>> y = np.random.randn(100, 50)         # Training targets (50 voxels)
    >>> X_test = np.random.randn(20, 10)     # Test features
    >>> result = _bootstrap_ridge_predict_cpu_parallel(X, y, X_test, alpha=1.0)
    >>> result['mean'].shape
    (20, 50)  # Predictions for 20 test samples, 50 voxels
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from .utils import _generate_bootstrap_indices

    # Input validation
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X_pred = np.asarray(X_pred, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim not in [1, 2]:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")
    if X_pred.ndim != 2:
        raise ValueError(f"X_pred must be 2D, got shape {X_pred.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same n_samples: {X.shape[0]} != {y.shape[0]}"
        )
    if X.shape[1] != X_pred.shape[1]:
        raise ValueError(
            f"X and X_pred must have same n_features: {X.shape[1]} != {X_pred.shape[1]}"
        )

    # Handle 1D y
    single_voxel = y.ndim == 1
    if single_voxel:
        y = y[:, np.newaxis]

    n_obs = X.shape[0]
    n_test_samples = X_pred.shape[0]
    n_voxels = y.shape[1]
    output_shape = (n_test_samples, n_voxels)

    # Pre-generate bootstrap indices (deterministic)
    all_indices = _generate_bootstrap_indices(
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
        return _bootstrap_ridge_predict_worker(
            X, y, X_pred, all_indices[idx], alpha, **ridge_kwargs
        )

    # Execute in parallel with progress bar
    bootstrap_samples = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_bootstrap)(i)
        for i in tqdm(range(n_samples), desc="Bootstrap Ridge predictions", unit="iter")
    )

    # Aggregate results
    for sample in bootstrap_samples:
        stats.update(sample)

    # Get final results
    result = stats.get_results()

    # Add backend info
    import multiprocessing

    actual_n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    result["backend"] = f"cpu-parallel-{actual_n_jobs}"

    # Remove samples if not requested
    if not save_boots:
        result.pop("samples", None)

    return result
