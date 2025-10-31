"""Bootstrap inference utilities with CPU/GPU support."""

import numpy as np
from typing import Tuple, Dict
from scipy.stats import norm


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
