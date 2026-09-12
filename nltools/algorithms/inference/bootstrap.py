"""Bootstrap resampling for simple statistics and fitted-model outputs.

Resamples observations with replacement `n_samples` times and summarizes the
resulting distribution with `BootstrapAccumulator`, a streaming aggregator that
keeps a running Welford variance plus a bounded per-element tail — enough order
statistics to reproduce the exact percentile interval without retaining every
replicate. The CPU engines dispatch in bounded windows and fold each one in
before opening the next, which is what makes that bound real: `joblib.Parallel`
queues finished results, so collecting the run would hold every replicate at
once whatever the accumulator does. The engines cover simple aggregations ('mean', 'median', 'std', ...),
ridge weights, and ridge predictions, each with a CPU-parallel path (`n_jobs`
workers) and, for ridge, a batched GPU path. `BrainData.bootstrap` and
`Adjacency.bootstrap` are the user-facing entry points that pick an engine and
wrap the arrays as a `BootstrapResult`.

Every engine returns the same four arrays — `estimate` (the statistic on the
unresampled full sample), `standard_error`, `ci_lower`, `ci_upper` — plus
`samples` when the caller asked to retain the complete distribution. Memory
budgeting, the output preflight, GPU batch sizing, and CPU worker sizing all
live in `nltools/algorithms/backends.py`; nothing here does budget arithmetic.
"""

import numpy as np
import warnings

from .validation import (
    validate_bootstrap_method,
    validate_bootstrap_data,
    validate_array_shape,
    validate_array_shape_range,
    validate_confidence_level,
    validate_memory_budget,
    validate_n_samples,
    validate_shape_compatibility,
)
from .random import generate_bootstrap_indices
from .utils import make_progress_bar
from nltools.utils import find_stack_level


# Constants for supported methods
SIMPLE_METHODS = ["mean", "median", "std", "sum", "min", "max"]
FITTED_METHODS = ["weights", "predict"]


def _advise_on_n_samples(n_samples: int) -> None:
    """Warn once per run when the replicate count is too low for a stable interval.

    The engines are the single emitter: every facade reaches one of them, and a
    direct engine call still gets advised, so the user sees the sentence exactly
    once however they entered.

    Args:
        n_samples (int): Number of bootstrap replicates.
    """
    if n_samples < 1000:
        warnings.warn(
            f"n_samples={n_samples} is low. For reliable confidence intervals, "
            f"use n_samples >= 1000. For hypothesis testing, use n_samples >= 5000.",
            UserWarning,
            stacklevel=find_stack_level(),
        )


def _advise_on_sample_size(data: np.ndarray) -> None:
    """Warn when the sample the bootstrap resamples from is very small.

    Args:
        data (np.ndarray): The validated 1D or 2D data.
    """
    n_samples = data.shape[0] if data.ndim == 2 else len(data)
    if n_samples < 10:
        warnings.warn(
            f"Only {n_samples} samples available. Bootstrap works best with n >= 30. "
            f"Results may be unreliable with very small sample sizes.",
            UserWarning,
            stacklevel=find_stack_level(),
        )


class BootstrapAccumulator:
    """Streaming bootstrap aggregator with a bounded, exact percentile tail.

    Holds a running Welford mean and variance plus, per output element, the `k`
    smallest and `k` largest replicate values seen so far, where
    `k = ceil((B - 1) * (1 - c) / 2) + 1`. Those are exactly the order
    statistics NumPy's linear-interpolation percentile can reach at either end,
    so the interval matches the one the complete distribution would give while
    storage scales with `(1 - c) * B` instead of `B`.

    Storage is memory-efficient, not constant: `k` grows with `B`. Retaining the
    complete distribution (`retain_samples=True`) adds the full sample list but
    changes nothing about how the interval is computed.

    Args:
        shape (tuple[int, ...]): Shape of one replicate's output.
        n_replicates (int): Number of replicates the run will produce, `B`,
            which fixes the retained tail size.
        confidence_level (float): Interval confidence level, `c`. Defaults to
            `0.95`.
        retain_samples (bool): Keep every replicate as well. Defaults to False.

    Attributes:
        n (int): Replicates folded in so far.
        n_replicates (int): Replicates the run was sized for.
        tail_size (int): Values retained per element at each end.

    Examples:
        ```python
        accumulator = BootstrapAccumulator((100,), n_replicates=1000)
        for sample in replicates:
            accumulator.update(sample)
        summary = accumulator.results()
        sorted(summary)  # → ['ci_lower', 'ci_upper', 'samples', 'standard_error']
        ```
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        n_replicates: int,
        confidence_level: float = 0.95,
        retain_samples: bool = False,
    ):
        from nltools.algorithms.backends import (
            BOOTSTRAP_TAIL_FLUSH_BLOCK,
            bootstrap_retained_tail_size,
        )

        self.shape = tuple(shape)
        self.confidence_level = float(confidence_level)
        self.retain_samples = bool(retain_samples)
        self.n_replicates = int(n_replicates)
        self.tail_size = bootstrap_retained_tail_size(
            n_replicates, confidence_level=self.confidence_level
        )
        self._flush_block = BOOTSTRAP_TAIL_FLUSH_BLOCK

        self.n = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.M2 = np.zeros(self.shape, dtype=np.float64)

        # Bounded order statistics, unordered within each block until `results`.
        self._low = np.empty((0, *self.shape), dtype=np.float64)
        self._high = np.empty((0, *self.shape), dtype=np.float64)
        self._pending: list[np.ndarray] = []

        # NaN is dropped by `np.partition`, so a per-element flag reproduces
        # `np.percentile`'s propagation instead of quietly skipping it.
        self._has_nan = np.zeros(self.shape, dtype=bool)

        # Preallocated rather than appended-and-stacked: `np.stack` on a list of
        # `n_replicates` arrays would briefly hold the distribution twice, which
        # is precisely the peak the preflight is meant to predict.
        self._samples = (
            np.empty((self.n_replicates, *self.shape), dtype=np.float64)
            if self.retain_samples
            else None
        )

    def update(self, sample: np.ndarray) -> None:
        """Fold one replicate into the running statistics and retained tails.

        Args:
            sample (np.ndarray): One replicate's output, shaped like `shape`.

        Raises:
            ValueError: If the sample's shape does not match `shape`, or the
                accumulator has already seen `n_replicates` replicates.
        """
        sample = np.asarray(sample, dtype=np.float64)
        if sample.shape != self.shape:
            raise ValueError(
                f"Sample shape {sample.shape} does not match expected shape {self.shape}"
            )
        if self.n >= self.n_replicates:
            # The retained tail is sized for exactly `n_replicates`; a further
            # replicate would make `results()` read past what it kept.
            raise ValueError(
                f"This accumulator was sized for {self.n_replicates} replicates "
                f"and has already seen that many. Size it with the run's total "
                f"replicate count."
            )

        self.n += 1
        delta = sample - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (sample - self.mean)

        self._has_nan |= np.isnan(sample)
        self._pending.append(sample.copy())
        if self._samples is not None:
            self._samples[self.n - 1] = sample
        if len(self._pending) >= self._flush_block:
            self._flush()

    def _flush(self) -> None:
        """Fold the buffered replicates into the two bounded tails."""
        if not self._pending:
            return
        block = np.stack(self._pending)
        self._pending = []
        self._low = self._keep_smallest(np.concatenate([self._low, block]))
        self._high = self._keep_largest(np.concatenate([self._high, block]))

    def _keep_smallest(self, values: np.ndarray) -> np.ndarray:
        """Reduce `values` to the `tail_size` smallest entries per element.

        The slice is copied because a view would keep the whole partition
        output alive, quietly holding `tail_size + flush block` arrays where
        the budget charges `tail_size`.
        """
        if values.shape[0] <= self.tail_size:
            return values
        return np.partition(values, self.tail_size - 1, axis=0)[: self.tail_size].copy()

    def _keep_largest(self, values: np.ndarray) -> np.ndarray:
        """Reduce `values` to the `tail_size` largest entries per element."""
        if values.shape[0] <= self.tail_size:
            return values
        cut = values.shape[0] - self.tail_size
        return np.partition(values, cut, axis=0)[cut:].copy()

    @classmethod
    def _empty_like(cls, reference: "BootstrapAccumulator") -> "BootstrapAccumulator":
        """An empty accumulator with `reference`'s run shape and retention policy.

        `merge` needs a target sized for the *whole* run, which the public
        constructor derives from `n_replicates`. Copying the settled values
        across says that directly instead of re-deriving them.
        """
        empty = cls(
            reference.shape,
            n_replicates=reference.n_replicates,
            confidence_level=reference.confidence_level,
            retain_samples=reference.retain_samples,
        )
        empty.tail_size = reference.tail_size
        return empty

    @staticmethod
    def merge(
        first: "BootstrapAccumulator", second: "BootstrapAccumulator"
    ) -> "BootstrapAccumulator":
        """Combine two accumulators over disjoint replicate blocks.

        Uses the Chan-Golub-LeVeque parallel variance update and merges the
        bounded tails, so a run split across workers or memory-driven batches
        summarizes to the same numbers as one sequential pass. Retained
        samples are concatenated in `first`-then-`second` order.

        Args:
            first (BootstrapAccumulator): Accumulator over the earlier block.
            second (BootstrapAccumulator): Accumulator over the later block.
                Both must have been sized with the run's total replicate count.

        Returns:
            BootstrapAccumulator: A new accumulator holding both blocks.

        Raises:
            ValueError: If the two accumulators describe different runs.
        """
        if first.shape != second.shape:
            raise ValueError(
                f"Cannot merge accumulators of shape {first.shape} and {second.shape}."
            )
        if first.confidence_level != second.confidence_level:
            raise ValueError(
                "Cannot merge accumulators with different confidence levels."
            )
        if first.tail_size != second.tail_size:
            # Both blocks must retain the tail the *whole* run needs, so size
            # every accumulator with the run's total replicate count.
            raise ValueError(
                f"Cannot merge accumulators retaining {first.tail_size} and "
                f"{second.tail_size} values per tail: both must be sized for "
                f"the whole run's replicate count."
            )
        if first.retain_samples != second.retain_samples:
            # Merging them could only produce a partial distribution, which is
            # worse than refusing: `results()` would report `samples=None` for a
            # run the caller asked to retain.
            raise ValueError(
                "Cannot merge one accumulator that retains every replicate with "
                "one that does not: both blocks of a run must agree on "
                "retain_samples."
            )

        merged = BootstrapAccumulator._empty_like(first)

        first._flush()
        second._flush()

        total = first.n + second.n
        merged.n = total
        if total:
            delta = second.mean - first.mean
            merged.mean = first.mean + delta * (second.n / total)
            merged.M2 = (
                first.M2 + second.M2 + delta * delta * (first.n * second.n / total)
            )
        merged._has_nan = first._has_nan | second._has_nan
        merged._low = merged._keep_smallest(np.concatenate([first._low, second._low]))
        merged._high = merged._keep_largest(np.concatenate([first._high, second._high]))
        if merged.retain_samples:
            merged._samples = np.concatenate(
                [first._samples[: first.n], second._samples[: second.n]]
            )
        return merged

    def results(self) -> dict:
        """Summarize the replicates seen so far.

        Returns:
            dict: `'standard_error'` (elementwise `ddof=1` deviation across
                replicates), `'ci_lower'` and `'ci_upper'` (the central
                percentile interval by linear interpolation), and `'samples'`
                (every replicate, bootstrap axis first, or None).

        Raises:
            ValueError: If fewer than two replicates have been folded in.
        """
        if self.n < 2:
            raise ValueError(
                f"Need at least 2 bootstrap replicates, got {self.n}. "
                "A standard error cannot be computed from fewer."
            )
        self._flush()

        standard_error = np.sqrt(self.M2 / (self.n - 1))
        ci_lower, ci_upper = self._interval()
        samples = self._samples[: self.n] if self._samples is not None else None
        return {
            "standard_error": standard_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "samples": samples,
        }

    def _interval(self) -> tuple[np.ndarray, np.ndarray]:
        """Central percentile interval read off the two retained tails."""
        half_alpha = (1 - self.confidence_level) / 2
        position = (self.n - 1) * half_alpha

        low_sorted = np.sort(self._low, axis=0)
        high_sorted = np.sort(self._high, axis=0)
        ci_lower = _interpolate_order_statistic(low_sorted, position, offset=0)
        ci_upper = _interpolate_order_statistic(
            high_sorted,
            (self.n - 1) - position,
            offset=self.n - high_sorted.shape[0],
        )

        # `np.percentile` returns NaN for any element whose distribution holds
        # one, and the spec requires the same propagation here.
        ci_lower = np.where(self._has_nan, np.nan, ci_lower)
        ci_upper = np.where(self._has_nan, np.nan, ci_upper)
        return ci_lower, ci_upper


def _interpolate_order_statistic(
    tail_sorted: np.ndarray, position: float, *, offset: int
) -> np.ndarray:
    """Read one linearly interpolated order statistic out of a retained tail.

    Args:
        tail_sorted (np.ndarray): Retained values, sorted ascending along axis 0.
        position (float): Fractional index into the *complete* sorted
            distribution, as NumPy's linear interpolation defines it.
        offset (int): Complete-distribution index of `tail_sorted[0]`.

    Returns:
        np.ndarray: The interpolated value per output element.
    """
    lower_index = int(np.floor(position))
    fraction = position - lower_index
    local = lower_index - offset
    lower = tail_sorted[local]
    if fraction == 0:
        return lower.copy()
    return lower + fraction * (tail_sorted[local + 1] - lower)


def _run_replicates(
    compute_one,
    n_samples: int,
    accumulator: BootstrapAccumulator,
    *,
    n_jobs: int,
    window: int,
    desc: str,
    progress_bar: bool,
) -> None:
    """Evaluate every replicate on CPU workers, aggregating one window at a time.

    Bounding the pipeline is the whole point. `joblib.Parallel` dispatches
    eagerly and queues finished results, so neither `pre_dispatch` nor
    `return_as="generator"` limits how many replicate arrays are alive at once —
    collecting the whole run, the obvious spelling, makes peak memory `O(B)` and
    turns `bootstrap_memory_preflight` into a budget the run ignores. Instead
    each window of `window` replicates is folded into the accumulator and
    released before the next window is dispatched, so peak memory is the
    accumulator's retained tail plus one window.

    Windows are consecutive and folded in submission order, so replicate order,
    the reported failure index, and Welford's accumulation order are identical
    to a sequential run — worker count stays numerically invisible.

    Args:
        compute_one (Callable): `(index) -> np.ndarray` for one replicate.
        n_samples (int): Number of replicates.
        accumulator (BootstrapAccumulator): Aggregator to fold results into.
        n_jobs (int): Worker count, already planned against the budget.
        window (int): Replicates dispatched before the next aggregation, from
            `backends.bootstrap_replicate_window`.
        desc (str): Progress-bar description.
        progress_bar (bool): Show a progress bar over replicates.

    Raises:
        RuntimeError: If a replicate fails, naming its index.
    """
    from joblib import Parallel, delayed

    def _guarded(index):
        try:
            return compute_one(index)
        except Exception as error:  # noqa: BLE001 - re-raised with the index
            raise RuntimeError(
                f"bootstrap replicate {index} failed: {error}"
            ) from error

    pbar = make_progress_bar(
        progress_bar=progress_bar, total=n_samples, desc=desc, unit="iter"
    )
    # One pool for the whole run; the context manager keeps workers alive
    # across windows so bounding memory does not cost a respawn per window.
    with Parallel(n_jobs=n_jobs) as parallel:
        for start in range(0, n_samples, window):
            stop = min(start + window, n_samples)
            for sample in parallel(
                delayed(_guarded)(index) for index in range(start, stop)
            ):
                accumulator.update(sample)
            # The bar advances on completed replicates, not dispatched ones.
            pbar.update(stop - start)
    pbar.close()


def _summarize(accumulator: BootstrapAccumulator, estimate: np.ndarray) -> dict:
    """Assemble the engine result from an accumulator and the full-sample estimate.

    Args:
        accumulator (BootstrapAccumulator): Aggregator over every replicate.
        estimate (np.ndarray): The statistic on the unresampled full sample.

    Returns:
        dict: `'estimate'`, `'standard_error'`, `'ci_lower'`, `'ci_upper'`, and
            `'samples'` when the complete distribution was retained.
    """
    summary = accumulator.results()
    result = {
        "estimate": np.asarray(estimate, dtype=np.float64),
        "standard_error": summary["standard_error"],
        "ci_lower": summary["ci_lower"],
        "ci_upper": summary["ci_upper"],
    }
    if summary["samples"] is not None:
        result["samples"] = summary["samples"]
    return result


def _bootstrap_simple_method_worker(
    data: np.ndarray,
    method: str,
    indices: np.ndarray,
) -> np.ndarray:
    """Apply one simple aggregation to a resampled copy of the data.

    The single home of the six basic reductions, used both for the replicates
    and — with the identity index — for the full-sample estimate. `'std'` is
    the population deviation (`ddof=0`), matching `BrainData.std()`.

    Args:
        data (np.ndarray): Data to bootstrap, shape (n_obs, n_features).
        method (str): Aggregation method, one of 'mean', 'median', 'std', 'sum',
            'min', or 'max'.
        indices (np.ndarray): Row indices of the replicate, shape (n_obs,).

    Returns:
        np.ndarray: Aggregated result, shape (n_features,).

    Raises:
        ValueError: If `method` is not one of the supported aggregations.
    """
    data_boot = data[indices]

    if method == "mean":
        return np.mean(data_boot, axis=0)
    if method == "median":
        return np.median(data_boot, axis=0)
    if method == "std":
        return np.std(data_boot, axis=0, ddof=0)
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
    *,
    confidence_level: float = 0.95,
    memory_budget_gb: float | None = None,
    return_samples: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap a simple aggregation across CPU workers.

    Bootstrap indices are pre-generated from `random_state`, replicates run in
    parallel with joblib, and results stream into `BootstrapAccumulator`.

    Args:
        data (np.ndarray): Data to bootstrap, shape (n_obs, n_features) or
            (n_obs,).
        method (str): Aggregation method, one of 'mean', 'median', 'std', 'sum',
            'min', or 'max'.
        n_samples (int): Number of bootstrap replicates. Defaults to 5000.
        confidence_level (float): Interval confidence level. Defaults to 0.95.
        memory_budget_gb (float | None): Working-memory budget in GB governing
            the output preflight and worker planning. None measures the host.
        return_samples (bool): Retain every replicate. Defaults to False.
        n_jobs (int): CPU worker ceiling (-1 = all cores). Defaults to -1.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over replicates. Defaults to False.

    Returns:
        dict[str, np.ndarray]: `'estimate'` (the aggregation on the unresampled
            data), `'standard_error'`, `'ci_lower'`, `'ci_upper'`, `'samples'`
            (only with `return_samples=True`), and `'backend'`.

    Examples:
        ```python
        data = np.random.randn(100, 50)  # 100 observations, 50 features
        result = _bootstrap_simple_cpu_parallel(data, "mean", n_samples=1000)
        result["estimate"].shape  # → (50,)
        ```
    """
    from nltools.algorithms.backends import (
        bootstrap_memory_preflight,
        bootstrap_n_jobs_cpu,
        bootstrap_replicate_window,
        _estimate_data_size_mb,
    )

    validate_bootstrap_method(method, SIMPLE_METHODS, FITTED_METHODS)
    validate_n_samples(n_samples)
    validate_confidence_level(confidence_level)
    validate_memory_budget(memory_budget_gb)

    data = np.asarray(data, dtype=np.float64)
    validate_bootstrap_data(data, method)

    _advise_on_n_samples(n_samples)
    _advise_on_sample_size(data)

    single_feature = data.ndim == 1
    if single_feature:
        data = data[:, np.newaxis]

    n_obs, n_features = data.shape
    output_shape = (1,) if single_feature else (n_features,)

    workers = bootstrap_n_jobs_cpu(
        _estimate_data_size_mb(data),
        n_samples,
        memory_budget_gb=memory_budget_gb,
        n_jobs=n_jobs,
    )
    bootstrap_memory_preflight(
        output_shape,
        n_samples,
        confidence_level=confidence_level,
        return_samples=return_samples,
        n_workers=workers,
        memory_budget_gb=memory_budget_gb,
    )
    window = bootstrap_replicate_window(n_samples, n_workers=workers)

    all_indices = generate_bootstrap_indices(
        n_obs, n_samples, random_state=random_state
    )
    estimate = _bootstrap_simple_method_worker(data, method, np.arange(n_obs))

    accumulator = BootstrapAccumulator(
        output_shape,
        n_replicates=n_samples,
        confidence_level=confidence_level,
        retain_samples=return_samples,
    )

    def _compute_one_bootstrap(index):
        return _bootstrap_simple_method_worker(data, method, all_indices[index])

    _run_replicates(
        _compute_one_bootstrap,
        n_samples,
        accumulator,
        n_jobs=workers,
        window=window,
        desc="Bootstrap iterations",
        progress_bar=progress_bar,
    )

    result = _summarize(accumulator, estimate)
    result["backend"] = f"cpu-parallel-{workers}"
    return result


def _as_feature_spaces(X) -> list[np.ndarray]:
    """Normalize training features to a list of 2-D arrays in coefficient order.

    Ordinary ridge supplies one matrix; banded ridge supplies one matrix per
    fitted feature space, already ordered by `Ridge.feature_space_names_`.

    Args:
        X (np.ndarray | Sequence[np.ndarray]): One matrix, or one per space.

    Returns:
        list[np.ndarray]: The feature spaces as float64 arrays.
    """
    spaces = list(X) if isinstance(X, (list, tuple)) else [X]
    return [np.asarray(space, dtype=np.float64) for space in spaces]


def _stack_feature_spaces(X) -> np.ndarray:
    """Concatenate prediction features into one matrix in coefficient order.

    Args:
        X (np.ndarray | Sequence[np.ndarray]): One matrix, or one per space.

    Returns:
        np.ndarray: A `(n_rows, n_features)` float64 matrix.
    """
    spaces = _as_feature_spaces(X)
    return spaces[0] if len(spaces) == 1 else np.concatenate(spaces, axis=1)


def _bootstrap_design(feature_spaces, y, backend=None):
    """Convert the training design onto the backend once for every replicate.

    Bootstrap replicates differ only in which rows they draw, so the design and
    the response are converted a single time and each replicate resamples rows
    in place. On a GPU that keeps the host-to-device transfer out of the loop;
    on the CPU it keeps the concatenation out of it.

    Args:
        feature_spaces (Sequence[np.ndarray]): Training features, one matrix per
            fitted space in coefficient order.
        y (np.ndarray): Training targets, shape (n_obs, n_voxels).
        backend (Backend | None): Resolved GPU backend, or None for the CPU.

    Returns:
        _ResidentDesign: The converted design, ready for repeated refits.
    """
    from nltools.models.ridge import _resident_design

    return _resident_design(feature_spaces, y, backend)


def _refit_resample(
    design,
    indices: np.ndarray,
    alpha: float | np.ndarray,
    feature_space_weights: np.ndarray | None = None,
    memory_budget_gb: float | None = None,
) -> np.ndarray:
    """Refit ridge weights on one bootstrap replicate, hyperparameters fixed.

    Every ridge-bootstrap replicate — CPU or GPU, ordinary or banded — routes
    through the package's single fixed-hyperparameter refit, so a replicate
    cannot drift from the full-data fit numerically and never reruns model
    selection. `indices` is applied to the design and the response together, so
    every feature space and the response resample with the same rows.

    Args:
        design (_ResidentDesign): The training design from `_bootstrap_design`.
        indices (np.ndarray): Row indices of the replicate.
        alpha (float | np.ndarray): Scalar or per-target regularization, taken
            from the fitted model and held fixed.
        feature_space_weights (np.ndarray | None): The fitted banded simplex
            weights, shared `(n_spaces,)` or per-target `(n_spaces, n_targets)`.
            None solves the unweighted ordinary system.
        memory_budget_gb (float | None): Budget used to size the refit's
            internal target batch.

    Returns:
        np.ndarray: Weights, shape (n_features, n_voxels).
    """
    from nltools.models.ridge import _refit_fixed_hyperparameters

    return _refit_fixed_hyperparameters(
        design,
        None,
        alpha,
        feature_space_weights,
        memory_budget_gb=memory_budget_gb,
        row_indices=indices,
    )


def _bootstrap_ridge_weights_cpu_parallel(
    X,
    y: np.ndarray,
    alpha: float,
    estimate: np.ndarray,
    n_samples: int = 5000,
    *,
    confidence_level: float = 0.95,
    memory_budget_gb: float | None = None,
    feature_space_weights: np.ndarray | None = None,
    return_samples: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap ridge weights across CPU workers.

    Each replicate calls the shared fixed-hyperparameter refit directly on numpy
    arrays (no `BrainData` serialization), which is 10-100x faster than a naive
    implementation.

    Args:
        X (np.ndarray | list[np.ndarray]): Feature matrix, shape (n_obs,
            n_features), or one matrix per banded feature space in fitted order.
        y (np.ndarray): Target matrix, shape (n_obs, n_voxels) or (n_obs,).
        alpha (float): Ridge regularization parameter, held fixed.
        estimate (np.ndarray): The fitted full-data coefficients, shape
            (n_features, n_voxels). Only the fitted model knows them, so the
            caller supplies them rather than the engine refitting.
        n_samples (int): Number of bootstrap replicates. Defaults to 5000.
        confidence_level (float): Interval confidence level. Defaults to 0.95.
        memory_budget_gb (float | None): Working-memory budget in GB governing
            the output preflight and worker planning. None measures the host.
        feature_space_weights (np.ndarray | None): Fitted banded simplex
            weights held fixed across replicates. None for ordinary ridge.
        return_samples (bool): Retain every replicate. Defaults to False.
        n_jobs (int): CPU worker ceiling (-1 = all cores). Defaults to -1.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over replicates. Defaults to False.

    Returns:
        dict[str, np.ndarray]: `'estimate'`, `'standard_error'`, `'ci_lower'`,
            `'ci_upper'`, `'samples'` (only with `return_samples=True`), and
            `'backend'`.

    Examples:
        ```python
        X = np.random.randn(100, 10)  # 100 observations, 10 features
        y = np.random.randn(100, 50)  # 100 observations, 50 voxels
        coef = _refit_fixed_hyperparameters([X], y, 1.0)
        result = _bootstrap_ridge_weights_cpu_parallel(X, y, 1.0, coef)
        result["estimate"].shape  # → (10, 50)
        ```
    """
    from nltools.algorithms.backends import (
        bootstrap_memory_preflight,
        bootstrap_n_jobs_cpu,
        bootstrap_replicate_window,
        _estimate_data_size_mb,
    )

    spaces = _as_feature_spaces(X)
    y = np.asarray(y, dtype=np.float64)

    validate_array_shape_range(y, 1, 2, name="y")
    for space in spaces:
        validate_array_shape(space, 2, name="X")
        validate_shape_compatibility(space, y, X_name="X", y_name="y")
    validate_n_samples(n_samples)
    validate_confidence_level(confidence_level)
    validate_memory_budget(memory_budget_gb)
    _advise_on_n_samples(n_samples)

    if y.ndim == 1:
        y = y[:, np.newaxis]

    n_obs = spaces[0].shape[0]
    n_features = sum(space.shape[1] for space in spaces)
    output_shape = (n_features, y.shape[1])

    workers = bootstrap_n_jobs_cpu(
        _estimate_data_size_mb(y),
        n_samples,
        memory_budget_gb=memory_budget_gb,
        n_jobs=n_jobs,
    )
    bootstrap_memory_preflight(
        output_shape,
        n_samples,
        confidence_level=confidence_level,
        return_samples=return_samples,
        n_workers=workers,
        memory_budget_gb=memory_budget_gb,
    )
    window = bootstrap_replicate_window(n_samples, n_workers=workers)

    all_indices = generate_bootstrap_indices(
        n_obs, n_samples, random_state=random_state
    )
    accumulator = BootstrapAccumulator(
        output_shape,
        n_replicates=n_samples,
        confidence_level=confidence_level,
        retain_samples=return_samples,
    )
    design = _bootstrap_design(spaces, y)

    def _compute_one_bootstrap(index):
        return _refit_resample(
            design,
            all_indices[index],
            alpha,
            feature_space_weights,
            memory_budget_gb=memory_budget_gb,
        )

    _run_replicates(
        _compute_one_bootstrap,
        n_samples,
        accumulator,
        n_jobs=workers,
        window=window,
        desc="Bootstrap Ridge weights",
        progress_bar=progress_bar,
    )

    result = _summarize(accumulator, estimate)
    result["backend"] = f"cpu-parallel-{workers}"
    return result


def _bootstrap_ridge_predict_cpu_parallel(
    X,
    y: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
    estimate: np.ndarray,
    n_samples: int = 5000,
    *,
    confidence_level: float = 0.95,
    memory_budget_gb: float | None = None,
    feature_space_weights: np.ndarray | None = None,
    return_samples: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap ridge predictions across CPU workers.

    Each replicate refits the ridge model on the resampled training data and
    applies the refitted coefficients to the unchanged `X_pred`.

    Args:
        X (np.ndarray | list[np.ndarray]): Training feature matrix, shape
            (n_obs, n_features), or one matrix per banded feature space in
            fitted order.
        y (np.ndarray): Training target matrix, shape (n_obs, n_voxels) or
            (n_obs,).
        X_pred (np.ndarray | list[np.ndarray]): Test feature matrix, shape
            (n_test, n_features), or one matrix per banded feature space in
            fitted order.
        alpha (float): Ridge regularization parameter, held fixed.
        estimate (np.ndarray): The fitted full-data model evaluated at `X_pred`,
            shape (n_test, n_voxels).
        n_samples (int): Number of bootstrap replicates. Defaults to 5000.
        confidence_level (float): Interval confidence level. Defaults to 0.95.
        memory_budget_gb (float | None): Working-memory budget in GB governing
            the output preflight and worker planning. None measures the host.
        feature_space_weights (np.ndarray | None): Fitted banded simplex
            weights held fixed across replicates. None for ordinary ridge.
        return_samples (bool): Retain every replicate. Defaults to False.
        n_jobs (int): CPU worker ceiling (-1 = all cores). Defaults to -1.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over replicates. Defaults to False.

    Returns:
        dict[str, np.ndarray]: `'estimate'`, `'standard_error'`, `'ci_lower'`,
            `'ci_upper'`, `'samples'` (only with `return_samples=True`), and
            `'backend'`.

    Examples:
        ```python
        X = np.random.randn(100, 10)  # training features
        y = np.random.randn(100, 50)  # training targets (50 voxels)
        X_test = np.random.randn(20, 10)  # test features
        coef = _refit_fixed_hyperparameters([X], y, 1.0)
        result = _bootstrap_ridge_predict_cpu_parallel(
            X, y, X_test, 1.0, X_test @ coef
        )
        result["estimate"].shape  # → (20, 50)
        ```
    """
    from nltools.algorithms.backends import (
        bootstrap_memory_preflight,
        bootstrap_n_jobs_cpu,
        bootstrap_replicate_window,
        _estimate_data_size_mb,
    )

    spaces = _as_feature_spaces(X)
    y = np.asarray(y, dtype=np.float64)
    X_pred = _stack_feature_spaces(X_pred)

    validate_array_shape_range(y, 1, 2, name="y")
    for space in spaces:
        validate_array_shape(space, 2, name="X")
        validate_shape_compatibility(space, y, X_name="X", y_name="y")
    validate_array_shape(X_pred, 2, name="X_pred")
    n_features = sum(space.shape[1] for space in spaces)
    if n_features != X_pred.shape[1]:
        raise ValueError(
            f"X and X_pred must have same n_features: {n_features} != {X_pred.shape[1]}"
        )
    validate_n_samples(n_samples)
    validate_confidence_level(confidence_level)
    validate_memory_budget(memory_budget_gb)
    _advise_on_n_samples(n_samples)

    if y.ndim == 1:
        y = y[:, np.newaxis]

    n_obs = spaces[0].shape[0]
    output_shape = (X_pred.shape[0], y.shape[1])

    workers = bootstrap_n_jobs_cpu(
        _estimate_data_size_mb(y),
        n_samples,
        memory_budget_gb=memory_budget_gb,
        n_jobs=n_jobs,
    )
    bootstrap_memory_preflight(
        output_shape,
        n_samples,
        confidence_level=confidence_level,
        return_samples=return_samples,
        n_workers=workers,
        memory_budget_gb=memory_budget_gb,
    )
    window = bootstrap_replicate_window(n_samples, n_workers=workers)

    all_indices = generate_bootstrap_indices(
        n_obs, n_samples, random_state=random_state
    )
    accumulator = BootstrapAccumulator(
        output_shape,
        n_replicates=n_samples,
        confidence_level=confidence_level,
        retain_samples=return_samples,
    )
    design = _bootstrap_design(spaces, y)

    def _compute_one_bootstrap(index):
        weights = _refit_resample(
            design,
            all_indices[index],
            alpha,
            feature_space_weights,
            memory_budget_gb=memory_budget_gb,
        )
        return X_pred @ weights

    _run_replicates(
        _compute_one_bootstrap,
        n_samples,
        accumulator,
        n_jobs=workers,
        window=window,
        desc="Bootstrap Ridge predictions",
        progress_bar=progress_bar,
    )

    result = _summarize(accumulator, estimate)
    result["backend"] = f"cpu-parallel-{workers}"
    return result


def _auto_batch_size_ridge(
    n_bootstrap: int,
    n_samples: int,
    n_features: int,
    n_voxels: int,
    output_shape: tuple[int, ...],
    max_memory_gb: float | None = None,
    backend=None,
) -> tuple[int, int]:
    """Determine the Ridge-bootstrap GPU batch size for a memory budget.

    Forwards to `backends.ridge_bootstrap_batch_size`, which owns the budget
    measurement, the working-set model, and the batch arithmetic. This wrapper
    only supplies the solver's working dtype size: MPS solves in float32,
    every other backend in float64.

    Args:
        n_bootstrap (int): Total number of bootstrap replicates.
        n_samples (int): Number of observations in the dataset.
        n_features (int): Total feature count across all feature spaces.
        n_voxels (int): Number of voxels/targets.
        output_shape (tuple[int, ...]): Shape of one retained replicate result.
        max_memory_gb (float | None): Explicit memory budget in GB. None
            (default) measures the device via `device_memory_budget`.
        backend (Backend | None): Resolved backend the work runs on.

    Returns:
        tuple[int, int]: `(batch_size, n_batches)`.
    """
    from nltools.algorithms.backends import ridge_bootstrap_batch_size

    device = getattr(backend, "device", None)
    return ridge_bootstrap_batch_size(
        n_bootstrap,
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_voxels,
        output_shape=output_shape,
        device_itemsize=4 if device == "mps" else 8,
        max_gpu_memory_gb=max_memory_gb,
        backend=backend,
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
    feature_spaces: list[np.ndarray],
    y: np.ndarray,
    alpha: float,
    *,
    estimate: np.ndarray,
    compute_sample,
    output_shape: tuple[int, ...],
    desc: str,
    confidence_level: float = 0.95,
    feature_space_weights: np.ndarray | None = None,
    n_samples: int = 5000,
    return_samples: bool = False,
    backend=None,
    memory_budget_gb: float | None = None,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Shared GPU bootstrap driver for ridge statistics, with automatic batching.

    Owns everything the weights and predict bootstraps have in common — pre-drawn
    resample indices, batch sizing via `_auto_batch_size_ridge`, the per-replicate
    refit inside an OOM-safe batch loop, `BootstrapAccumulator` aggregation on
    the CPU, the progress bar, and result formatting. The per-replicate statistic
    is injected via `compute_sample`.

    Each replicate goes through the package's shared fixed-hyperparameter refit
    (`nltools.models.ridge._refit_fixed_hyperparameters`) under the scoped
    Himalaya GPU backend, so the GPU path solves the same equation as the CPU
    path and supports banded models. The batch loop exists for memory control:
    it bounds how many resamples are in flight before aggregation.

    Args:
        feature_spaces (list[np.ndarray]): Training features, one matrix per
            fitted space in coefficient order.
        y (np.ndarray): Target matrix, shape (n_obs, n_voxels), 2-D.
        alpha (float): Ridge regularization held fixed across replicates.
        estimate (np.ndarray): The full-data statistic, shape `output_shape`.
        compute_sample (Callable): `(coef) -> np.ndarray`, mapping one
            replicate's coefficients, shape (n_features, n_voxels), to the
            statistic aggregated on the CPU (the weights themselves, or
            predictions from them).
        output_shape (tuple[int, ...]): Shape of each `compute_sample` result.
        desc (str): Progress-bar description.
        confidence_level (float): Interval confidence level. Defaults to 0.95.
        feature_space_weights (np.ndarray | None): Fitted banded simplex
            weights held fixed across replicates. None for ordinary ridge.
        n_samples (int): Number of bootstrap replicates. Defaults to 5000.
        return_samples (bool): Retain every replicate. Defaults to False.
        backend (Backend | None): Backend instance (must be a GPU torch backend).
            None auto-selects.
        memory_budget_gb (float | None): Working-memory budget in GB. None
            (default) measures the device's available memory.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over replicates. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Bootstrap statistics in the same format as the CPU
            engines, with `'backend'` set to `'gpu-<device>'`.

    Raises:
        RuntimeError: If a replicate fails, naming its index.
    """
    from nltools.algorithms.backends import (
        auto_select_backend,
        bootstrap_memory_preflight,
        compute_oom_safe,
        is_oom_error,
    )

    n_obs = feature_spaces[0].shape[0]
    n_features = sum(space.shape[1] for space in feature_spaces)
    n_voxels = y.shape[1]

    if backend is None:
        backend = auto_select_backend(n_obs, n_features)
    _validate_gpu_backend(backend)

    validate_n_samples(n_samples)
    validate_confidence_level(confidence_level)
    validate_memory_budget(memory_budget_gb)
    _advise_on_n_samples(n_samples)

    bootstrap_memory_preflight(
        output_shape,
        n_samples,
        confidence_level=confidence_level,
        return_samples=return_samples,
        memory_budget_gb=memory_budget_gb,
        backend=backend,
    )

    all_indices = generate_bootstrap_indices(
        n_obs, n_samples, random_state=random_state
    )

    batch_size, n_batches = _auto_batch_size_ridge(
        n_samples,
        n_obs,
        n_features,
        n_voxels,
        output_shape,
        max_memory_gb=memory_budget_gb,
        backend=backend,
    )

    # One host-to-device transfer for the whole run; replicates resample rows
    # on the device.
    design = _bootstrap_design(feature_spaces, y, backend)

    def _compute_batch(batch: np.ndarray, replicates: np.ndarray) -> np.ndarray:
        """GPU ridge statistics for one (sub-)batch of pre-drawn resample indices.

        `replicates` carries each row's global replicate index so a terminal
        failure names it even after OOM recovery has split the batch. An OOM
        itself is re-raised untouched, because `compute_oom_safe` recovers from
        it by retrying the same rows in smaller pieces.
        """
        batch_results = []
        for replicate, indices in zip(replicates, batch):
            try:
                coef = _refit_resample(
                    design,
                    indices,
                    alpha,
                    feature_space_weights,
                    memory_budget_gb=memory_budget_gb,
                )
            except Exception as error:
                if is_oom_error(error):
                    raise
                raise RuntimeError(
                    f"bootstrap replicate {int(replicate)} failed: {error}"
                ) from error
            batch_results.append(compute_sample(coef))
        return np.array(batch_results)

    accumulator = BootstrapAccumulator(
        output_shape,
        n_replicates=n_samples,
        confidence_level=confidence_level,
        retain_samples=return_samples,
    )

    pbar = make_progress_bar(
        progress_bar=progress_bar,
        total=n_samples,
        desc=desc,
        unit="iter",
        disable=n_batches == 1,
    )
    replicate_ids = np.arange(n_samples)

    for batch_index in range(n_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, n_samples)

        # Bootstrap indices for this batch were all pre-drawn (all_indices),
        # so OOM recovery reuses them exactly; the replicate ids split with them.
        batch_results = compute_oom_safe(
            _compute_batch, all_indices[start:end], replicate_ids[start:end]
        )
        for sample in batch_results:
            accumulator.update(sample)

        pbar.update(end - start)

    pbar.close()

    result = _summarize(accumulator, estimate)
    result["backend"] = f"gpu-{backend.device}"
    return result


def _bootstrap_ridge_weights_gpu_batched(
    X,
    y: np.ndarray,
    alpha: float,
    estimate: np.ndarray,
    n_samples: int = 5000,
    *,
    confidence_level: float = 0.95,
    feature_space_weights: np.ndarray | None = None,
    return_samples: bool = False,
    backend=None,
    memory_budget_gb: float | None = None,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap ridge weights on the GPU with automatic batching.

    Thin wrapper over `_bootstrap_ridge_gpu_batched` whose per-replicate
    statistic is the ridge coefficients themselves.

    Args:
        X (np.ndarray | list[np.ndarray]): Feature matrix, shape (n_obs,
            n_features), or one matrix per banded feature space in fitted order.
        y (np.ndarray): Target matrix, shape (n_obs, n_voxels) or (n_obs,).
        alpha (float): Ridge regularization parameter, held fixed.
        estimate (np.ndarray): The fitted full-data coefficients.
        n_samples (int): Number of bootstrap replicates. Defaults to 5000.
        confidence_level (float): Interval confidence level. Defaults to 0.95.
        feature_space_weights (np.ndarray | None): Fitted banded simplex
            weights held fixed across replicates. None for ordinary ridge.
        return_samples (bool): Retain every replicate. Defaults to False.
        backend (Backend | None): Backend instance (must be a GPU torch backend).
            None auto-selects.
        memory_budget_gb (float | None): Working-memory budget in GB. None
            (default) measures the device's available memory.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over replicates. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Bootstrap statistics in the same format as the CPU
            engine.
    """
    spaces = _as_feature_spaces(X)
    y = np.asarray(y, dtype=np.float64)

    validate_array_shape_range(y, 1, 2, name="y")
    for space in spaces:
        validate_array_shape(space, 2, name="X")
        validate_shape_compatibility(space, y, X_name="X", y_name="y")

    if y.ndim == 1:
        y = y[:, np.newaxis]
    n_features = sum(space.shape[1] for space in spaces)

    return _bootstrap_ridge_gpu_batched(
        spaces,
        y,
        alpha,
        estimate=estimate,
        compute_sample=lambda coef: coef,
        output_shape=(n_features, y.shape[1]),
        desc="GPU bootstrap Ridge weights",
        confidence_level=confidence_level,
        feature_space_weights=feature_space_weights,
        n_samples=n_samples,
        return_samples=return_samples,
        backend=backend,
        memory_budget_gb=memory_budget_gb,
        random_state=random_state,
        progress_bar=progress_bar,
    )


def _bootstrap_ridge_predict_gpu_batched(
    X,
    y: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
    estimate: np.ndarray,
    n_samples: int = 5000,
    *,
    confidence_level: float = 0.95,
    feature_space_weights: np.ndarray | None = None,
    return_samples: bool = False,
    backend=None,
    memory_budget_gb: float | None = None,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Bootstrap ridge predictions on the GPU with automatic batching.

    Thin wrapper over `_bootstrap_ridge_gpu_batched` whose per-replicate
    statistic is `X_pred @ coef`.

    Args:
        X (np.ndarray | list[np.ndarray]): Training feature matrix, shape
            (n_obs, n_features), or one matrix per banded feature space in
            fitted order.
        y (np.ndarray): Training target matrix, shape (n_obs, n_voxels) or
            (n_obs,).
        X_pred (np.ndarray | list[np.ndarray]): Test feature matrix, shape
            (n_test, n_features), or one matrix per banded feature space
            in fitted order.
        alpha (float): Ridge regularization parameter, held fixed.
        estimate (np.ndarray): The fitted full-data model evaluated at `X_pred`.
        n_samples (int): Number of bootstrap replicates. Defaults to 5000.
        confidence_level (float): Interval confidence level. Defaults to 0.95.
        feature_space_weights (np.ndarray | None): Fitted banded simplex
            weights held fixed across replicates. None for ordinary ridge.
        return_samples (bool): Retain every replicate. Defaults to False.
        backend (Backend | None): Backend instance (must be a GPU torch backend).
            None auto-selects.
        memory_budget_gb (float | None): Working-memory budget in GB. None
            (default) measures the device's available memory.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over replicates. Defaults to False.

    Returns:
        dict[str, np.ndarray]: Bootstrap statistics in the same format as the CPU
            engine.
    """
    spaces = _as_feature_spaces(X)
    y = np.asarray(y, dtype=np.float64)
    X_pred = _stack_feature_spaces(X_pred)

    validate_array_shape_range(y, 1, 2, name="y")
    for space in spaces:
        validate_array_shape(space, 2, name="X")
        validate_shape_compatibility(space, y, X_name="X", y_name="y")
    validate_array_shape(X_pred, 2, name="X_pred")
    n_features = sum(space.shape[1] for space in spaces)
    if n_features != X_pred.shape[1]:
        raise ValueError(
            f"X and X_pred must have same n_features: {n_features} != {X_pred.shape[1]}"
        )

    if y.ndim == 1:
        y = y[:, np.newaxis]

    return _bootstrap_ridge_gpu_batched(
        spaces,
        y,
        alpha,
        estimate=estimate,
        compute_sample=lambda coef: X_pred @ coef,
        output_shape=(X_pred.shape[0], y.shape[1]),
        desc="GPU bootstrap Ridge predictions",
        confidence_level=confidence_level,
        feature_space_weights=feature_space_weights,
        n_samples=n_samples,
        return_samples=return_samples,
        backend=backend,
        memory_budget_gb=memory_budget_gb,
        random_state=random_state,
        progress_bar=progress_bar,
    )
