"""Bootstrap resampling for `BrainData` — aggregate statistics and Ridge model statistics.

`BrainData.bootstrap` delegates here.
"""

import numpy as np

from .utils import _result_from_array

#: Statistics that reduce `bd.data` directly and need no fitted model.
SIMPLE_STATS = ("mean", "median", "std", "sum", "min", "max")

#: Statistics that resample a fitted `Ridge` and therefore need explicit
#: training features. `'predict'` additionally needs evaluation features.
FITTED_STATS = ("weights", "predict")


def bootstrap(
    bd,
    statistic,
    *,
    X=None,
    X_test=None,
    n_samples=5000,
    confidence_level=0.95,
    device="cpu",
    memory_budget_gb=None,
    return_samples=False,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
):
    """Bootstrap a statistic and its uncertainty, on CPU workers or a GPU.

    Resamples observations with replacement and aggregates the replicates as
    they complete, into a running Welford variance plus just enough retained
    order statistics per output element to reproduce the exact percentile
    interval. What the run holds is that retained tail — about
    `(1 - confidence_level)` of the replicates per element — plus one dispatch
    window, rather than all `n_samples` maps. This is memory-efficient, not
    constant-memory: the tail still grows with `n_samples`, and
    `return_samples=True` keeps the whole distribution.

    A Ridge bootstrap resamples the explicitly supplied training `X` together
    with `bd.data`, using the same row indices for every feature space, and
    refits with the fitted model's selected `alpha_` — and, for a banded model,
    its `feature_space_weights_` — held fixed. It never reruns cross-validation
    or the banded random search.

    Args:
        bd (BrainData): Data to resample.
        statistic (str): Statistic to bootstrap. Basic aggregates: ``'mean'``,
            ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'``. Model
            statistics (require a fitted `Ridge`): ``'weights'`` or
            ``'predict'``.
        X (np.ndarray | Mapping[str, np.ndarray] | None): Training features in
            their original row order, required by both Ridge statistics and
            rejected by the basic ones. A matrix for ordinary Ridge; a mapping
            with exactly the fitted feature-space names for banded Ridge.
        X_test (np.ndarray | Mapping[str, np.ndarray] | None): Evaluation
            features for ``statistic='predict'``, in the same structure as `X`.
            It may have any row count.
        n_samples (int): Number of bootstrap replicates, at least two. Default
            ``5000``.
        confidence_level (float): Confidence level of the reported interval,
            strictly between zero and one. Default ``0.95``.
        device (str): Compute device for Ridge refits: ``'cpu'`` (default) or
            ``'gpu'`` (PyTorch on CUDA/MPS; raises if neither is available).
            Basic statistics reject ``'gpu'``.
        memory_budget_gb (float | None): Working-memory budget in GB. It
            governs the output preflight and CPU-worker planning for every
            statistic, and GPU batch sizing for the Ridge ones. ``None``
            (default) measures the device.
        return_samples (bool): Retain and return every replicate. Default
            ``False``. It changes retention only, never interval semantics.
        n_jobs (int): CPU worker ceiling. Default ``-1`` (all cores); the
            planner may use fewer.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar. Default ``False``.

    Returns:
        BootstrapResult: `estimate` (the statistic on the unresampled full
            sample), `standard_error`, `ci_lower` and `ci_upper` as `BrainData`
            maps of identical shape, plus `samples` as a NumPy array with the
            bootstrap axis first when ``return_samples=True``.

    Raises:
        ValueError: If `statistic` is unknown, a basic statistic is given `X`,
            `X_test`, or ``device='gpu'``, a Ridge statistic is missing `X` (or
            `X_test` for ``'predict'``), the fitted model is not a `Ridge`, `X`
            does not match the fitted feature structure and observation count,
            an argument is out of range, or the retained output cannot fit the
            memory budget.

    Examples:
        ```python
        boot = brain.bootstrap('mean', n_samples=1000)
        boot.estimate.plot()

        brain.fit(model='ridge', X=features, ridge_alpha=1.0)
        boot = brain.bootstrap('weights', X=features, n_samples=1000)
        ```

    Note:
        This is an IID row bootstrap: rows must be exchangeable for the
        interval to mean anything. Fitting retains no hidden copy of the
        training features, so omitting `X` raises even when the same features
        were supplied to `fit`.
    """
    from nltools.algorithms.inference.bootstrap import (
        _bootstrap_simple_cpu_parallel,
        _bootstrap_ridge_weights_cpu_parallel,
        _bootstrap_ridge_predict_cpu_parallel,
        _bootstrap_ridge_weights_gpu_batched,
        _bootstrap_ridge_predict_gpu_batched,
    )

    _validate_statistic(statistic)
    _validate_arguments(n_samples, confidence_level, memory_budget_gb, device)

    if statistic in SIMPLE_STATS:
        _reject_model_arguments(statistic, X, X_test, device)
        result = _bootstrap_simple_cpu_parallel(
            bd.data,
            method=statistic,
            n_samples=n_samples,
            confidence_level=confidence_level,
            memory_budget_gb=memory_budget_gb,
            return_samples=return_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )
        return _as_bootstrap_result(bd, result)

    model = _fitted_ridge(bd, statistic)
    spaces = _training_feature_spaces(model, X, statistic, bd.shape[0])
    if statistic == "weights":
        if X_test is not None:
            raise ValueError(
                "bootstrap('weights') summarizes training coefficients and "
                "takes no X_test; use statistic='predict' to evaluate on new rows."
            )
    elif X_test is None:
        raise ValueError(
            "X_test parameter required for bootstrap(statistic='predict'). "
            "Provide test features: bootstrap('predict', X=..., X_test=...)"
        )

    backend = _resolve_device(device)
    alpha = model.alpha_
    feature_space_weights = model.feature_space_weights_
    coefficients = np.asarray(model.coef_, dtype=np.float64)

    shared = {
        "n_samples": n_samples,
        "confidence_level": confidence_level,
        "feature_space_weights": feature_space_weights,
        "return_samples": return_samples,
        "random_state": random_state,
        "progress_bar": progress_bar,
    }

    if statistic == "weights":
        if backend is None:
            result = _bootstrap_ridge_weights_cpu_parallel(
                spaces,
                bd.data,
                alpha,
                coefficients,
                memory_budget_gb=memory_budget_gb,
                n_jobs=n_jobs,
                **shared,
            )
        else:
            result = _bootstrap_ridge_weights_gpu_batched(
                spaces,
                bd.data,
                alpha,
                coefficients,
                backend=backend,
                memory_budget_gb=memory_budget_gb,
                **shared,
            )
        return _as_bootstrap_result(bd, result)

    # `Ridge` owns the alignment of both feature arguments, through the one
    # seam this facade uses; the engines concatenate in coefficient order.
    test_spaces = model._aligned_feature_spaces(X_test)
    predictions = _stacked(test_spaces) @ coefficients

    if backend is None:
        result = _bootstrap_ridge_predict_cpu_parallel(
            spaces,
            bd.data,
            test_spaces,
            alpha,
            predictions,
            memory_budget_gb=memory_budget_gb,
            n_jobs=n_jobs,
            **shared,
        )
    else:
        result = _bootstrap_ridge_predict_gpu_batched(
            spaces,
            bd.data,
            test_spaces,
            alpha,
            predictions,
            backend=backend,
            memory_budget_gb=memory_budget_gb,
            **shared,
        )

    return _as_bootstrap_result(bd, result)


def _validate_statistic(statistic):
    """Reject anything outside the closed set of eight supported statistics.

    Args:
        statistic (str): The requested statistic.

    Raises:
        ValueError: If `statistic` is not one of the eight names.
    """
    if statistic in SIMPLE_STATS or statistic in FITTED_STATS:
        return
    raise ValueError(
        f"Unsupported statistic '{statistic}'. "
        f"Supported basic statistics: {list(SIMPLE_STATS)}. "
        f"Supported fitted model statistics: {list(FITTED_STATS)}. "
        f"For fitted statistics, you must call .fit() first."
    )


def _validate_arguments(n_samples, confidence_level, memory_budget_gb, device):
    """Check every mode-independent argument before any resampling begins.

    Range-checking only. The quality advisory for a low `n_samples` belongs to
    the engine, which every path reaches, so the user sees it exactly once.

    Args:
        n_samples (int): Requested replicate count.
        confidence_level (float): Requested interval level.
        memory_budget_gb (float | None): Requested working-memory budget.
        device (str): Requested compute device.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range, or `device` is not
            ``'cpu'`` or ``'gpu'``.
    """
    from nltools.algorithms.inference.bootstrap import (
        _validate_confidence_level,
        _validate_memory_budget,
        _validate_n_samples,
    )

    _validate_n_samples(n_samples)
    _validate_confidence_level(confidence_level)
    _validate_memory_budget(memory_budget_gb)
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")


def _reject_model_arguments(statistic, X, X_test, device):
    """Reject the arguments a basic statistic has no use for.

    Args:
        statistic (str): The requested basic statistic, for the message.
        X (Any): Training features, which must be absent.
        X_test (Any): Evaluation features, which must be absent.
        device (str): Requested compute device, which must be ``'cpu'``.

    Raises:
        ValueError: If features were supplied, or a GPU was requested.
    """
    if X is not None or X_test is not None:
        raise ValueError(
            f"bootstrap({statistic!r}) reduces the data itself and takes no "
            f"features; X and X_test belong to statistic='weights' or 'predict'."
        )
    if device != "cpu":
        raise ValueError(
            f"bootstrap({statistic!r}) is a NumPy reduction over rows and runs "
            f"on the CPU; device='gpu' applies only to the Ridge statistics."
        )


def _stacked(spaces):
    """Concatenate aligned feature spaces into one matrix in coefficient order.

    Args:
        spaces (Sequence[np.ndarray]): One matrix per fitted feature space.

    Returns:
        np.ndarray: A `(n_rows, n_features)` float64 matrix.
    """
    matrices = [np.asarray(space, dtype=np.float64) for space in spaces]
    return matrices[0] if len(matrices) == 1 else np.concatenate(matrices, axis=1)


def _fitted_ridge(bd, statistic):
    """Return the fitted `Ridge` a model bootstrap needs, or raise.

    Args:
        bd (BrainData): The object being resampled.
        statistic (str): The requested model statistic, for the error message.

    Returns:
        Ridge: The fitted estimator.

    Raises:
        ValueError: If nothing is fitted, or the fit is not a `Ridge`.
    """
    from nltools.models import Ridge

    model = getattr(bd, "model_", None)
    if model is None or not getattr(model, "is_fitted_", False):
        raise ValueError(
            f"Must call .fit(model='ridge', X=features) before bootstrap('{statistic}')"
        )
    if not isinstance(model, Ridge):
        raise ValueError(
            f"bootstrap('{statistic}') only supports a fitted Ridge, but this "
            f"BrainData holds a fitted {type(model).__name__}."
        )
    return model


def _training_feature_spaces(model, X, statistic, n_obs):
    """Align the explicit training `X` to the fitted feature-space order.

    Args:
        model (Ridge): The fitted estimator.
        X (np.ndarray | Mapping | None): Caller-supplied training features.
        statistic (str): The requested model statistic, for the error message.
        n_obs (int): Observation count of the `BrainData` being resampled.

    Returns:
        list[np.ndarray]: One matrix per fitted feature space, in coefficient
            order.

    Raises:
        ValueError: If `X` is missing, does not match the fitted feature
            structure, or has a different number of rows than the response.
    """
    if X is None:
        raise ValueError(
            f"bootstrap('{statistic}') requires the training features as X=. "
            f"Fitting keeps no copy of them, so pass the same features you "
            f"passed to fit()."
        )
    spaces = model._aligned_feature_spaces(X)
    rows = spaces[0].shape[0]
    if rows != n_obs:
        raise ValueError(
            f"X has {rows} rows, but the fitted BrainData has {n_obs} "
            f"observations; the training features must be in their original "
            f"row order."
        )
    return spaces


def _resolve_device(device):
    """Resolve the bootstrap `device` request to a GPU backend, or None for CPU.

    Args:
        device (str): ``'cpu'`` or ``'gpu'``.

    Returns:
        Backend | None: A resolved GPU backend, or None to stay on the CPU.

    Raises:
        ValueError: If `device` is not one of the two supported values, or
            ``'gpu'`` was requested with no accelerator available.
    """
    from nltools.algorithms.backends import check_gpu_available, resolve_backend

    if device == "cpu":
        return None
    if device != "gpu":
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if not check_gpu_available()[0]:
        raise ValueError(
            "GPU requested via device='gpu' but no CUDA or MPS device is "
            "available. Use device='cpu'."
        )
    return resolve_backend("gpu")


def _as_bootstrap_result(bd, result):
    """Wrap an engine's arrays as a `BootstrapResult` of `BrainData` maps.

    Args:
        bd (BrainData): Template whose mask and spatial state the outputs
            inherit; row metadata and fitted state are cleared.
        result (dict): Engine output with `'estimate'`, `'standard_error'`,
            `'ci_lower'`, `'ci_upper'`, and optionally `'samples'`.

    Returns:
        BootstrapResult: The four summaries as `BrainData`, and the retained
            replicates as a NumPy array when present.
    """
    from nltools.data.results import BootstrapResult

    return BootstrapResult(
        estimate=_result_from_array(bd, result["estimate"], rows="clear"),
        standard_error=_result_from_array(bd, result["standard_error"], rows="clear"),
        ci_lower=_result_from_array(bd, result["ci_lower"], rows="clear"),
        ci_upper=_result_from_array(bd, result["ci_upper"], rows="clear"),
        samples=result.get("samples"),
    )
