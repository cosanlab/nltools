"""Bootstrap resampling for `BrainData` — aggregate statistics and Ridge model statistics.

`BrainData.bootstrap` delegates here.
"""

from .utils import _result_from_array

#: Statistics that reduce `bd.data` directly and need no fitted model.
SIMPLE_STATS = ("mean", "median", "std", "sum", "min", "max")

#: Statistics that resample a fitted `Ridge` and therefore need explicit
#: training features. `'predict'` additionally needs evaluation features.
FITTED_STATS = ("weights", "predict")


def bootstrap(
    bd,
    stat,
    *,
    X=None,
    X_test=None,
    n_samples=5000,
    save_boots=False,
    percentiles=(2.5, 97.5),
    device="cpu",
    memory_budget_gb=None,
    tail=2,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
):
    """Bootstrap statistics with CPU parallelization or GPU acceleration.

    Supports simple aggregation statistics and fitted `Ridge` statistics. A
    Ridge bootstrap resamples the explicitly supplied training `X` together
    with `bd.data`, using the same row indices for every feature space, and
    refits with the fitted model's selected `alpha_` — and, for a banded model,
    its `feature_space_weights_` — held fixed. It never reruns cross-validation
    or the banded random search.

    Note: the CPU path pre-generates all resample indices and collects every
    per-sample result, so peak memory grows with ``n_samples`` (it is not a
    streaming/online accumulator).

    Args:
        bd (BrainData): Data to resample.
        stat (str): Statistic to bootstrap. Simple aggregates: ``'mean'``,
            ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'``. Model
            statistics (require a fitted `Ridge`): ``'weights'`` or
            ``'predict'``.
        X (np.ndarray | Mapping[str, np.ndarray] | None): Training features in
            their original row order, required by both Ridge statistics and
            rejected by the simple ones. A matrix for ordinary Ridge; a mapping
            with exactly the fitted feature-space names for banded Ridge.
        X_test (np.ndarray | Mapping[str, np.ndarray] | None): Evaluation
            features for ``stat='predict'``, in the same structure as `X`. It
            may have any row count.
        n_samples (int): Number of bootstrap iterations. Default ``5000``.
        save_boots (bool): Keep every bootstrap sample (memory intensive).
            Default ``False``.
        percentiles (tuple[float, float]): Percentiles for the confidence
            interval. Default ``(2.5, 97.5)``.
        device (str): Compute device for Ridge bootstraps: ``'cpu'`` (default)
            or ``'gpu'`` (PyTorch on CUDA/MPS; raises if neither is available).
        memory_budget_gb (float | None): Working-memory budget in GB used to
            size GPU batches. ``None`` (default) measures the device.
        tail (int | str): ``2``/``'two'`` for two-tailed (default);
            ``1``/``'one'`` for one-tailed (statistic > 0; negate the data for
            the other direction).
        n_jobs (int): CPU workers for parallelization. Default ``-1`` (all CPUs).
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar. Default ``False``.

    Returns:
        BrainData | dict: For simple stats with ``save_boots=False``, a
            `BrainData` holding the bootstrap mean. For model stats, a dict with
            keys ``'mean'``, ``'std'``, ``'Z'``, ``'p'``, ``'ci_lower'``,
            ``'ci_upper'`` (all `BrainData`). With ``save_boots=True``, a dict
            (even for simple stats) with an added ``'samples'`` key holding the
            raw sample array.

    Raises:
        ValueError: If `stat` is unknown, a simple statistic is given `X` or
            `X_test`, a Ridge statistic is missing `X` (or `X_test` for
            ``'predict'``), the fitted model is not a `Ridge`, or `X` does not
            match the fitted feature structure and observation count.

    Examples:
        ```python
        # Simple aggregation: returns a BrainData holding the bootstrap mean
        boot = brain.bootstrap(stat='mean', n_samples=1000)

        # Ridge weights bootstrap: the training features are passed explicitly
        brain.fit(model='ridge', X=features, ridge_alpha=1.0)
        boot = brain.bootstrap(stat='weights', X=features, n_samples=1000)

        # Ridge prediction bootstrap, GPU accelerated
        boot = brain.bootstrap(
            stat='predict', X=features, X_test=X_new, n_samples=1000, device='gpu'
        )
        ```

    Note:
        Fitting retains no hidden copy of the training features, so omitting
        `X` raises even when the same features were supplied to `fit`.
    """
    from nltools.algorithms.inference.bootstrap import (
        _bootstrap_simple_cpu_parallel,
        _bootstrap_ridge_weights_cpu_parallel,
        _bootstrap_ridge_predict_cpu_parallel,
        _bootstrap_ridge_weights_gpu_batched,
        _bootstrap_ridge_predict_gpu_batched,
    )

    if stat in SIMPLE_STATS:
        if X is not None or X_test is not None:
            raise ValueError(
                f"bootstrap(stat={stat!r}) reduces the data itself and takes no "
                f"features; X and X_test belong to stat='weights' or 'predict'."
            )
        result = _bootstrap_simple_cpu_parallel(
            bd.data,
            method=stat,
            n_samples=n_samples,
            save_boots=save_boots,
            n_jobs=n_jobs,
            random_state=random_state,
            percentiles=percentiles,
            tail=tail,
            progress_bar=progress_bar,
        )
        return convert_bootstrap_results_to_brain_data(
            bd, result, save_boots=save_boots, return_dict=False
        )

    if stat not in FITTED_STATS:
        raise ValueError(
            f"Unsupported stat '{stat}'. "
            f"Supported simple stats: {list(SIMPLE_STATS)}. "
            f"Supported fitted model stats: {list(FITTED_STATS)}. "
            f"For fitted stats, you must call .fit() first."
        )

    model = _fitted_ridge(bd, stat)
    spaces = _training_feature_spaces(model, X, stat, bd.shape[0])
    if stat == "weights":
        if X_test is not None:
            raise ValueError(
                "bootstrap(stat='weights') summarizes training coefficients and "
                "takes no X_test; use stat='predict' to evaluate on new rows."
            )
    elif X_test is None:
        raise ValueError(
            "X_test parameter required for bootstrap(stat='predict'). "
            "Provide test features: bootstrap(stat='predict', X=..., X_test=...)"
        )

    backend = _resolve_device(device)
    alpha = model.alpha_
    feature_space_weights = model.feature_space_weights_

    if stat == "weights":
        if backend is None:
            result = _bootstrap_ridge_weights_cpu_parallel(
                spaces,
                bd.data,
                alpha,
                n_samples=n_samples,
                save_boots=save_boots,
                feature_space_weights=feature_space_weights,
                n_jobs=n_jobs,
                random_state=random_state,
                percentiles=percentiles,
                tail=tail,
                progress_bar=progress_bar,
            )
        else:
            result = _bootstrap_ridge_weights_gpu_batched(
                spaces,
                bd.data,
                alpha,
                n_samples=n_samples,
                save_boots=save_boots,
                feature_space_weights=feature_space_weights,
                backend=backend,
                max_gpu_memory_gb=memory_budget_gb,
                random_state=random_state,
                percentiles=percentiles,
                tail=tail,
                progress_bar=progress_bar,
            )
        return convert_bootstrap_results_to_brain_data(
            bd, result, save_boots=save_boots, return_dict=True
        )

    # `Ridge` owns the alignment of both feature arguments, through the one
    # seam this facade uses; the engines concatenate in coefficient order.
    test_spaces = model._aligned_feature_spaces(X_test)

    if backend is None:
        result = _bootstrap_ridge_predict_cpu_parallel(
            spaces,
            bd.data,
            test_spaces,
            alpha,
            n_samples=n_samples,
            save_boots=save_boots,
            feature_space_weights=feature_space_weights,
            n_jobs=n_jobs,
            random_state=random_state,
            percentiles=percentiles,
            tail=tail,
            progress_bar=progress_bar,
        )
    else:
        result = _bootstrap_ridge_predict_gpu_batched(
            spaces,
            bd.data,
            test_spaces,
            alpha,
            n_samples=n_samples,
            save_boots=save_boots,
            feature_space_weights=feature_space_weights,
            backend=backend,
            max_gpu_memory_gb=memory_budget_gb,
            random_state=random_state,
            percentiles=percentiles,
            tail=tail,
            progress_bar=progress_bar,
        )

    return convert_bootstrap_results_to_brain_data(
        bd, result, save_boots=save_boots, return_dict=True
    )


def _fitted_ridge(bd, stat):
    """Return the fitted `Ridge` a model bootstrap needs, or raise.

    Args:
        bd (BrainData): The object being resampled.
        stat (str): The requested model statistic, for the error message.

    Returns:
        Ridge: The fitted estimator.

    Raises:
        ValueError: If nothing is fitted, or the fit is not a `Ridge`.
    """
    from nltools.models import Ridge

    model = getattr(bd, "model_", None)
    if model is None or not getattr(model, "is_fitted_", False):
        raise ValueError(
            f"Must call .fit(model='ridge', X=features) before bootstrap(stat='{stat}')"
        )
    if not isinstance(model, Ridge):
        raise ValueError(
            f"bootstrap(stat='{stat}') only supports a fitted Ridge, but this "
            f"BrainData holds a fitted {type(model).__name__}."
        )
    return model


def _training_feature_spaces(model, X, stat, n_obs):
    """Align the explicit training `X` to the fitted feature-space order.

    Args:
        model (Ridge): The fitted estimator.
        X (np.ndarray | Mapping | None): Caller-supplied training features.
        stat (str): The requested model statistic, for the error message.
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
            f"bootstrap(stat='{stat}') requires the training features as X=. "
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


def convert_bootstrap_results_to_brain_data(
    bd, result, save_boots=False, return_dict=False
):
    """Wrap the arrays from a bootstrap run as `BrainData` objects.

    Args:
        bd (BrainData): Template whose mask and metadata the outputs inherit.
        result (dict): Bootstrap output with keys ``'mean'``, ``'std'``,
            ``'Z'``, ``'p'``, ``'ci_lower'``, ``'ci_upper'``, and optionally
            ``'samples'``.
        save_boots (bool): Include the ``'samples'`` key in the output.
        return_dict (bool): Always return a dict. If ``False`` and
            ``save_boots=False``, return a `BrainData` holding the mean.

    Returns:
        BrainData | dict: A `BrainData` with the bootstrap mean when
            ``return_dict=False`` and ``save_boots=False``; otherwise a dict of
            `BrainData` per statistic, where the optional ``'samples'`` entry is
            a raw ndarray.
    """
    if save_boots:
        # Return dict with samples
        out = {}
        for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
            if key in result:
                # Reshape 1D arrays to 2D (1, n_voxels) for BrainData
                data_2d = (
                    result[key] if result[key].ndim == 2 else result[key].reshape(1, -1)
                )
                out[key] = _result_from_array(bd, data_2d, rows="clear")
        if "samples" in result:
            out["samples"] = result["samples"]
        return out
    if return_dict:
        # Return dict format (for model stats)
        out = {}
        for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
            if key in result:
                out[key] = _result_from_array(bd, result[key], rows="clear")
        return out
    # Return BrainData with mean (for simple stats)
    # Reshape 1D arrays to 2D (1, n_voxels) for BrainData
    mean_2d = (
        result["mean"] if result["mean"].ndim == 2 else result["mean"].reshape(1, -1)
    )
    boot_mean = _result_from_array(bd, mean_2d, rows="clear")
    return boot_mean
