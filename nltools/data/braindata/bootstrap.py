"""Bootstrap resampling for `BrainData` — aggregate statistics and Ridge model statistics.

`BrainData.bootstrap` delegates here.
"""

import numpy as np

from .utils import _copy_without_fit_state


def bootstrap(
    bd,
    stat,
    *,
    n_samples=5000,
    save_boots=False,
    percentiles=(2.5, 97.5),
    X_test=None,
    device="cpu",
    max_gpu_memory_gb=None,
    tail=2,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
):
    """Bootstrap statistics with CPU parallelization or GPU acceleration.

    Supports simple aggregation statistics and fitted model statistics (Ridge).
    Note: the CPU path pre-generates all resample indices and collects every
    per-sample result, so peak memory grows with ``n_samples`` (it is not a
    streaming/online accumulator).

    Args:
        bd (BrainData): Data to resample.
        stat (str): Statistic to bootstrap. Simple aggregates: ``'mean'``,
            ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'``. Model
            statistics (require a fitted Ridge model): ``'weights'``, or
            ``'predict'`` (also requires ``X_test``).
        n_samples (int): Number of bootstrap iterations. Default ``5000``.
        save_boots (bool): Keep every bootstrap sample (memory intensive).
            Default ``False``.
        percentiles (tuple[float, float]): Percentiles for the confidence
            interval. Default ``(2.5, 97.5)``.
        X_test (np.ndarray | None): Test features for ``stat='predict'``.
        device (str): Compute device for Ridge bootstraps: ``'cpu'`` (default),
            ``'gpu'`` (PyTorch on CUDA/MPS; raises if none is available), or
            ``'auto'`` (a GPU if present, else CPU). Ignored for simple stats.
        max_gpu_memory_gb (float | None): Explicit GPU memory budget in GB when
            ``device`` is ``'gpu'`` or ``'auto'``. ``None`` (default) measures
            the device.
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

    Examples:
        ```python
        # Simple aggregation: returns a BrainData holding the bootstrap mean
        boot = brain.bootstrap(stat='mean', n_samples=1000)
        assert isinstance(boot, BrainData)

        # Ridge weights bootstrap (CPU): returns a dict of BrainData
        brain.fit(X=dm, model='ridge', alpha=1.0)
        boot = brain.bootstrap(stat='weights', n_samples=1000)
        assert isinstance(boot['mean'], BrainData)

        # Ridge weights bootstrap (GPU accelerated)
        boot = brain.bootstrap(stat='weights', n_samples=1000, device='gpu')

        # Ridge predict bootstrap
        boot = brain.bootstrap(stat='predict', X_test=X_new, n_samples=1000)

        # Summarize pre-existing bootstrap samples (a BrainData with one image per
        # sample) with OnlineBootstrapStats instead
        from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats

        stats = OnlineBootstrapStats(shape=(brain.shape[1],), save_samples=False)
        for sample in bootstrap_samples:
            stats.update(sample.data)
        result = stats.get_results()  # keys: mean, std, Z, p, ci_lower, ci_upper
        mean_brain = _copy_without_fit_state(brain, copy_data=False)
        mean_brain.data = result['mean']
        ```

    Note:
        Use ``stat='mean'`` to bootstrap an aggregate; use ``stat='weights'`` or
        ``stat='predict'`` to also get Z and p maps. To summarize bootstrap
        samples you already have, feed them to `OnlineBootstrapStats` directly
        (see Examples).
    """
    from nltools.algorithms.inference.bootstrap import (
        _bootstrap_simple_cpu_parallel,
        _bootstrap_ridge_weights_cpu_parallel,
        _bootstrap_ridge_predict_cpu_parallel,
        _bootstrap_ridge_weights_gpu_batched,
        _bootstrap_ridge_predict_gpu_batched,
    )
    from nltools.data import DesignMatrix
    from nltools.algorithms.backends import (
        check_gpu_available,
        auto_select_backend,
        resolve_backend,
    )

    # Determine if we should use GPU. `device='gpu'` demands a real GPU;
    # `device='auto'` uses one when present and silently falls back to CPU.
    # The resolved `Backend` instance is threaded to the algorithm-layer GPU
    # helpers (which keep the internal `backend=` name).
    use_gpu = False
    backend = None
    if device in ("gpu", "auto"):
        if check_gpu_available()[0]:
            use_gpu = True
            if device == "auto":
                backend = auto_select_backend(bd.data.shape[0], bd.data.shape[1])
            else:
                backend = resolve_backend("gpu")
        elif device == "gpu":
            raise ValueError(
                "GPU requested via device='gpu' but no GPU is available. "
                "Use device='cpu' or device='auto' for CPU fallback."
            )

    # Get data as numpy array
    data = bd.data  # Shape: (n_samples, n_voxels)

    # Route to appropriate bootstrap function
    SIMPLE_STATS = ["mean", "median", "std", "sum", "min", "max"]
    FITTED_STATS = ["weights", "predict"]

    if stat in SIMPLE_STATS:
        # Simple aggregation bootstrap
        result = _bootstrap_simple_cpu_parallel(
            data,
            method=stat,
            n_samples=n_samples,
            save_boots=save_boots,
            n_jobs=n_jobs,
            random_state=random_state,
            percentiles=percentiles,
            tail=tail,
            progress_bar=progress_bar,
        )

        # Convert result to BrainData format
        return convert_bootstrap_results_to_brain_data(
            bd, result, save_boots=save_boots, return_dict=False
        )

    if stat not in FITTED_STATS:
        raise ValueError(
            f"Unsupported stat '{stat}'. "
            f"Supported simple stats: {SIMPLE_STATS}. "
            f"Supported fitted model stats: {FITTED_STATS}. "
            f"For fitted stats, you must call .fit() first."
        )

    # Check if model is fitted
    if not hasattr(bd, "model_") or bd.model_ is None:
        raise ValueError(
            f"Must call .fit(model='ridge', X=design_matrix) before bootstrap(stat='{stat}')"
        )

    # Check if Ridge model
    if not hasattr(bd.model_, "coef_") or not hasattr(bd.model_, "alpha_"):
        raise ValueError(
            f"Bootstrap stat='{stat}' only supports Ridge models. "
            f"Got model type: {type(bd.model_)}"
        )

    # Get design matrix from stored X_
    if not hasattr(bd, "X_") or bd.X_ is None:
        raise ValueError(
            "Design matrix not found. Must call .fit(model='ridge', X=design_matrix) "
            "with X parameter."
        )

    # Convert DesignMatrix to numpy if needed
    if isinstance(bd.X_, DesignMatrix):
        X = bd.X_.to_numpy()
    else:
        X = np.asarray(bd.X_)

    # Get alpha from model
    alpha = bd.model_.alpha_ if hasattr(bd.model_, "alpha_") else bd.model_.alpha

    if stat == "weights":
        # Ridge weights bootstrap
        if use_gpu:
            result = _bootstrap_ridge_weights_gpu_batched(
                X,
                data,
                alpha=alpha,
                n_samples=n_samples,
                save_boots=save_boots,
                backend=backend,
                max_gpu_memory_gb=max_gpu_memory_gb,
                random_state=random_state,
                percentiles=percentiles,
                tail=tail,
                progress_bar=progress_bar,
            )
        else:
            result = _bootstrap_ridge_weights_cpu_parallel(
                X,
                data,
                alpha=alpha,
                n_samples=n_samples,
                save_boots=save_boots,
                n_jobs=n_jobs,
                random_state=random_state,
                percentiles=percentiles,
                tail=tail,
                progress_bar=progress_bar,
            )

        return convert_bootstrap_results_to_brain_data(
            bd, result, save_boots=save_boots, return_dict=True
        )

    # stat == "predict"
    if X_test is None:
        raise ValueError(
            "X_test parameter required for bootstrap(stat='predict'). "
            "Provide test features: bootstrap(stat='predict', X_test=...)"
        )

    X_test = np.asarray(X_test)

    if use_gpu:
        result = _bootstrap_ridge_predict_gpu_batched(
            X,
            data,
            X_test,
            alpha=alpha,
            n_samples=n_samples,
            save_boots=save_boots,
            backend=backend,
            max_gpu_memory_gb=max_gpu_memory_gb,
            random_state=random_state,
            percentiles=percentiles,
            tail=tail,
            progress_bar=progress_bar,
        )
    else:
        result = _bootstrap_ridge_predict_cpu_parallel(
            X,
            data,
            X_test,
            alpha=alpha,
            n_samples=n_samples,
            save_boots=save_boots,
            n_jobs=n_jobs,
            random_state=random_state,
            percentiles=percentiles,
            tail=tail,
            progress_bar=progress_bar,
        )

    return convert_bootstrap_results_to_brain_data(
        bd, result, save_boots=save_boots, return_dict=True
    )


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
                out[key] = _copy_without_fit_state(bd, copy_data=False)
                # Reshape 1D arrays to 2D (1, n_voxels) for BrainData
                data_2d = (
                    result[key] if result[key].ndim == 2 else result[key].reshape(1, -1)
                )
                out[key].data = data_2d
        if "samples" in result:
            out["samples"] = result["samples"]
        return out
    if return_dict:
        # Return dict format (for model stats)
        out = {}
        for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
            if key in result:
                out[key] = _copy_without_fit_state(bd, copy_data=False)
                out[key].data = result[key]
        return out
    # Return BrainData with mean (for simple stats)
    boot_mean = _copy_without_fit_state(bd, copy_data=False)
    # Reshape 1D arrays to 2D (1, n_voxels) for BrainData
    mean_2d = (
        result["mean"] if result["mean"].ndim == 2 else result["mean"].reshape(1, -1)
    )
    boot_mean.data = mean_2d
    return boot_mean
