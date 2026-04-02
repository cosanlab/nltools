"""Bootstrap functions extracted from BrainData methods."""

import numpy as np


def bootstrap(
    bd,
    stat,
    n_samples=5000,
    save_boots=False,
    n_jobs=-1,
    random_state=None,
    percentiles=(2.5, 97.5),
    X_test=None,
    **kwargs,
):
    """Bootstrap statistics using efficient online algorithms.

    Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
    Supports simple aggregation statistics and fitted model statistics (Ridge).

    Args:
        bd: BrainData instance.
        stat: (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std',
            'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model,
            'predict' requires fitted Ridge model + X_test).
        n_samples: (int) Number of bootstrap iterations. Default: 5000
        save_boots: (bool) If True, store all bootstrap samples (memory intensive).
                   Default: False
        n_jobs: (int) Number of CPU cores for parallelization. -1 means all CPUs.
        random_state: (int, optional) Random seed for reproducibility
        percentiles: (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5)
        X_test: (np.ndarray, optional) Test features for 'predict' bootstrap.
               Required if stat='predict'
        backend: (str, optional) Backend for computation ('numpy', 'torch', 'auto').
                If 'torch' and GPU available, uses optimized GPU acceleration with
                inline Ridge computation (no CPU round-trips). Default: None (CPU).
        max_gpu_memory_gb: (float) Maximum GPU memory to use in GB. Default: 4.0
        **kwargs: Additional parameters (currently unused, reserved for future extensions)

    Returns:
        BrainData or dict:
            - For simple stats: Returns BrainData with bootstrap mean
            - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',
              'ci_lower', 'ci_upper' (all BrainData objects)
            - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

    Examples:
        >>> # Simple aggregation
        >>> boot = brain.bootstrap(stat='mean', n_samples=1000)
        >>> assert isinstance(boot, BrainData)

        >>> # Ridge weights bootstrap (CPU)
        >>> brain.fit(X=dm, model='ridge', alpha=1.0)
        >>> boot = brain.bootstrap(stat='weights', n_samples=1000)
        >>> assert 'mean' in boot
        >>> assert isinstance(boot['mean'], BrainData)

        >>> # Ridge weights bootstrap (GPU accelerated)
        >>> brain.fit(X=dm, model='ridge', alpha=1.0)
        >>> boot = brain.bootstrap(stat='weights', n_samples=1000, backend='torch')
        >>> assert 'mean' in boot
        >>> assert isinstance(boot['mean'], BrainData)

        >>> # Ridge predict bootstrap
        >>> brain.fit(X=dm, model='ridge', alpha=1.0)
        >>> boot = brain.bootstrap(stat='predict', X_test=X_new, n_samples=1000)
        >>> assert 'mean' in boot
        >>> assert isinstance(boot['mean'], BrainData)

    Note:
        This method replaces the deprecated `summarize_bootstrap()` function from
        `nltools.stats`. To reproduce `summarize_bootstrap()` functionality:

        **Old API (deprecated):**
        >>> from nltools.stats import summarize_bootstrap
        >>> bootstrap_samples = BrainData(list_of_samples)  # Multiple samples
        >>> result = summarize_bootstrap(bootstrap_samples, save_weights=False)
        >>> # Returns: {'mean': BrainData, 'Z': BrainData, 'p': BrainData}

        **New API (recommended):**
        >>> # Option 1: Use BrainData.bootstrap() for generating bootstrap samples
        >>> boot = brain.bootstrap(stat='mean', n_samples=1000, save_boots=False)
        >>> # Returns BrainData with bootstrap mean
        >>> # To get Z and p, use stat='weights' or 'predict' which returns dict

        >>> # Option 2: For existing bootstrap samples (BrainData with multiple images),
        >>> # use OnlineBootstrapStats directly:
        >>> from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
        >>> stats = OnlineBootstrapStats(shape=(brain.shape[1],), save_samples=False)
        >>> for sample in bootstrap_samples:  # Iterate over samples
        ...     stats.update(sample.data)
        >>> result = stats.get_results()
        >>> # Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
        >>> #           'ci_lower': array, 'ci_upper': array}
        >>> # Convert to BrainData if needed:
        >>> mean_brain = brain._shallow_copy_with_data()
        >>> mean_brain.data = result['mean']
    """
    from nltools.algorithms.inference.bootstrap import (
        _bootstrap_simple_cpu_parallel,
        _bootstrap_ridge_weights_cpu_parallel,
        _bootstrap_ridge_predict_cpu_parallel,
        _bootstrap_ridge_weights_gpu_batched,
        _bootstrap_ridge_predict_gpu_batched,
    )
    from nltools.data import DesignMatrix
    from nltools.backends import Backend, check_gpu_available, auto_select_backend

    # Extract backend parameter from kwargs
    backend = kwargs.pop("backend", None)
    max_gpu_memory_gb = kwargs.pop("max_gpu_memory_gb", 4.0)

    # Determine if we should use GPU
    use_gpu = False
    if backend == "torch" or backend == "auto":
        if check_gpu_available()[0]:
            use_gpu = True
            if backend == "auto":
                backend = auto_select_backend(bd.data.shape[0], bd.data.shape[1])
            else:
                backend = Backend("torch")
        elif backend == "torch":
            raise ValueError(
                "GPU backend requested but GPU not available. "
                "Use backend=None or backend='auto' for CPU fallback."
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
        )

        # Convert result to BrainData format
        return convert_bootstrap_results_to_brain_data(
            bd, result, save_boots=save_boots, return_dict=False
        )

    elif stat in FITTED_STATS:
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
                    **kwargs,
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
                    **kwargs,
                )

            # Convert results to BrainData format
            return convert_bootstrap_results_to_brain_data(
                bd, result, save_boots=save_boots, return_dict=True
            )

        elif stat == "predict":
            # Ridge predict bootstrap
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
                    **kwargs,
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
                    **kwargs,
                )

            # Convert results to BrainData format
            return convert_bootstrap_results_to_brain_data(
                bd, result, save_boots=save_boots, return_dict=True
            )

    else:
        # Invalid stat
        raise ValueError(
            f"Unsupported stat '{stat}'. "
            f"Supported simple stats: {SIMPLE_STATS}. "
            f"Supported fitted model stats: {FITTED_STATS}. "
            f"For fitted stats, you must call .fit() first."
        )


def convert_bootstrap_results_to_brain_data(
    bd, result, save_boots=False, return_dict=False
):
    """Convert bootstrap results dictionary to BrainData format.

    Helper method to convert numpy arrays from bootstrap functions into
    BrainData objects or dicts of BrainData objects.

    Args:
        bd: BrainData instance.
        result: (dict) Result dictionary from bootstrap function with keys:
                'mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', and optionally 'samples'
        save_boots: (bool) If True, include 'samples' key in output
        return_dict: (bool) If True, always return dict even for simple stats.
                    If False, return BrainData for simple stats (when save_boots=False)

    Returns:
        BrainData or dict:
            - If return_dict=False and save_boots=False: Returns BrainData with mean
            - Otherwise: Returns dict with BrainData objects for each statistic
    """
    if save_boots:
        # Return dict with samples
        out = {}
        for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
            if key in result:
                out[key] = bd._shallow_copy_with_data()
                # Reshape 1D arrays to 2D (1, n_voxels) for BrainData
                data_2d = (
                    result[key] if result[key].ndim == 2 else result[key].reshape(1, -1)
                )
                out[key].data = data_2d
        if "samples" in result:
            out["samples"] = result["samples"]
        return out
    elif return_dict:
        # Return dict format (for model stats)
        out = {}
        for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
            if key in result:
                out[key] = bd._shallow_copy_with_data()
                out[key].data = result[key]
        return out
    else:
        # Return BrainData with mean (for simple stats)
        boot_mean = bd._shallow_copy_with_data()
        # Reshape 1D arrays to 2D (1, n_voxels) for BrainData
        mean_2d = (
            result["mean"]
            if result["mean"].ndim == 2
            else result["mean"].reshape(1, -1)
        )
        boot_mean.data = mean_2d
        return boot_mean
