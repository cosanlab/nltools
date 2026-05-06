"""
BrainData modeling functions.

Standalone functions extracted from BrainData class methods for model fitting,
cross-validation, GLM estimation, Ridge regression, and contrast computation.
"""

import numpy as np

from .utils import shallow_copy


def cv(
    bd,
    k=None,
    method="kfold",
    split_by=None,
    groups=None,
    n=1000,
    random_state=None,
):
    """Create a cross-validation pipeline for this BrainData.

    Returns a Pipeline object that enables fluent, chainable transforms
    with cross-validation. Terminal methods like .predict() execute the
    pipeline and return results.

    Args:
        bd: BrainData instance.
        k: Number of folds (for kfold method). Defaults to 5.
        method: CV scheme type. Options:
            - 'kfold': k-fold cross-validation (default)
            - 'loro': leave-one-run-out (requires split_by='runs' or groups)
            - 'bootstrap': bootstrap with out-of-bag test sets
        split_by: Attribute name for group splits (e.g., 'runs').
            If provided and groups is None, will try to get groups from
            bd.X[split_by] if bd.X is a DataFrame.
        groups: Explicit group labels for CV splits.
        n: Number of iterations for bootstrap/permutation methods. Default 1000.
        random_state: Random seed for reproducibility.

    Returns:
        BrainDataPipeline: A pipeline object for method chaining.

    Examples:
        >>> # Simple 5-fold CV with prediction
        >>> result = brain.cv(k=5).predict(y, algorithm='ridge')
        >>> print(f"Mean score: {result.mean_score:.3f}")

        >>> # With preprocessing
        >>> result = (brain
        ...     .cv(k=5)
        ...     .normalize()
        ...     .reduce(n_components=50)
        ...     .predict(y))

        >>> # Leave-one-run-out CV
        >>> result = brain.cv(method='loro', groups=run_labels).predict(y)

    See Also:
        BrainDataPipeline: For available transforms and terminal methods.
        CVScheme: For CV scheme configuration details.
    """
    from nltools.pipelines.cv import CVScheme
    from nltools.data.braindata.pipeline import BrainDataPipeline

    # Handle split_by -> groups conversion
    if groups is None and split_by is not None:
        if hasattr(bd, "X") and bd.X is not None:
            if hasattr(bd.X, "__getitem__") and split_by in bd.X:
                groups = np.array(bd.X[split_by])

    # Create CV scheme (CVScheme retains internal `scheme=` parameter name)
    cv_scheme = CVScheme(
        k=k,
        scheme=method,
        split_by=split_by,
        n=n,
        random_state=random_state,
    )

    # Create and return pipeline
    return BrainDataPipeline(bd, cv=cv_scheme, groups=groups)


def fit(
    bd,
    model="glm",
    *,
    X=None,
    cv=None,
    local_alpha=True,
    fit_intercept=False,
    inplace=True,
    progress_bar=None,
    scale=True,
    scale_value=100.0,
    design_clean=True,
    design_clean_thresh=0.95,
    design_clean_exclude_confounds=False,
    design_clean_fill_na=0,
    **kwargs,
):
    """Fit a model to brain imaging data.

    Creates and fits a model from string specification. The brain data
    (bd.data) is always used as the target variable. Model and results
    are stored for later use with predict().

    Args:
        bd: BrainData instance.
        model (str): Model type: 'ridge', 'glm', or future model names
        X (array-like or DataFrame): Design matrix or feature matrix, shape (n_samples, n_features)
            - For GLM: Design matrix with regressors (n_samples must match bd.data)
            - For Ridge: Feature matrix for prediction (n_samples must match bd.data)
        cv (int, 'auto', or sklearn CV splitter, optional): Cross-validation specification (Ridge only):
            - int: Number of folds for k-fold CV (returns CV scores)
            - 'auto': Triggers alpha selection via CV (implies alpha='auto')
            - sklearn CV object: Custom CV splitter (e.g., KFold(3, shuffle=True))
            - None: No CV (default, backward compatible)
        inplace (bool, default=True): If True, mutate bd and return bd (backward compatible).
            If False, return Fit dataclass with results (bd unchanged).
        progress_bar (bool, optional): Display progress bar during fitting.
            - If None: Uses bd.verbose (default)
            - If True: Shows progress bar for long-running operations
            - If False: No progress bar
        scale (bool, default=True): Apply grand-mean scaling before fitting. Calls
            bd.scale(scale_value) which divides all values by the global mean
            and multiplies by scale_value. This puts data in percent signal change
            units, which is standard for fMRI analysis.
        scale_value (float, default=100.0): Target value for mean after scaling.
            Only used if scale=True.
        design_clean (bool, default=True): GLM only. If True, run
            ``DesignMatrix.clean()`` on ``X`` before fitting to drop highly
            correlated regressors. Coerces ``X`` to ``DesignMatrix`` if needed.
            Ignored when ``model='ridge'``.
        design_clean_thresh (float, default=0.95): GLM only. Correlation
            threshold passed to ``DesignMatrix.clean()`` (drops if
            ``abs(r) >= thresh``). Ignored when ``model='ridge'``.
        design_clean_exclude_confounds (bool, default=False): GLM only. If
            True, ``DesignMatrix.clean()`` skips confound columns when
            checking correlations. Ignored when ``model='ridge'``.
        design_clean_fill_na (int, float, or None, default=0): GLM only.
            Fill value for NaNs before correlation check in
            ``DesignMatrix.clean()``. Ignored when ``model='ridge'``.
        **kwargs (dict): Additional arguments passed to model constructor
            - Ridge: alpha, alphas, backend, random_state
            - Glm: noise_model, minimize_memory, etc.

    Returns:
        BrainData or Fit: If ``inplace=True``, returns bd (fitted BrainData).
            If ``inplace=False``, returns Fit dataclass with results.

    Attributes:
        The following are set on bd when ``inplace=True``:

        ``model_`` (BaseModel): Fitted model instance (Ridge, Glm, etc.)
        ``X_`` (ndarray): Training data X, stored for predict() default
        ``cv_results_`` (dict, optional): Cross-validation results dict with keys 'scores',
        'mean_score', 'predictions', 'folds', 'best_alpha', 'alpha_scores' (if cv is not None).
        glm_betas (BrainData, optional): Beta coefficients (for model='glm')
        glm_t (BrainData, optional): T-statistics (for model='glm')
        glm_p (BrainData, optional): P-values (for model='glm')
        glm_se (BrainData, optional): Standard errors (for model='glm')
        glm_residual (BrainData, optional): Residuals (for model='glm')
        glm_predicted (BrainData, optional): Fitted values (for model='glm')
        glm_r2 (BrainData, optional): R-squared values (for model='glm')
        ridge_weights (BrainData, optional): Model coefficients (for model='ridge')
        ridge_fitted_values (BrainData, optional): Fitted values (for model='ridge')
        ridge_scores (BrainData, optional): R-squared scores (for model='ridge')

    Examples:
        >>> # Old behavior (backward compatible): mutate self
        >>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
        >>> print(f"CV R2: {brain_data.cv_results_['mean_score'].mean():.3f}")
        >>> weights = brain_data.ridge_weights  # Access as attribute
        >>>
        >>> # New behavior: return Fit dataclass (self unchanged)
        >>> fit = brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features, inplace=False)
        >>> assert isinstance(fit, Fit)
        >>> assert 'weights' in fit.available()
        >>> assert not hasattr(brain_data, 'ridge_weights')  # brain_data unchanged
        >>> print(f"CV R2: {fit.cv_mean_score.mean():.3f}")
        >>>
        >>> # GLM with Fit dataclass
        >>> fit_glm = brain_data.fit(model='glm', X=design_matrix, inplace=False)
        >>> assert 'betas' in fit_glm.available()
        >>> assert 't_stats' in fit_glm.available()
    """
    from nltools.models import Ridge, Glm

    # Validate inputs
    if model not in ["glm", "ridge"]:
        raise TypeError("supported models are 'glm' (default) and 'ridge'")
    if X is None:
        raise TypeError("X must be provided")

    # For GLM: preserve DataFrame/DesignMatrix (don't convert to numpy)
    # For Ridge: convert to numpy array for sklearn compatibility
    if model == "glm":
        X_model = X  # Keep as-is (DataFrame or DesignMatrix)
        # Validate shape using underlying array
        X_array = np.asarray(X)
        if X_array.shape[0] != bd.shape[0]:
            raise ValueError(
                f"X has {X_array.shape[0]} samples, but brain data has {bd.shape[0]} samples. "
                f"number of samples must match."
            )
        if design_clean:
            from nltools.data.designmatrix import DesignMatrix

            if not isinstance(X_model, DesignMatrix):
                X_model = DesignMatrix(X_model)
            X_model = X_model.clean(
                fill_na=design_clean_fill_na,
                exclude_confounds=design_clean_exclude_confounds,
                thresh=design_clean_thresh,
                verbose=bd.verbose,
            )
    else:
        # Ridge: handle list (banded ridge) or array (regular ridge)
        if isinstance(X, list):
            # Banded ridge: keep as list, validate each element
            X_model = X
            for i, Xi in enumerate(X):
                Xi_array = np.asarray(Xi)
                if Xi_array.shape[0] != bd.shape[0]:
                    raise ValueError(
                        f"X[{i}] has {Xi_array.shape[0]} samples, but brain data has {bd.shape[0]} samples. "
                        f"number of samples must match."
                    )
        else:
            # Regular ridge: convert to numpy
            X_model = np.asarray(X)
            if X_model.shape[0] != bd.shape[0]:
                raise ValueError(
                    f"X has {X_model.shape[0]} samples, but brain data has {bd.shape[0]} samples. "
                    f"number of samples must match."
                )

    # Always store model_ and X_ for predict() to work (even if inplace=False)
    bd.X_ = X_model

    # Determine progress_bar setting (default to bd.verbose if not specified)
    if progress_bar is None:
        progress_bar = bd.verbose

    # Create temporary copy if inplace=False to avoid mutating result attributes
    if inplace:
        target = bd
    else:
        # Create temporary copy for fitting (to avoid mutating bd's result attributes)
        target = bd.copy()
        # Set X_ on copy (will be set below)
        target.X_ = X_model
        # Clean up any existing result attributes from the copy
        for attr in [
            "ridge_weights",
            "ridge_fitted_values",
            "ridge_scores",
            "glm_betas",
            "glm_t",
            "glm_p",
            "glm_se",
            "glm_residual",
            "glm_predicted",
            "glm_r2",
            "cv_results_",
        ]:
            if hasattr(target, attr):
                delattr(target, attr)

    # Apply scaling before fitting (puts data in percent signal change units)
    if scale:
        scaled = target.scale(scale_value)
        target.data = scaled.data

    # Create model based on string
    if model == "ridge":
        # Forward progress_bar to Ridge model's progress_bar kwarg
        ridge_kwargs = kwargs.copy()
        if "progress_bar" not in ridge_kwargs:
            ridge_kwargs["progress_bar"] = progress_bar
        # Explicit-signature kwargs win over **kwargs forwarding so calls
        # like bd.fit(model='ridge', local_alpha=False, ...) reach Ridge.
        ridge_kwargs.setdefault("local_alpha", local_alpha)
        ridge_kwargs.setdefault("fit_intercept", fit_intercept)
        target.model_ = Ridge(**ridge_kwargs)
        fit_ridge(target, X_model, cv=cv, **kwargs)
    elif model == "glm":
        if cv is not None:
            raise NotImplementedError(
                "Cross-validation not yet supported for GLM models"
            )
        # Pass mask from BrainData to Glm to prevent resampling during GLM estimation
        # The mask must match the one used to mask the data initially
        glm_kwargs = kwargs.copy()
        if "mask" not in glm_kwargs:
            glm_kwargs["mask"] = target.mask
        # Forward progress_bar to GLM's progress_bar kwarg
        if "progress_bar" not in glm_kwargs:
            glm_kwargs["progress_bar"] = progress_bar
        target.model_ = Glm(**glm_kwargs)
        fit_glm(target, X_model)

    # If inplace=False, copy model_ to bd (needed for predict()) and return Fit
    if not inplace:
        # Store model_ and X_ on bd for predict() to work
        bd.model_ = target.model_
        # Also store design_matrix for GLM compute_contrasts()
        if model == "glm" and hasattr(target, "design_matrix"):
            bd.design_matrix = target.design_matrix
        # Return Fit dataclass with results
        return to_fit_dataclass(target, model=model)
    return bd


def fit_ridge(bd, X, cv=None, **kwargs):
    """Fit Ridge model and extract results.

    Args:
        bd: BrainData instance.
        X (ndarray): Training features
        cv (int, 'auto', or sklearn CV splitter, optional): Cross-validation specification
        **kwargs (dict): Additional arguments for CV (alpha, alphas, backend, etc.)

    Note:
        Sets ridge_weights, ridge_fitted_values, ridge_scores, and
        cv_results_ (if cv provided) on bd.
    """
    alpha = bd.model_.alpha if hasattr(bd.model_, "alpha") else None

    if cv is not None and alpha == "auto":
        # Delegate per-voxel α selection + full-data refit entirely to the
        # model layer (which calls solve_ridge_cv). The BrainData layer
        # only assembles cv_results_ from the model's attributes plus a
        # held-out-prediction pass.
        bd.model_.cv = _normalize_cv(cv)
        bd.model_.fit(X, bd.data)
        bd.cv_results_ = _assemble_ridge_cv_results(bd, X, cv)
    elif cv is not None:
        # Fixed-α + CV evaluation: alpha is set, we just want held-out
        # scores under it. compute_ridge_cv handles this branch.
        bd.cv_results_ = compute_ridge_cv(bd, X, cv, **kwargs)
        bd.model_.fit(X, bd.data)
    else:
        bd.model_.fit(X, bd.data)

    _populate_ridge_attributes(bd, X)


def _normalize_cv(cv):
    """Reject single-use generators and bad cv values; pass through ints and
    splitter objects.

    BrainData's CV path needs a re-iterable splitter — alpha selection
    iterates folds once for scoring, then ``cross_val_predict_ridge``
    iterates them again for held-out predictions.
    """
    is_splitter = hasattr(cv, "split") and hasattr(cv, "get_n_splits")
    if hasattr(cv, "__next__") and not is_splitter:
        raise TypeError(
            "Got a generator for `cv` (e.g. `splitter.split(X, ...)`). "
            "Pass an sklearn CV splitter object instead — "
            "KFold(5, shuffle=True), GroupKFold(8), etc. — so the "
            "BrainData layer can iterate it more than once."
        )
    if not isinstance(cv, int) and not is_splitter:
        raise ValueError(f"cv must be an int or sklearn CV splitter object; got {cv!r}")
    if isinstance(cv, int) and cv < 2:
        # Defer KFold's own message verbatim ("k-fold cross-validation
        # requires at least one train/test split") — we just trip it
        # eagerly so the caller doesn't get an opaque error mid-CV.
        from sklearn.model_selection import KFold

        KFold(n_splits=cv)  # raises ValueError
    return cv


def _populate_ridge_attributes(bd, X):
    """Set ridge_weights / ridge_fitted_values / ridge_scores from bd.model_."""
    # Ridge.coef_ is (n_features, n_voxels); no transpose.
    bd.ridge_weights = shallow_copy(bd)
    bd.ridge_weights.data = bd.model_.coef_

    fitted = bd.model_.predict(X)
    bd.ridge_fitted_values = shallow_copy(bd)
    bd.ridge_fitted_values.data = fitted

    scores = bd.model_.score(X, bd.data)  # (n_voxels,)
    bd.ridge_scores = shallow_copy(bd)
    bd.ridge_scores.data = scores.reshape(1, -1)


def _assemble_ridge_cv_results(bd, X, cv):
    """Build cv_results_ dict from the fitted Ridge model + held-out preds.

    Pure assembly — no math beyond picking per-voxel best-α scores out of
    the model's (n_splits, n_alphas, n_voxels) cube and calling
    ``cross_val_predict_ridge`` to get held-out predictions under the
    selected per-voxel α. This is the contract that ``BrainData.fit(
    model='ridge', alpha='auto', cv=K)`` produces:

        - 'best_alpha':   (n_voxels,) per-voxel selected α (or scalar
                          when ``local_alpha=False``).
        - 'alpha_scores': (n_splits, n_alphas, n_voxels) raw α-grid CV
                          scores from the solver.
        - 'scores':       (n_splits, n_voxels) per-fold R² *at the
                          selected α* — extracted by indexing into
                          ``alpha_scores`` per voxel.
        - 'mean_score':   (n_voxels,) mean of ``scores`` across folds.
        - 'predictions':  BrainData of held-out predictions on the
                          original BOLD scale (uses per-voxel α).
        - 'folds':        (n_samples,) fold index per sample.
    """
    from nltools.algorithms.ridge import cross_val_predict_ridge

    cv_splitter = _normalize_cv(cv) if not isinstance(cv, int) else cv

    alpha_scores = bd.model_.cv_scores_  # (n_splits, n_alphas, n_voxels)
    n_splits, n_alphas, n_voxels = alpha_scores.shape

    # Per-voxel selected α (already on the model). May be scalar when the
    # model squeezed a single-target case, but for multi-voxel BrainData
    # we'll always have a (n_voxels,) array — broadcast just in case.
    best_alpha = bd.model_.alpha_
    if not isinstance(best_alpha, np.ndarray):
        best_alpha_arr = np.full(n_voxels, float(best_alpha))
    else:
        best_alpha_arr = best_alpha

    # Per-voxel best-α index → per-fold scores at that α.
    # alpha_scores has the candidate alphas in the order solve_ridge_cv saw
    # them (i.e., the model's `alphas` attr). Recover that order to do the
    # lookup.
    alpha_grid = np.asarray(bd.model_.alphas)
    best_idx = np.searchsorted(alpha_grid, best_alpha_arr)
    # Guard against floating-point ties: clip and verify.
    best_idx = np.clip(best_idx, 0, n_alphas - 1)

    # scores[s, v] = alpha_scores[s, best_idx[v], v]
    fold_arange = np.arange(n_splits)[:, None]
    voxel_arange = np.arange(n_voxels)[None, :]
    scores = alpha_scores[fold_arange, best_idx[None, :], voxel_arange]
    mean_score = scores.mean(axis=0)

    # Held-out predictions under per-voxel α (delegates to the same
    # backend-aware refit pipeline solve_ridge_cv uses).
    fit_intercept = bool(getattr(bd.model_, "fit_intercept", False))
    parallel = (
        "gpu"
        if getattr(bd.model_.backend_, "device", None) in ("cuda", "mps")
        else "cpu"
    )
    pred_result = cross_val_predict_ridge(
        X,
        bd.data,
        alphas=best_alpha_arr,
        cv=cv_splitter,
        fit_intercept=fit_intercept,
        parallel=parallel,
    )

    cv_predictions_brain = shallow_copy(bd)
    cv_predictions_brain.data = pred_result["predictions"]

    return {
        "best_alpha": best_alpha_arr
        if isinstance(best_alpha, np.ndarray)
        else best_alpha,
        "alpha_scores": alpha_scores,
        "scores": scores,
        "mean_score": mean_score,
        "predictions": cv_predictions_brain,
        "folds": pred_result["folds"],
    }


def compute_ridge_cv(bd, X, cv, alpha=None, backend="auto", **kwargs):
    """Held-out CV scores under a fixed Ridge α.

    Used only for the *fixed-α* + CV branch — alpha selection is now
    handled by ``Ridge.fit`` (which delegates to ``solve_ridge_cv``) and
    assembled into ``cv_results_`` by ``_assemble_ridge_cv_results``.

    Args:
        bd: BrainData instance.
        X (ndarray): Training features, shape (n_samples, n_features).
        cv (int or sklearn CV splitter): Cross-validation specification.
        alpha (float, optional): Fixed regularization strength. If None,
            extracted from ``bd.model_.alpha``.
        backend (str): Computational backend ('numpy', 'torch', 'auto'). Default: 'auto'
        **kwargs: Additional kwargs (forward-compatibility).

    Returns:
        dict: ``{"scores", "mean_score", "predictions", "folds"}``.
    """
    from nltools.algorithms.ridge import cross_val_predict_ridge

    cv_splitter = _normalize_cv(cv)

    if isinstance(X, list):
        raise ValueError(
            "Cross-validation for banded ridge should be handled by the model. "
            "Use alpha='auto' with cv parameter in fit()."
        )

    if alpha is None:
        alpha = bd.model_.alpha if hasattr(bd.model_, "alpha") else 1.0

    fit_intercept = bool(getattr(bd.model_, "fit_intercept", False))

    # Translate the BrainData-layer 'backend' string vocabulary to the
    # ridge layer's 'parallel' vocabulary. Kept here for compatibility
    # with callers that still pass backend='auto'/'torch'/etc.
    if backend in ("auto", "numpy", None):
        parallel = "cpu"
    elif backend == "torch":
        parallel = "gpu"
    elif backend in ("cpu", "gpu"):
        parallel = backend
    else:
        parallel = "cpu"

    n_voxels = bd.data.shape[1]
    pred_result = cross_val_predict_ridge(
        X,
        bd.data,
        alphas=np.full(n_voxels, float(alpha)),
        cv=cv_splitter,
        fit_intercept=fit_intercept,
        parallel=parallel,
    )

    cv_predictions_brain = shallow_copy(bd)
    cv_predictions_brain.data = pred_result["predictions"]

    return {
        "scores": pred_result["scores"],
        "mean_score": pred_result["scores"].mean(axis=0),
        "predictions": cv_predictions_brain,
        "folds": pred_result["folds"],
    }


def fit_glm(bd, X):
    """Fit GLM model and extract results (same logic as current regress()).

    Args:
        bd: BrainData instance.
        X: Design matrix (DataFrame or DesignMatrix).

    Note:
        Sets glm_betas, glm_t, glm_p, glm_se, glm_residual, glm_predicted,
        glm_r2, and design_matrix on bd.
    """
    from nltools.data.designmatrix import DesignMatrix
    from nltools.data import BrainData

    # Ensure X is DesignMatrix
    if not isinstance(X, DesignMatrix):
        X = DesignMatrix(X)

    # Store design matrix for compute_contrasts()
    bd.design_matrix = X

    # Convert data to 4D nifti for nilearn
    data_4d = bd.to_nifti()

    # Fit Glm model
    bd.model_.fit(data_4d, design_matrices=[X])

    # Extract results for each regressor (same as regress() lines 802-815)
    n_regressors = X.shape[1]
    beta_maps = []
    t_maps = []
    p_maps = []
    se_maps = []

    for i, col in enumerate(X.columns):
        # Create contrast vector (1 for this regressor, 0 for others)
        contrast = np.zeros(n_regressors)
        contrast[i] = 1

        # Compute contrast using Glm's compute_contrast method
        results = bd.model_.compute_contrast(contrast, output_type="all")

        # Store maps
        beta_maps.append(results["effect_size"])
        t_maps.append(results["stat"])
        p_maps.append(results["p_value"])
        se_maps.append(results["effect_variance"])

    # Convert results to BrainData objects
    bd.glm_betas = BrainData(beta_maps, mask=bd.mask)
    bd.glm_t = BrainData(t_maps, mask=bd.mask)
    bd.glm_p = BrainData(p_maps, mask=bd.mask)

    # Convert effect variance to standard error (same as regress() lines 826-831)
    se_data = []
    for se_img in se_maps:
        se_brain = BrainData(se_img, mask=bd.mask)
        se_brain.data = np.sqrt(np.abs(se_brain.data))
        se_data.append(se_brain)
    bd.glm_se = BrainData(data=se_data, mask=bd.mask)

    # Get residuals
    bd.glm_residual = BrainData(bd.model_.residuals, mask=bd.mask)

    # Predicted = original - residuals
    bd.glm_predicted = bd.copy()
    bd.glm_predicted.data = bd.data - bd.glm_residual.data

    # R-squared calculation
    ss_total = np.sum((bd.data - bd.data.mean(axis=0)) ** 2, axis=0)
    ss_residual = np.sum(bd.glm_residual.data**2, axis=0)
    r2_values = 1 - (ss_residual / (ss_total + 1e-10))

    # Create single-image BrainData for R-squared
    bd.glm_r2 = bd[0].copy()
    bd.glm_r2.data = r2_values.reshape(1, -1)


def to_fit_dataclass(bd, model):
    """Convert BrainData fit results to Fit dataclass.

    Args:
        bd: BrainData instance.
        model (str): Model type ('ridge' or 'glm')

    Returns:
        Fit: Dataclass containing fit results
    """
    from nltools.data.fitresults import Fit

    if model == "ridge":
        # Extract Ridge results
        fitted_values = bd.ridge_fitted_values.data  # (n_samples, n_voxels)
        weights = bd.ridge_weights.data  # (n_features, n_voxels)
        scores = bd.ridge_scores.data  # (1, n_voxels)
        # Squeeze first dimension to get (n_voxels,)
        if scores.ndim > 1 and scores.shape[0] == 1:
            scores = scores[0]  # (n_voxels,)
        else:
            scores = scores.squeeze()  # (n_voxels,)

        # Extract CV results if available
        cv_scores = None
        cv_mean_score = None
        cv_predictions = None
        cv_folds = None
        cv_best_alpha = None
        cv_alpha_scores = None

        if hasattr(bd, "cv_results_") and bd.cv_results_ is not None:
            cv_results = bd.cv_results_
            cv_scores = cv_results.get("scores")  # (n_folds, n_voxels)
            cv_mean_score = cv_results.get("mean_score")  # (n_voxels,)

            # Extract predictions from BrainData
            if "predictions" in cv_results:
                cv_predictions = cv_results["predictions"].data  # (n_samples, n_voxels)

            cv_folds = cv_results.get("folds")  # (n_samples,)
            cv_best_alpha = cv_results.get("best_alpha")  # float or None
            cv_alpha_scores = cv_results.get(
                "alpha_scores"
            )  # (n_folds, n_alphas, n_voxels) or None

        return Fit(
            fitted_values=fitted_values,
            weights=weights,
            scores=scores,
            cv_scores=cv_scores,
            cv_mean_score=cv_mean_score,
            cv_predictions=cv_predictions,
            cv_folds=cv_folds,
            cv_best_alpha=cv_best_alpha,
            cv_alpha_scores=cv_alpha_scores,
        )

    if model == "glm":
        # Extract GLM results
        fitted_values = bd.glm_predicted.data  # (n_samples, n_voxels)
        betas = bd.glm_betas.data  # (n_regressors, n_voxels)
        t_stats = bd.glm_t.data  # (n_regressors, n_voxels)
        p_values = bd.glm_p.data  # (n_regressors, n_voxels)
        se = bd.glm_se.data  # (n_regressors, n_voxels)
        residuals = bd.glm_residual.data  # (n_samples, n_voxels)
        r2 = bd.glm_r2.data  # (1, n_voxels)
        # Squeeze first dimension to get (n_voxels,)
        if r2.ndim > 1 and r2.shape[0] == 1:
            r2 = r2[0]  # (n_voxels,)
        else:
            r2 = r2.squeeze()  # (n_voxels,)

        return Fit(
            fitted_values=fitted_values,
            betas=betas,
            t_stats=t_stats,
            p_values=p_values,
            se=se,
            residuals=residuals,
            r2=r2,
        )

    raise AssertionError(f"unvalidated model passed to to_fit_dataclass: {model!r}")


def _signed_z_from_p(t_like_arr, p_arr):
    """Compute a signed two-tailed z-score map from a p-value map.

    ``|z| = norm.isf(p/2)`` so that p=0.05 → |z|≈1.96, matching nilearn's
    ``output_type='z_score'`` convention. The sign is copied from the
    accompanying statistic (t-stat or mean) so the returned map has both
    direction and magnitude.
    """
    from scipy.stats import norm

    p_clipped = np.clip(np.asarray(p_arr), np.finfo(float).tiny, 1.0)
    z_abs = norm.isf(p_clipped / 2.0)
    return np.sign(np.asarray(t_like_arr)) * z_abs


def ttest(
    bd,
    popmean=0.0,
    permutation=False,
    n_permute=5000,
    tail=2,
    return_null=False,
    n_jobs=-1,
    random_state=None,
):
    """One-sample voxelwise t-test across images (axis 0).

    For a BrainData stack of images (e.g. subject-level contrast maps with
    shape ``(n_samples, n_voxels)``), test whether the per-voxel mean differs
    from ``popmean``.

    Args:
        bd: BrainData instance (must contain multiple images).
        popmean: Population mean to test against. Default 0.0.
        permutation: If True, use sign-flip permutation test via
            ``nltools.stats.one_sample_permutation_test``; the p-values come
            from the empirical null and the parametric t-statistic is still
            reported alongside for reference.
        n_permute: Number of permutations (used only when
            ``permutation=True``). Default 5000.
        tail: Tail of the test (1 or 2). Default 2.
        return_null: If True, also return the null distribution. Default False.
        n_jobs: Number of parallel jobs. Default -1 (all cores).
        random_state: Random seed for reproducibility.

    Returns:
        dict with four BrainData keys:

            - ``"mean"``: voxelwise mean across images (effect-size estimate).
            - ``"t"``: parametric one-sample t-statistic.
            - ``"z"``: signed z-score, ``sign(t) * norm.isf(p/2)``, matching
              nilearn's ``output_type='z_score'``. Useful for thresholding
              on z at small df where t tails are heavier than normal.
            - ``"p"``: p-value (parametric, or permutation-based when
              ``permutation=True``).

        The effect size is always returned alongside the inferential maps so
        group-level code never has to compute the mean separately.

    Raises:
        ValueError: If ``bd`` contains fewer than 2 images.
    """
    from scipy.stats import ttest_1samp

    from . import BrainData

    if bd.data.ndim < 2 or bd.data.shape[0] < 2:
        raise ValueError(
            "t-test requires multiple images (got shape[0] < 2). "
            "Stack subject-level maps into a single BrainData first."
        )

    # Parametric t / p are always computed — they're the cheap reference
    # even on the permutation path.
    t_arr, p_param = ttest_1samp(bd.data, popmean, axis=0)
    mean_arr = np.asarray(bd.data).mean(axis=0) - popmean

    if permutation:
        from nltools.stats import one_sample_permutation_test

        perm = one_sample_permutation_test(
            bd.data,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        p_arr = np.asarray(perm["p"])
        # Permutation gave us its own mean — prefer it for numerical
        # consistency with the reported p.
        mean_arr = np.asarray(perm["mean"])
    else:
        p_arr = np.asarray(p_param)

    z_arr = _signed_z_from_p(t_arr, p_arr)

    return {
        "mean": BrainData(np.asarray(mean_arr), mask=bd.mask),
        "t": BrainData(np.asarray(t_arr), mask=bd.mask),
        "z": BrainData(z_arr, mask=bd.mask),
        "p": BrainData(p_arr, mask=bd.mask),
    }


def ttest2(bd, other, equal_var=True):
    """Two-sample voxelwise t-test between two BrainData stacks.

    Args:
        bd: First BrainData (shape ``(n1, n_voxels)``).
        other: Second BrainData (shape ``(n2, n_voxels)``).
        equal_var: If True (default), standard two-sample t-test. If False,
            Welch's t-test.

    Returns:
        dict: ``{"t": BrainData, "p": BrainData}``.

    Raises:
        ValueError: If the two BrainData objects have mismatched n_voxels.
    """
    from scipy.stats import ttest_ind

    from . import BrainData

    if bd.data.shape[1] != other.data.shape[1]:
        raise ValueError(
            f"BrainData objects must have same n_voxels. "
            f"Got {bd.data.shape[1]} and {other.data.shape[1]}."
        )

    t_arr, p_arr = ttest_ind(bd.data, other.data, axis=0, equal_var=equal_var)
    t_bd = BrainData(mask=bd.mask)
    t_bd.data = np.asarray(t_arr)
    p_bd = BrainData(mask=bd.mask)
    p_bd.data = np.asarray(p_arr)
    return {"t": t_bd, "p": p_bd}


def regress(bd, design_matrix=None, method="ols", mode=None):
    """Deprecated: Use fit(model='glm', X=design_matrix) instead.

    Args:
        bd: BrainData instance.
        design_matrix: Design matrix (unused, raises error).
        method: Noise model (unused, raises error).
        mode: Mode (unused, raises error).
    """
    raise NotImplementedError(
        "The regress() method has been removed in v0.6.0. "
        "Use fit(model='glm', X=design_matrix) instead. "
        "See migration guide for examples."
    )


_CONTRAST_OUTPUT_TYPES = {
    "t": "stat",
    "z": "z_score",
    "p": "p_value",
    "beta": "effect_size",
    "effect_size": "effect_size",
    "all": "all",
}

# nilearn 'all' output → cleaner nltools keys. effect_variance is stored
# as the standard error after a sqrt, for symmetry with bd.glm_se.
_CONTRAST_ALL_KEY_MAP = {
    "effect_size": "beta",
    "stat": "t",
    "z_score": "z",
    "p_value": "p",
    "effect_variance": "se",
}


def _contrast_all_to_bd(img_dict, mask):
    """Convert nilearn's output_type='all' dict into a dict of BrainData.

    ``effect_variance`` is stored as the (voxel-wise) standard error, i.e.
    ``sqrt(|effect_variance|)`` — matches how ``bd.glm_se`` is computed in
    :func:`fit_glm` and what a user thinks of as "SE of the contrast".
    """
    from . import BrainData

    out = {}
    for src_key, dst_key in _CONTRAST_ALL_KEY_MAP.items():
        img = img_dict[src_key]
        if dst_key == "se":
            bd_map = BrainData(img, mask=mask, verbose=False)
            bd_map.data = np.sqrt(np.abs(np.asarray(bd_map.data)))
            out[dst_key] = bd_map
        else:
            out[dst_key] = BrainData(img, mask=mask, verbose=False)
    return out


def compute_contrasts(bd, contrasts, contrast_type="t"):
    """Compute contrasts from a fitted GLM.

    Delegates to the underlying ``nilearn.FirstLevelModel.compute_contrast`` so
    t-statistics are computed with the full parameter covariance matrix —
    linear-combination-of-stored-betas cannot do this correctly for multi-
    regressor contrasts (it would ignore off-diagonal covariance and produce
    an effect-size map, not a t-map).

    Must be called after ``.fit(model='glm', X=design_matrix)`` has been run.

    Args:
        bd: BrainData instance.
        contrasts: Can be:

            - str: a contrast expressed in terms of column names, e.g.
              ``"conditionA - conditionB"`` or ``"2*conditionA - conditionB - conditionC"``
            - array-like: a numeric contrast vector, one weight per regressor
              (e.g. ``[1, -1, 0, 0]``)
            - dict: ``{name: contrast}`` for multiple contrasts at once
        contrast_type (str): What to return per contrast. One of:

            - ``"t"`` (default): t-statistic map (for thresholding /
              single-subject inference)
            - ``"z"``: z-score map
            - ``"p"``: p-value map
            - ``"beta"`` / ``"effect_size"``: effect-size (β) map — use this
              when feeding into a second-level (group) analysis
            - ``"all"``: a bundle dict ``{"beta", "t", "z", "p", "se"}``
              of BrainData maps for this one contrast. One fit, one call,
              every view — effect size *and* inferential maps together so
              group-level code never has to recompute beta separately.

    Returns:
        Depends on inputs:

            - single contrast (str or array) + scalar ``contrast_type``:
              a single BrainData.
            - single contrast + ``contrast_type="all"``: a flat dict of five
              BrainData keyed by ``"beta"``/``"t"``/``"z"``/``"p"``/``"se"``.
            - dict of contrasts + scalar ``contrast_type``: a dict
              ``{name: BrainData}``.
            - dict of contrasts + ``contrast_type="all"``: a nested dict
              ``{name: {"beta", "t", "z", "p", "se"}}``.

    Raises:
        RuntimeError: if ``.fit(model='glm')`` has not been run.
        ValueError: if the contrast vector length or a column name is invalid,
            or if ``contrast_type`` is not one of the supported values.

    Examples:
        >>> data.fit(model="glm", X=dm)
        >>> # Single-subject t-map, ready to threshold
        >>> tmap = data.compute_contrasts("conditionA - conditionB")
        >>> # Effect-size map for use as input to a group-level analysis
        >>> beta = data.compute_contrasts(
        ...     "conditionA - conditionB", contrast_type="beta"
        ... )
        >>> # Everything at once: threshold on res["t"], feed group on res["beta"]
        >>> res = data.compute_contrasts(
        ...     "conditionA - conditionB", contrast_type="all"
        ... )
        >>> res["t"].plot(threshold=3.09)
        >>> group_effects.append(res["beta"])

    Notes:
        - String contrasts support coefficients: ``"2*A - B"`` or ``"0.5*A + 0.5*B"``.
        - Column names must match design matrix columns exactly (case-sensitive).
        - For group analysis, stack per-subject effect-size maps
          (``contrast_type="beta"`` or ``res["beta"]`` from ``contrast_type="all"``)
          and run a second-level test (e.g. ``BrainData.ttest``). Mixing first-level
          t-maps into a group one-sample test conflates effect magnitude with precision.
    """
    from . import BrainData

    if not hasattr(bd, "glm_betas"):
        raise RuntimeError(
            "Must run .fit(model='glm', X=design_matrix) before computing contrasts"
        )
    if not hasattr(bd, "model_") or bd.model_ is None:
        raise RuntimeError(
            "BrainData has glm_* results but no model_ attached; refit with "
            ".fit(model='glm', X=design_matrix) to enable compute_contrasts."
        )

    if contrast_type not in _CONTRAST_OUTPUT_TYPES:
        raise ValueError(
            f"contrast_type must be one of {sorted(_CONTRAST_OUTPUT_TYPES)}; "
            f"got {contrast_type!r}"
        )
    output_type = _CONTRAST_OUTPUT_TYPES[contrast_type]
    want_all = contrast_type == "all"

    # Normalize contrasts → {name: contrast_def}
    if isinstance(contrasts, (str, list, np.ndarray)):
        contrast_dict = {"contrast": contrasts}
        single_contrast = True
    elif isinstance(contrasts, dict):
        contrast_dict = contrasts
        single_contrast = False
    else:
        raise TypeError("contrasts must be str, array, or dict")

    n_regressors = bd.glm_betas.shape[0]
    results = {}
    for name, contrast_def in contrast_dict.items():
        if isinstance(contrast_def, str):
            contrast_vector = parse_contrast_string(bd, contrast_def)
        else:
            contrast_vector = np.asarray(contrast_def, dtype=float)

        if len(contrast_vector) != n_regressors:
            raise ValueError(
                f"Contrast vector length ({len(contrast_vector)}) must match "
                f"number of regressors ({n_regressors})"
            )

        contrast_out = bd.model_.compute_contrast(
            contrast_vector, output_type=output_type
        )
        if want_all:
            results[name] = _contrast_all_to_bd(contrast_out, bd.mask)
        else:
            results[name] = BrainData(contrast_out, mask=bd.mask, verbose=False)

    if single_contrast:
        return results["contrast"]
    return results


def parse_contrast_string(bd, contrast_str):
    """Parse a contrast string into a numeric contrast vector.

    Args:
        bd: BrainData instance.
        contrast_str (str): Contrast string like "A - B" or "2*A - B - C"

    Returns:
        np.array: Numeric contrast vector

    Raises:
        RuntimeError: If design_matrix not found (fit() not called)
        ValueError: If column name not found in design_matrix
    """
    import re

    if getattr(bd, "design_matrix", None) is None:
        raise RuntimeError(
            "No design matrix found. Run .fit(model='glm', X=design_matrix) first."
        )

    col_names = list(bd.design_matrix.columns)

    # Initialize contrast vector
    contrast_vector = np.zeros(len(col_names))

    # Parse the string
    # Split by + and - (keeping the operators)
    tokens = re.split(r"(\+|\-)", contrast_str)
    tokens = [t.strip() for t in tokens if t.strip()]

    # Process tokens
    sign = 1  # Start with positive
    for token in tokens:
        if token == "+":
            sign = 1
        elif token == "-":
            sign = -1
        else:
            # Parse coefficient and variable
            if "*" in token:
                coef_str, var_name = token.split("*")
                coef = float(coef_str.strip())
                var_name = var_name.strip()
            else:
                coef = 1
                var_name = token

            # Find column index
            if var_name in col_names:
                idx = col_names.index(var_name)
                contrast_vector[idx] = sign * coef
            else:
                raise ValueError(f"Column '{var_name}' not found in design matrix")

    return contrast_vector
