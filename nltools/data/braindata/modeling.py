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
    scheme="kfold",
    split_by=None,
    groups=None,
    random_state=None,
    **kwargs,
):
    """Create a cross-validation pipeline for this BrainData.

    Returns a Pipeline object that enables fluent, chainable transforms
    with cross-validation. Terminal methods like .predict() execute the
    pipeline and return results.

    Args:
        bd: BrainData instance.
        k: Number of folds (for kfold scheme). Defaults to 5.
        scheme: CV scheme type. Options:
            - 'kfold': k-fold cross-validation (default)
            - 'loro': leave-one-run-out (requires split_by='runs' or groups)
            - 'bootstrap': bootstrap with out-of-bag test sets
        split_by: Attribute name for group splits (e.g., 'runs').
            If provided and groups is None, will try to get groups from
            bd.X[split_by] if bd.X is a DataFrame.
        groups: Explicit group labels for CV splits.
        random_state: Random seed for reproducibility.
        **kwargs: Additional arguments passed to CVScheme.

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
        >>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)

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

    # Create CV scheme
    cv_scheme = CVScheme(
        k=k,
        scheme=scheme,
        split_by=split_by,
        random_state=random_state,
        **kwargs,
    )

    # Create and return pipeline
    return BrainDataPipeline(bd, cv=cv_scheme, groups=groups)


def fit(
    bd,
    model=None,
    X=None,
    cv=None,
    inplace=True,
    progress_bar=None,
    scale=True,
    scale_value=100.0,
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
    if model is None:
        raise TypeError("model must be provided")
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
        # Pass progress_bar to Ridge model
        ridge_kwargs = kwargs.copy()
        if "progress_bar" not in ridge_kwargs:
            ridge_kwargs["progress_bar"] = progress_bar
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
        # Pass progress_bar to GLM (not verbose - we use tqdm progress bar)
        if "progress_bar" not in glm_kwargs:
            glm_kwargs["progress_bar"] = progress_bar
        target.model_ = Glm(**glm_kwargs)
        fit_glm(target, X_model)
    else:
        raise ValueError(f"Unknown model '{model}'. Must be one of: 'ridge', 'glm'")

    # If inplace=False, copy model_ to bd (needed for predict()) and return Fit
    if not inplace:
        # Store model_ and X_ on bd for predict() to work
        bd.model_ = target.model_
        # Also store design_matrix for GLM compute_contrasts()
        if model == "glm" and hasattr(target, "design_matrix"):
            bd.design_matrix = target.design_matrix
        # Return Fit dataclass with results
        return to_fit_dataclass(target, model=model)
    else:
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
    # Perform cross-validation if requested
    if cv is not None:
        bd.cv_results_ = compute_ridge_cv(bd, X, cv, **kwargs)

    # Always fit full model on all training data
    bd.model_.fit(X, bd.data)

    # Extract weights as BrainData
    # Ridge.coef_ is already (n_features, n_voxels) - no transpose needed
    bd.ridge_weights = shallow_copy(bd)
    bd.ridge_weights.data = bd.model_.coef_

    # Compute fitted values
    fitted = bd.model_.predict(X)
    bd.ridge_fitted_values = shallow_copy(bd)
    bd.ridge_fitted_values.data = fitted

    # Compute R-squared scores per voxel (using model's score method which now returns per-voxel)
    scores = bd.model_.score(X, bd.data)  # (n_voxels,)
    bd.ridge_scores = shallow_copy(bd)
    bd.ridge_scores.data = scores.reshape(1, -1)  # (1, n_voxels)


def compute_ridge_cv(bd, X, cv, alpha=None, alphas=None, backend="auto", **kwargs):
    """Compute cross-validation results for Ridge regression.

    Args:
        bd: BrainData instance.
        X (ndarray or list): Training features. If ndarray, shape (n_samples, n_features).
            If list, list of feature spaces for banded ridge.
        cv (int, 'auto', or sklearn CV splitter): Cross-validation specification
        alpha (float or 'auto', optional): Regularization strength (extracted from model if not provided)
        alphas (array-like, optional): Alpha values to try for alpha selection
        backend (str): Computational backend ('numpy', 'torch', 'auto'). Default: 'auto'
        **kwargs (dict): Additional arguments (currently unused, for future extensibility)

    Returns:
        dict: Dictionary containing:
            - 'scores': (n_folds, n_voxels) array of R-squared per fold
            - 'mean_score': (n_voxels,) array of mean R-squared across folds
            - 'predictions': BrainData of out-of-fold predictions
            - 'folds': (n_samples,) array of fold indices
            - 'best_alpha': Selected alpha (if alpha selection performed)
            - 'alpha_scores': (n_folds, n_alphas, n_voxels) array (if alpha selection)
    """
    from sklearn.model_selection import check_cv
    from sklearn.metrics import r2_score
    from nltools.algorithms.ridge import ridge_cv, ridge_svd

    # Convert backend to parallel parameter for ridge functions
    # ridge_svd/ridge_cv expect: None, 'cpu', or 'gpu'
    if backend in ("auto", "numpy", None):
        parallel = "cpu"
    elif backend == "torch":
        parallel = "gpu"
    elif backend in ("cpu", "gpu"):
        parallel = backend
    else:
        parallel = "cpu"  # Safe default

    # Get alpha from model if not explicitly provided
    if alpha is None:
        alpha = bd.model_.alpha if hasattr(bd.model_, "alpha") else 1.0

    # Normalize cv parameter
    if cv == "auto":
        # cv='auto' implies alpha selection with default 5 folds
        cv_splitter = check_cv(5)
        perform_alpha_selection = True
    elif isinstance(cv, int):
        cv_splitter = check_cv(cv)
        perform_alpha_selection = alpha == "auto"
    else:
        # Assume sklearn CV object
        cv_splitter = cv
        perform_alpha_selection = alpha == "auto"

    # Handle list (banded ridge) vs array (regular ridge)
    if isinstance(X, list):
        # Banded ridge CV is handled by the model itself, not here
        # This shouldn't be called for banded ridge
        raise ValueError(
            "Cross-validation for banded ridge should be handled by the model. "
            "Use alpha='auto' with cv parameter in fit()."
        )

    n_samples, n_features = X.shape
    n_voxels = bd.data.shape[1]

    # Initialize results
    cv_predictions = np.zeros_like(bd.data)
    fold_indices = np.zeros(n_samples, dtype=int)

    # Perform alpha selection if requested
    if perform_alpha_selection:
        # Use efficient ridge_cv for alpha selection
        if alphas is None:
            alphas = np.logspace(-2, 4, 20)  # Default alpha grid

        result = ridge_cv(
            X, bd.data, alphas=alphas, cv=cv_splitter.n_splits, parallel=parallel
        )

        best_alpha = result["alpha"]
        alpha_scores = result["cv_scores"]  # (n_folds, n_alphas, n_targets)

        # Update model's alpha to best_alpha
        bd.model_.alpha = best_alpha
        alpha = best_alpha

        cv_results = {
            "best_alpha": best_alpha,
            "alpha_scores": alpha_scores,
        }
    else:
        cv_results = {}

    # Perform CV with fixed alpha to get predictions and per-fold scores
    fold_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = bd.data[train_idx], bd.data[test_idx]

        # Fit Ridge on training fold
        coef = ridge_svd(X_train, y_train, alpha=alpha, parallel=parallel)

        # Predict on test fold
        y_pred = X_test @ coef

        # Store out-of-fold predictions
        cv_predictions[test_idx] = y_pred

        # Store fold indices
        fold_indices[test_idx] = fold_idx

        # Compute R-squared for this fold (per-voxel)
        fold_r2 = np.array(
            [r2_score(y_test[:, j], y_pred[:, j]) for j in range(n_voxels)]
        )
        fold_scores.append(fold_r2)

    # Aggregate results
    fold_scores = np.array(fold_scores)  # (n_folds, n_voxels)
    mean_scores = fold_scores.mean(axis=0)  # (n_voxels,)

    # Store predictions as BrainData
    cv_predictions_brain = shallow_copy(bd)
    cv_predictions_brain.data = cv_predictions

    # Package results
    cv_results.update(
        {
            "scores": fold_scores,
            "mean_score": mean_scores,
            "predictions": cv_predictions_brain,
            "folds": fold_indices,
        }
    )

    return cv_results


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

    elif model == "glm":
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

    else:
        raise ValueError(f"Unknown model '{model}'. Must be 'ridge' or 'glm'")


def ttest(bd, popmean=0.0, permutation=False, **kwargs):
    """One-sample voxelwise t-test across images (axis 0).

    For a BrainData stack of images (e.g. subject-level contrast maps with
    shape ``(n_samples, n_voxels)``), test whether the per-voxel mean differs
    from ``popmean``.

    Args:
        bd: BrainData instance (must contain multiple images).
        popmean: Population mean to test against. Default 0.0.
        permutation: If True, use sign-flip permutation test via
            ``nltools.stats.one_sample_permutation_test``. ``**kwargs`` are
            forwarded (e.g. ``n_permute``, ``tail``, ``parallel``,
            ``random_state``).
        **kwargs: Forwarded to the permutation test when ``permutation=True``.

    Returns:
        dict: ``{"t": BrainData, "p": BrainData}`` for the parametric case, or
        ``{"mean": BrainData, "p": BrainData}`` when ``permutation=True``
        (mirrors ``Adjacency.ttest``).

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

    if permutation:
        from nltools.stats import one_sample_permutation_test

        result = one_sample_permutation_test(bd.data, **kwargs)
        mean_bd = BrainData(mask=bd.mask)
        mean_bd.data = np.asarray(result["mean"])
        p_bd = BrainData(mask=bd.mask)
        p_bd.data = np.asarray(result["p"])
        return {"mean": mean_bd, "p": p_bd}

    t_arr, p_arr = ttest_1samp(bd.data, popmean, axis=0)
    t_bd = BrainData(mask=bd.mask)
    t_bd.data = np.asarray(t_arr)
    p_bd = BrainData(mask=bd.mask)
    p_bd.data = np.asarray(p_arr)
    return {"t": t_bd, "p": p_bd}


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


def regress(bd, design_matrix=None, noise_model="ols", mode=None, **kwargs):
    """Deprecated: Use fit(model='glm', X=design_matrix) instead.

    Args:
        bd: BrainData instance.
        design_matrix: Design matrix (unused, raises error).
        noise_model: Noise model (unused, raises error).
        mode: Mode (unused, raises error).
        **kwargs: Additional arguments (unused, raises error).
    """
    raise NotImplementedError(
        "The regress() method has been removed in v0.6.0. "
        "Use fit(model='glm', X=design_matrix) instead. "
        "See migration guide for examples."
    )


def compute_contrasts(bd, contrasts, contrast_type="t"):
    """Compute contrasts from fitted GLM results.

    This method computes contrasts as linear combinations of the GLM beta coefficients.
    Must be called after .fit(model='glm', X=design_matrix) has been run.

    Args:
        bd: BrainData instance.
        contrasts: Can be:

            - str: A string specifying the contrast using column names
              e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC"
            - dict: Dictionary with contrast names as keys and contrast strings/vectors as values
              e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]}
            - array: Numeric contrast vector matching the number of regressors
              e.g., [1, -1, 0, 0] for a 4-regressor model
        contrast_type (str): Type of contrast statistic ('t' or 'F'). Default: 't'
            Note: Currently only 't' contrasts are supported.

    Returns:
        BrainData or dict: If single contrast, returns BrainData object with contrast map.
                           If multiple contrasts (dict input), returns dict of BrainData objects.

    Raises:
        RuntimeError: If .fit(model='glm') hasn't been called yet
        ValueError: If contrast vector length doesn't match number of regressors
        ValueError: If column name in string contrast not found in design matrix

    Examples:
        >>> # Fit GLM model
        >>> design_matrix = pd.DataFrame({
        ...     'intercept': np.ones(n_samples),
        ...     'conditionA': signal_a,
        ...     'conditionB': signal_b
        ... })
        >>> brain.fit(model='glm', X=design_matrix)
        >>>
        >>> # Simple numeric contrast: A - B
        >>> contrast1 = brain.compute_contrasts([0, 1, -1])
        >>>
        >>> # String-based contrast (more readable)
        >>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
        >>>
        >>> # Multiple contrasts at once
        >>> contrasts = {
        ...     "A_vs_B": "conditionA - conditionB",
        ...     "avg_effect": [0, 0.5, 0.5],
        ...     "weighted": "2*conditionA - conditionB"
        ... }
        >>> results = brain.compute_contrasts(contrasts)
        >>> # results is a dict: {"A_vs_B": BrainData, "avg_effect": BrainData, ...}

    Notes:
        - String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
        - Column names must match design matrix columns exactly (case-sensitive)
        - Contrast weights should sum to zero for proper inference in most cases
    """
    # Check that regression has been run
    if not hasattr(bd, "glm_betas"):
        raise RuntimeError(
            "Must run .fit(model='glm', X=design_matrix) before computing contrasts"
        )

    # Parse contrasts
    if isinstance(contrasts, str):
        # Single string contrast
        contrast_dict = {"contrast": contrasts}
        single_contrast = True
    elif isinstance(contrasts, (list, np.ndarray)):
        # Single numeric contrast
        contrast_dict = {"contrast": contrasts}
        single_contrast = True
    elif isinstance(contrasts, dict):
        # Multiple contrasts
        contrast_dict = contrasts
        single_contrast = False
    else:
        raise TypeError("contrasts must be str, array, or dict")

    # Process each contrast
    results = {}
    for name, contrast_def in contrast_dict.items():
        # Parse string contrasts to create contrast vector
        if isinstance(contrast_def, str):
            # Parse string like "conditionA - conditionB"
            contrast_vector = parse_contrast_string(bd, contrast_def)
        else:
            # Use numeric contrast directly
            contrast_vector = np.array(contrast_def)

        # Validate contrast vector length
        if len(contrast_vector) != bd.glm_betas.shape[0]:
            raise ValueError(
                f"Contrast vector length ({len(contrast_vector)}) must match "
                f"number of regressors ({bd.glm_betas.shape[0]})"
            )

        # Compute contrast by linear combination of betas
        contrast_data = np.zeros((1, bd.glm_betas.shape[1]))
        for i, weight in enumerate(contrast_vector):
            if weight != 0:
                contrast_data += weight * bd.glm_betas[i].data

        # Create BrainData object for contrast
        contrast_brain = bd[0].copy()
        contrast_brain.data = contrast_data

        # Store result
        results[name] = contrast_brain

    # Return single contrast or dict of contrasts
    if single_contrast:
        return results["contrast"]
    else:
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
