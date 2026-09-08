"""Model fitting, contrasts, and statistical tests for `BrainData`.

GLM and ridge fitting (with cross-validation), contrast computation, one- and
two-sample t-tests, and the design-matrix diagnostics `fit` runs before a GLM.
`BrainData` methods delegate here.
"""

import warnings
from copy import deepcopy

import numpy as np

from nltools.utils import find_stack_level
from .utils import _clear_fit_state, _copy_for_fit, _result_from_array


def resolve_preprocessing_defaults(model, scale, standardize):
    """Resolve the ``'auto'`` scale/standardize sentinels to concrete values.

    Per-model defaults for ``BrainData.fit``. ``scale`` (percent-signal-change)
    is opt-in for both models. Ridge standardizes its targets by default so a
    shared alpha regularizes voxels fairly; GLM does neither so betas stay in
    native units.

    Args:
        model (str): ``'ridge'`` or ``'glm'``.
        scale (bool or 'auto'): Requested scale flag.
        standardize (str, None, or 'auto'): Requested standardize method.

    Returns:
        tuple: ``(scale, standardize)`` with any ``'auto'`` resolved.
    """
    if scale == "auto":
        scale = False
    if standardize == "auto":
        standardize = "zscore" if model == "ridge" else None
    return scale, standardize


class RankDeficientDesignWarning(UserWarning):
    """The design matrix supplied to ``fit()`` is rank deficient.

    Subclasses ``UserWarning`` so it participates in default filtering, while
    remaining individually silenceable:
    ``warnings.filterwarnings("ignore", category=RankDeficientDesignWarning)``.
    """


class NearCollinearDesignWarning(UserWarning):
    """The design matrix supplied to ``fit()`` is full rank but nearly collinear.

    Subclasses ``UserWarning`` so it participates in default filtering, while
    remaining individually silenceable:
    ``warnings.filterwarnings("ignore", category=NearCollinearDesignWarning)``.
    """


# Pairwise |correlation| at or above which a full-rank design is flagged as
# near-collinear. Heritage of the removed v0.5 ``design_clean`` default
# threshold — designs it used to silently prune now warn instead.
NEAR_COLLINEAR_CORR_THRESHOLD = 0.95

# Condition number of the column-standardized design above which multi-column
# near-dependence is flagged — the classic cutoff from Belsley, Kuh & Welsch
# (1980). Catches dependence spread across 3+ columns that no pairwise
# correlation reveals.
NEAR_COLLINEAR_CONDITION_THRESHOLD = 30.0


def _redundant_column_names(finite, rank, columns):
    """Name the columns most likely responsible for a rank deficiency.

    Uses pivoted QR: the pivots beyond the numerical rank are the columns QR
    would discard as linear combinations of the ones before them. Which member
    of a dependent set gets blamed is arbitrary (that arbitrariness is exactly
    why the deficiency matters), so the result is a "likely involved" hint,
    not a verdict. Truncated so a wide design cannot flood the warning.
    """
    from scipy.linalg import qr

    _, _, pivots = qr(finite, mode="economic", pivoting=True)
    redundant = sorted(int(i) for i in pivots[rank:])
    names = [
        str(columns[i]) if columns is not None else f"column {i}" for i in redundant
    ]
    if len(names) > 5:
        names = names[:5] + [f"... and {len(names) - 5} more"]
    return names


def _warn_if_rank_deficient(X_array, X_model):
    """Warn when a design matrix is rank deficient.

    A rank-deficient design has no unique least-squares solution. The GLM still
    returns betas — nilearn falls back to a pseudo-inverse — but the effect is
    split arbitrarily across the linearly dependent columns, so any contrast
    touching that subspace is not interpretable. Silence here is dangerous
    because the failure is invisible in the output: the betas come back finite
    and plausible.

    The warning diagnoses the deficiency (naming the likely-involved columns,
    or the p > n shape when that is the cause) and offers the fixes: inspect
    with ``DesignMatrix.vif()``, drop redundant columns with
    ``DesignMatrix.clean()`` (order-dependent for correlated pairs), or use
    regularization (``fit(model='ridge')``), whose solution is unique and
    order-invariant even when ``X'X`` is singular.

    We warn rather than raise because over-parameterized designs can still have
    estimable contrasts, and because raising would break pipelines currently
    relying (silently) on the pseudo-inverse.

    Args:
        X_array (np.ndarray): Design matrix as a 2-D array.
        X_model: The design object supplied by the caller, used for column
            names when it has them.

    Returns:
        bool: True if the warning fired (the design is rank deficient), so the
            caller can skip the near-collinearity check — an exactly deficient
            design should raise only this warning, never both.
    """
    if X_array.ndim != 2 or X_array.shape[1] < 2:
        return False

    finite = X_array[np.isfinite(X_array).all(axis=1)]
    if finite.shape[0] == 0:
        # Nothing to assess; the fit itself will fail loudly on the NaNs.
        return False

    n_cols = X_array.shape[1]
    rank = int(np.linalg.matrix_rank(finite))
    if rank >= n_cols:
        return False

    columns = getattr(X_model, "columns", None)
    if finite.shape[0] < n_cols:
        # More regressors than (finite) timepoints: deficient by construction,
        # no matter what the columns contain.
        diagnosis = (
            f"the design has more columns ({n_cols}) than usable rows "
            f"({finite.shape[0]}), so it cannot be full rank"
        )
    else:
        names = _redundant_column_names(finite, rank, columns)
        diagnosis = (
            f"{n_cols - rank} column(s) are linear combinations of the others "
            f"(likely involved: {', '.join(names)})"
        )
    warnings.warn(
        f"Design matrix is rank deficient: rank {rank} of {n_cols} columns — "
        f"{diagnosis}. The OLS betas are not uniquely determined, and "
        "contrasts touching the dependent columns are not interpretable: the "
        "fit silently returns one of infinitely many solutions. Possible "
        "fixes: (1) inspect the collinearity with `DesignMatrix.vif()`; "
        "(2) try `DesignMatrix.clean()` to drop redundant columns before "
        "fitting (note: which of a correlated pair survives depends on the "
        "order the design was built in); (3) try regularization — "
        "`fit(model='ridge')` keeps every regressor and has a unique, "
        "order-invariant solution.",
        RankDeficientDesignWarning,
        stacklevel=find_stack_level(),
    )
    return True


def _near_collinear_pairs(sub, varying, columns):
    """Describe the column pairs correlated at or above the pairwise threshold.

    ``sub`` holds only the varying (non-constant) columns; ``varying`` maps its
    column positions back to the original design so names stay correct.
    Sorted by |r| descending and truncated so a wide design cannot flood the
    warning.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(sub, rowvar=False)
    abs_corr = np.abs(corr)
    rows, cols = np.triu_indices_from(abs_corr, k=1)
    over = [
        (float(abs_corr[i, j]), int(i), int(j))
        for i, j in zip(rows, cols)
        if abs_corr[i, j] >= NEAR_COLLINEAR_CORR_THRESHOLD
    ]
    over.sort(reverse=True)

    def name(k):
        orig = int(varying[k])
        return str(columns[orig]) if columns is not None else f"column {orig}"

    descs = [f"{name(i)} & {name(j)} (|r| = {r:.2f})" for r, i, j in over]
    if len(descs) > 5:
        descs = descs[:5] + [f"... and {len(descs) - 5} more"]
    return descs


def _warn_if_near_collinear(X_array, X_model):
    """Warn when a full-rank design matrix is nearly collinear.

    A near-collinear design has a unique least-squares solution, but a fragile
    one: the variance of the betas on the correlated columns is inflated, so
    small perturbations of the data can flip their signs or magnitudes. Two
    complementary signals, either of which fires the warning (the message says
    which): a pairwise |correlation| at or above
    ``NEAR_COLLINEAR_CORR_THRESHOLD`` — the direct heritage of the threshold
    the removed v0.5 ``design_clean`` pruned at — and a condition number of
    the column-standardized design above
    ``NEAR_COLLINEAR_CONDITION_THRESHOLD``, which catches near-dependence
    spread across three or more columns that no pairwise correlation reveals.

    Constant columns (generated intercepts, all-ones baselines) are excluded
    from both signals, consistent with ``DesignMatrix.vif()`` (which drops
    generated intercepts) and ``DesignMatrix.clean()`` (which treats constant
    columns as r = 0): a constant has no correlation with anything and cannot
    be standardized.

    This is a warning, never an error, and nothing is dropped — v0.6.0
    deliberately removed the implicit ``design_clean`` auto-dropping; the fix
    is a modeling decision that belongs to the caller. ``fit`` calls this only
    when the exact-rank check stayed silent, so a rank-deficient design raises
    ``RankDeficientDesignWarning`` alone.

    Args:
        X_array (np.ndarray): Design matrix as a 2-D array.
        X_model: The design object supplied by the caller, used for column
            names when it has them.
    """
    if X_array.ndim != 2 or X_array.shape[1] < 2:
        return

    finite = X_array[np.isfinite(X_array).all(axis=1)]
    if finite.shape[0] < 3:
        return

    columns = getattr(X_model, "columns", None)
    varying = np.flatnonzero(finite.var(axis=0) > 0)
    if varying.size < 2:
        return
    sub = finite[:, varying]

    signals = []
    pair_descs = _near_collinear_pairs(sub, varying, columns)
    if pair_descs:
        signals.append(
            f"column pair(s) correlated at |r| >= "
            f"{NEAR_COLLINEAR_CORR_THRESHOLD}: {', '.join(pair_descs)}"
        )
    standardized = (sub - sub.mean(axis=0)) / sub.std(axis=0)
    condition_number = float(np.linalg.cond(standardized))
    if condition_number > NEAR_COLLINEAR_CONDITION_THRESHOLD:
        signals.append(
            f"the condition number of the standardized design is "
            f"{condition_number:.0f} (> "
            f"{NEAR_COLLINEAR_CONDITION_THRESHOLD:.0f}), indicating "
            f"near-linear dependence spread across several columns"
        )
    if not signals:
        return

    warnings.warn(
        f"Design matrix is nearly collinear (full rank, but ill-conditioned): "
        f"{'; '.join(signals)}. The OLS betas are estimable but unstable: "
        "their variance is inflated, and small changes in the data can flip "
        "their signs or magnitudes. Nothing was dropped or modified. Possible "
        "fixes: (1) inspect the collinearity with `DesignMatrix.vif()`; "
        "(2) consider `DesignMatrix.clean()` to drop near-duplicate columns "
        "before fitting (note: which of a correlated pair survives depends on "
        "the order the design was built in); (3) try regularization — "
        "`fit(model='ridge')` keeps every regressor and shrinks correlated "
        "coefficients together.",
        NearCollinearDesignWarning,
        stacklevel=find_stack_level(),
    )


def fit(  # nosemgrep: kwargs-internal-forwarding  # forwards model params to the nilearn FirstLevelModel / ridge estimator
    bd,
    model="glm",
    *,
    X=None,
    cv=None,
    device="cpu",
    local_alpha=True,
    fit_intercept=False,
    inplace=True,
    progress_bar=False,
    scale="auto",
    standardize="auto",
    **kwargs,
):
    """Fit a model to brain imaging data.

    Creates and fits a model from string specification. The brain data
    (bd.data) is always used as the target variable. Model and results
    are stored for later use with predict().

    For ``model='glm'`` the design is diagnosed before estimation, as warnings
    only — nothing is ever dropped, modified, or raised on. An exactly
    rank-deficient design fires `RankDeficientDesignWarning`; a full-rank but
    near-collinear design (a column pair with |r| >= 0.95, or a
    column-standardized condition number above 30) fires
    `NearCollinearDesignWarning` instead — never both. Each has its own
    category so it can be silenced surgically with
    ``warnings.filterwarnings``.

    **Results stored on the returned `BrainData`:**

    - `model_` — the fitted `Ridge` or `Glm` instance (always set, so `predict()`
      works).
    - `X_` — the training design/features, used as the `predict()` default.
    - `cv_results_` — dict with keys `'scores'`, `'mean_score'`, `'predictions'`,
      `'folds'`, `'best_alpha'`, `'alpha_scores'` (ridge with `cv` only).
    - GLM: `glm_betas`, `glm_t`, `glm_p`, `glm_se`, `glm_residual`,
      `glm_predicted`, `glm_r2` (each a `BrainData`).
    - Ridge: `ridge_weights`, `ridge_fitted_values`, `ridge_scores` (each a
      `BrainData`).

    Args:
        bd (BrainData): Data whose `.data` is the regression target.
        model (str): `'glm'` (default) or `'ridge'`.
        X (array-like | DataFrame | DesignMatrix): Design matrix (GLM) or feature
            matrix (ridge) of shape `(n_samples, n_features)`; `n_samples` must
            match `bd.data`. For banded ridge, a list of such matrices.
        cv (int | str | CV splitter | None): Cross-validation specification, ridge
            only. An int is the number of k-fold splits (returns CV scores);
            `'auto'` selects alpha via CV (implies `alpha='auto'`); an sklearn
            splitter (e.g. `KFold(3, shuffle=True)`) is used as given; None
            (default) runs no CV.
        device (str): Ridge only. Compute device for the ridge solve/CV: `'cpu'`
            (NumPy), `'gpu'` (PyTorch on CUDA/MPS when available), or `'auto'`
            (GPU if present, else CPU). Forwarded to `Ridge` and the CV
            evaluation. Ignored for `model='glm'`. Default: `'cpu'`.
        local_alpha (bool): Ridge only. If True, select a separate best alpha per
            voxel; if False, select a single shared alpha across all voxels.
            Forwarded to `Ridge`. Default: True.
        fit_intercept (bool): Ridge only. If True, fit an intercept term.
            Redundant (and warned against) when the data is already centered via
            `scale` or `standardize`. Forwarded to `Ridge`. Default: False.
        inplace (bool): If True, mutate `bd` and return it. If False, fit and
            return an independent `BrainData` copy while leaving every part of
            `bd` untouched. Default: True.
        progress_bar (bool): Display a progress bar for long-running
            operations. Default: False.
        scale (bool | str): Apply percent-signal-change scaling to the data
            before fitting, via nilearn's per-voxel `mean_scaling` (each voxel's
            time-series is divided by its own temporal mean, de-meaned, and
            multiplied by 100). `'auto'` (default) resolves to False for both
            models — PSC is opt-in. Useful for GLM (interpretable % betas); for
            ridge it is redundant with `standardize='zscore'` (a warning is
            raised for that combination). Applied before `standardize`.
        standardize (str | None): Standardize each voxel across observations
            after scaling: `'center'` (subtract the mean), `'zscore'` (subtract
            mean, divide by std), or None (off). `'auto'` (default) resolves to
            `'zscore'` for `model='ridge'` (so a shared alpha regularizes voxels
            fairly) and None for `model='glm'`.
        **kwargs (dict): Additional arguments passed to the model constructor —
            for `Ridge`: `alpha`, `alphas`, `random_state`; for `Glm`:
            `noise_model`, `minimize_memory`, etc.

    Returns:
        BrainData: `bd` itself when `inplace=True`; otherwise an independently
            owned fitted copy.

    Examples:
        ```python
        # inplace=True (default): results are stored as attributes on brain_data
        brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
        print(f"CV R2: {brain_data.cv_results_['mean_score'].mean():.3f}")
        weights = brain_data.ridge_weights

        # inplace=False: fit a copy; brain_data remains completely unchanged
        fitted = brain_data.fit(
            model='ridge', alpha=1.0, cv=5, X=features, inplace=False
        )
        weights = fitted.ridge_weights
        assert not hasattr(brain_data, 'ridge_weights')
        print(f"CV R2: {fitted.cv_results_['mean_score'].mean():.3f}")

        # The returned GLM copy can compute contrasts
        fitted_glm = brain_data.fit(model='glm', X=design_matrix, inplace=False)
        contrast = fitted_glm.compute_contrasts('conditionA - conditionB')
        ```
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
        if not _warn_if_rank_deficient(X_array, X_model):
            _warn_if_near_collinear(X_array, X_model)
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

    target = bd if inplace else _copy_for_fit(bd)
    if inplace:
        _clear_fit_state(target)
    if isinstance(X_model, list):
        target.X_ = [np.array(part, copy=True) for part in X_model]
    elif hasattr(X_model, "copy"):
        target.X_ = X_model.copy()
    else:
        target.X_ = deepcopy(X_model)

    # Resolve per-model preprocessing defaults ('auto' sentinel).
    scale, standardize = resolve_preprocessing_defaults(model, scale, standardize)

    # scale (percent-signal-change) is redundant with z-scoring: per-voxel,
    # zscore(mean_scaling(Y)) == zscore(Y), so the scale step does nothing.
    # Warn rather than silently doing pointless work (and, with per-voxel
    # standardize other than zscore, scaling a regression target only distorts
    # its regularization — which is why ridge standardizes instead of scaling).
    if scale and standardize == "zscore":
        warnings.warn(
            "scale=True is redundant with standardize='zscore': z-scoring "
            "already absorbs percent-signal-change scaling, so the scale step "
            "has no effect. Drop scale or use standardize='center'/None.",
            UserWarning,
            stacklevel=find_stack_level(),
        )

    # A ridge intercept is redundant once the targets are centered (any scaling
    # or standardization de-means them), so fitting one adds nothing and usually
    # signals a misunderstanding of the preprocessing. Warn loudly. Intercepts
    # are for the raw-offset case (scale=False, standardize=None) — see
    # test_ridge_intercept_no_centering_ok.
    if model == "ridge" and fit_intercept and (scale or standardize is not None):
        warnings.warn(
            "fit_intercept=True is redundant for ridge when the data is centered "
            "by scale/standardize (the default standardize='zscore' already "
            "de-means each voxel), so the intercept is ~0 and adds nothing. Use "
            "fit_intercept=True only with scale=False, standardize=None.",
            UserWarning,
            stacklevel=find_stack_level(),
        )

    # Preprocess before fitting: scale (percent signal change) THEN standardize.
    # scale uses nilearn's per-voxel mean_scaling — the same transform
    # FirstLevelModel applies internally — called explicitly here so it is
    # user-controlled rather than an inherited nilearn default.
    if scale:
        from nilearn.glm.first_level import mean_scaling

        target.data = mean_scaling(target.data, axis=0)[0]
    if standardize is not None:
        from .analysis import standardize as standardize_data

        target.data = standardize_data(target, axis=0, method=standardize).data

    # Create model based on string
    if model == "ridge":
        # Device selection is a first-class facade kwarg (`device=`), not a
        # passthrough. Reject the retired algorithm-layer aliases loudly rather
        # than letting them silently reach Ridge (which no longer accepts them).
        for banned in ("backend", "parallel"):
            if banned in kwargs:
                raise TypeError(
                    f"`{banned}=` is not a valid ridge kwarg; use "
                    f"`device='cpu'|'gpu'|'auto'` to select the compute device."
                )
        # Forward progress_bar to Ridge model's progress_bar kwarg
        ridge_kwargs = kwargs.copy()
        if "progress_bar" not in ridge_kwargs:
            ridge_kwargs["progress_bar"] = progress_bar
        # Explicit-signature kwargs win over **kwargs forwarding so calls
        # like bd.fit(model='ridge', local_alpha=False, ...) reach Ridge.
        ridge_kwargs.setdefault("device", device)
        ridge_kwargs.setdefault("local_alpha", local_alpha)
        ridge_kwargs.setdefault("fit_intercept", fit_intercept)
        target.model_ = Ridge(**ridge_kwargs)
        fit_ridge(target, target.X_, cv=cv, device=device, **kwargs)
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
        fit_glm(target, target.X_)

    return target


def fit_ridge(  # nosemgrep: kwargs-internal-forwarding  # forwards ridge params (alpha) to compute_ridge_cv
    bd, X, cv=None, device="cpu", **kwargs
):
    """Fit Ridge model and extract results.

    Args:
        bd (BrainData): Data with `bd.model_` already set to a `Ridge` instance.
        X (np.ndarray | list[np.ndarray]): Training features (a list for banded
            ridge).
        cv (int | str | CV splitter | None): Cross-validation specification; see
            `fit`.
        device (str): Compute device (`'cpu'`/`'gpu'`/`'auto'`) for the held-out
            CV evaluation, forwarded to `compute_ridge_cv`. Default: `'cpu'`.
        **kwargs (dict): Additional ridge arguments for CV (`alpha`, etc.).

    Note:
        Sets `ridge_weights`, `ridge_fitted_values`, `ridge_scores`, and
        `cv_results_` (if `cv` is given) on `bd`.
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
        bd.cv_results_ = compute_ridge_cv(bd, X, cv, device=device, **kwargs)
        bd.model_.fit(X, bd.data)
    else:
        bd.model_.fit(X, bd.data)

    _populate_ridge_attributes(bd, X)


def _normalize_cv(cv):
    """Validate and normalize a cross-validation specification.

    Reject single-use generators and bad cv values; pass through ints and
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
    bd.ridge_weights = _result_from_array(
        bd, np.array(bd.model_.coef_, copy=True), rows="clear"
    )

    fitted = bd.model_.predict(X)
    bd.ridge_fitted_values = _result_from_array(
        bd, np.array(fitted, copy=True), rows="preserve"
    )

    scores = bd.model_.score(X, bd.data)  # (n_voxels,)
    bd.ridge_scores = _result_from_array(
        bd, np.array(scores, copy=True).reshape(1, -1), rows="clear"
    )


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

    alpha_scores = np.array(
        bd.model_.cv_scores_, copy=True
    )  # (n_splits, n_alphas, n_voxels)
    n_splits, n_alphas, n_voxels = alpha_scores.shape

    # Per-voxel selected α (already on the model). May be scalar when the
    # model squeezed a single-target case, but for multi-voxel BrainData
    # we'll always have a (n_voxels,) array — broadcast just in case.
    best_alpha = bd.model_.alpha_
    if not isinstance(best_alpha, np.ndarray):
        best_alpha_arr = np.full(n_voxels, float(best_alpha))
    else:
        best_alpha_arr = np.array(best_alpha, copy=True)

    # Per-voxel best-α index → per-fold scores at that α.
    # alpha_scores has the candidate alphas in the order solve_ridge_cv saw
    # them (i.e., the model's `alphas` attr). Recover that order to do the
    # lookup.
    # Match by nearest VALUE, not searchsorted: the alpha grid is whatever
    # the user passed and may be unsorted, so searchsorted would return the
    # wrong column (and thus per-fold scores for the wrong alpha).
    alpha_grid = np.asarray(bd.model_.alphas)
    best_idx = np.argmin(np.abs(alpha_grid[:, None] - best_alpha_arr[None, :]), axis=0)
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

    cv_predictions_brain = _result_from_array(
        bd, np.array(pred_result["predictions"], copy=True), rows="preserve"
    )

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


def compute_ridge_cv(bd, X, cv, alpha=None, device="cpu"):
    """Held-out CV scores under a fixed Ridge α.

    Used only for the *fixed-α* + CV branch. When `alpha='auto'`, alpha selection
    is handled by `Ridge.fit` (which delegates to `solve_ridge_cv`) and `fit`
    assembles `cv_results_` from the fitted model instead.

    Args:
        bd (BrainData): Data with `bd.model_` set to a `Ridge` instance.
        X (np.ndarray): Training features, shape `(n_samples, n_features)`.
        cv (int | CV splitter): Cross-validation specification.
        alpha (float | None): Fixed regularization strength. If None, taken from
            `bd.model_.alpha`.
        device (str): Compute device (`'cpu'`/`'gpu'`/`'auto'`). Default: `'cpu'`.

    Returns:
        dict: Keys `'scores'`, `'mean_score'`, `'predictions'`, `'folds'`.
    """
    from nltools.algorithms.ridge import cross_val_predict_ridge
    from nltools.algorithms.backends import resolve_backend

    cv_splitter = _normalize_cv(cv)

    if isinstance(X, list):
        raise ValueError(
            "Cross-validation for banded ridge should be handled by the model. "
            "Use alpha='auto' with cv parameter in fit()."
        )

    if alpha is None:
        alpha = bd.model_.alpha if hasattr(bd.model_, "alpha") else 1.0

    fit_intercept = bool(getattr(bd.model_, "fit_intercept", False))

    # Translate the facade 'device' selector to the ridge layer's 'parallel'
    # vocabulary by resolving to a concrete backend: 'gpu' requires an
    # accelerator, while 'auto' may fall back to CPU.
    backend_obj = resolve_backend(device)
    parallel = "gpu" if backend_obj.device in ("cuda", "mps") else "cpu"

    n_voxels = bd.data.shape[1]
    pred_result = cross_val_predict_ridge(
        X,
        bd.data,
        alphas=np.full(n_voxels, float(alpha)),
        cv=cv_splitter,
        fit_intercept=fit_intercept,
        parallel=parallel,
    )

    cv_predictions_brain = _result_from_array(
        bd, np.array(pred_result["predictions"], copy=True), rows="preserve"
    )

    return {
        "scores": pred_result["scores"],
        "mean_score": pred_result["scores"].mean(axis=0),
        "predictions": cv_predictions_brain,
        "folds": pred_result["folds"],
    }


def fit_glm(bd, X):
    """Fit GLM model and extract results.

    Args:
        bd (BrainData): Data with `bd.model_` set to a `Glm` instance.
        X (DataFrame | DesignMatrix): Design matrix.

    Note:
        Sets `glm_betas`, `glm_t`, `glm_p`, `glm_se`, `glm_residual`,
        `glm_predicted`, `glm_r2`, and `design_matrix` on `bd`.
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

    # Betas come straight from the cached coef_ (assembled from run_glm theta),
    # so the per-regressor maps stay in masked-array space with no Nifti
    # round-trip. coef_ is (n_regressors, n_voxels).
    bd.glm_betas = _result_from_array(
        bd, np.array(bd.model_.coef_, copy=True), rows="clear"
    )

    # Per-regressor t / p / se via nilearn's FUNCTIONAL compute_contrast on the
    # fitted (labels_, results_): arrays in masked space, no unmask. Correct for
    # both OLS and AR noise models (per-voxel covariance lives in results_).
    from nilearn.glm import compute_contrast as _compute_contrast

    labels = bd.model_.glm_.labels_[0]
    results = bd.model_.glm_.results_[0]
    n_regressors = X.shape[1]
    t_maps, p_maps, se_maps = [], [], []
    for i in range(n_regressors):
        con = np.zeros(n_regressors)
        con[i] = 1.0
        contrast = _compute_contrast(labels, results, con)
        t_maps.append(contrast.stat().ravel())
        p_maps.append(contrast.p_value().ravel())
        se_maps.append(np.sqrt(np.abs(contrast.effect_variance().ravel())))

    bd.glm_t = _result_from_array(bd, np.vstack(t_maps), rows="clear")
    bd.glm_p = _result_from_array(bd, np.vstack(p_maps), rows="clear")
    bd.glm_se = _result_from_array(bd, np.vstack(se_maps), rows="clear")

    # Residuals stay from nilearn: for AR noise models these are the whitened
    # residuals, which Y - X@coef_ does not reproduce, so keep nilearn's.
    bd.glm_residual = _result_from_array(
        bd, BrainData(bd.model_.residuals, mask=bd.mask).data, rows="preserve"
    )

    # Predicted = original - residuals
    bd.glm_predicted = _result_from_array(
        bd, bd.data - bd.glm_residual.data, rows="preserve"
    )

    # R-squared calculation
    ss_total = np.sum((bd.data - bd.data.mean(axis=0)) ** 2, axis=0)
    ss_residual = np.sum(bd.glm_residual.data**2, axis=0)
    r2_values = 1 - (ss_residual / (ss_total + 1e-10))

    # Create single-image BrainData for R-squared
    bd.glm_r2 = _result_from_array(bd, r2_values.reshape(1, -1), rows="clear")


def ttest(
    bd,
    *,
    popmean=0.0,
    permutation=False,
    n_permute=5000,
    tail=2,
    return_null=False,
    n_jobs=-1,
    random_state=None,
):
    """Run a one-sample voxelwise t-test across images (axis 0).

    For a BrainData stack of images (e.g. subject-level contrast maps with
    shape `(n_images, n_voxels)`), test whether the per-voxel mean differs from
    `popmean`. Delegates the statistics to the shared one-sample contract in
    `nltools.algorithms.inference.one_sample`.

    Args:
        bd (BrainData): Stack of two or more images.
        popmean (float): Population mean to test against. Default 0.0.
        permutation (bool): If True, take p from a sign-flip permutation test on
            `images - popmean` via `one_sample_permutation_test`. The reported
            `t` stays the observed parametric statistic. Default False.
        n_permute (int): Number of permutations, used only when
            `permutation=True`. Default 5000.
        tail (int | str): `2` or `'two'` for two-tailed (default); `1` or `'one'`
            for one-tailed (mean > `popmean`).
        return_null (bool): If True, also return the permutation null. Has no
            effect on the parametric path, which computes no null. Default False.
        n_jobs (int): Number of parallel jobs. Default -1 (all cores).
        random_state (int | None): Random seed for reproducibility.

    Returns:
        dict: `"mean"`, `"t"`, `"z"` and `"p"` as independent `BrainData` images
            with observation metadata cleared. `"mean"` is the voxelwise mean
            minus `popmean` — the effect relative to the tested null, equal to
            the raw mean only when `popmean=0`. `"t"` is the observed one-sample
            t-statistic on both paths. `"p"` is parametric, or the empirical
            sign-flip p-value when `permutation=True`. `"z"` is the tail-aware
            normal score of `p` (`sign(t) * norm.isf(p/2)` two-tailed), matching
            nilearn's `output_type='z_score'`. With `permutation=True` and
            `return_null=True` the dict also holds `"null_dist"`, an owned
            `(n_permute, n_voxels)` array of centered means in the units of
            `"mean"`. Maps are unthresholded. Apply a cutoff or a
            multiple-comparison correction afterwards.

    Raises:
        ValueError: If `bd` contains fewer than 2 images.

    Examples:
        ```python
        result = contrast_maps.ttest()
        significant = result["z"].data * (result["p"].data < 0.001)

        perm = contrast_maps.ttest(
            permutation=True, n_permute=5000, return_null=True, random_state=0
        )
        perm["null_dist"].shape  # → (5000, n_voxels)
        ```
    """
    from nltools.algorithms.inference.one_sample import _one_sample_statistics

    if bd.data.ndim < 2 or bd.data.shape[0] < 2:
        raise ValueError(
            "t-test requires multiple images (got shape[0] < 2). "
            "Stack subject-level maps into a single BrainData first."
        )

    stats = _one_sample_statistics(
        bd.data,
        popmean=popmean,
        permutation=permutation,
        n_permute=n_permute,
        tail=tail,
        return_null=return_null,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    results = {
        key: _result_from_array(bd, stats[key], rows="clear")
        for key in ("mean", "t", "z", "p")
    }
    if "null_dist" in stats:
        results["null_dist"] = stats["null_dist"]
    return results


_CONTRAST_OUTPUT_TYPES = {
    "t": "stat",
    "z": "z_score",
    "p": "p_value",
    "beta": "effect_size",
    "effect_size": "effect_size",
    "all": "all",
}


def _functional_contrast(labels, run_results, con_vec, statistic):
    """Compute contrast statistic(s) as masked-space arrays.

    Uses nilearn's FUNCTIONAL ``compute_contrast`` on the fitted
    ``(labels_, results_)`` — the full per-voxel parameter covariance (correct
    for OLS and AR) — with no unmasking to a Nifti. For ``statistic='all'``
    returns a dict of arrays keyed ``beta/t/z/p/se``; otherwise a single array.
    ``se`` is ``sqrt(|effect_variance|)``, matching ``bd.glm_se``.
    """
    from nilearn.glm import compute_contrast as _nl_compute_contrast

    con = _nl_compute_contrast(labels, run_results, np.asarray(con_vec, dtype=float))
    if statistic == "all":
        return {
            "beta": con.effect_size().ravel(),
            "t": con.stat().ravel(),
            "z": con.z_score().ravel(),
            "p": con.p_value().ravel(),
            "se": np.sqrt(np.abs(con.effect_variance().ravel())),
        }
    getters = {
        "t": con.stat,
        "z": con.z_score,
        "p": con.p_value,
        "beta": con.effect_size,
        "effect_size": con.effect_size,
    }
    return getters[statistic]().ravel()


def compute_contrasts(bd, contrasts, statistic="t"):
    """Compute contrasts from a fitted GLM.

    Uses nilearn's functional ``compute_contrast`` on the fitted
    ``(labels_, results_)`` so t-statistics are computed with the full per-voxel
    parameter covariance (correct for OLS and AR) — a linear combination of
    stored betas cannot do this for multi-regressor contrasts (it would ignore
    off-diagonal covariance and produce an effect-size map, not a t-map).
    Contrast maps stay in masked-array space; no unmasking to a Nifti.

    Must be called after ``.fit(model='glm', X=design_matrix)`` has been run.

    **Contrast forms.** A string names columns with optional coefficients
    (``"conditionA - conditionB"``, ``"2*conditionA - conditionB - conditionC"``,
    ``"0.5*A + 0.5*B"``; names are case-sensitive and must match the design
    exactly). An array-like is a numeric contrast vector with one weight per
    regressor (``[1, -1, 0, 0]``). A dict ``{name: contrast}`` evaluates several
    contrasts at once.

    **Statistics.** ``"t"`` (default) is the t-statistic map for thresholding /
    single-subject inference; ``"z"`` the z-score map; ``"p"`` the p-value map;
    ``"beta"`` / ``"effect_size"`` the effect-size (β) map to feed into a
    second-level (group) analysis; ``"all"`` returns every view for one fit — a
    dict ``{"beta", "t", "z", "p", "se"}`` of `BrainData` maps — so group-level
    code never has to recompute beta separately.

    Contrast p-values are **one-sided** (the nilearn/SPM directional-contrast
    convention: a contrast tests "A > B"; flip the contrast for the other
    direction). This is the documented exception to the library's two-tailed
    default.

    Args:
        bd (BrainData): Data fitted with ``model='glm'``.
        contrasts (str | array-like | dict): One contrast (string or numeric
            vector) or a ``{name: contrast}`` dict of several; see above.
        statistic (str): ``"t"`` (default), ``"z"``, ``"p"``, ``"beta"`` /
            ``"effect_size"``, or ``"all"``; see above.

    Returns:
        BrainData | dict: A single `BrainData` for one contrast with a scalar
            ``statistic``; a dict keyed ``"beta"``/``"t"``/``"z"``/``"p"``/``"se"``
            for one contrast with ``statistic="all"``; ``{name: BrainData}`` for
            a dict of contrasts with a scalar ``statistic``; and a nested
            ``{name: {"beta", "t", "z", "p", "se"}}`` for a dict of contrasts
            with ``statistic="all"``.

    Raises:
        RuntimeError: if ``.fit(model='glm')`` has not been run.
        ValueError: if the contrast vector length or a column name is invalid,
            or if ``statistic`` is not one of the supported values.

    Examples:
        ```python
        data.fit(model="glm", X=dm)

        # Single-subject t-map, ready to threshold
        tmap = data.compute_contrasts("conditionA - conditionB")

        # Effect-size map for use as input to a group-level analysis
        beta = data.compute_contrasts("conditionA - conditionB", statistic="beta")

        # Everything at once: threshold on res["t"], feed the group on res["beta"]
        res = data.compute_contrasts("conditionA - conditionB", statistic="all")
        res["t"].plot(threshold=3.09)
        group_effects.append(res["beta"])
        ```

    Note:
        For group analysis, stack per-subject effect-size maps
        (``statistic="beta"`` or ``res["beta"]`` from ``statistic="all"``) and
        run a second-level test (e.g. ``BrainData.ttest``). Mixing first-level
        t-maps into a group one-sample test conflates effect magnitude with
        precision.
    """

    if not hasattr(bd, "glm_betas"):
        raise RuntimeError(
            "Must run .fit(model='glm', X=design_matrix) before computing contrasts"
        )
    if not hasattr(bd, "model_") or bd.model_ is None:
        raise RuntimeError(
            "BrainData has glm_* results but no model_ attached; refit with "
            ".fit(model='glm', X=design_matrix) to enable compute_contrasts."
        )

    if statistic not in _CONTRAST_OUTPUT_TYPES:
        raise ValueError(
            f"statistic must be one of {sorted(_CONTRAST_OUTPUT_TYPES)}; "
            f"got {statistic!r}"
        )
    want_all = statistic == "all"

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
    # Read the fitted run_glm results once; contrasts are computed as arrays via
    # nilearn's functional compute_contrast (no Nifti round-trip). Single run.
    labels = bd.model_.glm_.labels_[0]
    run_results = bd.model_.glm_.results_[0]
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

        vals = _functional_contrast(labels, run_results, contrast_vector, statistic)
        if want_all:
            results[name] = {
                key: _result_from_array(bd, arr, rows="clear")
                for key, arr in vals.items()
            }
        else:
            results[name] = _result_from_array(bd, vals, rows="clear")

    if single_contrast:
        return results["contrast"]
    return results


def parse_contrast_string(bd, contrast_str):
    """Parse a contrast string into a numeric contrast vector.

    Args:
        bd (BrainData): Data with a `design_matrix` from a prior GLM fit.
        contrast_str (str): Contrast string like `"A - B"` or `"2*A - B - C"`.

    Returns:
        np.ndarray: Numeric contrast vector, one weight per design column.

    Raises:
        RuntimeError: If no design matrix is attached (`fit()` not called).
        ValueError: If a column name is not in the design matrix.
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
