"""Model fitting, contrasts, and statistical tests for `BrainData`.

GLM and ridge fitting (with cross-validation), contrast computation, one- and
two-sample t-tests, and the design-matrix diagnostics `fit` runs before a GLM.
`BrainData` methods delegate here.
"""

import dataclasses
import warnings
from copy import deepcopy

import numpy as np

from nltools.utils import find_stack_level
from .utils import _clear_fit_state, _copy_for_fit, _result_from_array


#: Which estimator each model-specific `BrainData.fit` option belongs to.
#: `BrainData.fit` exposes both estimators' options on one signature, so an
#: option supplied for the estimator `model=` did not select is rejected
#: rather than silently ignored.
_ESTIMATOR_OPTION_OWNERS = {
    "cv": "ridge",
    "device": "ridge",
    "per_target_alpha": "ridge",
    "progress_bar": "ridge",
    "glm_noise_model": "glm",
    "glm_bins": "glm",
    "glm_n_jobs": "glm",
}


def check_unselected_estimator_options(model, supplied):
    """Reject `fit` options belonging to the estimator `model` did not select.

    Args:
        model (str): The selected model, `'glm'` or `'ridge'`.
        supplied (Iterable[str]): Names of the model-specific options the
            caller gave a non-default value.

    Raises:
        ValueError: If any supplied name belongs to the unselected estimator.
    """
    wrong = sorted(
        name for name in supplied if _ESTIMATOR_OPTION_OWNERS.get(name, model) != model
    )
    if wrong:
        raise ValueError(
            f"model={model!r} does not accept {wrong}: those options belong to "
            f"the unselected estimator."
        )


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


def fit(  # nosemgrep: kwargs-internal-forwarding  # ridge-only passthrough pending the ridge_* signature (Kata e5y6)
    bd,
    model="glm",
    *,
    X=None,
    cv=None,
    device="cpu",
    per_target_alpha=True,
    glm_noise_model="ols",
    glm_bins=100,
    glm_n_jobs=1,
    inplace=True,
    random_state=None,
    progress_bar=False,
    **kwargs,
):
    """Fit a model to brain imaging data.

    `bd.data` is always the response. The estimator and its results are stored
    on the returned `BrainData` for later use with `predict` and, for a GLM,
    `compute_contrasts`.

    For `model='glm'` the design is diagnosed before estimation, as warnings
    only — nothing is ever dropped, modified, or raised on. An exactly
    rank-deficient design fires `RankDeficientDesignWarning`; a full-rank but
    near-collinear design (a column pair with |r| >= 0.95, or a
    column-standardized condition number above 30) fires
    `NearCollinearDesignWarning` instead — never both. Each has its own
    category so it can be silenced surgically with `warnings.filterwarnings`.

    The facade does not preprocess the response. Compose `scale()` and
    `standardize()` before `fit` when you want them, so the fitted object's
    data, predictions, residuals, and coefficients stay in the response space
    you supplied.

    GLM options carry a `glm_` prefix. The ridge options (`cv`, `device`,
    `per_target_alpha`, `progress_bar`, and additional `Ridge` constructor
    arguments) keep their bare names for now, and `random_state` keeps its
    bare name because both estimators use it. A non-default option belonging
    to the estimator `model` did not select raises `ValueError`.

    **Results stored on the returned `BrainData`:**

    - `model_` — the fitted `Ridge` or `Glm`.
    - GLM: `glm_betas`, `glm_residual`, `glm_predicted`, `glm_r2`.
    - Ridge: `ridge_weights`, `ridge_fitted_values`, `ridge_scores`.

    Args:
        bd (BrainData): Data whose `.data` is the regression target.
        model (str): `'glm'` (default) or `'ridge'`.
        X (DesignMatrix | array-like | Mapping): Design matrix for a GLM — a
            precomputed `DesignMatrix` with `n_samples` matching `bd` — or a
            feature matrix for ridge. For banded ridge, a mapping of
            feature-space names to matrices.
        cv (int | CV splitter | None): Ridge only. Cross-validation
            specification. An int is the number of unshuffled k-fold splits; an
            sklearn splitter is used as given; None (default) fits a fixed
            alpha.
        device (str): Ridge only. Compute device for the ridge solve: `'cpu'`
            (default, NumPy) or `'gpu'` (PyTorch on CUDA/MPS, or an error when
            neither is available).
        per_target_alpha (bool): Ridge only. If True (default), select a
            separate best alpha per voxel; if False, one shared alpha.
        glm_noise_model (str): GLM only. `'ols'` (default) or `'arN'` for
            Nilearn's autoregressive model of order N (`'ar1'`, `'ar2'`, ...).
        glm_bins (int): GLM only. Nilearn's discretization of the estimated AR
            coefficients. Default 100.
        glm_n_jobs (int): GLM only. CPUs Nilearn uses to fit autoregressive
            groups in parallel. The default OLS fit does not use this path.
            Default 1.
        inplace (bool): If True (default), mutate `bd` and return it. If False,
            fit and return an independent `BrainData` copy while leaving every
            part of `bd` untouched.
        random_state (int | None): Seed shared by both estimators. Default None.
        progress_bar (bool): Ridge only. Display a progress bar during fitting.
            Default False.
        **kwargs (dict): Ridge only. Additional `Ridge` constructor arguments
            (`alpha`, `search_iterations`, ...).

    Returns:
        BrainData: `bd` itself when `inplace=True`; otherwise an independently
            owned fitted copy.

    Raises:
        TypeError: If `model` is unknown, `X` is missing, `model='glm'` gets a
            design that is not a `DesignMatrix`, or `model='glm'` gets an
            unknown keyword.
        ValueError: If `X` and `bd` disagree on sample count, or a non-default
            option belongs to the unselected estimator.

    Examples:
        ```python
        # inplace=True (default): results are stored on brain_data
        brain_data.fit(model='ridge', alpha=1.0, X=features)
        weights = brain_data.ridge_weights

        # inplace=False: fit a copy; brain_data remains completely unchanged
        fitted = brain_data.fit(model='glm', X=design, inplace=False)
        effect = fitted.compute_contrasts('conditionA - conditionB')
        ```
    """
    from nltools.data.designmatrix import DesignMatrix
    from nltools.models import Glm, Ridge

    if model not in ("glm", "ridge"):
        raise TypeError("supported models are 'glm' (default) and 'ridge'")
    if X is None:
        raise TypeError("X must be provided")

    check_unselected_estimator_options(
        model,
        [
            name
            for name, value, default in (
                ("cv", cv, None),
                ("device", device, "cpu"),
                ("per_target_alpha", per_target_alpha, True),
                ("progress_bar", progress_bar, False),
                ("glm_noise_model", glm_noise_model, "ols"),
                ("glm_bins", glm_bins, 100),
                ("glm_n_jobs", glm_n_jobs, 1),
            )
            if value != default
        ],
    )

    if model == "glm":
        if kwargs:
            raise TypeError(
                f"fit(model='glm') got unexpected keyword argument(s) "
                f"{sorted(kwargs)}. The GLM takes X, glm_noise_model, glm_bins, "
                f"glm_n_jobs, inplace, and random_state."
            )
        if not isinstance(X, DesignMatrix):
            raise TypeError(
                f"fit(model='glm') requires a precomputed DesignMatrix for X, "
                f"got {type(X).__name__}. Build one with "
                f"`DesignMatrix(...)` before fitting."
            )
        X_model = X
        X_array = X.to_numpy()
        if X_array.shape[0] != bd.shape[0]:
            raise ValueError(
                f"X has {X_array.shape[0]} samples, but brain data has "
                f"{bd.shape[0]} samples. number of samples must match."
            )
        if not _warn_if_rank_deficient(X_array, X_model):
            _warn_if_near_collinear(X_array, X_model)
    elif isinstance(X, list):
        # Banded ridge: keep as list, validate each element
        X_model = X
        for i, Xi in enumerate(X):
            Xi_array = np.asarray(Xi)
            if Xi_array.shape[0] != bd.shape[0]:
                raise ValueError(
                    f"X[{i}] has {Xi_array.shape[0]} samples, but brain data "
                    f"has {bd.shape[0]} samples. number of samples must match."
                )
    else:
        X_model = np.asarray(X)
        if X_model.shape[0] != bd.shape[0]:
            raise ValueError(
                f"X has {X_model.shape[0]} samples, but brain data has "
                f"{bd.shape[0]} samples. number of samples must match."
            )

    target = bd if inplace else _copy_for_fit(bd)
    if inplace:
        _clear_fit_state(target)

    if model == "glm":
        fit_glm(
            target,
            X_model,
            Glm(
                noise_model=glm_noise_model,
                bins=glm_bins,
                n_jobs=glm_n_jobs,
                random_state=random_state,
            ),
        )
        return target

    # Device selection is a first-class facade kwarg (`device=`), not a
    # passthrough. Reject the retired algorithm-layer aliases loudly rather
    # than letting them silently reach Ridge (which no longer accepts them).
    for banned in ("backend", "parallel"):
        if banned in kwargs:
            raise TypeError(
                f"`{banned}=` is not a valid ridge kwarg; use "
                f"`device='cpu'|'gpu'|'auto'` to select the compute device."
            )
    if isinstance(X_model, list):
        target.X_ = [np.array(part, copy=True) for part in X_model]
    elif hasattr(X_model, "copy"):
        target.X_ = X_model.copy()
    else:
        target.X_ = deepcopy(X_model)

    ridge_kwargs = kwargs.copy()
    # Explicit-signature kwargs win over **kwargs forwarding so calls
    # like bd.fit(model='ridge', per_target_alpha=False, ...) reach Ridge.
    ridge_kwargs.setdefault("progress_bar", progress_bar)
    ridge_kwargs.setdefault("device", device)
    ridge_kwargs.setdefault("per_target_alpha", per_target_alpha)
    ridge_kwargs.setdefault("random_state", random_state)
    if cv is not None:
        ridge_kwargs["cv"] = _normalize_cv(cv)
    target.model_ = Ridge(**ridge_kwargs)
    fit_ridge(target, target.X_)
    return target


def fit_ridge(bd, X):
    """Fit `bd.model_` and attach the ridge results to `bd`.

    Alpha selection and the banded search belong to `Ridge`; this layer only
    stores the results the facade owns.

    Args:
        bd (BrainData): Data with `bd.model_` already set to a `Ridge` instance.
        X (np.ndarray | Mapping[str, np.ndarray]): Training features.

    Note:
        Sets `ridge_weights`, `ridge_fitted_values`, and `ridge_scores` on `bd`.
    """
    bd.model_.fit(X, bd.data)
    _populate_ridge_attributes(bd, X)


def _normalize_cv(cv):
    """Validate and normalize a cross-validation specification.

    Reject single-use generators and bad cv values; pass through ints and
    splitter objects. `Ridge` traverses the splits more than once, so the
    specification has to be re-iterable.

    Args:
        cv (int | BaseCrossValidator): Fold count or scikit-learn splitter.

    Returns:
        int | BaseCrossValidator: The validated specification.

    Raises:
        TypeError: If `cv` is a single-use split generator.
        ValueError: If `cv` is neither an int fold count nor a splitter.
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


def fit_glm(bd, X, model):
    """Fit `model` on `X` and attach it and the GLM results the facade owns.

    Numerical fitting, coefficients, predictions, residuals, and R-squared all
    come from `Glm`; this layer only wraps them as independently owned
    `BrainData` results. `model_` is attached only once the fit succeeds, so a
    failed fit never leaves an unfitted estimator on `bd`.

    Args:
        bd (BrainData): Data whose `.data` is the response.
        X (DesignMatrix): The training design.
        model (Glm): An unfitted estimator.

    Note:
        Sets `model_`, `glm_betas` (one map per design column), `glm_predicted`
        and `glm_residual` (one row per training observation, row metadata
        retained), and `glm_r2` (one fit-quality map). `glm_r2` carries
        Nilearn's whitened variance-ratio semantics: conventional R-squared for
        an OLS fit with an intercept, a pseudo-R-squared in the whitened space
        for an autoregressive one.
    """
    model.fit(X, bd.data)
    bd.model_ = model
    bd.glm_betas = _result_from_array(
        bd, np.array(model.coef_, copy=True), rows="clear"
    )
    bd.glm_predicted = _result_from_array(
        bd, np.array(model.predicted_, copy=True), rows="preserve"
    )
    bd.glm_residual = _result_from_array(
        bd, np.array(model.residuals_, copy=True), rows="preserve"
    )
    bd.glm_r2 = _result_from_array(
        bd, np.array(model.r2_, copy=True).reshape(1, -1), rows="clear"
    )


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


def compute_contrasts(bd, contrasts, *, inference=False):
    """Compute contrasts on a fitted GLM.

    Pure forwarding: the fitted `Glm` parses every contrast definition and
    computes every number. This layer wraps each per-voxel array as an
    independently owned `BrainData` map with cleared row metadata, because a
    contrast map's leading axis no longer represents training observations.

    Args:
        bd (BrainData): Data fitted with `model='glm'`.
        contrasts (str | array-like | Mapping): One contrast definition — a
            string expression over design column names or a flat numeric weight
            vector — or a mapping of names to those definitions.
        inference (bool): If True, return `ContrastResult` records instead of
            bare effect maps. Default False.

    Returns:
        BrainData | ContrastResult | dict: One effect map, or one
            `ContrastResult` of maps when `inference=True`; a dictionary with
            the same keys for a mapping.

    Raises:
        RuntimeError: If no model has been fitted.
        ValueError: If the fitted model is not a `Glm`.

    Examples:
        ```python
        data.fit(model="glm", X=design)

        # Effect map — the input a second-level model consumes
        effect = data.compute_contrasts("conditionA - conditionB")

        # First-level inference: every statistic in one record
        result = data.compute_contrasts("conditionA - conditionB", inference=True)
        result.statistic.plot(threshold=3.09)
        ```

    Note:
        Contrast p-values are one-sided (the nilearn/SPM directional-contrast
        convention): the contrast tests "A > B", so negate it for the other
        direction. This is the documented exception to the library's two-tailed
        default.
    """
    from nltools.models import Glm

    model = getattr(bd, "model_", None)
    if model is None:
        raise RuntimeError(
            "compute_contrasts requires a fitted GLM. Run "
            ".fit(model='glm', X=design_matrix) first."
        )
    if not isinstance(model, Glm):
        raise ValueError(
            f"compute_contrasts requires a fitted Glm, but this BrainData holds "
            f"a fitted {type(model).__name__}. Refit with model='glm'."
        )

    computed = model.compute_contrasts(contrasts, inference=inference)
    if isinstance(computed, dict):
        return {name: _contrast_maps(bd, value) for name, value in computed.items()}
    return _contrast_maps(bd, computed)


def _contrast_maps(bd, computed):
    """Wrap one `Glm` contrast return as independently owned `BrainData` maps.

    Every per-target statistic becomes its own map; `degrees_of_freedom` stays a
    scalar or an array because it describes the fit, not the voxel axis. The
    payload field names come from `ContrastResult` itself so a field added to
    the record cannot silently go unwrapped here.
    """
    from nltools.models import ContrastResult

    if not isinstance(computed, ContrastResult):
        return _result_from_array(bd, np.array(computed, copy=True), rows="clear")
    payload_fields = [
        field.name
        for field in dataclasses.fields(ContrastResult)
        if field.name != "degrees_of_freedom"
    ]
    degrees_of_freedom = computed.degrees_of_freedom
    if isinstance(degrees_of_freedom, np.ndarray):
        degrees_of_freedom = degrees_of_freedom.copy()
    return ContrastResult(
        **{
            name: _result_from_array(
                bd, np.array(getattr(computed, name), copy=True), rows="clear"
            )
            for name in payload_fields
        },
        degrees_of_freedom=degrees_of_freedom,
    )
