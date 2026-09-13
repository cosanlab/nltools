"""Model fitting, contrasts, and statistical tests for `BrainData`.

GLM and ridge fitting (with cross-validation), contrast computation, one- and
two-sample t-tests, and the design-matrix diagnostics `fit` runs before a GLM.
`BrainData` methods delegate here.
"""

import dataclasses
import warnings
from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from nltools.utils import DesignMatrixWarning, _find_stack_level
from .utils import _clear_fit_state, _copy_for_fit, _is_default, _result_from_array


#: Which estimator each model-specific `BrainData.fit` option belongs to.
#: `BrainData.fit` exposes both estimators' options on one signature, so an
#: option supplied for the estimator `model=` did not select is rejected
#: rather than silently ignored.
_ESTIMATOR_OPTION_OWNERS = {
    "ridge_alpha": "ridge",
    "ridge_cv": "ridge",
    "ridge_search_iterations": "ridge",
    "ridge_dirichlet_concentration": "ridge",
    "ridge_device": "ridge",
    "ridge_memory_budget_gb": "ridge",
    "ridge_per_target_alpha": "ridge",
    "ridge_prefer_conservative_alpha": "ridge",
    "ridge_progress_bar": "ridge",
    "glm_noise_model": "glm",
    "glm_bins": "glm",
    "glm_n_jobs": "glm",
}


def _check_unselected_estimator_options(model, supplied):
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


def _warn_if_rank_deficient(X_array):
    """Warn when a design matrix is rank deficient.

    A rank-deficient design has no unique least-squares solution. The GLM still
    returns betas — nilearn falls back to a pseudo-inverse — but the effect is
    split arbitrarily across the linearly dependent columns, so any contrast
    touching that subspace is not interpretable. Silence here is dangerous
    because the failure is invisible in the output: the betas come back finite
    and plausible.

    The warning diagnoses the deficiency (how many columns are dependent, or
    the p > n shape when that is the cause) and offers the fixes: inspect
    with ``DesignMatrix.vif()``, drop redundant columns with
    ``DesignMatrix.clean()`` (order-dependent for correlated pairs), or use
    regularization (``fit(model='ridge')``), whose solution is unique and
    order-invariant even when ``X'X`` is singular.

    We warn rather than raise because over-parameterized designs can still have
    estimable contrasts, and because raising would break pipelines currently
    relying (silently) on the pseudo-inverse.

    Args:
        X_array (np.ndarray): Design matrix as a 2-D array.
    """
    if X_array.ndim != 2 or X_array.shape[1] < 2:
        return

    finite = X_array[np.isfinite(X_array).all(axis=1)]
    if finite.shape[0] == 0:
        # Nothing to assess; the fit itself will fail loudly on the NaNs.
        return

    n_cols = X_array.shape[1]
    rank = int(np.linalg.matrix_rank(finite))
    if rank >= n_cols:
        return

    if finite.shape[0] < n_cols:
        # More regressors than (finite) timepoints: deficient by construction,
        # no matter what the columns contain.
        diagnosis = (
            f"the design has more columns ({n_cols}) than usable rows "
            f"({finite.shape[0]}), so it cannot be full rank"
        )
    else:
        diagnosis = f"{n_cols - rank} column(s) are linear combinations of the others"
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
        DesignMatrixWarning,
        stacklevel=_find_stack_level(),
    )


def _fit(
    bd,
    model="glm",
    *,
    X=None,
    ridge_alpha=1.0,
    ridge_cv=None,
    ridge_search_iterations=100,
    ridge_dirichlet_concentration=(0.1, 1.0),
    ridge_device="cpu",
    ridge_memory_budget_gb=None,
    ridge_per_target_alpha=True,
    ridge_prefer_conservative_alpha=False,
    ridge_progress_bar=False,
    glm_noise_model="ols",
    glm_bins=100,
    glm_n_jobs=1,
    inplace=True,
    random_state=None,
):
    """Fit a model to brain imaging data.

    `bd.data` is always the response. The fit is recorded on the returned
    `BrainData` as `model`, a frozen `FitResult`, for later use with `predict`
    and, for a GLM, `compute_contrasts`.

    For `model='glm'` the design is diagnosed before estimation, as a warning
    only — nothing is ever dropped, modified, or raised on. A rank-deficient
    design fires `DesignMatrixWarning`, which has its own category so
    it can be silenced surgically with `warnings.filterwarnings`.

    The facade does not preprocess the response. Compose `scale()` and
    `standardize()` before `fit` when you want them, so the fitted object's
    data, predictions, residuals, and coefficients stay in the response space
    you supplied.

    Every model-specific option carries a `glm_` or `ridge_` prefix that names
    the estimator it configures; `random_state` keeps its bare name because
    both estimators accept it. A non-default option belonging to the estimator
    `model` did not select raises `ValueError`.

    **What the fit records.** `bd.model` is a `FitResult` holding `kind`,
    the `betas`, `predicted`, `residual` and `r2` maps, the `design`, and —
    for ridge — the selected `alpha` map and resolved `cv`.

    Args:
        bd (BrainData): Data whose `.data` is the regression target.
        model (str): `'glm'` (default) or `'ridge'`.
        X (DesignMatrix | array-like | Mapping): Design matrix for a GLM — a
            precomputed `DesignMatrix` with `n_samples` matching `bd` — or a
            feature matrix for ridge. For banded ridge, a mapping of
            feature-space names to matrices.
        ridge_alpha (float | Sequence[float]): Ridge only. A positive scalar
            fits a fixed alpha and requires `ridge_cv=None`; a sequence selects
            an alpha by cross-validation and requires `ridge_cv`. Default 1.0.
        ridge_cv (int | CV splitter | None): Ridge only. An int is the number
            of unshuffled k-fold splits; an sklearn splitter is used as given.
            Default None.
        ridge_search_iterations (int): Ridge only, banded. Number of sampled
            feature-space weight vectors. Default 100.
        ridge_dirichlet_concentration (float | Sequence[float]): Ridge only,
            banded. Concentration of the Dirichlet distribution the candidate
            weights are drawn from. Default `(0.1, 1.0)`.
        ridge_device (str): Ridge only. `'cpu'` (default) or `'gpu'` (PyTorch
            on CUDA/MPS, or an error when neither is available).
        ridge_memory_budget_gb (float | None): Ridge only. Working-memory
            budget in GB for the solver's internal batching. None (default)
            measures the selected device.
        ridge_per_target_alpha (bool): Ridge only. If True (default), select a
            separate best alpha per voxel; if False, one shared alpha.
        ridge_prefer_conservative_alpha (bool): Ridge only. If True, select the
            largest alpha within one standard deviation of the best score.
            Default False.
        ridge_progress_bar (bool): Ridge only. Display a progress bar over the
            banded search. Default False.
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

    Returns:
        BrainData: `bd` itself when `inplace=True`; otherwise an independently
            owned fitted copy.

    Raises:
        TypeError: If `model` is unknown, `X` is missing, or `model='glm'` gets
            a design that is not a `DesignMatrix`.
        ValueError: If `X` and `bd` disagree on sample count, or a non-default
            option belongs to the unselected estimator.

    Examples:
        ```python
        # inplace=True (default): the fit is recorded on brain_data
        brain_data.fit(model='ridge', ridge_alpha=1.0, X=features)
        weights = brain_data.model.betas

        # inplace=False: fit a copy; brain_data remains completely unchanged
        fitted = brain_data.fit(model='glm', X=design, inplace=False)
        effect = fitted.compute_contrasts('conditionA - conditionB')
        ```
    """
    from nltools.data.designmatrix import DesignMatrix
    from nltools.models import _Glm, _Ridge

    if model not in ("glm", "ridge"):
        raise TypeError("supported models are 'glm' (default) and 'ridge'")
    if X is None:
        raise TypeError("X must be provided")

    _check_unselected_estimator_options(
        model,
        [
            name
            for name, value, default in (
                ("ridge_alpha", ridge_alpha, 1.0),
                ("ridge_cv", ridge_cv, None),
                ("ridge_search_iterations", ridge_search_iterations, 100),
                (
                    "ridge_dirichlet_concentration",
                    ridge_dirichlet_concentration,
                    (0.1, 1.0),
                ),
                ("ridge_device", ridge_device, "cpu"),
                ("ridge_memory_budget_gb", ridge_memory_budget_gb, None),
                ("ridge_per_target_alpha", ridge_per_target_alpha, True),
                (
                    "ridge_prefer_conservative_alpha",
                    ridge_prefer_conservative_alpha,
                    False,
                ),
                ("ridge_progress_bar", ridge_progress_bar, False),
                ("glm_noise_model", glm_noise_model, "ols"),
                ("glm_bins", glm_bins, 100),
                ("glm_n_jobs", glm_n_jobs, 1),
            )
            if not _is_default(value, default)
        ],
    )

    if model == "glm":
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
        _warn_if_rank_deficient(X_array)
    elif isinstance(X, Mapping):
        # Banded ridge: one named feature space per entry.
        X_model = {name: np.asarray(space) for name, space in X.items()}
        for name, space in X_model.items():
            if space.ndim != 2 or space.shape[0] != bd.shape[0]:
                raise ValueError(
                    f"feature space {name!r} has shape {space.shape}, but brain "
                    f"data has {bd.shape[0]} samples. number of samples must match."
                )
    else:
        X_model = np.asarray(X)
        if X_model.ndim != 2 or X_model.shape[0] != bd.shape[0]:
            raise ValueError(
                f"X has shape {X_model.shape}, but brain data has "
                f"{bd.shape[0]} samples. number of samples must match."
            )

    target = bd if inplace else _copy_for_fit(bd)
    if inplace:
        _clear_fit_state(target)

    if model == "glm":
        _fit_glm(
            target,
            X_model,
            _Glm(
                noise_model=glm_noise_model,
                bins=glm_bins,
                n_jobs=glm_n_jobs,
                random_state=random_state,
            ),
        )
        return target

    # Prefix translation is the whole job here: every `ridge_*` facade keyword
    # maps onto the identically-named `_Ridge` constructor argument.
    estimator = _Ridge(
        alpha=ridge_alpha,
        cv=ridge_cv,
        search_iterations=ridge_search_iterations,
        dirichlet_concentration=ridge_dirichlet_concentration,
        device=ridge_device,
        memory_budget_gb=ridge_memory_budget_gb,
        per_target_alpha=ridge_per_target_alpha,
        prefer_conservative_alpha=ridge_prefer_conservative_alpha,
        random_state=random_state,
        progress_bar=ridge_progress_bar,
    )
    _fit_ridge(target, X_model, estimator)
    return target


def _fit_ridge(bd, X, model):
    """Fit `model` on `X` and record the result on `bd.model`.

    Alpha selection and the banded search belong to `_Ridge`; this layer only
    shapes the results the facade owns. The record is set only once the fit
    succeeds, so a failed fit never leaves an unfitted estimator behind.

    Args:
        bd (BrainData): Data whose `.data` is the response.
        X (np.ndarray | Mapping[str, np.ndarray]): Training features.
        model (_Ridge): An unfitted estimator.
    """
    model.fit(X, bd.data)
    bd.model = _ridge_fit_result(bd, X, model)


def _owned_design(X):
    """Copy the training design so the record never aliases the caller's input.

    The maps `_result_from_array` builds are independently owned already; the
    design comes straight from the caller, so this is the one payload the
    producer has to detach.
    """
    from nltools.data.designmatrix import DesignMatrix

    if isinstance(X, DesignMatrix):
        return DesignMatrix(X)
    if isinstance(X, Mapping):
        return {name: np.array(space, copy=True) for name, space in X.items()}
    return np.array(X, copy=True)


def _ridge_fit_result(bd, X, model):
    """Build the `FitResult` a ridge fit leaves on `BrainData.model`.

    Ridge fits no intercept, so the residual is the response minus the fitted
    values with nothing else subtracted. The selected penalty is broadcast to
    one value per voxel whether it was chosen per voxel or shared, so `alpha`
    is one map in every case.
    """
    from nltools.data.results import FitResult

    n_voxels = bd.data.shape[-1]
    # _Ridge.coef_ is (n_features, n_voxels); no transpose.
    fitted = np.asarray(model.predict(X))
    alpha = np.broadcast_to(np.asarray(model.alpha_, dtype=float), (n_voxels,))
    return FitResult(
        kind="ridge",
        betas=_result_from_array(bd, np.asarray(model.coef_), rows="clear"),
        predicted=_result_from_array(bd, fitted, rows="preserve"),
        residual=_result_from_array(bd, bd.data - fitted, rows="preserve"),
        r2=_result_from_array(
            bd, np.asarray(model.score(X, bd.data)).reshape(1, -1), rows="clear"
        ),
        design=_owned_design(X),
        alpha=_result_from_array(bd, alpha.reshape(1, -1), rows="clear"),
        cv=deepcopy(model._resolved_cv()),
        _estimator=model,
    )


def _fit_glm(bd, X, model):
    """Fit `model` on `X` and record the result on `bd.model`.

    Numerical fitting, coefficients, predictions, residuals, and R-squared all
    come from `_Glm`; this layer only wraps them as `BrainData` maps. The
    record is set only once the fit succeeds, so a failed fit never leaves an
    unfitted estimator behind.

    Args:
        bd (BrainData): Data whose `.data` is the response.
        X (DesignMatrix): The training design.
        model (_Glm): An unfitted estimator.
    """
    model.fit(X, bd.data)
    bd.model = _glm_fit_result(bd, X, model)


def _glm_fit_result(bd, X, model):
    """Build the `FitResult` a GLM fit leaves on `BrainData.model`.

    `betas` carries one map per design column, in column order; `predicted`
    and `residual` keep one row per training observation with its row
    metadata. `r2` carries Nilearn's whitened variance-ratio semantics:
    conventional R-squared for an OLS fit with an intercept, a pseudo-R-squared
    in the whitened space for an autoregressive one.
    """
    from nltools.data.results import FitResult

    return FitResult(
        kind="glm",
        betas=_result_from_array(bd, np.asarray(model.coef_), rows="clear"),
        predicted=_result_from_array(bd, np.asarray(model.predicted_), rows="preserve"),
        residual=_result_from_array(bd, np.asarray(model.residuals_), rows="preserve"),
        r2=_result_from_array(bd, np.asarray(model.r2_).reshape(1, -1), rows="clear"),
        design=_owned_design(X),
        _estimator=model,
    )


def _fit_result_from_storage(bd, stored):
    """Rebuild the `FitResult` an HDF5 file stored, with no estimator behind it.

    The file keeps the maps, the design and the kind, so the record answers
    every read a live fit answers. `_estimator` stays None, which is what
    `compute_contrasts(inference=True)`, `predict(X=...)` and `bootstrap` check
    before refusing.

    Args:
        bd (BrainData): The object being loaded, already carrying its mask and
            row metadata.
        stored (dict): What `nltools.io.h5._read_fit_record` returned.

    Returns:
        FitResult: The restored record.
    """
    from nltools.io.h5 import _FIT_MAP_ROWS

    from nltools.data.results import FitResult

    maps = {
        name: _result_from_array(bd, array, rows=_FIT_MAP_ROWS[name])
        for name, array in stored["maps"].items()
    }
    return FitResult(
        kind=str(stored["kind"]),
        betas=maps["betas"],
        predicted=maps["predicted"],
        residual=maps["residual"],
        r2=maps["r2"],
        design=stored["design"],
        alpha=maps.get("alpha"),
    )


def _contrasts_from_betas(bd, fit, contrasts, *, inference):
    """Compute effect maps from a restored fit's stored betas.

    A contrast effect is a linear combination of the coefficients, so it needs
    the betas and the design's column names and nothing else. Every other
    statistic needs the parameter covariance, which only the fitted model has.

    Raises:
        RuntimeError: If `inference=True`, with the one-line refit.
    """
    if inference:
        raise RuntimeError(
            "compute_contrasts(inference=True) needs the fitted GLM itself, "
            "which an HDF5 file does not store — it keeps the maps, the design "
            "and the kind. Refit with "
            "data.fit(model='glm', X=data.model.design), then ask again."
        )
    columns = tuple(fit.design.columns)
    if isinstance(contrasts, Mapping):
        return {
            name: _effect_from_betas(bd, fit, contrast, columns)
            for name, contrast in contrasts.items()
        }
    return _effect_from_betas(bd, fit, contrasts, columns)


def _effect_from_betas(bd, fit, contrast, columns):
    """Wrap one contrast's `weights @ betas` as an independently owned map."""
    from nltools.models.glm import _resolve_contrast

    weights = _resolve_contrast(contrast, columns)
    return _result_from_array(bd, weights @ fit.betas.data, rows="clear")


def _ttest(
    bd,
    *,
    popmean=0.0,
    permutation=False,
    n_permute=5000,
    tail=2,
    return_null=False,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
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
        progress_bar (bool): If True, show a progress bar. Default False.

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
        progress_bar=progress_bar,
    )
    results = {
        key: _result_from_array(bd, stats[key], rows="clear")
        for key in ("mean", "t", "z", "p")
    }
    if "null_dist" in stats:
        results["null_dist"] = stats["null_dist"]
    return results


def _compute_contrasts(bd, contrasts, *, inference=False):
    """Compute contrasts on a fitted GLM.

    Pure forwarding: the fitted `_Glm` parses every contrast definition and
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
        ValueError: If the fitted model is not a `_Glm`.

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
    from .utils import _NO_FIT_EXPLANATION

    fit = bd.model
    if fit is None:
        raise RuntimeError(
            "compute_contrasts requires a fitted GLM. Run "
            f".fit(model='glm', X=design_matrix) first. {_NO_FIT_EXPLANATION}"
        )
    if fit.kind != "glm":
        raise ValueError(
            f"compute_contrasts requires a GLM fit, but this BrainData holds a "
            f"{fit.kind} fit. Refit with model='glm'."
        )
    if fit._estimator is None:
        return _contrasts_from_betas(bd, fit, contrasts, inference=inference)

    computed = fit._estimator.compute_contrasts(contrasts, inference=inference)
    if isinstance(computed, dict):
        return {name: _contrast_maps(bd, value) for name, value in computed.items()}
    return _contrast_maps(bd, computed)


def _contrast_maps(bd, computed):
    """Wrap one `_Glm` contrast return as independently owned `BrainData` maps.

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
