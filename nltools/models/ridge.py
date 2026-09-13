"""Ridge and banded Ridge regression, with the numerics delegated to Himalaya.

`Ridge` owns argument names, validation, named feature-space alignment, device
and memory policy, and fitted-state normalization. Himalaya owns decomposition,
the cross-validation loss, alpha selection, the Dirichlet search, and
coefficient refitting. Nothing here re-implements a solver.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..algorithms.backends import (
    auto_batch_size,
    device_memory_budget,
    resolve_backend,
)
from .validation import _check_is_fitted


#: Constructor defaults for the arguments that only the banded random search
#: consumes. Passing any of them a different value while fitting ordinary Ridge
#: is an error rather than a silent no-op. `random_state` is deliberately absent:
#: `BrainData.fit` forwards one unprefixed `random_state` to whichever estimator
#: it builds (see specs/braindata.md), so rejecting it here would break a
#: documented, universally accepted facade keyword.
_BANDED_ONLY_DEFAULTS = {
    "search_iterations": 100,
    "dirichlet_concentration": (0.1, 1.0),
}

#: nltools `Backend.device` -> Himalaya backend name.
_HIMALAYA_BACKEND_FOR_DEVICE = {
    "cpu": "numpy",
    "cuda": "torch_cuda",
    "mps": "torch_mps",
}

#: Allocation factor covering the intermediates Himalaya holds alongside the
#: dominant working set of one batched item.
_WORKING_SET_OVERHEAD = 5.0


@contextmanager
def _scoped_himalaya_backend(name: str):
    """Set Himalaya's process-global backend for the block and restore it after.

    Himalaya stores its backend in a module-level global, so a fit that changed
    it would leak into unrelated code. This restores the previous backend on
    success and on exception alike.

    Args:
        name (str): Himalaya backend name, e.g. `"numpy"` or `"torch_mps"`.

    Yields:
        None: The block runs with `name` as the active Himalaya backend.
    """
    from himalaya.backend import get_backend, set_backend

    previous = get_backend().name
    set_backend(name, on_error="raise")
    try:
        yield
    finally:
        set_backend(previous, on_error="raise")


def _himalaya_backend_name(backend) -> str:
    """Map a resolved nltools `Backend` to its Himalaya backend name.

    Args:
        backend (Backend): Backend returned by `resolve_backend`.

    Returns:
        str: One of `"numpy"`, `"torch_cuda"`, or `"torch_mps"`.

    Raises:
        RuntimeError: If the backend's device has no Himalaya equivalent.
    """
    try:
        return _HIMALAYA_BACKEND_FOR_DEVICE[backend.device]
    except KeyError:
        raise RuntimeError(
            f"no Himalaya backend for device {backend.device!r}; "
            "use device='cpu' or device='gpu'"
        ) from None


def _batch_sizes(
    backend,
    memory_budget_gb: float | None,
    n_samples: int,
    n_features: int,
    n_targets: int,
    n_alphas: int,
    itemsize: int,
) -> dict[str, int]:
    """Size the batches of a whole cross-validated or banded fit.

    Only the per-item working-set estimates live here; the budget itself and
    the batch arithmetic come from `nltools.algorithms.backends`. The estimates
    follow Himalaya's dominant allocations: decomposition matrices of
    `(n_alphas_batch, n_features, n_samples)`, cross-validated predictions of
    `(n_alphas_batch, n_samples, n_targets_batch)`, and refit weights of
    `(n_features, n_targets_batch_refit)`.

    Returns:
        dict[str, int]: `n_targets_batch`, `n_targets_batch_refit`, and
            `n_alphas_batch`.
    """
    budget_gb = device_memory_budget(
        backend, max_gpu_memory_gb=memory_budget_gb, cap_for_batching=True
    )
    n_alphas_batch, _ = auto_batch_size(
        n_alphas,
        n_features * n_samples * itemsize,
        budget_gb=budget_gb,
        overhead=_WORKING_SET_OVERHEAD,
    )
    n_targets_batch, _ = auto_batch_size(
        n_targets,
        n_alphas_batch * n_samples * itemsize,
        budget_gb=budget_gb,
        overhead=_WORKING_SET_OVERHEAD,
    )
    n_targets_batch_refit, _ = auto_batch_size(
        n_targets,
        n_alphas_batch * n_features * itemsize,
        budget_gb=budget_gb,
        overhead=_WORKING_SET_OVERHEAD,
    )
    return {
        "n_targets_batch": n_targets_batch,
        "n_targets_batch_refit": n_targets_batch_refit,
        "n_alphas_batch": n_alphas_batch,
    }


def _refit_targets_batch(
    backend,
    memory_budget_gb: float | None,
    n_samples: int,
    n_targets: int,
    itemsize: int,
    *,
    per_target_alpha: bool,
    n_features: int,
) -> int:
    """Size the target batch of the fixed-hyperparameter refit.

    With one shared alpha Himalaya reuses a single shrinkage operator, so a
    target costs only its own columns of `Y` and of the weights. With a
    per-target alpha it instead holds an `(n_targets_batch, n_samples,
    n_samples)` block, which dominates everything else.

    Returns:
        int: Target batch size in `[1, n_targets]`.
    """
    budget_gb = device_memory_budget(
        backend, max_gpu_memory_gb=memory_budget_gb, cap_for_batching=True
    )
    if per_target_alpha:
        bytes_per_target = n_samples * n_samples * itemsize
    else:
        bytes_per_target = (n_samples + n_features) * itemsize
    batch, _ = auto_batch_size(
        n_targets,
        bytes_per_target,
        budget_gb=budget_gb,
        overhead=_WORKING_SET_OVERHEAD,
    )
    return batch


def _prepare_feature_space_weights(candidates, dtype) -> np.ndarray:
    """Validate candidate feature-space weights and convert them for Himalaya.

    Candidates must be finite, strictly positive, and sum to one per row before
    conversion. After conversion to `dtype`, weights below that dtype's
    smallest positive normal value are raised to it: that is the smallest
    weight a float32 device can take `sqrt` of and then divide back out without
    destroying the reusable feature buffer. The floor is a numerical boundary
    only — it never makes a zero or negative weight valid.

    The caller's array is never modified.

    Args:
        candidates (array-like): Shape `(n_candidates, n_spaces)` weights on the
            simplex.
        dtype (np.dtype | type): Floating dtype the feature matrices use.

    Returns:
        np.ndarray: A new `(n_candidates, n_spaces)` array of `dtype`.

    Raises:
        ValueError: If the candidates are not 2-D, or any weight is
            non-finite, not strictly positive, or a row does not sum to one.
    """
    values = np.array(candidates, dtype=np.float64, copy=True)
    if values.ndim != 2:
        raise ValueError(
            "feature-space weight candidates must be 2D with shape "
            f"(n_candidates, n_spaces), got {values.ndim}D"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("feature-space weight candidates must all be finite")
    if not np.all(values > 0):
        raise ValueError(
            "feature-space weight candidates must all be strictly positive"
        )
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-6):
        raise ValueError(
            "each feature-space weight candidate must sum to one; "
            f"observed sums between {row_sums.min():.6g} and {row_sums.max():.6g}"
        )
    converted = values.astype(dtype, copy=True)
    tiny = np.finfo(converted.dtype).tiny
    converted[converted < tiny] = tiny
    return converted


def _working_dtype(spaces, y, backend) -> np.dtype:
    """Choose the floating dtype a solve runs in.

    Himalaya solves in the dtype it is handed, so this is the one rule every
    entry point shares. An integer design would make the shrinkage arithmetic
    truncate to zero and return silently wrong (all-zero) coefficients, and MPS
    is a float32-only device that would otherwise downcast float64 inputs with
    a warning on every call.

    Args:
        spaces (Sequence[np.ndarray]): Feature matrices.
        y (np.ndarray): Targets.
        backend (Backend | None): Resolved backend, or None for the CPU.

    Returns:
        np.dtype: `float32` on MPS, otherwise the promoted dtype of the inputs
            with a `float32` floor.
    """
    if backend is not None and getattr(backend, "device", None) == "mps":
        return np.dtype(np.float32)
    dtypes = [np.asarray(space).dtype for space in spaces] + [np.asarray(y).dtype]
    return np.dtype(np.promote_types(np.result_type(*dtypes), np.float32))


@dataclass(frozen=True)
class _ResidentDesign:
    """A concatenated design and its targets, already on Himalaya's backend.

    Bootstrap refits solve thousands of replicates against the same training
    data. Converting once and resampling rows in place keeps the host-to-device
    transfer out of the replicate loop; on the CPU it also avoids re-running the
    concatenation per replicate.

    Attributes:
        design: `(n_samples, n_features)` in `feature_space_names_` order, on
            the backend.
        targets: `(n_samples, n_targets)`, on the backend.
        sizes (tuple[int, ...]): Feature count per space, in the same order.
        dtype (np.dtype): The working dtype both arrays were converted to.
        backend: The resolved nltools `Backend`, or None for the CPU.
        backend_name (str): Himalaya's name for that backend.
    """

    design: Any
    targets: Any
    sizes: tuple[int, ...]
    dtype: np.dtype
    backend: Any
    backend_name: str


def _resident_design(feature_spaces, y, backend=None) -> _ResidentDesign:
    """Convert a design and its targets onto the backend once.

    Args:
        feature_spaces (Sequence[np.ndarray]): One or more `(n_samples,
            n_features_k)` matrices in coefficient order.
        y (np.ndarray): Targets of shape `(n_samples, n_targets)`.
        backend (Backend | None): Resolved backend; None stays on the CPU.

    Returns:
        _ResidentDesign: The converted design, ready for repeated refits.
    """
    spaces = [np.asarray(space) for space in feature_spaces]
    y = np.asarray(y)
    dtype = _working_dtype(spaces, y, backend)
    backend_name = "numpy" if backend is None else _himalaya_backend_name(backend)
    stacked = np.ascontiguousarray(
        spaces[0] if len(spaces) == 1 else np.concatenate(spaces, axis=1), dtype=dtype
    )
    with _scoped_himalaya_backend(backend_name):
        design, targets = _on_active_backend(
            stacked, np.ascontiguousarray(y, dtype=dtype)
        )
    return _ResidentDesign(
        design=design,
        targets=targets,
        sizes=tuple(space.shape[1] for space in spaces),
        dtype=dtype,
        backend=backend,
        backend_name=backend_name,
    )


def _take_rows(array, indices):
    """Select rows of a backend-resident array with host integer indices.

    Args:
        array: A NumPy array or a torch tensor on any device.
        indices (np.ndarray | None): Row indices, or None to take every row.

    Returns:
        The selected rows, on the same backend and device as `array`.
    """
    if indices is None:
        return array
    indices = np.asarray(indices, dtype=np.int64)
    if hasattr(array, "detach"):
        import torch

        return array[torch.as_tensor(indices, device=array.device)]
    return array[indices]


def _take_columns(array, columns):
    """Select columns of a backend-resident array with host integer indices.

    Args:
        array: A NumPy array or a torch tensor on any device.
        columns (np.ndarray): Column indices.

    Returns:
        The selected columns, on the same backend and device as `array`.
    """
    columns = np.asarray(columns, dtype=np.int64)
    if hasattr(array, "detach"):
        import torch

        return array[:, torch.as_tensor(columns, device=array.device)]
    return np.ascontiguousarray(array[:, columns])


def _weight_groups(feature_space_weights, n_targets):
    """Group targets that selected the same feature-space weight vector.

    Sharing one decomposition across such targets is an implementation detail
    that does not change the result.

    Args:
        feature_space_weights (np.ndarray | None): `(n_spaces,)` shared or
            `(n_spaces, n_targets)` per-target weights, or None.
        n_targets (int): Number of targets.

    Returns:
        list[tuple[np.ndarray | None, np.ndarray]]: `(gamma, columns)` pairs;
            `gamma` is None for the unweighted system.
    """
    if feature_space_weights is None:
        return [(None, np.arange(n_targets))]
    gammas = np.asarray(feature_space_weights, dtype=np.float64)
    # Every target sharing one weight vector is the common case (shared weights,
    # or a search that converged on the same row). Skip the sort in np.unique.
    if gammas.ndim == 1:
        return [(gammas, np.arange(n_targets))]
    if bool(np.all(gammas == gammas[:, :1])):
        return [(gammas[:, 0], np.arange(n_targets))]
    _, inverse = np.unique(gammas.T, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    return [
        (
            gammas[:, np.flatnonzero(inverse == label)[0]],
            np.flatnonzero(inverse == label),
        )
        for label in np.unique(inverse)
    ]


def _refit_fixed_hyperparameters(
    feature_spaces,
    y,
    alpha,
    feature_space_weights=None,
    backend=None,
    memory_budget_gb=None,
    n_targets_batch=None,
    row_indices=None,
):
    """Refit Ridge coefficients with the hyperparameters held fixed.

    The one fixed-hyperparameter solve in the package: the final banded refit,
    ordinary fixed-alpha fitting, and every Ridge bootstrap replicate use it, so
    they cannot drift apart numerically. Feature space `k` is scaled by
    `sqrt(gamma[k])` before the solve and the resulting coefficients are scaled
    back, which is equivalent to the per-space penalty `alpha / gamma[k]`.

    Targets that selected the same weight vector share one decomposition. The
    grouping is an implementation detail and does not change the result.

    Args:
        feature_spaces (Sequence[np.ndarray] | _ResidentDesign): One or more
            `(n_samples, n_features_k)` matrices in coefficient order, or a
            design already converted onto the backend by `_resident_design`.
        y (np.ndarray | None): Targets of shape `(n_samples, n_targets)`. None
            when `feature_spaces` is a `_ResidentDesign`, which carries them.
        alpha (float | np.ndarray): Scalar or `(n_targets,)` regularization.
        feature_space_weights (np.ndarray | None): `(n_spaces,)` shared or
            `(n_spaces, n_targets)` weights. None solves the unweighted system.
        backend (Backend | None): Resolved backend; None runs on the CPU.
            Ignored when a `_ResidentDesign` is supplied, which records its own.
        memory_budget_gb (float | None): Budget used to size the target batch
            when `n_targets_batch` is not given.
        n_targets_batch (int | None): Himalaya target batch size. None derives
            one from the memory budget.
        row_indices (np.ndarray | None): Rows to solve on, applied to the design
            and the targets alike. None uses every row. Bootstrap replicates
            pass their resample here so the design is converted only once.

    Returns:
        np.ndarray: Coefficients of shape `(n_features, n_targets)` in the
            original, unscaled feature coordinates, as CPU NumPy.
    """
    from himalaya.ridge import solve_ridge_svd

    if isinstance(feature_spaces, _ResidentDesign):
        resident = feature_spaces
    else:
        resident = _resident_design(feature_spaces, y, backend)

    dtype = resident.dtype
    sizes = resident.sizes
    n_features = sum(sizes)
    n_targets = resident.targets.shape[1]
    n_samples = (
        len(row_indices) if row_indices is not None else resident.design.shape[0]
    )

    alphas = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (n_targets,))
    coef = np.zeros((n_features, n_targets), dtype=np.float64)

    with _scoped_himalaya_backend(resident.backend_name):
        stacked = _take_rows(resident.design, row_indices)
        all_targets = _take_rows(resident.targets, row_indices)

        for gamma, columns in _weight_groups(feature_space_weights, n_targets):
            if gamma is None:
                design = stacked
                scale = None
            else:
                scale = np.concatenate(
                    [np.full(size, np.sqrt(g)) for size, g in zip(sizes, gamma)]
                ).astype(dtype)
                design = stacked * _on_active_backend(scale)[0]
            # A shared alpha needs one shrinkage vector; a per-target alpha
            # makes Himalaya hold an (n_targets_batch, n_samples, n_samples)
            # block instead, so the two paths get different batch estimates.
            group_alphas = alphas[columns]
            shared_alpha = bool(np.all(group_alphas == group_alphas[0]))
            batch = n_targets_batch
            if batch is None:
                batch = _refit_targets_batch(
                    resident.backend,
                    memory_budget_gb,
                    n_samples,
                    len(columns),
                    dtype.itemsize,
                    per_target_alpha=not shared_alpha,
                    n_features=n_features,
                )
            targets = (
                all_targets
                if len(columns) == n_targets
                else _take_columns(all_targets, columns)
            )
            solved = solve_ridge_svd(
                design,
                targets,
                alpha=dtype.type(group_alphas[0])
                if shared_alpha
                else group_alphas.astype(dtype),
                fit_intercept=False,
                n_targets_batch=batch,
                warn=False,
            )
            solved = np.asarray(_to_cpu_numpy(solved), dtype=np.float64)
            if scale is not None:
                solved = solved * scale[:, None]
            coef[:, columns] = solved

    return coef


def _on_active_backend(*arrays):
    """Move arrays onto Himalaya's active backend and device.

    Some Himalaya solvers build scratch arrays with `ones_like(X)`, which
    assumes `X` already lives on the active backend. Converting up front keeps
    the GPU backends usable with NumPy inputs.

    Args:
        *arrays: NumPy arrays to convert.

    Returns:
        list: The arrays as the active backend's array type.
    """
    from himalaya.backend import get_backend

    backend = get_backend()
    return [backend.asarray(array) for array in arrays]


def _to_cpu_numpy(array) -> np.ndarray:
    """Return `array` as CPU NumPy, whatever backend produced it.

    Args:
        array: A NumPy array or a torch tensor from any device.

    Returns:
        np.ndarray: A NumPy view or copy on the host.
    """
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _snap_to_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Round selected alphas back onto the candidate grid they came from.

    Himalaya returns the selected alpha through a log/exp round trip, which on
    a float32 device drifts by a few ULPs. Matching in log space recovers the
    exact candidate the search chose.

    Args:
        values (np.ndarray): Recovered alphas, shape `(n_targets,)`.
        grid (np.ndarray): Candidate alphas, shape `(n_alphas,)`.

    Returns:
        np.ndarray: Values drawn from `grid`, shape `(n_targets,)`.
    """
    log_grid = np.log(np.asarray(grid, dtype=np.float64))
    log_values = np.log(np.asarray(values, dtype=np.float64))
    nearest = np.argmin(np.abs(log_grid[:, None] - log_values[None, :]), axis=0)
    return np.asarray(grid, dtype=np.float64)[nearest]


class Ridge:
    """Ridge regression over one or several named feature spaces.

    Fits `argmin_b ||X @ b - y||^2 + alpha * ||b||^2` without an intercept.
    Callers own preprocessing: `Ridge` never centers, scales, standardizes, or
    adds an intercept column. Constructor arguments are validated once, at
    construction, and must not be reassigned afterwards.

    A two-dimensional `X` fits ordinary Ridge. A mapping from names to
    two-dimensional arrays fits banded Ridge, which searches feature-space
    weights on the simplex jointly with the alphas.

    Himalaya defines the numerical behavior: the cross-validation loss is
    negative mean squared error, and alpha selection, the Dirichlet search, and
    coefficient refitting all come from its solvers.

    Args:
        alpha (float | Sequence[float] | np.ndarray): A positive finite scalar
            fits a fixed alpha and requires `cv=None`. A non-empty
            one-dimensional collection of positive finite values selects an
            alpha by cross-validation and requires `cv`. Default: `1.0`.
        cv (int | BaseCrossValidator | None): An integer builds unshuffled
            K-fold splits; a reusable scikit-learn cross-validator is used as
            given. A single-use split generator is invalid because fitting
            traverses the splits more than once. Default: `None`.
        search_iterations (int): Number of sampled feature-space weight vectors
            for banded Ridge. Default: `100`.
        dirichlet_concentration (float | Sequence[float]): Concentration
            parameter(s) of the Dirichlet distribution the candidate weights are
            drawn from. A list is cycled through across candidates.
            Default: `(0.1, 1.0)`.
        device (str): `'cpu'` or `'gpu'`. An explicit `'gpu'` resolves to CUDA
            or MPS or raises; it never falls back to a CPU backend.
            Default: `'cpu'`.
        memory_budget_gb (float | None): Working-memory budget in GB used to
            size Himalaya's internal batches. None measures the device with
            conservative headroom. It is a budget, not a hard process limit.
            Default: `None`.
        per_target_alpha (bool): True selects the best alpha separately per
            target; False averages each candidate's fold scores across targets
            and selects one shared alpha. Default: `True`.
        prefer_conservative_alpha (bool): True selects the largest alpha whose
            mean score beats the best alpha's mean score minus that alpha's
            standard deviation across folds. Invalid with
            `per_target_alpha=False`. Default: `False`.
        random_state (int | None): Seed for the banded random search only; the
            cross-validator controls split randomness. Ordinary Ridge accepts it
            and ignores it — it has no randomness of its own — so that
            `BrainData.fit` can keep forwarding one shared `random_state` to
            whichever estimator it builds. Default: `None`.
        progress_bar (bool): Show a progress bar over the banded search.
            Default: `False`.

    Attributes:
        coef_ (np.ndarray): `(n_features,)` for one-dimensional `y`, otherwise
            `(n_features, n_targets)`, in concatenated feature-space order.
        alpha_ (float | np.ndarray): Scalar for a fixed or shared alpha,
            otherwise `(n_targets,)`.
        cv_scores_ (float | np.ndarray | None): None for a fixed-alpha fit. For
            ordinary Ridge, the fold-averaged negative-MSE score at the selected
            alpha. For banded Ridge, `(search_iterations,)` or
            `(search_iterations, n_targets)` fold-averaged scores.
        feature_space_weights_ (np.ndarray | None): None for ordinary Ridge.
            Strictly positive weights whose columns sum to one, shaped
            `(n_spaces,)` or `(n_spaces, n_targets)`.
        feature_space_names_ (tuple[str, ...] | None): Fitted mapping keys in
            coefficient order; None for ordinary Ridge.
        feature_space_sizes_ (tuple[int, ...] | None): Feature counts aligned
            with `feature_space_names_`; None for ordinary Ridge.
        backend_ (Backend): The resolved execution backend.
        n_samples_ (int): Fitted sample count.
        n_features_in_ (int): Total fitted feature count across spaces.
        is_fitted_ (bool): True after a successful fit.

    Examples:
        ```python
        import numpy as np
        from nltools.models import Ridge

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        model = Ridge(alpha=1.0).fit(X, y)
        predictions = model.predict(X)

        # Banded ridge over two named feature spaces
        spaces = {"motion": np.random.randn(100, 6), "task": np.random.randn(100, 12)}
        banded = Ridge(alpha=[1.0, 10.0, 100.0], cv=5, search_iterations=20)
        banded.fit(spaces, y)
        print(banded.feature_space_weights_)
        ```
    """

    def __init__(
        self,
        *,
        alpha: float | Sequence[float] | np.ndarray = 1.0,
        cv=None,
        search_iterations: int = 100,
        dirichlet_concentration: float | Sequence[float] = (0.1, 1.0),
        device: str = "cpu",
        memory_budget_gb: float | None = None,
        per_target_alpha: bool = True,
        prefer_conservative_alpha: bool = False,
        random_state: int | None = None,
        progress_bar: bool = False,
    ) -> None:
        self.alpha = alpha
        self.cv = cv
        self.search_iterations = search_iterations
        self.dirichlet_concentration = dirichlet_concentration
        self.device = device
        self.memory_budget_gb = memory_budget_gb
        self.per_target_alpha = per_target_alpha
        self.prefer_conservative_alpha = prefer_conservative_alpha
        self.random_state = random_state
        self.progress_bar = progress_bar
        self.is_fitted_ = False
        #: The normalized alpha `fit` solves with; `alpha` is validated once here.
        self._normalized_alpha = self._validate_parameters()

    # ---------------------------------------------------------------- validation

    def _validate_parameters(self) -> np.ndarray | float:
        """Check the constructor arguments and return the normalized alpha.

        Returns:
            float | np.ndarray: The scalar alpha, or the one-dimensional array
                of candidate alphas.

        Raises:
            ValueError: If any argument or combination of arguments is invalid.
        """
        if self.device not in ("cpu", "gpu"):
            raise ValueError(
                f"device must be 'cpu' or 'gpu', got {self.device!r}; "
                "there is no 'auto' device for Ridge"
            )
        if self.memory_budget_gb is not None:
            budget = self.memory_budget_gb
            if not isinstance(budget, numbers.Real) or isinstance(budget, bool):
                raise ValueError(
                    f"memory_budget_gb must be a positive number or None, "
                    f"got {budget!r}"
                )
            if not np.isfinite(budget) or budget <= 0:
                raise ValueError(
                    f"memory_budget_gb must be positive and finite, got {budget!r}"
                )
        if self.prefer_conservative_alpha and not self.per_target_alpha:
            raise ValueError(
                "prefer_conservative_alpha=True requires per_target_alpha=True; "
                "the conservative rule needs per-target fold scores"
            )
        if not isinstance(self.search_iterations, numbers.Integral) or isinstance(
            self.search_iterations, bool
        ):
            raise ValueError(
                f"search_iterations must be a positive int, "
                f"got {self.search_iterations!r}"
            )
        if self.search_iterations < 1:
            raise ValueError(
                f"search_iterations must be at least 1, got {self.search_iterations}"
            )

        alpha = self._validate_alpha()
        self._validate_cv()

        scalar_alpha = np.isscalar(alpha) or np.ndim(alpha) == 0
        if scalar_alpha and self.cv is not None:
            raise ValueError(
                f"scalar alpha={alpha!r} fits a fixed alpha and requires cv=None; "
                f"got cv={self.cv!r}. Pass a sequence of alphas to select one."
            )
        if not scalar_alpha and self.cv is None:
            raise ValueError(
                "a sequence of alphas requires cv to select among them; got cv=None. "
                "Pass a scalar alpha for a fixed-alpha fit."
            )
        return alpha

    def _validate_alpha(self) -> np.ndarray | float:
        """Normalize `alpha` into a positive scalar or a 1-D candidate array.

        Returns:
            float | np.ndarray: The validated alpha.

        Raises:
            ValueError: If `alpha` is `"auto"`, empty, multidimensional, or
                holds non-finite or non-positive values.
        """
        alpha = self.alpha
        if isinstance(alpha, str):
            raise ValueError(
                f"alpha must be a positive number or a sequence of positive "
                f"numbers, got {alpha!r}; alpha='auto' is not supported — pass "
                "the candidate alphas and a cv"
            )
        if isinstance(alpha, numbers.Real) and not isinstance(alpha, bool):
            if not np.isfinite(alpha) or alpha <= 0:
                raise ValueError(f"alpha must be positive and finite, got {alpha!r}")
            return float(alpha)

        values = np.asarray(alpha, dtype=np.float64)
        if values.ndim == 0:
            if not np.isfinite(values) or values <= 0:
                raise ValueError(f"alpha must be positive and finite, got {alpha!r}")
            return float(values)
        if values.ndim != 1:
            raise ValueError(
                f"alpha must be a scalar or a 1D collection, got a "
                f"{values.ndim}D array of shape {values.shape}"
            )
        if values.size == 0:
            raise ValueError("alpha must not be an empty collection")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"alpha values must all be finite, got {alpha!r}")
        if not np.all(values > 0):
            raise ValueError(f"alpha values must all be positive, got {alpha!r}")
        return values

    def _validate_cv(self) -> None:
        """Check that `cv` is None, an int fold count, or a reusable splitter.

        Raises:
            ValueError: If `cv` is an integer below 2 or an unsupported type.
            TypeError: If `cv` is a single-use split generator.
        """
        cv = self.cv
        if cv is None:
            return
        is_splitter = hasattr(cv, "split") and hasattr(cv, "get_n_splits")
        if hasattr(cv, "__next__") and not is_splitter:
            raise TypeError(
                "cv got a single-use generator (e.g. `splitter.split(X)`). Pass "
                "the splitter itself — KFold(5), GroupKFold(8) — because fitting "
                "iterates the splits more than once."
            )
        if is_splitter:
            return
        if isinstance(cv, numbers.Integral) and not isinstance(cv, bool):
            if cv < 2:
                raise ValueError(f"cv must be at least 2 folds, got {cv}")
            return
        raise ValueError(
            f"cv must be None, an int fold count, or a scikit-learn "
            f"cross-validator, got {cv!r}"
        )

    def _resolved_cv(self):
        """Return the cross-validator Himalaya should use.

        Returns:
            BaseCrossValidator | None: An unshuffled `KFold` for an int `cv`,
                the caller's splitter, or None.
        """
        if self.cv is None:
            return None
        if isinstance(self.cv, numbers.Integral) and not isinstance(self.cv, bool):
            from sklearn.model_selection import KFold

            return KFold(n_splits=int(self.cv), shuffle=False)
        return self.cv

    def _check_banded_only_arguments_unused(self) -> None:
        """Reject banded-only arguments left at non-default values.

        Raises:
            ValueError: If a banded-only argument was set while fitting
                ordinary Ridge.
        """
        for name, default in _BANDED_ONLY_DEFAULTS.items():
            value = getattr(self, name)
            if isinstance(default, tuple):
                same = isinstance(value, (tuple, list)) and tuple(value) == default
            else:
                same = value == default
            if not same:
                raise ValueError(
                    f"{name}={value!r} only applies to banded Ridge, but X is a "
                    "single feature matrix. Pass a mapping of named feature "
                    f"spaces, or leave {name} at its default {default!r}."
                )

    # -------------------------------------------------------------------- inputs

    @staticmethod
    def _as_feature_spaces(X):
        """Split `X` into ordered feature spaces and their names.

        Args:
            X (np.ndarray | Mapping[str, np.ndarray]): A single feature matrix
                or a mapping of named feature spaces.

        Returns:
            tuple[list[np.ndarray], tuple[str, ...] | None]: The matrices in
                coefficient order and their names, or None for ordinary Ridge.

        Raises:
            ValueError: If `X` is an empty mapping, has a non-string name, or
                holds anything but equally sampled 2-D numeric arrays.
        """
        if isinstance(X, Mapping):
            if len(X) == 0:
                raise ValueError(
                    "X is an empty mapping; banded Ridge needs at least one "
                    "named feature space"
                )
            names = []
            spaces = []
            for name, space in X.items():
                if not isinstance(name, str):
                    raise ValueError(
                        f"feature-space names must be strings, got {name!r} of "
                        f"type {type(name).__name__}"
                    )
                array = np.asarray(space)
                if array.ndim != 2:
                    raise ValueError(
                        f"feature space {name!r} must be a 2D array, got "
                        f"{array.ndim}D with shape {array.shape}"
                    )
                names.append(name)
                spaces.append(array)
            sample_counts = {name: s.shape[0] for name, s in zip(names, spaces)}
            if len(set(sample_counts.values())) > 1:
                raise ValueError(
                    "all banded feature spaces must have the same number of "
                    f"samples, got {sample_counts}"
                )
            return spaces, tuple(names)

        try:
            array = np.asarray(X)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "X must be a 2D feature matrix or a mapping of names to 2D "
                f"feature matrices; could not interpret it as an array ({error})"
            ) from None
        if array.dtype == object or array.ndim != 2:
            raise ValueError(
                "X must be a 2D feature matrix or a mapping of names to 2D "
                f"feature matrices, got an array with shape {array.shape} and "
                f"dtype {array.dtype}"
            )
        return [array], None

    # ----------------------------------------------------------------------- fit

    def fit(self, X, y) -> Ridge:
        """Fit the model.

        Args:
            X (np.ndarray | Mapping[str, np.ndarray]): A `(n_samples,
                n_features)` matrix for ordinary Ridge, or a non-empty mapping
                of unique names to equally sampled 2-D matrices for banded
                Ridge.
            y (np.ndarray): Targets of shape `(n_samples,)` or `(n_samples,
                n_targets)`.

        Returns:
            Ridge: `self`.

        Raises:
            ValueError: If any input or argument combination is invalid.
            RuntimeError: If `device='gpu'` and no accelerator is available.
        """
        alpha = self._normalized_alpha
        spaces, names = self._as_feature_spaces(X)
        is_banded = names is not None

        y = np.asarray(y)
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be 1D or 2D, got {y.ndim}D array")
        n_samples = spaces[0].shape[0]
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y have inconsistent sample counts: X has {n_samples}, "
                f"y has {y.shape[0]}"
            )
        y_was_1d = y.ndim == 1
        y_2d = y[:, None] if y_was_1d else y
        if y_2d.shape[1] < 1:
            raise ValueError(
                f"y must have at least one target column, got shape {y.shape}"
            )

        scalar_alpha = np.ndim(alpha) == 0
        if is_banded:
            if scalar_alpha:
                raise ValueError(
                    f"banded Ridge needs a sequence of candidate alphas and an "
                    f"explicit cv, got scalar alpha={alpha!r}"
                )
        else:
            self._check_banded_only_arguments_unused()

        backend = resolve_backend("cpu" if self.device == "cpu" else "gpu")
        dtype = _working_dtype(spaces, y_2d, backend)
        spaces = [np.ascontiguousarray(space, dtype=dtype) for space in spaces]
        targets = np.ascontiguousarray(y_2d, dtype=dtype)

        sizes = tuple(space.shape[1] for space in spaces)
        n_features = int(sum(sizes))
        n_targets = targets.shape[1]
        alphas = np.atleast_1d(np.asarray(alpha, dtype=np.float64))

        self.backend_ = backend
        if scalar_alpha:
            # The fixed-alpha refit sizes its own batch from the same budget; it
            # never runs the cross-validation or alpha loops the others measure.
            self._fit_fixed_alpha(spaces, targets, float(alpha))
        else:
            try:
                batches = _batch_sizes(
                    backend,
                    self.memory_budget_gb,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_targets=n_targets,
                    n_alphas=alphas.size,
                    itemsize=dtype.itemsize,
                )
            except ValueError as error:
                raise ValueError(
                    f"memory_budget_gb={self.memory_budget_gb!r} is too small for a "
                    f"fit with n_samples={n_samples}, n_features={n_features}, "
                    f"n_targets={n_targets}, n_alphas={alphas.size} ({error})"
                ) from error
            if is_banded:
                self._fit_banded(spaces, targets, alphas, dtype, batches)
            else:
                self._fit_ordinary_cv(spaces, targets, alphas, dtype, batches)

        self.feature_space_names_ = names
        self.feature_space_sizes_ = sizes if is_banded else None
        self.n_samples_ = int(n_samples)
        self.n_features_in_ = n_features
        if y_was_1d:
            self._squeeze_single_target()
        self.is_fitted_ = True
        return self

    def _fit_fixed_alpha(self, spaces, targets, alpha) -> None:
        """Solve a fixed-alpha ordinary Ridge and store the fitted state.

        Args:
            spaces (list[np.ndarray]): One feature matrix.
            targets (np.ndarray): `(n_samples, n_targets)` targets.
            alpha (float): The fixed regularization strength.
        """
        self.coef_ = _refit_fixed_hyperparameters(
            spaces,
            targets,
            alpha,
            backend=self.backend_,
            memory_budget_gb=self.memory_budget_gb,
        )
        self.alpha_ = float(alpha)
        self.cv_scores_ = None
        self.feature_space_weights_ = None

    def _fit_ordinary_cv(self, spaces, targets, alphas, dtype, batches) -> None:
        """Select an alpha by cross-validation and store the fitted state.

        Args:
            spaces (list[np.ndarray]): One feature matrix.
            targets (np.ndarray): `(n_samples, n_targets)` targets.
            alphas (np.ndarray): Candidate alphas.
            dtype (np.dtype): Working dtype.
            batches (dict[str, int]): Himalaya batch sizes.
        """
        from himalaya.ridge import solve_ridge_cv_svd
        from himalaya.scoring import l2_neg_loss

        with _scoped_himalaya_backend(_himalaya_backend_name(self.backend_)):
            design, y_device, alpha_device = _on_active_backend(
                spaces[0], targets, alphas.astype(dtype)
            )
            best_alphas, coefs, cv_scores = solve_ridge_cv_svd(
                design,
                y_device,
                alphas=alpha_device,
                fit_intercept=False,
                score_func=l2_neg_loss,
                cv=self._resolved_cv(),
                local_alpha=self.per_target_alpha,
                conservative=self.prefer_conservative_alpha,
                warn=False,
                **batches,
            )

        self.coef_ = np.asarray(_to_cpu_numpy(coefs), dtype=np.float64)
        selected = _snap_to_grid(_to_cpu_numpy(best_alphas), alphas)
        self.alpha_ = float(selected[0]) if not self.per_target_alpha else selected
        self.cv_scores_ = np.asarray(
            _to_cpu_numpy(cv_scores), dtype=np.float64
        ).reshape(-1)
        self.feature_space_weights_ = None

    def _fit_banded(self, spaces, targets, alphas, dtype, batches) -> None:
        """Run the banded random search and store the fitted state.

        Args:
            spaces (list[np.ndarray]): Feature matrices in coefficient order.
            targets (np.ndarray): `(n_samples, n_targets)` targets.
            alphas (np.ndarray): Candidate alphas.
            dtype (np.dtype): Working dtype.
            batches (dict[str, int]): Himalaya batch sizes.
        """
        from himalaya.kernel_ridge import generate_dirichlet_samples
        from himalaya.ridge import solve_group_ridge_random_search
        from himalaya.scoring import l2_neg_loss

        # Himalaya's sampler ends with `get_backend().asarray(gammas)`, so the
        # candidates would otherwise take the dtype and device of whatever
        # backend happened to be globally active. They are validated and clamped
        # on the host, so draw them under an explicit numpy scope.
        with _scoped_himalaya_backend("numpy"):
            candidates = _to_cpu_numpy(
                generate_dirichlet_samples(
                    n_samples=self.search_iterations,
                    n_kernels=len(spaces),
                    concentration=self._concentration_for_himalaya(),
                    random_state=self.random_state,
                )
            )
        candidates = _prepare_feature_space_weights(candidates, dtype)

        with _scoped_himalaya_backend(_himalaya_backend_name(self.backend_)):
            converted = _on_active_backend(*spaces, targets, alphas.astype(dtype))
            designs, y_device, alpha_device = (
                converted[:-2],
                converted[-2],
                converted[-1],
            )
            deltas, refit_weights, cv_scores = solve_group_ridge_random_search(
                designs,
                y_device,
                n_iter=candidates,
                alphas=alpha_device,
                fit_intercept=False,
                score_func=l2_neg_loss,
                cv=self._resolved_cv(),
                return_weights=True,
                local_alpha=self.per_target_alpha,
                random_state=self.random_state,
                progress_bar=self.progress_bar,
                conservative=self.prefer_conservative_alpha,
                warn=False,
                **batches,
            )

        deltas = np.asarray(_to_cpu_numpy(deltas), dtype=np.float64)
        self.coef_ = np.asarray(_to_cpu_numpy(refit_weights), dtype=np.float64)
        self.cv_scores_ = np.asarray(_to_cpu_numpy(cv_scores), dtype=np.float64)

        # deltas = log(gamma / alpha) with each gamma column summing to one, so
        # the simplex weights and the selected alpha both fall out of a
        # log-sum-exp over the spaces.
        shifted = deltas - deltas.max(axis=0, keepdims=True)
        weights = np.exp(shifted)
        self.feature_space_weights_ = weights / weights.sum(axis=0, keepdims=True)
        log_total = deltas.max(axis=0) + np.log(np.exp(shifted).sum(axis=0))
        selected = _snap_to_grid(np.exp(-log_total), alphas)
        self.alpha_ = selected if self.per_target_alpha else float(selected[0])

    def _concentration_for_himalaya(self):
        """Return `dirichlet_concentration` in the form Himalaya's sampler takes.

        Returns:
            float | list[float]: A scalar, or a list Himalaya cycles through.
        """
        values = np.atleast_1d(
            np.asarray(self.dirichlet_concentration, dtype=np.float64)
        )
        if values.size == 1:
            return float(values[0])
        return [float(value) for value in values]

    def _squeeze_single_target(self) -> None:
        """Drop the trailing target axis after fitting one-dimensional `y`."""
        self.coef_ = self.coef_[:, 0]
        if isinstance(self.alpha_, np.ndarray):
            self.alpha_ = float(self.alpha_[0])
        if self.cv_scores_ is not None:
            self.cv_scores_ = (
                float(self.cv_scores_[0])
                if self.cv_scores_.ndim == 1
                else self.cv_scores_[:, 0]
            )
        if self.feature_space_weights_ is not None:
            self.feature_space_weights_ = self.feature_space_weights_[:, 0]

    # ------------------------------------------------------------------- predict

    def _design_matrix(self, X) -> np.ndarray:
        """Align `X` to the fitted feature structure and concatenate it.

        Args:
            X (np.ndarray | Mapping[str, np.ndarray]): Prediction features in
                the structure used for fitting.

        Returns:
            np.ndarray: A `(n_samples, n_features_in_)` matrix in fitted order.

        Raises:
            ValueError: If the structure, names, or feature counts differ from
                the fitted model.
        """
        spaces = self._aligned_feature_spaces(X)
        return spaces[0] if len(spaces) == 1 else np.concatenate(spaces, axis=1)

    def _aligned_feature_spaces(self, X) -> list[np.ndarray]:
        """Align `X` to the fitted feature structure, one matrix per space.

        The single place that validates prediction and bootstrap features
        against the fitted model: `_design_matrix` concatenates the result, and
        the `BrainData` Ridge bootstrap resamples the spaces separately so a
        banded refit can rescale each one by its own simplex weight.

        Args:
            X (np.ndarray | Mapping[str, np.ndarray]): Features in the
                structure used for fitting. A banded mapping may be in any
                order; it is aligned to `feature_space_names_`.

        Returns:
            list[np.ndarray]: One `(n_samples, n_features_k)` matrix per fitted
                feature space, in coefficient order.

        Raises:
            ValueError: If the structure, names, feature counts, or sample
                counts differ from the fitted model.
        """
        names = self.feature_space_names_
        sizes = self.feature_space_sizes_
        if names is None or sizes is None:
            if isinstance(X, Mapping):
                raise ValueError(
                    "this model was fitted on a single feature matrix, but X is "
                    f"a mapping with names {tuple(X)}"
                )
            spaces, _ = self._as_feature_spaces(X)
            matrix = spaces[0]
            if matrix.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {matrix.shape[1]} features, but Ridge was fitted "
                    f"with {self.n_features_in_} features"
                )
            return [matrix]

        if not isinstance(X, Mapping):
            raise ValueError(
                "this model was fitted on named feature spaces "
                f"{names}, so X must be a mapping with the "
                f"same names, got {type(X).__name__}"
            )
        missing = [name for name in names if name not in X]
        extra = [name for name in X if name not in names]
        if missing or extra:
            raise ValueError(
                f"X must contain exactly the fitted feature spaces "
                f"{names}; missing {tuple(missing)}, "
                f"unexpected {tuple(extra)}"
            )
        ordered = []
        for name, size in zip(names, sizes):
            space = np.asarray(X[name])
            if space.ndim != 2:
                raise ValueError(
                    f"feature space {name!r} must be a 2D array, got "
                    f"{space.ndim}D with shape {space.shape}"
                )
            if space.shape[1] != size:
                raise ValueError(
                    f"feature space {name!r} has {space.shape[1]} features, but "
                    f"Ridge was fitted with {size}"
                )
            ordered.append(space)
        counts = {name: space.shape[0] for name, space in zip(names, ordered)}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"all feature spaces must have the same number of samples, got {counts}"
            )
        return ordered

    def predict(self, X) -> np.ndarray:
        """Predict targets for `X`.

        Args:
            X (np.ndarray | Mapping[str, np.ndarray]): Features in the
                structure used for fitting. Banded mappings may be in any
                order; they are aligned to `feature_space_names_`.

        Returns:
            np.ndarray: `(n_samples,)` when fitted on one-dimensional `y`,
                otherwise `(n_samples, n_targets)`.

        Raises:
            ValueError: If the model is not fitted, or `X` does not match the
                fitted feature structure.
        """
        _check_is_fitted(self)
        return self._design_matrix(X) @ self.coef_

    def score(self, X, y) -> float | np.ndarray:
        """Return the coefficient of determination for each target.

        Args:
            X (np.ndarray | Mapping[str, np.ndarray]): Features in the fitted
                structure.
            y (np.ndarray): True targets, `(n_samples,)` or `(n_samples,
                n_targets)`.

        Returns:
            float | np.ndarray: A `float` for one-dimensional `y`, otherwise an
                array of shape `(n_targets,)`. A constant target scores zero.

        Raises:
            ValueError: If the model is not fitted, or the shapes disagree.
        """
        _check_is_fitted(self)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be 1D or 2D, got {y.ndim}D array")
        predictions = np.asarray(self.predict(X), dtype=np.float64)
        if y.shape[0] != predictions.shape[0]:
            raise ValueError(
                f"X and y have inconsistent sample counts: X gives "
                f"{predictions.shape[0]} predictions, y has {y.shape[0]}"
            )
        was_1d = y.ndim == 1
        y_2d = y[:, None] if was_1d else y
        predicted_2d = predictions[:, None] if predictions.ndim == 1 else predictions
        if y_2d.shape[1] != predicted_2d.shape[1]:
            raise ValueError(
                f"y has {y_2d.shape[1]} targets, but Ridge predicts "
                f"{predicted_2d.shape[1]}"
            )

        residual = np.sum((y_2d - predicted_2d) ** 2, axis=0)
        total = np.sum((y_2d - y_2d.mean(axis=0)) ** 2, axis=0)
        scores = np.zeros(y_2d.shape[1], dtype=np.float64)
        varying = total > 0
        scores[varying] = 1.0 - residual[varying] / total[varying]
        return float(scores[0]) if was_1d else scores

    def __repr__(self) -> str:
        """Return a short constructor-style summary of the model."""
        return f"Ridge(alpha={self.alpha!r}, device={self.device!r})"
