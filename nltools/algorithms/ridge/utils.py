"""Utility functions for ridge regression.

Helpers for target batching, the shared SVD decomposition across alphas, alpha
selection from cross-validation scores, R² scoring, and Dirichlet sampling of
feature-space weights for banded ridge (patterns follow the himalaya library).
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from collections.abc import Iterator
from sklearn.utils import check_random_state

if TYPE_CHECKING:
    from nltools.algorithms.backends import Backend


def _auto_n_targets_batch(
    max_gpu_memory_gb: float | None,
    elements_per_target: int,
    n_targets: int,
    backend=None,
) -> int:
    """Derive a GPU target-batch size from a memory budget.

    Used by the GPU ridge paths when `n_targets_batch` is left unset so
    `max_gpu_memory_gb` actually bounds GPU allocation instead of processing
    all targets at once. Thin adapter over the batching layer in
    `nltools.algorithms.backends`.

    `elements_per_target` is the caller's estimate of the dominant
    target-scaling working set (in float32 elements) for a single target
    column — e.g. the alpha-batched prediction block `n_alphas_batch *
    n_samples` in the CV solvers, or `rank + n_features + n_samples` in the
    single-fit SVD path. A 5× overhead factor keeps peak allocation clear of
    the budget.

    Args:
        max_gpu_memory_gb (float | None): GPU memory budget in GB. None measures
            the device via `device_memory_budget`.
        elements_per_target (int): Dominant float32 working-set size per target.
        n_targets (int): Total number of targets (columns of Y).
        backend (Backend | None): Resolved backend the work runs on (used only to
            measure the budget when `max_gpu_memory_gb` is None).

    Returns:
        int: Target batch size in `[1, n_targets]`.
    """
    from ..backends import auto_batch_size, device_memory_budget

    budget_gb = device_memory_budget(
        backend, max_gpu_memory_gb=max_gpu_memory_gb, cap_for_batching=True
    )
    n_targets_batch, _ = auto_batch_size(
        n_targets,
        elements_per_target * 4,  # float32
        budget_gb=budget_gb,
        overhead=5.0,
    )
    return n_targets_batch


def generate_dirichlet_samples(
    n_samples: int,
    n_kernels: int,
    concentration: float | list[float] = [0.1, 1.0],
    random_state: int | None = None,
) -> np.ndarray:
    """Generate samples from a Dirichlet distribution.

    Used to draw candidate feature-space weights (gamma) for the random search
    in banded ridge regression.

    Args:
        n_samples (int): Number of samples to generate.
        n_kernels (int): Number of dimensions (feature spaces) of the distribution.
        concentration (float | list[float]): Concentration parameter(s). A value
            of 1 samples uniformly over the simplex; `np.inf` gives equal
            weights; a list alternates between its values across samples.
            Defaults to `[0.1, 1.0]`.
        random_state (int | None): Random generator seed; use an int for
            deterministic samples. Defaults to None.

    Returns:
        np.ndarray: Samples of shape (n_samples, n_kernels); each row sums to 1.

    Examples:
        ```python
        gammas = generate_dirichlet_samples(10, 3, concentration=[0.1, 1.0])
        gammas.shape  # → (10, 3)
        np.allclose(gammas.sum(axis=1), 1.0)  # → True
        ```
    """
    random_generator = check_random_state(random_state)

    concentration_arr = np.atleast_1d(concentration)
    n_concentrations = len(concentration_arr)
    n_samples_per_concentration = int(np.ceil(n_samples / float(n_concentrations)))

    # Generate the gammas
    gammas = []
    for conc in concentration_arr:
        if conc == np.inf:
            # Equal weights for all spaces
            gamma = np.full(n_kernels, fill_value=1.0 / n_kernels)[None]
            gamma = np.tile(gamma, (n_samples_per_concentration, 1))
        else:
            # Sample from Dirichlet distribution
            gamma = random_generator.dirichlet(
                [conc] * n_kernels, n_samples_per_concentration
            )
        gammas.append(gamma)
    gammas = np.vstack(gammas)

    # Reorder the gammas to alternate between concentrations:
    # [a0, a1, a2, a0, a1, a2] instead of [a0, a0, a1, a1, a2, a2]
    gammas = gammas.reshape(n_concentrations, n_samples_per_concentration, n_kernels)
    gammas = np.swapaxes(gammas, 0, 1)
    gammas = gammas.reshape(n_concentrations * n_samples_per_concentration, n_kernels)

    # Remove extra gammas if we generated more than requested
    gammas = gammas[:n_samples]

    return gammas


def _decompose_ridge(
    Xtrain: np.ndarray,
    alphas: np.ndarray,
    n_alphas_batch: int | None = None,
    method: str = "svd",
    backend: Backend | None = None,
) -> Iterator[tuple[np.ndarray, slice]]:
    """Yield ridge resolution matrices `(X.T @ X + alpha * I)^-1 @ X.T`, batched over alphas.

    Ridge predictions are `Ytest_hat = Xtest @ matrices @ Ytrain`. One SVD of
    `Xtrain` serves every alpha, and the matrices are produced in alpha batches
    so only one batch is alive at a time.

    Args:
        Xtrain (np.ndarray): Training features of shape (n_samples_train, n_features).
        alphas (np.ndarray): Ridge regularization parameters, shape (n_alphas,).
        n_alphas_batch (int | None): Number of alphas per yielded batch; smaller
            batches use less memory. None processes all alphas at once.
            Defaults to None.
        method (str): Decomposition method; only `"svd"` is supported.
            Defaults to `"svd"`.
        backend (Backend | None): Backend providing `.svd`, `.matmul`, and
            `.expand_dims`. Effectively required: the default None is not
            handled and raises `AttributeError`.

    Yields:
        tuple[np.ndarray, slice]: `(matrices, alpha_batch)` — resolution matrices
            of shape (n_alphas_batch, n_features, n_samples_train) and the slice
            of `alphas` they correspond to.

    Raises:
        ValueError: If `method` is not `"svd"`.

    Examples:
        ```python
        X = np.random.randn(100, 50)
        alphas = np.array([0.1, 1.0, 10.0])
        for matrices, batch in _decompose_ridge(X, alphas, n_alphas_batch=2, backend=backend):
            print(alphas[batch])  # → [0.1 1.0] then [10.0]
        ```
    """

    # Default: process all alphas at once
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    # SVD decomposition: X = U @ diag(s) @ Vt
    if method == "svd":
        U, s, Vt = backend.svd(Xtrain, full_matrices=False)
    else:
        raise ValueError(f"Unknown method={method!r}")

    # Yield batches of resolution matrices
    for start in range(0, len(alphas), n_alphas_batch):
        batch = slice(start, start + n_alphas_batch)
        alphas_batch = alphas[batch]

        # Compute eigenvalue weighting for this alpha batch
        # Shape: (n_alphas_batch, n_features)
        if len(alphas_batch.shape) == 0:  # Scalar
            alphas_batch = backend.expand_dims(alphas_batch, 0)

        # Ridge solution: (XtX + alpha*I)^-1 = V @ diag(s/(s^2 + alpha)) @ Vt
        # We compute: s / (alpha + s^2) for each alpha
        ev_weighting = s[None, :] / (alphas_batch[:, None] + s[None, :] ** 2)

        # Resolution matrices: Vt.T @ diag(ev_weighting) @ U.T
        # Shape: (n_alphas_batch, n_features, n_samples_train)
        matrices = backend.matmul(Vt.T, ev_weighting[:, :, None] * U.T[None, :, :])

        yield matrices, batch

        # Delete to free memory (generator cleanup)
        del matrices


def _select_best_alphas(
    scores: np.ndarray,
    alphas: np.ndarray,
    local_alpha: bool,
    backend: Backend | None = None,
    conservative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the best alphas from cross-validation scores.

    Scores are averaged over splits, with a tiny bias toward larger alphas so
    ties resolve toward more regularization (himalaya's convention).

    Args:
        scores (np.ndarray): Cross-validation scores of shape
            (n_splits, n_alphas, n_targets).
        alphas (np.ndarray): Ridge regularization parameters, shape (n_alphas,).
        local_alpha (bool): If True, pick the best alpha independently per
            target; if False, pick one alpha for all targets.
        backend (Backend | None): Backend providing `.asarray`, `.xp`, and
            `.full`. Effectively required: the default None is not handled and
            raises `AttributeError`.
        conservative (bool): If True (and `local_alpha=True`), pick the largest
            alpha whose mean score is within one standard deviation of the best,
            trading a little fit for more regularization. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(alphas_argmax, best_scores_mean)` — the
            index of the selected alpha for each target and its split-averaged
            score, both of shape (n_targets,).

    Raises:
        NotImplementedError: If `conservative=True` with `local_alpha=False`.
    """

    # Ensure scores and alphas are on the same device
    scores = backend.asarray(scores)
    alphas = backend.asarray(alphas)

    # Average scores over CV splits
    # Shape: (n_alphas, n_targets)
    scores_mean = backend.xp.mean(scores, axis=0)

    # Add tiny epsilon slope to prefer larger alphas when scores are equal
    # This provides a principled tiebreaker toward more regularization
    scores_mean = scores_mean + (backend.xp.log(alphas) * 1e-10)[:, None]

    if local_alpha:
        # Select best alpha independently for each target
        alphas_argmax = backend.xp.argmax(scores_mean, axis=0)  # ty: ignore[unknown-argument]

        if conservative:
            # Conservative: take largest alpha within 1 std of best
            scores_std = backend.xp.std(scores, axis=0)
            best_scores = scores_mean[alphas_argmax, np.arange(len(alphas_argmax))]
            threshold = (
                best_scores - scores_std[alphas_argmax, np.arange(len(alphas_argmax))]
            )

            # Find which alphas beat the threshold
            beats_threshold = scores_mean > threshold[None, :]
            # Bias toward larger alphas
            beats_threshold = (
                beats_threshold.astype(np.float32)
                + backend.xp.log(alphas)[:, None] * 1e-4
            )
            alphas_argmax = backend.xp.argmax(beats_threshold, axis=0)  # ty: ignore[unknown-argument]

    else:
        # Global: select single best alpha for all targets
        if conservative:
            raise NotImplementedError(
                "conservative=True with local_alpha=False not implemented"
            )

        # Mean over targets, then argmax over alphas
        global_scores = backend.xp.mean(scores_mean, axis=1)
        best_alpha_idx = backend.xp.argmax(global_scores)

        # Broadcast to all targets
        alphas_argmax = backend.full(
            scores_mean.shape[1], fill_value=best_alpha_idx, dtype="int64"
        )

    # Get best scores for selected alphas
    best_scores_mean = scores_mean[alphas_argmax, np.arange(scores_mean.shape[1])]

    return alphas_argmax, best_scores_mean  # ty: ignore[invalid-return-type]


def _r2_score(
    y_true: np.ndarray, y_pred: np.ndarray, backend: Backend | None = None
) -> np.ndarray:
    """Compute the R² score (coefficient of determination) per target.

    `R² = 1 - SS_res / SS_tot`, with the residual and total sums of squares
    taken over samples; a `1e-10` guard avoids division by zero. Works with
    NumPy or PyTorch arrays through the backend.

    Args:
        y_true (np.ndarray): True targets of shape (n_samples, n_targets).
        y_pred (np.ndarray): Predictions of shape (n_samples, n_targets) or
            (n_alphas, n_samples, n_targets).
        backend (Backend | None): Backend providing `.xp` for the reductions.
            Effectively required: the default None is not handled and raises
            `AttributeError`.

    Returns:
        np.ndarray: R² per target, shape (n_targets,) or (n_alphas, n_targets).
    """
    # Handle both 2D and 3D predictions (with alpha dimension)
    if len(y_pred.shape) == 3:
        # Shape: (n_alphas, n_samples, n_targets)
        ss_res = backend.xp.sum((y_true[None, :, :] - y_pred) ** 2, axis=1)
        ss_tot = backend.xp.sum(
            (y_true[None, :, :] - backend.xp.mean(y_true, axis=0)[None, None, :]) ** 2,
            axis=1,
        )
    else:
        # Shape: (n_samples, n_targets)
        ss_res = backend.xp.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = backend.xp.sum(
            (y_true - backend.xp.mean(y_true, axis=0)[None, :]) ** 2, axis=0
        )

    # R² = 1 - SS_res / SS_tot
    # Avoid division by zero
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    return r2
