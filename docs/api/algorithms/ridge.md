---
title: algorithms.ridge
---

Ridge regression algorithms and utilities.

This package contains ridge regression implementations with GPU acceleration.

Features:
- Cross-validation with per-target or global alpha selection
- Memory-efficient batching for large-scale problems
- GPU acceleration (10-100x speedup on large datasets)
- Banded ridge for multiple feature spaces

**Methods:**

Name | Description
---- | -----------
[`cross_val_predict_ridge`](#algorithms-ridge-cross-val-predict-ridge) | Held-out ridge predictions per CV fold under a (per-target) alpha.
[`generate_dirichlet_samples`](#algorithms-ridge-generate-dirichlet-samples) | Generate samples from a Dirichlet distribution.
[`ridge_cv`](#algorithms-ridge-ridge-cv) | Ridge regression with cross-validation for hyperparameter selection.
[`ridge_svd`](#algorithms-ridge-ridge-svd) | Solve ridge regression using Singular Value Decomposition.
[`solve_banded_ridge_cv`](#algorithms-ridge-solve-banded-ridge-cv) | Solve banded ridge regression with cross-validation using random search.
[`solve_ridge_cv`](#algorithms-ridge-solve-ridge-cv) | Solve ridge regression with cross-validation.



**Examples:**

```python
import numpy as np
from nltools.algorithms.ridge import solve_ridge_cv

X = np.random.randn(100, 50)
Y = np.random.randn(100, 10)
result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
```

## Methods

(algorithms-ridge-cross-val-predict-ridge)=
### `cross_val_predict_ridge`

```python
cross_val_predict_ridge(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, n_targets_batch: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None) -> dict[str, Any]
```

Held-out ridge predictions per CV fold under a (per-target) alpha.

For each fold, refits ridge with the supplied alpha (per-target or
scalar) on the training fold and predicts the held-out fold. Targets
sharing the same alpha share an SVD of the training fold via
`_refit_banded_ridge`, so the cost scales with the number of
*unique* alphas, not the number of targets.

Designed to be the BrainData CV layer's source of held-out predictions
when alpha selection has already been done by ``solve_ridge_cv``: pass
the selected per-voxel alphas back through here to get the fold-by-fold
predictions and per-fold R² needed for ``cv_results_``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Feature matrix of shape (n_samples, n_features). | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). 1D ``Y`` is promoted to (n_samples, 1). | *required*
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray)</code> | Per-target alpha array of shape (n_targets,) or a scalar (broadcast to every target). | *required*
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits (no shuffling). Generators (e.g. ``KFold(5).split(X)``) are rejected — pass the splitter object instead. | <code>5</code>
`fit_intercept` | <code>[bool](#bool)</code> | If True, center X and Y on the *training fold's* mean per fold (sklearn convention) and add the intercept back so predictions live on the original Y scale. | <code>False</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during refit (for memory efficiency). If None, processes all targets at once. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas. If None, processes all unique alphas at once. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to backend device as needed (recommended for large neuroimaging Y). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Per-fold scoring function ``(y_true, y_pred) -> per-target scores``. If None, uses R² in NumPy on CPU (cheap at one fold's size and decoupled from backend ops to avoid stray transfers). | <code>None</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'predictions': (n_samples, n_targets) held-out per-target       predictions on the original Y scale (CPU numpy).     - 'folds': (n_samples,) int fold index per row (CPU numpy).     - 'scores': (n_splits, n_targets) per-fold R² (or       ``score_func``) at the supplied alpha (CPU numpy).     - 'backend': Backend used (for transparency).

(algorithms-ridge-generate-dirichlet-samples)=
### `generate_dirichlet_samples`

```python
generate_dirichlet_samples(n_samples: int, n_kernels: int, concentration: float | list[float] = [0.1, 1.0], random_state: int | None = None) -> np.ndarray
```

Generate samples from a Dirichlet distribution.

This function generates random samples from a Dirichlet distribution,
which is used for sampling feature space weights (gamma) in banded ridge
regression random search.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>[int](#int)</code> | Number of samples to generate. | *required*
`n_kernels` | <code>[int](#int)</code> | Number of dimensions (feature spaces) of the distribution. | *required*
`concentration` | <code>[float](#float) \| [list](#list)[[float](#float)]</code> | Concentration parameters of the Dirichlet distribution. - A value of 1 corresponds to uniform sampling over the simplex. - A value of infinity corresponds to equal weights. - If a list, samples cycle through the list. Defaults to [0.1, 1.0]. | <code>[0.1, 1.0]</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic samples. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Dirichlet samples of shape (n_samples, n_kernels).     Each row sums to 1 (lies on simplex).

**Examples:**

```pycon
>>> # Generate 10 samples for 3 feature spaces
>>> gammas = generate_dirichlet_samples(10, 3, concentration=[0.1, 1.0])
>>> gammas.shape
(10, 3)
>>> # Each row sums to 1
>>> np.allclose(gammas.sum(axis=1), 1.0)
True
```

(algorithms-ridge-ridge-cv)=
### `ridge_cv`

```python
ridge_cv(X: np.ndarray, y: np.ndarray, *, alphas: np.ndarray | None = None, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict
```

Ridge regression with cross-validation for hyperparameter selection.

Performs k-fold cross-validation to select the best alpha parameter,
then fits a final model on all data using the selected alpha.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Training data features with shape (n_samples, n_features) | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target values with shape (n_samples,) or (n_samples, n_targets) | *required*
`alphas` | <code>[ndarray](#numpy.ndarray)</code> | Array of alpha values to try. If None, uses default range: np.logspace(-2, 4, 20) = [0.01, 0.015, ..., 10000] | <code>None</code>
`cv` | <code>int or sklearn CV splitter</code> | Number of folds (int) or an sklearn cross-validator (anything with ``.split(X)`` and ``.get_n_splits()``, e.g. ``KFold(5, shuffle=True)`` or ``GroupKFold(8)``). Splitters are honored for the actual fold iteration, so leave-one-run-out and shuffled-K-fold give different results from contiguous K-fold. Defaults to 5. | <code>5</code>
`fit_intercept` | <code>[bool](#bool)</code> | If True, center X and y on the training mean before fitting and recover the intercept after. The returned ``coef`` is on the centered scale; the recovered intercept is returned under the ``intercept`` key. Defaults to False. | <code>False</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch. Requires torch installed   (raises ImportError otherwise); degrades to torch-CPU only when no   GPU device is present. Use "auto" for torch-optional CPU fallback. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed (not currently used, kept for consistency). Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary containing:<br>    - 'alpha' (float): Best alpha value selected by CV     - 'coef' (np.ndarray): Coefficients using best alpha on full dataset     - 'cv_scores' (np.ndarray): Cross-validation R**2 scores for each fold, alpha, and target         with shape (n_folds, n_alphas, n_targets)     - 'backend' (str): Backend used for computation

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> result = ridge_cv(X, y, cv=3)
>>> result['alpha']  # Best alpha selected
1.0
>>> result['coef'].shape
(50,)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Uses R**2 (coefficient of determination) as the scoring metric
- For multi-target regression, selects alpha that maximizes mean R**2 across targets
- parallel='gpu' requires torch installed; with torch present but no GPU device it
  runs on torch-CPU. It does not fall back to NumPy when torch is absent — use
  parallel='auto' for that.

</details>

(algorithms-ridge-ridge-svd)=
### `ridge_svd`

```python
ridge_svd(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0, parallel: str | None = None, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> np.ndarray
```

Solve ridge regression using Singular Value Decomposition.

This function implements ridge regression using SVD, which provides
numerical stability and efficiency for high-dimensional problems.
The implementation is inspired by the himalaya library.

<details class="algorithm" open markdown="1">
<summary>Algorithm</summary>

The ridge regression solution is:
    beta = (X.T @ X + alpha*I)^(-1) @ X.T @ y

Using SVD of X = U @ diag(s) @ V.T, this becomes:
    beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y

This formulation avoids explicit matrix inversion and is numerically stable.
The shrinkage factor s / (s**2 + alpha) regularizes small singular values.

</details>

<details class="performance" open markdown="1">
<summary>Performance</summary>

- Time complexity: O(n_samples × n_features × min(n_samples, n_features))
- Space complexity: O(n_samples × n_features)
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)
- See `solve_ridge_cv()` for cross-validation with GPU support

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Training data features with shape (n_samples, n_features) | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target values with shape (n_samples,) or (n_samples, n_targets). Can be 1D for single-target or 2D for multi-target | *required*
`alpha` | <code>[float](#float)</code> | Regularization strength. Must be positive. Higher values increase regularization (shrink coefficients toward zero). Defaults to 1.0. | <code>1.0</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch. Requires torch installed   (raises ImportError otherwise); degrades to torch-CPU only when no   GPU device is present. Use "auto" for torch-optional CPU fallback. Defaults to None. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed (not currently used, kept for consistency). Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Ridge regression coefficients     - shape (n_features,) for single-target regression     - shape (n_features, n_targets) for multi-target regression

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> beta = ridge_svd(X, y, alpha=1.0)
>>> beta.shape
(50,)
```

```pycon
>>> # Multi-target regression
>>> Y = np.random.randn(100, 5)
>>> beta = ridge_svd(X, Y, alpha=1.0)
>>> beta.shape
(50, 5)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Time complexity: O(n_samples * n_features * min(n_samples, n_features))
- Space complexity: O(n_samples * n_features)
- For alpha→0, this reduces to ordinary least squares (OLS). Use alpha=1e-6
  for OLS in practice (more numerically stable than alpha=0)
- Supports both CPU (NumPy) and GPU (PyTorch) backends
- See `nltools.algorithms.ridge.solvers.solve_ridge_cv()` for cross-validation
- See `nltools.algorithms.ridge.utils._decompose_ridge()` for generator pattern

</details>

(algorithms-ridge-solve-banded-ridge-cv)=
### `solve_banded_ridge_cv`

```python
solve_banded_ridge_cv(Xs: list[np.ndarray], Y: np.ndarray, *, n_iter: int | np.integer | np.ndarray = 100, concentration: float | list[float] = [0.1, 1.0], alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, jitter_alphas: bool = False, return_weights: bool = True, diagonalize_method: str = 'svd', warn: bool = True, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Solve banded ridge regression with cross-validation using random search.

This function implements true banded/group ridge regression (as in Himalaya).
It searches over feature space weights (gamma) sampled from a Dirichlet
distribution, combined with alpha grid search.

Banded ridge (also called group ridge) applies different scaling weights
per feature space: Z_i = sqrt(gamma_i) * X_i, then solves standard ridge
regression on the scaled concatenated features. This allows optimizing
the relative importance of different feature spaces.

The feature spaces are scaled by sqrt(gamma) for each gamma sample, then
standard ridge regression is applied with alpha grid search.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`Xs` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Feature matrices for different feature spaces. Each array has shape (n_samples, n_features_i). All must have the same n_samples. | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). | *required*
`n_iter` | <code>[int](#int) \| [integer](#numpy.integer) \| [ndarray](#numpy.ndarray)</code> | Number of feature-space weights combination to search, or array of shape (n_iter, n_spaces). If an array is given, the solver uses it as the list of weights to try, instead of sampling from a Dirichlet distribution. Defaults to 100. | <code>100</code>
`concentration` | <code>[float](#float) \| [list](#list)[[float](#float)]</code> | Concentration parameters of the Dirichlet distribution. - A value of 1 corresponds to uniform sampling over the simplex. - A value of infinity corresponds to equal weights. - If a list, iteratively cycle through the list. Not used if n_iter is an array. Defaults to [0.1, 1.0]. | <code>[0.1, 1.0]</code>
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray) \| [list](#list)[[float](#float)]</code> | Range of ridge regularization parameters to try. Can be float or array of shape (n_alphas,). Defaults to [0.1, 1.0, 10.0]. | <code>[0.1, 1.0, 10.0]</code>
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits. Defaults to 5. | <code>5</code>
`local_alpha` | <code>[bool](#bool)</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. Defaults to True. | <code>True</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during CV (for memory efficiency). If None, processes all targets at once. Defaults to None. | <code>None</code>
`n_targets_batch_refit` | <code>[int](#int) \| None</code> | Batch size for targets during refit. If None, uses n_targets_batch value. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas (for memory efficiency). If None, processes all alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to GPU as needed. This prevents OOM when Y is large (e.g., 300k voxels). Defaults to True (recommended for neuroimaging). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Scoring function (y_true, y_pred) -> scores. If None, uses R² score. Defaults to None. | <code>None</code>
`fit_intercept` | <code>[bool](#bool)</code> | Whether to fit an intercept. If False, X and Y should be centered. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display progress bar (requires tqdm). Defaults to False. | <code>False</code>
`conservative` | <code>[bool](#bool)</code> | If True, select largest alpha within 1 std of best score. Defaults to False. | <code>False</code>
`jitter_alphas` | <code>[bool](#bool)</code> | If True, alphas range is slightly jittered for each gamma. Defaults to False. | <code>False</code>
`return_weights` | <code>[bool](#bool)</code> | Whether to refit on the entire dataset and return the weights. Defaults to True. | <code>True</code>
`diagonalize_method` | <code>[str](#str)</code> | Method used to diagonalize the features. Currently only "svd" is supported. Defaults to "svd". | <code>'svd'</code>
`warn` | <code>[bool](#bool)</code> | If True, warn if the number of samples is smaller than the number of features. Defaults to True. | <code>True</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'deltas': Best log feature-space weights for each target,         shape (n_spaces, n_targets). deltas = log(gamma / alpha), where         gamma are the feature space weights.     - 'cv_scores': Cross-validation scores per iteration, averaged over splits,         for the best alpha, shape (n_iter, n_targets). Always returned on CPU         (numpy array).     - 'coefs': Ridge coefficients refit on entire dataset using best hyperparameters,         shape (n_features_total, n_targets), or None if return_weights=False.         Always returned on CPU (numpy array).     - 'intercept': Intercept of shape (n_targets,), or None if         fit_intercept=False or return_weights=False.     - 'backend': Backend used (for transparency).

**Examples:**

```pycon
>>> # Multiple feature spaces (banded ridge with random search)
>>> X1 = np.random.randn(100, 30)  # First feature space
>>> X2 = np.random.randn(100, 20)  # Second feature space
>>> Y = np.random.randn(100, 10)
>>> result = solve_banded_ridge_cv(
...     [X1, X2], Y, n_iter=50, alphas=[0.1, 1.0, 10.0]
... )
>>> deltas = result['deltas']
>>> coefs = result['coefs']
>>> scores = result['cv_scores']
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

This implements true banded/group ridge regression (as in Himalaya's
solve_group_ridge_random_search) with:
- Dirichlet sampling for feature space weights (gamma)
- Scaling each feature space by sqrt(gamma) for each gamma sample
- Cross-validation with alpha grid search
- Per-target selection of best gamma and alpha combination

This is the correct implementation of banded/group ridge regression, which
allows different scaling weights per feature space. For single feature space
ridge regression, use solve_ridge_cv instead.

Algorithm details:

- Random search: Samples gamma weights from Dirichlet distribution
- Banded ridge: Scales each feature space by sqrt(gamma_i), then solves standard ridge
- Cross-validation: Evaluates each (gamma, alpha) combination via k-fold CV
- Best selection: Chooses (gamma, alpha) that maximizes CV score per target

Memory efficiency strategies (Principle 2: automatic memory efficiency):

- Generator pattern for alpha batching (via _decompose_ridge): Processes alphas
  in batches to avoid storing all resolution matrices simultaneously
- Target batching (n_targets_batch): Processes targets in chunks to fit GPU memory
- Y_in_cpu strategy: Keeps large Y on CPU, transfers only batches needed
  for computation
- Immediate cleanup with del statements: Explicitly frees memory after each batch

Performance:

- Time complexity: O(n_iter × n_splits × (n_alphas_batch × n_features^2 + n_targets_batch × n_samples))
- Memory complexity: O(n_features × n_targets_batch) per batch
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)

See ``nltools.algorithms.ridge.utils._decompose_ridge()`` for generator pattern details.
See ``docs/development/ridge-internals.md`` for detailed algorithm explanation.

</details>

(algorithms-ridge-solve-ridge-cv)=
### `solve_ridge_cv`

```python
solve_ridge_cv(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Solve ridge regression with cross-validation.

This function solves ridge regression for a single feature space with
cross-validation for hyperparameter selection.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Feature matrix of shape (n_samples, n_features). | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). | *required*
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray) \| [list](#list)[[float](#float)]</code> | Ridge regularization parameters to try. Defaults to [0.1, 1.0, 10.0]. | <code>[0.1, 1.0, 10.0]</code>
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits. Defaults to 5. | <code>5</code>
`local_alpha` | <code>[bool](#bool)</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. Defaults to True. | <code>True</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during CV (for memory efficiency). If None, processes all targets at once. Defaults to None. | <code>None</code>
`n_targets_batch_refit` | <code>[int](#int) \| None</code> | Batch size for targets during refit. If None, uses n_targets_batch value. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas (for memory efficiency). If None, processes all alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to GPU as needed. This prevents OOM when Y is large (e.g., 300k voxels). Defaults to True (recommended for neuroimaging). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Scoring function (y_true, y_pred) -> scores. If None, uses R² score. Defaults to None. | <code>None</code>
`fit_intercept` | <code>[bool](#bool)</code> | Whether to fit an intercept. If False, X and Y should be centered. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display progress bar (requires tqdm). Defaults to False. | <code>False</code>
`conservative` | <code>[bool](#bool)</code> | If True, select largest alpha within 1 std of best score. Defaults to False. | <code>False</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'best_alphas': Selected best alpha for each target (or same alpha repeated         if local_alpha=False), shape (n_targets,).     - 'coefs': Ridge coefficients refit on entire dataset using best alphas,         shape (n_features, n_targets). Always returned on CPU (numpy array).     - 'cv_scores': Cross-validation scores for best alphas, shape (n_splits, n_alphas, n_targets).         Always returned on CPU (numpy array).     - 'intercept': Per-target intercept of shape (n_targets,). Only present         when ``fit_intercept=True``.     - 'backend': Backend used (for transparency).

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> Y = np.random.randn(100, 10)
>>> result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
>>> alphas = result['best_alphas']
>>> coefs = result['coefs']
>>> scores = result['cv_scores']
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

This is the efficient implementation for single feature space ridge regression
with cross-validation. For multiple feature spaces (banded/group ridge),
use solve_banded_ridge_cv instead.

Algorithm details:

- Cross-validation: k-fold CV evaluates each alpha value
- Alpha selection: Chooses best alpha per target (or globally if local_alpha=False)
- Refit: Fits final model on full dataset using best alpha(s)

Memory efficiency strategies (Principle 2: automatic memory efficiency):

- Generator pattern for alpha batching (via _decompose_ridge): Processes alphas
  in batches to avoid storing all resolution matrices simultaneously
- Target batching (n_targets_batch): Processes targets in chunks to fit GPU memory
- Y_in_cpu strategy: Keeps large Y on CPU, transfers only batches needed
  for computation
- Immediate cleanup with del statements: Explicitly frees memory after each batch

Performance:

- Time complexity: O(n_splits × (n_alphas_batch × n_features^2 + n_targets_batch × n_samples))
- Memory complexity: O(n_features × n_targets_batch) per batch
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)

See ``nltools.algorithms.ridge.utils._decompose_ridge()`` for generator pattern details.
See ``docs/development/ridge-internals.md`` for detailed algorithm explanation.

</details>
