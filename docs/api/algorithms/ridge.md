---
title: algorithms.ridge
label: page-algorithms-ridge
---

Ridge regression algorithms and utilities.

Ridge regression solvers with cross-validated alpha selection (per target or
global), memory-efficient batching for large problems, optional GPU acceleration
(roughly 10-100× faster on large datasets), and banded ridge for multiple
feature spaces. `solve_ridge_cv` and `solve_banded_ridge_cv` are the main entry
points; `ridge_svd` and `ridge_cv` are simpler single-alpha and CV solvers.
`BrainData.fit(model='ridge')` wraps these for brain data.

**Functions:**

Name | Description
---- | -----------
[`cross_val_predict_ridge`](#algorithms-ridge-cross-val-predict-ridge) | Held-out ridge predictions per CV fold at a fixed (per-target) alpha.
[`generate_dirichlet_samples`](#algorithms-ridge-generate-dirichlet-samples) | Generate samples from a Dirichlet distribution.
[`ridge_cv`](#algorithms-ridge-ridge-cv) | Ridge regression with cross-validated selection of a single global alpha.
[`ridge_svd`](#algorithms-ridge-ridge-svd) | Solve ridge regression for one alpha using the singular value decomposition.
[`solve_banded_ridge_cv`](#algorithms-ridge-solve-banded-ridge-cv) | Solve banded (group) ridge regression with cross-validated random search.
[`solve_ridge_cv`](#algorithms-ridge-solve-ridge-cv) | Solve ridge regression for one feature space with cross-validated alphas.



**Examples:**

```python
import numpy as np
from nltools.algorithms.ridge import solve_ridge_cv

X = np.random.randn(100, 50)
Y = np.random.randn(100, 10)
result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
```

## Functions

(algorithms-ridge-cross-val-predict-ridge)=
### `cross_val_predict_ridge`

```python
cross_val_predict_ridge(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, n_targets_batch: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None) -> dict[str, Any]
```

Held-out ridge predictions per CV fold at a fixed (per-target) alpha.

For each fold, ridge is refit on the training fold with the supplied alpha
(scalar or per target) and the held-out fold is predicted. Targets sharing
an alpha share one SVD of the training fold, so the cost scales with the
number of *unique* alphas, not the number of targets.

This is how `BrainData` obtains held-out predictions once `solve_ridge_cv`
has selected alphas: pass the selected per-voxel alphas back through here to
get the fold-by-fold predictions and per-fold R² for `cv_results_`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | Feature matrix of shape (n_samples, n_features). | *required*
`Y` | <code>ndarray</code> | Targets of shape (n_samples, n_targets); a 1D `Y` is promoted to (n_samples, 1). | *required*
`alphas` | <code>float \| ndarray</code> | Per-target alphas of shape (n_targets,), or a scalar broadcast to every target. | *required*
`cv` | <code>int \| BaseCrossValidator</code> | Number of folds for unshuffled `KFold`, or an sklearn cross-validator. Generators (e.g. `KFold(5).split(X)`) are rejected; pass the splitter object. Defaults to 5. | <code>5</code>
`fit_intercept` | <code>bool</code> | If True, center `X` and `Y` on the training fold's means (sklearn convention) and add the intercept back so predictions are on the original `Y` scale. Defaults to False. | <code>False</code>
`n_targets_batch` | <code>int \| None</code> | Targets per batch during the refit. None processes all targets at once on the CPU; with `parallel="gpu"` None derives a batch size from `max_gpu_memory_gb`. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>int \| None</code> | Alphas per batch. None processes all unique alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>bool</code> | If True, keep `Y` on the CPU and move one fold's training targets at a time to the device (recommended for large neuroimaging `Y`). Defaults to True. | <code>True</code>
`score_func` | <code>Callable \| None</code> | Per-fold scoring function `(y_true, y_pred) -> per-target scores`, evaluated on NumPy arrays. None uses R², computed in NumPy on the CPU. Defaults to None. | <code>None</code>
`parallel` | <code>str \| None</code> | Execution backend. `None` or `"cpu"` runs on NumPy; `"gpu"` runs on PyTorch (requires torch; falls back to the torch CPU device when no GPU is present); `"auto"` uses torch when installed and NumPy otherwise. Defaults to `"cpu"`. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB used to derive `n_targets_batch` when `parallel="gpu"`. None measures the device. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'predictions'` (np.ndarray, held-out predictions on the     original `Y` scale, shape (n_samples, n_targets)), `'folds'`     (np.ndarray, fold index per row, shape (n_samples,)), `'scores'`     (np.ndarray, per-fold R² or `score_func` output, shape (n_splits,     n_targets)), and `'backend'` (str, backend name). Arrays are NumPy on     the CPU.

**Raises:**

Type | Description
---- | -----------
<code>TypeError</code> | If `cv` is a single-use generator rather than a splitter.
<code>ValueError</code> | If `alphas` does not broadcast to `(n_targets,)`, or `X` and `Y` disagree on `n_samples`.

(algorithms-ridge-generate-dirichlet-samples)=
### `generate_dirichlet_samples`

```python
generate_dirichlet_samples(n_samples: int, n_kernels: int, concentration: float | list[float] = [0.1, 1.0], random_state: int | None = None) -> np.ndarray
```

Generate samples from a Dirichlet distribution.

Used to draw candidate feature-space weights (gamma) for the random search
in banded ridge regression.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>int</code> | Number of samples to generate. | *required*
`n_kernels` | <code>int</code> | Number of dimensions (feature spaces) of the distribution. | *required*
`concentration` | <code>float \| list[float]</code> | Concentration parameter(s). A value of 1 samples uniformly over the simplex; `np.inf` gives equal weights; a list alternates between its values across samples. Defaults to `[0.1, 1.0]`. | <code>[0.1, 1.0]</code>
`random_state` | <code>int \| None</code> | Random generator seed; use an int for deterministic samples. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Samples of shape (n_samples, n_kernels); each row sums to 1.

**Examples:**

```python
gammas = generate_dirichlet_samples(10, 3, concentration=[0.1, 1.0])
gammas.shape  # → (10, 3)
np.allclose(gammas.sum(axis=1), 1.0)  # → True
```

(algorithms-ridge-ridge-cv)=
### `ridge_cv`

```python
ridge_cv(X: np.ndarray, y: np.ndarray, *, alphas: np.ndarray | None = None, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict
```

Ridge regression with cross-validated selection of a single global alpha.

Scores every alpha by out-of-fold R² on each fold, picks the alpha with the
highest mean R² across folds and targets, then refits on all the data with
it. For per-target alphas, memory-bounded batching, and GPU-batched folds
use `solve_ridge_cv`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | Training features, shape (n_samples, n_features). | *required*
`y` | <code>ndarray</code> | Targets, shape (n_samples,) or (n_samples, n_targets). | *required*
`alphas` | <code>ndarray \| None</code> | Alpha values to try. None uses `np.logspace(-2, 4, 20)` (0.01 to 10000). Defaults to None. | <code>None</code>
`cv` | <code>int \| BaseCrossValidator</code> | Number of folds, or an sklearn cross-validator (anything with `.split(X)` and `.get_n_splits()`, e.g. `KFold(5, shuffle=True)` or `GroupKFold(8)`). The splitter drives the actual fold iteration, so leave-one-run-out and shuffled K-fold give different results from contiguous K-fold. Defaults to 5. | <code>5</code>
`fit_intercept` | <code>bool</code> | If True, center `X` and `y` on their means before fitting and recover the intercept afterwards. The returned `coef` is on the centered scale; the intercept is returned under the `'intercept'` key. Defaults to False. | <code>False</code>
`parallel` | <code>str \| None</code> | Execution backend. `None` or `"cpu"` runs on NumPy; `"gpu"` runs on PyTorch (requires torch, raising ImportError otherwise, and falls back to the torch CPU device when no GPU is present — it never falls back to NumPy); `"auto"` uses torch when installed and NumPy otherwise. Defaults to `"cpu"`. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB for batching over targets (torch backends only). None measures the device. Defaults to None. | <code>None</code>
`random_state` | <code>int \| None</code> | Unused; accepted for signature consistency. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'alpha'` (float, the selected alpha), `'coef'` (np.ndarray,     coefficients refit on all data with that alpha), `'cv_scores'`     (np.ndarray, out-of-fold R² with shape (n_folds, n_alphas,     n_targets)), `'backend'` (str, backend name), and — only when     `fit_intercept=True` — `'intercept'` (float or np.ndarray).

**Raises:**

Type | Description
---- | -----------
<code>TypeError</code> | If `cv` is a generator rather than a re-iterable splitter.

**Examples:**

```python
X = np.random.randn(100, 50)
y = np.random.randn(100)
result = ridge_cv(X, y, cv=3)
result["alpha"]  # → the selected alpha
result["coef"].shape  # → (50,)
```

(algorithms-ridge-ridge-svd)=
### `ridge_svd`

```python
ridge_svd(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0, parallel: str | None = None, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> np.ndarray
```

Solve ridge regression for one alpha using the singular value decomposition.

With `X = U @ diag(s) @ V.T` the solution is
`beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y`; the shrinkage factor
`s / (s**2 + alpha)` damps small singular values without an explicit matrix
inverse. Time is `O(n_samples × n_features × min(n_samples, n_features))`
and memory `O(n_samples × n_features)`. As `alpha → 0` this approaches
ordinary least squares; use `alpha=1e-6` rather than 0 for a stable OLS fit.
For cross-validated alpha selection use `solve_ridge_cv`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | Training features, shape (n_samples, n_features). | *required*
`y` | <code>ndarray</code> | Targets, shape (n_samples,) for a single target or (n_samples, n_targets) for several. | *required*
`alpha` | <code>float</code> | Regularization strength; must be non-negative. Larger values shrink the coefficients harder toward zero. Defaults to 1.0. | <code>1.0</code>
`parallel` | <code>str \| None</code> | Execution backend. `None` or `"cpu"` runs on NumPy; `"gpu"` runs on PyTorch (requires torch, raising ImportError otherwise, and falls back to the torch CPU device when no GPU is present); `"auto"` uses torch when installed and NumPy otherwise. Defaults to None. | <code>None</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB for batching over targets (torch backends only). None measures the device. Defaults to None. | <code>None</code>
`random_state` | <code>int \| None</code> | Unused; accepted for signature consistency. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Coefficients, shape (n_features,) for a single target or     (n_features, n_targets) for several.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `alpha` is negative, `X` is not 2D, `y` is not 1D or 2D, or the sample counts differ.

**Examples:**

```python
X = np.random.randn(100, 50)
y = np.random.randn(100)
ridge_svd(X, y, alpha=1.0).shape  # → (50,)

Y = np.random.randn(100, 5)  # multi-target
ridge_svd(X, Y, alpha=1.0).shape  # → (50, 5)
```

(algorithms-ridge-solve-banded-ridge-cv)=
### `solve_banded_ridge_cv`

```python
solve_banded_ridge_cv(Xs: list[np.ndarray], Y: np.ndarray, *, n_iter: int | np.integer | np.ndarray = 100, concentration: float | list[float] = [0.1, 1.0], alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, jitter_alphas: bool = False, return_weights: bool = True, diagonalize_method: str = 'svd', warn: bool = True, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Solve banded (group) ridge regression with cross-validated random search.

Banded ridge gives each feature space its own scale: `Z_i = sqrt(gamma_i) * X_i`,
then ordinary ridge is solved on the concatenated `Z`, so the relative
importance of the feature spaces is learned. The weights `gamma` are drawn
from a Dirichlet distribution (`n_iter` draws; the first is equal weights);
for each draw every alpha is scored by k-fold cross-validation, and each
target keeps the `(gamma, alpha)` pair with the best score (himalaya's
`solve_group_ridge_random_search`). For a single feature space use
`solve_ridge_cv`.

**Memory.** Alphas are processed in batches of `n_alphas_batch` from one SVD
per fold, targets in batches of `n_targets_batch`, and with `Y_in_cpu=True`
only the current target batch is moved to the device. Time scales as
`O(n_iter × n_splits × (n_alphas_batch × n_features² + n_targets_batch ×
n_samples))`, memory as `O(n_features × n_targets_batch)` per batch; a GPU
is roughly 10-100× faster once `n_features` exceeds ~10K.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`Xs` | <code>list[ndarray]</code> | One feature matrix per feature space, each of shape (n_samples, n_features_i) with the same `n_samples`. | *required*
`Y` | <code>ndarray</code> | Targets of shape (n_samples, n_targets). | *required*
`n_iter` | <code>int \| ndarray</code> | Number of feature-space weight combinations to sample, or an explicit array of weights with shape (n_iter, n_spaces) to try instead of sampling. Defaults to 100. | <code>100</code>
`concentration` | <code>float \| list[float]</code> | Dirichlet concentration parameter(s). A value of 1 samples uniformly over the simplex; `np.inf` gives equal weights; a list alternates between its values. Ignored when `n_iter` is an array. Defaults to `[0.1, 1.0]`. | <code>[0.1, 1.0]</code>
`alphas` | <code>float \| ndarray \| list[float]</code> | Ridge regularization parameters to try. Defaults to `[0.1, 1.0, 10.0]`. | <code>[0.1, 1.0, 10.0]</code>
`cv` | <code>int \| BaseCrossValidator</code> | Number of folds for unshuffled `KFold`, or an sklearn cross-validator. Defaults to 5. | <code>5</code>
`local_alpha` | <code>bool</code> | If True, pick the best alpha independently per target; if False, one alpha for all targets. Defaults to True. | <code>True</code>
`n_targets_batch` | <code>int \| None</code> | Targets per batch during CV. None processes all targets at once on the CPU; with `parallel="gpu"` None derives a batch size from `max_gpu_memory_gb`. Defaults to None. | <code>None</code>
`n_targets_batch_refit` | <code>int \| None</code> | Targets per batch during the refit. None reuses `n_targets_batch`. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>int \| None</code> | Alphas per batch. None processes all alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>bool</code> | If True, keep `Y` on the CPU and move one target batch at a time to the device, which avoids running out of GPU memory on large `Y` (e.g. 300k voxels). Defaults to True. | <code>True</code>
`score_func` | <code>Callable \| None</code> | Scoring function `(y_true, y_pred) -> per-target scores`. None uses R². Defaults to None. | <code>None</code>
`fit_intercept` | <code>bool</code> | If True, center `X` and `Y` per training fold and return the intercept. If False, `X` and `Y` should already be centered. Defaults to False. | <code>False</code>
`progress_bar` | <code>bool</code> | Show a progress bar over the gamma draws (requires tqdm). Defaults to False. | <code>False</code>
`conservative` | <code>bool</code> | If True, pick the largest alpha within one standard deviation of the best score (more regularization at similar performance). Defaults to False. | <code>False</code>
`jitter_alphas` | <code>bool</code> | If True, multiply the alpha grid by a random factor in `[10^-0.5, 10^0.5]` for each gamma draw. Defaults to False. | <code>False</code>
`return_weights` | <code>bool</code> | If True, refit on the full data with the selected hyperparameters and return the coefficients. Defaults to True. | <code>True</code>
`diagonalize_method` | <code>str</code> | Feature decomposition; only `"svd"` is supported. Defaults to `"svd"`. | <code>'svd'</code>
`warn` | <code>bool</code> | If True, warn when `n_samples < n_features`, where banded ridge is slower than kernel ridge. Defaults to True. | <code>True</code>
`parallel` | <code>str \| None</code> | Execution backend. `None` or `"cpu"` runs on NumPy; `"gpu"` runs on PyTorch (requires torch; falls back to the torch CPU device when no GPU is present); `"auto"` uses torch when installed and NumPy otherwise. Defaults to `"cpu"`. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB used to derive `n_targets_batch` when `parallel="gpu"`. None measures the device. Defaults to None. | <code>None</code>
`random_state` | <code>int \| None</code> | Random generator seed; use an int for a deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'deltas'` (np.ndarray, `log(gamma / alpha)` per feature space     and target, shape (n_spaces, n_targets)), `'cv_scores'` (np.ndarray,     split-averaged score at the selected alpha for each gamma draw,     shape (n_iter, n_targets)), `'backend'` (str, backend name), and —     only when `return_weights=True` — `'coefs'` (np.ndarray, refit     coefficients on the unscaled features, shape (n_features_total,     n_targets)) plus, when `fit_intercept=True` as well, `'intercept'`     (np.ndarray, shape (n_targets,)). Arrays are always NumPy on the CPU.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `Xs` is empty, the feature spaces or `Y` disagree on `n_samples`, or `n_iter` is neither an integer nor a 2D array with `n_spaces` columns.

**Examples:**

```python
X1 = np.random.randn(100, 30)  # first feature space
X2 = np.random.randn(100, 20)  # second feature space
Y = np.random.randn(100, 10)
result = solve_banded_ridge_cv([X1, X2], Y, n_iter=50, alphas=[0.1, 1.0, 10.0])
result["deltas"].shape  # → (2, 10)
result["coefs"].shape  # → (50, 10)
result["cv_scores"].shape  # → (50, 10)
```

(algorithms-ridge-solve-ridge-cv)=
### `solve_ridge_cv`

```python
solve_ridge_cv(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Solve ridge regression for one feature space with cross-validated alphas.

Every alpha is scored by k-fold cross-validation, the best alpha is chosen
per target (or once for all targets with `local_alpha=False`), and the model
is refit on the full data with the chosen alphas. For several feature spaces
use `solve_banded_ridge_cv`; for held-out predictions at already-selected
alphas use `cross_val_predict_ridge`.

**Memory.** Alphas are processed in batches of `n_alphas_batch` from one SVD
per fold, targets in batches of `n_targets_batch`, and with `Y_in_cpu=True`
only the current target batch is moved to the device. Time scales as
`O(n_splits × (n_alphas_batch × n_features² + n_targets_batch × n_samples))`,
memory as `O(n_features × n_targets_batch)` per batch; a GPU is roughly
10-100× faster once `n_features` exceeds ~10K.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | Feature matrix of shape (n_samples, n_features). | *required*
`Y` | <code>ndarray</code> | Targets of shape (n_samples, n_targets). | *required*
`alphas` | <code>float \| ndarray \| list[float]</code> | Ridge regularization parameters to try. Defaults to `[0.1, 1.0, 10.0]`. | <code>[0.1, 1.0, 10.0]</code>
`cv` | <code>int \| BaseCrossValidator</code> | Number of folds for unshuffled `KFold`, or an sklearn cross-validator. Defaults to 5. | <code>5</code>
`local_alpha` | <code>bool</code> | If True, pick the best alpha independently per target; if False, one alpha for all targets. Defaults to True. | <code>True</code>
`n_targets_batch` | <code>int \| None</code> | Targets per batch during CV. None processes all targets at once on the CPU; with `parallel="gpu"` None derives a batch size from `max_gpu_memory_gb`. Defaults to None. | <code>None</code>
`n_targets_batch_refit` | <code>int \| None</code> | Targets per batch during the refit. None reuses `n_targets_batch`. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>int \| None</code> | Alphas per batch. None processes all alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>bool</code> | If True, keep `Y` on the CPU and move one target batch at a time to the device, which avoids running out of GPU memory on large `Y` (e.g. 300k voxels). Defaults to True. | <code>True</code>
`score_func` | <code>Callable \| None</code> | Scoring function `(y_true, y_pred) -> per-target scores`. None uses R². Defaults to None. | <code>None</code>
`fit_intercept` | <code>bool</code> | If True, center `X` and `Y` per training fold and return the intercept. If False, `X` and `Y` should already be centered. Defaults to False. | <code>False</code>
`progress_bar` | <code>bool</code> | Accepted for API symmetry with `solve_banded_ridge_cv`; this solver shows no progress bar. Defaults to False. | <code>False</code>
`conservative` | <code>bool</code> | If True, pick the largest alpha within one standard deviation of the best score (more regularization at similar performance). Defaults to False. | <code>False</code>
`parallel` | <code>str \| None</code> | Execution backend. `None` or `"cpu"` runs on NumPy; `"gpu"` runs on PyTorch (requires torch; falls back to the torch CPU device when no GPU is present); `"auto"` uses torch when installed and NumPy otherwise. Defaults to `"cpu"`. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB used to derive `n_targets_batch` when `parallel="gpu"`. None measures the device. Defaults to None. | <code>None</code>
`random_state` | <code>int \| None</code> | Unused by this solver (the search is deterministic); accepted for signature consistency. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'best_alphas'` (np.ndarray, selected alpha per target — the     same value repeated when `local_alpha=False` — shape (n_targets,)),     `'coefs'` (np.ndarray, coefficients refit on the full data, shape     (n_features, n_targets)), `'cv_scores'` (np.ndarray, per-fold score     of every alpha, shape (n_splits, n_alphas, n_targets)), `'backend'`     (str, backend name), and — only when `fit_intercept=True` —     `'intercept'` (np.ndarray, shape (n_targets,)). Arrays are always     NumPy on the CPU.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `X` and `Y` disagree on `n_samples`.

**Examples:**

```python
X = np.random.randn(100, 50)
Y = np.random.randn(100, 10)
result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
result["best_alphas"].shape  # → (10,)
result["coefs"].shape  # → (50, 10)
result["cv_scores"].shape  # → (5, 3, 10)
```
