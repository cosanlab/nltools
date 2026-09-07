---
title: Prediction & cross-validation
label: page-tasks-prediction
---

Decode or predict from brain data. `BrainData.predict` and `BrainCollection.predict_group` run the workflow. Listed here are the pieces they accept or return: the cross-validation schemes (`resolve_cv` turns an int, a name, or an sklearn splitter into one), the ridge solvers behind `model='ridge'` (CPU or GPU), `Roc` for a classifier's output, and the plots of weights, margins, and predictions.

**Classes:**

Name | Description
---- | -----------
[`KFoldStratified`](#tasks-prediction-kfoldstratified) | Stratify continuous targets across K-fold cross-validation.
[`Roc`](#tasks-prediction-roc) | Compute receiver operating characteristic curves for single-interval or forced-choice classification.

**Functions:**

Name | Description
---- | -----------
[`resolve_cv`](#tasks-prediction-resolve-cv) | Resolve a cv spec (int, sklearn-style name, or splitter) into an sklearn splitter.
[`ridge_cv`](#tasks-prediction-ridge-cv) | Ridge regression with cross-validated selection of a single global alpha.
[`ridge_svd`](#tasks-prediction-ridge-svd) | Solve ridge regression for one alpha using the singular value decomposition.
[`plot_roc`](#tasks-prediction-plot-roc) | Plot 1-Specificity by Sensitivity.
[`plot_dist_from_hyperplane`](#tasks-prediction-plot-dist-from-hyperplane) | Plot SVM Classification Distance from Hyperplane.
[`plot_probability`](#tasks-prediction-plot-probability) | Plot Classification Probability.
[`plot_scatter`](#tasks-prediction-plot-scatter) | Plot Prediction Scatterplot.

## Classes

(tasks-prediction-kfoldstratified)=
### `KFoldStratified`

```python
KFoldStratified(n_splits = 3, *, shuffle = False, random_state = None)
```

Stratify continuous targets across K-fold cross-validation.

Unlike the scikit-learn equivalent, this iterator stratifies continuous data.

Provides train/test indices to split data in train test sets. Samples are
ordered by their continuous target `y` and dealt round-robin into k folds
so each fold spans the full range of `y`. Each fold is then used as a
validation set once while the k - 1 remaining folds form the training set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_splits` | <code>int</code> | Number of folds. Must be at least 2. Defaults to 3. | <code>3</code>
`shuffle` | <code>bool</code> | Whether to break ties in `y` randomly before dealing samples into folds. Default False. | <code>False</code>
`random_state` | <code>int \| RandomState</code> | Seed or RandomState for the tie-break shuffle. If None, use the default numpy RNG. | <code>None</code>

**Methods:**

Name | Description
---- | -----------
[`split`](#tasks-prediction-split) | Generate indices to split data into training and test set.



#### Methods

(tasks-prediction-split)=
##### `split`

```python
split(X, y = None, groups = None)
```

Generate indices to split data into training and test set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>array - like</code> | Training data of shape `(n_samples, n_features)`. Only `y` is needed to generate the splits, so `np.zeros(n_samples)` works as a placeholder. | *required*
`y` | <code>array - like</code> | Continuous target of shape `(n_samples,)`; stratification is based on its ordering. | <code>None</code>
`groups` | <code>array - like</code> | Always ignored; exists for sklearn compatibility. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>tuple[ndarray, ndarray]</code> | `(train, test)` — the training set indices     and the testing set indices for that split.

(tasks-prediction-roc)=
### `Roc`

```python
Roc(*, input_values = None, binary_outcome = None, method = 'optimal_overall', forced_choice = None)
```

Compute receiver operating characteristic curves for single-interval or forced-choice classification.

The Roc class is based on Tor Wager's Matlab roc_plot.m function and
allows a user to easily run different types of receiver operator
characteristic curves.  For example, one might be interested in single
interval or forced choice.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` | <code>array - like</code> | 1-D continuous decision values, one per observation. | <code>None</code>
`binary_outcome` | <code>array - like</code> | Boolean class label per observation. | <code>None</code>
`method` | <code>str</code> | Threshold-selection variant, one of `'optimal_overall'`, `'optimal_balanced'`, `'minimum_sdt_bias'`. | <code>'optimal_overall'</code>
`forced_choice` | <code>array - like</code> | Subject id per observation for forced-choice classification (each subject contributes one positive and one negative observation). | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`input_values` | <code>ndarray</code> | Decision values.
`binary_outcome` | <code>ndarray</code> | Boolean labels.
`method` | <code>str</code> | Threshold-selection variant.
`forced_choice` | <code>ndarray \| None</code> | Subject ids for forced-choice classification.
`criterion_values` | <code>ndarray</code> | Thresholds at which `tpr`/`fpr` were evaluated; set by `calculate`.
`tpr` | <code>ndarray</code> | True positive rate per criterion value; set by `calculate`.
`fpr` | <code>ndarray</code> | False positive rate per criterion value; set by `calculate`.
`auc` | <code>float</code> | Area under the ROC curve; set by `calculate`.
`class_thr` | <code>float</code> | Selected classification threshold; set by `calculate`.
`sensitivity` | <code>float</code> | Sensitivity at `class_thr`; set by `calculate`.
`specificity` | <code>float</code> | Specificity at `class_thr`; set by `calculate`.
`ppv` | <code>float</code> | Positive predictive value at `class_thr`; set by `calculate`.
`accuracy` | <code>float</code> | Classification accuracy; set by `calculate`.
`accuracy_se` | <code>float</code> | Standard error of the accuracy; set by `calculate`.
`accuracy_p` | <code>BinomTestResult</code> | `scipy.stats.binomtest` result comparing accuracy against chance (read `.pvalue`); set by `calculate`.

**Methods:**

Name | Description
---- | -----------
[`calculate`](#tasks-prediction-calculate) | Calculate ROC metrics and store them on the instance.
[`plot`](#tasks-prediction-plot) | Create a ROC plot.
[`summary`](#tasks-prediction-summary) | Display a formatted summary of ROC analysis.



#### Methods

(tasks-prediction-calculate)=
##### `calculate`

```python
calculate(*, input_values = None, binary_outcome = None, criterion_values = None, method = 'optimal_overall', forced_choice = None, balanced_acc = False, tail = 2)
```

Calculate ROC metrics and store them on the instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` | <code>array - like</code> | 1-D continuous decision values, one per observation. Defaults to the values given at construction. | <code>None</code>
`binary_outcome` | <code>array - like</code> | Boolean class label per observation. Defaults to the labels given at construction. | <code>None</code>
`criterion_values` | <code>array - like</code> | Thresholds at which to evaluate `fpr` and `tpr`. Defaults to a dense grid over the range of `input_values`. | <code>None</code>
`method` | <code>str</code> | Threshold-selection variant, one of `'optimal_overall'`, `'optimal_balanced'`, `'minimum_sdt_bias'`. | <code>'optimal_overall'</code>
`forced_choice` | <code>array - like</code> | Subject id per observation for forced-choice classification. | <code>None</code>
`balanced_acc` | <code>bool</code> | Report balanced accuracy (mean of sensitivity and specificity) instead of overall accuracy. Only affects the accuracy estimate, not the p-value or the threshold used for sensitivity/specificity. | <code>False</code>
`tail` | <code>int \| str</code> | `2`/`'two'` for two-tailed (default); `1`/`'one'` for one-tailed (accuracy > chance) in the binomial test for `accuracy_p`. | <code>2</code>

(tasks-prediction-plot)=
##### `plot`

```python
plot(*, method = 'gaussian', balanced_acc = False)
```

Create a ROC plot.

Runs `calculate` first, then plots either a Gaussian-smoothed ROC curve fit
to the decision values or the observed empirical curve.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | Type of plot, `'gaussian'` or `'observed'`. | <code>'gaussian'</code>
`balanced_acc` | <code>bool</code> | Passed to `calculate`; report balanced accuracy. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The ROC figure.

(tasks-prediction-summary)=
##### `summary`

```python
summary()
```

Display a formatted summary of ROC analysis.

## Functions

(tasks-prediction-resolve-cv)=
### `resolve_cv`

```python
resolve_cv(cv, *, groups = None, classifier: bool = False, shuffle: bool = False, random_state: int | None = None)
```

Resolve a cv spec (int, sklearn-style name, or splitter) into an sklearn splitter.

The single cv-resolution rule shared by `BrainData.predict`,
`BrainCollection.predict`, and `BrainCollection.predict_group`. String
names follow sklearn's splitter classes. An int spec honors `groups` when
one is supplied (it becomes a `GroupKFold` variant rather than a plain
`KFold`, which would ignore the groups).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cv` | <code>str \| int \| BaseCrossValidator</code> | `'loo'` (`LeaveOneOut`), `'logo'` (`LeaveOneGroupOut` — pass the grouping variable via `groups`), an int fold count, or an sklearn splitter (returned unchanged). | *required*
`groups` | <code>array - like</code> | Group labels. Only consulted for int specs. | <code>None</code>
`classifier` | <code>bool</code> | Whether the downstream model is a classifier — an int spec becomes the stratified variant (`StratifiedKFold`, or `StratifiedGroupKFold` with groups) for classifiers. | <code>False</code>
`shuffle` | <code>bool</code> | Whether an int spec's KFold variant shuffles samples before splitting. Ignored for the group variants (fold membership is set by `groups`). | <code>False</code>
`random_state` | <code>int</code> | Seed for `shuffle`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>BaseCrossValidator</code> | An sklearn splitter instance.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | On an unknown string spec, including the pre-v0.6.0 names `'loso'` / `'loro'` (use `'logo'` with `groups=`).

(tasks-prediction-ridge-cv)=
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
`parallel` | <code>str \| None</code> | Execution backend. `None` or `"cpu"` runs on NumPy; `"gpu"` requires a CUDA or MPS accelerator; `"auto"` may use a Torch CPU backend when no accelerator is available. Defaults to `"cpu"`. | <code>'cpu'</code>
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

(tasks-prediction-ridge-svd)=
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
`parallel` | <code>str \| None</code> | Execution backend. `None` or `"cpu"` runs on NumPy; `"gpu"` requires a CUDA or MPS accelerator; `"auto"` may use a Torch CPU backend when no accelerator is available. Defaults to None. | <code>None</code>
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

(tasks-prediction-plot-roc)=
### `plot_roc`

```python
plot_roc(fpr, tpr)
```

Plot 1-Specificity by Sensitivity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fpr` | <code>ndarray</code> | False positive rate per criterion value, from `Roc.calculate`. | *required*
`tpr` | <code>ndarray</code> | True positive rate per criterion value, from `Roc.calculate`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The ROC figure.

(tasks-prediction-plot-dist-from-hyperplane)=
### `plot_dist_from_hyperplane`

```python
plot_dist_from_hyperplane(stats_output)
```

Plot SVM Classification Distance from Hyperplane.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` | <code>DataFrame</code> | Prediction output table (e.g. from `BrainData.predict`). | *required*

**Returns:**

Type | Description
---- | -----------
<code>FacetGrid</code> | Distance from the hyperplane per sample.

(tasks-prediction-plot-probability)=
### `plot_probability`

```python
plot_probability(stats_output)
```

Plot Classification Probability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` | <code>DataFrame</code> | Prediction output table (e.g. from `BrainData.predict`). | *required*

**Returns:**

Type | Description
---- | -----------
<code>FacetGrid</code> | Scatterplot.

(tasks-prediction-plot-scatter)=
### `plot_scatter`

```python
plot_scatter(stats_output)
```

Plot Prediction Scatterplot.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` | <code>DataFrame</code> | Prediction output table (e.g. from `BrainData.predict`). | *required*

**Returns:**

Type | Description
---- | -----------
<code>FacetGrid</code> | Scatterplot.
