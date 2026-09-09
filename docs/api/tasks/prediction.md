---
title: Prediction & cross-validation
label: page-tasks-prediction
---

Decode or predict from brain data. `BrainData.predict` runs the workflow. Listed here are the pieces it accepts or returns: the cross-validation schemes (`resolve_cv` turns an int, a name, or an sklearn splitter into one), the `Ridge` estimator behind `model='ridge'` (CPU or GPU), `Roc` for a classifier's output, and the plots of weights, margins, and predictions.

**Classes:**

Name | Description
---- | -----------
[`KFoldStratified`](#tasks-prediction-kfoldstratified) | Stratify continuous targets across K-fold cross-validation.
[`Ridge`](#tasks-prediction-ridge) | Ridge regression over one or several named feature spaces.
[`Roc`](#tasks-prediction-roc) | Compute receiver operating characteristic curves for single-interval or forced-choice classification.

**Functions:**

Name | Description
---- | -----------
[`resolve_cv`](#tasks-prediction-resolve-cv) | Resolve a cv spec (int, sklearn-style name, or splitter) into an sklearn splitter.
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

(tasks-prediction-ridge)=
### `Ridge`

```python
Ridge(*, alpha: float | Sequence[float] | np.ndarray = 1.0, cv: float | Sequence[float] | np.ndarray = None, search_iterations: int = 100, dirichlet_concentration: float | Sequence[float] = (0.1, 1.0), device: str = 'cpu', memory_budget_gb: float | None = None, per_target_alpha: bool = True, prefer_conservative_alpha: bool = False, random_state: int | None = None, progress_bar: bool = False)
```

Ridge regression over one or several named feature spaces.

Fits `argmin_b ||X @ b - y||^2 + alpha * ||b||^2` without an intercept.
Callers own preprocessing: `Ridge` never centers, scales, standardizes, or
adds an intercept column.

A two-dimensional `X` fits ordinary Ridge. A mapping from names to
two-dimensional arrays fits banded Ridge, which searches feature-space
weights on the simplex jointly with the alphas.

Himalaya defines the numerical behavior: the cross-validation loss is
negative mean squared error, and alpha selection, the Dirichlet search, and
coefficient refitting all come from its solvers.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>float \| Sequence[float] \| ndarray</code> | A positive finite scalar fits a fixed alpha and requires `cv=None`. A non-empty one-dimensional collection of positive finite values selects an alpha by cross-validation and requires `cv`. Default: `1.0`. | <code>1.0</code>
`cv` | <code>int \| BaseCrossValidator \| None</code> | An integer builds unshuffled K-fold splits; a reusable scikit-learn cross-validator is used as given. A single-use split generator is invalid because fitting traverses the splits more than once. Default: `None`. | <code>None</code>
`search_iterations` | <code>int</code> | Number of sampled feature-space weight vectors for banded Ridge. Default: `100`. | <code>100</code>
`dirichlet_concentration` | <code>float \| Sequence[float]</code> | Concentration parameter(s) of the Dirichlet distribution the candidate weights are drawn from. A list is cycled through across candidates. Default: `(0.1, 1.0)`. | <code>(0.1, 1.0)</code>
`device` | <code>str</code> | `'cpu'` or `'gpu'`. An explicit `'gpu'` resolves to CUDA or MPS or raises; it never falls back to a CPU backend. Default: `'cpu'`. | <code>'cpu'</code>
`memory_budget_gb` | <code>float \| None</code> | Working-memory budget in GB used to size Himalaya's internal batches. None measures the device with conservative headroom. It is a budget, not a hard process limit. Default: `None`. | <code>None</code>
`per_target_alpha` | <code>bool</code> | True selects the best alpha separately per target; False averages each candidate's fold scores across targets and selects one shared alpha. Default: `True`. | <code>True</code>
`prefer_conservative_alpha` | <code>bool</code> | True selects the largest alpha whose mean score beats the best alpha's mean score minus that alpha's standard deviation across folds. Invalid with `per_target_alpha=False`. Default: `False`. | <code>False</code>
`random_state` | <code>int \| None</code> | Seed for the banded random search only; the cross-validator controls split randomness. Ordinary Ridge accepts it and ignores it — it has no randomness of its own — so that `BrainData.fit` can keep forwarding one shared `random_state` to whichever estimator it builds. Default: `None`. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar over the banded search. Default: `False`. | <code>False</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`coef_` | <code>ndarray</code> | `(n_features,)` for one-dimensional `y`, otherwise `(n_features, n_targets)`, in concatenated feature-space order.
`alpha_` | <code>float \| ndarray</code> | Scalar for a fixed or shared alpha, otherwise `(n_targets,)`.
`cv_scores_` | <code>float \| ndarray \| None</code> | None for a fixed-alpha fit. For ordinary Ridge, the fold-averaged negative-MSE score at the selected alpha. For banded Ridge, `(search_iterations,)` or `(search_iterations, n_targets)` fold-averaged scores.
`feature_space_weights_` | <code>ndarray \| None</code> | None for ordinary Ridge. Strictly positive weights whose columns sum to one, shaped `(n_spaces,)` or `(n_spaces, n_targets)`.
`feature_space_names_` | <code>tuple[str, ...] \| None</code> | Fitted mapping keys in coefficient order; None for ordinary Ridge.
`feature_space_sizes_` | <code>tuple[int, ...] \| None</code> | Feature counts aligned with `feature_space_names_`; None for ordinary Ridge.
`backend_` | <code>[Backend](#backends-backend)</code> | The resolved execution backend.
`n_samples_` | <code>int</code> | Fitted sample count.
`n_features_in_` | <code>int</code> | Total fitted feature count across spaces.
`is_fitted_` | <code>bool</code> | True after a successful fit.

**Methods:**

Name | Description
---- | -----------
[`fit`](#tasks-prediction-fit) | Fit the model.
[`predict`](#tasks-prediction-predict) | Predict targets for `X`.
[`score`](#tasks-prediction-score) | Return the coefficient of determination for each target.



**Examples:**

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

#### Methods

(tasks-prediction-fit)=
##### `fit`

```python
fit(X, y) -> Ridge
```

Fit the model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray \| Mapping[str, ndarray]</code> | A `(n_samples, n_features)` matrix for ordinary Ridge, or a non-empty mapping of unique names to equally sampled 2-D matrices for banded Ridge. | *required*
`y` | <code>ndarray</code> | Targets of shape `(n_samples,)` or `(n_samples, n_targets)`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Ridge](#tasks-prediction-ridge)</code> | `self`.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If any input or argument combination is invalid.
<code>RuntimeError</code> | If `device='gpu'` and no accelerator is available.

(tasks-prediction-predict)=
##### `predict`

```python
predict(X) -> np.ndarray
```

Predict targets for `X`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray \| Mapping[str, ndarray]</code> | Features in the structure used for fitting. Banded mappings may be in any order; they are aligned to `feature_space_names_`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | `(n_samples,)` when fitted on one-dimensional `y`,     otherwise `(n_samples, n_targets)`.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the model is not fitted, or `X` does not match the fitted feature structure.

(tasks-prediction-score)=
##### `score`

```python
score(X, y) -> float | np.ndarray
```

Return the coefficient of determination for each target.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray \| Mapping[str, ndarray]</code> | Features in the fitted structure. | *required*
`y` | <code>ndarray</code> | True targets, `(n_samples,)` or `(n_samples, n_targets)`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray</code> | A `float` for one-dimensional `y`, otherwise an     array of shape `(n_targets,)`. A constant target scores zero.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the model is not fitted, or the shapes disagree.

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

The cv-resolution rule used by `BrainData.predict`. String
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
