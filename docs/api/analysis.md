(analysis-roc)=
## `roc`

ROC (Receiver Operating Characteristic) analysis for single-interval classification.

These tools provide the ability to quickly run receiver operating characteristic
analyses on the output of machine-learning models applied to imaging data.

**Classes:**

Name | Description
---- | -----------
[`Roc`](#analysis-roc) | Compute receiver operating characteristic curves for single-interval or forced-choice classification.



### Classes

#### `Roc`

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
`input_values` |  | 1-D array/vector of continuous decision values (one per observation) | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`method` |  | threshold-selection variant, one of `'optimal_overall'`, `'optimal_balanced'`, `'minimum_sdt_bias'` | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject (default=None) | <code>None</code>

**Methods:**

Name | Description
---- | -----------
[`calculate`](#analysis-calculate) | Calculate ROC metrics for single-interval classification.
[`plot`](#analysis-plot) | Create a ROC plot.
[`summary`](#analysis-summary) | Display a formatted summary of ROC analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`binary_outcome` |  | 
`forced_choice` |  | 
`input_values` |  | 
`method` |  | 

##### Methods

(analysis-calculate)=
###### `calculate`

```python
calculate(*, input_values = None, binary_outcome = None, criterion_values = None, method = 'optimal_overall', forced_choice = None, balanced_acc = False)
```

Calculate ROC metrics for single-interval classification.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | 1-D array/vector of continuous decision values (one per observation) | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`criterion_values` |  | (optional) criterion values for calculating fpr             & tpr | <code>None</code>
`method` |  | threshold-selection variant, one of `'optimal_overall'`,             `'optimal_balanced'`, `'minimum_sdt_bias'` | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject             (default=None) | <code>None</code>
`balanced_acc` |  | balanced accuracy for single-interval classification             (bool). THIS IS NOT COMPLETELY IMPLEMENTED BECAUSE             IT AFFECTS ACCURACY ESTIMATES, BUT NOT P-VALUES OR             THRESHOLD AT WHICH TO EVALUATE SENS/SPEC | <code>False</code>

(analysis-plot)=
###### `plot`

```python
plot(*, method = 'gaussian', balanced_acc = False)
```

Create a ROC plot.

Create a specific kind of ROC curve plot, based on input values
along a continuous distribution and a binary outcome variable (logical)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | type of plot, one of `'gaussian'`, `'observed'` | <code>'gaussian'</code>
`balanced_acc` |  | balanced accuracy for single-interval classification | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | fig

(analysis-summary)=
###### `summary`

```python
summary()
```

Display a formatted summary of ROC analysis.



### Methods