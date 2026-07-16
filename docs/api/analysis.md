(analysis-roc)=
## `roc`

NeuroLearn Analysis Tools
=========================
These tools provide the ability to quickly run
machine-learning analyses on imaging data

**Classes:**

Name | Description
---- | -----------
[`Roc`](#analysis-roc) | Roc Class



### Classes

#### `Roc`

```python
Roc(input_values = None, binary_outcome = None, threshold_type = 'optimal_overall', forced_choice = None, **kwargs)
```

Roc Class

The Roc class is based on Tor Wager's Matlab roc_plot.m function and
allows a user to easily run different types of receiver operator
characteristic curves.  For example, one might be interested in single
interval or forced choice.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | nibabel data instance | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`threshold_type` |  | ['optimal_overall', 'optimal_balanced',             'minimum_sdt_bias'] | <code>'optimal_overall'</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction         algorithm | <code>{}</code>

**Methods:**

Name | Description
---- | -----------
[`calculate`](#analysis-calculate) | Calculate Receiver Operating Characteristic plot (ROC) for
[`plot`](#analysis-plot) | Create ROC Plot
[`summary`](#analysis-summary) | Display a formatted summary of ROC analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`binary_outcome` |  | 
`forced_choice` |  | 
`input_values` |  | 
`threshold_type` |  | 

##### Methods

(analysis-calculate)=
###### `calculate`

```python
calculate(input_values = None, binary_outcome = None, criterion_values = None, threshold_type = 'optimal_overall', forced_choice = None, balanced_acc = False)
```

Calculate Receiver Operating Characteristic plot (ROC) for
single-interval classification.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | nibabel data instance | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`criterion_values` |  | (optional) criterion values for calculating fpr             & tpr | <code>None</code>
`threshold_type` |  | ['optimal_overall', 'optimal_balanced',             'minimum_sdt_bias'] | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject             (default=None) | <code>None</code>
`balanced_acc` |  | balanced accuracy for single-interval classification             (bool). THIS IS NOT COMPLETELY IMPLEMENTED BECAUSE             IT AFFECTS ACCURACY ESTIMATES, BUT NOT P-VALUES OR             THRESHOLD AT WHICH TO EVALUATE SENS/SPEC | <code>False</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction             algorithm | *required*

(analysis-plot)=
###### `plot`

```python
plot(plot_method = 'gaussian', balanced_acc = False, **kwargs)
```

Create ROC Plot

Create a specific kind of ROC curve plot, based on input values
along a continuous distribution and a binary outcome variable (logical)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`plot_method` |  | type of plot ['gaussian','observed'] | <code>'gaussian'</code>
`binary_outcome` |  | vector of training labels | *required*
`**kwargs` |  | Additional keyword arguments to pass to the prediction         algorithm | <code>{}</code>

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