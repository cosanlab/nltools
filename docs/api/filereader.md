## `nltools.io.file_reader`

NeuroLearn File Reading Tools
=============================

**Functions:**

Name | Description
---- | -----------
[`onsets_to_dm`](#nltools.io.file_reader.onsets_to_dm) | Read 1 or more file paths and return 1 or more design matrices.



### Classes

### Functions#### `nltools.io.file_reader.onsets_to_dm`

```python
onsets_to_dm(timings: str | Path | pd.DataFrame | list[str | Path | pd.DataFrame], run_length: int | list[int], TR: float, hrf_model: str | Callable | None = 'glover', drift_model: str | None = None, high_pass: float = 0.01, drift_order: int = 0, fill_na: Any = None, **kwargs: Any) -> DesignMatrix | list[DesignMatrix]
```

Read 1 or more file paths and return 1 or more design matrices.

Your timing file needs have the following column names:

- 'onset': required
- 'duration': required
- 'trial_type': optional
- 'modulation': optional

This function is a wrapper around [`nilearn.glm.first_level.make_first_level_design_matrix`](https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.make_first_level_design_matrix.html#nilearn.glm.first_level.make_first_level_design_matrix) which is more robust that older implementations.

However, the default options are **different** and create a design matrix with minimal additional modifications. You can use kwargs to control settings to also convolve predictors with a variety of HRF functions, add nuisance parameters, drift and cosine functions, etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`timings` | <code>([str](#str), [Path](#pathlib.Path), [DataFrame](#pandas.DataFrame), [list](#list))</code> | file(s) or dataframe(s) containing stimulus timing | *required*
`run_length` | <code>([int](#int), [list](#list))</code> | number or list of numbers for run lengths in TRs | *required*
`TR` | <code>[float](#float)</code> | repetition time in seconds. Defaults to None. | *required*
`hrf_model` | <code>[str](#str)</code> | convolve each column of the design matrix (e.g. 'glover'). Defaults to None. | <code>'glover'</code>
`drift_model` | <code>[str](#str)</code> | how to add drift ('cosine' or 'polynomial'). Defaults to None. | <code>None</code>
`high_pass` | <code>[float](#float)</code> | high-pass frequency if drift_model='cosine'. Defaults to 0.01 | <code>0.01</code>
`drift_order` | <code>[int](#int)</code> | what order if drift_model='polynomial'. Defaults to 0. | <code>0</code>
`fill_na` | <code>[Any](#typing.Any)</code> | value to fill NaN entries with. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#nltools.data.DesignMatrix) \| [list](#list)[[DesignMatrix](#nltools.data.DesignMatrix)]</code> | DesignMatrix | list[DesignMatrix]: Single DesignMatrix if one timing file provided, or list of DesignMatrices if multiple timing files provided.

