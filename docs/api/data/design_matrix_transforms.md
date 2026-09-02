---
title: data.designmatrix.transforms
label: data-design-matrix-transforms
---

Standalone transform functions for DesignMatrix.

Each function takes a DesignMatrix instance as the first argument (`dm`)
and returns a new DesignMatrix via `copy_with(dm,...)`.

**Functions:**

Name | Description
---- | -----------
[`downsample`](#data-design-matrix-transforms-downsample) | Reduce temporal resolution using Polars-native operations.
[`standardize`](#data-design-matrix-transforms-standardize) | Standardize columns using the specified method.
[`upsample`](#data-design-matrix-transforms-upsample) | Increase temporal resolution using Polars-native interpolation.
[`zscore`](#data-design-matrix-transforms-zscore) | Z-score standardize columns to mean zero and unit variance.

## Functions

(data-design-matrix-transforms-downsample)=
### `downsample`

```python
downsample(dm: DesignMatrix, target: float, method: str = 'mean') -> DesignMatrix
```

Reduce temporal resolution using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#data-design-matrix)</code> | DesignMatrix instance to transform. | *required*
`target` | <code>float</code> | Target sampling frequency in Hz (must be < current sampling_freq). | *required*
`method` | <code>str</code> | Aggregation method - 'mean' or 'median'. Default: 'mean'. | <code>'mean'</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#data-design-matrix)</code> | Downsampled DesignMatrix with updated sampling_freq.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If sampling_freq is not set, target >= current sampling_freq, or method is invalid.

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": list(range(100))}, sampling_freq=1.0)
>>> dm_down = downsample(dm, target=0.5)  # 1 Hz -> 0.5 Hz (100 -> 50 samples)
```

(data-design-matrix-transforms-standardize)=
### `standardize`

```python
standardize(dm: DesignMatrix, columns: list[str] | None = None, method: str = 'zscore') -> DesignMatrix
```

Standardize columns using the specified method.

This method provides a consistent API with BrainData and Collection
for data normalization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#data-design-matrix)</code> | DesignMatrix instance to transform. | *required*
`columns` | <code>list[str] \| None</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>
`method` | <code>str</code> | Standardization method. Options are: - 'zscore': Z-score standardization (mean=0, std=1) [default] - 'center': Mean centering only (mean=0) | <code>'zscore'</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#data-design-matrix)</code> | New DesignMatrix with standardized columns.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If an invalid method is specified.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3))
>>> dm_z = standardize(dm, method='zscore')  # z-score all columns
>>> dm_c = standardize(dm, method='center')  # center only
```

(data-design-matrix-transforms-upsample)=
### `upsample`

```python
upsample(dm: DesignMatrix, target: float, method: str = 'linear') -> DesignMatrix
```

Increase temporal resolution using Polars-native interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#data-design-matrix)</code> | DesignMatrix instance to transform. | *required*
`target` | <code>float</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>str</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#data-design-matrix)</code> | Upsampled DesignMatrix with updated sampling_freq.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If sampling_freq is not set, target <= current sampling_freq, or method is invalid.

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)
>>> dm_up = upsample(dm, target=2.0)  # 1 Hz -> 2 Hz (10 -> 18 samples)
```

(data-design-matrix-transforms-zscore)=
### `zscore`

```python
zscore(dm: DesignMatrix, columns: list[str] | None = None) -> DesignMatrix
```

Z-score standardize columns to mean zero and unit variance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#data-design-matrix)</code> | DesignMatrix instance to transform. | *required*
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#data-design-matrix)</code> | New DesignMatrix with standardized columns
