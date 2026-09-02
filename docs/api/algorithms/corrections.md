---
title: algorithms.corrections
label: algorithms-corrections
---

Multiple comparison corrections and thresholding.

**Functions:**

Name | Description
---- | -----------
[`fdr`](#algorithms-corrections-fdr) | Determine an FDR threshold for an array of p-values.
[`holm_bonf`](#algorithms-corrections-holm-bonf) | Compute Holm-Bonferroni-corrected p-values.
[`multi_threshold`](#algorithms-corrections-multi-threshold) | Threshold test image by multiple p-values from p image.
[`threshold`](#algorithms-corrections-threshold) | Threshold test image by p-value from p image.



## Functions

(algorithms-corrections-fdr)=
### `fdr`

```python
fdr(p, q = 0.05)
```

Determine an FDR threshold for an array of p-values.

Uses the desired false discovery rate ``q``. Written by Tal Yarkoni.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` |  | (np.array) vector of p-values | *required*
`q` |  | (float) false discovery rate level | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | p-value threshold based on independence or positive dependence.

(algorithms-corrections-holm-bonf)=
### `holm_bonf`

```python
holm_bonf(p, alpha = 0.05)
```

Compute Holm-Bonferroni-corrected p-values.

This step-down procedure applies iteratively less correction to the highest
p-values. It is a bit more conservative than FDR, but much more powerful than
vanilla Bonferroni correction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` |  | (np.array) vector of p-values | *required*
`alpha` |  | (float) alpha level | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | p-value threshold based on the Bonferroni step-down procedure.

(algorithms-corrections-multi-threshold)=
### `multi_threshold`

```python
multi_threshold(t_map, p_map, thresh)
```

Threshold test image by multiple p-values from p image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_map` |  | (BrainData) BrainData instance of statistic metric (e.g., t-statistic, beta, etc) | *required*
`p_map` |  | (BrainData) BrainData instance of p-values | *required*
`thresh` |  | (list) list of p-values to threshold stat image | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#data-brain-data)</code> | Thresholded BrainData instance with cumulative map. Positive     values indicate how many thresholds were passed for positive stats;     negative values indicate how many thresholds were passed for negative     stats.

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique cumulative threshold map functionality:
- Creates a single map showing which thresholds were passed
- Different from calling threshold() multiple times (which would give separate images)
- Useful for visualizing threshold hierarchies
- nilearn.threshold_img() does not support cumulative multi-threshold maps

</details>

(algorithms-corrections-threshold)=
### `threshold`

```python
threshold(stat, p, thr = 0.05, return_mask = False)
```

Threshold test image by p-value from p image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (BrainData) BrainData instance of arbitrary statistic metric   (e.g., beta, t, etc) | *required*
`p` |  | (BrainData) BrainData instance of p-values | *required*
`thr` |  | (float) p-value threshold to apply | <code>0.05</code>
`return_mask` |  | (bool) optionally return the thresholding mask; default False | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#data-brain-data) \| tuple[[BrainData](#data-brain-data), [BrainData](#data-brain-data)]</code> | The thresholded BrainData instance;     if `return_mask=True`, a tuple `(out, mask)` where `mask` is the     BrainData instance of the thresholding mask.

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique functionality not available in nilearn:
- Thresholds stat image based on p-values from separate p-value image
- Neither nilearn.threshold_img nor BrainData.threshold() support this
- BrainData.threshold() thresholds based on stat values themselves
- nilearn.threshold_img() thresholds based on image intensity values

</details>
