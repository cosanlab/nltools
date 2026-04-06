## `datasets`

NeuroLearn datasets
===================

Functions to help download datasets from Neurovault and other sources.

**Methods:**

Name | Description
---- | -----------
[`download_collection`](#download_collection) | Download images and metadata from Neurovault collection.
[`download_nifti`](#download_nifti) | Download an image from a URL to a nifti file.
[`fetch_emotion_ratings`](#fetch_emotion_ratings) | Download and load emotion rating dataset from Neurovault.
[`fetch_haxby`](#fetch_haxby) | Download and load Haxby2001 dataset from nilearn.
[`fetch_neurovault_collection`](#fetch_neurovault_collection) | Download images and metadata from a Neurovault collection.
[`fetch_pain`](#fetch_pain) | Download and load pain dataset from Neurovault.
[`get_collection_image_metadata`](#get_collection_image_metadata) | Get image metadata associated with collection.



### Classes

### Methods

#### `download_collection`

```python
download_collection(collection = None, data_dir = None, overwrite = False, resume = True, verbose = 1)
```

Download images and metadata from Neurovault collection.

.. deprecated::
    This function is deprecated and will be removed in a future version.
    Please use fetch_neurovault_collection instead.

#### `download_nifti`

```python
download_nifti(url, data_dir = None)
```

Download an image from a URL to a nifti file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`url` | <code>[str](#str)</code> | URL of the image to download | *required*
`data_dir` | <code>[str](#str)</code> | Directory to save the file. If None, uses current directory. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Path to the downloaded file

#### `fetch_emotion_ratings`

```python
fetch_emotion_ratings(data_dir = None, verbose = 1)
```

Download and load emotion rating dataset from Neurovault.

This downloads the Chang et al. (2015) emotion ratings dataset from
Neurovault collection 1964.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_dir` | <code>[str](#str)</code> | Path of the data directory. Used to force data storage in a specified location. Default: None | <code>None</code>
`verbose` | <code>[int](#int)</code> | Verbosity level. Default: 1 | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData object with downloaded data. X=metadata

<details class="references" open markdown="1">
<summary>References</summary>

Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
A sensitive and specific neural signature for picture-induced negative affect.
PLoS biology, 13(6), e1002180.

</details>

#### `fetch_haxby`

```python
fetch_haxby(n_subjects = 1, data_dir = None, verbose = 1, mask = 'haxby_mask', resample = False)
```

Download and load Haxby2001 dataset from nilearn.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_subjects` | <code>int, None, or 'all'</code> | Which subject to load (1-6), or None/'all' for all subjects. Default: 1. - `n_subjects=1`: Returns all runs for subject 1 - `n_subjects=2`: Returns all runs for subject 2 - `n_subjects=None` or `'all'`: Returns all runs for all subjects (nested lists) | <code>1</code>
`data_dir` | <code>[str](#str)</code> | Directory to store downloaded data. Default: None | <code>None</code>
`verbose` | <code>[int](#int)</code> | Verbosity level. Default: 1 | <code>1</code>
`mask` | <code>str, nibabel.Nifti1Image, or None, default="haxby_mask"</code> | Brain mask to use. - `"haxby_mask"`: Use the default mask provided with the Haxby dataset (default) - `None`: Use default MNI template mask - Other: Passed directly to BrainData (file path, nibabel object, etc.) | <code>'haxby_mask'</code>
`resample` | <code>bool, default=False</code> | Whether to automatically resample data to mask space. See BrainData.__init__() for details. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | - If n_subjects is int: (list of BrainData, list of DesignMatrix) - all runs for that subject - If n_subjects is None or 'all': (list of lists of BrainData, list of lists of DesignMatrix)   First level: subjects, second level: runs per subject

**Examples:**

```pycon
>>> # Load all runs for subject 1
>>> brain_data, design_matrix = fetch_haxby(n_subjects=1)
>>> len(brain_data)  # Number of runs
>>>
>>> # Load all runs for subject 2
>>> brain_data, design_matrix = fetch_haxby(n_subjects=2)
>>>
>>> # Load all runs for all subjects
>>> brain_data_nested, design_matrix_nested = fetch_haxby(n_subjects='all')
>>> len(brain_data_nested)  # Number of subjects
>>> len(brain_data_nested[0])  # Number of runs for first subject
```

#### `fetch_neurovault_collection`

```python
fetch_neurovault_collection(collection_id, data_dir = None, verbose = 1)
```

Download images and metadata from a Neurovault collection.

This function uses the modern nilearn API to download collections from Neurovault.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`collection_id` | <code>[int](#int)</code> | Neurovault collection ID | *required*
`data_dir` | <code>[str](#str)</code> | Directory to store downloaded data. If None, uses nilearn's default data directory. | <code>None</code>
`verbose` | <code>[int](#int)</code> | Verbosity level. Default: 1 | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (metadata DataFrame, list of image file paths)

#### `fetch_pain`

```python
fetch_pain(data_dir = None, verbose = 1)
```

Download and load pain dataset from Neurovault.

This downloads the Chang et al. (2015) pain dataset from Neurovault collection 504.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_dir` | <code>[str](#str)</code> | Path of the data directory. Used to force data storage in a specified location. Default: None | <code>None</code>
`verbose` | <code>[int](#int)</code> | Verbosity level. Default: 1 | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData object with downloaded data. X=metadata

<details class="references" open markdown="1">
<summary>References</summary>

Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
A sensitive and specific neural signature for picture-induced negative affect.
PLoS biology, 13(6), e1002180.

</details>

#### `get_collection_image_metadata`

```python
get_collection_image_metadata(collection = None, data_dir = None, limit = 10)
```

Get image metadata associated with collection.

.. deprecated::
    This function is deprecated and will be removed in a future version.
    Please use fetch_neurovault_collection instead.

