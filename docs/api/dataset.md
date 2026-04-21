## `datasets`

NeuroLearn datasets
===================

Functions to help download datasets from Neurovault and other sources.

**Methods:**

Name | Description
---- | -----------
[`download_nifti`](#download_nifti) | Download an image from a URL to a nifti file.
[`fetch_emotion_ratings`](#fetch_emotion_ratings) | Download and load emotion rating dataset from Neurovault.
[`fetch_haxby`](#fetch_haxby) | Download and load Haxby2001 dataset from nilearn.
[`fetch_neurovault_collection`](#fetch_neurovault_collection) | Download images and metadata from a Neurovault collection.
[`fetch_pain`](#fetch_pain) | Download and load pain dataset from Neurovault.
[`load_haxby_example`](#load_haxby_example) | Load a small synthetic Haxby-like dataset, entirely in-memory.



### Classes

### Methods

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
`tuple` |  | (metadata polars.DataFrame, list of image file paths)

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

#### `load_haxby_example`

```python
load_haxby_example(n_runs = 1, random_state = 42)
```

Load a small synthetic Haxby-like dataset, entirely in-memory.

Matches the return shape of `fetch_haxby(n_subjects=1)` — paired lists of
`BrainData` and `DesignMatrix`, one entry per run — but generates a tiny
synthetic volume (10 x 10 x 5 = 500 voxels) with condition-specific signal
injected into disjoint voxel clusters. No network I/O, no disk I/O, no
nilearn fetcher dependency. Runs in well under a second.

Intended for tutorials, documentation examples, and Pyodide / in-browser
environments where `fetch_haxby` cannot download the real dataset. The
eight conditions match the real Haxby object-recognition experiment
(face, house, cat, bottle, scissors, shoe, chair, scrambledpix), arranged
in a randomized 9-TR block design with TR=2.5s.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_runs` | <code>[int](#int)</code> | Number of runs to generate. Default 1. | <code>1</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for reproducible output. Default 42. | <code>42</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | `(list[BrainData], list[DesignMatrix])`, each of length n_runs. The DesignMatrix columns are the eight condition names plus a "constant" column, matching `fetch_haxby`'s output shape.

**Examples:**

```pycon
>>> from nltools.datasets import load_haxby_example
>>> brain_data, design_matrices = load_haxby_example()
>>> data, dm = brain_data[0], design_matrices[0]
>>> data.shape
(72, 500)
>>> "face" in dm.columns
True
```

