## `datasets`

Dataset download and example-data utilities.

# NeuroLearn datasets

Functions to help download datasets from Neurovault and other sources.

**Methods:**

Name | Description
---- | -----------
[`download_nifti`](#download_nifti) | Download an image from a URL to a nifti file.
[`fetch_emotion_ratings`](#fetch_emotion_ratings) | Download and load emotion rating dataset from Neurovault.
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

Returns paired lists of `BrainData` and `DesignMatrix`, one entry per
run, generated from a tiny synthetic volume (10 x 10 x 5 = 500 voxels)
with condition-specific signal injected into disjoint voxel clusters.
No network I/O, no disk I/O, no nilearn fetcher dependency. Runs in
well under a second.

Intended for tutorials, documentation examples, and Pyodide / in-browser
environments where downloading a real fMRI dataset is impractical. The
eight conditions match the real Haxby 2001 object-recognition experiment
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
`tuple` |  | `(list[BrainData], list[DesignMatrix])`, each of length n_runs. The DesignMatrix columns are the eight condition names suffixed with ``_c0`` (HRF-convolved boxcars).

**Examples:**

```pycon
>>> from nltools.datasets import load_haxby_example
>>> brain_data, design_matrices = load_haxby_example()
>>> data, dm = brain_data[0], design_matrices[0]
>>> data.shape
(72, 500)
>>> "face_c0" in dm.columns
True
```

