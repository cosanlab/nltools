(dataset-datasets)=
## `datasets`

Dataset download and example-data utilities.

# NeuroLearn datasets

Functions to help download example datasets. The curated example datasets
(`fetch_pain`, `fetch_emotion_ratings`) are hosted on the ``nltools/niftis``
Hugging Face dataset and resolve through the same `fetch_resource` /
`seed_resources` machinery as the MNI templates and atlases, so they work both
on a normal Python kernel and in Pyodide / JupyterLite (pre-seed with
`seed_resources` there). Arbitrary Neurovault collections are still available
via `fetch_neurovault_collection`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`EMOTION_METADATA` |  | Relpath of the emotion dataset's metadata table (its filename manifest).
`PAIN_RESOURCES` | <code>[list](#list)[[str](#str)]</code> | Every `fetch_resource` relpath the pain dataset needs (metadata + 84 images).



**Methods:**

Name | Description
---- | -----------
[`download_nifti`](#dataset-download-nifti) | Download an image from a URL to a nifti file.
[`emotion_resources`](#dataset-emotion-resources) | List every `fetch_resource` relpath the emotion dataset needs.
[`fetch_emotion_ratings`](#dataset-fetch-emotion-ratings) | Download and load the emotion-rating dataset from the nltools HF dataset.
[`fetch_neurovault_collection`](#dataset-fetch-neurovault-collection) | Download images and metadata from a Neurovault collection.
[`fetch_pain`](#dataset-fetch-pain) | Download and load the pain dataset from the nltools HF dataset.
[`load_haxby_example`](#dataset-load-haxby-example) | Load a small synthetic Haxby-like dataset, entirely in-memory.

### Classes

### Methods

(dataset-download-nifti)=
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

(dataset-emotion-resources)=
#### `emotion_resources`

```python
emotion_resources() -> list[str]
```

List every `fetch_resource` relpath the emotion dataset needs.

The emotion image filenames are keyed by Neurovault id (not a generable
grid like `PAIN_RESOURCES`), so this reads `EMOTION_METADATA` to enumerate
them. To pre-seed the Pyodide / JupyterLite cache, seed the metadata file
first (it is read here), then seed the images:

```python
await seed_resources([EMOTION_METADATA])
await seed_resources(emotion_resources())
```

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | list[str]: `[EMOTION_METADATA, ...679 image relpaths]`.

(dataset-fetch-emotion-ratings)=
#### `fetch_emotion_ratings`

```python
fetch_emotion_ratings(verbose = 0)
```

Download and load the emotion-rating dataset from the nltools HF dataset.

Loads the Chang et al. (2015) IAPS emotion-rating study: 679 whole-brain
contrast images across 150 subjects, each rating images 1-5, with a
built-in train/test holdout split. `X` carries the full portable Neurovault
metadata (key columns: `SubjectID`, `Rating`, `Holdout`, `AGE`, `SEX`).

Data is hosted on the ``nltools/niftis`` Hugging Face dataset and cached
locally on first use, so this works on a normal Python kernel with no extra
setup. In Pyodide / JupyterLite, pre-seed the cache first (see
`emotion_resources`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`verbose` | <code>[int](#int)</code> | Verbosity passed to `BrainData` while loading. Default: 0 | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | `BrainData` with the 679 images; `X` holds the metadata table.

<details class="references" open markdown="1">
<summary>References</summary>

Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
A sensitive and specific neural signature for picture-induced negative affect.
PLoS biology, 13(6), e1002180.

</details>

(dataset-fetch-neurovault-collection)=
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

(dataset-fetch-pain)=
#### `fetch_pain`

```python
fetch_pain(verbose = 0)
```

Download and load the pain dataset from the nltools HF dataset.

Loads the Chang et al. (2015) pain-perception study: 28 subjects x 3
stimulus-intensity conditions = 84 whole-brain contrast images, with a
curated metadata table (`SubjectID`, `PainLevel`, `PainIntensity`, `Age`,
`Sex`, provenance `neurovault_id` / `name`).

Data is hosted on the ``nltools/niftis`` Hugging Face dataset and cached
locally on first use, so this works on a normal Python kernel with no extra
setup. In Pyodide / JupyterLite, pre-seed the cache first:
``await seed_resources(PAIN_RESOURCES)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`verbose` | <code>[int](#int)</code> | Verbosity passed to `BrainData` while loading. Default: 0 | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | `BrainData` with the 84 images; `X` holds the metadata table.

<details class="references" open markdown="1">
<summary>References</summary>

Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
A sensitive and specific neural signature for picture-induced negative affect.
PLoS biology, 13(6), e1002180.

</details>

(dataset-load-haxby-example)=
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

