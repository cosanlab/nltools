## `nltools.data.collection.transforms`

BrainCollection transform functions.

Extracted from BrainCollection methods — each function takes a BrainCollection
as its first argument (``bc``) instead of ``self``.

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.collection.transforms.align) | Align subjects using local functional alignment.
[`detrend`](#nltools.data.collection.transforms.detrend) | Remove trend from each image.
[`filter_collection`](#nltools.data.collection.transforms.filter_collection) | Filter collection by predicate.
[`map_axis0`](#nltools.data.collection.transforms.map_axis0) | Map function over images (axis=0).
[`map_axis1`](#nltools.data.collection.transforms.map_axis1) | Map function over timepoints (axis=1).
[`map_axis2`](#nltools.data.collection.transforms.map_axis2) | Map function over voxels (axis=2) per image.
[`map_collection`](#nltools.data.collection.transforms.map_collection) | Apply function across specified axis.
[`smooth`](#nltools.data.collection.transforms.smooth) | Spatially smooth each image.
[`standardize`](#nltools.data.collection.transforms.standardize) | Standardize each image.
[`threshold`](#nltools.data.collection.transforms.threshold) | Threshold each image.



### Classes

### Functions#### `nltools.data.collection.transforms.align`

```python
align(bc: 'BrainCollection', method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: 'nib.Nifti1Image | None' = None, n_features: int | None = None, n_iter: int = 3, parallel: str | None = 'cpu', n_jobs: int = -1, return_model: bool = False, show_progress: bool = True) -> 'BrainCollection | tuple[BrainCollection, object]'
```

Align subjects using local functional alignment.

Performs neighborhood-based functional alignment across subjects using
LocalAlignment. Each subject's data is aligned to a common template space
using local transforms learned within searchlight spheres or parcels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to align. | *required*
`method` | <code>[str](#str)</code> | Alignment method. Options: - 'procrustes': Orthogonal Procrustes (default, preserves dimensions) - 'srm': Shared Response Model (dimensionality reduction) - 'hyperalignment': Hyperalignment (iterative Procrustes) | <code>'procrustes'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Options: - 'searchlight': Overlapping spheres with center-only aggregation - 'piecewise': Non-overlapping parcels (requires parcellation) | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for searchlight scheme. | <code>10.0</code>
`parcellation` | <code>'nib.Nifti1Image \| None'</code> | Parcellation image for piecewise scheme (required if scheme='piecewise'). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full dimensions. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. | <code>3</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. Options: - None: Single-threaded - 'cpu': CPU parallelization with joblib - 'gpu': GPU acceleration via PyTorch | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs for CPU mode (-1 = auto). | <code>-1</code>
`return_model` | <code>[bool](#bool)</code> | If True, return (aligned_collection, model) tuple for fit/transform workflow with new data. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | BrainCollection with aligned data. If return_model=True, returns
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | tuple of (aligned_collection, LocalAlignment_model).

**Examples:**

```pycon
>>> # Basic searchlight alignment
>>> aligned_bc = bc.align(method='procrustes', radius_mm=10.0)
```

```pycon
>>> # Piecewise alignment with parcellation
>>> aligned_bc = bc.align(
...     scheme='piecewise',
...     parcellation=parcellation_img,
...     method='srm',
...     n_features=50
... )
```

```pycon
>>> # Fit/transform workflow for train/test split
>>> aligned_train, model = train_bc.align(return_model=True)
>>> aligned_test = model.transform(test_data_list)
```

```pycon
>>> # GPU-accelerated alignment
>>> aligned_bc = bc.align(parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used for searchlight to preserve local orthogonality of transforms.

</details>

<details class="see-also" open markdown="1">
<summary>See Also</summary>

nltools.algorithms.alignment.LocalAlignment: Underlying alignment class.

</details>

#### `nltools.data.collection.transforms.detrend`

```python
detrend(bc: 'BrainCollection', method: str = 'linear', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Remove trend from each image.

Delegates to BrainData.detrend() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to detrend. | *required*
`method` | <code>[str](#str)</code> | 'linear' or 'constant'. | <code>'linear'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with detrended images.

**Examples:**

```pycon
>>> bc.detrend()  # Remove linear trend
>>> bc.detrend(method='constant')  # Remove mean only
```

#### `nltools.data.collection.transforms.filter_collection`

```python
filter_collection(bc: 'BrainCollection', predicate: 'Callable | list | np.ndarray') -> 'BrainCollection'
```

Filter collection by predicate.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to filter. | *required*
`predicate` | <code>'Callable \| list \| np.ndarray'</code> | Filter condition. Can be: - callable: fn(BrainData) -> bool - list/ndarray: Boolean mask of length n_images - pd.Series: Boolean series (index ignored) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with subset of images matching predicate.

**Examples:**

```pycon
>>> # Filter by callable
>>> bc.filter(lambda bd: bd.data.mean() > 0)
```

```pycon
>>> # Filter by boolean mask
>>> mask = [True, False, True]
>>> bc.filter(mask)
```

```pycon
>>> # Filter by metadata condition
>>> bc.filter(bc.metadata['group'] == 'control')
```

#### `nltools.data.collection.transforms.map_axis0`

```python
map_axis0(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over images (axis=0).

#### `nltools.data.collection.transforms.map_axis1`

```python
map_axis1(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over timepoints (axis=1).

#### `nltools.data.collection.transforms.map_axis2`

```python
map_axis2(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over voxels (axis=2) per image.

#### `nltools.data.collection.transforms.map_collection`

```python
map_collection(bc: 'BrainCollection', fn: Callable, axis: int | str = 0, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Apply function across specified axis.

This is the general-purpose transformation method. For common operations,
use convenience methods like standardize(), smooth(), etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to transform. | *required*
`fn` | <code>[Callable](#collections.abc.Callable)</code> | Function to apply. Signature depends on axis: - axis=0: fn(BrainData) -> BrainData (per image) - axis=1: fn(BrainData) -> BrainData (per timepoint slice) - axis=2: fn(ndarray[n_obs]) -> ndarray (per voxel timeseries) | *required*
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to iterate over: - 0 or 'images': Apply fn to each image independently - 1 or 'time': Apply fn to each timepoint across images - 2 or 'voxels': Apply fn to each voxel timeseries per image | <code>0</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. -1 for all cores. Default 1. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. Default True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with transformed data.

**Examples:**

```pycon
>>> # Per-image operation
>>> bc.map(lambda bd: bd.standardize())
```

```pycon
>>> # Per-voxel timeseries (e.g., detrend each voxel)
>>> from scipy.signal import detrend
>>> bc.map(detrend, axis=2)
```

```pycon
>>> # Parallel processing
>>> bc.map(expensive_fn, n_jobs=-1)
```

#### `nltools.data.collection.transforms.smooth`

```python
smooth(bc: 'BrainCollection', fwhm: float, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Spatially smooth each image.

Delegates to BrainData.smooth() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to smooth. | *required*
`fwhm` | <code>[float](#float)</code> | Full width at half maximum of Gaussian kernel in mm. | *required*
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with smoothed images.

**Examples:**

```pycon
>>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
```

#### `nltools.data.collection.transforms.standardize`

```python
standardize(bc: 'BrainCollection', axis: int = 0, method: str = 'center', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Standardize each image.

Delegates to BrainData.standardize() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to standardize. | *required*
`axis` | <code>[int](#int)</code> | Axis for standardization within each image: - 0: Standardize across observations (time) per voxel - 1: Standardize across voxels per observation | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' (subtract mean) or 'zscore' (subtract mean, divide std) | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with standardized images.

**Examples:**

```pycon
>>> bc.standardize()  # Center each image across time
>>> bc.standardize(method='zscore')  # Z-score each image
>>> bc.standardize(axis=1)  # Standardize across voxels
```

#### `nltools.data.collection.transforms.threshold`

```python
threshold(bc: 'BrainCollection', upper: float | str | None = None, lower: float | str | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Threshold each image.

Delegates to BrainData.threshold() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to threshold. | *required*
`upper` | <code>[float](#float) \| [str](#str) \| None</code> | Upper cutoff. String interpreted as percentile. | <code>None</code>
`lower` | <code>[float](#float) \| [str](#str) \| None</code> | Lower cutoff. String interpreted as percentile. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | Return binary mask. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | Replace NaN with 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with thresholded images.

**Examples:**

```pycon
>>> bc.threshold(lower=0)  # Zero out negative values
>>> bc.threshold(upper='95%')  # Keep top 5%
>>> bc.threshold(lower=2, binarize=True)  # Binary mask
```

