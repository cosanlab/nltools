## `constructors`

Constructor functions for BrainCollection.

Standalone functions that create BrainCollection instances from various sources
(BIDS datasets, glob patterns, stacked BrainData).

**Methods:**

Name | Description
---- | -----------
[`from_bids`](#from_bids) | Create BrainCollection from a BIDS dataset.
[`from_glob`](#from_glob) | Create BrainCollection from glob pattern.
[`from_stacked`](#from_stacked) | Create BrainCollection by splitting a stacked BrainData.



### Classes

### Methods

#### `from_bids`

```python
from_bids(layout: Any, mask: 'nib.Nifti1Image | Path | str', *, task: str | None = None, subject: str | list[str] | None = None, session: str | list[str] | None = None, run: int | list[int] | None = None, space: str | None = None, suffix: str = 'bold', extension: str = 'nii.gz', **bids_filters: str) -> 'BrainCollection'
```

Create BrainCollection from a BIDS dataset.

Requires pybids to be installed: `pip install pybids`

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`layout` | <code>[Any](#typing.Any)</code> | pybids BIDSLayout object or path to BIDS dataset. | *required*
`mask` | <code>'nib.Nifti1Image \| Path \| str'</code> | Shared mask (required). | *required*
`task` | <code>[str](#str) \| None</code> | BIDS task filter. | <code>None</code>
`subject` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Subject ID(s) to include. | <code>None</code>
`session` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Session ID(s) to include. | <code>None</code>
`run` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Run number(s) to include. | <code>None</code>
`space` | <code>[str](#str) \| None</code> | BIDS space filter (e.g., 'MNI152NLin2009cAsym'). | <code>None</code>
`suffix` | <code>[str](#str)</code> | BIDS suffix (default 'bold'). | <code>'bold'</code>
`extension` | <code>[str](#str)</code> | File extension (default 'nii.gz'). | <code>'nii.gz'</code>
`**bids_filters` |  | Additional BIDS entity filters. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with metadata extracted from BIDS entities.

**Examples:**

```pycon
>>> bc = from_bids(
...     '/data/bids_dataset',
...     mask='2mm-MNI152-2009c',
...     task='rest',
...     space='MNI152NLin2009cAsym'
... )
```

#### `from_glob`

```python
from_glob(pattern: str, mask: 'nib.Nifti1Image | Path | str', *, pattern_groups: 'dict[str, int] | str | None' = None, sort: bool = True) -> 'BrainCollection'
```

Create BrainCollection from glob pattern.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern (e.g., ``'/data/*/func/*_bold.nii.gz'``). | *required*
`mask` | <code>'nib.Nifti1Image \| Path \| str'</code> | Shared mask (required). | *required*
`pattern_groups` | <code>'dict[str, int] \| str \| None'</code> | Regex pattern with named groups for metadata extraction. Example: ``r'sub-(?P<subject>\w+)/.*run-(?P<run>\d+)'`` | <code>None</code>
`sort` | <code>[bool](#bool)</code> | Sort files alphabetically (default True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with optional metadata from pattern groups.

**Examples:**

```pycon
>>> bc = from_glob(
...     '/data/sub-*/func/*_bold.nii.gz',
...     mask=mask,
...     pattern_groups=r'sub-(?P<subject>\w+)'
... )
```

#### `from_stacked`

```python
from_stacked(brain_data: 'BrainData', splits: list[int] | None = None, n_images: int | None = None) -> 'BrainCollection'
```

Create BrainCollection by splitting a stacked BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` | <code>'BrainData'</code> | BrainData with shape (n_total_obs, n_voxels). | *required*
`splits` | <code>[list](#list)[[int](#int)] \| None</code> | List of observation counts per image. Must sum to n_total_obs. | <code>None</code>
`n_images` | <code>[int](#int) \| None</code> | Number of images (splits evenly). Mutually exclusive with splits. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with data split according to specification.

**Examples:**

```pycon
>>> # Split evenly into 3 images
>>> bc = from_stacked(bd, n_images=3)
```

```pycon
>>> # Split with explicit counts
>>> bc = from_stacked(bd, splits=[100, 100, 150])
```

