## `io`

I/O functions for BrainCollection.

Provides save path resolution and write functionality extracted from BrainCollection.

**Methods:**

Name | Description
---- | -----------
[`write`](#write) | Write all images in collection to files.



### Classes

### Methods

#### `write`

```python
write(bc: BrainCollection, directory: str | Path, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write all images in collection to files.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to write. | *required*
`directory` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Output directory path. Will be created if it doesn't exist. | *required*
`pattern` | <code>[str](#str)</code> | Filename pattern with {i} placeholder for image index. Default: "image_{i:04d}.nii.gz" produces image_0000.nii.gz, etc. | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | Optional filename for metadata CSV. Set to None to skip. Default: "metadata.csv" | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Path](#pathlib.Path)]</code> | List of paths to written files.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> paths = write(bc, "output/")
>>> # Creates: output/image_0000.nii.gz, image_0001.nii.gz, etc.
```

```pycon
>>> # Custom pattern
>>> write(bc, "output/", pattern="sub-{i:02d}_bold.nii.gz")
>>> # Creates: output/sub-00_bold.nii.gz, sub-01_bold.nii.gz, etc.
```

```pycon
>>> # With BIDS-style naming using metadata
>>> bc.metadata["filename"] = [f"sub-{s}_bold.nii.gz" for s in subjects]
>>> for i, bd in enumerate(bc):
...     bd.write(f"output/{bc.metadata.loc[i, 'filename']}")
```

