---
title: io
label: io
---

nltools I/O utilities.

HDF5 serialization for neuroimaging data types.

**Functions:**

Name | Description
---- | -----------
[`is_h5_path`](#io-is-h5-path) | Check if a file path indicates an HDF5 file.
[`load_brain_data_h5`](#io-load-brain-data-h5) | Load BrainData from HDF5 file.
[`to_h5`](#io-to-h5) | Save BrainData or Adjacency objects to HDF5 files.



## Functions

(io-is-h5-path)=
### `is_h5_path`

```python
is_h5_path(file_name) -> bool
```

Check if a file path indicates an HDF5 file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` |  | Path to check (str or Path object). | *required*

**Returns:**

Type | Description
---- | -----------
<code>bool</code> | True if the file has an HDF5 extension (.h5 or .hdf5).

**Examples:**

```pycon
>>> is_h5_path("data.h5")
True
>>> is_h5_path("data.csv")
False
>>> is_h5_path(Path("results.hdf5"))
True
```

(io-load-brain-data-h5)=
### `load_brain_data_h5`

```python
load_brain_data_h5(file_path, mask = None)
```

Load BrainData from HDF5 file.

Supports the v0.6 layout (X/Y as h5py groups with ``columns`` + ``values``)
and the legacy deepdish/PyTables layout written by nltools <= 0.5.1
(X/Y as flat datasets with sibling ``X_columns``/``X_index`` nodes).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_path` |  | Path to HDF5 file. | *required*
`mask` |  | Optional mask to use. If None, loads mask from file if available. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dictionary containing loaded data, X, Y, and optionally mask info.

(io-to-h5)=
### `to_h5`

```python
to_h5(obj, file_name, obj_type = 'brain_data', h5_compression = 'gzip')
```

Save BrainData or Adjacency objects to HDF5 files.

Uses h5py for both types; X/Y (BrainData) and Y (Adjacency) are stored
as polars-compatible groups with ``columns`` and ``values`` datasets.
A BrainData mask is always stored by value (data + affine datasets); its
filename is stored alongside only when the mask is file-backed, so
in-memory masks serialize without one and round-trip by value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`obj` |  | Object to save (BrainData or Adjacency). | *required*
`file_name` |  | Path to save file to. | *required*
`obj_type` |  | Type of object ('brain_data' or 'adjacency'). | <code>'brain_data'</code>
`h5_compression` |  | Compression type for h5py datasets. | <code>'gzip'</code>
