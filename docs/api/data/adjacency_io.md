---
title: data.adjacency.io
label: page-data-adjacency-io
---

I/O functions for Adjacency objects.

**Functions:**

Name | Description
---- | -----------
[`read_h5`](#data-adjacency-io-read-h5) | Read current and legacy vector layouts into a normalized Adjacency.
[`to_graph`](#data-adjacency-io-to-graph) | Convert Adjacency into networkx graph.
[`write`](#data-adjacency-io-write) | Write an Adjacency to a `.csv` or `.h5` file.



## Functions

(data-adjacency-io-read-h5)=
### `read_h5`

```python
read_h5(file_name, matrix_type = None)
```

Read current and legacy vector layouts into a normalized Adjacency.

(data-adjacency-io-to-graph)=
### `to_graph`

```python
to_graph(adj)
```

Convert Adjacency into networkx graph.

Only works on single matrices for now.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance (must be a single matrix). | *required*

**Returns:**

Type | Description
---- | -----------
<code>Graph or DiGraph</code> | Graph representation of the     adjacency matrix. Uses DiGraph for directed matrices.

(data-adjacency-io-write)=
### `write`

```python
write(adj, file_name, method = 'long')
```

Write an Adjacency to a `.csv` or `.h5` file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency object to write. | *required*
`file_name` | <code>str \| Path</code> | Output path; an `.h5`/`.hdf5` suffix writes HDF5. | *required*
`method` | <code>str</code> | Layout for CSV output, `'long'` (vectorized rows) or `'square'` (single matrix only). | <code>'long'</code>
