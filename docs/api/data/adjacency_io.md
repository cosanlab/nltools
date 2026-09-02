---
title: data.adjacency.io
label: data-adjacency-io
---

I/O functions for Adjacency objects.

**Functions:**

Name | Description
---- | -----------
[`to_graph`](#data-adjacency-io-to-graph) | Convert Adjacency into networkx graph.
[`write`](#data-adjacency-io-write) | Write out Adjacency object to csv file.



## Functions

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
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance (must be a single matrix). | *required*

**Returns:**

Type | Description
---- | -----------
<code>Graph or DiGraph</code> | Graph representation of the     adjacency matrix. Uses DiGraph for directed matrices.

(data-adjacency-io-write)=
### `write`

```python
write(adj, file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | Adjacency object to write | *required*
`file_name` | <code>str</code> | name of file name to write | *required*
`method` | <code>str</code> | method to write out data ['long','square'] | <code>'long'</code>
