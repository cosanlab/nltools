## `nltools.data.adjacency.io`

I/O functions for Adjacency objects.

**Functions:**

Name | Description
---- | -----------
[`to_graph`](#nltools.data.adjacency.io.to_graph) | Convert Adjacency into networkx graph.
[`write`](#nltools.data.adjacency.io.write) | Write out Adjacency object to csv file.



### Functions#### `nltools.data.adjacency.io.to_graph`

```python
to_graph(adj)
```

Convert Adjacency into networkx graph.

Only works on single matrices for now.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance (must be a single matrix). | *required*

**Returns:**

Type | Description
---- | -----------
 | networkx.Graph or networkx.DiGraph: Graph representation of the adjacency matrix. Uses DiGraph for directed matrices.

#### `nltools.data.adjacency.io.write`

```python
write(adj, file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | Adjacency object to write | *required*
`file_name` | <code>[str](#str)</code> | name of file name to write | *required*
`method` | <code>[str](#str)</code> | method to write out data ['long','square'] | <code>'long'</code>

