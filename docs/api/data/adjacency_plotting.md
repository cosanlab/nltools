(data-adjacency-plotting-plotting)=
## `plotting`

Plotting functions for Adjacency matrices.

**Methods:**

Name | Description
---- | -----------
[`plot_adjacency`](#data-adjacency-plotting-plot-adjacency) | Create Heatmap of Adjacency Matrix.
[`plot_mds`](#data-adjacency-plotting-plot-mds) | Plot Multidimensional Scaling.



### Methods

(data-adjacency-plotting-plot-adjacency)=
#### `plot_adjacency`

```python
plot_adjacency(adj, limit = 3, axes = None, *args, **kwargs)
```

Create Heatmap of Adjacency Matrix.

Can pass in any ``sns.heatmap`` argument.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency object to plot. | *required*
`limit` | <code>[int](#int)</code> | Number of heatmaps to plot if object contains multiple adjacencies (default: 3). | <code>3</code>
`axes` |  | Matplotlib axis handle. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | None

(data-adjacency-plotting-plot-mds)=
#### `plot_mds`

```python
plot_mds(adj, n_components = 2, metric = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, *args, **kwargs)
```

Plot Multidimensional Scaling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency object to plot (must be a distance matrix). | *required*
`n_components` | <code>[int](#int)</code> | Number of dimensions to project (can be 2 or 3). | <code>2</code>
`metric` | <code>[bool](#bool)</code> | Perform metric or non-metric dimensional scaling. Default True. | <code>True</code>
`labels` | <code>[list](#list)</code> | Can override labels stored in Adjacency Class. | <code>None</code>
`labels_color` | <code>[list](#list)</code> | List of colors for labels. | <code>None</code>
`cmap` |  | Colormap instance (default: ``plt.cm.hot_r``). | <code>None</code>
`view` | <code>[tuple](#tuple)</code> | View for 3-Dimensional plot. Default (30, 20). | <code>(30, 20)</code>
`figsize` | <code>[list](#list)</code> | Figure size. Default [12, 8]. | <code>None</code>
`ax` |  | Matplotlib axis handle. | <code>None</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
 | None

