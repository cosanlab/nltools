---
title: data.adjacency.plotting
label: data-adjacency-plotting
---

Plotting functions for Adjacency matrices.

**Functions:**

Name | Description
---- | -----------
[`plot_adjacency`](#data-adjacency-plotting-plot-adjacency) | Create Heatmap of Adjacency Matrix.
[`plot_mds`](#data-adjacency-plotting-plot-mds) | Plot Multidimensional Scaling.



## Functions

(data-adjacency-plotting-plot-adjacency)=
### `plot_adjacency`

```python
plot_adjacency(adj, limit = 3, axes = None, *args, **kwargs)
```

Create Heatmap of Adjacency Matrix.

Can pass in any ``sns.heatmap`` argument.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency object to plot. | *required*
`limit` | <code>int</code> | Number of heatmaps to plot if object contains multiple adjacencies (default: 3). | <code>3</code>
`axes` |  | Matplotlib axis handle. | <code>None</code>

(data-adjacency-plotting-plot-mds)=
### `plot_mds`

```python
plot_mds(adj, *, n_components = 2, metric_mds = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, **kwargs)
```

Plot Multidimensional Scaling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency object to plot (must be a distance matrix). | *required*
`n_components` | <code>int</code> | Number of dimensions to project (can be 2 or 3). | <code>2</code>
`metric_mds` | <code>bool</code> | Perform metric (True) or non-metric (False) dimensional scaling. Default True. | <code>True</code>
`labels` | <code>list</code> | Can override labels stored in Adjacency Class. | <code>None</code>
`labels_color` | <code>list</code> | List of colors for labels. | <code>None</code>
`cmap` |  | Colormap instance (default: ``plt.cm.hot_r``). | <code>None</code>
`view` | <code>tuple</code> | View for 3-Dimensional plot. Default (30, 20). | <code>(30, 20)</code>
`figsize` | <code>list</code> | Figure size. Default [12, 8]. | <code>None</code>
`ax` |  | Matplotlib axis handle. | <code>None</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. | <code>-1</code>
