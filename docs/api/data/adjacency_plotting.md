---
title: data.adjacency.plotting
label: page-data-adjacency-plotting
---

Plotting functions for Adjacency matrices.

**Functions:**

Name | Description
---- | -----------
[`plot_adjacency`](#data-adjacency-plotting-plot-adjacency) | Create a heatmap of an Adjacency matrix.
[`plot_mds`](#data-adjacency-plotting-plot-mds) | Plot multidimensional scaling.



## Functions

(data-adjacency-plotting-plot-adjacency)=
### `plot_adjacency`

```python
plot_adjacency(adj, limit = 3, axes = None, *args, **kwargs)
```

Create a heatmap of an Adjacency matrix.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency object to plot. | *required*
`limit` | <code>int</code> | Number of heatmaps to plot if the object contains multiple matrices. Default 3. | <code>3</code>
`axes` | <code>Axes</code> | Axis to draw on (single matrix only). | <code>None</code>
`*args` | <code>tuple</code> | Forwarded positionally to `seaborn.heatmap`. | <code>()</code>
`**kwargs` | <code>dict</code> | Forwarded to `seaborn.heatmap`. | <code>{}</code>

(data-adjacency-plotting-plot-mds)=
### `plot_mds`

```python
plot_mds(adj, *, n_components = 2, metric_mds = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, **kwargs)
```

Plot multidimensional scaling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency object to plot (must be a single distance matrix). | *required*
`n_components` | <code>int</code> | Number of dimensions to project (2 or 3). | <code>2</code>
`metric_mds` | <code>bool</code> | Perform metric (True) or non-metric (False) scaling. Default True. | <code>True</code>
`labels` | <code>list</code> | Overrides the labels stored on `adj`. | <code>None</code>
`labels_color` | <code>list</code> | One color per label. | <code>None</code>
`cmap` | <code>Colormap</code> | Colormap. Default `plt.cm.hot_r`. | <code>None</code>
`view` | <code>tuple</code> | Elevation/azimuth for a 3-D plot. Default (30, 20). | <code>(30, 20)</code>
`figsize` | <code>list</code> | Figure size. Default [12, 8]. | <code>None</code>
`ax` | <code>Axes</code> | Axis to draw on. | <code>None</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. | <code>-1</code>
`**kwargs` | <code>dict</code> | Forwarded to `sklearn.manifold.MDS`. | <code>{}</code>
