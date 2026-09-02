---
title: data.designmatrix.plotting
label: page-data-design-matrix-plotting
---

Visualize a DesignMatrix as a heatmap, overlaid time courses, or a correlation matrix.

`DesignMatrix.plot` dispatches over `method` to `plot_matrix`,
`plot_timeseries`, and `plot_corr`, mirroring `BrainData.plot`.

**Functions:**

Name | Description
---- | -----------
[`plot_corr`](#data-design-matrix-plotting-plot-corr) | Render a labeled correlation heatmap of the columns.
[`plot_designmatrix`](#data-design-matrix-plotting-plot-designmatrix) | Visualize a DesignMatrix, dispatching over `method`.
[`plot_matrix`](#data-design-matrix-plotting-plot-matrix) | Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).
[`plot_timeseries`](#data-design-matrix-plotting-plot-timeseries) | Plot regressor time courses as overlaid lines.

## Functions

(data-design-matrix-plotting-plot-corr)=
### `plot_corr`

```python
plot_corr(dm: DesignMatrix, *, columns: list[str] | None = None, metric: str = 'pearson', figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Render a labeled correlation heatmap of the columns.

Reuses `DesignMatrix.corr`, which returns a similarity ``Adjacency``
with the unit diagonal dropped; the diagonal is restored to ``1.0`` here so
the heatmap reads as a standard correlation matrix.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#page-data-design-matrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>list[str] \| None</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>
`metric` | <code>str</code> | ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`figsize` | <code>tuple \| None</code> | Figure size; scales with the number of columns when omitted. | <code>None</code>
`title` | <code>str \| None</code> | Axis title. | <code>None</code>
`cmap` | <code>str \| None</code> | Colormap name. Default: ``'RdBu_r'``. | <code>None</code>
`ax` | <code>Axes \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>str \| None</code> | Path to save the figure. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to ``seaborn.heatmap`` (e.g. ``annot=False``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The rendered figure.

(data-design-matrix-plotting-plot-designmatrix)=
### `plot_designmatrix`

```python
plot_designmatrix(dm: DesignMatrix, method: str = 'matrix', *, columns: list[str] | None = None, rescale: bool = True, metric: str = 'pearson', ax: plt.Axes | None = None, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, save: str | None = None, **kwargs: str | None)
```

Visualize a DesignMatrix, dispatching over `method`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#page-data-design-matrix)</code> | DesignMatrix instance. | *required*
`method` | <code>str</code> | ``'matrix'`` (SPM-style heatmap), ``'timeseries'`` (overlaid line plot), or ``'corr'`` (correlation heatmap). Default: ``'matrix'``. | <code>'matrix'</code>
`columns` | <code>list[str] \| None</code> | Subset of columns to plot. Defaults to all. | <code>None</code>
`rescale` | <code>bool</code> | ``'matrix'`` only; rescale each column by its L2 norm. Default: True. | <code>True</code>
`metric` | <code>str</code> | ``'corr'`` only; ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`ax` | <code>Axes \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`figsize` | <code>tuple \| None</code> | Figure size; per-method default when omitted. | <code>None</code>
`title` | <code>str \| None</code> | Axis title. | <code>None</code>
`cmap` | <code>str \| None</code> | Colormap (``'matrix'`` / ``'corr'``). | <code>None</code>
`save` | <code>str \| None</code> | Path to save the figure. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to the underlying plotter (``seaborn.heatmap`` for ``'matrix'`` / ``'corr'``; ``matplotlib.axes.Axes.plot`` for ``'timeseries'``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The figure containing the plot.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `method` is not one of the three supported values.

(data-design-matrix-plotting-plot-matrix)=
### `plot_matrix`

```python
plot_matrix(dm: DesignMatrix, *, columns: list[str] | None = None, rescale: bool = True, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#page-data-design-matrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>list[str] \| None</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`rescale` | <code>bool</code> | If True, rescale each column by its L2 norm so columns with different native magnitudes are visually comparable (SPM/nilearn convention). Default: True. | <code>True</code>
`figsize` | <code>tuple \| None</code> | Figure size; defaults to ``(4, 6)`` when a new figure is made. | <code>None</code>
`title` | <code>str \| None</code> | Axis title. | <code>None</code>
`cmap` | <code>str \| None</code> | Colormap name. Default: ``'gray'``. | <code>None</code>
`ax` | <code>Axes \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>str \| None</code> | Path to save the figure. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to ``seaborn.heatmap``. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The rendered figure.

(data-design-matrix-plotting-plot-timeseries)=
### `plot_timeseries`

```python
plot_timeseries(dm: DesignMatrix, *, columns: list[str] | None = None, figsize: tuple | None = None, title: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Plot regressor time courses as overlaid lines.

One line is drawn per column. Pass the same ``ax`` across calls to overlay
multiple DesignMatrices (e.g. original vs. convolved).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#page-data-design-matrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>list[str] \| None</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`figsize` | <code>tuple \| None</code> | Figure size; defaults to ``(8, 4)`` when a new figure is made. | <code>None</code>
`title` | <code>str \| None</code> | Axis title. | <code>None</code>
`ax` | <code>Axes \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>str \| None</code> | Path to save the figure. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to ``matplotlib.axes.Axes.plot`` for each line. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The rendered figure.
