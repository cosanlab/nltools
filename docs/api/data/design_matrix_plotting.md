## `plotting`

DesignMatrix visualization functions.

Standalone functions extracted from ``DesignMatrix`` methods. Each takes a
``DesignMatrix`` instance (``dm``) as its first argument. ``DesignMatrix.plot``
dispatches over ``method`` to the helpers here, mirroring ``BrainData.plot``.

**Methods:**

Name | Description
---- | -----------
[`plot_corr`](#plot_corr) | Render a labeled correlation heatmap of the columns.
[`plot_designmatrix`](#plot_designmatrix) | Visualize a DesignMatrix, dispatching over ``method``.
[`plot_matrix`](#plot_matrix) | Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).
[`plot_timeseries`](#plot_timeseries) | Plot regressor time courses as overlaid lines.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`VALID_PLOT_METHODS`](#VALID_PLOT_METHODS) |  | 

### Methods

#### `plot_corr`

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
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>
`metric` | <code>[str](#str)</code> | ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`figsize` | <code>[tuple](#tuple) \| None</code> | Figure size; scales with the number of columns when omitted. | <code>None</code>
`title` | <code>[str](#str) \| None</code> | Optional axis title. | <code>None</code>
`cmap` | <code>[str](#str) \| None</code> | Colormap name. Default: ``'RdBu_r'``. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.pyplot.Axes) \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Optional path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to ``seaborn.heatmap`` (e.g. ``annot=False``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

#### `plot_designmatrix`

```python
plot_designmatrix(dm: DesignMatrix, method: str = 'matrix', *, columns: list[str] | None = None, rescale: bool = True, metric: str = 'pearson', ax: plt.Axes | None = None, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, save: str | None = None, **kwargs: str | None)
```

Visualize a DesignMatrix, dispatching over ``method``.

See `DesignMatrix.plot` for the full argument documentation.

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure: The figure containing the plot.

#### `plot_matrix`

```python
plot_matrix(dm: DesignMatrix, *, columns: list[str] | None = None, rescale: bool = True, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`rescale` | <code>[bool](#bool)</code> | If True, rescale each column by its L2 norm so columns with different native magnitudes are visually comparable (SPM/nilearn convention). Default: True. | <code>True</code>
`figsize` | <code>[tuple](#tuple) \| None</code> | Figure size; defaults to ``(4, 6)`` when a new figure is made. | <code>None</code>
`title` | <code>[str](#str) \| None</code> | Optional axis title. | <code>None</code>
`cmap` | <code>[str](#str) \| None</code> | Colormap name. Default: ``'gray'``. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.pyplot.Axes) \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Optional path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to ``seaborn.heatmap``. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

#### `plot_timeseries`

```python
plot_timeseries(dm: DesignMatrix, *, columns: list[str] | None = None, figsize: tuple | None = None, title: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Plot regressor time courses as overlaid lines.

One line is drawn per column. Pass the same ``ax`` across calls to overlay
multiple DesignMatrices (e.g. original vs. convolved).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`figsize` | <code>[tuple](#tuple) \| None</code> | Figure size; defaults to ``(8, 4)`` when a new figure is made. | <code>None</code>
`title` | <code>[str](#str) \| None</code> | Optional axis title. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.pyplot.Axes) \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Optional path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to ``matplotlib.axes.Axes.plot`` for each line. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

