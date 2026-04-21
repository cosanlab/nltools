## `plotting`

BrainData plotting functions.

**Methods:**

Name | Description
---- | -----------
[`auto_select_colormap`](#auto_select_colormap) | Auto-select colormap based on data characteristics.
[`plot_brain`](#plot_brain) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap_brain`](#plot_flatmap_brain) | Plot brain data on cortical flatmap.
[`prepare_save_paths`](#prepare_save_paths) | Prepare save paths for multiple plot outputs.



### Methods

#### `auto_select_colormap`

```python
auto_select_colormap(data)
```

Auto-select colormap based on data characteristics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | numpy array of brain data | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Colormap name

#### `plot_brain`

```python
plot_brain(bd, method = 'glass', upper = None, lower = None, threshold = None, cut_coords = None, cmap = None, bg_img = None, ax = None, title = None, colorbar = True, save = None, stat = 'mean', **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`method` | <code>[str](#str)</code> | Visualization type ('glass', 'slices', 'timeseries', 'histogram'). | <code>'glass'</code>
`upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter. If positive, sets upper (shows values above threshold). If negative, sets lower (shows values below threshold). | <code>None</code>
`cut_coords` | <code>[list](#list)</code> | Cut coordinates for multi-slice views. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Background image for slice views. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis to plot on. | <code>None</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Valid options: 'mean', 'median', 'std'. | <code>'mean'</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Display or matplotlib Figure.

#### `plot_flatmap_brain`

```python
plot_flatmap_brain(bd, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`threshold` | <code>[float](#float)</code> | Values below this absolute threshold are masked. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Matplotlib colormap for data. Default: 'RdBu_r'. | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap. | <code>None</code>
`template` | <code>[str](#str)</code> | fsaverage resolution. Default: 'fsaverage5'. | <code>'fsaverage5'</code>
`with_curvature` | <code>[bool](#bool)</code> | Show sulcal/gyral pattern. Default: True. | <code>True</code>
`curvature_contrast` | <code>[float](#float)</code> | Contrast of curvature. Default: 0.5. | <code>0.5</code>
`curvature_brightness` | <code>[float](#float)</code> | Mean brightness of curvature. Default: 0.5. | <code>0.5</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size. Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | sampling radius in mm for vol_to_surf. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation for vol_to_surf. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

#### `prepare_save_paths`

```python
prepare_save_paths(save)
```

Prepare save paths for multiple plot outputs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`save` |  | Base save path (str or Path) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with 'glass' and 'slices' keys containing save paths

