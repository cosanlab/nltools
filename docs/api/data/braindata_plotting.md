(data-braindata-plotting-plotting)=
## `plotting`

BrainData plotting functions.

**Methods:**

Name | Description
---- | -----------
[`auto_select_colormap`](#data-braindata-plotting-auto-select-colormap) | Auto-select colormap based on data characteristics.
[`plot_brain`](#data-braindata-plotting-plot-brain) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap_brain`](#data-braindata-plotting-plot-flatmap-brain) | Plot brain data on cortical flatmap.
[`prepare_save_paths`](#data-braindata-plotting-prepare-save-paths) | Prepare save paths for multiple plot outputs.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`DEFAULT_SLICE_CUT_COORDS` |  | 

### Methods

(data-braindata-plotting-auto-select-colormap)=
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

(data-braindata-plotting-plot-brain)=
#### `plot_brain`

```python
plot_brain(bd, method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`method` | <code>[str](#str)</code> | Visualization type ('glass', 'slices', 'timeseries', 'histogram'). | <code>'glass'</code>
`upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold applied to the data (nltools semantics; may be a percentile string like ``"95%"``). | <code>None</code>
`lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold applied to the data (nltools semantics). | <code>None</code>
`threshold` | <code>[float](#float)</code> | Absolute-value transparency cutoff forwarded to the underlying nilearn plot function. Voxels with ``|value| < threshold`` are rendered transparent. Must be >= 0. Use ``upper``/``lower`` for one-sided data thresholding. | <code>None</code>
`view` | <code>[str](#str)</code> | For ``method="slices"``, any non-empty combination of ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``). Default: ``"z"``. | <code>'z'</code>
`cut_coords` | <code>[list](#list) or [dict](#dict)</code> | Cut coordinates for multi-slice views. If provided, takes precedence over ``view``-based defaults. Either a list of per-axis coordinate sequences whose length matches ``view``, or a dict keyed by axis letter (``{"x": [...], "z": [...]}``) from which entries for each axis in ``view`` are looked up. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Background image for slice views. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis to plot on. | <code>None</code>
`figsize` | <code>[tuple](#tuple)</code> | default figure size if no axis (8, 6) | <code>(8, 6)</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Valid options: 'mean', 'median', 'std'. | <code>'mean'</code>
`limit` | <code>[int](#int)</code> | Maximum number of images to render when ``bd`` contains multiple maps and ``method`` is ``"glass"`` or ``"slices"``. Default: 3. A warning is emitted if the data has more images than ``limit``. Ignored for single-image data and for matplotlib-based methods (``"timeseries"``, ``"histogram"``), which already aggregate across images. | <code>3</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure or list[matplotlib.figure.Figure]: For
 | single-image data, the figure object (last one created if
 | ``method="slices"`` produced multiple per-axis figures). For
 | multi-image data with ``method`` in ``{"glass", "slices"}``, a list
 | of figures (one per image for glass; one per image-and-view pair for
 | slices). All figures auto-display in notebooks.

(data-braindata-plotting-plot-flatmap-brain)=
#### `plot_flatmap_brain`

```python
plot_flatmap_brain(bd, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
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

(data-braindata-plotting-prepare-save-paths)=
#### `prepare_save_paths`

```python
prepare_save_paths(save, idx = None)
```

Prepare save paths for multiple plot outputs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`save` |  | Base save path (str or Path) | *required*
`idx` | <code>[int](#int)</code> | Image index appended as ``_img{idx}`` to the base filename. Used to disambiguate saves across multiple images. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with 'glass' and 'slices' keys containing save paths

