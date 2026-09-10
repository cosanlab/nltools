---
title: data.braindata.plotting
label: page-data-braindata-plotting
---

Glass-brain, slice, flatmap, timeseries, and histogram plots for `BrainData`.

**Functions:**

Name | Description
---- | -----------
[`auto_select_colormap`](#data-braindata-plotting-auto-select-colormap) | Auto-select colormap based on data characteristics.
[`plot_brain`](#data-braindata-plotting-plot-brain) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap_brain`](#data-braindata-plotting-plot-flatmap-brain) | Plot brain data on cortical flatmap.
[`prepare_save_paths`](#data-braindata-plotting-prepare-save-paths) | Prepare save paths for multiple plot outputs.



## Functions

(data-braindata-plotting-auto-select-colormap)=
### `auto_select_colormap`

```python
auto_select_colormap(data)
```

Auto-select colormap based on data characteristics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | Brain data values. | *required*

**Returns:**

Type | Description
---- | -----------
<code>str</code> | ``'Reds'`` for positive-only data, ``'Blues_r'`` for     negative-only data, otherwise ``'RdBu_r'``.

(data-braindata-plotting-plot-brain)=
### `plot_brain`

```python
plot_brain(bd, *, method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to plot. | *required*
`method` | <code>str</code> | Visualization type ('glass', 'slices', 'timeseries', 'histogram'). | <code>'glass'</code>
`upper` | <code>str \| float \| None</code> | Upper threshold applied to the data (nltools semantics; may be a percentile string like ``"95%"``). | <code>None</code>
`lower` | <code>str \| float \| None</code> | Lower threshold applied to the data (nltools semantics). | <code>None</code>
`threshold` | <code>float \| str</code> | Absolute-value transparency cutoff forwarded to nilearn. Percentile strings such as ``"95%"`` are resolved over finite, nonzero magnitudes. Must be >= 0. | <code>None</code>
`view` | <code>str</code> | For ``method="slices"``, any non-empty combination of ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``). Default: ``"z"``. | <code>'z'</code>
`cut_coords` | <code>list or dict</code> | Cut coordinates for multi-slice views. If provided, takes precedence over ``view``-based defaults. Either a list of per-axis coordinate sequences whose length matches ``view``, or a dict keyed by axis letter (``{"x": [...], "z": [...]}``) from which entries for each axis in ``view`` are looked up. | <code>None</code>
`cmap` | <code>str</code> | Colormap name. By default, positive-only maps use ``"Reds"``, negative-only maps use ``"Blues_r"``, and mixed maps use ``"RdBu_r"``. | <code>None</code>
`bg_img` | <code>Nifti1Image or str</code> | Background image for slice views. | <code>None</code>
`ax` | <code>Axes</code> | Matplotlib axis to plot on. | <code>None</code>
`figsize` | <code>tuple</code> | default figure size if no axis (8, 6) | <code>(8, 6)</code>
`title` | <code>str</code> | Plot title. | <code>None</code>
`colorbar` | <code>bool</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>str</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>str</code> | Statistic for timeseries plots. Valid options: 'mean', 'median', 'std'. | <code>'mean'</code>
`limit` | <code>int</code> | Maximum number of images to render when ``bd`` contains multiple maps and ``method`` is ``"glass"`` or ``"slices"``. Default: 3. A warning is emitted if the data has more images than ``limit``. Ignored for single-image data and for matplotlib-based methods (``"timeseries"``, ``"histogram"``), which already aggregate across images. | <code>3</code>
`**kwargs` | <code>dict</code> | Additional arguments forwarded to `nilearn.plotting.plot_glass_brain` / `plot_stat_map`. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure \| list[Figure]</code> | For     single-image data, the figure object (last one created if     `method="slices"` produced multiple per-axis figures). For     multi-image data with `method` in `{"glass", "slices"}`, a list of     figures (one per image for glass; one per image-and-view pair for     slices). All figures auto-display in notebooks.

(data-braindata-plotting-plot-flatmap-brain)=
### `plot_flatmap_brain`

```python
plot_flatmap_brain(bd, *, threshold = None, cmap = None, vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to plot (must be in standard MNI space). | *required*
`threshold` | <code>float \| str</code> | Values below this absolute threshold are masked; percentile strings are accepted. | <code>None</code>
`cmap` | <code>str</code> | Matplotlib colormap. The default is sign-aware. | <code>None</code>
`vmax` | <code>float</code> | Maximum value; inferred from displayed data. | <code>None</code>
`vmin` | <code>float</code> | Minimum value; inferred from displayed data. | <code>None</code>
`template` | <code>str</code> | fsaverage resolution. Default: 'fsaverage5'. | <code>'fsaverage5'</code>
`with_curvature` | <code>bool</code> | Show sulcal/gyral pattern. Default: True. | <code>True</code>
`curvature_contrast` | <code>float</code> | Contrast of curvature. Default: 0.5. | <code>0.5</code>
`curvature_brightness` | <code>float</code> | Mean brightness of curvature. Default: 0.5. | <code>0.5</code>
`transparency` | <code>str or float or array - like</code> | Transparency/alpha applied to the surface data. ``'auto'`` (default) lets the renderer choose. | <code>'auto'</code>
`colorbar` | <code>bool</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>str</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>tuple</code> | Figure size. Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>str</code> | Figure title. | <code>None</code>
`radius` | <code>float</code> | sampling radius in mm for vol_to_surf. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>str</code> | Interpolation for vol_to_surf. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>Axes</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>str</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The rendered figure.

(data-braindata-plotting-prepare-save-paths)=
### `prepare_save_paths`

```python
prepare_save_paths(save, idx = None)
```

Prepare save paths for multiple plot outputs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`save` | <code>str \| Path</code> | Base save path; its extension is reused (default `png`). | *required*
`idx` | <code>int \| None</code> | Image index appended as ``_img{idx}`` to the base filename, to disambiguate saves across multiple images. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | `'glass'` maps to one path; `'slices'` maps to a dict of per-axis     (`'x'`, `'y'`, `'z'`) paths.
