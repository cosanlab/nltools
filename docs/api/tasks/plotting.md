---
title: Brain plotting
label: page-tasks-plotting
---

Render a volume on the cortical surface, as a flatmap, or in an interactive viewer, and browse ICA/PCA components. `BrainData.plot` and `BrainData.iplot` call these. Plots of model output sit with their workflow. ROC and prediction plots are under [Prediction & cross-validation](prediction.md); adjacency-matrix plots are under [Similarity & RSA](similarity.md).

**Functions:**

Name | Description
---- | -----------
[`plot_surf`](#tasks-plotting-plot-surf) | Plot volumetric data on fsaverage surfaces in a tight montage.
[`plot_flatmap`](#tasks-plotting-plot-flatmap) | Plot brain data on cortical flatmap.
[`plot_interactive_brain`](#tasks-plotting-plot-interactive-brain) | Create an interactive brain visualization with nilearn.
[`component_viewer`](#tasks-plotting-component-viewer) | Interactively view the results of a `BrainData.decompose()` run.

## Functions

(tasks-plotting-plot-surf)=
### `plot_surf`

```python
plot_surf(brain, *, hemi = 'both', view = 'montage', surface = 'pial', template = 'fsaverage5', threshold = None, cmap = None, vmin = None, vmax = None, transparency = 'auto', bg_on_data = False, colorbar = True, colorbar_orientation = 'horizontal', figsize = (10, 8), title = None, radius = 3.0, interpolation = 'linear', zoom = 1.2, axes = None, save = None)
```

Plot volumetric data on fsaverage surfaces in a tight montage.

Like nilearn's `plot_img_on_surf` but with tight framing (via
`Axes3D.set_box_aspect(zoom=...)` + `set_axis_off`), an auto-applied
transparency mask (same convention as `plot_flatmap`), and a single shared
colorbar instead of one per subplot.

The grid is `len(view) × len(hemi)` — rows are views, columns are hemispheres.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path</code> | MNI-space image to plot. | *required*
`hemi` | <code>str \| list</code> | `'left'`, `'right'`, `'both'` (default), or a list subset like `['left']`. | <code>'both'</code>
`view` | <code>str \| list</code> | `'montage'` (default, → `['lateral', 'medial']`), a single view string, or any list subset of `'lateral'`, `'medial'`, `'dorsal'`, `'ventral'`, `'anterior'`, `'posterior'`. | <code>'montage'</code>
`surface` | <code>str</code> | fsaverage mesh to render on. One of `'pial'` (default), `'inflated'`, `'white'`, `'sphere'`. | <code>'pial'</code>
`template` | <code>str</code> | fsaverage resolution (`'fsaverage3'` … `'fsaverage'`). Default `'fsaverage5'`. | <code>'fsaverage5'</code>
`threshold` | <code>float \| str</code> | Absolute cutoff (`0.3`) or percentile string (`'95%'`). | <code>None</code>
`cmap` | <code>str</code> | Matplotlib colormap. By default, positive-only maps use ``"Reds"``, negative-only maps use ``"Blues_r"``, and mixed maps use ``"RdBu_r"``. | <code>None</code>
`vmin` | <code>float</code> | Colormap lower bound. Defaults to zero for positive-only maps, the data minimum for negative-only maps, and negative max-absolute value for mixed maps. | <code>None</code>
`vmax` | <code>float</code> | Colormap upper bound. Defaults to the data maximum for positive-only maps, zero for negative-only maps, and max-absolute value for mixed maps. | <code>None</code>
`transparency` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path \| None</code> | Binary mask used to NaN-out vertices outside the mask so the background shines through. `'auto'` (default) uses `BrainData.mask`; None disables masking. | <code>'auto'</code>
`bg_on_data` | <code>bool</code> | Whether to multiply data by the background. Default False. | <code>False</code>
`colorbar` | <code>bool</code> | Show a single shared colorbar. Default True. | <code>True</code>
`colorbar_orientation` | <code>str</code> | `'horizontal'` (default) or `'vertical'`. | <code>'horizontal'</code>
`figsize` | <code>tuple</code> | Figure size. Default (10, 8). | <code>(10, 8)</code>
`title` | <code>str</code> | Figure title. | <code>None</code>
`radius` | <code>float</code> | `vol_to_surf` sampling radius. Default 3.0. | <code>3.0</code>
`interpolation` | <code>str</code> | `vol_to_surf` interpolation. Default `'linear'`. | <code>'linear'</code>
`zoom` | <code>float</code> | Zoom factor for each 3-D axis (`Axes3D.set_box_aspect`). Default 1.2; try 1.4 for the tightest clean framing. | <code>1.2</code>
`axes` | <code>ndarray</code> | Pre-existing `Axes3D` array to draw into, of shape `(len(view), len(hemi))`. | <code>None</code>
`save` | <code>str</code> | Path to save the figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The surface figure.

(tasks-plotting-plot-flatmap)=
### `plot_flatmap`

```python
plot_flatmap(brain, *, threshold = None, cmap = None, vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

Projects MNI152 volumetric data onto an fsaverage surface and renders
as a 2D flattened cortical map. Uses nilearn's vol_to_surf for projection
and matplotlib's tripcolor for rendering.

This function provides publication-quality flatmap visualizations without
requiring external dependencies like pycortex.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path</code> | Image to plot. Data must be in MNI152 space. | *required*
`threshold` | <code>float or str</code> | Values below this absolute threshold are masked. Can be a float or percentile string like '95%'. Defaults to None (no threshold). | <code>None</code>
`cmap` | <code>str</code> | Matplotlib colormap. The default is ``"Reds"`` for positive-only maps, ``"Blues_r"`` for negative-only maps, and ``"RdBu_r"`` for mixed maps. | <code>None</code>
`vmax` | <code>float</code> | Maximum value. Defaults to the positive data maximum, zero for negative-only data, or max-absolute value for mixed data. | <code>None</code>
`vmin` | <code>float</code> | Minimum value. Defaults to zero for positive-only data, the negative data minimum, or negative max-absolute value for mixed data. | <code>None</code>
`template` | <code>str</code> | fsaverage resolution. Options: 'fsaverage3' (642 vertices), 'fsaverage4' (2562), 'fsaverage5' (10242, default), 'fsaverage6' (40962), 'fsaverage' (163842, full resolution). | <code>'fsaverage5'</code>
`with_curvature` | <code>bool</code> | Show sulcal/gyral pattern as grayscale background. Defaults to True. | <code>True</code>
`curvature_contrast` | <code>float</code> | Contrast of curvature (0=flat gray, 1=full contrast). Defaults to 0.5. | <code>0.5</code>
`curvature_brightness` | <code>float</code> | Mean brightness of curvature (0=dark, 1=bright). Defaults to 0.5. | <code>0.5</code>
`transparency` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path \| None</code> | Binary mask used to render vertices outside the mask as transparent (so the curvature shows through). `'auto'` (default) uses the input `BrainData`'s `.mask` when available, matching the behavior of the volumetric `.plot()`. Pass None to disable masking entirely. | <code>'auto'</code>
`colorbar` | <code>bool</code> | Show colorbar. Defaults to True. | <code>True</code>
`colorbar_orientation` | <code>str</code> | 'horizontal' or 'vertical'. Defaults to 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>tuple</code> | Figure size (width, height). Defaults to (12, 6). | <code>(12, 6)</code>
`title` | <code>str</code> | Figure title. Defaults to None. | <code>None</code>
`radius` | <code>float</code> | Sampling radius in mm for vol_to_surf projection. Larger values provide smoother projections. Defaults to 3.0. | <code>3.0</code>
`interpolation` | <code>str</code> | Interpolation for vol_to_surf. Options: 'linear', 'nearest_most_frequent'. Defaults to 'linear'. | <code>'linear'</code>
`axes` | <code>Axes</code> | Existing axes to plot on. If None, creates new figure. Defaults to None. | <code>None</code>
`save` | <code>str</code> | File path to save figure. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The figure containing the flatmap.

**Examples:**

Basic flatmap with default settings:

```python
from nltools.plotting import plot_flatmap
from nltools.data import BrainData

brain = BrainData("stats.nii.gz")
fig = plot_flatmap(brain)
```

Thresholded with custom colormap:

```python
fig = plot_flatmap(brain, threshold=2.5, cmap="hot")
```

Percentile threshold, no curvature:

```python
fig = plot_flatmap(brain, threshold="95%", with_curvature=False)
```

High resolution for publication:

```python
fig = plot_flatmap(brain, template="fsaverage6", figsize=(16, 8))
fig.savefig("flatmap.pdf", dpi=300)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Data is projected from MNI152 space to fsaverage surface space, so small
alignment differences are expected at boundaries. Higher resolution
templates (fsaverage6, fsaverage) produce sharper images but take longer
to render. The flat surfaces are cached by nilearn after the first
download (~50MB for fsaverage5).

</details>

(tasks-plotting-plot-interactive-brain)=
### `plot_interactive_brain`

```python
plot_interactive_brain(brain, *, threshold = 1e-06, surface = False, percentile_threshold = False, anatomical = None, **kwargs)
```

Create an interactive brain visualization with nilearn.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain` | <code>[BrainData](#page-data-brain-data)</code> | A 1-D (single volume) or 2-D (stack of volumes) instance. | *required*
`threshold` | <code>float \| str</code> | Initial threshold; a percentile string such as `'95%'` switches on `percentile_threshold`. Default 1e-6. | <code>1e-06</code>
`surface` | <code>bool</code> | Whether to create a surface-based plot. Default False. | <code>False</code>
`percentile_threshold` | <code>bool</code> | Whether to interpret threshold values as percentiles. Default False. | <code>False</code>
`anatomical` | <code>Nifti1Image \| str</code> | Background image; defaults to nilearn's MNI152 template. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to `nilearn.plotting.view_img` or `nilearn.plotting.view_img_on_surf`. | <code>{}</code>

<details class="note" open markdown="1">
<summary>Note</summary>

Returns nothing; the widgets render inline.

</details>

(tasks-plotting-component-viewer)=
### `component_viewer`

```python
component_viewer(output, tr = 2.0)
```

Interactively view the results of a `BrainData.decompose()` run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output` | <code>dict</code> | Output dictionary from `BrainData.decompose()` (keys `'components'`, `'weights'`, `'decomposition_object'`). | *required*
`tr` | <code>float</code> | Repetition time of the data in seconds. Default 2.0. | <code>2.0</code>

<details class="note" open markdown="1">
<summary>Note</summary>

Returns nothing; the interactive widgets render inline.

</details>
