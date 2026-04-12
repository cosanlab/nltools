## `plotting`

nltools.plotting — Visualization utilities for neuroimaging analysis.

This package provides standalone plotting functions organized into
focused submodules:

- **brain**: Surface plots, flatmaps, and interactive brain viewers
- **adjacency**: Adjacency matrix visualizations (stacked, silhouette, distance)
- **prediction**: Model output plots (ROC, SVM margin, regression, logistic)
- **decomposition**: ICA/PCA component viewer

All public functions are re-exported here for convenience::

    from nltools.plotting import plot_surface, plot_roc, component_viewer  # all work

**Modules:**

Name | Description
---- | -----------
[`adjacency`](#adjacency) | Adjacency matrix visualization — stacked plots, distance, and silhouette.
[`brain`](#brain) | Brain visualization — surface plots, flatmaps, and interactive viewers.
[`decomposition`](#decomposition) | ICA/PCA component viewer — interactive decomposition explorer.
[`prediction`](#prediction) | Model output visualization — ROC, SVM margin, regression, and logistic plots.

**Methods:**

Name | Description
---- | -----------
[`component_viewer`](#component_viewer) | This a function to interactively view the results of a decomposition analysis
[`plot_between_label_distance`](#plot_between_label_distance) | Heatmap of average pairwise distance between every label pair.
[`plot_dist_from_hyperplane`](#plot_dist_from_hyperplane) | Plot SVM Classification Distance from Hyperplane
[`plot_flatmap`](#plot_flatmap) | Plot brain data on cortical flatmap.
[`plot_interactive_brain`](#plot_interactive_brain) | This function leverages nilearn's new javascript based brain viewer functions to create interactive plotting functionality.
[`plot_mean_label_distance`](#plot_mean_label_distance) | Violin plot of within- vs between-label distances.
[`plot_probability`](#plot_probability) | Plot Classification Probability
[`plot_roc`](#plot_roc) | Plot 1-Specificity by Sensitivity
[`plot_scatter`](#plot_scatter) | Plot Prediction Scatterplot
[`plot_silhouette`](#plot_silhouette) | Silhouette plot indicating between- vs within-label distance.
[`plot_stacked_adjacency`](#plot_stacked_adjacency) | Create stacked adjacency to illustrate similarity.
[`plot_surface`](#plot_surface) | Plot neuroimaging data on cortical surface.



### Methods

#### `component_viewer`

```python
component_viewer(output, tr = 2.0)
```

This a function to interactively view the results of a decomposition analysis

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output` |  | (dict) output dictionary from running Brain_data.decompose() | *required*
`tr` |  | (float) repetition time of data | <code>2.0</code>

#### `plot_between_label_distance`

```python
plot_between_label_distance(distance, labels, ax = None, permutation_test = True, n_permute = 5000, fontsize = 18, **kwargs)
```

Heatmap of average pairwise distance between every label pair.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | Square pairwise distance matrix (np.ndarray or polars DataFrame). | *required*
`labels` |  | Array-like of length N giving a group label for each row/column. | *required*
`ax` |  | Matplotlib axis to plot on (optional). | <code>None</code>
`permutation_test` |  | If True, also compute mean-difference and p-value matrices. | <code>True</code>
`n_permute` |  | Number of permutations. | <code>5000</code>
`fontsize` |  | Reserved for future use; currently unused. | <code>18</code>
`**kwargs` |  | Passed to seaborn.heatmap. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Without permutation_test: (long_df, within_mean_df)
 | With permutation_test: (long_df, within_mean_df, mean_diff_df, p_df)
 | All frames are polars DataFrames. `long_df` has columns
 | [Distance, Group, Comparison]. The three square-matrix-like frames
 | are long format with columns [label1, label2, <value>] so they can
 | be pivoted to a matrix if needed.

#### `plot_dist_from_hyperplane`

```python
plot_dist_from_hyperplane(stats_output)
```

Plot SVM Classification Distance from Hyperplane

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` |  | a pandas file with prediction output | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fig` |  | Will return a seaborn plot of distance from hyperplane

#### `plot_flatmap`

```python
plot_flatmap(brain, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
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
`brain` |  | BrainData, nibabel Nifti1Image, or file path to NIfTI image. Data must be in MNI152 space. | *required*
`threshold` | <code>[float](#float) or [str](#str)</code> | Values below this absolute threshold are masked. Can be a float or percentile string like '95%'. Defaults to None (no threshold). | <code>None</code>
`cmap` | <code>[str](#str)</code> | Matplotlib colormap for data. Defaults to 'RdBu_r' (diverging red-blue). | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap. If None, uses symmetric max of absolute values. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap. If None and vmax is set, uses -vmax for diverging maps. | <code>None</code>
`template` | <code>[str](#str)</code> | fsaverage resolution. Options: 'fsaverage3' (642 vertices), 'fsaverage4' (2562), 'fsaverage5' (10242, default), 'fsaverage6' (40962), 'fsaverage' (163842, full resolution). | <code>'fsaverage5'</code>
`with_curvature` | <code>[bool](#bool)</code> | Show sulcal/gyral pattern as grayscale background. Defaults to True. | <code>True</code>
`curvature_contrast` | <code>[float](#float)</code> | Contrast of curvature (0=flat gray, 1=full contrast). Defaults to 0.5. | <code>0.5</code>
`curvature_brightness` | <code>[float](#float)</code> | Mean brightness of curvature (0=dark, 1=bright). Defaults to 0.5. | <code>0.5</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Defaults to True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Defaults to 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size (width, height). Defaults to (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. Defaults to None. | <code>None</code>
`radius` | <code>[float](#float)</code> | Sampling radius in mm for vol_to_surf projection. Larger values provide smoother projections. Defaults to 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation for vol_to_surf. Options: 'linear', 'nearest_most_frequent'. Defaults to 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. If None, creates new figure. Defaults to None. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure: The figure containing the flatmap.

**Examples:**

Basic flatmap with default settings:

```pycon
>>> from nltools.plotting import plot_flatmap
>>> from nltools.data import BrainData
>>> brain = BrainData('stats.nii.gz')
>>> fig = plot_flatmap(brain)
```

Thresholded with custom colormap:

```pycon
>>> fig = plot_flatmap(brain, threshold=2.5, cmap='hot')
```

Percentile threshold, no curvature:

```pycon
>>> fig = plot_flatmap(brain, threshold='95%', with_curvature=False)
```

High resolution for publication:

```pycon
>>> fig = plot_flatmap(brain, template='fsaverage6', figsize=(16, 8))
>>> fig.savefig('flatmap.pdf', dpi=300)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Data is projected from MNI152 space to fsaverage surface space.
  Small alignment differences are expected at boundaries.
- Higher resolution templates (fsaverage6, fsaverage) produce
  sharper images but take longer to render.
- The flat surfaces are cached by nilearn after first download
  (~50MB for fsaverage5).

</details>

#### `plot_interactive_brain`

```python
plot_interactive_brain(brain, threshold = 1e-06, surface = False, percentile_threshold = False, anatomical = None, **kwargs)
```

This function leverages nilearn's new javascript based brain viewer functions to create interactive plotting functionality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain` | <code>[BrainData](#nltools.BrainData)</code> | a BrainData instance of 1d or 2d shape (i.e. 3d or 4d volume) | *required*
`threshold` | <code>[float](#float) / [str](#str)</code> | threshold to initialize the visualization, maybe be a percentile string; default 0 | <code>1e-06</code>
`surface` | <code>[bool](#bool)</code> | whether to create a surface-based plot; default False | <code>False</code>
`percentile_threshold` | <code>[bool](#bool)</code> | whether to interpret threshold values as percentiles | <code>False</code>
`kwargs` |  | optional arguments to nilearn.view_img or nilearn.view_img_on_surf | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | interactive brain viewer widget

#### `plot_mean_label_distance`

```python
plot_mean_label_distance(distance, labels, ax = None, permutation_test = False, n_permute = 5000, fontsize = 18, **kwargs)
```

Violin plot of within- vs between-label distances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | Square pairwise distance matrix (np.ndarray or polars DataFrame). | *required*
`labels` |  | Array-like of length N giving a group label for each row/column. | *required*
`ax` |  | Matplotlib axis to plot on (optional). | <code>None</code>
`permutation_test` |  | If True, run a two-sample permutation test per group. | <code>False</code>
`n_permute` |  | Number of permutations. | <code>5000</code>
`fontsize` |  | Font size for plot labels. | <code>18</code>
`**kwargs` |  | Passed to seaborn.violinplot. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | pl.DataFrame with columns [Distance, Group, Type] in long format.
 | If permutation_test=True, returns (pl.DataFrame, dict of per-group stats).

#### `plot_probability`

```python
plot_probability(stats_output)
```

Plot Classification Probability

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` |  | a pandas file with prediction output | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fig` |  | Will return a seaborn scatterplot

#### `plot_roc`

```python
plot_roc(fpr, tpr)
```

Plot 1-Specificity by Sensitivity

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fpr` |  | false positive rate from Roc.calculate | *required*
`tpr` |  | true positive rate from Roc.calculate | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fig` |  | Will return a matplotlib ROC plot

#### `plot_scatter`

```python
plot_scatter(stats_output)
```

Plot Prediction Scatterplot

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` |  | a pandas file with prediction output | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fig` |  | Will return a seaborn scatterplot

#### `plot_silhouette`

```python
plot_silhouette(distance, labels, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Silhouette plot indicating between- vs within-label distance.

Uses the simplified silhouette definition from the original nltools
implementation: within(i) = mean distance to other points in the same
cluster; between(i) = mean distance to all points in other clusters
(not the strict Rousseeuw min-over-clusters). Score is
(between - within) / max(between, within).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | Square pairwise distance matrix (np.ndarray or polars DataFrame). | *required*
`labels` |  | Array-like of length N giving a cluster label per row/column. | *required*
`ax` |  | Matplotlib axis to plot on (optional). | <code>None</code>
`permutation_test` |  | If True, run a one-sample permutation test per cluster on positive-mean silhouette scores. | <code>True</code>
`n_permute` |  | Number of permutations. | <code>5000</code>
`**kwargs` |  | Optional. `colors` (list of RGB triplets, one per cluster) and `figsize` (tuple) control the plot appearance. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | pl.DataFrame with columns [label, mean_silhouette]. If permutation_test
 | is True, adds a `p` column (1.0 for clusters with non-positive mean).

#### `plot_stacked_adjacency`

```python
plot_stacked_adjacency(adjacency1, adjacency2, normalize = True, **kwargs)
```

Create stacked adjacency to illustrate similarity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adjacency1` |  | Adjacency instance 1. | *required*
`adjacency2` |  | Adjacency instance 2. | *required*
`normalize` |  | Normalize matrices before stacking. Default True. | <code>True</code>
`**kwargs` |  | Passed through to seaborn.heatmap. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib axes with the stacked heatmap.

#### `plot_surface`

```python
plot_surface(brain, surface = 'inflated', bg_map = 'curvature', hemi = 'both', view = 'montage', threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, darkness = None, bg_on_data = False, colorbar = False, figsize = (10, 10), n_samples = 1, radius = 0.0, interpolation = 'linear', engine = 'matplotlib', axes = None, save = None, **kwargs)
```

Plot neuroimaging data on cortical surface.

Intelligently projects volumetric NIfTI data onto cortical surfaces
and displays in customizable montage layouts. Automatically handles
hemispheric parsing and uses included MNI152 template surfaces.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain` |  | BrainData, nibabel Nifti1Image, or file path to NIfTI image. If BrainData has multiple images, plots the first one. | *required*
`surface` | <code>[str](#str)</code> | Surface mesh type. Options: 'pial', 'inflated', 'midthickness', 'white'. Defaults to 'inflated'. | <code>'inflated'</code>
`bg_map` | <code>[str](#str) or None</code> | Background map. Options: 'curvature', 'sulc', None, or file path to custom background map. Defaults to 'curvature'. | <code>'curvature'</code>
`hemi` | <code>[str](#str)</code> | Hemisphere to plot. Options: 'left', 'right', 'both'. Defaults to 'both'. | <code>'both'</code>
`view` | <code>[str](#str) or [list](#list)</code> | View type. Options: 'lateral', 'medial', 'montage', or list of views. Defaults to 'montage' (2x2 grid). | <code>'montage'</code>
`threshold` | <code>[float](#float) or [str](#str)</code> | Threshold value. Can be a float or percentile string like '95%'. Defaults to None. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. Defaults to 'RdBu_r'. | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap scaling. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap scaling. | <code>None</code>
`darkness` | <code>[float](#float) or None</code> | Background darkness (0-1). Defaults to None. | <code>None</code>
`bg_on_data` | <code>[bool](#bool)</code> | Overlay background on data. Defaults to False. | <code>False</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Defaults to False. | <code>False</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size tuple (width, height). Defaults to (10, 10). | <code>(10, 10)</code>
`n_samples` | <code>[int](#int)</code> | Number of samples for vol_to_surf projection. Defaults to 1. | <code>1</code>
`radius` | <code>[float](#float)</code> | Sampling radius for vol_to_surf projection. Defaults to 0.0. | <code>0.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation method for projection. Options: 'linear', 'nearest_most_frequent'. Defaults to 'linear'. | <code>'linear'</code>
`engine` | <code>[str](#str)</code> | Rendering engine. Options: 'matplotlib', 'plotly'. Defaults to 'matplotlib'. | <code>'matplotlib'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes) or [list](#list)</code> | Custom matplotlib axes for montage layout. If None, creates new figure. Defaults to None. | <code>None</code>
`save` | <code>[str](#str) or None</code> | File path to save plot. If None, plot is displayed but not saved. Defaults to None. | <code>None</code>
`**kwargs` |  | Additional arguments passed to plot_surf_stat_map. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure or plotly.graph_objects.Figure: Figure object
 | containing the surface plot(s).

**Examples:**

Plot BrainData with default 2x2 montage:

```pycon
>>> from nltools.plotting import plot_surface
>>> from nltools.data import BrainData
>>> brain = BrainData('data.nii.gz')
>>> fig = plot_surface(brain)
```

Single hemisphere, lateral view:

```pycon
>>> fig = plot_surface(brain, hemi='left', view='lateral')
```

Custom colormap and threshold:

```pycon
>>> fig = plot_surface(brain, cmap='hot', threshold=0.5)
```

Percentile threshold with custom background:

```pycon
>>> fig = plot_surface(brain, threshold='95%', bg_map='sulc')
```



### Modules

#### `adjacency`

Adjacency matrix visualization — stacked plots, distance, and silhouette.

**Methods:**

Name | Description
---- | -----------
[`plot_between_label_distance`](#plot_between_label_distance) | Heatmap of average pairwise distance between every label pair.
[`plot_mean_label_distance`](#plot_mean_label_distance) | Violin plot of within- vs between-label distances.
[`plot_silhouette`](#plot_silhouette) | Silhouette plot indicating between- vs within-label distance.
[`plot_stacked_adjacency`](#plot_stacked_adjacency) | Create stacked adjacency to illustrate similarity.



##### Methods

###### `plot_between_label_distance`

```python
plot_between_label_distance(distance, labels, ax = None, permutation_test = True, n_permute = 5000, fontsize = 18, **kwargs)
```

Heatmap of average pairwise distance between every label pair.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | Square pairwise distance matrix (np.ndarray or polars DataFrame). | *required*
`labels` |  | Array-like of length N giving a group label for each row/column. | *required*
`ax` |  | Matplotlib axis to plot on (optional). | <code>None</code>
`permutation_test` |  | If True, also compute mean-difference and p-value matrices. | <code>True</code>
`n_permute` |  | Number of permutations. | <code>5000</code>
`fontsize` |  | Reserved for future use; currently unused. | <code>18</code>
`**kwargs` |  | Passed to seaborn.heatmap. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Without permutation_test: (long_df, within_mean_df)
 | With permutation_test: (long_df, within_mean_df, mean_diff_df, p_df)
 | All frames are polars DataFrames. `long_df` has columns
 | [Distance, Group, Comparison]. The three square-matrix-like frames
 | are long format with columns [label1, label2, <value>] so they can
 | be pivoted to a matrix if needed.

###### `plot_mean_label_distance`

```python
plot_mean_label_distance(distance, labels, ax = None, permutation_test = False, n_permute = 5000, fontsize = 18, **kwargs)
```

Violin plot of within- vs between-label distances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | Square pairwise distance matrix (np.ndarray or polars DataFrame). | *required*
`labels` |  | Array-like of length N giving a group label for each row/column. | *required*
`ax` |  | Matplotlib axis to plot on (optional). | <code>None</code>
`permutation_test` |  | If True, run a two-sample permutation test per group. | <code>False</code>
`n_permute` |  | Number of permutations. | <code>5000</code>
`fontsize` |  | Font size for plot labels. | <code>18</code>
`**kwargs` |  | Passed to seaborn.violinplot. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | pl.DataFrame with columns [Distance, Group, Type] in long format.
 | If permutation_test=True, returns (pl.DataFrame, dict of per-group stats).

###### `plot_silhouette`

```python
plot_silhouette(distance, labels, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Silhouette plot indicating between- vs within-label distance.

Uses the simplified silhouette definition from the original nltools
implementation: within(i) = mean distance to other points in the same
cluster; between(i) = mean distance to all points in other clusters
(not the strict Rousseeuw min-over-clusters). Score is
(between - within) / max(between, within).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | Square pairwise distance matrix (np.ndarray or polars DataFrame). | *required*
`labels` |  | Array-like of length N giving a cluster label per row/column. | *required*
`ax` |  | Matplotlib axis to plot on (optional). | <code>None</code>
`permutation_test` |  | If True, run a one-sample permutation test per cluster on positive-mean silhouette scores. | <code>True</code>
`n_permute` |  | Number of permutations. | <code>5000</code>
`**kwargs` |  | Optional. `colors` (list of RGB triplets, one per cluster) and `figsize` (tuple) control the plot appearance. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | pl.DataFrame with columns [label, mean_silhouette]. If permutation_test
 | is True, adds a `p` column (1.0 for clusters with non-positive mean).

###### `plot_stacked_adjacency`

```python
plot_stacked_adjacency(adjacency1, adjacency2, normalize = True, **kwargs)
```

Create stacked adjacency to illustrate similarity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adjacency1` |  | Adjacency instance 1. | *required*
`adjacency2` |  | Adjacency instance 2. | *required*
`normalize` |  | Normalize matrices before stacking. Default True. | <code>True</code>
`**kwargs` |  | Passed through to seaborn.heatmap. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib axes with the stacked heatmap.

#### `brain`

Brain visualization — surface plots, flatmaps, and interactive viewers.

**Methods:**

Name | Description
---- | -----------
[`plot_flatmap`](#plot_flatmap) | Plot brain data on cortical flatmap.
[`plot_interactive_brain`](#plot_interactive_brain) | This function leverages nilearn's new javascript based brain viewer functions to create interactive plotting functionality.
[`plot_surface`](#plot_surface) | Plot neuroimaging data on cortical surface.

##### Methods

###### `plot_flatmap`

```python
plot_flatmap(brain, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
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
`brain` |  | BrainData, nibabel Nifti1Image, or file path to NIfTI image. Data must be in MNI152 space. | *required*
`threshold` | <code>[float](#float) or [str](#str)</code> | Values below this absolute threshold are masked. Can be a float or percentile string like '95%'. Defaults to None (no threshold). | <code>None</code>
`cmap` | <code>[str](#str)</code> | Matplotlib colormap for data. Defaults to 'RdBu_r' (diverging red-blue). | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap. If None, uses symmetric max of absolute values. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap. If None and vmax is set, uses -vmax for diverging maps. | <code>None</code>
`template` | <code>[str](#str)</code> | fsaverage resolution. Options: 'fsaverage3' (642 vertices), 'fsaverage4' (2562), 'fsaverage5' (10242, default), 'fsaverage6' (40962), 'fsaverage' (163842, full resolution). | <code>'fsaverage5'</code>
`with_curvature` | <code>[bool](#bool)</code> | Show sulcal/gyral pattern as grayscale background. Defaults to True. | <code>True</code>
`curvature_contrast` | <code>[float](#float)</code> | Contrast of curvature (0=flat gray, 1=full contrast). Defaults to 0.5. | <code>0.5</code>
`curvature_brightness` | <code>[float](#float)</code> | Mean brightness of curvature (0=dark, 1=bright). Defaults to 0.5. | <code>0.5</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Defaults to True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Defaults to 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size (width, height). Defaults to (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. Defaults to None. | <code>None</code>
`radius` | <code>[float](#float)</code> | Sampling radius in mm for vol_to_surf projection. Larger values provide smoother projections. Defaults to 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation for vol_to_surf. Options: 'linear', 'nearest_most_frequent'. Defaults to 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. If None, creates new figure. Defaults to None. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure: The figure containing the flatmap.

**Examples:**

Basic flatmap with default settings:

```pycon
>>> from nltools.plotting import plot_flatmap
>>> from nltools.data import BrainData
>>> brain = BrainData('stats.nii.gz')
>>> fig = plot_flatmap(brain)
```

Thresholded with custom colormap:

```pycon
>>> fig = plot_flatmap(brain, threshold=2.5, cmap='hot')
```

Percentile threshold, no curvature:

```pycon
>>> fig = plot_flatmap(brain, threshold='95%', with_curvature=False)
```

High resolution for publication:

```pycon
>>> fig = plot_flatmap(brain, template='fsaverage6', figsize=(16, 8))
>>> fig.savefig('flatmap.pdf', dpi=300)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Data is projected from MNI152 space to fsaverage surface space.
  Small alignment differences are expected at boundaries.
- Higher resolution templates (fsaverage6, fsaverage) produce
  sharper images but take longer to render.
- The flat surfaces are cached by nilearn after first download
  (~50MB for fsaverage5).

</details>

###### `plot_interactive_brain`

```python
plot_interactive_brain(brain, threshold = 1e-06, surface = False, percentile_threshold = False, anatomical = None, **kwargs)
```

This function leverages nilearn's new javascript based brain viewer functions to create interactive plotting functionality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain` | <code>[BrainData](#nltools.BrainData)</code> | a BrainData instance of 1d or 2d shape (i.e. 3d or 4d volume) | *required*
`threshold` | <code>[float](#float) / [str](#str)</code> | threshold to initialize the visualization, maybe be a percentile string; default 0 | <code>1e-06</code>
`surface` | <code>[bool](#bool)</code> | whether to create a surface-based plot; default False | <code>False</code>
`percentile_threshold` | <code>[bool](#bool)</code> | whether to interpret threshold values as percentiles | <code>False</code>
`kwargs` |  | optional arguments to nilearn.view_img or nilearn.view_img_on_surf | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | interactive brain viewer widget

###### `plot_surface`

```python
plot_surface(brain, surface = 'inflated', bg_map = 'curvature', hemi = 'both', view = 'montage', threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, darkness = None, bg_on_data = False, colorbar = False, figsize = (10, 10), n_samples = 1, radius = 0.0, interpolation = 'linear', engine = 'matplotlib', axes = None, save = None, **kwargs)
```

Plot neuroimaging data on cortical surface.

Intelligently projects volumetric NIfTI data onto cortical surfaces
and displays in customizable montage layouts. Automatically handles
hemispheric parsing and uses included MNI152 template surfaces.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain` |  | BrainData, nibabel Nifti1Image, or file path to NIfTI image. If BrainData has multiple images, plots the first one. | *required*
`surface` | <code>[str](#str)</code> | Surface mesh type. Options: 'pial', 'inflated', 'midthickness', 'white'. Defaults to 'inflated'. | <code>'inflated'</code>
`bg_map` | <code>[str](#str) or None</code> | Background map. Options: 'curvature', 'sulc', None, or file path to custom background map. Defaults to 'curvature'. | <code>'curvature'</code>
`hemi` | <code>[str](#str)</code> | Hemisphere to plot. Options: 'left', 'right', 'both'. Defaults to 'both'. | <code>'both'</code>
`view` | <code>[str](#str) or [list](#list)</code> | View type. Options: 'lateral', 'medial', 'montage', or list of views. Defaults to 'montage' (2x2 grid). | <code>'montage'</code>
`threshold` | <code>[float](#float) or [str](#str)</code> | Threshold value. Can be a float or percentile string like '95%'. Defaults to None. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. Defaults to 'RdBu_r'. | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap scaling. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap scaling. | <code>None</code>
`darkness` | <code>[float](#float) or None</code> | Background darkness (0-1). Defaults to None. | <code>None</code>
`bg_on_data` | <code>[bool](#bool)</code> | Overlay background on data. Defaults to False. | <code>False</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Defaults to False. | <code>False</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size tuple (width, height). Defaults to (10, 10). | <code>(10, 10)</code>
`n_samples` | <code>[int](#int)</code> | Number of samples for vol_to_surf projection. Defaults to 1. | <code>1</code>
`radius` | <code>[float](#float)</code> | Sampling radius for vol_to_surf projection. Defaults to 0.0. | <code>0.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation method for projection. Options: 'linear', 'nearest_most_frequent'. Defaults to 'linear'. | <code>'linear'</code>
`engine` | <code>[str](#str)</code> | Rendering engine. Options: 'matplotlib', 'plotly'. Defaults to 'matplotlib'. | <code>'matplotlib'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes) or [list](#list)</code> | Custom matplotlib axes for montage layout. If None, creates new figure. Defaults to None. | <code>None</code>
`save` | <code>[str](#str) or None</code> | File path to save plot. If None, plot is displayed but not saved. Defaults to None. | <code>None</code>
`**kwargs` |  | Additional arguments passed to plot_surf_stat_map. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure or plotly.graph_objects.Figure: Figure object
 | containing the surface plot(s).

**Examples:**

Plot BrainData with default 2x2 montage:

```pycon
>>> from nltools.plotting import plot_surface
>>> from nltools.data import BrainData
>>> brain = BrainData('data.nii.gz')
>>> fig = plot_surface(brain)
```

Single hemisphere, lateral view:

```pycon
>>> fig = plot_surface(brain, hemi='left', view='lateral')
```

Custom colormap and threshold:

```pycon
>>> fig = plot_surface(brain, cmap='hot', threshold=0.5)
```

Percentile threshold with custom background:

```pycon
>>> fig = plot_surface(brain, threshold='95%', bg_map='sulc')
```

#### `decomposition`

ICA/PCA component viewer — interactive decomposition explorer.

**Methods:**

Name | Description
---- | -----------
[`component_viewer`](#component_viewer) | This a function to interactively view the results of a decomposition analysis

##### Methods

###### `component_viewer`

```python
component_viewer(output, tr = 2.0)
```

This a function to interactively view the results of a decomposition analysis

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output` |  | (dict) output dictionary from running Brain_data.decompose() | *required*
`tr` |  | (float) repetition time of data | <code>2.0</code>

#### `prediction`

Model output visualization — ROC, SVM margin, regression, and logistic plots.

**Methods:**

Name | Description
---- | -----------
[`plot_dist_from_hyperplane`](#plot_dist_from_hyperplane) | Plot SVM Classification Distance from Hyperplane
[`plot_probability`](#plot_probability) | Plot Classification Probability
[`plot_roc`](#plot_roc) | Plot 1-Specificity by Sensitivity
[`plot_scatter`](#plot_scatter) | Plot Prediction Scatterplot



##### Methods

###### `plot_dist_from_hyperplane`

```python
plot_dist_from_hyperplane(stats_output)
```

Plot SVM Classification Distance from Hyperplane

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` |  | a pandas file with prediction output | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fig` |  | Will return a seaborn plot of distance from hyperplane

###### `plot_probability`

```python
plot_probability(stats_output)
```

Plot Classification Probability

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` |  | a pandas file with prediction output | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fig` |  | Will return a seaborn scatterplot

###### `plot_roc`

```python
plot_roc(fpr, tpr)
```

Plot 1-Specificity by Sensitivity

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fpr` |  | false positive rate from Roc.calculate | *required*
`tpr` |  | true positive rate from Roc.calculate | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fig` |  | Will return a matplotlib ROC plot

###### `plot_scatter`

```python
plot_scatter(stats_output)
```

Plot Prediction Scatterplot

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stats_output` |  | a pandas file with prediction output | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fig` |  | Will return a seaborn scatterplot

