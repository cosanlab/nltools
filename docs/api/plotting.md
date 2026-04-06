## `nltools.plotting`

NeuroLearn Plotting Tools
=========================

Numerous functions to plot data

**Functions:**

Name | Description
---- | -----------
[`dist_from_hyperplane_plot`](#nltools.plotting.dist_from_hyperplane_plot) | Plot SVM Classification Distance from Hyperplane
[`plot_between_label_distance`](#nltools.plotting.plot_between_label_distance) | Create a heatmap indicating average between label distance
[`plot_brain`](#nltools.plotting.plot_brain) | More complete brain plotting of a BrainData instance
[`plot_flatmap`](#nltools.plotting.plot_flatmap) | Plot brain data on cortical flatmap.
[`plot_interactive_brain`](#nltools.plotting.plot_interactive_brain) | This function leverages nilearn's new javascript based brain viewer functions to create interactive plotting functionality.
[`plot_mean_label_distance`](#nltools.plotting.plot_mean_label_distance) | Create a violin plot indicating within and between label distance.
[`plot_silhouette`](#nltools.plotting.plot_silhouette) | Create a silhouette plot indicating between relative to within label distance
[`plot_stacked_adjacency`](#nltools.plotting.plot_stacked_adjacency) | Create stacked adjacency to illustrate similarity.
[`plot_t_brain`](#nltools.plotting.plot_t_brain) | Takes a brain data object and computes a 1 sample t-test across it's first axis. If a list is provided will compute difference between brain data objects in list (i.e. paired samples t-test).
[`probability_plot`](#nltools.plotting.probability_plot) | Plot Classification Probability
[`roc_plot`](#nltools.plotting.roc_plot) | Plot 1-Specificity by Sensitivity
[`scatterplot`](#nltools.plotting.scatterplot) | Plot Prediction Scatterplot
[`surface_plot`](#nltools.plotting.surface_plot) | Plot neuroimaging data on cortical surface.



### Attributes

### Functions#### `nltools.plotting.dist_from_hyperplane_plot`

```python
dist_from_hyperplane_plot(stats_output)
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

#### `nltools.plotting.plot_between_label_distance`

```python
plot_between_label_distance(distance, labels, ax = None, permutation_test = True, n_permute = 5000, fontsize = 18, **kwargs)
```

Create a heatmap indicating average between label distance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | (pandas dataframe) brain_distance matrix | *required*
`labels` |  | (pandas dataframe) group labels | *required*
`ax` |  | axis to plot (default=None) | <code>None</code>
`permutation_test` |  | (boolean) | <code>True</code>
`n_permute` |  | (int) number of samples for permuation test | <code>5000</code>
`fontsize` |  | (int) size of font for plot | <code>18</code>

Returns:
    f: heatmap
    out: pandas dataframe of pairwise distance between conditions
    within_dist_out: average pairwise distance matrix
    mn_dist_out: (optional if permutation_test=True) average difference in distance between conditions
    p_dist_out: (optional if permutation_test=True) p-value for difference in distance between conditions

#### `nltools.plotting.plot_brain`

```python
plot_brain(objIn, how = 'full', thr_upper = None, thr_lower = None, save = False, verbose = False, **kwargs)
```

More complete brain plotting of a BrainData instance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`obj` | <code>[BrainData](#nltools.data.BrainData)</code> | object to plot | *required*
`how` | <code>[str](#str)</code> | whether to plot a glass brain 'glass', 3 view-multi-slice mni 'mni', or both 'full' | <code>'full'</code>
`thr_upper` | <code>[str](#str) / [float](#float)</code> | thresholding of image. Can be string for percentage, or float for data units (see BrainData.threshold() | <code>None</code>
`thr_lower` | <code>[str](#str) / [float](#float)</code> | thresholding of image. Can be string for percentage, or float for data units (see BrainData.threshold() | <code>None</code>
`save` | <code>[str](#str)</code> | if a string file name or path is provided plots will be saved into this directory appended with the orientation they belong to | <code>False</code>
`kwargs` |  | optionals args to nilearn plot functions (e.g. vmax) | <code>{}</code>

#### `nltools.plotting.plot_flatmap`

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
`interpolation` | <code>[str](#str)</code> | Interpolation for vol_to_surf. Options: 'linear', 'nearest'. Defaults to 'linear'. | <code>'linear'</code>
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

#### `nltools.plotting.plot_interactive_brain`

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

#### `nltools.plotting.plot_mean_label_distance`

```python
plot_mean_label_distance(distance, labels, ax = None, permutation_test = False, n_permute = 5000, fontsize = 18, **kwargs)
```

Create a violin plot indicating within and between label distance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | pandas dataframe of distance | *required*
`labels` |  | labels indicating columns and rows to group | *required*
`ax` |  | matplotlib axis to plot on | <code>None</code>
`permutation_test` |  | (bool) indicates whether to run permuatation test or not | <code>False</code>
`n_permute` |  | (int) number of permutations to run | <code>5000</code>
`fontsize` |  | (int) fontsize for plot labels | <code>18</code>

Returns:
    f: heatmap
    stats: (optional if permutation_test=True) permutation results

#### `nltools.plotting.plot_silhouette`

```python
plot_silhouette(distance, labels, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Create a silhouette plot indicating between relative to within label distance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` |  | (pandas dataframe) brain_distance matrix | *required*
`labels` |  | (pandas dataframe) group labels | *required*
`ax` |  | axis to plot (default=None) | <code>None</code>
`permutation_test` |  | (boolean) | <code>True</code>
`n_permute` |  | (int) number of samples for permuation test | <code>5000</code>

<details class="optional-keyword-args" open markdown="1">
<summary>Optional keyword args</summary>

figsize: (list) dimensions of silhouette plot
colors: (list) color triplets for silhouettes. Length must equal number of unique labels

</details>

**Returns:**

Type | Description
---- | -----------
 | # f: heatmap
 | # out: pandas dataframe of pairwise distance between conditions
 | # within_dist_out: average pairwise distance matrix
 | # mn_dist_out: (optional if permutation_test=True) average difference in distance between conditions
 | # p_dist_out: (optional if permutation_test=True) p-value for difference in distance between conditions

#### `nltools.plotting.plot_stacked_adjacency`

```python
plot_stacked_adjacency(adjacency1, adjacency2, normalize = True, **kwargs)
```

Create stacked adjacency to illustrate similarity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`matrix1` |  | Adjacency instance 1 | *required*
`matrix2` |  | Adjacency instance 2 | *required*
`normalize` |  | (boolean) Normalize matrices. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib figure

#### `nltools.plotting.plot_t_brain`

```python
plot_t_brain(objIn, how = 'full', thr = 'unc', alpha = None, nperm = None, cut_coords = [], **kwargs)
```

Takes a brain data object and computes a 1 sample t-test across it's first axis. If a list is provided will compute difference between brain data objects in list (i.e. paired samples t-test).
Args:
    objIn (list/BrainData): if list will compute difference map first
    how (list): whether to plot a glass brain 'glass', 3 view-multi-slice mni 'mni', or both 'full'
    thr (str): what method to use for multiple comparisons correction unc, fdr, or tfce
    alpha (float): p-value threshold
    nperm (int): number of permutations for tcfe; default 1000
    cut_coords (list): x,y,z coords to plot brain slice
    kwargs: optionals args to nilearn plot functions (e.g. vmax)

#### `nltools.plotting.probability_plot`

```python
probability_plot(stats_output)
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

#### `nltools.plotting.roc_plot`

```python
roc_plot(fpr, tpr)
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

#### `nltools.plotting.scatterplot`

```python
scatterplot(stats_output)
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

#### `nltools.plotting.surface_plot`

```python
surface_plot(brain, surface = 'inflated', bg_map = 'curvature', hemi = 'both', view = 'montage', threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, darkness = None, bg_on_data = False, colorbar = False, figsize = (10, 10), n_samples = 1, radius = 0.0, interpolation = 'linear', engine = 'matplotlib', axes = None, save = None, **kwargs)
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
`view` | <code>[str](#str) or [list](#list)</code> | View type. Options: 'lateral', 'medial', 'montage', or list of views. Defaults to 'montage' (2×2 grid). | <code>'montage'</code>
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

Plot BrainData with default 2×2 montage:

```pycon
>>> from nltools.plotting import surface_plot
>>> from nltools.data import BrainData
>>> brain = BrainData('data.nii.gz')
>>> fig = surface_plot(brain)
```

Single hemisphere, lateral view:

```pycon
>>> fig = surface_plot(brain, hemi='left', view='lateral')
```

Custom colormap and threshold:

```pycon
>>> fig = surface_plot(brain, cmap='hot', threshold=0.5)
```

Percentile threshold with custom background:

```pycon
>>> fig = surface_plot(brain, threshold='95%', bg_map='sulc')
```

