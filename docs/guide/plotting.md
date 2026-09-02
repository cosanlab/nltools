---
title: Plotting
---

Each data class knows how to draw itself. [`BrainData.plot`](../api/data/brain_data.md#data-brain-data-plot)
does most of the work: `method='glass'` for a whole-brain projection, `'slices'` for orthogonal cuts,
`'timeseries'` or `'histogram'` for the data as numbers rather than anatomy. Surfaces are
[`plot_surf`](../api/data/brain_data.md#data-brain-data-plot-surf) and
[`plot_flatmap`](../api/data/brain_data.md#data-brain-data-plot-flatmap); the interactive WebGL
viewer is [`iplot`](../api/data/brain_data.md#data-brain-data-iplot). Everything returns a
matplotlib figure or axes, takes `save=` for a path, and forwards unrecognized keyword arguments to
the underlying nilearn or seaborn call.

Two thresholds share the page and are easy to confuse. `threshold=` is a *transparency*
cutoff passed to nilearn: voxels with `abs(value) < threshold` are drawn as background.
`upper=` and `lower=` actually censor the data before plotting, one-sided, and accept percentile
strings like `'95%'`.

Object | Plot | Notes
--- | --- | ---
`BrainData` volume | [`plot`](../api/data/brain_data.md#data-brain-data-plot)`(method='glass'\|'slices')` | `view='xyz'` picks slice axes; `cut_coords=`, `bg_img=`, `cmap=` as usual
`BrainData` numbers | `plot(method='timeseries'\|'histogram')` | `stat='mean'\|'median'\|'std'` for `'timeseries'`
`BrainData` surface | [`plot_surf`](../api/data/brain_data.md#data-brain-data-plot-surf) | `hemi=`, `view=` (`'montage'` = lateral + medial), `surface='pial'\|'inflated'`
`BrainData` flatmap | [`plot_flatmap`](../api/data/brain_data.md#data-brain-data-plot-flatmap) | Curvature underlay on by default
`BrainData` interactive | [`iplot`](../api/data/brain_data.md#data-brain-data-iplot) / [`plot_interactive_brain`](../api/tasks/plotting.md#tasks-plotting-plot-interactive-brain) | Needs a live kernel (Jupyter, marimo); static pages show a placeholder
`DesignMatrix` | [`plot`](../api/data/design_matrix.md#data-design-matrix-plot)`(method='matrix'\|'timeseries'\|'corr')` | `'matrix'` is the SPM-style heatmap; `'corr'` shows regressor collinearity
`Adjacency` matrix | [`plot`](../api/data/adjacency.md#data-adjacency-plot) | `limit=` caps how many matrices from a stack are drawn
`Adjacency` structure | [`plot_mds`](../api/data/adjacency.md#data-adjacency-plot-mds), [`plot_silhouette`](../api/data/adjacency.md#data-adjacency-plot-silhouette), [`plot_label_distance`](../api/data/adjacency.md#data-adjacency-plot-label-distance) | All take `labels=`, one per node
Two matrices at once | [`plot_stacked_adjacency`](../api/tasks/similarity.md#tasks-similarity-plot-stacked-adjacency) | See [Similarity & RSA](similarity-and-rsa.md)
`Predict` result | [`plot_roc`](../api/tasks/prediction.md#tasks-prediction-plot-roc), [`plot_scatter`](../api/tasks/prediction.md#tasks-prediction-plot-scatter), [`plot_dist_from_hyperplane`](../api/tasks/prediction.md#tasks-prediction-plot-dist-from-hyperplane), [`plot_probability`](../api/tasks/prediction.md#tasks-prediction-plot-probability) | Or `Roc.plot()` / `Roc.summary()`
`decompose` output | [`component_viewer`](../api/tasks/plotting.md#tasks-plotting-component-viewer) | ipywidgets; live kernel only

## Volumes

```python
stat_map.plot(method="glass", threshold=2.0, title="group z")
stat_map.plot(method="slices", view="xz", threshold=2.0, cmap="RdBu_r")
stat_map.plot(method="slices", upper=2.0, lower=-2.0, save="zmap.png")
```

`save=` writes the figure and still returns it. For a background other than the bundled MNI T1, pass
`bg_img=` a path or image. That matters for un-normalized single-subject data, where the template
would be misleading.

```python
pain[:20].plot(method="timeseries", stat="mean")
pain[:20].plot(method="histogram")
pain[:3].plot(method="glass", limit=3)      # limit caps glass/slices renders

stat_map.plot_surf(hemi="left", view="lateral", threshold=2.0)
```

## Designs and matrices

```python
dm.plot(method="matrix")
dm.plot(method="corr", metric="pearson")

rdm.plot()
rdm.plot_mds(labels=labels, n_jobs=1)
rdm.plot_silhouette(labels=labels, n_permute=1000)
rdm.plot_label_distance(labels=labels)
```

`plot_silhouette` and `plot_label_distance` run a permutation test as they draw
(`permutation_test=True` by default), so they cost more than a plain heatmap. Turn it off while
iterating on a figure.

## Gotchas

- `iplot` and `component_viewer` render through a live kernel. In a statically built page they
  degrade to a placeholder, so use `plot` for anything that has to survive a docs build.
- `plot_surf` and `plot_flatmap` interpolate volume data onto an fsaverage mesh with a
  `radius_mm=3.0` sampling ball. They are visualizations of volume data, not surface analyses.
- `Adjacency.plot_mds` returns nothing; it draws into the current or supplied axes.
- Set a non-interactive matplotlib backend (`matplotlib.use("Agg")`) before importing in a script,
  or figures will try to open windows.

Next: [Atlases & cluster reports](atlases.md) to put names on what you just drew.
