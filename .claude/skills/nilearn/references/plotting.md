# nilearn.plotting — Visualization

Plotting utilities for brain data: 2D/3D matplotlib views (orthogonal slices, glass brain, mosaic/tiled), surface meshes, connectomes, GLM design matrices, and interactive HTML viewers built on plotly/three.js. Most volume plots return display objects (slicers/projectors) that support overlays, contours, markers, edges, and `savefig`.

**Source:** https://nilearn.github.io/dev/modules/plotting.html

## When to use
- Visualize statistical maps, ROIs, anatomy, or atlases on slices or glass brain
- Project volume stat maps onto surface meshes (fsaverage, fslr)
- Plot connectomes (graph) on a brain schematic, or a connectivity matrix
- Inspect GLM design matrices, contrasts, and event timing
- Produce interactive shareable HTML viewers for notebooks / web pages
- Find good cut coordinates automatically

## Picker: which plotting function?

| Want to plot | Use |
|---|---|
| Statistical map on slices | `plot_stat_map` |
| Stat map on glass brain | `plot_glass_brain` |
| Anatomy alone | `plot_anat` |
| EPI volume | `plot_epi` |
| Arbitrary 3D image | `plot_img` |
| ROI / parcellation overlay | `plot_roi` |
| Probabilistic atlas (4D) | `plot_prob_atlas` |
| Connectome (graph on brain) | `plot_connectome` |
| Network nodes (markers) on glass brain | `plot_markers` |
| Connectivity / correlation matrix | `plot_matrix` |
| Carpet plot of fMRI | `plot_carpet` |
| Stat map on inflated surface (volume input) | `plot_img_on_surf` |
| Stat map on surface (already-projected data) | `plot_surf_stat_map` |
| Surface ROI / labels | `plot_surf_roi` |
| ROI contours on surface | `plot_surf_contours` |
| Generic surface with map | `plot_surf` |
| Design matrix | `plot_design_matrix` |
| Design matrix correlation | `plot_design_matrix_correlation` |
| Contrast vector for a design | `plot_contrast_matrix` |
| Event timing diagram | `plot_event` |
| Compare two images (Bland-Altman) | `plot_bland_altman` |
| Compare image lists (correlation) | `plot_img_comparison` |
| Interactive HTML viewer | `view_img`, `view_img_on_surf`, `view_connectome`, `view_markers`, `view_surf` |

## Volume plots

### plot_stat_map
```python
from nilearn import plotting

display = plotting.plot_stat_map(
    stat_map_img, bg_img=MNI152,
    threshold=3.0,
    display_mode='ortho',  # 'ortho'|'tiled'|'mosaic'|'x'|'y'|'z'|'yx'|'xz'|'yz'
    cut_coords=None,       # None=auto, int=n_cuts, list=explicit coords
    cmap='RdBu_r',
    symmetric_cbar='auto',
    colorbar=True,
    title='Activation',
    dim='auto',            # background dimming -2 to 2
    annotate=True,
    draw_cross=True,
    output_file=None,      # if set, saves and returns None
)
```

### plot_glass_brain
```python
plotting.plot_glass_brain(
    stat_map_img,
    threshold='auto',
    plot_abs=True,         # if False, signed values + diverging cmap
    display_mode='ortho',  # 'ortho'|'l'|'r'|'lr'|'lzr'|'lyr'|'lzry'|'lyrz'
    cmap=None,
    colorbar=False,
    symmetric_cbar='auto',
    alpha=0.7,
    title=None,
)
```

### plot_anat
```python
plotting.plot_anat(
    anat_img=None,         # None -> MNI152 template
    display_mode='ortho',
    cut_coords=None,
    cmap='gray',
    dim=-0.5,
    annotate=True,
)
```

### plot_roi
```python
plotting.plot_roi(
    roi_img, bg_img=MNI152,
    display_mode='ortho',
    cmap='gist_ncar',
    alpha=0.7,
    view_type='continuous',          # 'continuous' | 'contours'
    resampling_interpolation='nearest',  # nearest for label images
    linewidths=2.5,                  # for contours mode
)
```

### plot_img
```python
plotting.plot_img(img, cut_coords=None, display_mode='ortho',
                  cmap=None, colorbar=False, threshold=None)
```
Generic 3D image plotter — base for `plot_stat_map`, `plot_anat`, `plot_epi`.

### plot_epi
```python
plotting.plot_epi(epi_img=None, display_mode='ortho',
                  cmap='gray', cut_coords=None)
```
Plot cuts of an EPI image.

### plot_prob_atlas
```python
plotting.plot_prob_atlas(maps_img, bg_img=MNI152,
                         view_type='auto',     # 'auto'|'contours'|'filled_contours'|'continuous'
                         threshold='auto',
                         linewidths=2.5,
                         cmap='gist_rainbow')
```
Plot a probabilistic (4D) atlas overlaid on anatomy.

### plot_carpet
```python
plotting.plot_carpet(img, mask_img=None,
                     mask_labels=None,         # dict {label_name: int} for grouped rows
                     t_r=None,
                     standardize='zscore_sample',
                     detrend=False,
                     title=None)
```
Voxel intensity x time (a.k.a. "Power plot").

### Connectome volume plots

#### plot_connectome
```python
plotting.plot_connectome(
    adjacency_matrix, node_coords,
    node_color='auto', node_size=50,
    edge_cmap='RdBu_r',
    edge_threshold='90%',  # str percentile or numeric
    edge_vmin=None, edge_vmax=None,
    display_mode='ortho',  # same options as glass brain
    colorbar=False,
    alpha=0.7,
    node_kwargs=None, edge_kwargs=None,
)
```

#### plot_markers
```python
plotting.plot_markers(
    node_values, node_coords,
    node_size='auto',
    node_cmap='RdBu_r',
    node_threshold=None,
    display_mode='ortho',
    annotate=True,
    colorbar=True,
)
```
Plot scalar-valued markers (e.g. node strength) at MNI coordinates on glass brain.

### Matrix plots

#### plot_matrix
```python
plotting.plot_matrix(
    mat,
    title=None,
    labels=None,
    figure=None, axes=None,
    colorbar=True,
    cmap='RdBu_r',
    tri='full',            # 'full' | 'lower' | 'diag'
    auto_fit=True,
    grid=False,
    reorder=False,         # False | True | 'single' | 'complete' | 'average'
    vmin=None, vmax=None,
)
```

## Surface plots

### Volume → surface in one call

#### plot_img_on_surf
```python
plotting.plot_img_on_surf(
    stat_map,
    surf_mesh='fsaverage5',   # or PolyMesh / fetched mesh
    views=['lateral', 'medial'],
    hemispheres=['left', 'right'],
    inflate=True,
    threshold=3.0,
    cmap='RdBu_r',
    symmetric_cbar='auto',
    colorbar=True,
    bg_on_data=True,
    title=None,
    output_file=None,
)
```
Internally projects volume → surface, then renders multiple `plot_surf_stat_map` panels.

### Direct surface plotting (already-projected data)

#### plot_surf
```python
plotting.plot_surf(
    surf_mesh, surf_map=None, bg_map=None,
    hemi='left',                 # 'left'|'right'|'both'
    view='lateral',              # 'lateral'|'medial'|'dorsal'|'ventral'|'anterior'|'posterior'|tuple
    engine='matplotlib',         # 'matplotlib' | 'plotly'
    cmap='cold_hot',
    avg_method='mean',
    threshold=None,
    alpha='auto',
    bg_on_data=False,
    darkness=0.7,
)
```

#### plot_surf_stat_map
```python
plotting.plot_surf_stat_map(
    surf_mesh, stat_map=None, bg_map=None,
    hemi='left', view='lateral',
    engine='matplotlib',         # 'plotly' enables interactive figure
    cmap='RdBu_r',
    threshold=1.0,
    symmetric_cbar='auto',
    colorbar=True,
)
```

#### plot_surf_roi
```python
plotting.plot_surf_roi(
    surf_mesh, roi_map=None, bg_map=None,
    hemi='left', view='lateral',
    cmap='gist_ncar',
    avg_method='median',
    threshold=1e-14,
    alpha='auto',
)
```

#### plot_surf_contours
```python
plotting.plot_surf_contours(
    surf_mesh=None, roi_map=None,
    hemi='left', view='lateral',
    levels=None,                 # which label values to draw
    labels=None, colors=None,
    legend=False,
    figure=None, axes=None,      # overlay onto an existing surface fig
)
```

## GLM plots

### plot_design_matrix
```python
plotting.plot_design_matrix(
    design_matrix,
    rescale=True,
    ax=None,
    output_file=None,
)
```

### plot_design_matrix_correlation
```python
plotting.plot_design_matrix_correlation(
    design_matrix,
    tri='full',
    cmap='RdBu_r',
    output_file=None,
)
```
Correlation matrix between design-matrix regressors.

### plot_contrast_matrix
```python
plotting.plot_contrast_matrix(
    contrast_def,                # str expression or array
    design_matrix,
    colorbar=True,
    ax=None,
    output_file=None,
)
```

### plot_event
```python
plotting.plot_event(
    model_event,                 # events DataFrame or list of DataFrames
    cmap=None,
    output_file=None,
)
```
Stripe plot of trial onsets/durations per condition.

## Image comparison

| Function | Renders |
|---|---|
| `plot_bland_altman(ref_img, src_img)` | Bland-Altman agreement plot for two 3D images |
| `plot_img_comparison(ref_imgs, src_imgs)` | Scatter + correlation plots between two image lists |

## Interactive HTML viewers (Jupyter / save to .html)

| Function | Renders |
|---|---|
| `view_img` | 3D volume statistical map with slicers (orthoview) |
| `view_img_on_surf` | Volume stat map projected to inflated surface (plotly) |
| `view_surf` | Surface map directly on a mesh |
| `view_connectome` | 3D connectome graph (nodes + edges) |
| `view_markers` | 3D marker locations on a brain |

```python
view = plotting.view_img(stat_map, threshold=3.0,
                         bg_img=MNI152, cmap='RdBu_r',
                         symmetric_cmap=True,
                         vmax=None, opacity=1.0,
                         width_view=600, black_bg=False)
view.save_as_html('viewer.html')   # or display directly in Jupyter

plotting.view_img_on_surf(stat_map, surf_mesh='fsaverage5',
                          threshold='90%', vmax=None, cmap='RdBu_r')

plotting.view_surf(surf_mesh, surf_map=data, bg_map=sulcal,
                   threshold=None, cmap='cold_hot')

plotting.view_connectome(adj_matrix, node_coords,
                         edge_threshold='90%', edge_cmap='bwr',
                         linewidth=6.0, node_size=3.0,
                         symmetric_cmap=True)

plotting.view_markers(marker_coords, marker_color='red',
                      marker_size=5.0, marker_labels=None)
```

## Coordinate finders

| Function | Returns |
|---|---|
| `find_cut_slices(img, direction='z', n_cuts=7, spacing='auto')` | List of "good" cut positions along an axis |
| `find_xyz_cut_coords(img, mask_img=None, activation_threshold=None)` | (x, y, z) of largest activation cluster |
| `find_parcellation_cut_coords(labels_img, background_label=0, return_label_names=False, label_hemisphere='left')` | Center of mass per label |
| `find_probabilistic_atlas_cut_coords(maps_img)` | Center per 4D component |

## Display objects (return values)

Volume plotting functions return a slicer or projector instance from `nilearn.plotting.displays`. These objects expose chaining methods:

| Method | Effect |
|---|---|
| `.add_overlay(img, cmap=..., threshold=...)` | Overlay another image |
| `.add_contours(img, levels=[...], colors=...)` | Draw contour outlines |
| `.add_markers(coords, marker_color=..., marker_size=...)` | Place markers |
| `.add_edges(img)` | Overlay edge map |
| `.add_graph(adjacency, coords)` | Add a connectome (glass brain) |
| `.annotate(...)` / `.title(...)` | Annotate or set title |
| `.savefig(path, dpi=...)` | Save figure |
| `.close()` | Release the matplotlib figure |

### Available display classes

- **Slicers** (volume cuts on anatomy): `OrthoSlicer`, `MosaicSlicer`, `TiledSlicer`, `XSlicer`, `YSlicer`, `ZSlicer`, `XZSlicer`, `YXSlicer`, `YZSlicer`, plus `BaseSlicer`
- **Projectors** (glass-brain projections): `OrthoProjector`, `LZRYProjector`, `LYRZProjector`, `LYRProjector`, `LZRProjector`, `LRProjector`, `LProjector`, `RProjector`, `XProjector`, `YProjector`, `ZProjector`, `XZProjector`, `YZProjector`, `YXProjector`
- **Axes**: `BaseAxes`, `CutAxes`, `GlassBrainAxes`
- **Surface figure**: `PlotlySurfaceFigure` (returned when `engine='plotly'`)
- **Helpers**: `get_slicer(display_mode)`, `get_projector(display_mode)`

```python
from nilearn.plotting.displays import get_slicer, get_projector
SlicerCls = get_slicer('ortho')      # -> OrthoSlicer
ProjCls   = get_projector('lyrz')    # -> LYRZProjector
```

### Misc

- `plotting.show()` — display all open matplotlib figures (wrapper around `plt.show`).

## Common patterns

### Stat map with overlays and markers
```python
display = plotting.plot_stat_map(zmap, threshold=3.1, display_mode='z',
                                  cut_coords=6, title='Faces > Houses')
display.add_contours(roi_img, levels=[0.5], colors='lime', linewidths=1.5)
display.add_markers([(0, -52, 18)], marker_color='yellow', marker_size=80)
display.savefig('faces.png', dpi=200)
display.close()
```

### Glass brain side-by-side with signed values
```python
plotting.plot_glass_brain(zmap, threshold=2.3, plot_abs=False,
                          display_mode='lyrz', colorbar=True,
                          cmap='RdBu_r', title='signed Z')
```

### Volume → surface projection
```python
fig = plotting.plot_img_on_surf(zmap, surf_mesh='fsaverage5',
                                 views=['lateral', 'medial'],
                                 hemispheres=['left', 'right'],
                                 inflate=True, threshold=3.0)
```

### Interactive viewer in Jupyter, save to disk
```python
view = plotting.view_img(zmap, threshold=3.0, bg_img=MNI152)
view  # auto-renders inline in Jupyter
view.save_as_html('report/zmap.html')
```

### Design matrix + contrast
```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plotting.plot_design_matrix(design_matrix, ax=axes[0])
plotting.plot_contrast_matrix('face - house', design_matrix, ax=axes[1])
```

### Connectome with custom edge threshold
```python
plotting.plot_connectome(corr_mat, coords,
                         edge_threshold='95%',
                         node_size=node_strength * 20,
                         display_mode='lzry',
                         colorbar=True)
```

### Carpet plot for QC
```python
plotting.plot_carpet(fmri_img, mask_img=brain_mask, t_r=2.0,
                     standardize='zscore_sample', title='Run-01 carpet')
```

### Mosaic slicer with explicit coords
```python
plotting.plot_stat_map(zmap, display_mode='mosaic',
                       cut_coords=(5, 5, 5))   # n cuts per axis
```

### Find coords automatically, then plot
```python
coords = plotting.find_xyz_cut_coords(zmap, activation_threshold=3.0)
plotting.plot_stat_map(zmap, cut_coords=coords, display_mode='ortho')
```

## Gotchas

1. **Default cmap changed in v0.13**: `'RdBu_r'` for diverging stat maps, `'gray'` for anat, `'inferno'` for sequential. Older notebooks relying on `'cold_hot'` defaults will look different.
2. **Use `'nearest'` interpolation for label/ROI images**, `'continuous'` for probabilistic / stat maps. Pass via `resampling_interpolation=...`.
3. **`plot_img_on_surf` does the volume→surface projection internally**; `plot_surf_stat_map` expects already-projected surface data (a `SurfaceImage` or array on the mesh).
4. **`view_*` returns an HTML object** — call `.save_as_html(path)` to persist or display inline in Jupyter (the object's repr renders as an iframe).
5. **`plot_glass_brain` `plot_abs=True` by default** — signed maps will be collapsed to magnitude. Set `plot_abs=False` for signed visualization with a diverging cmap.
6. **`output_file=...` returns `None`**: when you set `output_file`, the function saves the figure and returns `None` instead of a display object — you cannot chain `.add_overlay(...)` after that.
7. **`display_mode` strings differ between slicers and projectors**: slicers use `'ortho'|'tiled'|'mosaic'|'x'|'y'|'z'|'yx'|'xz'|'yz'`; glass brain projectors add `'l'|'r'|'lr'|'lzr'|'lyr'|'lzry'|'lyrz'`.
8. **`cut_coords` semantics depend on `display_mode`**: `int` = n cuts (auto-spaced); `list/tuple` = explicit MNI coords; `tuple of 3 ints` for `'mosaic'` = n cuts per axis.
9. **`engine='plotly'` for surface plots** returns a `PlotlySurfaceFigure`, not a matplotlib figure — use its `.figure` attribute for plotly-specific manipulation, and `.savefig()` for export.
10. **`symmetric_cbar='auto'`** chooses symmetric only when data spans both signs and `vmin/vmax` not set; force with `True`/`False` if you need consistent scales across plots.
11. **`plot_matrix` `reorder=True`** uses average linkage by default — pass `'single'`/`'complete'`/`'average'` explicitly for reproducibility, and labels are reordered too.
12. **`plot_connectome` requires N x N adjacency + N x 3 coords** in MNI space — coordinates from atlas labels via `find_parcellation_cut_coords(labels_img)`.
13. **Always `display.close()` in loops** — repeated `plot_*` calls leak matplotlib figures; close after `savefig` to avoid OOM in batch reports.
14. **`bg_img=None` vs default**: passing `None` removes the anatomical underlay; omitting the argument uses the MNI152 template.

## See also
- `references/surface.md` — for surface mesh inputs (`SurfaceImage`, `PolyMesh`, `vol_to_surf`)
- `references/datasets.md` — `load_mni152_template`, `load_fsaverage` for backgrounds
- `references/glm.md` — for design matrices and contrasts feeding the GLM plots
- https://nilearn.github.io/dev/modules/plotting.html
