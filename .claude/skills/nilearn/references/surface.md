# nilearn.surface — Manipulating Surface Data

Functions for surface manipulation. Provides the modern `SurfaceImage` / `PolyMesh` / `PolyData` data model used by surface maskers and surface plotting, as well as legacy loaders for raw GIFTI/FreeSurfer files and a volume-to-surface projection routine.

**Source:** https://nilearn.github.io/dev/modules/surface.html

## When to use

- Bundle a mesh + per-vertex data into a single object: `SurfaceImage`
- Hold left/right hemisphere meshes together: `PolyMesh`
- Hold left/right hemisphere data arrays together: `PolyData`
- Build a mesh from arrays vs. a file: `InMemoryMesh` / `FileMesh`
- Load arbitrary GIFTI/FreeSurfer surface files: `load_surf_mesh`, `load_surf_data`
- Project a volume onto a surface mesh: `vol_to_surf` (or `SurfaceImage.from_volume`)

## Inventory

### Classes

| Class | Purpose |
|---|---|
| `FileMesh` | A surface mesh stored in a Gifti or Freesurfer file. |
| `InMemoryMesh` | A surface mesh stored as in-memory numpy arrays. |
| `PolyData` | A collection of data arrays. |
| `PolyMesh` | A collection of meshes. |
| `SurfaceImage` | Surface image containing meshes & data for both hemispheres. |
| `SurfaceMesh` | A surface mesh having vertex, coordinates and faces (triangles). |

### Functions

| Function | Purpose |
|---|---|
| `load_surf_data(surf_data)` | Load data to be represented on a surface mesh. |
| `load_surf_mesh(surf_mesh)` | Load a surface mesh geometry. |
| `vol_to_surf(img, surf_mesh, ...)` | Extract surface data from a Nifti image. |

## SurfaceMesh (abstract base)

Abstract base class for a surface mesh with vertex coordinates and triangle faces.

Common attributes (on concrete subclasses):
- `coordinates` — `(n_vertices, 3)` array
- `faces` — `(n_faces, 3)` integer array
- `n_vertices` — int

## InMemoryMesh

```python
InMemoryMesh(coordinates, faces)
```

Holds the mesh as numpy arrays. Use when constructing meshes programmatically or after loading a custom format.

## FileMesh

```python
FileMesh(file_path)
```

Lazy-loaded mesh from GIFTI (`.gii`) or FreeSurfer (`lh.pial`, `rh.white`, ...) files. `coordinates` and `faces` are read on access.

Method: `loaded()` — returns an `InMemoryMesh` with arrays materialized.

## PolyMesh

```python
PolyMesh(left=None, right=None)
```

Container holding `left` and `right` `SurfaceMesh` objects. This is the format returned by `nilearn.datasets.load_fsaverage()`.

Attributes:
- `parts` — dict `{'left': mesh, 'right': mesh}`
- `n_vertices` — sum across parts

## PolyData

```python
PolyData(left=None, right=None, dtype=None)
```

Container holding `left` and `right` per-vertex data arrays. Shape per part is `(n_vertices,)` for a single map or `(n_vertices, n_timepoints)` for a time series.

Attributes:
- `parts` — dict `{'left': array, 'right': array}`
- `shape` — combined shape

## SurfaceImage

```python
SurfaceImage(mesh, data, dtype=None)
```

Holds a `PolyMesh` plus a `PolyData` for one image (or 4D time series). Counterpart to `Nifti1Image` for surface data.

Attributes:
- `mesh` — `PolyMesh`
- `data` — `PolyData`
- `shape`

Class methods:
- `SurfaceImage.from_volume(mesh, volume_img, **vol_to_surf_kwargs)` — project a volume onto the given mesh and wrap the result.

## load_surf_mesh

```python
load_surf_mesh(surf_mesh)
```

Load a single hemisphere mesh from a path (GIFTI or FreeSurfer) or pass through an existing mesh object. Returns an `InMemoryMesh`.

## load_surf_data

```python
load_surf_data(surf_data)
```

Load per-vertex data from GIFTI / FreeSurfer (`.curv`, `.thickness`, `.annot`, `.label`) / `.mgz` / `.nii` / `.csv` / `.txt`. Returns a numpy array.

## vol_to_surf

```python
vol_to_surf(
    img,                        # Niimg-like (3D or 4D)
    surf_mesh,                  # path, mesh, PolyMesh, or in-memory mesh
    radius=3.0,                 # mm; sampling distance from the surface
    interpolation='linear',     # 'linear' | 'nearest' | 'nearest_most_frequent'
    kind='auto',                # 'auto' | 'depth' | 'line' | 'ball'
    n_samples=None,             # number of samples per vertex
    mask_img=None,              # Niimg-like restricting voxels considered
    inner_mesh=None,            # white-matter surface for 'depth' sampling
    depth=None,                 # list of depths in mm for 'depth' kind
)
```

Returns a numpy array of shape `(n_vertices,)` for 3D input, or `(n_vertices, n_timepoints)` for 4D input.

`kind` choices:
- `'auto'` — picks `'depth'` if `inner_mesh` is given, else `'line'`
- `'depth'` — sample between `surf_mesh` (pial) and `inner_mesh` (white) at given relative `depth`s
- `'line'` — sample along the normal at distances up to `radius`
- `'ball'` — average voxels within a ball of radius `radius`

`interpolation` choices:
- `'linear'` — trilinear
- `'nearest'` — nearest-voxel
- `'nearest_most_frequent'` — modal label within the sampling region (use for integer label volumes)

## Common patterns

### Load fsaverage with the modern API

```python
from nilearn.datasets import load_fsaverage, load_fsaverage_data

mesh = load_fsaverage('fsaverage5')         # PolyMesh
sulcal = load_fsaverage_data(mesh_type='inflated', data_type='sulcal')
```

### Project volume to surface

```python
from nilearn.surface import SurfaceImage
from nilearn.datasets import load_fsaverage

mesh = load_fsaverage('fsaverage5')
surf_img = SurfaceImage.from_volume(mesh=mesh, volume_img=stat_map)
```

### Manual projection with sampling control

```python
from nilearn.surface import vol_to_surf

texture = vol_to_surf(
    fmri_4d, mesh.parts['left'],
    radius=3.0, interpolation='linear', kind='line', n_samples=10,
)
```

### Project an integer atlas

```python
from nilearn.surface import vol_to_surf

labels = vol_to_surf(
    atlas_int_img, mesh.parts['left'],
    interpolation='nearest_most_frequent', kind='ball', radius=3.0,
)
```

### Build a SurfaceImage from arrays

```python
from nilearn.surface import SurfaceImage, PolyMesh, PolyData, InMemoryMesh

left_mesh = InMemoryMesh(coords_lh, faces_lh)
right_mesh = InMemoryMesh(coords_rh, faces_rh)
mesh = PolyMesh(left=left_mesh, right=right_mesh)
data = PolyData(left=lh_values, right=rh_values)
img = SurfaceImage(mesh=mesh, data=data)
```

### Plot surface data

```python
from nilearn import plotting

plotting.plot_surf_stat_map(
    mesh.parts['left'], stat_map=surf_img.data.parts['left'],
    hemi='left', view='lateral', engine='matplotlib',
    cmap='RdBu_r', threshold=1.0,
)
```

### Surface masker pipeline

```python
from nilearn.maskers import SurfaceMasker

masker = SurfaceMasker(standardize='zscore_sample')
X = masker.fit_transform(surf_img)            # (n_timepoints, n_vertices)
back = masker.inverse_transform(X)            # SurfaceImage
```

## Gotchas

- The new surface API (`load_fsaverage()` -> `PolyMesh`) is preferred; the legacy `fetch_surf_fsaverage()` returns the old dict-of-paths format and is incompatible with `SurfaceImage`.
- `SurfaceImage.from_volume` is the convenience wrapper around `vol_to_surf` for both hemispheres at once.
- For integer label volumes (atlases), use `interpolation='nearest_most_frequent'` — `'linear'` will produce nonsensical fractional labels.
- `kind='depth'` requires `inner_mesh` (typically the white-matter surface). With both pial + white available, prefer depth sampling for cortical signals.
- `vol_to_surf(img, mesh)` for a `PolyMesh` projects a single hemisphere; pass `mesh.parts['left']` or `mesh.parts['right']` explicitly when working hemisphere-wise.
- `PolyData.parts['left']` shape is `(n_vertices,)` for a 3D image and `(n_vertices, n_timepoints)` for 4D — surface maskers transpose to `(n_timepoints, n_vertices)` on `transform`.
- `FileMesh` is lazy: file errors surface only when `.coordinates` / `.faces` are accessed.

## See also

- `references/maskers.md` — `SurfaceMasker`, `SurfaceLabelsMasker`, `SurfaceMapsMasker`
- `references/datasets.md` — `load_fsaverage`, `load_fsaverage_data`, `fetch_atlas_surf_destrieux`
- Source: https://nilearn.github.io/dev/modules/surface.html
