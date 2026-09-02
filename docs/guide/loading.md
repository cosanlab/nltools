---
title: Loading data & masks
---

[`BrainData`](../api/data/brain_data.md) is the entry point. Hand its constructor a NIfTI path, a
list of paths, a URL, an `.h5` bundle, a nibabel image, another `BrainData`, or a numpy array plus
a `mask=`. A list of paths stacks into one `(n_images, n_voxels)` object. The optional `X` and `Y`
arguments attach per-image tables (design/covariates and targets) that travel with the data.

**The mask decides the grid.** With no `mask=`, nltools uses the bundled MNI template at the
brain space's current resolution. If your data sits on a grid no bundled template matches, say
4 mm, nltools resamples it to the closest bundled 1/2/3 mm template and raises a
[`ResamplingWarning`](../api/tasks/design-and-glm.md#tasks-design-and-glm-resamplingwarning)
naming the fallback. To keep the native resolution, pass `mask=` with a mask in your data's own
space. Template names follow the
`'{res}mm-MNI152-2009{version}'` pattern, where the version code is `fsl` (default, 2/3 mm),
`a` (nilearn, 1/2/3 mm), or `c` (fmriprep, 1/2 mm).

Goal | Use | Notes
--- | --- | ---
Load one or many images | `BrainData(path_or_list)` | List input stacks; mixed grids are resampled to the mask
Use a specific grid | `BrainData(..., mask='3mm-MNI152-2009fsl')` | Also accepts a `Nifti1Image` or a mask path
Change the global default | [`set_brainspace`](../api/tasks/loading.md#tasks-loading-set-brainspace) / [`with_brainspace`](../api/tasks/loading.md#tasks-loading-with-brainspace) | `with_brainspace` is a context manager; [`reset_brainspace`](../api/tasks/loading.md#tasks-loading-reset-brainspace) restores defaults
Save / reload with metadata | [`BrainData.write`](../api/data/brain_data.md#data-brain-data-write) → `.h5` | HDF5 round-trips `X`, `Y`, and the mask; `.nii.gz` does not
Example data | [`fetch_pain`](../api/tasks/loading.md#tasks-loading-fetch-pain), [`fetch_emotion_ratings`](../api/tasks/loading.md#tasks-loading-fetch-emotion-ratings), [`load_haxby_example`](../api/tasks/loading.md#tasks-loading-load-haxby-example) | Cached on first use; `load_haxby_example` is synthetic and needs no network
Bundled masks and atlases | [`list_resources`](../api/tasks/loading.md#tasks-loading-list-resources), [`fetch_resource`](../api/tasks/loading.md#tasks-loading-fetch-resource) | Returns a local path; parcellations live under `masks/`
Published maps | [`fetch_neurovault_collection`](../api/tasks/loading.md#tasks-loading-fetch-neurovault-collection), [`download_nifti`](../api/tasks/loading.md#tasks-loading-download-nifti) | `BrainData` also accepts a URL directly
Build a mask | [`create_sphere`](../api/tasks/loading.md#tasks-loading-create-sphere), [`expand_mask`](../api/tasks/loading.md#tasks-loading-expand-mask), [`collapse_mask`](../api/tasks/loading.md#tasks-loading-collapse-mask) | `expand_mask` turns one labeled atlas into per-ROI binary masks
Many subjects | [`BrainCollection.from_paths`](../api/data/brain_collection.md#data-brain-collection-from-paths) / [`from_bids`](../api/data/brain_collection.md#data-brain-collection-from-bids) / [`from_glob`](../api/data/brain_collection.md#data-brain-collection-from-glob) | See [Working with many subjects](collections.md)
Stack objects | [`concatenate`](../api/tasks/loading.md#tasks-loading-concatenate) | Works on lists of `BrainData` or `Adjacency`

## Loading

```python
from nltools.data import BrainData
from nltools.datasets import fetch_pain

pain = fetch_pain()          # 84 images; the metadata table is in .X
stack = BrainData(paths, mask="3mm-MNI152-2009fsl")
```

## Brain space and HDF5

`set_brainspace` changes the template every mask-less object falls back on for the rest of the
session. Prefer `with_brainspace` when you only need the change for a few lines.

```python
from nltools.templates import get_brainspace, set_brainspace, with_brainspace

set_brainspace(resolution=3)             # 3 mm from here on
with with_brainspace(resolution=2):      # 2 mm inside the block only
    coarse = BrainData(paths[0])

pain[:5].write("subset.h5")
subset = BrainData("subset.h5")          # X, Y, and mask come back intact
```

## Bundled files

`list_resources(prefix=...)` browses the `nltools/niftis` dataset without downloading;
`fetch_resource` downloads one file and returns its local path.

```python
from nltools.templates import fetch_resource, list_resources
from nltools.mask import create_sphere

list_resources(prefix="default")
fetch_resource("default/2mm-MNI152-2009fsl-mask.nii.gz")
sphere = create_sphere([0, 20, 30], radius=8)
```

Next: [Design matrices & GLM](design-and-glm.md), or the
[BrainData tutorial](../tutorials/basics/01_brain_data.md) for a worked walkthrough.
