# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Working with BrainData — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Working with BrainData

    `BrainData` is the object almost every nltools analysis starts from: imaging
    data as an images-by-voxels matrix, one row per image and one column per
    voxel inside the mask. That shape is what makes it behave like a dataframe —
    index it, slice it, do arithmetic on it, iterate over it, and the metadata
    follows.

    ## Load

    `fetch_pain()` retrieves the pain dataset from
    [Chang et al., 2015](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002180):
    28 subjects with one contrast image each at low, medium and high pain
    intensity, cached locally on first use. The `joblib` cache below keeps the
    docs build from reloading 84 images on every run; call `fetch_pain()`
    directly in your own code.

    Printing an object reports its shape, grid and mask, and `len()` is the
    number of images. `.data` is the `numpy` array underneath, `.X` a `polars`
    DataFrame of per-image metadata, and `.Y` outcomes or labels the same way:
    """)
    return


@app.cell
def _():
    from joblib import Memory

    from nltools.data import BrainData
    from nltools.datasets import fetch_pain

    memory = Memory(".tutorial-cache", verbose=0)
    data = memory.cache(fetch_pain)()

    print(data)
    print(f".data is a {type(data.data).__name__} of shape {data.data.shape}")
    data.X.head()
    return BrainData, data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Your own data loads by path, and many files load together as a list.
    `BrainData` also takes a URL, a `nibabel` image, or a list of any of those.

    ```python
    one = BrainData("sub-01_pain-high.nii.gz")
    many = BrainData(["sub-01_pain-high.nii.gz", "sub-02_pain-high.nii.gz"])
    ```

    Pass `mask` to put an object on a specific grid — see
    [Brain space and resolution](#brain-space-and-resolution) for what decides
    the grid when you do not.

    ## Index and slice

    Indexing works the Python way, and every form carries `.X` and `.Y` along
    with the images it keeps: an integer gives one image, a slice a range of them
    on the same voxels and grid, a list of integers any images in any order, and
    a boolean array — usually from a column of `.X` — filters images. `append`
    concatenates along the image axis, and objects are iterable:
    """)
    return


@app.cell
def _(data):
    high_pain = data[data.X["PainLevel"] == 3]

    print(data[0])
    print(data[:5])
    print(data[[0, 10, 20, 30]])
    print(f"{len(high_pain)} high-pain images")
    print(data[:2].append(data[4]))
    [round(float(image.data.mean()), 2) for image in data[:5]]
    return (high_pain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arithmetic and statistics

    Scalars broadcast over every voxel, and two objects add and subtract
    voxelwise, so subtracting two images gives one contrast map. Multiplying by a
    vector as long as the stack is a weighted sum across images instead, which is
    how you write a contrast over more than two conditions. `mean` and `std`
    reduce across images by default (`axis=0`), giving one value per voxel;
    `axis=1` reduces across voxels. Methods chain, so temporal signal-to-noise
    ratio is one expression.

    `standardize` z-scores or centers each voxel across images. `threshold` zeros
    everything below `upper`, everything above `lower`, or everything between the
    two when you pass both, resolving a percentile string over the nonzero
    voxels. `binarize=True` turns the survivors into ones, which is how you get a
    mask out of a map:
    """)
    return


@app.cell
def _(data):
    print((data + 10) * 2)
    print(data[1] - data[0])

    mean_map = data.mean()
    tsnr = mean_map / data.std()
    tsnr.plot(title="Temporal signal-to-noise ratio")

    z_scored = data.standardize(method="zscore")
    print(f"z-scored mean across voxels: {z_scored.data.mean():.6f}")

    top_voxels = mean_map.threshold(upper="95%", binarize=True)
    print(f"top 5% of the mean map: {top_voxels.data.sum():.0f} voxels")
    mean_map.threshold(upper="95%").plot(cmap="Blues", title="Top 5% of voxels")
    return (mean_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save

    `write` saves to NIfTI, or to HDF5 when the file name ends in `.h5` or
    `.hdf5`. Both load back with `BrainData`, but only HDF5 preserves `.X`, `.Y`
    and the mask: a NIfTI file holds the images and nothing else, so a round trip
    through it loses the metadata.
    """)
    return


@app.cell
def _(BrainData, data):
    import tempfile
    from pathlib import Path

    tmpdir = Path(tempfile.mkdtemp())
    nifti_path = tmpdir / "pain_subset.nii.gz"
    h5_path = tmpdir / "pain_subset.h5"

    data[:3].write(str(nifti_path))
    data[:3].write(str(h5_path))

    from_nifti = BrainData(str(nifti_path))
    from_h5 = BrainData(str(h5_path))

    print(f"NIfTI {nifti_path.stat().st_size / 1e6:.1f} MB")
    print(f"  -> {from_nifti.shape}, X has {from_nifti.X.shape[1]} columns")
    print(f"HDF5  {h5_path.stat().st_size / 1e6:.1f} MB")
    print(f"  -> {from_h5.shape}, X has {from_h5.X.shape[1]} columns")
    return Path, tempfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot

    `plot` draws a glass brain by default, and `method='slices'` draws
    cross-sections, one row per axis in `view`; the remaining keywords go to the
    matching [nilearn](https://nilearn.github.io) plot. On a stack it draws one
    figure per image up to `limit`, which defaults to 3 so an 84-image object does
    not silently produce 84 figures. `plot_surf` projects the volume onto the
    fsaverage pial surface as a lateral and medial montage, `plot_flatmap` onto a
    flattened surface, and `to_nifti` hands back a `nibabel` image — 3D for one
    image, 4D for a stack — for anything these do not cover:
    """)
    return


@app.cell
def _(data, mean_map):
    from nilearn.plotting import plot_stat_map

    mean_map.plot(title="Mean activation")
    data[:4].plot(limit=4)
    mean_map.plot(method="slices", view="xyz")
    mean_map.plot_surf()
    mean_map.plot_flatmap()

    _display = plot_stat_map(mean_map.to_nifti(), display_mode="z", cut_coords=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `iplot` returns an interactive [niivue](https://niivue.com) viewer: a WebGL
    `anywidget` with a threshold slider above the volume. Drag the slider (or
    right-drag the image) to window the map, scroll through slices, scrub 4D
    frames, render in 3D, and overlay an atlas with hover-to-label.
    `controls=False` hides the slider, `colorbar=False` the colorbar. It needs a
    live kernel, so this static page shows a placeholder instead:
    """)
    return


@app.cell
def _(mean_map):
    mean_map.iplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Masks and ROIs

    A mask restricts an object to the voxels you care about. Masks come from three
    places: a sphere around a coordinate, a parcellation split into its regions,
    and a thresholded statistic map.

    `create_sphere` draws binary spheres on the current brain space's grid.
    Centers are MNI millimeter coordinates and `radius` is in millimeters, so the
    same request covers the same physical volume whatever grid you are on.
    `apply_mask` rewrites the voxel axis to the mask's own support: a sphere
    inside the brain leaves a much narrower object, and the voxels it drops are
    gone rather than zeroed. It needs the mask on the object's grid already;
    `extract_roi` is the one that resamples, collapsing each region of a mask to
    one number per image — here the mean signal in the sphere for each of the 84
    images:
    """)
    return


@app.cell
def _(data):
    import matplotlib.pyplot as plt

    from nltools.mask import create_sphere

    sphere = create_sphere([0, 0, 0], radius=30)
    sphere_data = data.apply_mask(sphere)

    print(f"whole brain:  {data.shape}")
    print(f"30 mm sphere: {sphere_data.shape}")
    sphere_data.mean().plot()

    roi_mean = data.extract_roi(sphere)
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(roi_mean)
    _ax.set(xlabel="image", ylabel="mean signal", title="30 mm sphere at [0, 0, 0]")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A parcellation is one image whose voxel values are integer region IDs.
    nltools ships several; this is a 50-region whole-brain parcellation, fetched
    from the nltools data repository on Hugging Face. `expand_mask` turns those
    IDs into a stack of 50 binary masks, one per region, and `collapse_mask`
    folds such a stack back into one labeled image, numbering the regions in
    stack order and dropping any overlap.

    A statistic map becomes a mask the same way: averaging the high-pain images
    and keeping the tails outside the middle 95% gives a thresholded map, and
    `regions` splits that into one image per blob — smoothing the map first,
    cutting each connected component at its local peaks, and dropping anything
    smaller than `min_region_size` (1350 mm³):
    """)
    return


@app.cell
def _(BrainData, high_pain):
    from nltools.datasets import fetch_resource
    from nltools.mask import collapse_mask, expand_mask

    _k50 = "masks/default/2mm-MNI152-2009fsl-k50.nii.gz"
    parcellation = BrainData(fetch_resource(_k50))
    regions = expand_mask(parcellation)

    print(f"{parcellation.shape} labeled image -> {regions.shape} binary masks")
    regions[:3].plot()
    collapse_mask(regions).plot(title="Collapsed back to one labeled image")

    blobs = high_pain.mean().threshold(lower="2.5%", upper="97.5%").regions()
    print(f"{len(blobs)} regions from the thresholded high-pain mean")
    blobs.plot(limit=len(blobs))
    return parcellation, regions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Masks earn their keep on a round trip: summarize each region, run an analysis
    over regions, paint the result back into a brain image. Multiplying each
    subject's three images by `[-1, 0, 1]` gives one high-minus-low pain contrast
    per subject; `extract_roi` on the labeled parcellation turns those into
    regions by subjects; correlation distance between the region profiles is a
    50-node `Adjacency`, thresholded at the 15th percentile of distances to keep
    the pairs whose profiles track each other across subjects. `to_graph` hands
    that matrix to `networkx` for any graph metric — degree counts how many
    regions each region is tied to — and `roi_to_brain` writes one value per
    region back into the expanded masks:
    """)
    return


@app.cell
def _(BrainData, data, parcellation, regions):
    import numpy as np
    from sklearn.metrics import pairwise_distances

    from nltools.data import Adjacency
    from nltools.mask import roi_to_brain

    contrast = BrainData(
        [
            data[data.X["SubjectID"] == subject] * np.array([-1, 0, 1])
            for subject in data.X["SubjectID"].unique(maintain_order=True)
        ]
    )
    region_profiles = contrast.extract_roi(parcellation)
    print(f"{contrast.shape} contrast images")
    print(f"{region_profiles.shape} (regions, subjects)")

    distance = Adjacency(
        pairwise_distances(region_profiles, metric="correlation"),
        matrix_type="distance",
    )
    connected = distance.threshold(lower="15%", binarize=True)
    connected.plot()

    graph = connected.to_graph()
    degree = np.array([d for _, d in sorted(graph.degree())])

    print(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"degree range: {degree.min()}-{degree.max()}")
    roi_to_brain(degree, regions).plot(title="Degree centrality")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Brain space and resolution

    Every object lives on a grid: a template, a resolution, and the brain mask
    that decides which voxels are in. nltools defaults to a 2 mm MNI152 template
    and gives three levels of control over that choice — a global setting, a
    scoped override, and a per-object mask. `get_brainspace()` reports the active
    configuration and the files it resolves to.

    `Simulator` has no grid of its own, so it adopts the current brain space,
    which makes it a convenient way to see where an object landed.
    `set_brainspace()` changes the default for everything that follows — set it
    once at the top of an analysis, since changing it midway leaves later objects
    on a different grid than earlier ones, a hard mistake to spot — and
    `reset_brainspace()` puts the default back. `with_brainspace()` does the same
    for the duration of a block, restoring the previous setting on exit even when
    the block raises.

    That setting is only a *default*, for objects with no grid of their own: a
    file already on a recognized template grid keeps it, so the 2 mm file written
    below loads as 2 mm data inside a 3 mm block:
    """)
    return


@app.cell
def _(BrainData, Path, tempfile):
    from nltools import (
        get_brainspace,
        reset_brainspace,
        set_brainspace,
        with_brainspace,
    )
    from nltools.data import Simulator

    print(get_brainspace())

    dummy_brain = Simulator(random_state=0).create_data([0, 1], 1, reps=3)
    brain_2mm = Path(tempfile.mkdtemp()) / "dummy_2mm_brain.nii.gz"
    dummy_brain.write(str(brain_2mm))
    print(f"simulated under the 2 mm default: {dummy_brain.shape[1]} voxels")

    set_brainspace(resolution=3)
    dummy_3mm = Simulator(random_state=0).create_data([0, 1], 1, reps=3)
    print(f"simulated under the 3 mm default: {dummy_3mm.shape[1]} voxels")
    default_space = reset_brainspace()

    with with_brainspace(resolution=3):
        scoped = Simulator(random_state=0).create_data([0, 1], 1, reps=1)
        loaded_under_3mm = BrainData(str(brain_2mm))

    print(f"simulated inside a 3 mm block: {scoped.shape[1]} voxels")
    print(f"2 mm file loaded inside it: {loaded_under_3mm.shape[1]} voxels")
    print(f"after the block: {get_brainspace().resolution} mm is active again")
    return brain_2mm, default_space, get_brainspace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To put one object on a specific grid regardless of the global setting, pass
    `mask`: a template name, a path to any NIfTI file, or a `nibabel` image. The
    data is resampled to match and the global setting is untouched. Template
    names follow `'{resolution}mm-MNI152-2009{version}'`, where the version letter
    picks the family: `fsl` for the bundled default, `a` for nilearn's, `c` for
    fMRIPrep. `get_brainspace().mask`, `.brain` and `.plot` give the resolved file
    paths when another tool needs them.
    """)
    return


@app.cell
def _(BrainData, brain_2mm, default_space, get_brainspace):
    on_3mm_grid = BrainData(str(brain_2mm), mask="3mm-MNI152-2009fsl")

    print(f"global default: {default_space.resolution} mm")
    print(f"named 3 mm mask: {on_3mm_grid.shape[1]} voxels")
    print(f"global default after that: {get_brainspace().resolution} mm")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NeuroVault

    [NeuroVault](https://neurovault.org) is a public repository of unthresholded
    statistical maps. nltools can pull a whole collection, fetch a single image by
    URL, and push your own maps back up.

    `fetch_neurovault_collection` takes a collection ID and returns the image
    metadata and the local file paths; files land in nilearn's data directory
    unless you pass `data_dir`, and are reused on later calls. Hand both to
    `BrainData` and the metadata rides along on `.X`, one row per image.
    Collection 2099 is a three-image parcellation set.

    `download_nifti` fetches one image by URL and returns the path it wrote —
    without `data_dir`, into the working directory. `BrainData` accepts a URL
    directly too, which saves a line but deletes the download as soon as the data
    is in memory, so use `download_nifti` when you want to keep the file:
    """)
    return


@app.cell
def _(BrainData, tempfile):
    from nltools.datasets import download_nifti, fetch_neurovault_collection

    nv_metadata, nv_files = fetch_neurovault_collection(2099, verbose=0)
    collection = BrainData(nv_files, X=nv_metadata)
    print(nv_metadata.select("id", "name", "map_type", "modality"))

    _url = "https://neurovault.org/media/images/2099/Neurosynth%20Parcellation_0.nii.gz"
    neurosynth = BrainData(download_nifti(_url, data_dir=tempfile.mkdtemp()))
    print(neurosynth)

    collection.plot(limit=len(collection))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `upload_neurovault` pushes an object to a new or existing collection, using an
    access token generated under your NeuroVault account settings.
    `collection_name` creates a new collection; `collection_id` adds to one you
    already have. `img_type` and `img_modality` are required, and anything else
    you pass is forwarded as image metadata, as are the columns of `.X`. Each
    image is named for its collection and its position in the object. The call
    below is not run when these docs are built, because it would write to a live
    public repository.

    ```python
    neurosynth.upload_neurovault(
        access_token="your_neurovault_api_key",
        collection_name="Neurosynth Parcellation",
        img_type="Pa",
        img_modality="Other",
        analysis_level="M",
    )
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
