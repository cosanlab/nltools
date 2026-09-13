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

    `BrainData` is the object almost every nltools analysis starts from. It holds
    imaging data as an images-by-voxels matrix: one row per image, one column per
    voxel inside the mask. That shape is what makes it feel like a dataframe — you
    index it, slice it, do arithmetic on it, and iterate over it with plain
    Python, and the metadata comes along.

    ## Load

    `fetch_pain()` retrieves the pain dataset from
    [Chang et al., 2015](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002180):
    28 subjects with three beta images each, at low, medium and high thermal pain.
    The files are cached locally on first use and loaded straight into a
    `BrainData`. The `joblib` cache below keeps the docs build from reloading 84
    images on every run; you can call `fetch_pain()` directly.
    """)
    return


@app.cell
def _():
    from joblib import Memory

    from nltools.data import BrainData
    from nltools.datasets import fetch_pain

    memory = Memory(".tutorial-cache", verbose=0)

    data = memory.cache(fetch_pain)()
    data
    return BrainData, data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `len()` is the number of images and `.shape` is images by voxels; `.data` is
    the `numpy` array underneath. Image metadata lives on `.X` as a `polars`
    DataFrame with one row per image, and `.Y` holds outcomes or labels the same
    way:
    """)
    return


@app.cell
def _(data):
    print(f"{len(data)} images x {data.shape[1]} in-mask voxels")
    print(f".data is a {type(data.data).__name__} of shape {data.data.shape}")
    data.X.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Your own data loads by path, and many files load together as a list.
    `BrainData` also takes a URL, a `nibabel` image, or a list of any of those.

    ```python
    from nltools.data import BrainData

    one = BrainData("sub-01_pain-high.nii.gz")
    many = BrainData(["sub-01_pain-high.nii.gz", "sub-02_pain-high.nii.gz"])
    remote = BrainData("https://neurovault.org/media/images/2099/some_map.nii.gz")
    ```

    Pass `mask` to put an object on a specific grid — see
    [Brain space and resolution](#brain-space-and-resolution) for what decides
    the grid when you do not.

    ## Index and slice

    Indexing works the Python way, and every form carries `.X` and `.Y` along with
    the images it keeps. An integer gives one image, a slice a range of them on the
    same voxels and the same grid, a list of integers any images in any order:
    """)
    return


@app.cell
def _(data):
    print(data[0])
    print(data[:5])
    data[[0, 10, 20, 30]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    A boolean array filters images, and a column of `.X` is the usual source of
    one. `append` concatenates along the image axis, and objects are iterable, so
    a comprehension gives one value per image:
    """)
    return


@app.cell
def _(data):
    high_pain = data[data.X["PainLevel"] == 3]
    print(f"{len(high_pain)} high-pain images")
    print(data[:2].append(data[4]))
    [round(float(image.data.mean()), 2) for image in data[:5]]
    return (high_pain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arithmetic and statistics

    Scalars broadcast over every voxel, and two objects add and subtract
    voxelwise, so subtracting two images gives one contrast map. `mean` and `std`
    reduce across images by default (`axis=0`), giving one value per voxel;
    `axis=1` reduces across voxels instead. Methods chain, so the mean of 84
    images is a single image and temporal signal-to-noise ratio is one expression:
    """)
    return


@app.cell
def _(data):
    print((data + 10) * 2)
    print(data[1] - data[0])

    mean_map = data.mean()
    tsnr = mean_map / data.std()
    tsnr.plot(title="Temporal signal-to-noise ratio")
    return (mean_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `standardize` z-scores or centers each voxel across images. `threshold` cuts a
    map at an absolute value or a percentile and keeps the surviving values;
    `binarize=True` turns the survivors into ones, which is how you get a mask out
    of a map:
    """)
    return


@app.cell
def _(data, mean_map):
    z_scored = data.standardize(method="zscore")
    print(f"z-scored mean across voxels: {z_scored.data.mean():.6f}")

    top_voxels = mean_map.threshold(upper="95%", binarize=True)
    print(f"top 5% of the mean map: {top_voxels.data.sum():.0f} voxels")
    mean_map.threshold(upper="95%").plot(cmap="Blues", title="Top 5% of voxels")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save

    `write` saves to NIfTI, or to HDF5 when the file name ends in `.h5`. Both load
    back with `BrainData`, but only HDF5 preserves `.X`, `.Y` and the mask: a NIfTI
    file holds the images and nothing else, so a round trip through it loses the
    metadata.
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

    `plot` draws a glass brain by default; `method` picks another view, and the
    remaining keywords go to the matching [nilearn](https://nilearn.github.io)
    plot. Calling it on a stack draws one figure per image. `limit` caps how many,
    and defaults to 3 so that an 84-image object does not silently produce 84
    figures; raise it, or index first, when you want more:
    """)
    return


@app.cell
def _(data, mean_map):
    mean_map.plot(title="Mean activation")
    data[:4].plot(limit=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `method='slices'` draws cross-sections, one row per axis in `view`. `plot_surf`
    projects the volume onto an inflated cortical surface, and `plot_flatmap` onto
    a flattened one:
    """)
    return


@app.cell
def _(mean_map):
    mean_map.plot(method="slices", view="xyz")
    mean_map.plot_surf()
    mean_map.plot_flatmap()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    For anything these do not cover, `to_nifti` converts back to a `nibabel` image
    — 3D for one image, 4D for a stack — which any other toolbox accepts:
    """)
    return


@app.cell
def _(mean_map):
    from nilearn.plotting import plot_stat_map

    _display = plot_stat_map(mean_map.to_nifti(), display_mode="z", cut_coords=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `iplot` returns an interactive [niivue](https://niivue.com) viewer: a WebGL
    `anywidget` with a threshold slider above the volume. Drag the slider (or
    right-drag on the image) to window the map, scroll through slices, scrub 4D
    frames, render in 3D, and overlay an atlas with hover-to-label. `controls=False`
    hides the slider and `colorbar=False` the colorbar. It needs a live kernel, so
    this static page shows a placeholder where the viewer would be:
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

    `create_sphere` draws binary spheres. Centers are MNI millimeter coordinates
    and `radius` is in millimeters, so the same request covers the same physical
    volume whatever grid you are on. `apply_mask` keeps only the voxels inside it,
    dropping the rest rather than zeroing them, so the masked object is narrower
    than the original:
    """)
    return


@app.cell
def _(data):
    from nltools.mask import create_sphere

    sphere = create_sphere([0, 0, 0], radius=30)
    sphere_data = data.apply_mask(sphere)

    print(f"whole brain:  {data.shape}")
    print(f"30 mm sphere: {sphere_data.shape}")
    sphere_data.mean().plot()
    return (sphere,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `extract_roi` collapses each region of a mask to one number per image. With a
    single binary region that is one value per image — the mean signal in the
    sphere for each of the 84 images. The mask is resampled onto the object's own
    grid first, so the two do not have to match resolutions:
    """)
    return


@app.cell
def _(data, sphere):
    import matplotlib.pyplot as plt

    roi_mean = data.extract_roi(sphere)

    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(roi_mean)
    _ax.set(xlabel="image", ylabel="mean signal", title="30 mm sphere at [0, 0, 0]")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A parcellation is one image whose voxel values are integer region IDs. nltools
    ships several; this is a 50-region whole-brain parcellation, fetched from the
    package's data repository. `expand_mask` turns those IDs into a stack of 50
    binary masks, one per region. `collapse_mask` is the inverse: it folds a stack
    of binary masks back into one labeled image, numbering the regions in stack
    order and dropping any overlap.
    """)
    return


@app.cell
def _(BrainData):
    from nltools.datasets import fetch_resource
    from nltools.mask import collapse_mask, expand_mask

    parcellation = BrainData(
        fetch_resource("masks/default/2mm-MNI152-2009fsl-k50.nii.gz")
    )
    regions = expand_mask(parcellation)

    print(f"{parcellation.shape} labeled image -> {regions.shape} binary masks")
    regions[:3].plot()
    collapse_mask(regions).plot(title="Collapsed back to one labeled image")
    return parcellation, regions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A statistic map becomes a mask the same way. Averaging the high-pain images and
    keeping the tails outside the middle 95% gives a thresholded map, and `regions`
    splits that into its spatially contiguous blobs, one image per blob:
    """)
    return


@app.cell
def _(high_pain):
    blobs = high_pain.mean().threshold(lower="2.5%", upper="97.5%").regions()
    print(f"{len(blobs)} contiguous regions")
    blobs.plot(limit=len(blobs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Masks earn their keep on a round trip: summarize each region, run an analysis
    over regions, and paint the result back into a brain image. Here we compute a
    linear pain contrast per subject — one image per subject, high minus low pain —
    correlate the 50 regions across subjects, threshold that correlation structure
    into a graph, and map each region's degree back onto the brain:
    """)
    return


@app.cell
def _(BrainData, data):
    import numpy as np

    contrast = BrainData(
        [
            data[data.X["SubjectID"] == subject] * np.array([1, -1, 0])
            for subject in data.X["SubjectID"].unique(maintain_order=True)
        ]
    )
    contrast
    return contrast, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `extract_roi` on a labeled parcellation gives regions by images — the profile of
    each region across the 28 subjects. Correlation distance between those profiles
    is a 50-node `Adjacency`, and thresholding it keeps only the region pairs that
    covary across subjects:
    """)
    return


@app.cell
def _(contrast, parcellation):
    from sklearn.metrics import pairwise_distances

    from nltools.data import Adjacency

    region_profiles = contrast.extract_roi(parcellation)
    print(f"{region_profiles.shape} (regions, subjects)")

    distance = Adjacency(
        pairwise_distances(region_profiles, metric="correlation"),
        matrix_type="distance",
    )
    connected = distance.threshold(upper=0.4, binarize=True)
    connected.plot()
    return (connected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `to_graph` hands the thresholded matrix to `networkx`, where any graph metric is
    available. Degree counts how many regions each region is tied to, and
    `roi_to_brain` writes one value per region back into the expanded masks:
    """)
    return


@app.cell
def _(connected, np, regions):
    from nltools.mask import roi_to_brain

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

    Every object lives on a grid: a template, a resolution, and the brain mask that
    decides which voxels are in. nltools defaults to a 2 mm MNI152 template and
    gives you three levels of control over that choice — a global setting, a scoped
    override, and a per-object mask. `get_brainspace()` reports the active
    configuration and the files it resolves to:
    """)
    return


@app.cell
def _():
    from nltools import (
        get_brainspace,
        reset_brainspace,
        set_brainspace,
        with_brainspace,
    )
    from nltools.data import Simulator

    get_brainspace()
    return (
        Simulator,
        get_brainspace,
        reset_brainspace,
        set_brainspace,
        with_brainspace,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Simulator` generates data with a known signal, which is a convenient way to see
    which grid an object landed on. It has no grid of its own, so it adopts the
    current brain space. Save one, so there is a 2 mm file on disk to load back in
    below. `set_brainspace()` changes the default for everything that follows — set
    it once at the top of an analysis, since changing it midway leaves later objects
    on a different grid than earlier ones, which is a hard mistake to spot.
    `reset_brainspace()` puts the default back:
    """)
    return


@app.cell
def _(Path, Simulator, reset_brainspace, set_brainspace, tempfile):
    dummy_brain = Simulator(random_state=0).create_data([0, 1], 1, reps=3)
    brain_2mm = Path(tempfile.mkdtemp()) / "dummy_2mm_brain.nii.gz"
    dummy_brain.write(str(brain_2mm))
    print(f"simulated under the 2 mm default: {dummy_brain.shape[1]} voxels")

    set_brainspace(resolution=3)
    dummy_3mm = Simulator(random_state=0).create_data([0, 1], 1, reps=3)
    print(f"simulated under the 3 mm default: {dummy_3mm.shape[1]} voxels")

    default_space = reset_brainspace()
    default_space
    return brain_2mm, default_space


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `with_brainspace()` does the same for the duration of a block and restores the
    previous setting on exit, including when the block raises. Prefer it when only
    part of an analysis belongs in another space.

    The setting is a *default* for objects that have no grid of their own. A file
    that already sits on a recognized template grid keeps it, so loading the 2 mm
    file written above inside the block gives 2 mm data even while the default says
    3 mm:
    """)
    return


@app.cell
def _(BrainData, Simulator, brain_2mm, get_brainspace, with_brainspace):
    with with_brainspace(resolution=3):
        scoped = Simulator(random_state=0).create_data([0, 1], 1, reps=1)
        loaded_under_3mm = BrainData(str(brain_2mm))

    print(f"simulated inside the block: {scoped.shape[1]} voxels")
    print(f"2 mm file loaded inside it: {loaded_under_3mm.shape[1]} voxels")
    print(f"after the block: {get_brainspace().resolution} mm is active again")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To put one object on a specific grid regardless of the global setting, pass
    `mask`. It takes a template name, a path to any NIfTI file, or a `nibabel`
    image, and the data is resampled to match; the global setting is untouched.
    Template names follow `'{resolution}mm-MNI152-2009{version}'`, where the version
    letter picks the family: `fsl` for the bundled default, `a` for nilearn's, `c`
    for the fMRIPrep template. `get_brainspace().mask`, `.brain` and `.plot` give
    the resolved file paths when you need to hand them to another tool.
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
    metadata and the local file paths. Files land in nilearn's data directory unless
    you pass `data_dir`, and are reused on later calls. Hand both to `BrainData` and
    the metadata rides along on `.X`, one row per image. Collection 2099 is a
    three-image parcellation set:
    """)
    return


@app.cell
def _(BrainData):
    from nltools.datasets import fetch_neurovault_collection

    nv_metadata, nv_files = fetch_neurovault_collection(2099, verbose=0)
    collection = BrainData(nv_files, X=nv_metadata)

    print(nv_metadata.select("id", "name", "map_type", "modality"))
    collection.plot(limit=len(collection))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `download_nifti` fetches one image by URL and returns the path it wrote. Without
    `data_dir` it writes to the working directory, so give it a location you
    control. `BrainData` also accepts a URL directly, which saves a line but drops
    the file in a temporary directory your system will eventually clear — use
    `download_nifti` when you want to keep it:
    """)
    return


@app.cell
def _(BrainData, tempfile):
    from nltools.datasets import download_nifti

    _url = "https://neurovault.org/media/images/2099/Neurosynth%20Parcellation_0.nii.gz"
    neurosynth = BrainData(download_nifti(_url, data_dir=tempfile.mkdtemp()))
    neurosynth
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `upload_neurovault` pushes an object to a new or existing collection. It needs an
    access token, generated under your NeuroVault account settings.
    `collection_name` creates a new collection; `collection_id` adds to one you
    already have. `img_type` and `img_modality` are required, and anything else you
    pass is forwarded as image metadata — as are the columns of `.X`, with the row
    index used as each image's name.

    The call below is not run when these docs are built, because it would write to a
    live public repository.

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
