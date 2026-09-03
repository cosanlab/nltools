# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# BrainData basics — marimo notebook. Source of truth for the docs page; rendered to MyST by scripts/marimo_to_myst.py.

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
    # BrainData Basics

    The `BrainData` class is the core data structure in `nltools` for working with
    neuroimaging data. It stores data as 2D arrays (images x voxels) for efficient
    computation, automatically handles resampling to standard MNI space (default),
    and supports standard Python operations like indexing, arithmetic, and iteration.
    """)
    return


@app.cell
def _():
    from nltools import BrainData

    # Empty brain
    BrainData()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading data

    You pass a file path, a `nilearn`/`nibabel` image, a file URL, or lists of any of
    those to `BrainData()` — it loads and resamples to MNI space if needed, e.g.
    `BrainData('myfile.nii.gz')`.

    To keep things simple we use one of the included datasets. `fetch_pain()`
    downloads a pain-perception study (Chang et al., 2015): 28 subjects x 3
    conditions = 84 images.
    """)
    return


@app.cell
def _():
    from nltools.datasets import fetch_pain

    brains = fetch_pain()
    return (brains,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The `BrainData` repr shows the shape (images x voxels) and whether metadata "
        "`polars` DataFrames (X, Y) are attached.
    """)
    return


@app.cell
def _(brains):
    brains
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Access the underlying data as a numpy array with the `.data` attribute:
    """)
    return


@app.cell
def _(brains):
    brains.data.shape  # (images, voxels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `BrainData` also stores metadata as `polars` DataFrames on `.X` and `.Y`:

    - **X**: design matrix / covariates for modeling
    - **Y**: outcome variables or labels
    """)
    return


@app.cell
def _(brains):
    # The pain dataset ships metadata in X
    brains.X.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving data

    `BrainData` saves as NIfTI (`.nii.gz`) or HDF5 (`.h5`). HDF5 preserves metadata
    (X, Y) and masks and produces smaller files:

    ```python
    brains.write("data.nii.gz")   # NIfTI
    brains.write("data.h5")       # HDF5, with X/Y/mask/etc.
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Indexing and slicing

    `BrainData` supports standard Python-style indexing, and all indexing preserves
    the X/Y metadata.
    """)
    return


@app.cell
def _(brains):
    # Single image
    brains[0]
    return


@app.cell
def _(brains):
    # Slicing
    first_five = brains[:5]

    # Same voxels, space, etc as brains
    first_five
    return


@app.cell
def _(brains):
    # List indexing
    selected = brains[[0, 10, 20, 30]]

    # The 0th, 10th, 20th, and 30th images
    selected
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Boolean indexing filters images by computed properties:
    """)
    return


@app.cell
def _(brains):
    # Keep the images whose global mean is above the average across images
    # (illustrative boolean-mask indexing)
    _global_mean = brains.mean(axis=1)
    _keep = _global_mean > _global_mean.mean()
    high_intensity = brains[_keep]
    print(f"Images kept: {len(high_intensity)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Use `.append()` to concatenate `BrainData` objects:
    """)
    return


@app.cell
def _(brains):
    # Append one image to another
    brains[0].append(brains[1]).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arithmetic operations

    `BrainData` supports element-wise arithmetic with scalars and other `BrainData`
    objects.
    """)
    return


@app.cell
def _(brains):
    # Addition (scalar, broadcast over every voxel)
    brains + 100
    return


@app.cell
def _(brains):
    # Subtraction of two images → single brain map
    brains[1] - brains[0]
    return


@app.cell
def _(brains):
    # Adding two BrainData objects element-wise
    brains + brains
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statistical operations

    `BrainData` exposes many statistical methods that reduce across images
    (`axis=0`) or across voxels (`axis=1`).
    """)
    return


@app.cell
def _(brains):
    # Mean across all images → single brain map
    brains.mean()
    return


@app.cell
def _(brains):
    # Standard deviation across images → single brain map
    brains.std()
    return


@app.cell
def _(brains):
    # Temporal signal-to-noise ratio, then plot it
    tsnr = brains.mean() / brains.std()
    tsnr.plot()
    return


@app.cell
def _(brains):
    # Standardization / z-scoring across images
    z_scored = brains.standardize(method="zscore")
    print(f"Z-scored mean: {z_scored.mean().data.mean():.6f}")
    print(f"Z-scored std:  {z_scored.std().data.mean():.4f}")
    return


@app.cell
def _(brains):
    # Gaussian spatial smoothing at a given FWHM (mm)
    _smoothed = brains[0].smooth(fwhm=6)
    print(f"Original range: [{brains[0].data.min():.2f}, {brains[0].data.max():.2f}]")
    print(f"Smoothed range: [{_smoothed.data.min():.2f}, {_smoothed.data.max():.2f}]")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Threshold by absolute value or percentile, optionally binarizing for a mask:
    """)
    return


@app.cell
def _(brains):
    # Keep only voxels in the top 5%
    brains.mean().threshold(upper="95%").plot(cmap='Blues')
    return


@app.cell
def _(brains):
    # Binarize for use as a mask
    _binary_mask = brains.mean().threshold(upper="95%", binarize=True)
    print(f"Mask voxels: {_binary_mask.data.sum():.0f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Masking

    Use `apply_mask` to restrict data to a region of interest.
    """)
    return


@app.cell
def _(brains):
    # Mean map, with color bounds captured so later plots stay comparable
    mean_brain = brains.mean()
    vmin, vmax = mean_brain.data.min(), mean_brain.data.max()
    mean_brain.plot(vmin=vmin, vmax=vmax)
    return mean_brain, vmax, vmin


@app.cell
def _(mean_brain):
    # An ROI mask from the top 10% of mean activation
    roi_mask = mean_brain.threshold(upper="90%", binarize=True)
    roi_mask.plot(vmin=0, vmax=1, cmap="gray_r")
    return (roi_mask,)


@app.cell
def _(mean_brain, roi_mask, vmax, vmin):
    # Apply it — voxels outside the mask render transparent
    masked_data = mean_brain.apply_mask(roi_mask)
    masked_data.plot(vmin=vmin, vmax=vmax, cmap="RdBu_r")
    return (masked_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization

    `BrainData.plot()` supports several visualization types via the `method`
    argument. Most wrap [`nilearn.plotting`](https://nilearn.github.io/dev/modules/plotting.html),
    so you can always drop down to `BrainData.to_nifti()` and call nilearn directly.

    ### Glass brain (default)
    """)
    return


@app.cell
def _(masked_data):
    masked_data.plot(title="Mean Activation", cmap='viridis')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Slices
    """)
    return


@app.cell
def _(masked_data):
    # Default: axial (z) slices only
    masked_data.plot(method="slices")
    return


@app.cell
def _(masked_data):
    # Any combination of x/y/z views, one row per axis
    masked_data.plot(method="slices", view="xyz")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Surface & flat-map
    """)
    return


@app.cell
def _(masked_data):
    masked_data.plot_surf(zoom=1.3)
    return


@app.cell
def _(masked_data):
    masked_data.plot_flatmap()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Timeseries & voxel distribution

    For multi-image `BrainData`, plot the mean signal over images; `histogram` shows
    the voxel-intensity distribution.
    """)
    return


@app.cell
def _(brains):
    brains.plot(method="timeseries", figsize=(6, 4))
    return


@app.cell
def _(mean_brain):
    mean_brain.plot(
        method="histogram", title="Voxel Intensity Distribution", figsize=(6, 4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive viewer

    `BrainData.iplot()` returns an interactive [niivue](https://niivue.com) viewer —
    a WebGL `anywidget` that drives `@niivue/niivue` directly: a threshold slider
    stacked above the viewer, with the stat-map colorbar shown. Drag the slider (or
    right-drag on the image) to window the map live; scroll through slices, scrub 4D
    frames, render in 3D, and overlay nltools atlases with hover-to-label. It speaks
    anywidget's standard model API, so it renders in any live kernel (marimo, Jupyter).

    Pass `controls=False` to hide the slider (right-drag windowing still works), and
    `colorbar=False` to hide the colorbar. No `ipywidgets` dependency needed.
    """)
    return


@app.cell
def _(masked_data):
    # Interactive niivue viewer with a threshold slider (an anywidget driving
    # @niivue/niivue directly). It needs a live kernel, so on this static page
    # only a placeholder appears; run the notebook to explore the volume.
    masked_data.iplot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
