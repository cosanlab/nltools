# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.0.2",
#     "nibabel",
#     "nilearn==0.13.1",
#     "scikit-learn",
#     "scipy",
#     "pandas",
#     "polars",
#     "seaborn",
#     "matplotlib",
#     "joblib>=1.5.3",
#     "huggingface-hub",
#     "pynv",
#     "ipyniivue",
#     "ipywidgets",
#     "pyodide-http; sys_platform == 'emscripten'",
# ]
# ///
# BrainData basics — runs entirely in the browser via marimo + Pyodide.
# Source of truth for the docs tutorial; exported to WASM by
# scripts/build_marimo_wasm.py. `nltools` itself is injected at build time
# (file:// wheel for --execute) and micropip-installed from a hosted URL in
# the browser; see the IN_WASM setup cells below.

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # BrainData Basics

        The `BrainData` class is the core data structure in `nltools` for working with
        neuroimaging data. It stores data as 2D arrays (images x voxels) for efficient
        computation, automatically handles resampling to standard MNI space (default),
        and supports standard Python operations like indexing, arithmetic, and iteration.

        /// admonition | Running live in your browser
        This page **is** a running notebook. The cells below execute in a Pyodide kernel
        inside the page — no install, no server. The first load boots the kernel and
        downloads the scientific stack + example data, which takes a minute; it's cached
        for later visits. Edit any cell and re-run to explore.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import sys

    IN_WASM = sys.platform == "emscripten"
    return (IN_WASM,)


@app.cell(hide_code=True)
async def _(IN_WASM):
    # In-browser only: install the nltools dev wheel. The PyPI stack comes from the
    # PEP 723 header (marimo micropips it automatically under WASM); nltools is not on
    # PyPI at this version, so we install the build-hosted wheel by absolute URL.
    # This cell runs in the Pyodide *web worker*, where js.location is the worker
    # script URL — resolve the wheel against the shared origin, not location.href.
    if IN_WASM:
        import micropip
        import js

        _ = await micropip.install(
            js.location.origin + "__NLTOOLS_WHEEL_URL__", deps=False
        )
    return


@app.cell(hide_code=True)
async def _(IN_WASM):
    # In-browser only: pre-seed the HF-hosted resources into the IDBFS cache so the
    # synchronous fetch_resource()/fetch_pain() calls below hit the cache instead of
    # doing (unsupported) sync HTTP. Persists across reloads via IndexedDB.
    if IN_WASM:
        from nltools.datasets import PAIN_RESOURCES
        from nltools.templates import seed_resources

        _ = await seed_resources(
            [
                "default/2mm-MNI152-2009fsl-mask.nii.gz",
                "default/2mm-MNI152-2009fsl-brain.nii.gz",
                "default/2mm-MNI152-2009fsl-T1.nii.gz",
                *PAIN_RESOURCES,
            ]
        )
    return


@app.cell
def _():
    from nltools import BrainData

    # Empty brain
    BrainData()
    return (BrainData,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Loading data

        You pass a file path, a `nilearn`/`nibabel` image, a file URL, or lists of any of
        those to `BrainData()` — it loads and resamples to MNI space if needed, e.g.
        `BrainData('myfile.nii.gz')`.

        To keep things simple we use one of the included datasets. `fetch_pain()`
        downloads a pain-perception study (Chang et al., 2015): 28 subjects x 3
        conditions = 84 images.
        """
    )
    return


@app.cell
def _():
    from nltools.datasets import fetch_pain

    brains = fetch_pain()
    return (brains,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "The `BrainData` repr shows the shape (images x voxels) and whether metadata "
        "`polars` DataFrames (X, Y) are attached."
    )
    return


@app.cell
def _(brains):
    brains
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Access the underlying data as a numpy array with the `.data` attribute:")
    return


@app.cell
def _(brains):
    brains.data.shape  # (images, voxels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        `BrainData` also stores metadata as `polars` DataFrames on `.X` and `.Y`:

        - **X**: design matrix / covariates for modeling
        - **Y**: outcome variables or labels
        """
    )
    return


@app.cell
def _(brains):
    # The pain dataset ships metadata in X
    brains.X.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Saving data

        `BrainData` saves as NIfTI (`.nii.gz`) or HDF5 (`.h5`). HDF5 preserves metadata
        (X, Y) and masks and produces smaller files:

        ```python
        brains.write("data.nii.gz")   # NIfTI
        brains.write("data.h5")       # HDF5, with X/Y/mask/etc.
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Indexing and slicing

        `BrainData` supports standard Python-style indexing, and all indexing preserves
        the X/Y metadata.
        """
    )
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
    print(f"Sliced: {first_five.shape}")
    return


@app.cell
def _(brains):
    # List indexing
    selected = brains[[0, 10, 20, 30]]
    print(f"Selected: {selected.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Boolean indexing filters images by computed properties:")
    return


@app.cell
def _(brains):
    # Filter images whose global mean exceeds twice their own global mean
    # (illustrative boolean-mask indexing)
    _global_mean = brains.mean(axis=1)
    _keep = _global_mean > _global_mean.mean()
    high_intensity = brains[_keep]
    print(f"Images kept: {len(high_intensity)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Use `.append()` to concatenate `BrainData` objects:")
    return


@app.cell
def _(brains):
    # Append one image to another
    brains[0].append(brains[1]).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Arithmetic operations

        `BrainData` supports element-wise arithmetic with scalars and other `BrainData`
        objects.
        """
    )
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
    mo.md(
        r"""
        ## Statistical operations

        `BrainData` exposes many statistical methods that reduce across images
        (`axis=0`) or across voxels (`axis=1`).
        """
    )
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
    mo.md("Threshold by absolute value or percentile, optionally binarizing for a mask:")
    return


@app.cell
def _(brains):
    # Keep only voxels in the top 5%
    brains.mean().threshold(upper="95%").plot()
    return


@app.cell
def _(brains):
    # Binarize for use as a mask
    _binary_mask = brains.mean().threshold(upper="95%", binarize=True)
    print(f"Mask voxels: {_binary_mask.data.sum():.0f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Masking

        Use `apply_mask` to restrict data to a region of interest.
        """
    )
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
    mo.md(
        r"""
        ## Visualization

        `BrainData.plot()` supports several visualization types via the `method`
        argument. Most wrap [`nilearn.plotting`](https://nilearn.github.io/dev/modules/plotting.html),
        so you can always drop down to `BrainData.to_nifti()` and call nilearn directly.

        ### Glass brain (default)
        """
    )
    return


@app.cell
def _(masked_data):
    masked_data.plot(title="Mean Activation")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Slices")
    return


@app.cell
def _(masked_data):
    # Default: all views
    masked_data.plot(method="slices")
    return


@app.cell
def _(masked_data):
    # Only the Z view
    masked_data.plot(method="slices", view="z")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Surface & flat-map")
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
    mo.md(
        r"""
        ### Timeseries & voxel distribution

        For multi-image `BrainData`, plot the mean signal over images; `histogram` shows
        the voxel-intensity distribution.
        """
    )
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
    mo.md(
        r"""
        ### Interactive viewer

        `BrainData.iplot()` returns an interactive [niivue](https://niivue.com) viewer
        built on a WebGL `ipyniivue.NiiVue` widget: a threshold slider stacked above the
        viewer, with the stat-map colorbar shown. Drag the slider (or right-drag on the
        image) to window the map live; scroll through slices, scrub 4D frames, render in
        3D, and overlay nltools atlases with hover-to-label. It needs a live kernel — so
        it renders right here in this notebook.

        The default (`controls=True`) needs the optional `ipywidgets` dependency
        (`pip install 'nltools[interactive_plots]'`). Pass `controls=False` for the bare
        `NiiVue`, and `colorbar=False` to hide the colorbar.
        """
    )
    return


@app.cell
def _(masked_data):
    # Bare NiiVue viewer (no ipywidgets needed) — 3D ortho view of the stat map
    masked_data.iplot(controls=False)
    return


if __name__ == "__main__":
    app.run()
