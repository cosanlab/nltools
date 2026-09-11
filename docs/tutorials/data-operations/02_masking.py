# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Masking — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Masking

    A mask restricts a `BrainData` object to the voxels you care about. This
    tutorial builds masks three ways — a sphere around a coordinate, a
    parcellation split into its regions, and a thresholded statistic map — then
    uses them to summarize data and to paint per-region results back onto the
    brain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load data

    We use the pain dataset throughout. The `joblib` cache keeps the docs build
    from reloading 84 images on every run; you can call `fetch_pain()` directly.
    """)
    return


@app.cell
def _():
    from joblib import Memory

    from nltools.datasets import fetch_pain

    memory = Memory(".tutorial-cache", verbose=0)

    data = memory.cache(fetch_pain)()
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A spherical mask

    `create_sphere` draws binary spheres. Centers are MNI millimeter coordinates
    and `radius` is in millimeters, so the same request covers the same physical
    volume whatever grid you are on. `apply_mask` then keeps only the voxels
    inside it.
    """)
    return


@app.cell
def _(data):
    from nltools.mask import create_sphere

    sphere = create_sphere([0, 0, 0], radius=30)
    masked_data = data.apply_mask(sphere)
    masked_data.mean().plot()
    return masked_data, sphere


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Masking drops voxels rather than zeroing them, so the masked object is
    narrower than the original:
    """)
    return


@app.cell
def _(data, masked_data):
    print(f"whole brain: {data.shape}")
    print(f"30 mm sphere: {masked_data.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Average within a region

    `extract_roi` collapses each region of a mask to one number per image. With a
    single binary region you get one value per image — the mean signal in that
    sphere for each of the 84 images. The mask is resampled onto the object's own
    grid first, so it does not have to match resolutions.
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
    ## Expand and collapse a parcellation

    A parcellation is one image whose voxel values are integer region IDs. nltools
    ships several; here is a 50-region whole-brain parcellation, fetched from the
    package's data repository.
    """)
    return


@app.cell
def _():
    from nltools.data import BrainData
    from nltools.templates import fetch_resource

    parcellation = BrainData(
        fetch_resource("masks/default/2mm-MNI152-2009fsl-k50.nii.gz")
    )
    parcellation.plot()
    return BrainData, parcellation


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `expand_mask` turns those IDs into a stack of 50 binary masks, one per region:
    """)
    return


@app.cell
def _(parcellation):
    from nltools.mask import expand_mask

    regions = expand_mask(parcellation)
    print(regions.shape)
    regions[:3].plot()
    return (regions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `collapse_mask` is the inverse: it folds a stack of binary masks back into one
    labeled image, numbering the regions in stack order and dropping any overlap.
    """)
    return


@app.cell
def _(regions):
    from nltools.mask import collapse_mask

    collapse_mask(regions).plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Masks from a statistic map

    `threshold` cuts a map at an absolute value or a percentile. Here we average
    the high-pain images and keep the tails outside the middle 95%.
    """)
    return


@app.cell
def _(data):
    high = data[data.X["PainLevel"] == 3].mean()
    high.threshold(lower="2.5%", upper="97.5%").plot()
    return (high,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `binarize=True` turns the survivors into a mask of ones:
    """)
    return


@app.cell
def _(high):
    high.threshold(lower="2.5%", upper="97.5%", binarize=True).plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `regions` goes further and splits a thresholded map into its spatially
    contiguous blobs, one image per blob. Pass `limit` to draw them all:
    """)
    return


@app.cell
def _(high):
    blobs = high.threshold(lower="2.5%", upper="97.5%").regions()
    print(f"{len(blobs)} contiguous regions")
    blobs.plot(limit=len(blobs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze regions, then map the answer back

    Masks are most useful as a round trip: summarize each region, run an analysis
    over regions, and paint the result back into a brain image.

    Here we compute a linear pain contrast per subject, correlate the 50 regions
    across subjects, threshold that correlation structure into a graph, and map
    each region's degree back onto the brain.
    """)
    return


@app.cell
def _(BrainData, data):
    import numpy as np

    # High minus low pain, one contrast image per subject
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
    `extract_roi` on a labeled parcellation gives regions by images — the profile
    of each region across the 28 subjects:
    """)
    return


@app.cell
def _(contrast, parcellation):
    region_profiles = contrast.extract_roi(parcellation)
    print(region_profiles.shape)  # (regions, subjects)
    return (region_profiles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Correlation distance between those profiles is a 50-node `Adjacency`.
    Thresholding it keeps only the region pairs that covary across subjects:
    """)
    return


@app.cell
def _(region_profiles):
    from sklearn.metrics import pairwise_distances

    from nltools.data import Adjacency

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
    `to_graph` hands the thresholded matrix to `networkx`, where any graph metric
    is available. Degree counts how many regions each region is tied to:
    """)
    return


@app.cell
def _(connected, np):
    graph = connected.to_graph()
    degree = np.array([d for _, d in sorted(graph.degree())])
    print(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"degree range: {degree.min()}-{degree.max()}")
    return (degree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `roi_to_brain` writes one value per region back into the expanded masks, giving
    a brain image of degree centrality:
    """)
    return


@app.cell
def _(degree, regions):
    from nltools.mask import roi_to_brain

    roi_to_brain(degree, regions).plot(title="Degree centrality")
    return


if __name__ == "__main__":
    app.run()
