# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# NeuroVault I/O — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # NeuroVault I/O

    [NeuroVault](https://neurovault.org) is a public repository of unthresholded
    statistical maps. nltools can pull a whole collection, a single image by URL,
    and push your own maps back up.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download a collection

    `fetch_neurovault_collection` takes a collection ID and returns the image
    metadata and the local file paths. Files land in nilearn's data directory
    unless you pass `data_dir`, and are reused on later calls.
    """)
    return


@app.cell
def _():
    from nltools.datasets import fetch_neurovault_collection

    metadata, files = fetch_neurovault_collection(2099, verbose=0)
    print(f"{len(files)} images")
    metadata.select("id", "name", "map_type", "modality").head()
    return files, metadata


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Hand both to `BrainData` and the metadata rides along on `.X`, one row per
    image:
    """)
    return


@app.cell
def _(files, metadata):
    from nltools.data import BrainData

    collection = BrainData(files, X=metadata)
    collection
    return BrainData, collection


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Collection 2099 is a three-image parcellation set, so all three fit under
    `plot`'s default limit of 3:
    """)
    return


@app.cell
def _(collection):
    collection.plot(limit=len(collection))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download a single image

    `download_nifti` fetches one image by URL and returns the path it wrote.
    Without `data_dir` it writes to the working directory, so give it a location
    you control. `BrainData` also accepts a URL directly, which saves a line but
    drops the file in a temporary directory your system will eventually clear —
    use `download_nifti` when you want to keep it.
    """)
    return


@app.cell
def _(BrainData):
    import tempfile

    from nltools.datasets import download_nifti

    _url = "https://neurovault.org/media/images/2099/Neurosynth%20Parcellation_0.nii.gz"
    parcellation_path = download_nifti(_url, data_dir=tempfile.mkdtemp())
    parcellation = BrainData(parcellation_path)
    parcellation
    return (parcellation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    From there it is an ordinary `BrainData`, so any nilearn plot works:
    """)
    return


@app.cell
def _(parcellation):
    from nilearn.plotting import plot_glass_brain

    _glass = plot_glass_brain(parcellation.to_nifti())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Upload to NeuroVault

    `upload_neurovault` pushes a `BrainData` object to a new or existing
    collection. It needs an access token, which you generate under your NeuroVault
    account settings. `collection_name` creates a new collection;
    `collection_id` adds to one you already have. `img_type` and `img_modality`
    are required, and anything else you pass is forwarded as image metadata — as
    are the columns of `.X`, with the row index used as each image's name.

    The cell below is not run when these docs are built, because it would write to
    a live public repository.

    ```python
    parcellation.upload_neurovault(
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
