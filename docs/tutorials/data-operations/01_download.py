# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Basic data operations — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Basic Data Operations

    Most of nltools is built around `BrainData`. It holds imaging data as an
    images-by-voxels matrix: one row per image, one column per in-mask voxel. That
    shape is what makes the class feel like a dataframe — you index it, slice it, do
    arithmetic on it, and iterate over it with plain Python.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download a dataset

    `fetch_pain()` retrieves the pain dataset from
    [Chang et al., 2015](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002180):
    28 subjects with three beta images each, at low, medium and high thermal pain.
    The files are cached locally on first use and loaded straight into a `BrainData`.
    """)
    return


@app.cell
def _():
    from nltools.datasets import fetch_pain

    data = fetch_pain()
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The image metadata comes with it, on `.X`, as a `polars` DataFrame with one row
    per image:
    """)
    return


@app.cell
def _(data):
    data.X.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load your own files

    A NIfTI file loads by path, and many files load together as a list. The grid
    comes from the file's own affine: nltools matches it to the closest bundled
    MNI template, and resamples only when the match is not exact. The active
    brain space is the default for data that has no grid of its own, not an
    override for data that does. Pass `mask=` to pin a specific grid — see
    [Brain Space and Resolution](03_brain_space.md).

    ```python
    from nltools.data import BrainData

    one = BrainData("sub-01_pain-high.nii.gz")
    many = BrainData(["sub-01_pain-high.nii.gz", "sub-02_pain-high.nii.gz"])
    remote = BrainData("https://neurovault.org/media/images/2099/some_map.nii.gz")
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic operations

    `len()` is the number of images:
    """)
    return


@app.cell
def _(data):
    len(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `.shape` is images by voxels:
    """)
    return


@app.cell
def _(data):
    data.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Index with integers, lists of integers, slices, or boolean arrays:
    """)
    return


@app.cell
def _(data):
    data[[1, 6, 2]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Reduce across images to get one value per voxel:
    """)
    return


@app.cell
def _(data):
    data.mean()
    return


@app.cell
def _(data):
    data.std()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Methods chain, so the mean of 84 images is a single image:
    """)
    return


@app.cell
def _(data):
    data.mean().shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Two `BrainData` objects add and subtract voxelwise:
    """)
    return


@app.cell
def _(data):
    data[1] + data[2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Scalars broadcast over every voxel — here, add 10 and scale by 2:
    """)
    return


@app.cell
def _(data):
    (data + 10) * 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `.copy()` gives you an independent object:
    """)
    return


@app.cell
def _(data):
    data.copy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `.to_nifti()` converts back to a nibabel image — a 3D volume for one image, 4D
    for a stack — which is how you hand data to
    [nilearn](https://nilearn.github.io) or any other toolbox:
    """)
    return


@app.cell
def _(data):
    data.to_nifti().shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `.append()` concatenates along the image axis:
    """)
    return


@app.cell
def _(data):
    data[:2].append(data[4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `.write()` saves to NIfTI or, with an `.h5` extension, to HDF5 — which is
    smaller and keeps `.X`, `.Y` and the mask:
    """)
    return


@app.cell
def _(data):
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as _tmpdir:
        _path = os.path.join(_tmpdir, "pain_subset.nii.gz")
        data[:3].write(_path)
        print(f"{os.path.getsize(_path) / 1e6:.1f} MB written to {_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Images are iterable, so a comprehension gives you one value per image:
    """)
    return


@app.cell
def _(data):
    [image.mean() for image in data[:5]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting

    Convert to nibabel and any nilearn plot works:
    """)
    return


@app.cell
def _(data):
    from nilearn.plotting import plot_glass_brain

    _glass = plot_glass_brain(data.mean().to_nifti())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `.plot()` is the built-in shortcut, and draws a glass brain by default:
    """)
    return


@app.cell
def _(data):
    data.mean().plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Calling it on a stack draws one figure per image. `limit` caps how many, and
    defaults to 3 so that an 84-image object does not silently produce 84 figures;
    raise it, or index first, when you want more:
    """)
    return


@app.cell
def _(data):
    data[:4].plot(limit=4)
    return


if __name__ == "__main__":
    app.run()
