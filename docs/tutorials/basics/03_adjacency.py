# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Only marimo + the emscripten HTTP shim load from this header. nltools and its whole
#     # runtime stack are micropip-installed by the IN_WASM setup cell (UNPINNED, so Pyodide's
#     # bundled builds win) — see that cell. Listing the stack here too makes marimo's header
#     # auto-install redundantly pull unpinned latest scikit-learn/scipy/pandas/matplotlib,
#     # which drag in `packaging>=26` (absent in Pyodide 0.27.7) and error out.
#     "marimo",
#     "pyodide-http; sys_platform == 'emscripten'",
# ]
# ///
# Adjacency basics — runs entirely in the browser via marimo + Pyodide.
# Source of truth for the docs tutorial; exported to WASM by
# scripts/build_marimo_wasm.py. `nltools` is micropip-installed in the browser from a
# build-hosted wheel URL by the IN_WASM setup cell below.

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
        # Adjacency Basics

        The `Adjacency` class represents connectivity or similarity matrices. It stores
        data efficiently as the upper-triangle vector and reconstructs the full square
        matrix on demand. Common use cases:

        - **Functional connectivity** — correlations between ROI timeseries
        - **Representational similarity** — pattern-similarity matrices (RSA)
        - **Behavioral similarity** — subject-level similarity from traits or responses

        It supports two matrix types: `"similarity"` (higher = more similar) and
        `"distance"` (higher = more dissimilar).

        /// admonition | Running live in your browser
        This page **is** a running notebook — the cells below execute in a Pyodide kernel
        inside the page. The first load boots the kernel and installs the scientific stack
        plus a small example dataset (about a minute; cached afterwards).
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
    # In-browser only: install nltools + its full runtime stack before any nltools import
    # runs, then hand `wasm_ready` to every nltools-importing cell to force ordering. We
    # can't rely on marimo's PEP 723 header auto-install alone: it races cell execution and
    # marimo never re-runs a cell that already failed with ModuleNotFoundError. Resolve the
    # wheel against the shared worker origin.
    wasm_ready = True
    if IN_WASM:
        import asyncio

        import micropip
        import js

        async def _pip(reqs, **kw):
            # Install packages ONE AT A TIME instead of a single concurrent
            # micropip.install([...]) call. The big concurrent batch download
            # occasionally returns a truncated wheel (BadZipFile); micropip then
            # caches the corrupt bytes so an in-session retry keeps failing — and
            # marimo never re-runs an errored cell, permanently bricking the
            # page. Sequential installs keep peak download concurrency low and
            # sidestep the corruption; a per-package retry still rides out
            # ordinary network blips. (see nltools#455 investigation)
            items = [reqs] if isinstance(reqs, str) else list(reqs)
            for _item in items:
                for _attempt in range(3):
                    try:
                        await micropip.install(_item, **kw)
                        break
                    except Exception:  # noqa: BLE001
                        if _attempt == 2:
                            raise
                        await asyncio.sleep(0.75 * (_attempt + 1))

        # Install the stack UNPINNED so micropip takes Pyodide's bundled builds (pinning to
        # nltools' host versions, e.g. joblib>=1.5.3, fails against Pyodide's bundled
        # joblib). nilearn is the exception: 0.14+ needs packaging>=26 (absent in Pyodide
        # 0.27.7), so pin the last 0.13.x. numpy/scipy/pandas/sklearn/matplotlib come in
        # transitively at their bundled versions.
        await _pip(
            [
                "nibabel",
                "nilearn==0.13.1",
                "seaborn",
                "polars",
                "pynv",
                "huggingface-hub",
                "anywidget",
            ]
        )
        # deps=False installs the wheel without re-checking nltools' own version pins.
        await _pip(
            js.location.origin + "__NLTOOLS_WHEEL_URL__", deps=False
        )
    return (wasm_ready,)


@app.cell(hide_code=True)
async def _(IN_WASM, wasm_ready):
    # In-browser only: pre-seed the MNI templates + pain dataset into the IDBFS cache so
    # the synchronous fetch_pain() below (used in the "From brain data" example) works.
    # `seeded` is threaded into the data-loading cell so fetch_pain() waits for the cache.
    _ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
    seeded = True
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
    return (seeded,)


@app.cell
def _(wasm_ready):
    _ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
    import numpy as np
    import matplotlib.pyplot as plt

    from nltools.data import Adjacency

    return Adjacency, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating Adjacency objects

        ### From a square matrix
        """
    )
    return


@app.cell
def _(Adjacency, np):
    n_nodes = 10

    # A random symmetric matrix with a zero diagonal
    _rng = np.random.default_rng(0)
    random_matrix = _rng.standard_normal((n_nodes, n_nodes))
    random_matrix = (random_matrix + random_matrix.T) / 2
    np.fill_diagonal(random_matrix, 0)

    adj = Adjacency(data=random_matrix, matrix_type="similarity")
    print(adj)
    return adj, n_nodes, random_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### From brain data

        `BrainData.distance()` computes pairwise distances between brain images and
        returns an `Adjacency`:
        """
    )
    return


@app.cell
def _(seeded, wasm_ready):
    _ = wasm_ready, seeded  # wheel installed + resources seeded first (WASM)
    from nltools.datasets import fetch_pain

    data = fetch_pain()

    # A subset keeps the pairwise distance quick
    subset = data[:20]
    dist_matrix = subset.distance(metric="correlation")
    print(f"Distance matrix: {dist_matrix.shape}")
    return (dist_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Shape and storage

        `Adjacency` distinguishes the logical shape (a square matrix) from the stored
        vector (its upper triangle):
        """
    )
    return


@app.cell
def _(adj, n_nodes):
    print(f"Logical shape:   {adj.shape}")
    print(f"Number of nodes: {adj.n_nodes}")
    print(f"Vector length:   {adj.vector_shape}")
    print(f"Expected n*(n-1)/2 = {n_nodes * (n_nodes - 1) // 2}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Reconstruct the full matrix with `squareform()`:")
    return


@app.cell
def _(adj, np):
    square = adj.squareform()
    print(f"Square matrix: {square.shape}")
    print(f"Symmetric:     {np.allclose(square, square.T)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Visualization

        ### Heatmap
        """
    )
    return


@app.cell
def _(adj, plt):
    adj.plot()
    plt.gca().set_title("Random Similarity Matrix")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### With labels")
    return


@app.cell
def _(Adjacency, n_nodes, plt, random_matrix):
    _roi_names = [f"ROI_{i}" for i in range(n_nodes)]
    adj_labeled = Adjacency(
        data=random_matrix, matrix_type="similarity", labels=_roi_names
    )
    adj_labeled.plot()
    plt.gca().set_title("Labeled Matrix")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "### MDS plot\n\nMultidimensional scaling lays out the structure of a distance "
        "matrix in 2D:"
    )
    return


@app.cell
def _(dist_matrix, plt):
    dist_matrix.plot_mds(n_components=2, figsize=(6, 5))
    plt.gca().set_title("MDS of Image Distances")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Thresholding

        Remove weak connections by absolute value or percentile, and optionally binarize:
        """
    )
    return


@app.cell
def _(adj, np):
    # Absolute threshold: keep edges > 0.3
    thresh = adj.threshold(lower=0.3)
    print(f"Edges above 0.3: {(thresh.data > 0).sum()} / {len(thresh.data)}")

    # Percentile threshold: keep the top 10%
    thresh_pct = adj.threshold(lower="90%")
    print(f"Top 10% edges:   {(thresh_pct.data > 0).sum()}")

    # Binarize
    binary = adj.threshold(lower=0.3, binarize=True)
    print(f"Binary values:   {np.unique(binary.data)}")
    return binary, thresh


@app.cell
def _(adj, binary, plt, thresh):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    adj.plot(axes=_axes[0])
    _axes[0].set_title("Original")
    thresh.plot(axes=_axes[1])
    _axes[1].set_title("Thresholded (> 0.3)")
    binary.plot(axes=_axes[2])
    _axes[2].set_title("Binarized")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Statistics

        ### Summary statistics
        """
    )
    return


@app.cell
def _(adj):
    print(f"Mean:   {adj.mean():.4f}")
    print(f"Std:    {adj.std():.4f}")
    print(f"Median: {adj.median():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "### Comparing two matrices\n\n`similarity()` tests whether two matrices are "
        "related, with permutation-based inference:"
    )
    return


@app.cell
def _(Adjacency, np):
    # Two related matrices
    _rng = np.random.default_rng(42)
    _m1 = _rng.standard_normal((15, 15))
    _m1 = (_m1 + _m1.T) / 2
    np.fill_diagonal(_m1, 0)

    _m2 = _m1 + _rng.standard_normal((15, 15)) * 0.5
    _m2 = (_m2 + _m2.T) / 2
    np.fill_diagonal(_m2, 0)

    adj1 = Adjacency(_m1, matrix_type="similarity")
    adj2 = Adjacency(_m2, matrix_type="similarity")

    _result = adj1.similarity(adj2, metric="spearman", n_permute=5000)
    print(f"Spearman r = {_result['correlation']:.3f}, p = {_result['p']:.4f}")
    return adj1, adj2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Fisher's r-to-z transform

        When averaging or comparing correlation matrices, apply the Fisher transform first:
        """
    )
    return


@app.cell
def _(Adjacency, np):
    _rng = np.random.default_rng(7)
    _corr_data = np.corrcoef(_rng.standard_normal((8, 50)))
    np.fill_diagonal(_corr_data, 0)
    corr_adj = Adjacency(_corr_data, matrix_type="similarity")

    _z_adj = corr_adj.r_to_z()
    print(f"Original range: [{corr_adj.data.min():.3f}, {corr_adj.data.max():.3f}]")
    print(f"Z-scored range: [{_z_adj.data.min():.3f}, {_z_adj.data.max():.3f}]")

    _r_adj = _z_adj.z_to_r()
    print(f"Round-trip check: {np.allclose(corr_adj.data, _r_adj.data, atol=1e-10)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Arithmetic

        `Adjacency` supports element-wise arithmetic:
        """
    )
    return


@app.cell
def _(adj1, adj2):
    _diff = adj1 - adj2
    _scaled = adj1 * 2
    print(f"Difference mean: {_diff.mean():.4f}")
    print(f"Scaled mean:     {_scaled.mean():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Application: functional connectivity")
    return


@app.cell
def _(Adjacency, np, plt):
    # Five ROI timeseries with a bit of correlation structure
    _rng = np.random.default_rng(0)
    _roi_ts = _rng.standard_normal((100, 5))
    _roi_ts[:, 1] = _roi_ts[:, 0] + _rng.standard_normal(100) * 0.3  # ROI 0-1 correlated
    _roi_ts[:, 4] = _roi_ts[:, 3] + _rng.standard_normal(100) * 0.3  # ROI 3-4 correlated

    _fc_matrix = np.corrcoef(_roi_ts.T)
    np.fill_diagonal(_fc_matrix, 0)

    _roi_labels = ["DLPFC_L", "DLPFC_R", "ACC", "Insula_L", "Insula_R"]
    fc = Adjacency(_fc_matrix, matrix_type="similarity", labels=_roi_labels)
    fc.plot()
    plt.gca().set_title("ROI-to-ROI Functional Connectivity")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Application: representational similarity analysis

        Simulate neural patterns with category structure — faces similar to faces,
        objects similar to objects — and build a representational dissimilarity matrix:
        """
    )
    return


@app.cell
def _(Adjacency, np, plt):
    _rng = np.random.default_rng(1)
    _patterns = _rng.standard_normal((6, 1000))
    _patterns[1] = _patterns[0] + _rng.standard_normal(1000) * 0.2
    _patterns[2] = _patterns[0] + _rng.standard_normal(1000) * 0.2
    _patterns[4] = _patterns[3] + _rng.standard_normal(1000) * 0.2
    _patterns[5] = _patterns[3] + _rng.standard_normal(1000) * 0.2

    _rdm = 1 - np.corrcoef(_patterns)
    np.fill_diagonal(_rdm, 0)

    _labels = ["Face1", "Face2", "Face3", "Object1", "Object2", "Object3"]
    rsa = Adjacency(_rdm, matrix_type="distance", labels=_labels)
    rsa.plot()
    plt.gca().set_title("Representational Dissimilarity Matrix")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "Notice the block-diagonal structure: faces are similar to faces (low "
        "dissimilarity), objects to objects."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Stacking multiple matrices

        Use `append()` to stack matrices (e.g. one per subject) for group-level analysis:
        """
    )
    return


@app.cell
def _(Adjacency, np):
    # Simulate an FC matrix per subject
    _rng = np.random.default_rng(3)
    _matrices = []
    for _ in range(5):
        _ts = _rng.standard_normal((100, 5))
        _ts[:, 1] = _ts[:, 0] + _rng.standard_normal(100) * 0.3
        _m = np.corrcoef(_ts.T)
        np.fill_diagonal(_m, 0)
        _matrices.append(Adjacency(_m, matrix_type="similarity"))

    stacked = _matrices[0]
    for _mat in _matrices[1:]:
        stacked = stacked.append(_mat)

    print(f"Stacked: {len(stacked)} matrices, {stacked.n_nodes} nodes each")

    _group_mean = stacked.mean()
    print(f"Group mean shape: {_group_mean.shape}")

    # One-sample t-test across subjects
    _ttest = stacked.ttest()
    print(f"T-test: {(_ttest['p'].data < 0.05).sum()} significant edges (p < 0.05)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## File I/O")
    return


@app.cell
def _(Adjacency, adj, np):
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as _tmpdir:
        _path = os.path.join(_tmpdir, "adjacency.csv")
        adj.write(_path, method="square")
        print(f"Saved: {adj.shape}")

        _loaded = Adjacency(_path, matrix_type="similarity")
        print(f"Loaded: {_loaded.shape}")
        print(f"Round-trip check: {np.allclose(adj.data, _loaded.data)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        In this tutorial you learned to:

        - **Create** `Adjacency` from square matrices, vectors, or `BrainData.distance()`
        - **Store** as an upper-triangle vector with `squareform()` reconstruction
        - **Visualize** with `plot()` heatmaps and `plot_mds()` for structure
        - **Threshold** by absolute value, percentile, and binarization
        - **Test** with `mean()`, `ttest()`, and `similarity()` permutation tests
        - **Transform** with `r_to_z()` / `z_to_r()` (Fisher transforms)
        - **Stack** with `append()` for group-level analysis

        For a complete representational-similarity workflow, see the **Multivariate
        Pattern Analysis** tutorial, which builds an RDM from real brain patterns and
        compares it to a model.
        """
    )
    return


if __name__ == "__main__":
    app.run()
