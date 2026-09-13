# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Working with Adjacency — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Working with Adjacency

    `Adjacency` holds square matrices over a set of nodes: similarity and
    distance matrices, functional connectivity, directed graphs. A symmetric
    matrix is stored as its strict upper triangle and rebuilt on demand, and one
    object can hold a stack of matrices — one per subject, region or timepoint.
    Most of its methods mirror `BrainData`'s. Representational similarity
    analysis is the archetypal use of these matrices and has its own tutorial.

    ## Create one

    The constructor takes a square matrix, an upper-triangle vector, a `polars`
    or `pandas` DataFrame, a `.csv` or `.h5` path, or a list of any of those to
    stack. `matrix_type` declares what the values mean: `'similarity'` (higher
    is more alike), `'distance'` (higher is further apart), or `'directed'`
    (asymmetric, stored in full). `labels` names the nodes and travels with the
    object. A matrix declared symmetric has to be symmetric, so noise is
    symmetrized before it is added, and the diagonal is discarded either way.
    The example below has three groups of four nodes, connected within group at
    strengths 1, 2 and 3 and unconnected across groups:
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.linalg import block_diag

    from nltools.data import Adjacency

    def symmetric_noise(rng, scale):
        noise = rng.standard_normal((12, 12)) * scale
        return (noise + noise.T) / 2

    m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
    m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
    m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))

    blocks = Adjacency(
        m1 + 2 * m2 + 3 * m3 + symmetric_noise(np.random.default_rng(0), 0.1),
        matrix_type="similarity",
        labels=[f"{group}.{node}" for group in ("C1", "C2", "C3") for node in range(4)],
    )

    print(blocks)
    return Adjacency, blocks, m1, m2, m3, np, symmetric_noise


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Printing reports the logical shape, the size of the metadata frame `Y`,
    symmetry, and the matrix type. `.shape` is that logical shape —
    `(n_matrices, n_nodes, n_nodes)` for a stack — while `.vector_shape` is what
    is stored, the n(n-1)/2 entries above the diagonal. `squareform()` rebuilds
    the square matrix as a plain `numpy` array with a zero diagonal.

    `BrainData.distance()` returns an `Adjacency` too: every image against every
    other under a `scikit-learn` metric. Twenty-one images from the pain dataset
    — seven subjects at low, medium and high intensity — give a 21-node distance
    matrix, and `joblib` keeps the build from redownloading them:
    """)
    return


@app.cell
def _(blocks):
    print(f"shape:        {blocks.shape}  ({blocks.n_nodes} nodes)")
    print(f"vector_shape: {blocks.vector_shape}")
    print(f"squareform(): {blocks.squareform().shape} array")
    return


@app.cell
def _():
    from joblib import Memory

    from nltools.datasets import fetch_pain

    memory = Memory(".tutorial-cache", verbose=0)
    subset = memory.cache(fetch_pain)()[:21]
    pain_distance = subset.distance(metric="correlation")
    pain_distance.labels = [f"s{subject}" for subject in subset.X["SubjectID"]]

    print(pain_distance)
    return pain_distance, subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot

    `plot()` draws a `seaborn` heatmap of the square form, using `labels` for
    the ticks when they are set. Keywords the signature does not name go to
    `seaborn.heatmap`, so `cmap`, `vmin` and `vmax` work. The three planted
    groups are the brighter diagonal squares below. The pain matrix holds
    distances, so its blocks are dark instead: each image is close to the same
    subject's other two images and far from everyone else's.
    """)
    return


@app.cell
def _(blocks):
    blocks.plot()
    return


@app.cell
def _(pain_distance):
    pain_distance.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Threshold and transform

    `threshold` zeroes part of the range and keeps the rest. `upper=` keeps
    values at or above the cutoff, `lower=` keeps values at or below it, and
    giving both keeps the two tails outside the band. Read each as the edge of
    the region you keep, not the region you drop: on a similarity matrix the
    interesting edges are the high ones, so `upper=` is what you want, and on a
    distance matrix the close pairs are the low ones, so `lower=` is. A string
    ending in `%` is a percentile of the stored values; `binarize=True` turns
    whatever survives into ones.
    """)
    return


@app.cell
def _(blocks, np):
    strong = blocks.threshold(upper=0.5)
    top_decile = blocks.threshold(upper="90%")
    binary = blocks.threshold(upper=0.5, binarize=True)

    print(f"upper=0.5    keeps {(strong.data != 0).sum()} of {blocks.vector_shape[0]}")
    print(f"upper='90%'  keeps {(top_decile.data != 0).sum()}")
    print(f"binarized values: {np.unique(binary.data)}")
    return (binary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `distance_to_similarity` converts a distance matrix into a similarity one:
    `metric='correlation'` returns `1 - d`, undoing the correlation distance
    above, and `metric='euclidean'` returns `exp(-beta * d / sd(d))`. There is
    no method for the other direction — a correlation similarity goes back as
    `1 - s`, with `matrix_type` set on the constructor. Correlations are also
    not on an interval scale, so averaging them raw understates the large ones;
    Fisher's r-to-z fixes that, and `r_to_z()` and `z_to_r()` apply the two
    directions elementwise to the stored triangle:
    """)
    return


@app.cell
def _(np, pain_distance):
    pain_similarity = pain_distance.distance_to_similarity(metric="correlation")
    pain_z = pain_similarity.r_to_z()

    _dist, _sim, _z = pain_distance.data, pain_similarity.data, pain_z.data

    print(f"distance   [{_dist.min():.2f}, {_dist.max():.2f}]")
    print(f"similarity [{_sim.min():.2f}, {_sim.max():.2f}]")
    print(f"z          [{_z.min():.2f}, {_z.max():.2f}]")
    print(f"round trip: {np.allclose(pain_z.z_to_r().data, _sim)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arithmetic and statistics

    `+`, `-`, `*` and `/` work elementwise against another `Adjacency` or a
    scalar. Two matrices have to agree on node count and on node labels and
    their order — nothing else would catch two matrices whose nodes are ordered
    differently. `mean`, `median`, `std` and `sum` collapse the object: on a
    single matrix they return one number over the stored edges, and on a stack
    `axis=0` averages across matrices into an `Adjacency` while `axis=1`
    collapses each matrix into one number. Subtracting the first group's mask
    leaves the other two groups and the noise:
    """)
    return


@app.cell
def _(Adjacency, blocks, m1):
    residual = blocks - Adjacency(m1, matrix_type="similarity", labels=blocks.labels)

    print(f"blocks:              mean {blocks.mean():.2f}, sd {blocks.std():.2f}")
    print(f"first group removed: mean {residual.mean():.2f}, sd {residual.std():.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A list of matrices makes a stack, and `append` adds to an existing one.
    Fifteen matrices stand in for fifteen subjects: the first group is connected
    in five of them, absent in the next five and connected again in the last
    five, under noise five times larger than above. Averaging across the stack
    recovers the planted group at two thirds of its strength, since a third of
    the matrices do not carry it. `ttest` then tests every edge against zero,
    returning `'mean'`, `'t'`, `'z'` and `'p'` as four `Adjacency` maps, and
    `permutation=True` swaps the parametric p-value for a sign-flip one. The
    maps are unthresholded: 66 edges is 66 tests, so correct for them.
    """)
    return


@app.cell
def _(Adjacency, blocks, m1, np, symmetric_noise):
    stack_rng = np.random.default_rng(1)
    stack = Adjacency(
        [m1 + symmetric_noise(stack_rng, 0.5) for _ in range(5)]
        + [symmetric_noise(stack_rng, 0.5) for _ in range(5)]
        + [m1 + symmetric_noise(stack_rng, 0.5) for _ in range(5)],
        matrix_type="similarity",
        labels=blocks.labels,
    )

    print(stack)
    return (stack,)


@app.cell
def _(stack):
    stack.mean().plot()
    return


@app.cell
def _(m1, np, stack):
    group = stack.ttest()
    _p = group["p"].data

    print(f"{(_p < 0.05).sum()} of {len(_p)} edges at p < .05")
    print(f"the planted group is {int(m1[np.triu_indices(12, 1)].sum())} edges")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regression

    `regress` answers two different questions and the design decides which.
    `tail` is keyword-only and takes `2` (two-tailed, the default) or `1`.

    Pass an `Adjacency` and this matrix is decomposed into a weighted sum of the
    predictor matrices: edges are the observations and matrices the predictors.
    The three group masks recover the strengths the data was built with, and
    since predictor and response must agree on node ordering, the design carries
    the same labels. Pass a `DesignMatrix` and each edge is regressed across the
    stack instead — the adjacency analogue of a mass-univariate imaging
    analysis, with the same multiple-comparisons problem. That result holds one
    `Adjacency` map per predictor, and thresholding the t map leaves the
    on-off-on group:
    """)
    return


@app.cell
def _(Adjacency, blocks, m1, m2, m3):
    design = Adjacency([m1, m2, m3], matrix_type="similarity", labels=blocks.labels)
    block_fit = blocks.regress(design)

    print(f"beta: {block_fit['beta'].round(2)}")
    print(f"t:    {block_fit['t'].round(1)}")
    print(f"df:   {block_fit['df']}")
    return


@app.cell
def _(np, stack):
    from nltools.data import DesignMatrix

    on_off_on = DesignMatrix(
        np.array([1] * 5 + [0] * 5 + [1] * 5).reshape(-1, 1),
        columns=["on"],
        sampling_freq=1.0,
    )
    edge_fit = stack.regress(on_off_on)

    edge_fit["t"].threshold(upper=2).plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multidimensional scaling

    `plot_mds` lays a distance matrix out in two or three dimensions. It needs a
    single distance matrix; metric scaling is the default and `metric_mds=False`
    asks for non-metric. Node names come from `labels` and `labels_color` takes
    one color per node, here each image's intensity.

    The images land in seven tight clusters, one per subject; intensity, the
    thing the experiment manipulated, does not organize the picture at all. The
    distances agree: two images of one subject are more than three times closer
    than two of different subjects, and across subjects sharing an intensity
    buys nothing:
    """)
    return


@app.cell
def _(pain_distance, subset):
    intensity_color = {"low": "#4c72b0", "medium": "#dd8452", "high": "#c44e52"}

    pain_distance.plot_mds(
        labels_color=[intensity_color[level] for level in subset.X["PainIntensity"]]
    )
    return


@app.cell
def _(np, pain_distance, subset):
    _rows, _cols = np.triu_indices(pain_distance.n_nodes, 1)
    _edges = pain_distance.squareform()[_rows, _cols]
    _subject = subset.X["SubjectID"].to_numpy()
    _level = subset.X["PainIntensity"].to_numpy()
    _across = _subject[_rows] != _subject[_cols]
    _same_level = _level[_rows] == _level[_cols]

    print(f"same subject:         {_edges[~_across].mean():.2f}")
    print(f"different subject:    {_edges[_across].mean():.2f}")
    print(f"  and same intensity: {_edges[_across & _same_level].mean():.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graphs

    `to_graph` hands a single matrix to `networkx` — a `Graph`, or a `DiGraph`
    for a directed matrix — with `labels` as the node names, so give the nodes
    distinct labels or two of them will merge into one. Every `networkx` metric
    and layout applies from there, and the binarized matrix from earlier is
    three disconnected cliques of four nodes, so every node has degree 3. A
    graph measure is one number per node, and when the nodes are brain regions
    `roi_to_brain` writes those numbers back into the parcellation as an image;
    the BrainData tutorial runs that round trip end to end.
    """)
    return


@app.cell
def _(binary):
    import networkx as nx

    clique_graph = binary.to_graph()
    _degrees = sorted(set(dict(clique_graph.degree()).values()))

    print(f"{len(clique_graph)} nodes, {clique_graph.number_of_edges()} edges")
    print(f"degrees: {_degrees}")
    return clique_graph, nx


@app.cell
def _(clique_graph, nx):
    nx.draw_circular(clique_graph, node_color="lightsteelblue", with_labels=True)
    return


if __name__ == "__main__":
    app.run()
