# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Adjacency matrices — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Adjacency Matrices

    `Adjacency` holds square matrices over a set of nodes: similarity and distance
    matrices, connectivity, directed graphs. Symmetric matrices are stored as
    their upper triangle and rebuilt on demand, and one object can hold a stack of
    matrices — one per region, subject, or timepoint. Most of its methods mirror
    the ones on `BrainData`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create one

    An `Adjacency` accepts a numpy array, a dataframe, a CSV path, or a list of
    any of those. You also declare the matrix type: `'similarity'` (symmetric,
    typically ones on the diagonal), `'distance'` (symmetric, zeros on the
    diagonal), or `'directed'` (not symmetric, stored in full). Labels are
    optional.

    Here is fake data with three blocks of four nodes each, at signal strengths 1,
    2 and 3. The noise is symmetrized before it is added, because a matrix
    declared symmetric has to be symmetric.
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.linalg import block_diag

    from nltools.data import Adjacency

    rng = np.random.default_rng(0)

    def symmetric_noise(scale):
        noise = rng.standard_normal((12, 12)) * scale
        return (noise + noise.T) / 2

    m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
    m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
    m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))

    blocks = Adjacency(
        (m1 * 1 + m2 * 2 + m3 * 3) + symmetric_noise(0.1),
        matrix_type="similarity",
        labels=["C1"] * 4 + ["C2"] * 4 + ["C3"] * 4,
    )
    blocks
    return Adjacency, blocks, m1, m2, m3, np, symmetric_noise


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `squareform()` rebuilds the full matrix from the stored triangle:
    """)
    return


@app.cell
def _(blocks):
    blocks.squareform().shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `plot()` draws it as a heatmap, where the three blocks are visible as
    brighter squares on the diagonal:
    """)
    return


@app.cell
def _(blocks):
    blocks.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `cluster_summary` averages edges by a grouping variable — here the labels.
    `scope='within'` summarizes edges inside each group; `'between'` summarizes
    edges that cross groups. The three block strengths come back out.
    """)
    return


@app.cell
def _(blocks):
    blocks.cluster_summary(clusters=blocks.labels, scope="within", summary="mean")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regression

    `regress` covers two different questions, and which one you get depends on
    what you pass as the design.

    ### Decomposing one matrix

    Pass an `Adjacency` and the matrix is decomposed into a weighted sum of the
    predictor matrices; edges are the observations and matrices the predictors.
    Using the three blocks as predictors recovers the weights the data was built
    with. Predictor and response must agree on node ordering, so the design
    carries the same labels.
    """)
    return


@app.cell
def _(Adjacency, blocks, m1, m2, m3):
    design = Adjacency([m1, m2, m3], matrix_type="similarity", labels=blocks.labels)
    block_fit = blocks.regress(design)

    print(f"beta:  {block_fit['beta'].round(3)}")
    print(f"t:     {block_fit['t'].round(1)}")
    print(f"df:    {block_fit['df']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regression at every edge

    Pass a `DesignMatrix` instead and each edge gets its own regression across a
    stack of matrices — the analogue of a mass-univariate imaging analysis, with
    the same multiple-comparisons problem.

    The data here is 15 matrices: five with the first block on, five with it off,
    five on again.
    """)
    return


@app.cell
def _(Adjacency, m1, np, symmetric_noise):
    from nltools.data import DesignMatrix

    stack = Adjacency(
        [m1 + symmetric_noise(0.5) for _ in range(5)]
        + [symmetric_noise(0.5) for _ in range(5)]
        + [m1 + symmetric_noise(0.5) for _ in range(5)],
        matrix_type="similarity",
    )

    on_off_on = DesignMatrix(
        np.array([1] * 5 + [0] * 5 + [1] * 5).reshape(-1, 1),
        columns=["on"],
        sampling_freq=1.0,
    )
    on_off_on.plot(title="Model")
    return on_off_on, stack


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The result is one map per predictor. Plotting the t map above a cutoff shows
    the edges that follow the on-off-on pattern, which is the block we planted:
    """)
    return


@app.cell
def _(on_off_on, stack):
    edge_fit = stack.regress(on_off_on)
    edge_fit["t"].plot(vmin=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Similarity and distance

    `similarity` compares two matrices and tests the result by permutation. It
    returns the correlation and a p-value; pass `random_state` to make the
    permutations reproducible.
    """)
    return


@app.cell
def _(Adjacency, blocks, m1):
    blocks.similarity(
        Adjacency(m1, matrix_type="similarity", labels=blocks.labels),
        metric="spearman",
        random_state=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `distance` measures every matrix in a stack against every other, returning a
    new `Adjacency` whose nodes are the original matrices. Any metric
    `scikit-learn`'s `pairwise_distances` accepts works. The five on-matrices,
    the five off-matrices and the final five on-matrices group as you would hope.
    """)
    return


@app.cell
def _(stack):
    matrix_distance = stack.distance(metric="correlation")
    matrix_distance.plot()
    return (matrix_distance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `distance_to_similarity` converts back the other way:
    """)
    return


@app.cell
def _(matrix_distance):
    matrix_distance.distance_to_similarity(metric="correlation").plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multidimensional scaling

    A distance matrix can be laid out in space. `plot_mds` does that in two or
    three dimensions, which is a quick way to see whether the on and off matrices
    separate.
    """)
    return


@app.cell
def _(matrix_distance):
    labeled_distance = matrix_distance.copy()
    labeled_distance.labels = ["On"] * 5 + ["Off"] * 5 + ["On"] * 5
    labeled_distance.plot_mds(n_components=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graphs

    `to_graph` hands a matrix to `networkx`, so every graph metric and layout is
    available. Here the three noiseless blocks make three disconnected cliques of
    four nodes, and every node has degree 3.
    """)
    return


@app.cell
def _(Adjacency, m1, m2, m3):
    import networkx as nx

    clique_graph = Adjacency(m1 + m2 + m3, matrix_type="similarity").to_graph()
    print(f"degree of each node: {dict(clique_graph.degree())}")

    nx.draw_circular(clique_graph, node_color="lightsteelblue", with_labels=True)
    return


if __name__ == "__main__":
    app.run()
