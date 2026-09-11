# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Similarity and distance — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Similarity and Distance

    Comparing images to each other needs no model at all. Two questions come up
    constantly:

    - How far is every image from every other one? That is a **distance matrix**,
      and it is what representational similarity analysis works on.
    - How much does one image look like a particular pattern? That is a
      **pattern response**, the number a published brain signature produces when
      you apply it to new data.

    Both run on the pain dataset: 28 subjects, three intensities each.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from joblib import Memory

    from nltools.datasets import fetch_pain

    memory = Memory(".tutorial-cache", verbose=0)

    data = memory.cache(fetch_pain)()
    data
    return data, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distance between every pair of images

    `distance` computes the pairwise spatial distance between the images in a
    `BrainData`, using any metric `scipy.spatial.distance.cdist` supports.
    Correlation distance — one minus the spatial correlation — is the usual
    choice, because it ignores differences in overall scale between images.

    The result is an `Adjacency`, the class for matrices over a set of nodes. It
    stores the 3,486 unique pairs rather than the full square, and knows that it
    is a distance matrix.
    """)
    return


@app.cell
def _(data):
    distances = data.distance(metric="correlation")
    print(distances)
    distances.plot()
    return (distances,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The block structure along the diagonal is the subjects: three consecutive
    images belong to one person, and a person's images resemble each other more
    than they resemble anybody else's. That similarity is a fact about the
    individual, not about pain, and it is the reason cross-validation has to hold
    subjects out together.
    """)
    return


@app.cell
def _(data, distances, np):
    square = distances.squareform()
    same_subject = (
        data.X["SubjectID"].to_numpy()[:, None]
        == data.X["SubjectID"].to_numpy()[None, :]
    )
    off_diagonal = ~np.eye(len(data), dtype=bool)

    print(f"within subject:  {square[same_subject & off_diagonal].mean():.3f}")
    print(f"between subject: {square[~same_subject].mean():.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Similarity to one pattern

    `similarity` compares every image in a `BrainData` to a single map and
    returns one value per image. Average the high-intensity images across
    subjects and you have a rough pain pattern; the similarity of each image to
    it is that image's pattern response.
    """)
    return


@app.cell
def _(data):
    high_pain = data[data.X["PainLevel"] == 3].mean()
    response = data.similarity(high_pain, metric="correlation")

    print(f"one value per image: {response.shape}")
    high_pain.plot(title="Mean high-intensity pain image")
    return high_pain, response


@app.cell
def _(data, np, plt, response):
    intensity = data.X["PainLevel"].to_numpy()

    similarity_figure, axes = plt.subplots(ncols=2, figsize=(10, 4))
    axes[0].hist(response, bins=20)
    axes[0].set_xlabel("spatial similarity")
    axes[0].set_ylabel("images")
    axes[0].set_title("Similarity to the mean high-pain image")

    jitter = np.random.default_rng(0).normal(0, 0.04, intensity.size)
    axes[1].scatter(intensity + jitter, response, alpha=0.6)
    axes[1].set_xticks([1, 2, 3], ["low", "medium", "high"])
    axes[1].set_xlabel("pain intensity")
    axes[1].set_ylabel("spatial similarity")
    axes[1].set_title("Pattern response by intensity")
    similarity_figure.tight_layout()
    similarity_figure
    return (intensity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Response rises with intensity, which is the point — the pattern carries
    information about the manipulation. The 28 high-intensity images are part of
    the average they are being compared to, so their similarity is inflated;
    a real pattern-response analysis uses a pattern estimated on other people.

    ### Metrics

    `metric` selects what "similar" means. Correlation and its `'pearson'` alias
    center both maps first; `'cosine'` and `'dot_product'` do not, so they are
    sensitive to a map's overall offset and scale. `'rank_correlation'` (alias
    `'spearman'`) compares the orderings, which blunts the influence of a few
    extreme voxels.
    """)
    return


@app.cell
def _(data, high_pain, intensity, np):
    for metric in ["correlation", "rank_correlation", "cosine", "dot_product"]:
        values = data.similarity(high_pain, metric=metric)
        print(
            f"{metric:18s} range [{values.min():8.2f}, {values.max():8.2f}]   "
            f"r with intensity {np.corrcoef(values, intensity)[0, 1]:.2f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All four order the intensities the same way; they differ in scale and in how
    much a handful of high-magnitude voxels can move them.

    ## Recap

    | Step | Call |
    |---|---|
    | Pairwise distance between images | `data.distance(metric="correlation")` → `Adjacency` |
    | Full square from the stored pairs | `adjacency.squareform()` |
    | Response of every image to one map | `data.similarity(pattern, metric=)` |
    | Build a pattern from a subset | `data[mask].mean()` |

    **Next steps**

    - [Functional Alignment](05_hyperalignment.md) — put subjects in a common
      functional space before comparing their patterns.
    - [Adjacency Matrices](../data-operations/06_adjacency.md) — what else the
      matrix `distance` returns can do.
    - [Multivariate Pattern Analysis](../workflows/03_mvpa.md) — representational
      similarity analysis, built on exactly this distance matrix.
    """)
    return


if __name__ == "__main__":
    app.run()
