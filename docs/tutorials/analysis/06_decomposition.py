# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Decomposition — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Decomposition

    Decomposition looks for structure with no labels at all: factor a dataset
    into a small number of components and see what they turn out to be. It is the
    tool for "what is in this data" rather than "does this data predict that".

    `BrainData.decompose` wraps scikit-learn's decompositions, so the choice of
    algorithm is one keyword. Here it runs on the pain dataset — 28 subjects at
    three intensities — and the components turn out to track pain.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from joblib import Memory

    from nltools.datasets import fetch_pain
    from nltools.utils import concatenate

    memory = Memory(".tutorial-cache", verbose=0)

    data = memory.cache(fetch_pain)()
    data
    return concatenate, data, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Center within subject

    Every subject contributes three images, and subjects differ from each other
    far more than intensities differ within a subject. Left alone, the first
    components would describe who each image belongs to. Subtracting each
    subject's own mean removes that, leaving the part of each image that is about
    the intensity.
    """)
    return


@app.cell
def _(concatenate, data):
    centered = concatenate(
        [
            data[data.X["SubjectID"] == subject].standardize()
            for subject in data.X["SubjectID"].unique(maintain_order=True)
        ]
    )
    centered
    return (centered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Factor the data

    `method` selects the algorithm: `'pca'`, `'ica'`, `'nnmf'`, `'fa'`,
    `'dictionary'` or `'kernelpca'`. `axis` decides which way round the
    factorization runs. `axis='images'` treats voxels as observations and images
    as features, so each component is a brain map and each image gets a loading
    on it. `axis='voxels'` does the opposite, which is what you want for
    resting-state networks over a timeseries.

    The result carries `decomposition_object` (the fitted scikit-learn
    estimator, with its parameters and explained variance), `components`
    (a `BrainData` of one map per component), and `weights` (an
    images-by-components array).
    """)
    return


@app.cell
def _(centered):
    N_COMPONENTS = 5

    factors = centered.decompose(method="fa", axis="images", n_components=N_COMPONENTS)

    print(sorted(factors))
    print(f"components: {factors['components'].shape}")
    print(f"weights:    {factors['weights'].shape}")
    return N_COMPONENTS, factors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The weight matrix says how much each image loads on each component, and the
    component maps say which voxels each one is made of.
    """)
    return


@app.cell
def _(N_COMPONENTS, factors, plt, sns):
    weight_figure, weight_axis = plt.subplots(figsize=(5, 8))
    sns.heatmap(factors["weights"], ax=weight_axis, center=0, cmap="RdBu_r")
    weight_axis.set_xlabel("component")
    weight_axis.set_ylabel("image")
    weight_axis.set_title("Image loadings")
    weight_figure.tight_layout()

    # Bound rather than left as the cell's last expression, so the page gets the
    # figures and not the list of Figure objects that `plot` returns.
    component_figures = factors["components"].plot(limit=N_COMPONENTS)
    return (component_figures,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Do any components track pain?

    Nothing about the decomposition knew the intensities, so this is a real test.
    Put the loadings beside each image's intensity and average within level.
    """)
    return


@app.cell
def _(N_COMPONENTS, centered, factors, pd, sns):
    loadings = pd.DataFrame(
        factors["weights"],
        columns=[str(index) for index in range(N_COMPONENTS)],
    )
    loadings["intensity"] = centered.X["PainIntensity"].to_list()

    long_loadings = loadings.melt(
        id_vars="intensity", var_name="component", value_name="weight"
    )

    grid = sns.catplot(
        data=long_loadings,
        x="intensity",
        y="weight",
        hue="component",
        order=["low", "medium", "high"],
        kind="point",
        aspect=1.5,
    )
    grid.set_axis_labels("pain intensity", "component loading")
    grid.figure
    return (loadings,)


@app.cell
def _(N_COMPONENTS, centered, loadings, np):
    level = centered.X["PainLevel"].to_numpy()
    for component in range(N_COMPONENTS):
        correlation = np.corrcoef(loadings[str(component)].to_numpy(), level)[0, 1]
        print(f"component {component}: r with intensity {correlation:+.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One component rises monotonically across the three levels and correlates
    strongly with intensity; the rest are flat. Nothing in the factorization was
    told that the intensities existed, so a component that orders them recovered
    real structure. Its map, the first in the panel above, is the one worth
    inspecting.

    ## Inspecting components interactively

    `component_viewer` puts the pieces of one component side by side — its brain
    map at an adjustable threshold, its loading across images, and the power
    spectrum of that loading — with sliders for the component index and the
    threshold. It needs `ipywidgets` and a live kernel, so it is shown here
    rather than run:

    ```python
    from nltools.plotting import component_viewer

    component_viewer(factors, tr=2.0)
    ```

    ## Recap

    | Step | Call |
    |---|---|
    | Remove between-subject differences | `data[...].standardize()` per subject, then `concatenate` |
    | Factor the data | `data.decompose(method=, axis=, n_components=)` |
    | Component brain maps | `result["components"]` — a `BrainData` |
    | Image-by-component loadings | `result["weights"]` |
    | The fitted estimator | `result["decomposition_object"]` |
    | Interactive inspection | `component_viewer(result, tr=)` |

    **Next steps**

    - [Univariate Regression](01_univariate_regression.md) — the supervised
      version of the same question.
    - [Multivariate Pattern Analysis](../workflows/03_mvpa.md) — decoding and
      representational similarity across spatial scales.
    """)
    return


if __name__ == "__main__":
    app.run()
