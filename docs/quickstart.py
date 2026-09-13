# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools==0.6.0.dev0",
# ]
# ///
# Quickstart — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Quickstart

    `nltools` keeps things simple by making use of a few key concepts that can be
    flexibly combined to perform a wide variety of analyses.

    Everything on this page runs in your browser — a real Python interpreter,
    nothing installed. Press Run to recompute a cell; the first Run downloads
    Python and nltools, and each cell needs the ones above it to have run first.

    ## Basics

    ### Working with neuroimaging data

    Start by choosing the grid every image lives on. `set_brainspace` sets it for
    the session, and 3 mm is what keeps this page inside a browser tab: 71,020
    voxels in the mask, against 238,955 at the 2 mm default.
    """)
    return


@app.cell
def _():
    import nltools
    from nltools import set_brainspace

    print(nltools.__version__)
    print(set_brainspace(resolution=3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `BrainData` is the object almost every analysis starts from: images by voxels,
    one row per image and one column per voxel inside the mask. `Simulator` fills it
    with synthetic images — a sphere of signal scaled to each level, plus Gaussian
    noise. `reps=2` repeats the three levels, and `.Y` holds the level each image
    was drawn at:
    """)
    return


@app.cell
def _():
    from nltools.data import Simulator

    sim = Simulator(random_state=0)
    data = sim.create_data([1, 2, 3], sigma=1, reps=2, center=[0, -18, 18])

    print(data)
    print(f"levels: {data.Y['y'].to_list()}")
    return Simulator, data, sim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `mean()` averages across images, one value per voxel, and `plot()` draws the
    result — the signal sphere should be the only thing left standing:
    """)
    return


@app.cell
def _(data):
    data.mean().plot(title="Mean of the six simulated images")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More in [Working with BrainData](tutorials/data-operations/01_brain_data.md),
    which covers loading, indexing, masks and ROIs, plotting and saving images.

    ### Working with an experimental design

    A `DesignMatrix` is a dataframe that knows it describes a timeseries: it carries
    a sampling frequency, and `convolve` applies a hemodynamic response function to
    the task regressors, renaming each one `<column>_c0`. Here is a 60-TR run with
    two conditions alternating:
    """)
    return


@app.cell
def _():
    import numpy as np

    from nltools.data import DesignMatrix

    faces = np.zeros(60)
    faces[[4, 5, 24, 25, 44, 45]] = 1
    houses = np.zeros(60)
    houses[[14, 15, 34, 35, 54, 55]] = 1

    design = DesignMatrix({"faces": faces, "houses": houses}, TR=2.0)
    convolved = design.convolve()

    print(convolved)
    return convolved, np


@app.cell
def _(convolved):
    convolved.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More in [Working with DesignMatrix](tutorials/data-operations/02_design_matrix.md),
    which covers confounds, polynomial terms, multi-run designs and events files.

    ### Working with similarities & distances

    `Adjacency` holds a square matrix over a set of nodes — a correlation matrix, a
    distance matrix, a network. `BrainData.distance` makes one out of images, and it
    is worth doing inside a region: over 71,020 voxels, two noisy images mostly
    differ by noise. `create_sphere` draws a region in MNI millimetres and
    `apply_mask` keeps the voxels inside it — here, the sphere the simulator put the
    signal in. These images differ in how strongly that sphere responds, not in the
    shape of the pattern across it — the difference euclidean distance sees:
    """)
    return


@app.cell
def _(data):
    from nltools.mask import create_sphere

    region = create_sphere([0, -18, 18], radius=10)
    patterns = data.apply_mask(region)
    neural = patterns.distance(metric="euclidean")

    print(patterns)
    print(neural)
    return neural, patterns, region


@app.cell
def _(neural):
    neural.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Six images make 36 cells, but only 15 distinct pairs, and that is all an
    `Adjacency` stores; the plot fills the diagonal back in with the zero every image
    has with itself. The palest cells are the 1-versus-3 pairs, furthest apart; the
    darkest off the diagonal are the pairs drawn at the same level.

    More in [Working with Adjacency](tutorials/data-operations/03_adjacency.md),
    which covers thresholds, Fisher z, stacking subjects, regression and graphs.

    ## Common analysis workflows

    Those three objects are all you need. A design fits to data and answers a
    question about conditions; features fit to data and answer a question about
    prediction; data become distances and answer a question about geometry. Each
    workflow below is one short example on simulated data, with the permutation and
    bootstrap tests cut to 200 resamples from their default 5,000 so that every cell
    finishes in about a second.

    ### Mapping neural responses

    Fitting a GLM is `fit(model="glm", X=design)`, and `compute_contrasts` asks it a
    question. Here one run is simulated so that the sphere follows the `faces`
    regressor, and the contrast should recover it:
    """)
    return


@app.cell
def _(Simulator, convolved):
    run = Simulator(random_state=1).create_data(
        convolved["faces_c0"].to_list(), 0.2, radius=10, center=[0, -18, 18]
    )
    run.fit(model="glm", X=convolved)

    run.compute_contrasts("faces_c0 - houses_c0").plot(title="faces - houses, one run")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A beta map is the model's answer at every voxel and the contrast is the question
    asked of it: how much more this voxel responds to faces than to houses. One
    subject is not a result, though — the map a paper reports is a test across
    subjects. Simulate five, stack their contrast maps with `concatenate`, and
    `ttest` gives the voxelwise one-sample test; `threshold` zeroes every voxel whose
    p-value misses a cutoff:
    """)
    return


@app.cell
def _(Simulator, convolved):
    from nltools import concatenate
    from nltools.algorithms import threshold

    group = concatenate(
        [
            Simulator(random_state=subject)
            .create_data(
                convolved["faces_c0"].to_list(), 0.2, radius=10, center=[0, -18, 18]
            )
            .fit(model="glm", X=convolved)
            .compute_contrasts("faces_c0 - houses_c0")
            for subject in range(2, 7)
        ]
    )
    group_t = group.ttest()

    threshold(group_t["t"], group_t["p"], thr=0.001).plot(title="group t, p < 0.001")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ttest` returns the `mean`, `t`, `z` and `p` maps as separate images.

    ### Predicting neural responses

    An encoding model turns the same equation around: instead of a handful of task
    regressors, a feature matrix with more columns than timepoints, and ridge
    regression to keep the fit from blowing up. Here 100 random features, the first
    ten of which drive the sphere, and two runs of the same experiment so the model
    can be scored on data it never saw. `ridge_cv` picks the penalty by
    cross-validation, one per voxel:
    """)
    return


@app.cell
def _(Simulator, np):
    rng = np.random.default_rng(0)
    features = rng.standard_normal((60, 100))
    signal = features[:, :10].sum(axis=1).tolist()

    run1 = Simulator(random_state=7).create_data(
        signal, 1.0, radius=10, center=[0, -18, 18]
    )
    run2 = Simulator(random_state=8).create_data(
        signal, 1.0, radius=10, center=[0, -18, 18]
    )
    run1.fit(
        model="ridge",
        X=features,
        ridge_alpha=[0.1, 1, 10, 100],
        ridge_cv=5,
        random_state=0,
    )

    chosen, counts = np.unique(run1.model_.alpha_, return_counts=True)
    print("penalty chosen, voxel count:")
    for alpha, count in zip(chosen, counts):
        print(f"  {alpha:>5}  {count}")
    return features, rng, run1, run2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fit leaves `ridge_weights` on the object, one map per feature, so predicting
    the second run is a matrix product. Correlating that prediction with what the
    second run actually did, voxel by voxel, gives a performance map: not how much
    this voxel responds, but how well the model accounts for it:
    """)
    return


@app.cell
def _(features, np, run1, run2):
    from nltools.data import BrainData

    def voxel_correlation(observed, predicted):
        """Correlate two runs of the same voxels, one voxel at a time."""
        observed = observed - observed.mean(axis=0)
        predicted = predicted - predicted.mean(axis=0)
        norms = np.sqrt((observed**2).sum(axis=0) * (predicted**2).sum(axis=0))
        return (observed * predicted).sum(axis=0) / norms

    encoding_scores = voxel_correlation(run2.data, features @ run1.ridge_weights.data)
    score_map = BrainData(encoding_scores[None, :], mask=run2.mask)

    score_map.plot(title="Held-out prediction, r per voxel")
    return BrainData, encoding_scores, voxel_correlation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A performance map is what you compare across models, the way a beta map is what
    you compare across conditions. Scoring the same weights against features whose
    timing has been shuffled says how much of the map above is the model rather than
    the noise floor:
    """)
    return


@app.cell
def _(encoding_scores, features, rng, run1, run2, voxel_correlation):
    shuffled = rng.permutation(features, axis=0)
    chance_scores = voxel_correlation(run2.data, shuffled @ run1.ridge_weights.data)

    for name, r in (("features", encoding_scores), ("shuffled", chance_scores)):
        print(f"{name:>9}: best r {r.max():.2f}, {(r > 0.6).sum()} voxels above 0.6")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analyzing neural patterns

    Representational similarity analysis compares geometries instead of voxels: how
    far apart the conditions are in the brain, against how far apart a model says
    they should be. Both sides are an `Adjacency`, and `similarity` correlates them
    — Spearman by default — with a permutation test that shuffles rows and columns
    together. The neural side is the distance matrix from the basics section; the
    model side says two images are far apart when their levels are:
    """)
    return


@app.cell
def _(data, neural, np):
    from nltools.data import Adjacency

    levels = np.array(data.Y["y"].to_list())
    model_rdm = Adjacency(
        np.abs(levels[:, None] - levels[None, :]), matrix_type="distance"
    )
    rsa = neural.similarity(model_rdm, n_permute=200, random_state=0, n_jobs=1)

    print(f"rho = {rsa['correlation']:.2f}, p = {rsa['p']:.3f} (200 permutations)")
    return (Adjacency,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Decoding asks the same region the opposite question: given the pattern, which
    condition was it? `predict` cross-validates a classifier over images and reports
    the accuracy per fold. Twenty images at two levels, five folds:
    """)
    return


@app.cell
def _(region, sim):
    two_levels = sim.create_data([1, 2], sigma=3, reps=10, center=[0, -18, 18])
    decoded = two_levels.apply_mask(region).predict(
        y="y", estimator="linear_svc", cv=5
    )

    print(f"accuracy {decoded.mean_score:.2f}, per fold {decoded.scores}")
    return (decoded,)


@app.cell
def _(decoded):
    decoded.weight_map.plot(title="Classifier weights")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The weight map is the pattern the classifier leaned on, refit on all twenty
    images.

    ### Analyzing intersubject similarity

    Intersubject correlation asks how much of a response is shared: with everyone
    watching the same thing, the part of one subject's timecourse that another
    subject also shows is the part the stimulus drove. `isc` correlates every pair of
    columns in an observations-by-subjects array and bootstraps subjects for the
    p-value of the median. These five carry the same two signals, mixed differently
    in each:
    """)
    return


@app.cell
def _(np):
    from nltools.algorithms import isc

    noise = np.random.default_rng(1).standard_normal((60, 5))
    clock = np.linspace(0, 6 * np.pi, 60)
    mixture = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    timeseries = np.column_stack(
        [
            (1 - m) * np.sin(clock) + m * np.cos(2 * clock) + 0.5 * noise[:, i]
            for i, m in enumerate(mixture)
        ]
    )
    isc_result = isc(timeseries, n_samples=200, random_state=0, n_jobs=1)

    print(
        f"median ISC {isc_result['isc']:.2f}, "
        f"p = {isc_result['p'].item():.3f} (200 bootstraps)"
    )
    return mixture, timeseries


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Intersubject RSA is the previous section's analysis with people as the items: a
    matrix of how similar each pair of subjects' timecourses are, against a matrix of
    how far apart their behavioural scores are. Subjects whose responses mix the two
    signals alike should also score alike:
    """)
    return


@app.cell
def _(Adjacency, mixture, np, timeseries):
    subject_similarity = Adjacency(np.corrcoef(timeseries.T), matrix_type="similarity")
    behaviour_scores = 40 + 20 * mixture
    behaviour = Adjacency(
        np.abs(behaviour_scores[:, None] - behaviour_scores[None, :]),
        matrix_type="distance",
    )
    isrsa = subject_similarity.similarity(
        behaviour, n_permute=200, random_state=0, n_jobs=1
    )

    print(f"rho = {isrsa['correlation']:.2f}, p = {isrsa['p']:.3f} (200 permutations)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The correlation is negative because one side is a similarity and the other a
    distance: subjects who respond alike are the ones whose scores are close.

    ### Aligning neural responses

    Two people watching the same film share the response, not the anatomy it sits in:
    the same information can live in different voxels, and functional alignment finds
    the transformation between them. Here the second subject is the first one's data
    with its voxels shuffled — nothing lost, only moved:
    """)
    return


@app.cell
def _(BrainData, np, patterns, region, voxel_correlation):
    voxels = patterns.shape[1]
    shuffle_rng = np.random.default_rng(2)
    responses = shuffle_rng.standard_normal((40, 5))
    tuning = shuffle_rng.standard_normal((5, voxels))
    order = shuffle_rng.permutation(voxels)

    sub1 = BrainData(
        responses @ tuning + 0.5 * shuffle_rng.standard_normal((40, voxels)),
        mask=region,
    )
    sub2 = BrainData(sub1.data[:, order], mask=region)
    aligned = sub2.align(sub1, method="procrustes")

    print(f"before: r = {voxel_correlation(sub1.data, sub2.data).mean():.2f}")
    print(
        "after:  r = "
        f"{voxel_correlation(sub1.data, aligned['transformed'].data).mean():.2f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shuffling voxels is an orthogonal transformation and that is exactly what
    `procrustes` solves for, so it recovers the data exactly — the idealized version
    of what alignment does when two brains carry the same response in different
    places. `BrainData.align` works one pair at a time; `nltools.algorithms.align`
    takes a list of subjects and, with `deterministic_srm` or `probabilistic_srm`,
    learns the shared space they all map into.

    ## Keep learning

    The [Reference](api/nltools.md) documents every namespace and the tutorials work
    through the same objects on real data. Two courses teach the neuroimaging itself
    with nltools end to end.

    - [DartBrains](https://dartbrains.org) — the fundamentals of fMRI analysis,
      from preprocessing to group statistics.
    - [Naturalistic Data](https://naturalistic-data.org) — movies, games and other
      naturalistic designs, where the intersubject methods above come from.
    - [Getting help](index.md#learning-nltools) — the Discourse forum, where
      questions and their answers stay findable.
    """)
    return


if __name__ == "__main__":
    app.run()
