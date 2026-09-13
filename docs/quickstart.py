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
    the session, and 3 mm is the grid the example dataset below is built on. That
    is 71,020 voxels in the mask against 238,955 at the 2 mm default, which is what
    keeps this page inside a browser tab.
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
    `load_haxby_example` returns a whole experiment with nothing to download: the
    eight object conditions of the Haxby task in a randomized block design,
    simulated on the MNI template with responses in the ventral-stream regions
    each category drives. It hands back one `BrainData` per run — images by
    voxels, one row per TR — and the `DesignMatrix` that generated it. For real
    data, [`nltools.datasets`](api/datasets.md) has the fetchers.
    """)
    return


@app.cell
def _():
    from nltools.datasets import load_haxby_example

    brains, design_matrices = load_haxby_example()
    data, design = brains[0], design_matrices[0]

    print(data)
    print(data.Y["condition"].value_counts().sort("condition"))
    return data, design, load_haxby_example


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `.Y` carries the condition of every TR, so ordinary indexing pulls one
    condition out and `mean()` averages it. Face blocks minus rest is the response
    to faces, and `iplot()` draws it in an interactive [niivue](https://niivue.com)
    viewer: run the cell below, then drag the sliders to rewindow the map and
    scroll a panel to move through slices.
    """)
    return


@app.cell
def _(data):
    faces = data[data.Y["condition"] == "face"].mean()
    baseline = data[data.Y["condition"] == "rest"].mean()

    (faces - baseline).iplot(threshold="95%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More in [Working with BrainData](tutorials/data-operations/01_brain_data.md),
    which covers loading, indexing, masks and ROIs, plotting and saving images.

    ### Working with an experimental design

    A `DesignMatrix` is a dataframe that knows it describes a timeseries: it carries
    a sampling frequency, and `convolve` applies a hemodynamic response function to
    the task regressors, renaming each one `<column>_c0`. The example's design comes
    back convolved — eight condition regressors and an intercept over 72 TRs:
    """)
    return


@app.cell
def _(design):
    print(design)
    return


@app.cell
def _(design):
    design.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More in [Working with DesignMatrix](tutorials/data-operations/02_design_matrix.md),
    which covers confounds, polynomial terms, multi-run designs and events files.

    ### Working with similarities & distances

    `Adjacency` holds a square matrix over a set of nodes — a correlation matrix, a
    distance matrix, a network. `BrainData.distance` makes one out of images, and it
    is worth doing inside a region: over 71,020 voxels, two patterns mostly differ
    by noise. `create_sphere` draws regions in MNI millimetres and `apply_mask`
    keeps the voxels inside them — here the eight ventral-stream spheres the example
    responds in. Correlation distance ignores how strongly a pattern responds
    overall and compares only its shape across those voxels:
    """)
    return


@app.cell
def _(data):
    from nltools import concatenate
    from nltools.data import Adjacency
    from nltools.mask import create_sphere

    ventral_stream = {
        "face": [40, -50, -20],  # right fusiform face area
        "cat": [-40, -50, -20],  # left fusiform
        "bottle": [46, -78, -6],  # right lateral occipital
        "scissors": [-46, -78, -6],  # left lateral occipital
        "shoe": [36, -66, -16],  # right posterior fusiform
        "chair": [-36, -66, -16],  # left posterior fusiform
        "house": [-26, -44, -10],  # left parahippocampal place area
        "scrambledpix": [0, -88, 2],  # early visual cortex
    }
    ventral_temporal = create_sphere(
        list(ventral_stream.values()), radius=8, mask=data.mask
    )
    patterns = concatenate(
        [data[data.Y["condition"] == name].mean() for name in ventral_stream]
    ).apply_mask(ventral_temporal)
    neural = Adjacency(
        patterns.distance(metric="correlation"), labels=list(ventral_stream)
    )

    print(patterns)
    print(neural)
    return Adjacency, concatenate, neural, ventral_temporal


@app.cell
def _(neural):
    neural.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Eight conditions make 28 distinct pairs, and that is all an `Adjacency` stores;
    the plot fills the diagonal back in with the zero every pattern has with itself.
    The dark cells are the within-category pairs — face with cat, and the four
    man-made objects with each other — while houses and scrambled pictures have no
    close partner.

    More in [Working with Adjacency](tutorials/data-operations/03_adjacency.md),
    which covers thresholds, Fisher z, stacking subjects, regression and graphs.

    ## Common analysis workflows

    Those three objects are all you need. A design fits to data and answers a
    question about conditions; features fit to data and answer a question about
    prediction; data become distances and answer a question about geometry. Every
    permutation and bootstrap test below is cut to 200 resamples from its default
    5,000 so that each cell finishes in about a second.

    ### Mapping neural responses

    Fitting a GLM is `fit(model="glm", X=design)`, and `compute_contrasts` asks it a
    question — here, how much more each voxel responds to faces than to houses:
    """)
    return


@app.cell
def _(data, design):
    data.fit(model="glm", X=design)

    data.compute_contrasts("face_c0 - house_c0").plot(
        title="faces - houses, one subject"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A beta map is the model's answer at every voxel and the contrast is the question
    asked of it. One subject is not a result, though — the map a paper reports is a
    test across subjects. `n_runs=5` returns five fresh draws of the experiment,
    each with its own block order and noise, to stand in for five subjects; stack
    their contrast maps with `concatenate`, and `ttest` gives the voxelwise
    one-sample test, while `threshold` zeroes every voxel whose p-value misses a
    cutoff:
    """)
    return


@app.cell
def _(concatenate, load_haxby_example):
    from nltools.algorithms import threshold

    subjects, subject_designs = load_haxby_example(n_runs=5)
    group = concatenate(
        [
            subject.fit(model="glm", X=subject_design).compute_contrasts(
                "face_c0 - house_c0"
            )
            for subject, subject_design in zip(subjects, subject_designs)
        ]
    )
    group_t = group.ttest()

    threshold(group_t["t"], group_t["p"], thr=0.001).plot(title="group t, p < 0.001")
    return (subjects,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ttest` returns the `mean`, `t`, `z` and `p` maps as separate images.

    ### Predicting neural responses

    An encoding model turns the same equation around: features go in, and the model
    is judged on data it never saw. The features here are every convolved condition
    regressor at twelve delays, 96 columns over 72 TRs, so ordinary least squares
    has no unique solution and ridge regression is what makes the fit possible at
    all. Two runs, one to train on and one held out, both z-scored because ridge
    fits no intercept and the data sit on a baseline of 100. `ridge_cv` picks the
    penalty by cross-validation, one per voxel, and the whole fit lands on `.model`:
    one weight map per feature in `betas`, the chosen penalty in `alpha`, the
    variance explained in `r2`:
    """)
    return


@app.cell
def _(load_haxby_example):
    import numpy as np

    def delayed(design):
        """The eight convolved regressors at twelve delays, side by side."""
        names = [name for name in design.columns if name.endswith("_c0")]
        regressors = np.column_stack([np.asarray(design[name]) for name in names])
        shifted = [np.pad(regressors, ((lag, 0), (0, 0))) for lag in range(12)]
        return np.column_stack([block[: len(regressors)] for block in shifted])

    runs, run_designs = load_haxby_example(n_runs=2)
    train = runs[0].standardize(method="zscore")
    held_out = runs[1].standardize(method="zscore")
    train.fit(
        model="ridge",
        X=delayed(run_designs[0]),
        ridge_alpha=[1, 10, 100, 1000],
        ridge_cv=5,
        random_state=0,
    )

    print(train.model.betas)
    print(f"penalty per voxel: {np.unique(train.model.alpha.data)}")
    print(f"best training r2: {train.model.r2.data.max():.2f}")
    return delayed, held_out, np, run_designs, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Applying those weights to the held-out run's design predicts its timecourse.
    Correlating that prediction with what the run actually did, voxel by voxel,
    gives a performance map: not how much a voxel responds, but how well the model
    accounts for it:
    """)
    return


@app.cell
def _(delayed, held_out, np, run_designs, train):
    from nltools.data import BrainData

    def voxel_correlation(observed, predicted):
        """Correlate two arrays over the same voxels, one voxel at a time."""
        observed = (observed - observed.mean(axis=0)) / observed.std(axis=0)
        predicted = (predicted - predicted.mean(axis=0)) / predicted.std(axis=0)
        return (observed * predicted).mean(axis=0)

    encoding_scores = voxel_correlation(
        held_out.data, train.predict(X=delayed(run_designs[1])).data
    )

    BrainData(encoding_scores[None, :], mask=held_out.mask).plot(
        title="Held-out run, r per voxel"
    )
    return BrainData, voxel_correlation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analyzing neural patterns

    Representational similarity analysis compares geometries instead of voxels: how
    far apart the conditions are in the brain, against how far apart a model says
    they should be. Both sides are an `Adjacency`, and `similarity` correlates them
    — Spearman by default — with a permutation test that shuffles rows and columns
    together. The neural side is the distance matrix from the basics section; the
    model side says two conditions are far apart when they belong to different
    categories:
    """)
    return


@app.cell
def _(Adjacency, neural, np):
    categories = np.array(["animate"] * 2 + ["object"] * 4 + ["scene", "control"])
    model_rdm = Adjacency(
        (categories[:, None] != categories[None, :]).astype(float),
        matrix_type="distance",
        labels=neural.labels,
    )
    rsa = neural.similarity(model_rdm, n_permute=200, random_state=0, n_jobs=1)

    print(f"rho = {rsa['correlation']:.2f}, p = {rsa['p']:.3f} (200 permutations)")
    return (model_rdm,)


@app.cell
def _(model_rdm, neural):
    neural.plot_stacked(
        model_rdm, upper_title="Correlation distance", lower_title="Category model"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_stacked` puts the two matrices in one square, the measured geometry above
    the diagonal and the model below it.

    Decoding asks the same region the opposite question: given the pattern, which
    condition was it? `predict` cross-validates a classifier over images and reports
    the accuracy per fold. Twelve face and house TRs, three folds:
    """)
    return


@app.cell
def _(data, ventral_temporal):
    decoded = (
        data[data.Y["condition"].is_in(["face", "house"])]
        .apply_mask(ventral_temporal)
        .predict(y="condition", estimator="linear_svc", cv=3)
    )

    print(f"accuracy {decoded.mean_score:.2f}, per fold {decoded.scores}")
    return (decoded,)


@app.cell
def _(decoded):
    decoded.weight_map.plot(title="Classifier weights, faces vs houses")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The weight map is the pattern the classifier leaned on, refit on all twelve TRs.
    Decoding runs inside a region rather than over the whole brain. One run labels
    twelve TRs against 71,020 voxels, and a whole-brain classifier fitted to that
    lands anywhere between chance and this, depending on the noise; masking to
    ventral temporal cortex first is how the real Haxby data are analyzed too.

    ### Analyzing intersubject similarity

    Intersubject correlation asks how much of a response is shared: with everyone
    watching the same thing, the part of one subject's timecourse that another
    subject also shows is the part the stimulus drove. `extract_roi` averages each
    subject's timecourse inside every parcel of a 50-region atlas, `isc` correlates
    every pair of subjects one parcel at a time and bootstraps subjects for the
    p-value, and `roi_to_brain` paints the answer back onto the brain. Each
    simulated subject saw the blocks in a different order, so line the TRs up by
    condition first:
    """)
    return


@app.cell
def _(BrainData, np, subjects):
    from nltools.algorithms import isc
    from nltools.mask import expand_mask, roi_to_brain
    from nltools.templates import fetch_resource

    parcellation = BrainData(
        fetch_resource("masks/default/3mm-MNI152-2009fsl-k50.nii.gz")
    )

    def time_locked(subject):
        """Reorder one subject's TRs into the sequence every subject shares."""
        order = subject.Y.with_row_index().sort(["condition", "index"])["index"]
        return subject[order.to_numpy()]

    parcel_timeseries = np.stack(
        [time_locked(subject).extract_roi(parcellation).T for subject in subjects],
        axis=1,
    )
    isc_result = isc(parcel_timeseries, n_samples=200, random_state=0, n_jobs=1)

    print(
        f"best parcel: ISC {isc_result['isc'].max():.2f}, "
        f"{(isc_result['p'] < 0.05).sum()} of 50 parcels at p < 0.05 (200 bootstraps)"
    )
    return expand_mask, isc_result, parcellation, roi_to_brain, time_locked


@app.cell
def _(expand_mask, isc_result, parcellation, roi_to_brain):
    roi_to_brain(isc_result["isc"], expand_mask(parcellation)).plot(
        title="Intersubject correlation, per parcel"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Intersubject RSA is the previous section's analysis with people as the items: a
    matrix of how similar each pair of subjects' responses are, against a matrix of
    how far apart their behavioural scores are. Here each subject's ventral-temporal
    response is a different blend of a face-driven and a house-driven profile, and
    their score on a face-recognition task tracks the blend:
    """)
    return


@app.cell
def _(Adjacency, np, subjects, ventral_temporal):
    blend = np.linspace(0, 1, len(subjects))

    def blended_response(subject, weight):
        """One subject's ventral-temporal response, blended from faces toward houses."""
        rest = subject[subject.Y["condition"] == "rest"].mean()
        to_faces = subject[subject.Y["condition"] == "face"].mean() - rest
        to_houses = subject[subject.Y["condition"] == "house"].mean() - rest
        blended = to_faces * (1 - weight) + to_houses * weight
        return blended.apply_mask(ventral_temporal).data

    profiles = [blended_response(s, w) for s, w in zip(subjects, blend)]
    neural_similarity = Adjacency(
        np.corrcoef(profiles),
        matrix_type="similarity",
        labels=[f"s{n + 1}" for n in range(len(subjects))],
    )
    scores = 100 - 40 * blend
    behaviour = Adjacency(
        np.abs(scores[:, None] - scores[None, :]), matrix_type="distance"
    )
    isrsa = neural_similarity.similarity(
        behaviour, n_permute=200, random_state=0, n_jobs=1
    )

    print(f"rho = {isrsa['correlation']:.2f}, p = {isrsa['p']:.3f} (200 permutations)")
    return behaviour, neural_similarity


@app.cell
def _(behaviour, neural_similarity):
    neural_similarity.plot_stacked(
        behaviour,
        upper_title="Response similarity",
        lower_title="Face-recognition score distance",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The correlation is negative because one side is a similarity and the other a
    distance: subjects who respond alike are the ones whose scores are close.

    ### Aligning neural responses

    Two people watching the same film share the response, not the anatomy it sits in:
    the same information can live in different voxels, and functional alignment finds
    the transformation between them. Here the second subject is the first one's
    ventral-temporal data with its voxels shuffled — nothing lost, only moved:
    """)
    return


@app.cell
def _(BrainData, np, subjects, time_locked, ventral_temporal, voxel_correlation):
    target = time_locked(subjects[0]).apply_mask(ventral_temporal)
    shuffle = np.random.default_rng(0).permutation(target.shape[1])
    scrambled = BrainData(target.data[:, shuffle], mask=ventral_temporal)
    aligned = scrambled.align(target, method="procrustes")["transformed"]

    print(f"before: r = {voxel_correlation(target.data, scrambled.data).mean():.2f}")
    print(f"after:  r = {voxel_correlation(target.data, aligned.data).mean():.2f}")
    return aligned, scrambled, target


@app.cell
def _(aligned, scrambled, target):
    import matplotlib.pyplot as plt

    _panels = {
        "subject 1 (target)": target,
        "subject 2, voxels shuffled": scrambled,
        "subject 2, after procrustes": aligned,
    }
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    for _ax, (_label, _image) in zip(_axes, _panels.items()):
        _ax.imshow(_image.data[:, :40], aspect="auto", cmap="RdBu_r")
        _ax.set(title=_label, xlabel="voxel")
    _axes[0].set_ylabel("TR")
    _fig.tight_layout()
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
