# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Functional alignment — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Functional Alignment

    Group analysis assumes that a voxel means the same thing in everybody.
    Normalizing to a template and smoothing make that roughly true of anatomy;
    they do not make it true of function. Neighbouring cortex can be tuned very
    differently in two people, and a multivariate model trained on one person's
    voxels then has nothing to say about another's.

    Functional alignment estimates a transform per subject from responses to a
    shared stimulus, putting everyone in one functional space.
    [Hyperalignment](https://academic.oup.com/cercor/article/26/6/2919/1754308)
    does it with an iterative Procrustes transform — rotate, reflect and scale
    each subject's voxels to match a common template. The
    [shared response model](https://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model.pdf)
    does it by factorizing everyone's data into one shared response and a
    per-subject transform, and can compress to fewer features than there are
    voxels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulate data to align

    Three subjects, 500 timepoints, one sphere of medial prefrontal cortex. In
    each subject a random 30% of voxels carry the same signal; the rest is noise.
    The *signal* is shared, the *voxels carrying it* are not — which is exactly
    the problem alignment solves.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from nltools.algorithms import align
    from nltools.data import BrainData
    from nltools.mask import create_sphere

    N_OBSERVATIONS = 500
    N_SUBJECTS = 3
    SIGNAL_FRACTION = 0.3

    signal = np.zeros(N_OBSERVATIONS)
    signal[75:150] = 4
    signal[200:250] = 10
    signal[300:475] = 7

    mask = create_sphere([0, 45, 0], radius=8)
    rng = np.random.default_rng(0)

    def simulate_subject():
        """One subject's timeseries inside the sphere: shared signal, own voxels."""
        subject = BrainData(mask).apply_mask(mask)
        n_voxels = subject.shape[0]
        carries_signal = rng.random(n_voxels) < SIGNAL_FRACTION
        activity = np.zeros((n_voxels, N_OBSERVATIONS))
        activity[carries_signal, :] = signal
        subject.data = (activity + rng.standard_normal(activity.shape)).T
        return subject

    subjects = [simulate_subject() for _ in range(N_SUBJECTS)]
    print(f"{len(subjects)} subjects, each {subjects[0].shape} (timepoints, voxels)")
    return N_SUBJECTS, align, np, plt, signal, subjects


@app.cell
def _(plt, signal, subjects):
    signal_figure, signal_axis = plt.subplots(figsize=(10, 2.5))
    signal_axis.plot(signal)
    signal_axis.set_xlabel("time")
    signal_axis.set_ylabel("signal")
    signal_axis.set_title("The signal every subject's active voxels carry")
    signal_figure.tight_layout()

    subjects[0].mean().plot(title="Simulated sphere, subject 1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperalign

    `align` takes a list of subjects and returns a dictionary: the `transformed`
    data, a `transformation_matrix` per subject, the `common_model` everyone was
    projected into, and an `isc` entry giving the mean intersubject correlation
    of each aligned unit — a direct read on how much shared response the model
    captured. Procrustes adds `disparity`, the multivariate distance from each
    subject to the common space, and `scale`.
    """)
    return


@app.cell
def _(align, subjects):
    hyperaligned = align(subjects, method="procrustes")

    print(sorted(hyperaligned))
    for subject_index, disparity in enumerate(hyperaligned["disparity"], start=1):
        print(f"subject {subject_index}: disparity {disparity:.3f}")
    return (hyperaligned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What changed

    Plot each subject's voxel-by-time matrix before and after. The stripes are
    the shared signal; before alignment they sit in different rows in each
    subject, and afterwards they sit in the same rows.
    """)
    return


@app.cell
def _(N_SUBJECTS, hyperaligned, plt, subjects):
    matrix_figure, matrix_axes = plt.subplots(
        nrows=2, ncols=N_SUBJECTS, figsize=(14, 5), sharex=True, sharey=True
    )
    for column, (original, aligned) in enumerate(
        zip(subjects, hyperaligned["transformed"])
    ):
        matrix_axes[0, column].imshow(original.data.T, aspect="auto")
        matrix_axes[1, column].imshow(aligned.data.T, aspect="auto")
        matrix_axes[0, column].set_title(f"Subject {column + 1}")
        matrix_axes[1, column].set_xlabel("time")
    matrix_axes[0, 0].set_ylabel("original voxels")
    matrix_axes[1, 0].set_ylabel("aligned features")
    matrix_figure.tight_layout()
    matrix_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Averaging across subjects makes the same point without the clutter. Averaging
    unaligned voxels blurs the signal away, because a row that is active in one
    person is noise in the next; averaging aligned features keeps it.
    """)
    return


@app.cell
def _(hyperaligned, np, plt, subjects):
    average_figure, average_axes = plt.subplots(
        ncols=2, figsize=(14, 4), sharex=True, sharey=True
    )
    average_axes[0].imshow(
        np.mean([subject.data.T for subject in subjects], axis=0), aspect="auto"
    )
    average_axes[1].imshow(
        np.mean([aligned.data.T for aligned in hyperaligned["transformed"]], axis=0),
        aspect="auto",
    )
    average_axes[0].set_title("Average of the original data")
    average_axes[1].set_title("Average of the aligned data")
    average_axes[0].set_ylabel("voxels")
    for average_axis in average_axes:
        average_axis.set_xlabel("time")
    average_figure.tight_layout()
    average_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back into each subject's own voxels

    The transformation matrices run both ways, so aligned data can go back to the
    voxels it came from. `align` builds each subject's matrix **R** as
    `aligned = original @ R`, and **R** is a rotation, so the inverse is its
    transpose: `original ≈ aligned @ R.T`. Check the orientation with a
    correlation rather than trusting it — the wrong way round produces a matrix
    of the right shape and no error at all.

    The recovery is close but not exact, because Procrustes also centers each
    subject and rescales it by its norm, and those steps are not undone here.
    """)
    return


@app.cell
def _(N_SUBJECTS, hyperaligned, np, plt, subjects):
    back_projected = [
        np.dot(aligned.data, transform.data.T)
        for aligned, transform in zip(
            hyperaligned["transformed"], hyperaligned["transformation_matrix"]
        )
    ]
    for recovery_index, recovery in enumerate(back_projected, start=1):
        agreement = np.corrcoef(
            recovery.ravel(), subjects[recovery_index - 1].data.ravel()
        )[0, 1]
        print(f"subject {recovery_index}: r with the original {agreement:.2f}")

    back_figure, back_axes = plt.subplots(
        nrows=3, ncols=N_SUBJECTS, figsize=(14, 8), sharex=True, sharey=True
    )
    rows = [
        ("original voxels", [subject.data.T for subject in subjects]),
        (
            "aligned features",
            [aligned.data.T for aligned in hyperaligned["transformed"]],
        ),
        ("back-projected voxels", [matrix.T for matrix in back_projected]),
    ]
    for row, (row_label, matrices) in enumerate(rows):
        for back_column, back_matrix in enumerate(matrices):
            back_axes[row, back_column].imshow(back_matrix, aspect="auto")
        back_axes[row, 0].set_ylabel(row_label)
    for header_column in range(N_SUBJECTS):
        back_axes[0, header_column].set_title(f"Subject {header_column + 1}")
        back_axes[2, header_column].set_xlabel("time")
    back_figure.tight_layout()
    back_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding a subject to an existing model

    A new subject joins a fitted common model without refitting it:
    `BrainData.align` against the model itself rather than against another
    subject. This is what makes cross-validation possible — fit the model on
    training subjects, then project the held-out ones in. Fitting the model on
    everybody and then decoding across subjects leaks.

    Both entry points store the matrix the same way round, so back-projection
    is `aligned @ R.T` here too. Again, the correlation is what tells you.
    """)
    return


@app.cell
def _(hyperaligned, np, plt, subjects):
    newcomer = subjects[2]
    projected = newcomer.align(hyperaligned["common_model"], method="procrustes")
    recovered = np.dot(
        projected["transformed"].data, projected["transformation_matrix"].data.T
    )
    print(
        "r with the original: "
        f"{np.corrcoef(recovered.ravel(), newcomer.data.ravel())[0, 1]:.2f}"
    )

    new_figure, new_axes = plt.subplots(
        ncols=3, figsize=(14, 4), sharex=True, sharey=True
    )
    for new_axis, (new_title, new_matrix) in zip(
        new_axes,
        [
            ("Original", newcomer.data.T),
            ("Projected into the model", projected["transformed"].data.T),
            ("Back-projected", recovered.T),
        ],
    ):
        new_axis.imshow(new_matrix, aspect="auto")
        new_axis.set_title(new_title)
        new_axis.set_xlabel("time")
    new_axes[0].set_ylabel("voxels")
    new_figure.tight_layout()
    new_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A lower-dimensional common space

    The shared response model can align into fewer features than there are
    voxels. Ten features here, from 256 voxels. Fewer features than voxels often
    generalizes better for a multivariate model trained on an ROI; Procrustes
    cannot do it, because an orthogonal rotation preserves dimensionality.

    The return shapes follow from that. `transformed` and `common_model` are
    plain arrays on the model's feature axis, since features are not voxels and
    cannot be written to a brain. `transformation_matrix` *is* a `BrainData`: it
    holds one voxel map per feature, already oriented features-by-voxels, so
    multiplying the aligned data by it puts the subject back in voxel space with
    no transpose.
    """)
    return


@app.cell
def _(N_SUBJECTS, align, np, plt, subjects):
    shared = align(subjects, method="probabilistic_srm", n_features=10)
    print(f"transformed:           {shared['transformed'][0].shape}")
    print(f"common_model:          {shared['common_model'].shape}")
    print(f"transformation_matrix: {shared['transformation_matrix'][0].shape}")

    srm_back = [
        aligned @ transform.data
        for aligned, transform in zip(
            shared["transformed"], shared["transformation_matrix"]
        )
    ]
    for srm_index, srm_recovery in enumerate(srm_back, start=1):
        srm_agreement = np.corrcoef(
            srm_recovery.ravel(), subjects[srm_index - 1].data.ravel()
        )[0, 1]
        print(f"subject {srm_index}: r with the original {srm_agreement:.2f}")

    srm_figure, srm_axes = plt.subplots(nrows=3, ncols=N_SUBJECTS, figsize=(14, 8))
    srm_rows = [
        ("original voxels", [subject.data.T for subject in subjects]),
        ("aligned features", [aligned.T for aligned in shared["transformed"]]),
        ("back-projected voxels", [matrix.T for matrix in srm_back]),
    ]
    for srm_row, (srm_label, srm_matrices) in enumerate(srm_rows):
        for srm_column, srm_matrix in enumerate(srm_matrices):
            srm_axes[srm_row, srm_column].imshow(srm_matrix, aspect="auto")
        srm_axes[srm_row, 0].set_ylabel(srm_label)
    for srm_column in range(N_SUBJECTS):
        srm_axes[0, srm_column].set_title(f"Subject {srm_column + 1}")
        srm_axes[2, srm_column].set_xlabel("time")
    srm_figure.tight_layout()
    srm_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | Step | Call |
    |---|---|
    | Align a group | `align(subjects, method="procrustes")` |
    | Align into fewer features | `align(subjects, method="probabilistic_srm", n_features=10)` |
    | What came back | `transformed`, `transformation_matrix`, `common_model`, `isc` (+ `disparity`, `scale` for Procrustes) |
    | Back into voxels | `aligned @ transform.data.T` — check with a correlation |
    | Add one subject to a model | `subject.align(common_model, method=...)` |

    **Next steps**

    - [Decomposition](06_decomposition.md) — the other way to find structure
      without labels.
    - [Inter-Subject Correlation](../workflows/04_isc.md) — the analysis that
      most often needs alignment first.
    """)
    return


if __name__ == "__main__":
    app.run()
