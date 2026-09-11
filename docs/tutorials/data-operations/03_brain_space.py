# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Brain space and resolution — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Brain Space and Resolution

    Every `BrainData` object lives on a grid: a template, a resolution, and the
    brain mask that decides which voxels are in. nltools defaults to a 2 mm MNI152
    template, and gives you three levels of control over that choice — a global
    setting, a scoped override, and a per-object mask.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The current space

    `get_brainspace()` reports the active configuration and the files it resolves
    to:
    """)
    return


@app.cell
def _():
    from nltools import (
        get_brainspace,
        reset_brainspace,
        set_brainspace,
        with_brainspace,
    )
    from nltools.data import BrainData, Simulator

    get_brainspace()
    return (
        BrainData,
        Simulator,
        get_brainspace,
        reset_brainspace,
        set_brainspace,
        with_brainspace,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulated data to work with

    `Simulator` generates `BrainData` with a known signal, which is a convenient
    way to see which grid an object landed on. `create_data` drops a spherical
    signal of each level into a noisy brain; `reps` sets how many images per level
    and the second argument sets the noise. Objects it creates have no grid of
    their own, so they adopt the current brain space — 2 mm here, about 240k
    voxels.
    """)
    return


@app.cell
def _(Simulator):
    dummy_brain = Simulator(random_state=0).create_data([0, 1], 1, reps=3)
    dummy_brain
    return (dummy_brain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The level of each image is recorded in `.Y`:
    """)
    return


@app.cell
def _(dummy_brain):
    dummy_brain.Y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Save it, so there is a 2 mm file on disk to load back in below:
    """)
    return


@app.cell
def _(dummy_brain):
    import tempfile
    from pathlib import Path

    brain_2mm = Path(tempfile.mkdtemp()) / "dummy_2mm_brain.nii.gz"
    dummy_brain.write(str(brain_2mm))
    print(f"{brain_2mm.name}: {brain_2mm.stat().st_size / 1e6:.1f} MB")
    return (brain_2mm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Changing the space globally

    `set_brainspace()` changes the default for everything that follows. Set it once
    at the top of an analysis: changing it midway leaves later objects on a
    different grid than earlier ones, which is a hard mistake to spot.
    """)
    return


@app.cell
def _(Simulator, brain_2mm, set_brainspace):
    print(f"{brain_2mm.name} was written on the 2 mm grid")
    print(set_brainspace(resolution=3))

    # A fresh simulation now lands on the 3 mm grid, about 71k voxels
    dummy_3mm = Simulator(random_state=0).create_data([0, 1], 1, reps=3)
    dummy_3mm
    return (dummy_3mm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `reset_brainspace()` puts the default back:
    """)
    return


@app.cell
def _(dummy_3mm, reset_brainspace):
    print(f"simulated on the 3 mm grid: {dummy_3mm.shape[-1]} voxels")
    default_space = reset_brainspace()
    default_space
    return (default_space,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Changing it temporarily

    `with_brainspace()` does the same thing for the duration of a block and
    restores the previous setting on exit, including when the block raises. Prefer
    it when only part of an analysis belongs in another space.
    """)
    return


@app.cell
def _(Simulator, default_space, get_brainspace, with_brainspace):
    print(f"before the block: {default_space.resolution} mm")

    with with_brainspace(resolution=3):
        scoped = Simulator(random_state=0).create_data([0, 1], 1, reps=1)

    print(f"inside the block:  {scoped.shape[-1]} voxels")
    print(f"after the block:   {get_brainspace().resolution} mm is active again")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the global setting does and does not control

    The setting is a *default* for objects that have no grid of their own. A file
    that already sits on a recognized template grid keeps it: loading the 2 mm file
    written above gives 2 mm data even while the global setting says 3 mm.
    """)
    return


@app.cell
def _(BrainData, brain_2mm, default_space, with_brainspace):
    with with_brainspace(resolution=3):
        loaded_under_3mm = BrainData(str(brain_2mm))

    print(f"global default: {default_space.resolution} mm")
    print(f"3 mm default, 2 mm file -> {loaded_under_3mm.shape[-1]} voxels")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-object masks

    To put one object on a specific grid regardless of the global setting, pass
    `mask`. It takes a template name, a path to any NIfTI file, or a nibabel image,
    and the data is resampled to match.
    """)
    return


@app.cell
def _(BrainData, brain_2mm, default_space):
    print(f"global default is still {default_space.resolution} mm")
    on_3mm_grid = BrainData(str(brain_2mm), mask="3mm-MNI152-2009fsl")
    print(f"named 3 mm mask -> {on_3mm_grid.shape[-1]} voxels")
    on_3mm_grid
    return (on_3mm_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The global setting is untouched by that:
    """)
    return


@app.cell
def _(get_brainspace, on_3mm_grid):
    print(f"the object above holds {on_3mm_grid.shape[-1]} voxels, and yet:")
    get_brainspace()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Template names follow `'{resolution}mm-MNI152-2009{version}'`, where the
    version letter picks the family: `fsl` for the bundled default, `a` for
    nilearn's, `c` for the fMRIPrep template. `get_brainspace().mask`, `.brain` and
    `.plot` give the resolved file paths when you need to hand them to another
    tool.
    """)
    return


if __name__ == "__main__":
    app.run()
