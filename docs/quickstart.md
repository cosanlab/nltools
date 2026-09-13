# Quickstart

nltools analyzes neuroimaging data through a few objects that behave like
dataframes: `BrainData` holds images as an images-by-voxels matrix,
`DesignMatrix` a timeseries design, `Adjacency` a similarity or distance matrix.
Index them, do arithmetic on them, fit models with them, and plot them.

Everything on this page runs in your browser — a real Python interpreter,
nothing installed on your machine.

!!! note "Before you press Run"

    The first Run downloads Python and the scientific stack into the page, so
    give it a minute or two. Every cell shares one session, so run them in
    order: the later cells use names the earlier ones defined. Edit any cell and
    run it again to see what changes.

## Install and import

```pyodide session="quickstart" install="nltools==0.6.0.dev0"
import nltools

print(nltools.__version__)
```

## Simulate some images

`Simulator` makes synthetic images on the default MNI template: a signal sphere
at the center of the brain, at the intensity of each level you ask for, plus
Gaussian noise. `reps=2` repeats the three levels, so this is six images. The
template downloads on first use.

```pyodide session="quickstart"
from nltools.data import Simulator

sim = Simulator(random_state=0)
data = sim.create_data(levels=[1, 2, 3], sigma=1, reps=2)

print(data)
print(f"{data.shape[0]} images by {data.shape[1]} voxels")
```

## Build a design matrix

A `DesignMatrix` is a dataframe that knows it describes a timeseries: it carries
a sampling frequency, and `convolve` applies a hemodynamic response function to
the task regressors. Here is a 60-TR run with two conditions alternating.

```pyodide session="quickstart"
import numpy as np

from nltools.data import DesignMatrix

faces = np.zeros(60)
faces[[4, 5, 24, 25, 44, 45]] = 1
houses = np.zeros(60)
houses[[14, 15, 34, 35, 54, 55]] = 1

design = DesignMatrix({"faces": faces, "houses": houses}, TR=2.0)
convolved = design.convolve()

print(convolved)
convolved.plot()
```

## Make an adjacency matrix

`Adjacency` holds square matrices over a set of nodes. This one has two blocks
of four nodes, so the heatmap shows two bright squares on the diagonal.

```pyodide session="quickstart"
from nltools.data import Adjacency

rng = np.random.default_rng(0)
noise = rng.standard_normal((8, 8)) * 0.2
blocks = np.block(
    [
        [np.ones((4, 4)), np.zeros((4, 4))],
        [np.zeros((4, 4)), np.ones((4, 4))],
    ]
)
matrix = blocks + (noise + noise.T) / 2
np.fill_diagonal(matrix, 1)

similarity = Adjacency(matrix, matrix_type="similarity", labels=["A"] * 4 + ["B"] * 4)

print(similarity)
similarity.plot()
```

## Plot a brain map

`mean()` averages across images, one value per voxel, and `plot()` draws the
result as a glass brain. The simulated sphere should be the only thing in it.

```pyodide session="quickstart"
data.mean().plot(title="Mean of the simulated images")
```

## Where next

The [tutorials](tutorials/data-operations/01_brain_data.md) work through real data, and the
[Reference](api/nltools.md) documents every namespace. To run nltools outside
the browser, `uv add nltools` or `pip install nltools`.
