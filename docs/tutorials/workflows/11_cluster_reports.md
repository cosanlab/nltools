---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Cluster Reports & Atlas Labeling

After thresholding a stat map, the next step is usually to summarize what you found: where are the surviving clusters, how big are they, and what anatomical regions do they overlap? `BrainData.cluster_report()` produces a tidy peak/cluster table plus optional figures, with anatomical labels drawn from one or more atlases.

The atlas data is bundled in the `nltools/niftis` HuggingFace dataset and fetched lazily on first use — no upfront download cost. Eleven atlases are available; see `nltools.data.atlases.list_atlases()`.

## Quickstart

```{code-cell} python3
from nltools.datasets import fetch_pain
from nltools.data.atlases import list_atlases

list_atlases()
```

```{code-cell} python3
brains = fetch_pain()
contrast = brains.mean()  # any 3D stat map works

report = contrast.cluster_report(
    stat_threshold=3.0,
    cluster_threshold=20,
)
```

The result is a `ClusterReport` with three pieces:

```{code-cell} python3
report.peaks.head()
```

```{code-cell} python3
report.clusters.head()
```

```{code-cell} python3
type(report.stat_img)  # the thresholded BrainData, ready for further analysis
```

## Choosing atlases

By default, `cluster_report` labels each peak/cluster against three atlases — Harvard-Oxford (probabilistic anatomical), AAL (deterministic anatomical), and Schaefer-200 (functional parcellation). Pass `atlas=` to override:

```{code-cell} python3
# Single atlas
report = contrast.cluster_report(stat_threshold=3.0, atlas="harvard_oxford")

# Multiple atlases — one column per atlas in the output
report = contrast.cluster_report(
    stat_threshold=3.0,
    atlas=["harvard_oxford", "juelich", "neuromorphometrics"],
)
report.peaks.columns
```

For probabilistic atlases (Harvard-Oxford, Juelich) each peak/cluster gets a `"42.0% Foo; 18.0% Bar"` formatted string sorted by descending probability. Deterministic atlases produce a single region name per peak and mass-weighted percentages per cluster.

## Pre-thresholded maps (FDR / FWER)

If you've already thresholded via nilearn's `threshold_stats_img` (FDR, FWER, voxel-level inference), pass `stat_threshold=None` so `cluster_report` keeps every non-zero voxel:

```{code-cell} python3
from nilearn.glm import threshold_stats_img
from nltools import BrainData

thr_img, thresh = threshold_stats_img(
    contrast.to_nifti(),
    alpha=0.05,
    height_control="fdr",
    cluster_threshold=10,
)
report = BrainData(thr_img).cluster_report(stat_threshold=None, cluster_threshold=10)
```

## Coordinate-only labeling

If you have a list of MNI coordinates (e.g. peaks from a published paper) and just want anatomical labels, `label_coords` skips the cluster-finding step:

```{code-cell} python3
from nltools.data.atlases import label_coords

coords = [
    (-42, -22, 56),   # left M1/S1 hand
    (+42, -22, 56),   # right M1/S1 hand
    (0, -78, 8),      # medial occipital
]
label_coords(coords, atlas=["harvard_oxford", "aal"])
```

## Saving outputs

`ClusterReport` writes both tables to disk and renders an overview + per-cluster figures:

```{code-cell} python3
:tags: [skip-execution]
report.to_csv("./out/")            # peaks.csv + clusters.csv
report.plot(output_dir="./out/")    # overview.png + cluster_NN.png
```

Without `output_dir`, `plot()` returns the matplotlib figures directly so you can embed them in a notebook.

## Citing the atlases

Each registered atlas carries a `citation` field. Be sure to cite the original sources for atlases you use, plus AtlasReader (Notter et al. 2019, JOSS) for the labeling logic:

```{code-cell} python3
from nltools.data.atlases import load_atlas

atlas = load_atlas("harvard_oxford")
print(atlas.citation)
```

License terms for each bundled atlas are listed in `LICENSES.md` at https://huggingface.co/datasets/nltools/niftis/blob/main/atlases/LICENSES.md.
