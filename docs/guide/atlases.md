---
title: Atlases & cluster reports
---

Eleven parcellations ship with nltools, fetched from the `nltools/niftis` Hugging Face dataset on
first use. [`list_atlases`](../api/tasks/atlases.md#nltools.data.atlases.list_atlases) names them;
[`load_atlas`](../api/tasks/atlases.md#nltools.data.atlases.load_atlas) returns an
[`Atlas`](../api/tasks/atlases.md#nltools.data.atlases.Atlas): the image, a polars table of labels, the
kind (`'deterministic'` or `'probabilistic'`), and the citation you owe the original authors.
Probabilistic atlases (`harvard_oxford`, `juelich`) are 4-D, with one probability map per region;
the rest are integer-labeled volumes.

[`BrainData.cluster_report`](../api/data/brain_data.md#nltools.data.braindata.BrainData.cluster_report) answers what
you found: it thresholds a statistic map, finds its clusters, and labels each peak against three
atlases at once. It returns a [`ClusterReport`](../api/tasks/atlases.md#nltools.data.atlases.ClusterReport)
with `clusters` and `peaks` tables, the thresholded `stat_img`, a `plot()` for per-cluster figures,
and `to_csv()`.

Goal | Use | Notes
--- | --- | ---
See what is available | [`list_atlases`](../api/tasks/atlases.md#nltools.data.atlases.list_atlases) | `aal`, `harvard_oxford`, `schaefer_200`, `destrieux`, `juelich`, and seven more
Load one | [`load_atlas`](../api/tasks/atlases.md#nltools.data.atlases.load_atlas)`(name)` | `.image`, `.labels`, `.kind`, `.citation`
Label MNI coordinates | [`label_coords`](../api/tasks/atlases.md#nltools.data.atlases.label_coords) | `atlas=` takes one name or a sequence; `prob_threshold=` filters probabilistic hits
Cluster table | [`BrainData.cluster_report`](../api/data/brain_data.md#nltools.data.braindata.BrainData.cluster_report) | `stat_threshold=`, `cluster_threshold=` (voxels), `min_distance=` (mm between peaks)
Save or draw the report | [`ClusterReport.to_csv`](../api/tasks/atlases.md#nltools.data.atlases.ClusterReport.to_csv), [`.plot`](../api/tasks/atlases.md#nltools.data.atlases.ClusterReport.plot) | `plot(output_dir=)` writes one figure per cluster
Summarize per parcel | [`BrainData.extract_roi`](../api/data/brain_data.md#nltools.data.braindata.BrainData.extract_roi)`(mask, method='mean')` | `'median'` or `'pca'` (with `n_components=`); returns parcels × images
Paint values back | [`roi_to_brain_from_atlas`](../api/tasks/atlases.md#nltools.mask.roi_to_brain_from_atlas), [`roi_to_brain`](../api/tasks/loading.md#nltools.mask.roi_to_brain) | The atlas version takes a labeled volume; `roi_to_brain` takes expanded binary masks
Split a map into blobs | [`BrainData.regions`](../api/data/brain_data.md#nltools.data.braindata.BrainData.regions) | Connected-component decomposition of a thresholded map
Turn an atlas into masks | [`expand_mask`](../api/tasks/loading.md#nltools.mask.expand_mask) / [`collapse_mask`](../api/tasks/loading.md#nltools.mask.collapse_mask) | Labeled volume ↔ stack of binary masks

## Naming a result

```python
from nltools.data.atlases import label_coords, list_atlases, load_atlas

list_atlases()
atlas = load_atlas("harvard_oxford")
atlas.kind, atlas.labels, atlas.citation

report = stat_map.cluster_report(stat_threshold=3.0, cluster_threshold=10)
report.clusters      # one row per cluster, with anatomical labels
report.peaks         # one row per local maximum
report.stat_img      # the thresholded map

label_coords([[0, -20, 20], [-40, -60, 0]], atlas="harvard_oxford")
```

`cluster_report` labels against `DEFAULT_ATLASES`: Harvard-Oxford, AAL, and Schaefer-200. The
`clusters` table carries a column per atlas, so you can see where they disagree. `two_sided=True`
(the default) reports negative clusters too.

## Parcel summaries

```python
from nltools.data import BrainData
from nltools.mask import roi_to_brain_from_atlas
from nltools.templates import fetch_resource

k50 = BrainData(fetch_resource("masks/default/2mm-MNI152-2009fsl-k50.nii.gz"))
parcel_means = brain.extract_roi(k50, method="mean")     # (n_parcels, n_images)

painted = roi_to_brain_from_atlas(
    parcel_means.mean(axis=1), k50, source_mask=k50.mask
)
```

`extract_roi` collapses each parcel to one number per image (`method='pca'` keeps
`n_components` per parcel instead); `roi_to_brain_from_atlas` does the inverse, filling every voxel
of parcel *i* with value *i*. That round trip turns per-region results, such as decoding accuracies,
RSA correlations, or ISC values, into a brain map.

`regions` is the other direction: it splits an already-thresholded map into spatially connected
blobs, one image per region.

```python
regions = stat_map.threshold(upper=3.0).regions(min_region_size=1350)
```

## Gotchas

- Atlases and bundled masks download on first use and are cached by `huggingface_hub`. The first
  call in a fresh environment needs a network.
- An MNI-space atlas is resampled onto your data by header affine alone. That is a grid change, not
  a spatial normalization, so parcel boundaries are approximate for un-normalized data.
- `atlas.labels` is a polars DataFrame with `index` and `name` columns. `index` is the integer
  stored in the image, not a row number: AAL's first region is `2001`. Join on it rather than
  assuming positions line up.

Next: the [Tutorials](../tutorials/index.md), where these calls appear inside worked analyses.
