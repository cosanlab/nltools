# nilearn.reporting — Reporting Functions

Helpers for generating cluster tables and HTML reports of analysis results. The primary GLM report path is now `model.generate_report()`; `make_glm_report` is retained but deprecated.

**Source:** https://nilearn.github.io/dev/modules/reporting.html

## Inventory

### Classes
| Class | Purpose |
|---|---|
| `HTMLReport` | A report rendered as HTML; supports `.save_as_html(path)` and `_repr_html_` for notebooks. |

### Functions
| Function | Purpose |
|---|---|
| `get_clusters_table(stat_img, stat_threshold, ...)` | Create a pandas DataFrame with cluster statistics for a stat map. |
| `make_glm_report(model, contrasts=None, ...)` | Return an `HTMLReport` of a fitted GLM. **Deprecated — prefer `model.generate_report()`.** |

## get_clusters_table

```python
from nilearn.reporting import get_clusters_table

table = get_clusters_table(
    stat_img,
    stat_threshold=3.0,             # voxel-level threshold (z, t, ...)
    cluster_threshold=0,            # min cluster size in voxels
    two_sided=False,                # also report negative clusters
    min_distance=8.0,               # mm; merges peaks within this radius
)
# Columns: Cluster ID, X, Y, Z (mm), Peak Stat, Cluster Size (mm3)
```

Sub-peaks within each cluster are reported as fractional rows (e.g., `1a`, `1b`).

## HTMLReport

```python
report = flm.generate_report(contrasts=['face - house'])
report.save_as_html('flm_report.html')
report.open_in_browser()             # opens in default browser
str(report)                          # raw HTML string
```

In Jupyter, displaying the report object renders inline.

## make_glm_report (deprecated)

```python
from nilearn.reporting import make_glm_report

# Prefer model.generate_report(...) for new code
report = make_glm_report(model, contrasts={'face - house': 'face - house'},
                          title='First-level GLM',
                          bg_img='MNI152TEMPLATE',
                          threshold=3.09, alpha=0.001,
                          cluster_threshold=15,
                          height_control='fpr',
                          min_distance=8.0,
                          plot_type='slice')
```

## Common patterns

Cluster table after thresholding:

```python
from nilearn.reporting import get_clusters_table
from nilearn.glm import threshold_stats_img

thr_img, thresh = threshold_stats_img(z_map, alpha=0.05,
                                       height_control='fdr',
                                       cluster_threshold=10)
table = get_clusters_table(thr_img, stat_threshold=thresh,
                            cluster_threshold=10, two_sided=True,
                            min_distance=8.0)
```

GLM report from a fitted model:

```python
report = flm.generate_report(
    contrasts=['face - house', 'house - face'],
    title='Subject 01 first-level',
    threshold=3.09, alpha=0.001, cluster_threshold=15,
    height_control='fpr',
)
report.save_as_html('sub-01_report.html')
```

Second-level report:

```python
slm.fit(z_maps, design_matrix=group_dm)
slm.generate_report(contrasts=['group_mean']).save_as_html('group_report.html')
```

## Gotchas

- `make_glm_report` is deprecated — call `model.generate_report(...)` on `FirstLevelModel` / `SecondLevelModel` instead. Same kwargs.
- `get_clusters_table` returns a DataFrame with one row per cluster (and possibly fractional rows for sub-peaks); `Cluster ID` like `1a`, `1b` are sub-peaks of cluster 1.
- `two_sided=True` reports negative clusters as separate rows with negative peak values.
- `min_distance` is in **mm**, not voxels — sub-peaks within `min_distance` of a stronger peak in the same cluster are merged.
- `stat_threshold` is voxel-level only; cluster-extent thresholding requires the separate `cluster_threshold` argument.
- HTML reports embed a lot of base64-encoded images and can be large (10-50 MB) — fine to ship, but don't commit them to git.

## See also

- `nilearn.glm.threshold_stats_img`, `cluster_level_inference`, `fdr_threshold`.
- `FirstLevelModel.generate_report`, `SecondLevelModel.generate_report`.
- All `*Masker.generate_report()` methods (rendered the same way).
- https://nilearn.github.io/dev/modules/reporting.html
