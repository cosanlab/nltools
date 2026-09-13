---
title: Atlases & cluster reports
---

# Atlases & cluster reports

Put anatomical names on a result. `list_atlases` and `load_atlas` fetch parcellations from the nltools Hugging Face dataset on first use, and `label_coords` looks MNI coordinates up in them. `BrainData.cluster_report` thresholds a statistical map and labels each cluster's peak. `roi_to_brain_from_atlas` paints per-parcel values back into a volume.

## Atlases

::: nltools.datasets
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - list_atlases
        - load_atlas
        - label_coords

## Mapping parcel values back to a volume

::: nltools.mask
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - roi_to_brain_from_atlas
