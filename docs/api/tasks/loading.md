---
title: Loading, masks & datasets
---

# Loading, masks & datasets

Get data into nltools. [BrainData](../data/brain_data.md) and the other data classes load NIfTI files and HDF5 bundles themselves. The functions here cover the rest: example datasets, bundled resources and Neurovault collections, sphere and ROI masks, `concatenate`, and the MNI template every object falls back on when it gets no mask (`set_brainspace`).

## Example datasets

::: nltools.datasets
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - fetch_pain
        - fetch_emotion_ratings
        - fetch_neurovault_collection
        - load_haxby_example
        - download_nifti
        - get_resource_path
        - fetch_resource
        - list_resources

## Masks and ROIs

::: nltools.mask
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - create_sphere
        - expand_mask
        - collapse_mask
        - roi_to_brain

## Combining objects

::: nltools
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - concatenate

## Templates and brain space

::: nltools
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - get_brainspace
        - set_brainspace
        - reset_brainspace
        - with_brainspace

::: nltools.data
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - BrainSpaceConfig
