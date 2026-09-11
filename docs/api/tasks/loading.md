---
title: Loading, masks & datasets
---

# Loading, masks & datasets

Get data into nltools. [BrainData](../data/brain_data.md) and the other data classes load NIfTI files and HDF5 bundles themselves. The functions here cover the rest: example datasets and Neurovault collections, sphere and ROI masks, `concatenate`, and the MNI template every object falls back on when it gets no mask (`set_brainspace`).

## HDF5 files

::: nltools.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - load_brain_data_h5
        - to_h5
        - is_h5_path

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

::: nltools.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - concatenate

## Templates and brain space

::: nltools.templates
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - BrainSpaceConfig
        - get_brainspace
        - set_brainspace
        - reset_brainspace
        - with_brainspace
        - fetch_resource
        - list_resources
        - get_bg_image
        - is_standard_space
        - detect_resolution
