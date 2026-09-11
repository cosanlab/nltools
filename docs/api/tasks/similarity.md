---
title: Similarity & RSA
---

# Similarity & RSA

Compare patterns and matrices. `compute_similarity` scores two arrays under a `metric=`; the Fisher transforms make correlations averageable. `matrix_permutation_test` (Mantel), `correlation_permutation_test`, and `distance_correlation` compare whole matrices. The plots summarize stacks of [Adjacency](../data/adjacency.md) matrices. Map per-ROI values with `roi_to_brain_from_atlas` and an explicit aligned atlas and label order; map searchlight values with the source mask in voxel order.

## Similarity and matrix statistics

::: nltools.algorithms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - compute_similarity
        - compute_multivariate_similarity
        - transform_pairwise
        - fisher_r_to_z
        - fisher_z_to_r
        - matrix_permutation_test
        - correlation_permutation_test
        - distance_correlation
        - double_center
        - u_center

## Plots

::: nltools.plotting
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - plot_stacked_adjacency
        - plot_mean_label_distance
        - plot_between_label_distance
        - plot_silhouette
