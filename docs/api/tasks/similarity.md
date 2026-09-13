---
title: Similarity & RSA
---

# Similarity & RSA

Compare patterns and matrices. `compute_similarity` scores two arrays under a `metric=`; the Fisher transforms make correlations averageable. `matrix_permutation_test` (Mantel), `correlation_permutation_test`, and `distance_correlation` compare whole matrices. Stacks of [Adjacency](../data/adjacency.md) matrices are summarized by the `Adjacency` plotting methods. Map per-ROI values with `roi_to_brain_from_atlas` and an explicit aligned atlas and label order; map searchlight values with the source mask in voxel order.

## Similarity and matrix statistics

::: nltools.algorithms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - compute_similarity
        - transform_pairwise
        - fisher_r_to_z
        - fisher_z_to_r
        - matrix_permutation_test
        - correlation_permutation_test
        - distance_correlation
