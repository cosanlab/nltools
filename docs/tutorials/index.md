---
title: Tutorials
---

Learn how to use nltools through hands-on examples. Start with the **Basics** to understand core data structures, then explore **Workflows** for complete analysis pipelines.

### Basics

These tutorials introduce the four core data classes in nltools. Each one covers creation, manipulation, and common operations.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} BrainData Basics
:link: basics/01_brain_data
Loading neuroimaging data, basic operations (indexing, slicing, arithmetic), and summary statistics.
:::

:::{grid-item-card} DesignMatrix Basics
:link: basics/02_design_matrix
Creating design matrices, building task regressors, HRF convolution, and nuisance covariates.
:::

:::{grid-item-card} Adjacency Basics
:link: basics/03_adjacency
Creating Adjacency objects, square vs. vector forms, thresholding, and binarizing matrices.
:::

:::{grid-item-card} BrainCollection Basics
:link: basics/04_brain_collection
Creating BrainCollection objects, 3-axis indexing, and group-level aggregations.
:::

::::

### Workflows

End-to-end analysis workflows that demonstrate how nltools classes and functions work together for real neuroimaging analyses.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} First-Level GLM Analysis
:link: workflows/01_glm
Complete GLM workflow with BrainData and DesignMatrix --- fitting models and computing contrasts.
:::

:::{grid-item-card} Group Analysis
:link: workflows/02_group_analysis
First-level GLM per subject, then second-level statistics across subjects.
:::

:::{grid-item-card} Representational Similarity Analysis
:link: workflows/04_rsa
Linking disparate data types via shared structure in similarity matrices.
:::

:::{grid-item-card} Multivariate Pattern Analysis (MVPA)
:link: workflows/05_decoding
Face vs. house classification using the Haxby dataset.
:::

:::{grid-item-card} Pipeline Basics
:link: workflows/06_pipeline_basics
The nltools v0.6 fluent pipeline API for cross-validated analysis workflows.
:::

:::{grid-item-card} Two-Stage GLM Analysis
:link: workflows/07_two_stage_glm
Summary statistics approach --- first-stage GLM per subject, second-stage group tests.
:::

:::{grid-item-card} Encoding Models
:link: workflows/08_encoding_models
Predicting brain activity from stimulus features using GLM and Ridge regression.
:::

:::{grid-item-card} Multi-Subject Classification
:link: workflows/08_multi_subject_decoding
Leave-One-Subject-Out cross-validation for group classification.
:::

:::{grid-item-card} Inter-Subject Correlation (ISC)
:link: workflows/09_isc_analysis
Measuring similarity of brain responses across subjects viewing naturalistic stimuli.
:::

:::{grid-item-card} Representational Similarity Analysis (RSA)
:link: workflows/10_rsa_analysis
Comparing similarity structure of neural representations with theoretical models.
:::

:::{grid-item-card} Cluster Reports & Atlas Labeling
:link: workflows/11_cluster_reports
Summarizing thresholded stat maps with peak coordinates, sub-peaks, and anatomical labels from 11 bundled atlases.
:::

::::

## Learning Resources

## [Dartbrains](https://dartbrains.org)

A fundamentals of neuroimaging undergraduate level course

## [Naturalistic Data](https://naturalistic-data.org)

A more advanced neuroimaging course for working with *naturalistic* datasets (e.g. watching movies, playing games, etc).

## [Discourse Community](https://www.askpbs.org/c/nltools/13)

A Stack Overflow like forum where you can view, contribute, and vote on FAQs regarding `nltools`. Please ask questions here *first* so that other users can benefit from the answers!

