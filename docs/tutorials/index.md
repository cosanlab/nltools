---
title: Tutorials
---

Learn how to use nltools through hands-on examples. Start with the **Basics** to understand core data structures, then explore **Workflows** for complete analysis pipelines.

Each tutorial runs end-to-end and shows its outputs inline. Every one is also a **live, interactive notebook** you can run entirely in your browser (via marimo + WebAssembly) — no install, no server — via the "Open interactive version" link at the top of each page.

### Basics

These tutorials introduce the four core data classes in nltools. Each one covers creation, manipulation, and common operations.

::::{grid} 1 2 2 2

:::{grid-item-card} BrainData Basics
:link: basics/01_brain_data
:link-type: doc
Loading neuroimaging data, basic operations (indexing, slicing, arithmetic), and summary statistics.
:::

:::{grid-item-card} DesignMatrix Basics
:link: basics/02_design_matrix
:link-type: doc
Creating design matrices, building task regressors, HRF convolution, and nuisance covariates.
:::

:::{grid-item-card} Adjacency Basics
:link: basics/03_adjacency
:link-type: doc
Creating Adjacency objects, square vs. vector forms, thresholding, and binarizing matrices.
:::

:::{grid-item-card} BrainCollection Basics
:link: basics/04_brain_collection
:link-type: doc
A parallel, memory-efficient iterator of BrainData: per-subject operations in parallel, path-backed caching, and group reductions across subjects.
:::

::::

### Workflows

End-to-end analysis workflows that demonstrate how nltools classes and functions work together for real neuroimaging analyses.

::::{grid} 1 2 2 2

:::{grid-item-card} GLM Analysis
:link: workflows/01_glm
:link-type: doc
First- and second-level GLM: build designs, fit models, compute contrasts, and run group statistics with multiple-comparisons correction.
:::

:::{grid-item-card} Encoding Models
:link: workflows/02_encoding
:link-type: doc
Predict brain activity from stimulus features with an FIR feature bank and ridge regression, scored by cross-validated R².
:::

:::{grid-item-card} Multivariate Pattern Analysis
:link: workflows/03_mvpa
:link-type: doc
Decoding and RSA across whole-brain, ROI, and searchlight scales.
:::

:::{grid-item-card} Inter-Subject Correlation
:link: workflows/04_isc
:link-type: doc
Shared, time-locked responses to naturalistic movies, measured across subjects.
:::

::::

## Learning Resources

## [Dartbrains](https://dartbrains.org)

A fundamentals of neuroimaging undergraduate level course

## [Naturalistic Data](https://naturalistic-data.org)

A more advanced neuroimaging course for working with *naturalistic* datasets (e.g. watching movies, playing games, etc).

## [Discourse Community](https://www.askpbs.org/c/nltools/13)

A Stack Overflow like forum where you can view, contribute, and vote on FAQs regarding `nltools`. Please ask questions here *first* so that other users can benefit from the answers!

