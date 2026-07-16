---
title: Tutorials
---

Learn how to use nltools through hands-on examples. Start with the **Basics** to understand core data structures, then explore **Workflows** for complete analysis pipelines.

Each tutorial is a **live, interactive notebook** that runs entirely in your browser (via marimo + WebAssembly) — no install, no server. The first load boots a Python kernel and downloads the scientific stack, which takes a minute; after that it's cached. Open one and edit any cell to explore.

### Basics

These tutorials introduce the four core data classes in nltools. Each one covers creation, manipulation, and common operations.

::::{grid} 1 2 2 2

:::{grid-item-card} BrainData Basics
:link: /tutorials/basics-01_brain_data.html
Loading neuroimaging data, basic operations (indexing, slicing, arithmetic), and summary statistics.
:::

:::{grid-item-card} DesignMatrix Basics
:link: /tutorials/basics-02_design_matrix.html
Creating design matrices, building task regressors, HRF convolution, and nuisance covariates.
:::

:::{grid-item-card} Adjacency Basics
:link: /tutorials/basics-03_adjacency.html
Creating Adjacency objects, square vs. vector forms, thresholding, and binarizing matrices.
:::

::::

### Workflows

End-to-end analysis workflows that demonstrate how nltools classes and functions work together for real neuroimaging analyses.

::::{grid} 1 2 2 2

:::{grid-item-card} GLM Analysis
:link: /tutorials/workflows-01_glm.html
First- and second-level GLM: build designs, fit models, compute contrasts, and run group statistics with multiple-comparisons correction.
:::

:::{grid-item-card} Encoding Models
:link: /tutorials/workflows-02_encoding.html
Predict brain activity from stimulus features with an FIR feature bank and ridge regression, scored by cross-validated R².
:::

:::{grid-item-card} Multivariate Pattern Analysis
:link: /tutorials/workflows-03_mvpa.html
Decoding and RSA across whole-brain, ROI, and searchlight scales.
:::

:::{grid-item-card} Inter-Subject Correlation
:link: /tutorials/workflows-04_isc.html
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

