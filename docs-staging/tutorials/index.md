---
title: Tutorials
---

Learn how to use nltools through hands-on examples. Start with the **Basics** for the core data structures, work through **Data operations** for the everyday tasks that surround an analysis, pick the **Analysis** that answers your question, then read a **Workflow** end to end.

Tutorials are worked analyses you read start to finish; when you already know what you want to do and just need the right call and its gotchas, go to the [User Guide](../guide/index.md) instead.

Every tutorial is a [marimo](https://marimo.io) notebook under [`docs/tutorials/`](https://github.com/cosanlab/nltools/tree/master/docs/tutorials), and each page runs its notebook when the site is built, so the figures, tables and printed output below the code are what that code produced. To run and edit the cells yourself, download the `.py` and open it with `uvx marimo edit --sandbox <notebook>.py`, or run it in the cloud with nothing to install through the **Open in molab** link on each page.

## Basics

These tutorials introduce the three core data classes in nltools. Each one covers creation, manipulation, and common operations.

- [BrainData Basics](basics/01_brain_data.md) — loading neuroimaging data, basic operations (indexing, slicing, arithmetic), and summary statistics.
- [DesignMatrix Basics](basics/02_design_matrix.md) — creating design matrices, building task regressors, HRF convolution, and nuisance covariates.
- [Adjacency Basics](basics/03_adjacency.md) — creating Adjacency objects, square vs. vector forms, thresholding, and binarizing matrices.

## Data operations

Task-focused walkthroughs of the everyday operations: getting data in, restricting it to the voxels you care about, and building the tables an analysis consumes.

- [Basic Data Operations](data-operations/01_download.md) — download a dataset, then index, slice, combine, save and plot it.
- [Masking](data-operations/02_masking.md) — spheres, parcellations and thresholded maps as masks, and mapping per-region results back onto the brain.
- [Brain Space and Resolution](data-operations/03_brain_space.md) — the template and resolution every object sits on, set globally, scoped to a block, or per object.
- [NeuroVault I/O](data-operations/04_neurovault.md) — download a collection or a single image, and upload your own maps.
- [Design Matrices](data-operations/05_design_matrix.md) — build a design by hand or from onsets files, convolve it, stack runs, and check it is estimable.
- [Adjacency Matrices](data-operations/06_adjacency.md) — similarity and distance matrices, regression over edges, MDS, and graphs.

## Analysis

One analysis family per page: the question it answers, the call that runs it, and how to read what comes back.

- [Univariate Regression](analysis/01_univariate_regression.md) — a two-level GLM: one regression per subject, then a group test with FDR correction.
- [Multivariate Prediction](analysis/02_multivariate_prediction.md) — predict a continuous variable from the whole pattern, with estimators and cross-validation schemes compared.
- [Multivariate Classification](analysis/03_multivariate_classification.md) — decode two conditions, then evaluate with single-interval and forced-choice ROC.
- [Similarity and Distance](analysis/04_similarity.md) — pairwise distance between images, and the response of each image to one pattern.
- [Functional Alignment](analysis/05_hyperalignment.md) — hyperalignment and the shared response model, including back-projection and adding a new subject.
- [Decomposition](analysis/06_decomposition.md) — factor a dataset into components and see what they track.

## Workflows

End-to-end analysis workflows that demonstrate how nltools classes and functions work together for real neuroimaging analyses.

- [GLM Analysis](workflows/01_glm.md) — first- and second-level GLM: build designs, fit models, compute contrasts, and run group statistics with multiple-comparisons correction.
- [Encoding Models](workflows/02_encoding.md) — predict brain activity from stimulus features with an FIR feature bank and ridge regression, scored by cross-validated R².
- [Multivariate Pattern Analysis](workflows/03_mvpa.md) — decoding and RSA across whole-brain, ROI, and searchlight scales.
- [Inter-Subject Correlation](workflows/04_isc.md) — shared, time-locked responses to naturalistic movies, measured across subjects.

## Learning nltools

- [DartBrains](https://dartbrains.org) — an undergraduate-level course on the fundamentals of neuroimaging, taught with nltools.
- [Naturalistic Data](https://naturalistic-data.org) — a more advanced course on working with *naturalistic* datasets (watching movies, playing games, etc.).
- [Discourse community](https://www.askpbs.org/c/nltools/13) — a Stack Overflow-like forum where you can view, contribute, and vote on FAQs about nltools. Ask questions here *first* so other users benefit from the answers.
- [GitHub issues](https://github.com/cosanlab/nltools/issues) — for bugs and anything else code-related.
