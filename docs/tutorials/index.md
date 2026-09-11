---
title: Tutorials
---

Learn how to use nltools through hands-on examples. Start with the **Basics** to understand core data structures, then explore **Workflows** for complete analysis pipelines.

Tutorials are worked analyses you read start to finish; when you already know what you want to do and just need the right call and its gotchas, go to the [User Guide](../guide/index.md) instead.

Every tutorial is a [marimo](https://marimo.io) notebook under [`docs/tutorials/`](https://github.com/cosanlab/nltools/tree/master/docs/tutorials). Download the `.py` and open it with `uvx marimo edit --sandbox <notebook>.py` to run and edit every cell yourself, or run it in the cloud with nothing to install through the **Open in molab** link on each page.

## Basics

These tutorials introduce the three core data classes in nltools. Each one covers creation, manipulation, and common operations.

- [BrainData Basics](basics/01-brain-data.md) — loading neuroimaging data, basic operations (indexing, slicing, arithmetic), and summary statistics.
- [DesignMatrix Basics](basics/02-design-matrix.md) — creating design matrices, building task regressors, HRF convolution, and nuisance covariates.
- [Adjacency Basics](basics/03-adjacency.md) — creating Adjacency objects, square vs. vector forms, thresholding, and binarizing matrices.

## Workflows

End-to-end analysis workflows that demonstrate how nltools classes and functions work together for real neuroimaging analyses.

- [GLM Analysis](workflows/01-glm.md) — first- and second-level GLM: build designs, fit models, compute contrasts, and run group statistics with multiple-comparisons correction.
- [Encoding Models](workflows/02-encoding.md) — predict brain activity from stimulus features with an FIR feature bank and ridge regression, scored by cross-validated R².
- [Multivariate Pattern Analysis](workflows/03-mvpa.md) — decoding and RSA across whole-brain, ROI, and searchlight scales.
- [Inter-Subject Correlation](workflows/04-isc.md) — shared, time-locked responses to naturalistic movies, measured across subjects.

## Learning Resources

- [Dartbrains](https://dartbrains.org) — an undergraduate-level course on the fundamentals of neuroimaging.
- [Naturalistic Data](https://naturalistic-data.org) — a more advanced course on working with *naturalistic* datasets (watching movies, playing games, etc.).
- [Discourse Community](https://www.askpbs.org/c/nltools/13) — a Stack Overflow-like forum where you can view, contribute, and vote on FAQs about `nltools`. Ask questions here *first* so other users can benefit from the answers.
