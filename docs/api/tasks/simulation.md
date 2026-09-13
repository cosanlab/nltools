---
title: Simulation
---

# Simulation

Synthetic data with a known signal, for testing a pipeline end to end. `Simulator` builds [BrainData](../data/brain_data.md) with Gaussian-blob signal, several subjects, and noise. `SimulateGrid` builds 2D grids, which is enough to exercise thresholding and multiple-comparison correction without a mask.

::: nltools.data.simulator
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - Simulator
        - SimulateGrid
