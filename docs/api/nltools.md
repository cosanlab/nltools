---
title: nltools
---

# nltools

The top-level namespace. The six data classes are re-exported here and each has
its own page: [`BrainData`](data/brain_data.md), [`Adjacency`](data/adjacency.md),
[`DesignMatrix`](data/design_matrix.md), [`Roc`](data/roc.md),
[`Simulator` and `SimulateGrid`](data/simulator.md). What follows is everything
else `import nltools` gives you: the function that combines objects, the four
functions that read and set the brain space every object falls back on, and the
package version.

::: nltools
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - concatenate
        - get_brainspace
        - set_brainspace
        - reset_brainspace
        - with_brainspace
        - __version__
