---
title: Brain plotting
---

# Brain plotting

Render a volume on the cortical surface, as a flatmap, or in an interactive viewer, and browse ICA/PCA components. `BrainData.plot` and `BrainData.iplot` call these. Plots of model output sit with their workflow. ROC and prediction plots are under [Prediction & cross-validation](prediction.md); adjacency-matrix plots are under [Similarity & RSA](similarity.md).

::: nltools.plotting
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - plot_surf
        - plot_flatmap
        - plot_interactive_brain
        - component_viewer
