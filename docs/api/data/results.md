---
title: Result types
---

# Result types

What the analysis methods hand back. These are read, not constructed:
`BrainData.predict` returns a `Predict`, `BrainData.bootstrap` a
`BootstrapResult`, a GLM contrast a `ContrastResult`, and the brain-space
functions a `BrainSpaceConfig`.

::: nltools.data
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - Predict
        - BootstrapResult
        - ContrastResult
        - BrainSpaceConfig
