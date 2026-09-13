---
title: Result types
---

# Result types

What the analysis methods hand back. These are read, not constructed:
`BrainData.fit` leaves a `FitResult` on `BrainData.model`, `BrainData.predict`
returns a `PredictResult`, `BrainData.bootstrap` a `BootstrapResult`, a GLM
contrast a `ContrastResult`, and the brain-space functions a `BrainSpaceConfig`.
The fit and contrast records also write themselves to a directory of NIfTI maps
with `write`.

::: nltools.data
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - FitResult
        - PredictResult
        - BootstrapResult
        - ContrastResult
        - BrainSpaceConfig
