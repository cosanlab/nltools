# `nltools.pipelines`

**Composable Multi-Subject Neuroimaging Pipelines**

The pipelines module provides a declarative framework for building reproducible neuroimaging analysis workflows. Pipelines compose transform steps (normalization, dimensionality reduction, alignment) with terminal operations (prediction, ISC, RSA) and cross-validation schemes.

## Architecture

A pipeline consists of three components:

1. **Steps** — Transform operations applied sequentially ({doc}`pipelines/steps`)
2. **Terminal** — The final analysis operation ({doc}`pipelines/terminals`)
3. **CV Scheme** — Cross-validation strategy ({doc}`pipelines/cv`)

## Quick Start

```python
from nltools.pipelines import (
    Pipeline,
    MultiSubjectPipeline,
    NormalizeStep,
    ReduceStep,
    AlignStep,
    PredictTerminal,
    CVScheme,
)

# Define a decoding pipeline
pipe = Pipeline(
    steps=[NormalizeStep(), ReduceStep(n_components=50), AlignStep()],
    terminal=PredictTerminal(),
)

# Run with cross-validation across subjects
ms = MultiSubjectPipeline(pipe, cv=CVScheme.leave_one_subject_out())
results = ms.fit(subjects_data, labels)
```

## Sub-pages

- {doc}`pipelines/pipeline` — Core `Pipeline` and `FittedStack` classes, protocols
- {doc}`pipelines/multi_subject` — `MultiSubjectPipeline` for leave-one-subject-out workflows
- {doc}`pipelines/cv` — `CVScheme` and `NestedCVScheme` configuration
- {doc}`pipelines/steps` — Transform steps (`NormalizeStep`, `ReduceStep`, `PipeStep`, `AlignStep`)
- {doc}`pipelines/terminals` — Terminal operations (`PredictTerminal`, `ISCTerminal`, `RSATerminal`)
- {doc}`pipelines/results` — Result containers (`CVResult`, `FoldResult`, `ISCResult`, `RSAResult`, etc.)
