# `nltools.pipelines.multi_subject`

**Multi-Subject Pipeline Execution**

`MultiSubjectPipeline` orchestrates leave-one-subject-out (LOSO) and other multi-subject cross-validation workflows. It takes a `Pipeline` and a `CVScheme`, fits the pipeline on training subjects, and evaluates on held-out subjects.

```{eval-rst}
.. autoclass:: nltools.pipelines.multi_subject.MultiSubjectPipeline
    :members:
    :show-inheritance:
    :exclude-members: data, cv, groups, steps
```

## See Also

- {doc}`../pipelines` — Module overview
- {doc}`pipeline` — Core `Pipeline` class
- {doc}`cv` — Cross-validation schemes
- {doc}`results` — Result containers
