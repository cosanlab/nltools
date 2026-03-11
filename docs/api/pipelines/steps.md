# `nltools.pipelines.steps`

**Transform Steps**

Built-in transform steps for pipeline composition:

- `NormalizeStep` / `FittedNormalize` — Z-scoring and mean-centering
- `ReduceStep` / `FittedReduce` — Dimensionality reduction (PCA/SVD)
- `PipeStep` / `FittedPipe` — Wrap arbitrary sklearn-compatible transformers
- `AlignStep` / `FittedAlign` — Functional alignment (hyperalignment, SRM)

Each step implements the `TransformStep` protocol and produces a corresponding `FittedTransform` object.

```{eval-rst}
.. autoclass:: nltools.pipelines.steps.NormalizeStep
    :members:
    :show-inheritance:
    :exclude-members: method, axis, invertible

.. autoclass:: nltools.pipelines.steps.FittedNormalize
    :members:
    :show-inheritance:
    :exclude-members: mean, std, method

.. autoclass:: nltools.pipelines.steps.ReduceStep
    :members:
    :show-inheritance:
    :exclude-members: method, n_components, random_state

.. autoclass:: nltools.pipelines.steps.FittedReduce
    :members:
    :show-inheritance:
    :exclude-members: model, method

.. autoclass:: nltools.pipelines.steps.PipeStep
    :members:
    :show-inheritance:
    :exclude-members: transformer

.. autoclass:: nltools.pipelines.steps.FittedPipe
    :members:
    :show-inheritance:
    :exclude-members: transformer

.. autoclass:: nltools.pipelines.steps.AlignStep
    :members:
    :show-inheritance:

.. autoclass:: nltools.pipelines.steps.FittedAlign
    :members:
    :show-inheritance:
    :exclude-members: model, method, new_subject_method
```

## See Also

- {doc}`../pipelines` — Module overview
- {doc}`pipeline` — `TransformStep` and `FittedTransform` protocols
