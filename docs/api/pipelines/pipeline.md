# `nltools.pipelines.base`

**Core Pipeline Classes and Protocols**

The `Pipeline` class composes transform steps into a sequential processing chain.
`FittedStack` holds the fitted state of a pipeline after training.

The module also defines the protocols (`TransformStep`, `FittedTransform`, `CVScheme`, `Terminal`) that all pipeline components must implement.

```{eval-rst}
.. autoclass:: nltools.pipelines.base.TransformStep
    :members:
    :exclude-members: invertible

.. autoclass:: nltools.pipelines.base.FittedTransform
    :members:

.. autoclass:: nltools.pipelines.base.CVScheme
    :members:

.. autoclass:: nltools.pipelines.base.Terminal
    :members:

.. autoclass:: nltools.pipelines.base.FittedStack
    :members:
    :exclude-members: steps

.. autoclass:: nltools.pipelines.base.Pipeline
    :members:
    :exclude-members: data, cv, steps
```

## See Also

- {doc}`../pipelines` — Module overview
- {doc}`steps` — Built-in transform steps
- {doc}`terminals` — Terminal operations
