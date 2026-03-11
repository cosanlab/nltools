# `nltools.pipelines.terminals`

**Terminal Operations**

Terminals define the final analysis step of a pipeline:

- `PredictTerminal` — Classification or regression decoding
- `ISCTerminal` — Inter-subject correlation analysis
- `RSATerminal` — Representational similarity analysis

Each terminal implements the `Terminal` protocol.

```{eval-rst}
.. autoclass:: nltools.pipelines.terminals.PredictTerminal
    :members:
    :show-inheritance:
    :exclude-members: y, algorithm, kwargs

.. autoclass:: nltools.pipelines.terminals.ISCTerminal
    :members:
    :show-inheritance:
    :exclude-members: method, metric, n_permute, parallel, kwargs

.. autoclass:: nltools.pipelines.terminals.RSATerminal
    :members:
    :show-inheritance:
    :exclude-members: model_rdm, method, n_permute, kwargs
```

## See Also

- {doc}`../pipelines` — Module overview
- {doc}`pipeline` — `Terminal` protocol definition
- {doc}`results` — Result containers returned by terminals
