# `nltools.pipelines.cv`

**Cross-Validation Schemes**

`CVScheme` configures how subjects are split into train/test folds.
`NestedCVScheme` provides nested cross-validation for hyperparameter tuning within the outer CV loop.

```{eval-rst}
.. autoclass:: nltools.pipelines.cv.CVScheme
    :members:
    :show-inheritance:
    :exclude-members: k, scheme, split_by, n, random_state

.. autoclass:: nltools.pipelines.cv.NestedCVScheme
    :members:
    :show-inheritance:
    :exclude-members: outer, inner
```

## See Also

- {doc}`../pipelines` — Module overview
- {doc}`multi_subject` — Multi-subject pipeline execution
- {doc}`pipeline` — `CVScheme` protocol definition in `base`
