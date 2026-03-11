# `nltools.pipelines.results` & `nltools.pipelines.pool`

**Result Containers**

Data classes that hold pipeline outputs:

**From `results`:**
`FoldResult`, `CVResult`, `ISCResult`, `RSAResult`, `PermutationResult`

**From `pool`:**
`PooledData`, `StatResult`, `ResultDict`

```{eval-rst}
.. autoclass:: nltools.pipelines.results.FoldResult
    :members:
    :show-inheritance:
    :exclude-members: score, predictions, train_idx, test_idx, fitted_stack

.. autoclass:: nltools.pipelines.results.CVResult
    :members:
    :show-inheritance:
    :exclude-members: fold_results, pipeline

.. autoclass:: nltools.pipelines.results.ISCResult
    :members:
    :show-inheritance:
    :exclude-members: isc, p, ci, method, metric, n_subjects

.. autoclass:: nltools.pipelines.results.RSAResult
    :members:
    :show-inheritance:
    :exclude-members: correlation, p_value, ci, method, n_conditions

.. autoclass:: nltools.pipelines.results.PermutationResult
    :members:
    :show-inheritance:
    :exclude-members: observed, null_distribution, p_value, n_permutations

.. autoclass:: nltools.pipelines.pool.PooledData
    :members:
    :show-inheritance:
    :exclude-members: data, param, condition_names, subject_ids, mask, fitted_state, save_path

.. autoclass:: nltools.pipelines.pool.StatResult
    :members:
    :show-inheritance:
    :exclude-members: t_map, f_map, p_map, contrast, df

.. autoclass:: nltools.pipelines.pool.ResultDict
    :members:
    :show-inheritance:
```

## See Also

- {doc}`../pipelines` — Module overview
- {doc}`terminals` — Terminal operations that produce these results
