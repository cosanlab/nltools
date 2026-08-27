(algorithms-regression-regression)=
## `regression`

Standalone OLS regression on numpy arrays.

Pedagogical helper used in tutorials and notebooks where callers want a
``(b, se, t, p, df, res)`` tuple from a design matrix ``X`` and response
``Y`` without constructing a `BrainData` or `Glm`. For
4D neuroimaging data use `BrainData.fit` with ``model='glm'``.

**Methods:**

Name | Description
---- | -----------
[`regress`](#algorithms-regression-regress) | Fit an OLS regression of ``Y`` on ``X``.



### Methods

(algorithms-regression-regress)=
#### `regress`

```python
regress(X, Y, *, method: str = 'ols', stats: str = 'full', tail: int | str = 2)
```

Fit an OLS regression of ``Y`` on ``X``.

Does not add an intercept — include one in ``X`` explicitly. If ``Y``
is 2D, a separate regression is fit to each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | Design matrix, shape ``(n_samples, n_regressors)``. | *required*
`Y` |  | Response, shape ``(n_samples,)`` or ``(n_samples, n_targets)``. | *required*
`method` | <code>[str](#str)</code> | Only ``'ols'`` is supported in v0.6.0. The legacy ``'robust'`` and ``'arma'`` methods were dropped; use statsmodels or a dedicated package if you need them. | <code>'ols'</code>
`stats` | <code>[str](#str)</code> | ``'full'`` returns the 6-tuple below; ``'betas'`` returns just ``b``; ``'tstats'`` returns ``(b, t)``. | <code>'full'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: beta > 0; negate a regressor for the other direction). | <code>2</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | ``(b, se, t, p, df, res)`` when ``stats='full'``:
 |  | - ``b``: coefficients
 |  | - ``se``: standard errors
 |  | - ``t``: t-statistics
 |  | - ``p``: p-values (per ``tail``)
 |  | - ``df``: residual degrees of freedom
 |  | - ``res``: residuals

