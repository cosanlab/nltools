# nilearn.mass_univariate — Mass-Univariate Analysis

A massively univariate linear model estimated with OLS and a permutation test. The single public entry point is `permuted_ols`, used for non-parametric group inference (with optional TFCE) over many features.

**Source:** https://nilearn.github.io/dev/modules/mass_univariate.html

## Inventory

### Functions
| Function | Purpose |
|---|---|
| `permuted_ols(tested_vars, target_vars, ...)` | Massively univariate group analysis with permuted OLS. |

## permuted_ols

```python
from nilearn.mass_univariate import permuted_ols

out = permuted_ols(
    tested_vars,                    # (n_samples, n_regressors) — variable(s) of interest
    target_vars,                    # (n_samples, n_features) — voxels/regions
    confounding_vars=None,          # (n_samples, n_covariates) — nuisance regressors
    model_intercept=True,
    n_perm=10000,
    two_sided_test=True,
    random_state=None,
    n_jobs=1,
    verbose=0,
    masker=None,                    # required if tfce=True or threshold is not None
    tfce=False,                     # threshold-free cluster enhancement
    threshold=None,                 # cluster-forming voxel-wise threshold
    output_type='legacy',           # 'legacy'|'dict'
)
```

Returns by `output_type`:
- `'legacy'` — tuple `(neg_log_pvals, t_scores_original, h0_fmax_part)` of arrays.
- `'dict'` — dict with keys `'t'`, `'logp_max_t'`, and (if applicable) `'tfce'`, `'logp_max_tfce'`, `'size'`, `'logp_max_size'`, `'mass'`, `'logp_max_mass'`.

## Common patterns

Voxel-wise group t-test with no confounds:

```python
from nilearn.mass_univariate import permuted_ols

neg_log_pvals, t_scores, h0 = permuted_ols(
    tested_vars=group_labels[:, None],   # (n_subjects, 1)
    target_vars=Y,                        # (n_subjects, n_voxels)
    n_perm=5000, two_sided_test=True, n_jobs=-1,
)
```

With confounds and TFCE (requires masker for spatial neighborhood):

```python
out = permuted_ols(
    tested_vars=age[:, None],
    target_vars=Y,
    confounding_vars=np.column_stack([sex, motion]),
    n_perm=10000, two_sided_test=True,
    tfce=True, masker=masker,
    output_type='dict',
)

tfce_logp = out['logp_max_tfce']      # FWE-corrected -log10(p) for TFCE
```

Cluster-mass / cluster-size inference with a forming threshold:

```python
out = permuted_ols(
    tested_vars=design[:, None],
    target_vars=Y,
    confounding_vars=cov,
    threshold=3.09,                       # cluster-forming voxel-wise z
    masker=masker,
    n_perm=10000,
    output_type='dict',
)

cluster_size_logp = out['logp_max_size']   # FWE-corrected
cluster_mass_logp = out['logp_max_mass']
```

Reconstruct images from the per-feature outputs:

```python
img_neg_log_p = masker.inverse_transform(neg_log_pvals.ravel())
img_t = masker.inverse_transform(t_scores.ravel())
```

## Gotchas

- `tfce=True` and any cluster-level inference (`threshold` is not None) **require** a `masker` so spatial connectivity can be inferred.
- `threshold` is the **cluster-forming** voxel-wise threshold (in t/z units), not an alpha. Choose e.g. `3.09` (~p < 0.001 one-sided) before permutation.
- Returned p-values are `-log10(p)` (so 1.30 ~ p = 0.05, 3 ~ p = 0.001), not raw p-values.
- The legacy 3-tuple return type is kept for backward compatibility — prefer `output_type='dict'` for new code.
- `n_perm` controls the resolution of the null; below ~1000 the smallest achievable p-value is too coarse for FWE correction.
- `tested_vars` must be 2D — wrap a 1D array as `arr[:, None]`.
- For one-sample t-tests pass `tested_vars=np.ones((n_subjects, 1))` and `target_vars=Y_centered`.

## See also

- `nilearn.glm.second_level.non_parametric_inference` — higher-level wrapper that takes a list of stat maps and a design matrix.
- `nilearn.glm.threshold_stats_img`, `cluster_level_inference`.
- `nilearn.maskers.NiftiMasker.inverse_transform` — to reconstruct images from per-feature stats.
- https://nilearn.github.io/dev/modules/mass_univariate.html
