# nilearn.connectome — Functional Connectivity

Tools for computing functional connectivity matrices and an implementation of an algorithm for sparse multi-subject learning of Gaussian graphical models. The main entry point is `ConnectivityMeasure`; group-sparse inverse covariance is supported via `GroupSparseCovariance` / `GroupSparseCovarianceCV`.

**Source:** https://nilearn.github.io/dev/modules/connectome.html

## Inventory

### Classes
| Class | Purpose |
|---|---|
| `ConnectivityMeasure` | Computes different kinds of functional connectivity matrices. |
| `GroupSparseCovariance` | Covariance and precision matrix estimator. |
| `GroupSparseCovarianceCV` | Sparse inverse covariance with cross-validated choice of regularization. |

### Functions
| Function | Purpose |
|---|---|
| `sym_matrix_to_vec(symmetric, discard_diagonal=False)` | Return the flattened lower-triangular part of an array. |
| `vec_to_sym_matrix(vec, diagonal=None)` | Return the symmetric matrix given its flattened lower-triangular part. |
| `group_sparse_covariance(subjects, alpha, ...)` | Compute sparse precision/covariance matrices across subjects. |
| `cov_to_corr(covariance)` | Return correlation matrix for a given covariance matrix. |
| `prec_to_partial(precision)` | Return partial correlation matrix for a given precision matrix. |

## ConnectivityMeasure

```python
ConnectivityMeasure(
    cov_estimator=None,         # default: LedoitWolf
    kind='covariance',          # 'covariance'|'correlation'|'partial correlation'|'tangent'|'precision'
    vectorize=False,            # flatten upper triangle to 1D
    discard_diagonal=False,
    standardize=True,
)
```

Methods: `fit(X)`, `transform(X)`, `fit_transform(X)`, `inverse_transform(connectivities, diagonal=None)`.

Input `X`: list of `(n_timepoints, n_regions)` arrays — one per subject.
Output: `(n_subjects, n_regions, n_regions)`, or `(n_subjects, n_features)` if `vectorize=True`.

Post-fit attrs:
- `mean_` — group mean connectivity (geometric mean for `kind='tangent'`).
- `whitening_` — whitening matrix (only `kind='tangent'`).
- `cov_estimator_` — fitted estimator.

## GroupSparseCovariance / GroupSparseCovarianceCV

```python
GroupSparseCovarianceCV(alphas=4, n_refinements=4, cv=3, tol=0.01, max_iter=10,
                        verbose=0, n_jobs=1, memory=None)
gsc.fit(subjects)               # list of (n_timepoints, n_regions) arrays
gsc.covariances_                # (n_features, n_features, n_subjects)
gsc.precisions_                 # shared sparsity pattern across subjects
gsc.alpha_                      # selected regularization
```

`group_sparse_covariance(subjects, alpha, max_iter=50, tol=0.001, ...)` is the underlying functional API returning `(emp_covs, precisions)`.

## Vectorize / unvectorize

```python
vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
mat = vec_to_sym_matrix(vec, diagonal=np.ones(n_regions))
```

When `vectorize=True` and `discard_diagonal=True`, the off-diagonal upper triangle is returned (length `n*(n-1)/2`).

## Common patterns

Single-subject correlation matrix:

```python
from nilearn.connectome import ConnectivityMeasure

conn = ConnectivityMeasure(kind='correlation', standardize=True)
mat = conn.fit_transform([time_series])[0]   # (n_regions, n_regions)
```

Multi-subject tangent-space features for classification:

```python
all_ts = [masker.fit_transform(img, confounds=conf)
          for img, conf in zip(imgs, confs)]

conn = ConnectivityMeasure(kind='tangent', vectorize=True, discard_diagonal=True)
features = conn.fit_transform(all_ts)        # (n_subjects, n_features)
mean_conn = conn.mean_                       # geometric mean
```

Conversions between covariance, correlation, partial correlation:

```python
from nilearn.connectome import cov_to_corr, prec_to_partial

corr = cov_to_corr(cov)
partial = prec_to_partial(precision)
```

Group-sparse inverse covariance:

```python
from nilearn.connectome import GroupSparseCovarianceCV

gsc = GroupSparseCovarianceCV(alphas=4, cv=3)
gsc.fit(subjects_list)
```

## Gotchas

- `kind='tangent'` requires multiple subjects and is the recommended kind for downstream classification — Pearson `'correlation'` features are noisier per subject.
- `vectorize=True` flattens the upper triangle (with diagonal unless `discard_diagonal=True`); use `vec_to_sym_matrix` to recover the full matrix.
- `ConnectivityMeasure` uses `LedoitWolf` shrinkage by default; pass a custom `cov_estimator` (any sklearn `*Covariance*` estimator) to override.
- `GroupSparseCovarianceCV` assumes the sparsity pattern is shared across subjects — this is a strong assumption.
- `standardize=True` z-scores each region before computing covariance; turn off only if your time series are already standardized.

## See also

- `nilearn.maskers.NiftiLabelsMasker` / `NiftiMapsMasker` — extract region time series first.
- `nilearn.plotting.plot_matrix`, `plot_connectome`, `view_connectome` — visualize results.
- https://nilearn.github.io/dev/modules/connectome.html
