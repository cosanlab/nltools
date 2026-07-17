---
title: Ridge internals
description: The six mathematical tricks behind nltools' GPU-accelerated ridge solver, and its backend abstraction.
---

# Ridge internals

Efficient, GPU-accelerated ridge regression for neuroimaging. Inspired by
[Himalaya](https://github.com/gallantlab/himalaya) (Dupré la Tour et al., 2022), the
implementation achieves large speedups on big problems through a handful of mathematical
tricks and memory-efficient batching.

This is design reference. For the public solver API (`solve_ridge_cv`,
`solve_banded_ridge_cv`, `cross_val_predict_ridge`, and the legacy `ridge_svd`/`ridge_cv`),
see the [Ridge API](../api/algorithms.md). Code lives in `nltools/algorithms/ridge/`
(`core.py`, `solvers.py`, `utils.py`) with the backend in `nltools/algorithms/backends.py`.

## The six mathematical tricks

### 1. SVD magic: solve once, use forever

The textbook ridge formula inverts a matrix for *every* alpha:

```python
beta = (X.T @ X + alpha * I)^(-1) @ X.T @ y  # O(n³) per alpha
```

The SVD approach decomposes `X` once, then reuses it for any alpha with arithmetic only:

```python
# Do ONCE:
U, s, Vt = svd(X)

# For ANY alpha (reuse same SVD):
shrinkage = s / (s**2 + alpha)          # just arithmetic, O(n)
beta = Vt.T @ (shrinkage[:, None] * U.T @ y)   # no inversion
```

Instead of solving the full system repeatedly, decompose `X` into principal directions
once, then adjust how much you trust each direction (shrinkage) per alpha. Trying 1000
alphas becomes trivial rather than impossible.

### 2. Generator pattern: process and forget

Storing a resolution matrix per alpha explodes memory (1000 alphas × 500 MB = 500 GB).
`_decompose_ridge` is a **generator** that computes one alpha batch, yields it, and frees
it before the next:

```python
def _decompose_ridge(Xtrain, alphas, n_alphas_batch=None, method="svd", backend=None):
    U, s, Vt = svd(Xtrain)                 # once
    for batch in batches(alphas, n_alphas_batch):
        matrices = compute_batch(U, s, Vt, batch)
        yield matrices, batch
        del matrices                       # freed before next iteration
```

Only one batch lives in RAM at a time. The generator is composable — it nests inside the
target-batch loop for two-dimensional batching. (The default is process-all-alphas in
one batch; set `n_alphas_batch` to bound memory for very large grids.)

### 3. Two-dimensional batching: divide and conquer

A realistic problem — 100 samples × 300k voxels × 10 alphas × 5 folds — is ~60 GB naive,
far past an 8 GB GPU. Batching over **both** targets (voxels) and alphas keeps each chunk
small:

```python
for target_batch in range(0, 300_000, 5_000):   # 5k voxels at a time
    for matrices, alpha_batch in _decompose_ridge(...):  # 10 alphas at a time
        # 100 samples × 5k voxels × 10 alphas ≈ 200 MB
        ...
```

Target batching handles massive output spaces; alpha batching handles large
hyperparameter grids. Together they run problems far larger than GPU RAM.

### 4. `Y_in_cpu` strategy: smart shuttle

Pre-loading all of `Y` to the GPU (300k voxels) OOMs once multiplied by folds and alphas.
With `Y_in_cpu=True` (the default), `Y` stays in RAM and only the current target batch is
shuttled to the device:

```python
Y = keep_on_cpu(Y)
for batch in target_batches:
    Y_batch = to_gpu(Y[:, batch])   # only 5k voxels on device
    compute(Y_batch)
    del Y_batch
```

~10% slower, but prevents OOM entirely and enables problems many times larger than GPU
RAM. `max_gpu_memory_gb` feeds an auto target-batch sizer (`_auto_n_targets_batch`,
with a 5× overhead factor on GPU) that picks `n_targets_batch` when it's unset.

### 5. Per-target alpha without extra cost

Per-voxel alpha selection sounds like one SVD per voxel (100k SVDs). In fact you only
need one SVD per *unique* selected alpha (typically ~10):

```python
unique_alphas = np.unique(best_alphas)     # e.g. [0.1, 1.0, 10.0]
for alpha in unique_alphas:                # ~10 iterations, not 100k
    mask = best_alphas == alpha
    weights[:, mask] = solve_ridge(X, Y[:, mask], alpha)
```

Per-voxel optimization at bulk-solve cost — like sorting mail by zip code before
delivery.

### 6. Resolution-matrix precomputation

Separate the X-dependent (expensive) piece from the Y-dependent (cheap) piece:

```python
# matrices = Vt.T @ diag(s / (s² + α)) @ U.T  ==  (XᵀX + αI)⁻¹ Xᵀ
pred_matrix = X_val @ matrices              # depends only on X
predictions = pred_matrix @ Y_train_batch   # cheap; reuse across all targets
```

The expensive SVD is reused across all targets, and the computation vectorizes over
alphas and targets simultaneously.

### How it fits together

```python
def solve_ridge_cv(X, Y, alphas, cv=5):
    scores = zeros(n_splits, n_alphas, n_targets)
    for fold in cv.split(X):
        X_train, X_val = X[fold]
        # Trick 1: single SVD per fold, reused across ALL alphas
        for matrices, alpha_batch in _decompose_ridge(X_train, alphas):  # Trick 2 + 6
            pred_matrix = X_val @ matrices
            for target_batch in batches(n_targets, 5000):                # Trick 3
                Y_batch = to_gpu(Y[:, target_batch])                     # Trick 4
                predictions = pred_matrix @ Y_batch
                scores[fold, alpha_batch, target_batch] = r2(Y_batch, predictions)
                del Y_batch
            del matrices, pred_matrix
    best_alphas = argmax(scores.mean(axis=0), axis=0)                    # Trick 5
    for alpha in np.unique(best_alphas):
        mask = best_alphas == alpha
        coefs[:, mask] = solve_ridge(X, Y[:, mask], alpha)
    return {"best_alphas": best_alphas, "coefs": coefs, "cv_scores": scores, "backend": ...}
```

> **Return shape.** `solve_ridge_cv` returns a **dict** with keys `best_alphas`, `coefs`,
> `cv_scores` (shaped `(n_splits, n_alphas, n_targets)`), and `backend` — not a tuple.
> `solve_banded_ridge_cv` returns a dict keyed on `deltas`/`cv_scores`/`backend` (plus
> optional `coefs`/`intercept`); there is no `best_alphas` key on the banded path.

## Backend abstraction

The backend is a single **`class Backend`** in `nltools/algorithms/backends.py` that
dispatches internally on its `name`. It is *not* a set of per-backend modules, and there
is no module-level `get_backend()`. Obtain one via `resolve_backend(parallel)` or by
constructing `Backend(...)` directly:

```python
backend = resolve_backend("gpu")        # or Backend("torch"), Backend("numpy")
X_dev = backend.asarray(X, device="cuda")
U, s, Vt = backend.svd(X_dev)
```

**Selection.** The public solvers take `parallel: None | 'cpu' | 'gpu'`, translated to a
concrete backend. The constructor also accepts explicit `"numpy"`, `"torch"`, `"auto"`.
Resolved `.name` values are hyphenated: `numpy`, `torch-cpu`, `torch-cuda`, `torch-mps`.
`"torch"` auto-detects and will pick CUDA/MPS if present — it is not CPU-only.

**Backends:**

| Selector | Resolves to | Notes |
|---|---|---|
| `parallel=None` / `'cpu'` / `"numpy"` | `numpy` | CPU, always available |
| `"torch"` | `torch-cuda` / `torch-mps` / `torch-cpu` | auto-detects best device |
| `parallel='gpu'` | `torch-cuda` / `torch-mps` | requires a GPU |

**MPS (Apple Metal)** is supported, including a precision workaround: SVD runs on CPU in
float64 then moves back to the device, since MPS float32 SVD is inaccurate.

### When to use the GPU

| Scenario | `parallel=` | Rationale |
|---|---|---|
| < 10k voxels | `None`/`'cpu'` | CPU fast enough |
| 10k–100k voxels | `'cpu'` (torch) | faster CPU implementation |
| > 100k voxels | `'gpu'` | large speedup |
| interactive / debugging | `None`/`'cpu'` | simpler |

Rule of thumb: reach for the GPU above ~50k voxels, ~1k alphas, or when you need
sub-minute runtime.

## Banded ridge

`solve_banded_ridge_cv` implements true banded/group ridge (Himalaya's
`solve_group_ridge_random_search`): feature-group-specific regularization via `sqrt(gamma)`
scaling, with the group weights `gamma` explored by Dirichlet random search.

```python
# Scale each feature space by sqrt(gamma_i) before a standard ridge, then unscale.
X_scaled[:, group_i] *= sqrt(gamma[i])
# ... solve ...
weights[group_i] *= sqrt(gamma[i])
```

The first Dirichlet sample is forced to equal weights. Use case: multiple feature spaces
with different characteristics (e.g. semantic vs visual features in voxelwise encoding
models). The per-space log-ratio `deltas = log(gamma / alpha)` are returned rather than a
single `best_alphas`.

## Alpha grid selection

| Use case | Alpha range | Notes |
|---|---|---|
| Exploratory | `[0.01, 0.1, 1, 10, 100]` | fast, coarse |
| Standard | `np.logspace(-2, 3, 10)` | publication quality |
| Thorough | `np.logspace(-3, 4, 20)` | capture nuance |

Use log-spaced alphas to cover a wide range efficiently.

## Key design decisions

- **Single SVD per fold** — mathematical equivalence (`shrinkage = s/(s²+α)` changes
  alpha, not the SVD) turns a per-alpha decomposition into a one-time cost.
- **`Y_in_cpu` default** — prevents OOM at ~10% runtime cost; override for small problems.
- **Per-target alpha** — different voxels need different regularization, at the cost of
  only `n_unique_alphas` SVDs.
- **Generator pattern** — one batch in memory at a time, explicit `del` between yields,
  composable across batching dimensions.
- **Single-class backend** — one dispatch surface keeps NumPy / Torch (CUDA/MPS/CPU)
  behind one API without per-backend module duplication.

## Performance

The speedups are large (order 50–100× on big problems: single SVD per fold ~10×, GPU an
additional ~10–20×, with generator + batching enabling problems that don't fit in RAM at
all). Concrete timings are hardware-dependent — benchmark on your own machine rather than
relying on fixed numbers.

## References

1. Hoerl & Kennard (1970). Ridge regression. *Technometrics* 12(1):55–67.
2. Hastie et al. (2009). *The Elements of Statistical Learning* (2nd ed). Springer.
3. Dupré la Tour et al. (2022). himalaya: Ridge regression with multiple solvers.
4. Nunez-Elizalde et al. (2019). Voxelwise encoding models with non-spherical priors.
   *Nature Neuroscience* 22:1060–1065.
5. Naselaris et al. (2011). Encoding and decoding in fMRI. *NeuroImage* 56(2):400–410.
6. Haxby et al. (2011). A common, high-dimensional model of ventral temporal cortex.
   *Neuron* 72(2):404–416.
