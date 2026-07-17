---
title: Inference internals
description: Permutation and bootstrap testing in nltools — algorithms, deterministic RNG, p-values, and numerical stability.
---

# Inference internals

Non-parametric permutation and bootstrap testing, with optional CPU-parallel and GPU
backends. This is design reference for the `nltools/algorithms/inference/` module; for
the public functions see the [Inference API](../api/algorithms/inference.md).

## Backend selection

Every permutation/bootstrap entry point takes `parallel: None | 'cpu' | 'gpu'` (not a
`backend=` argument):

| `parallel=` | Meaning | Trade-off |
|---|---|---|
| `None` | Sequential NumPy | simple, deterministic, slow |
| `'cpu'` | Joblib CPU-parallel | fast, no GPU needed |
| `'gpu'` | PyTorch, batched | large speedup on big problems; requires a GPU |

Matrix permutation (Mantel) is CPU/`None` only — GPU indexing is inefficient for the
symmetric double-permutation, enforced by a validation guard.

## Core algorithms

### One-sample test (sign-flipping)

Test whether a mean differs from zero by randomly flipping signs. Assumes a symmetric
error distribution around zero.

```python
signs = random_choice([+1, -1], size=n_samples)
null_stat[i] = mean(data * signs)
```

### Two-sample test (group permutation)

Test whether group means differ by permuting labels. Assumes exchangeability under H₀.

```python
combined = concatenate([data1, data2])
shuffled = combined[random_permutation(n_total)]
null_stat[i] = mean(shuffled[:n1]) - mean(shuffled[n1:])
```

### Correlation test (index permutation)

Test whether a correlation differs from zero by permuting **one** variable. Metrics:
Pearson (linear), Spearman (rank, robust), Kendall (concordance, most robust).

```python
shuffled_x = x[random_permutation(n_samples)]
null_stat[i] = correlation(shuffled_x, y)   # y unchanged
```

Randomizing one variable tests H₀: ρ = 0 (what users expect); randomizing both tests a
different hypothesis. For autocorrelated data use the time-series methods instead.

### Time-series tests (autocorrelation-preserving)

Standard permutation inflates Type I error with autocorrelated data. Two surrogate
methods preserve temporal structure, randomizing only one variable:

1. **Circle shift** — `x_perm = circshift(x, random_amount)` preserves autocorrelation.
2. **Phase randomize** — `x_perm = ifft(fft(x) * exp(i·random_phases))` preserves the
   power spectrum.

### Matrix permutation (Mantel test)

Test the correlation between two matrices via symmetric permutation:

```python
perm = random_permutation(n_items)
matrix2_perm = matrix2[perm, :][:, perm]     # symmetric indexing
null_stat[i] = correlation(flatten(matrix1), flatten(matrix2_perm))
```

Element extraction: upper triangle (default), lower triangle, or full matrix. A related
public function, `distance_correlation` (with `double_center`/`u_center` helpers),
provides a distance-covariance test validated against R's `energy` and Python's `dcor`.

### Intersubject correlation (ISC)

Two computation modes:

1. **Leave-one-out (LOO)** — `ISC_i = corr(subject_i, mean(others))`; O(n_subjects),
   recommended for large N.
2. **Pairwise** — all `n(n-1)/2` correlations; traditional, complete structure.

Null via subject-wise bootstrap (resample with replacement), circle shift, or phase
randomize; the LOO/pairwise compute has a GPU path selectable with `parallel='gpu'`. A
companion `isc_group_permutation_test` tests a two-group ISC difference. `isc_test`
re-centers the bootstrap null at zero before computing p (fixing a pre-0.6.0 regression).

### Bootstrap inference

Estimate a sampling distribution and confidence intervals via resampling with
replacement. Two modes:

1. **Efficient (default)** — online statistics (Welford), `O(output_shape)` memory,
   normal-approximation CIs.
2. **Full (`save_samples=True`)** — store all samples, `O(n_samples × output_shape)`
   memory, exact percentile CIs, any statistic computable post-hoc.

```python
# Welford's online algorithm (efficient mode)
delta  = sample - mean
mean  += delta / (i + 1)
M2    += delta * (sample - mean)
# finalize
std = sqrt(M2 / (n_samples - 1))
z   = mean / std
p   = 2 * (1 - norm.cdf(abs(z)))     # two-tailed normal approx
```

Beyond `mean`, the simple path supports `median`/`std`/`sum`/`min`/`max`. For Ridge
models, bootstrap farms out to `ridge_svd()` directly (bypassing `BrainData` overhead)
and has a **GPU-batched** implementation for the weights and predict paths.

## P-value calculation

Phipson-Smyth correction:

```text
p = (count + 1) / (n_permute + 1)
```

where `count` = number of null statistics ≥ |observed|. This prevents `p = 0`
(statistically invalid), gives a minimum p-value of `1 / (n_permute + 1)`, and is
standard practice (scipy, FSL, AFNI). Two-tailed uses `|null| ≥ |observed|`; one-tailed
`'upper'`/`'lower'` are also supported.

## Deterministic RNG (cross-backend consistency)

The load-bearing pattern (matching MNE-Python): pre-generate an independent seed per
permutation, then give each permutation its own `RandomState`. This makes results
identical across backends and joblib worker execution orders.

```python
MAX_INT = 2**31 - 1
seeds = root_rng.randint(MAX_INT, size=n_permute)
for i in range(n_permute):
    perm_rng = np.random.RandomState(seeds[i])
    # generate permutation i ...
```

The randomizations (seeds, sign-flip matrices) are generated **before** the parallel
block; joblib workers only *consume* them and never touch RNG state — so
NumPy ↔ CPU-parallel results are bit-identical, and NumPy ↔ GPU differ only by float32
rounding. Memory cost is negligible (4 bytes/permutation plus a bounded sign-flip
matrix).

## CPU parallelization (joblib)

```python
randomizations = generate_all_randomizations(n_permute, random_state)
Parallel(n_jobs=-1)(
    delayed(compute_stat)(randomizations[i]) for i in range(n_permute)
)
```

Worker count is adaptively capped by a memory budget (`_auto_n_jobs_cpu` /
`_verify_n_jobs_memory_constraint`): it estimates per-worker serialization cost, leaves
headroom, and emits a `UserWarning` if it reduces the requested `n_jobs`.

## GPU batching (PyTorch)

Permutations are processed in memory-bounded batches (default budget 4 GB via
`max_gpu_memory_gb`):

```python
memory_per_perm = n_samples * n_features * 4        # float32 bytes
batch_size = int(max_memory_gb * 1e9 / memory_per_perm)
batch_size = max(100, min(batch_size, n_permute))
```

Device compute is float32 (negligible p-value impact vs float64). Kendall correlation has
no GPU kernel — the GPU path falls back to CPU with a `UserWarning` (tracked as EJO-453).

## Numerical stability

Correlations guard against constant/degenerate data with a small epsilon on the
denominator:

```python
from .utils import EPSILON        # 1e-10
correlation = numerator / (denominator + EPSILON)
```

`EPSILON = 1e-10` sits well above float64 machine epsilon (2.2e-16), is small enough for
negligible error, and is safe for float32 GPU math (machine epsilon 1.2e-7). Kendall
guards NaN → 0.0; the bootstrap Z-score is computed under `np.errstate` protection.

## Choosing `n_permute` / `n_samples`

Guidance, not hard limits:

- **Permutation:** ≥ 5,000 for publication (Nichols & Holmes 2002); minimum resolvable
  p-value is `1 / (n_permute + 1)`.
- **Bootstrap:** ≥ 1,000 for reliable CIs, ≥ 5,000 for publication CIs. Use efficient
  (online) mode by default; use full mode only when you need exact percentile CIs or a
  custom post-hoc statistic.

## Key design decisions

- **Pre-generate randomizations** — reproducibility (same seed → identical results across
  backends), inspectable null distributions, and replication of published results.
- **Independent `RandomState` per permutation** — eliminates joblib worker-order effects
  and gives perfect cross-backend consistency.
- **Phipson-Smyth correction** — prevents `p = 0`; standard in neuroimaging software.
- **Welford for bootstrap** — numerically stable and single-pass, `O(output_shape)`
  memory instead of storing every sample.
- **Farm Ridge bootstrap to `ridge_svd()`** — avoids `BrainData` overhead (object
  creation, attribute access, serialization) for a large speedup while keeping Ridge
  correctness.
- **Dual bootstrap modes** — efficient (normal-approx CIs) for most uses, full (exact
  percentile CIs) opt-in for custom statistics or distribution visualization.

## Performance

CPU-parallel gives a several-fold speedup over sequential; the GPU path gives a much
larger one on big problems (many voxels / many permutations). Actual timings are
hardware-dependent — benchmark on your own machine.

## References

1. Nichols & Holmes (2002). Nonparametric permutation tests for functional neuroimaging.
   *HBM* 15(1):1–25.
2. Winkler et al. (2014). Permutation inference for the GLM. *NeuroImage* 92:381–397.
3. Phipson & Smyth (2010). Permutation p-values should never be zero.
   *Stat Appl Genet Mol Biol* 9(1):Article 39.
4. Good (2000). *Permutation Tests: A Practical Guide*. Springer.
5. Theiler et al. (1992). Testing for nonlinearity in time series. *Physica D* 58:77–94.
6. Lancaster et al. (2018). Surrogate data for hypothesis testing.
   *Physics Reports* 748:1–60.
7. Chen et al. (2016). Untangling correlations at the group level. *NeuroImage*
   142:248–259.
8. Efron & Tibshirani (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
9. Welford (1962). Note on a method for calculating corrected sums of squares.
   *Technometrics* 4(3):419–420.
