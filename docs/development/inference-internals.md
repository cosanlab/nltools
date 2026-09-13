---
title: Inference internals
description: Non-parametric permutation and bootstrap testing on CPU workers — algorithms, deterministic RNG, p-values, and numerical stability.
---

# Inference internals

Non-parametric permutation and bootstrap testing on CPU workers. This is design
reference for the `nltools/algorithms/inference/` module; for the public
functions see [`nltools.algorithms`](../api/algorithms.md).

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
   power spectrum. Conjugate pairing matters: `pos_freq`/`neg_freq` are built already
   in conjugate order, so the negative frequencies take the *same* phases negated —
   never reversed (a reversed pairing leaves the spectrum non-Hermitian, and taking
   `.real` of the ifft silently distorts the surrogate).

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
randomize. A companion `isc_group_permutation_test` tests a two-group ISC
difference. `isc_test` re-centers the bootstrap null at zero before computing p
(fixing a pre-0.6.0 regression).

### Bootstrap inference

Estimate a sampling distribution and a confidence interval by resampling rows
with replacement. There is one mode, not two: replicates stream through
`BootstrapAccumulator`, which keeps a running Welford variance plus a bounded
per-element tail — exactly the order statistics NumPy's linear-interpolation
percentile can reach at either end. For `B` replicates at confidence level `c`
it retains, per output element, this many of the smallest and largest values:

```text
k = ceil((B - 1) * (1 - c) / 2) + 1
```

That reproduces the interval the complete distribution would give while storage
scales with `(1 - c) * B` instead of `B` — roughly 5% of the replicates at 95%
confidence. It is memory-efficient, not constant-memory: `k` still grows with
`B`. `return_samples=True` additionally keeps every replicate, for plotting or a
post-hoc statistic; it never changes how the interval is computed.

Streaming only pays off if the engine never materializes the distribution it is
avoiding. `joblib.Parallel` dispatches eagerly and queues finished results, so
neither `pre_dispatch` nor `return_as="generator"` bounds how many replicate
arrays are alive — collecting the run into a list, the obvious spelling, would
make peak memory `O(B)` and reduce the preflight to a number the run ignores.
`_run_replicates` therefore dispatches in windows of
`backends.bootstrap_replicate_window(n_samples, n_workers=...)` and folds each
window into the accumulator before opening the next. The window scales with the
worker count and never with `B`, and windows are consecutive and folded in
order, so Welford's accumulation order — and therefore every reported number —
is bitwise identical to a sequential run.

```python
# Welford's online update, per replicate
delta  = sample - mean
mean  += delta / n
M2    += delta * (sample - mean)
# finalize
standard_error = sqrt(M2 / (n - 1))          # ddof=1 across replicates
ci_lower, ci_upper = interpolate(retained_tails, (1 - c) / 2)
```

`estimate` is the statistic on the *unresampled* full sample, not the replicate
mean: the basic reduction on `bd.data`, the fitted `coef_` for `'weights'`, and
`X_test @ coef_` for `'predict'`. The result exposes no replicate mean and no
`z`, `p`, or `tail` output — a bootstrap hypothesis test is a separate,
not-yet-defined API. Non-finite replicate values propagate: `np.partition`
drops NaN, so the accumulator tracks a per-element NaN flag and reproduces
`np.percentile`'s propagation instead of quietly skipping it.

Aggregation runs on the CPU and is mergeable: `BootstrapAccumulator.merge`
combines two blocks with the Chan-Golub-LeVeque parallel variance update and a
tail merge, so a run split across workers or memory-driven batches summarizes to
the same numbers as one sequential pass. Both blocks must be sized with the
run's *total* replicate count, or the merge refuses — a block sized for its own
length would retain too short a tail.

Memory is planned in `nltools/algorithms/backends.py` and nowhere else.
`bootstrap_memory_preflight` charges eight bytes for every output-sized array a
run holds at once — the two bounded tails, the replicates buffered before the
next flush and the two temporaries that flush builds, one dispatch window, every
replicate when `return_samples=True`, and the two Welford accumulators plus the
four summary payloads — and raises *before* resampling if that exceeds the
budget, naming the requirement, the measured budget, and the `memory_budget_gb`
override. It never weakens the interval, reduces `n_samples`, or disables
`return_samples`. `bootstrap_n_jobs_cpu` treats `n_jobs` as a ceiling and lowers
it when a worker's copy of the data would not fit;
`bootstrap_replicate_window` turns that worker count into the dispatch window.
`BOOTSTRAP_TAIL_FLUSH_BLOCK` lives there too rather than in the engine, so the
one budget owner sees every constant it has to charge for.

`nltools/tests/core/test_bootstrap.py::TestBootstrapPeakMemory` measures peak
allocation with `tracemalloc` against that figure. It runs at `n_jobs=1`
deliberately — `tracemalloc` sees only this process, and the parent is where the
unbounded allocation used to live.

Beyond `mean`, the simple path supports `median`/`std`/`sum`/`min`/`max`, each
the exact NumPy reduction over rows; `'std'` is the *population* deviation
(`ddof=0`), matching `BrainData.std()` and distinct from the `ddof=1`
`standard_error` across replicates. For Ridge models, bootstrap farms out to the
shared fixed-hyperparameter refit in `nltools/models/ridge.py` directly
(bypassing `BrainData` overhead), on the CPU and on the GPU alike: the design is
converted onto the backend once and `_refit_resample` — the one replicate
implementation — resamples rows in place, so the GPU driver differs only in
which backend it converts to and in how many replicate results it holds before
aggregation. `backends.ridge_bootstrap_batch_size` sizes that batch: it charges
both the resident replicate and the host-side float64 results, so a prediction
bootstrap with a wide `X_test` shrinks the batch instead of overrunning the
budget. The refit holds `alpha_` — and, for a banded model,
`feature_space_weights_` — fixed; a resample never reruns cross-validation or the
banded random search. Training features are supplied explicitly by the caller
(`BrainData.bootstrap(..., X=...)`); every feature space and the response resample
with the same row indices. A terminal replicate failure raises the whole call and
names the replicate index; failed replicates are never dropped and replacements
are never drawn.

## P-value calculation

Phipson-Smyth correction:

```text
p = (count + 1) / (n_permute + 1)
```

where `count` = number of null statistics ≥ |observed|. This prevents `p = 0`
(statistically invalid), gives a minimum p-value of `1 / (n_permute + 1)`, and is
standard practice (scipy, FSL, AFNI). Two-tailed uses `|null| ≥ |observed|`; one-tailed
`'upper'`/`'lower'` are also supported.

## Deterministic RNG (worker-count consistency)

The load-bearing pattern (matching MNE-Python): pre-generate an independent seed per
permutation, then give each permutation its own `RandomState`. This makes results
identical regardless of joblib worker count.

```python
MAX_INT = 2**31 - 1
seeds = root_rng.randint(MAX_INT, size=n_permute)
for i in range(n_permute):
    perm_rng = np.random.RandomState(seeds[i])
    # generate permutation i ...
```

The randomizations (seeds, sign-flip matrices) are generated **before** the parallel
block; joblib workers only *consume* them and never touch RNG state — so results
are bit-identical for any `n_jobs`. Memory cost is negligible (4 bytes per
permutation plus a bounded sign-flip matrix).

## CPU parallelization (joblib)

```python
randomizations = generate_all_randomizations(n_permute, random_state)
Parallel(n_jobs=-1)(
    delayed(compute_stat)(randomizations[i]) for i in range(n_permute)
)
```

Worker count is adaptively capped by a memory budget (`_auto_n_jobs_cpu`, living in
`algorithms.backends`): it estimates per-worker serialization cost, leaves headroom,
and never exceeds `max_jobs` (pass `min(requested, cpu_count)` to cap an explicit
request).

Batch and memory arithmetic for the one remaining batched path — the Ridge
bootstrap on the GPU — lives in `algorithms.backends` and is documented in
[Ridge internals](ridge-internals.md); the permutation engines here allocate
nothing beyond one worker's copy of the data.

## Numerical stability

Correlations guard against constant/degenerate data with a small epsilon on the
denominator:

```python
from .utils import EPSILON        # 1e-10
correlation = numerator / (denominator + EPSILON)
```

`EPSILON = 1e-10` sits well above float64 machine epsilon (2.2e-16) and is small enough
for negligible error. Kendall
guards NaN → 0.0; the bootstrap accumulator propagates non-finite replicate values
rather than substituting `nan*` reductions.

## Choosing `n_permute` / `n_samples`

Guidance, not hard limits:

- **Permutation:** ≥ 5,000 for publication (Nichols & Holmes 2002); minimum resolvable
  p-value is `1 / (n_permute + 1)`.
- **Bootstrap:** ≥ 1,000 for reliable CIs, ≥ 5,000 for publication CIs. The interval is
  always the exact percentile interval; `return_samples=True` costs memory and buys only
  the replicates themselves, for plotting or a custom post-hoc statistic.

## Key design decisions

- **Pre-generate randomizations** — reproducibility (same seed → identical results),
  inspectable null distributions, and replication of published results.
- **Independent `RandomState` per permutation** — eliminates joblib worker-order effects,
  so the worker count is numerically invisible.
- **Phipson-Smyth correction** — prevents `p = 0`; standard in neuroimaging software.
- **Welford plus a bounded tail for bootstrap** — numerically stable and single-pass,
  and the exact percentile interval at `O((1 - c) * B * output_shape)` retained memory
  instead of storing every replicate. Memory-efficient, not constant-memory.
- **Windowed dispatch** — the retained bound is only real if the engine aggregates as it
  goes; joblib will happily queue every result otherwise.
- **Farm Ridge bootstrap to the shared fixed refit** — avoids `BrainData` overhead
  (object creation, attribute access, serialization) for a large speedup, and keeps
  resamples on the same numerical path as the full-data fit.
- **One bootstrap interval** — the exact percentile interval in every mode, so a result
  never depends on whether the caller happened to ask for the replicates.
- **Loud, early memory refusal** — a preflight against the measured budget beats an OOM
  half-way through a five-thousand-replicate run.

## Performance

Spreading permutations across workers gives a several-fold speedup over a single
worker. Actual timings are hardware-dependent — benchmark on your own machine.

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
