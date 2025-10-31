# GPU-Accelerated Permutation Testing - Design Reference

**Quick technical reference for nltools inference module developers.**

---

## Core Algorithms

### One-Sample Test (Sign-Flipping)

**Method**: Test if mean ≠ 0 by randomly flipping signs.

**Algorithm**:
```python
# For each permutation i:
signs = random_choice([+1, -1], size=n_samples)
null_stat[i] = mean(data * signs)
```

**Assumption**: Symmetric error distribution around zero.

**References**: Nichols & Holmes (2002) *HBM* 15(1):1-25; Winkler et al. (2014) *NeuroImage* 92:381-397.

---

### Two-Sample Test (Group Permutation)

**Method**: Test if group means differ by permuting labels.

**Algorithm**:
```python
# For each permutation i:
combined = concatenate([data1, data2])
shuffled = combined[random_permutation(n_total)]
null_stat[i] = mean(shuffled[:n1]) - mean(shuffled[n1:])
```

**Assumption**: Exchangeability (group assignment arbitrary under H₀).

**References**: Good (2000) *Permutation Tests*; Nichols & Holmes (2002).

---

### Correlation Test (Index Permutation)

**Method**: Test if correlation ≠ 0 by permuting one variable.

**Metrics**: Pearson (linear), Spearman (rank, robust), Kendall (concordance, most robust).

**Algorithm**:
```python
# For each permutation i:
shuffled_x = x[random_permutation(n_samples)]
null_stat[i] = correlation(shuffled_x, y)  # y unchanged
```

**Assumption**: Independence (i.i.d. observations). For autocorrelated data, use time-series methods.

**References**: Nichols & Holmes (2002); Good (2000).

---

### Time-Series Tests (Autocorrelation-Preserving)

**Why needed**: Standard permutation inflates Type I error with autocorrelated data.

**Methods**:

1. **Circle Shift**: `x_perm = circshift(x, random_amount)` - preserves autocorrelation
2. **Phase Randomize**: `x_perm = ifft(fft(x) * exp(i*random_phases))` - preserves power spectrum

**Critical**: Only randomize ONE variable to test correlation (not both).

**References**: Theiler et al. (1992) *Physica D* 58:77-94; Lancaster et al. (2018) *Physics Reports* 748:1-60.

---

### Matrix Permutation (Mantel Test)

**Method**: Test correlation between matrices via symmetric permutation.

**Algorithm**:
```python
# For each permutation i:
perm = random_permutation(n_items)
matrix2_perm = matrix2[perm, :][:, perm]  # Symmetric indexing
null_stat[i] = correlation(flatten(matrix1), flatten(matrix2_perm))
```

**Element extraction**: Upper triangle (default), lower triangle, or full matrix.

**Implementation**: CPU-parallel only (GPU indexing inefficient for this operation).

**References**: Chen et al. (2016) *NeuroImage* 142:248-259; Mantel (1967) *Cancer Research* 27:209-220.

---

### Intersubject Correlation (ISC)

**Two computation modes**:

1. **Leave-One-Out (LOO)**: `ISC_i = corr(subject_i, mean(others))`
   - O(n_subjects), efficient, recommended for large N

2. **Pairwise**: All n×(n-1)/2 pairwise correlations
   - Complete structure, traditional ISC

**Permutation**: Subject-wise bootstrap (resample with replacement), circle shift, or phase randomize.

**Implementation**: Two-phase (1) GPU-accelerated ISC computation, (2) CPU-parallel bootstrap.

**References**: Chen et al. (2016); BrainIAK implementation.

---

## P-Value Calculation

**Formula** (Phipson-Smyth correction):

```
p = (count + 1) / (n_permute + 1)
```

Where `count` = number of null statistics ≥ |observed|.

**Properties**:
- Prevents p = 0 (statistically invalid)
- Minimum p-value = 1/(n_permute + 1)
- Unbiased for hypothesis testing

**Reference**: Phipson & Smyth (2010) *Stat Appl Genet Mol Biol* 9(1):Article 39.

---

## Implementation Details

### Deterministic RNG (Cross-Backend Consistency)

**Pattern** (matches MNE-Python):
```python
# Pre-generate unique seed per permutation
MAX_INT = 2**31 - 1
seeds = root_rng.randint(MAX_INT, size=n_permute)

# Each permutation uses independent RandomState
for i in range(n_permute):
    perm_rng = np.random.RandomState(seeds[i])
    # Generate permutation...
```

**Why**: Ensures 0.000% variance across backends (NumPy, CPU-parallel, GPU), regardless of execution order.

**Memory cost**: Seeds (4 bytes/permutation) + sign-flip matrix (<100 MB worst case) = negligible.

---

### CPU Parallelization (Joblib)

**Strategy**: Pre-generate randomizations, parallelize computation.

```python
# Pre-generate (deterministic)
randomizations = generate_all_randomizations(n_permute, random_state)

# Parallelize (no RNG in workers)
Parallel(n_jobs=-1)(
    delayed(compute_stat)(randomizations[i])
    for i in tqdm(range(n_permute))
)
```

**Speedup**: 4-8× on 8-core machines.

---

### GPU Batching (PyTorch)

**Challenge**: Limited memory (can't load all permutations).

**Solution**: Automatic batching with conservative budget (default 4 GB).

```python
# Estimate batch size
memory_per_perm = n_samples * n_features * 4  # float32 bytes
batch_size = int(max_memory_gb * 1e9 / memory_per_perm)
batch_size = max(100, min(batch_size, n_permute))

# Process in batches
for batch_idx in range(n_batches):
    # Generate batch randomizations
    # Transfer to GPU, compute statistics
    # Accumulate results
```

**Precision**: float32 on GPU (negligible impact on p-values, <0.1% variance vs float64).

---

### Backend Selection

| Backend | When to Use | Pros | Cons |
|---------|-------------|------|------|
| `'numpy'` | Debugging, small problems | Simple, deterministic | Slow |
| `None` (CPU-parallel) | Standard problems (n=30-100) | Fast, no GPU needed | CPU-limited |
| `'torch'` | Large problems (>10K voxels/perms) | 10-100× speedup | Requires GPU |
| `'auto'` | Let algorithm decide | Intelligent selection | - |

---

## Numerical Stability

### Division by Zero

**Approach**: Add small epsilon to denominator.

```python
from .utils import EPSILON  # 1e-10

# Correlation with constant data protection
correlation = numerator / (denominator + EPSILON)
```

**Value**: `EPSILON = 1e-10`
- Standard in scientific computing for float64
- Well above machine epsilon (2.2e-16)
- Small enough for negligible error (<1e-10 relative)
- Safe for float32 GPU operations (machine epsilon 1.2e-07)

---

## Performance Benchmarks

**Problem**: 30 subjects, 50K voxels, 5K permutations

| Backend | Time | Speedup |
|---------|------|---------|
| NumPy (sequential) | 120s | 1× |
| CPU-parallel (8 cores) | 18s | 6.7× |
| PyTorch GPU (RTX 3090) | 2s | 60× |

**Memory**: Sign-flip matrix <5 MB typical, <100 MB extreme (negligible vs GB data).

---

## Quick Reference

### Minimum n_permute for Publication

| n_permute | Min p-value | Precision | Use Case |
|-----------|-------------|-----------|----------|
| 1,000 | 0.001 | ±6% | Exploratory |
| **5,000** | **0.0002** | **±3%** | **Publication standard** |
| 10,000 | 0.0001 | ±2% | High-precision |

**Rule**: n_permute ≥ 5,000 for publication (Nichols & Holmes 2002).

---

### Backend Recommendations

| Scenario | Backend | Time (5K perms) |
|----------|---------|----------------|
| ROI analysis (n=30, 500 voxels) | CPU-parallel | 0.3s |
| Whole-brain (n=30, 50K voxels) | CPU-parallel | 18s |
| High-res (n=100, 200K voxels) | GPU | 5s |
| Searchlight RSA (1K searchlights) | CPU-parallel | 1min |

**GPU threshold**: Use for >50K voxels or >10K permutations.

---

## Test Suite Validation

**Coverage**: 179 tests covering all methods, backends, edge cases.

**Cross-backend determinism**:
- NumPy ↔ CPU-parallel: 0.000% variance (bit-for-bit identical)
- NumPy ↔ GPU: <0.1% variance (float32 precision only)

**Validated**:
- ✅ Statistical correctness (vs published results)
- ✅ Determinism (same seed → identical results)
- ✅ Edge cases (constant data, single feature, extreme values)
- ✅ Backward compatibility (~1-2% variance vs old implementation, acceptable for v0.6.0)

---

## Key Design Decisions

**Why pre-generate randomizations?**
- Reproducibility: same seed → identical results across backends
- Debugging: inspect null distribution
- Scientific validity: replication of published results

**Why independent RandomState per permutation?**
- Eliminates worker execution order effects (joblib)
- Perfect cross-backend consistency (0.000% variance)

**Why Phipson-Smyth correction?**
- Standard practice in statistical software (scipy, FSL, AFNI)
- Prevents p=0 (statistically invalid)
- Conservative for neuroimaging's massive multiple comparisons

**Why randomize ONE variable for correlation?**
- Tests H₀: ρ=0 (what users expect)
- Standard practice (Theiler 1992, Lancaster 2018, all ISC literature)
- Randomizing both tests different hypothesis (both are white noise)

---

## References

1. Nichols & Holmes (2002). Nonparametric permutation tests for functional neuroimaging. *HBM* 15(1):1-25.
2. Winkler et al. (2014). Permutation inference for the GLM. *NeuroImage* 92:381-397.
3. Phipson & Smyth (2010). Permutation p-values should never be zero. *Stat Appl Genet Mol Biol* 9(1):Article 39.
4. Good (2000). *Permutation Tests: A Practical Guide*. Springer.
5. Theiler et al. (1992). Testing for nonlinearity in time series. *Physica D* 58:77-94.
6. Lancaster et al. (2018). Surrogate data for hypothesis testing. *Physics Reports* 748:1-60.
7. Chen et al. (2016). Untangling correlations at the group level. *NeuroImage* 142:248-259.

---

**Last Updated**: 2025-10-31
**Module Version**: 0.6.0
**Authors**: nltools development team
