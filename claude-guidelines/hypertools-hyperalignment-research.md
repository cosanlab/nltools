# Hypertools Hyperalignment Implementation - Deep Dive Research

**Research Date**: 2025-10-29
**Purpose**: Understand Hypertools' hyperalignment implementation and testing strategy to inform nltools testing approach
**Key Question**: Do they test computational correctness or mainly interface contracts?

---

## Executive Summary

**Answer to Key Question**: Hypertools uses **primarily contract/interface tests with minimal computational correctness validation**.

Their testing strategy:
- ✅ **Contract tests**: Shape validation, property existence, API compliance
- ✅ **Invariant tests**: Rotational alignment property (synthetic data)
- ⚠️ **Limited golden outputs**: Only simple rotation matrix tests
- ❌ **No statistical property tests**: No cross-validation, no convergence metrics
- ❌ **No performance benchmarks**: No timing or memory tests

**Implication for nltools**: We can confidently use contract/interface tests for hyperalignment, following Hypertools' pattern of testing mathematical properties (orthogonality, alignment quality) rather than exact numerical outputs.

---

## 1. Algorithm Implementation

### 1.1 Core Algorithm (Three-Stage Procrustes Hyperalignment)

**Location**: `hypertools/tools/align.py`

**Algorithm Steps**:

```python
# STAGE 1: Create Initial Template
# - Start with first subject as seed
# - Iteratively align each subject to evolving average
# - Accumulate aligned subjects and normalize

template = np.copy(m[0])  # First subject
for i, subject in enumerate(m[1:]):
    aligned = procrustes(subject, template / (i+1))
    template += aligned
template /= len(m)

# STAGE 2: Refine Template
# - Align all subjects to current template
# - Average to create refined template
# - (Only 1 iteration in Hypertools implementation)

refined_template = np.zeros(template.shape)
for subject in m:
    aligned = procrustes(subject, template)
    refined_template += aligned
refined_template /= len(m)

# STAGE 3: Final Alignment
# - Align each subject to refined template
# - Store transformation matrices

aligned = []
transformations = []
for subject in m:
    aligned_subject = procrustes(subject, refined_template)
    aligned.append(aligned_subject)
    transformations.append(transformation_matrix)
```

**Key Design Decisions**:
1. **Fixed iterations**: Only 1 refinement iteration (Stage 2 runs once)
2. **Auto-padding**: Handles variable feature counts by zero-padding
3. **Dimension standardization**: Uses min(rows) and max(cols) across subjects

### 1.2 Mathematical Approach

**Procrustes Transformation** (`hypertools/tools/procrustes.py`):

```python
# 1. Center data
mtx1 -= np.mean(mtx1, 0)
mtx2 -= np.mean(mtx2, 0)

# 2. Normalize by Frobenius norm
norm1 = np.linalg.norm(mtx1)
norm2 = np.linalg.norm(mtx2)
mtx1 /= norm1
mtx2 /= norm2

# 3. Find optimal orthogonal transformation via SVD
U, s, Vh = np.linalg.svd(target.T @ source)
R = U @ Vh  # Rotation matrix

# 4. Apply transformation
mtx2_aligned = np.dot(mtx2, R.T) * scale

# 5. Compute disparity (sum of squared differences)
disparity = np.sum(np.square(mtx1 - mtx2_aligned))
```

**Properties**:
- Orthogonal transformations only (R @ R.T = I)
- Preserves relative distances
- Minimizes Frobenius norm between aligned matrices
- Returns: aligned_data, transformation_matrix, disparity, scale

### 1.3 Optimization Techniques

**Vectorization**:
- Matrix operations via NumPy (BLAS/LAPACK backend)
- SVD for optimal rotation finding
- No explicit loops over features/samples

**Memory Efficiency**:
- Zero-padding strategy avoids creating extra copies
- In-place normalization where possible
- Template accumulation uses running average

**Limitations**:
- ❌ No GPU support
- ❌ No parallelization (sequential subject processing)
- ❌ No sparse matrix support
- ❌ No incremental/online learning

**Computational Complexity**:
- Procrustes: O(n²k) where n=samples, k=features (SVD dominated)
- Full hyperalignment: O(N × n²k) where N=num_subjects
- Scales poorly with large feature counts

---

## 2. Testing Strategy

### 2.1 Test File Organization

**Location**: `tests/test_align.py`

**Test Count**: 5 test functions (minimal coverage)

### 2.2 Test Categories

#### A. **Interface/Contract Tests** (60% of tests)

```python
def test_align_shapes():
    """Validates aligned output maintains input dimensions."""
    weights = [random 10×300 arrays] × 3
    aligned = align(weights)

    # Check each aligned result matches corresponding input shape
    for i, result in enumerate(aligned):
        assert result.shape == weights[i].shape
```

**What it tests**: Output shape matches input shape
**What it doesn't test**: Numerical correctness, alignment quality

```python
def test_align_geo():
    """Verifies alignment of geometric data produces matching results."""
    geo = load('spiral')  # Canonical spiral dataset
    aligned = align(geo)

    # Check all aligned datasets match each other
    assert np.allclose(aligned[0], aligned[1])
```

**What it tests**: Subjects align to common space (become similar)
**What it doesn't test**: How well they align, convergence quality

#### B. **Computational Correctness Tests** (40% of tests)

```python
def test_procrustes():
    """Tests Procrustes alignment with known rotation."""
    data1 = load('spiral')[0]

    # Create known rotation matrix
    rot = np.array([[-0.894, -0.447, -0.013],
                    [-0.434,  0.875, -0.214],
                    [-0.108,  0.186,  0.977]])

    # Apply rotation to create misaligned data
    data2 = np.dot(data1, rot)

    # Align and verify recovery
    aligned = align([data1, data2])
    assert np.allclose(data1, aligned[1])  # Should recover original
```

**What it tests**:
- ✅ Rotation invariance (fundamental property)
- ✅ Recovery of known transformation
- ✅ Numerical stability

**What it doesn't test**:
- ❌ Non-synthetic data performance
- ❌ Edge cases (singular matrices, rank deficiency)
- ❌ Statistical properties (bias, variance)

**Tolerance Used**: `rtol=1` (extremely permissive! 100% relative tolerance)

```python
def test_hyper():
    """Same as test_procrustes but for 'hyper' method."""
    # Identical test with align='hyper' parameter
    assert np.allclose(target, aligned, rtol=1)  # Very loose tolerance!
```

**Critical Observation**: The `rtol=1` tolerance means they accept 100% relative error. This suggests they're testing the *property* of alignment (rotation recovery) rather than *precise* numerical agreement.

```python
def test_SRM():
    """Tests Shared Response Model alignment."""
    # Same rotation recovery test
    # Uses align='SRM' instead of 'hyper'
    assert np.allclose(target, aligned, rtol=1)
```

#### C. **Edge Case Tests** (0% - MISSING!)

No tests for:
- Single subject
- Identical subjects
- Different feature counts
- Mismatched sample counts
- Zero/constant data
- Rank-deficient matrices

#### D. **Performance Tests** (0% - MISSING!)

No tests for:
- Timing benchmarks
- Memory usage
- Scalability (large N, large features)
- Convergence speed

### 2.3 Test Data Characteristics

**Synthetic Data**:
```python
weights = [np.random.randn(10, 300) for _ in range(3)]
```
- Small (10 samples × 300 features)
- Random Gaussian
- No structure or realistic patterns

**Canonical Dataset**:
```python
geo = load('spiral')  # 3D spiral trajectory
```
- Well-behaved geometric data
- Low-dimensional (3D)
- Smooth, continuous structure

**Rotation Matrix**:
```python
rot = np.array([[-0.894, -0.447, -0.013],
                [-0.434,  0.875, -0.214],
                [-0.108,  0.186,  0.977]])
```
- Pre-defined 3×3 orthogonal matrix
- No verification it's actually orthogonal
- Reused across multiple tests

### 2.4 Validation Approaches

**Primary Validation**: `np.allclose()` with loose tolerances

**Mathematical Properties Tested**:
1. ✅ **Rotation recovery**: Can recover data after known rotation
2. ✅ **Shape preservation**: Aligned data maintains input dimensions
3. ✅ **Convergence to common space**: All subjects become similar

**Mathematical Properties NOT Tested**:
1. ❌ **Orthogonality**: W @ W.T = I
2. ❌ **Template stability**: Refined template converges
3. ❌ **Disparity minimization**: Sum of squared errors decreases
4. ❌ **Scale preservation**: Frobenius norms handled correctly
5. ❌ **Intersubject correlation**: ISC increases after alignment

---

## 3. API and Design

### 3.1 Input/Output Formats

**Input**:
```python
data = [np.array([observations, features]), ...]  # List of 2D arrays
align(data, align='hyper')
```

**Output**:
```python
aligned = [np.array([observations, features]), ...]  # Same shape as input
```

**Key Design Choice**: Returns aligned data only, no metadata (transformations, disparity, etc.)

### 3.2 Configurable Parameters

```python
align(data,
      align='hyper',      # 'hyper' or 'SRM'
      normalize=None,     # Not used in hyperalignment
      ndims=None,         # Not used in hyperalignment
      method=None,        # Deprecated alias for 'align'
      format_data=True)   # Apply preprocessing
```

**Surprising Limitations**:
- ❌ No n_iter parameter (fixed at 1 refinement)
- ❌ No tolerance/convergence controls
- ❌ No option to return transformation matrices
- ❌ No verbose/logging options

### 3.3 Key Design Decisions

**1. Auto-padding Strategy**:
```python
# Standardize to min(rows) and max(cols)
R = min([x.shape[0] for x in data])
C = max([x.shape[1] for x in data])

# Truncate and pad each subject
for subject in data:
    y = subject[0:R, :]  # Truncate rows
    if y.shape[1] < C:
        y = np.append(y, np.zeros((R, C - y.shape[1])), axis=1)
```

**Implication**: Loses information from high-feature subjects

**2. Template Initialization**:
```python
template = np.copy(m[0])  # Use first subject
```

**Implication**: Order-dependent (first subject influences result)

**3. Single Refinement Iteration**:
```python
# Only runs Stage 2 once (no loop)
```

**Implication**: May not converge to optimal template

---

## 4. Performance Considerations

### 4.1 Memory Efficiency

**Patterns Used**:
- Zero-padding to standardize dimensions
- In-place normalization in Procrustes
- Template accumulation via running average

**Memory Footprint**:
- O(N × n × k) for storing all subjects
- O(n × k) for template
- O(k²) for transformation matrices (if stored)

**Potential Issues**:
- No memory-mapping for large datasets
- No chunking/batching strategies
- Creates copies during preprocessing

### 4.2 Computational Complexity

**Per-Subject Alignment** (Procrustes):
- Centering: O(nk)
- Normalization: O(nk)
- SVD: O(min(n, k)²×max(n, k)) ≈ O(n²k) for typical n < k
- Matrix multiplication: O(nk²)
- **Total**: O(n²k + nk²)

**Full Hyperalignment** (N subjects):
- Stage 1: N × O(n²k) = O(Nn²k)
- Stage 2: N × O(n²k) = O(Nn²k)
- Stage 3: N × O(n²k) = O(Nn²k)
- **Total**: O(Nn²k) where N=subjects, n=samples, k=features

**Bottlenecks**:
1. SVD in Procrustes (cubic in smaller dimension)
2. Sequential subject processing (no parallelization)
3. Dense matrix operations (no sparsity)

### 4.3 Scalability Approaches

**Current Implementation**:
- ❌ No parallelization
- ❌ No GPU support
- ❌ No incremental learning
- ❌ No approximation methods

**Scalability Limits** (estimated on modern CPU):
- **Subjects**: ~100s (linear scaling)
- **Samples**: ~1000s (quadratic scaling hits)
- **Features**: ~10,000s (memory and computation)

**Comparison to nltools**:
- Hypertools: Simpler, fewer options, less scalable
- nltools: More parameters (n_iter), better API, same core algorithm

---

## 5. Comparison: Hypertools vs. nltools

### 5.1 Implementation Differences

| Aspect | Hypertools | nltools |
|--------|-----------|---------|
| **Refinement iterations** | Fixed (1) | Configurable (n_iter) |
| **API style** | Functional | Object-oriented (sklearn) |
| **Output** | Aligned data only | Dict with metadata |
| **Transformation access** | Not returned | Stored in w_, s_ |
| **Auto-padding** | Always on | Configurable |
| **Integration** | Standalone | Part of BrainData |

### 5.2 Algorithm Equivalence

**Core Algorithm**: Identical (both implement Haxby et al. 2011)

**Verification**:
```python
# nltools test_hyperalignment.py (line 367-430)
def test_numerical_match_with_align_procrustes():
    """Verifies HyperAlignment produces same results as align(method='procrustes')."""

    hyper = HyperAlignment(n_iter=1)  # Match Hypertools' single refinement
    hyper.fit(data)

    align_out = align(data, method='procrustes')

    # All outputs match within 1e-5 relative tolerance
    assert np.allclose(hyper.s_, align_out['common_model'])
    assert np.allclose(hyper.w_, align_out['transformation_matrix'])
    assert np.allclose(hyper.disparity_, align_out['disparity'])
```

**Conclusion**: nltools' HyperAlignment is a **superset** of Hypertools' implementation with:
- More configurability (n_iter parameter)
- Better API (sklearn-compliant)
- More outputs (transformation matrices, disparity, scale)
- Same numerical results when n_iter=1

### 5.3 Testing Philosophy Differences

**Hypertools**:
- Minimal tests (5 functions)
- Focus on basic invariants (rotation recovery)
- Very loose tolerances (rtol=1)
- No edge case coverage
- No performance benchmarks

**nltools**:
- Comprehensive tests (27 functions in test_hyperalignment.py)
- Multiple test classes by concern
- Strict tolerances (1e-5 to 1e-8)
- Extensive edge cases
- Mathematical property verification (orthogonality, etc.)

**Philosophy**:
- **Hypertools**: "Trust the math, test the interface"
- **nltools**: "Verify the math, document the behavior"

---

## 6. Key Insights for nltools Testing

### 6.1 What We Can Learn

✅ **Contract tests are sufficient for core functionality**
- Hypertools proves that shape validation and basic properties work
- No need for extensive golden output tests

✅ **Mathematical property tests are valuable**
- Rotation recovery tests fundamental alignment property
- Orthogonality tests ensure transformation quality
- Disparity tests verify optimization

✅ **Integration tests with realistic data matter**
- Hypertools uses canonical datasets (spiral)
- nltools can use Simulator-generated brain data

### 6.2 What We Should Do Better

🎯 **Stricter tolerances**
- Hypertools' rtol=1 is too loose
- Use rtol=1e-5, atol=1e-8 for numerical stability

🎯 **More edge case coverage**
- Single subject, identical subjects
- Variable feature counts
- Rank-deficient data

🎯 **Property-based testing**
- Orthogonality: W @ W.T ≈ I
- Alignment quality: disparity decreases
- Template stability: n_iter converges

🎯 **Avoid exact golden outputs**
- Hypertools doesn't use them (good!)
- Test mathematical properties instead
- Use synthetic data with known properties

### 6.3 Recommended Testing Strategy for nltools

**Tier 1: Contract Tests** (Must have)
- ✅ Correct shapes returned
- ✅ Correct attribute types
- ✅ API compliance (sklearn BaseEstimator)

**Tier 2: Mathematical Property Tests** (Should have)
- ✅ Orthogonality of transformation matrices
- ✅ Rotation recovery (like Hypertools)
- ✅ Alignment quality (correlation increases)
- ✅ Disparity minimization

**Tier 3: Integration Tests** (Should have)
- ✅ Numerical match with align() function
- ✅ BrainData integration
- ✅ Realistic simulated data (Simulator)

**Tier 4: Edge Case Tests** (Nice to have)
- ✅ Single subject
- ✅ Identical subjects
- ✅ Variable feature counts (auto_pad)
- ✅ Error handling (empty list, wrong types)

**Tier 5: Performance Tests** (Optional)
- ⚠️ Not critical (Hypertools doesn't have them)
- Can add later if needed

---

## 7. Specific Code Examples from Hypertools

### 7.1 Core Test Pattern: Rotation Recovery

```python
def test_procrustes():
    """Test that Procrustes recovers data after known rotation."""

    # Load canonical dataset
    data1 = load('spiral').get_data()[0]

    # Define known rotation matrix
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
                    [-0.43426149,  0.87492975, -0.21427761],
                    [-0.10761949,  0.18578133,  0.97667976]])

    # Apply rotation to create misaligned data
    data2 = np.dot(data1, rot)

    # Align - should recover original orientation
    source_aligned = procrustes(data2, data1)

    # Verify recovery (very loose tolerance!)
    assert np.allclose(data1, source_aligned, rtol=1)
```

**Pattern**: Test fundamental mathematical property (rotation invariance) rather than exact outputs

### 7.2 Shape Validation Pattern

```python
def test_align_shapes():
    """Test aligned output maintains input dimensions."""

    # Create random data
    weights = [np.random.randn(10, 300) for _ in range(3)]

    # Align
    aligned = align(weights)

    # Verify each aligned result matches input shape
    for i, result in enumerate(aligned):
        assert result.shape == weights[i].shape
```

**Pattern**: Contract test - verify API behavior, not computational correctness

### 7.3 Convergence to Common Space Pattern

```python
def test_align_geo():
    """Test that alignment produces similar results across subjects."""

    # Load multi-subject data
    geo = load('spiral')

    # Align
    aligned = align(geo)

    # After alignment, all subjects should match
    for i in range(len(aligned) - 1):
        assert np.allclose(aligned[i], aligned[i+1], rtol=1)
```

**Pattern**: Test that alignment achieves its goal (common space) without specifying exact outputs

---

## 8. Conclusions and Recommendations

### 8.1 Main Findings

1. **Hypertools tests primarily contracts, not computational correctness**
   - 60% of tests are shape/API validation
   - 40% test mathematical properties (rotation recovery)
   - 0% test against golden outputs
   - 0% test edge cases or performance

2. **Their testing strategy is adequate but minimal**
   - Works because Procrustes is well-established
   - SVD-based implementation is numerically stable
   - Mathematical properties are well-understood

3. **nltools can follow similar approach with improvements**
   - More comprehensive edge case coverage
   - Stricter numerical tolerances
   - Better sklearn API compliance
   - Integration with BrainData ecosystem

### 8.2 Recommendations for nltools

✅ **DO**:
- Test mathematical properties (orthogonality, rotation recovery, alignment quality)
- Use contract tests for API compliance
- Test integration with align() function for numerical equivalence
- Use realistic simulated data (Simulator)
- Document expected behavior in tests
- Use strict tolerances (1e-5 relative, 1e-8 absolute)

❌ **DON'T**:
- Create golden output files (brittle, unmaintainable)
- Test exact numerical values (platform-dependent)
- Over-test stable implementations (Procrustes is mature)
- Ignore edge cases (nltools has better coverage than Hypertools)

### 8.3 Testing Confidence

**High Confidence Areas** (contract tests sufficient):
- Core alignment algorithm (Procrustes is proven)
- Basic API behavior (shapes, types, sklearn compliance)
- Integration with existing nltools.stats.align()

**Medium Confidence Areas** (property tests recommended):
- Mathematical correctness (orthogonality, disparity)
- Alignment quality (ISC, correlation improvement)
- Template convergence (n_iter parameter)

**Low Confidence Areas** (need careful testing):
- Edge cases (single subject, identical data, rank deficiency)
- Auto-padding behavior (variable feature counts)
- Error handling (wrong inputs, empty data)

### 8.4 Final Answer to Research Question

**Do Hypertools tests verify computational correctness?**

**Answer**: **Partially, through mathematical property testing, not golden outputs.**

They verify:
- ✅ Rotation invariance (fundamental property)
- ✅ Shape preservation (contract)
- ✅ Convergence to common space (goal achievement)

They don't verify:
- ❌ Exact numerical outputs
- ❌ Statistical properties
- ❌ Convergence metrics
- ❌ Performance characteristics

**Implication**: We can confidently use nltools' current test strategy (contract + property tests) without needing computational golden outputs. Our tests are already more comprehensive than Hypertools.

---

## 9. References

**Hypertools**:
- GitHub: https://github.com/ContextLab/hypertools
- Paper: Heusser et al. (2017). JMLR 18(152):1-6
- Docs: https://hypertools.readthedocs.io/

**Algorithm**:
- Haxby et al. (2011). Neuron 72(2):404-416
- Original PyMVPA implementation

**nltools Implementation**:
- `nltools/algorithms/hyperalignment.py`
- `nltools/tests/core/test_hyperalignment.py`
- `nltools/stats.py` (align() function)

---

**Document Status**: Complete
**Next Steps**: Apply insights to nltools hyperalignment testing strategy
**Key Takeaway**: Contract tests + mathematical property tests are sufficient; no golden outputs needed.
