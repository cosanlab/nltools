# TDD Plan: Extract HyperAlignment from align() into Separate Class

## Design Decisions (Confirmed)
1. ✅ Expose `n_iter` parameter (default=2) for template refinement
2. ✅ Use `s_` as primary attribute, `common_model_` as property alias
3. ✅ Expose `auto_pad` kwarg for padding control (default=True for backward compatibility)
4. ✅ Maintain exact numerical backward compatibility

## Phase 1: Write Tests for Current Procrustes Behavior (TDD Red Phase)

**Goal**: Document all current procrustes behavior with comprehensive tests before refactoring

### 1.1 Create Test File Structure
```bash
# Create new test file
nltools/tests/core/test_hyperalignment.py
```

### 1.2 Write Tests for Core HyperAlignment Class API
Based on SRM pattern and current align() behavior:

**Test categories**:
1. **Initialization tests** (n_iter parameter)
2. **fit() method tests**:
   - Basic fit with 2 subjects
   - Fit with 5 subjects (current test_align uses 5)
   - Fit with different-sized matrices (zero-padding)
   - Fit with single subject (edge case)
   - Fit with identical subjects (edge case)
3. **transform() method tests**:
   - Transform training data
   - Verify orthogonality of transformations
   - Check transformed data shape consistency
4. **transform_subject() method tests**:
   - Transform a new subject to common space
   - Verify alignment quality
5. **Attribute tests**:
   - `w_` (transformation matrices)
   - `s_` (common template)
   - `common_model_` (property alias)
   - `disparity_` (per-subject quality)
   - `scale_` (per-subject scale factors)
6. **Three-stage refinement tests**:
   - Test that n_iter controls refinement iterations
   - Verify stage 1 → stage 2 → stage 3 progression
7. **Padding behavior tests**:
   - Test auto_pad=True (automatic padding)
   - Test auto_pad=False (caller must pad)
   - Test error handling for mismatched sizes with auto_pad=False
8. **Numerical correctness tests**:
   - Compare outputs to current align() implementation
   - Verify exact numerical match (backward compatibility)

**Estimated**: ~25-30 tests

### 1.3 Write Integration Tests for align() Using HyperAlignment Class
Update existing tests in `nltools/tests/core/test_stats.py`:

1. Keep existing `test_align()` structure (currently skipped)
2. Add tests specifically for procrustes method:
   - Matrix input with method='procrustes'
   - Brain_Data input with method='procrustes'
   - axis=1 alignment
3. Verify backward compatibility (same output structure as current)

**Expected outcome**: All new tests FAIL (Red phase) because HyperAlignment class doesn't exist yet

---

## Phase 2: Implement HyperAlignment Class (TDD Green Phase)

**Goal**: Implement minimal HyperAlignment class to make tests pass

### 2.1 Create Class Skeleton
```bash
# Create new algorithm file
nltools/algorithms/hyperalignment.py
```

**Implementation order** (incremental, guided by failing tests):

1. **Basic class structure**:
   ```python
   class HyperAlignment(BaseEstimator, TransformerMixin):
       """Hyperalignment using iterative Procrustes alignment.

       Three-stage iterative process for aligning multi-subject data:
       1. Create initial average template
       2. Refine template through n_iter iterations
       3. Final alignment of all subjects to refined template

       This implements the Procrustes-based hyperalignment method commonly
       used in multi-subject neuroimaging analysis.

       Parameters
       ----------
       n_iter : int, default=2
           Number of template refinement iterations (stages 1-2).
       auto_pad : bool, default=True
           If True, automatically zero-pad matrices to standardize sizes.
           If False, caller must ensure all matrices have same dimensions.

       Attributes
       ----------
       w_ : list of array, element i has shape=[features_i, features]
           The transformation matrices (rotation + reflection) for each subject.
       s_ : array, shape=[features, samples]
           The aligned common template (shared response).
       common_model_ : array, shape=[features, samples]
           Alias for s_ (for backward compatibility with align() output).
       disparity_ : list of float
           Disparity (sum of squared differences) for each subject.
       scale_ : list of float
           Scale factors for each subject.

       Examples
       --------
       Basic multi-subject alignment:

       >>> from nltools.algorithms import HyperAlignment
       >>> import numpy as np
       >>>
       >>> # Create sample data (3 subjects)
       >>> data = [np.random.randn(100, 50) for _ in range(3)]
       >>>
       >>> # Fit hyperalignment
       >>> hyper = HyperAlignment(n_iter=2)
       >>> hyper.fit(data)
       >>>
       >>> # Transform to common space
       >>> aligned = hyper.transform(data)
       >>>
       >>> # Access common template
       >>> template = hyper.s_  # or hyper.common_model_
       >>>
       >>> # Align a new subject
       >>> new_subject = np.random.randn(100, 50)
       >>> new_transform = hyper.transform_subject(new_subject)

       References
       ----------
       Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
       Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
       A common, high-dimensional model of the representational space in
       human ventral temporal cortex. Neuron, 72(2), 404-416.
       """
       def __init__(self, n_iter=2, auto_pad=True):
           self.n_iter = n_iter
           self.auto_pad = auto_pad
   ```

2. **fit() method - Stage 0** (Standardization):
   - Implement size standardization
   - Implement zero-padding (if auto_pad=True)
   - Store original shapes for reverse mapping
   - Run tests → some should pass

3. **fit() method - Stage 1** (Initial template):
   - Use first subject as initial template
   - Iteratively align and accumulate
   - Run tests → more should pass

4. **fit() method - Stage 2** (Refined template):
   - Align to stage 1 template
   - Refine n_iter times
   - Run tests → more should pass

5. **fit() method - Stage 3** (Final alignment):
   - Align all subjects to refined template
   - Store w_, disparity_, scale_
   - Compute common model (s_)
   - Add common_model_ property alias
   - Run tests → fit() tests should pass

6. **transform() method**:
   - Apply stored transformations
   - Handle padding if needed
   - Run tests → transform() tests should pass

7. **transform_subject() method**:
   - Align new subject to s_
   - Use pairwise procrustes function
   - Run tests → all tests should pass

### 2.2 Extract Helper Functions
Move these from stats.py to hyperalignment.py:

1. **_procrustes_pairwise()** (current `procrustes()` function):
   - Lines 1457-1541 in stats.py
   - Rename to make clear it's internal helper
   - Keep as static method or module-level function
   - Used internally by HyperAlignment class

2. **Keep for reference**: `procrustes()` function in stats.py should remain (public API)

### 2.3 Update Module Exports
```python
# In nltools/algorithms/__init__.py
from .hyperalignment import HyperAlignment

# In nltools/algorithms/hyperalignment.py
__all__ = ['HyperAlignment']
```

**Expected outcome**: All HyperAlignment class tests PASS (Green phase)

---

## Phase 3: Integrate HyperAlignment Class into align() (TDD Green Phase)

**Goal**: Replace inline procrustes code in align() with HyperAlignment class

### 3.1 Modify align() Function
In `nltools/stats.py`, update the `align()` function:

**Before** (lines 1353-1412):
```python
elif method == "procrustes":
    # ... ~60 lines of inline implementation ...
```

**After**:
```python
elif method == "procrustes":
    from nltools.algorithms import HyperAlignment

    # Transpose handled by align() for axis parameter
    # data is already in correct orientation [features, samples]

    hyper = HyperAlignment(n_iter=2, auto_pad=True)
    hyper.fit(data)

    out['transformed'] = hyper.transform(data)
    out['common_model'] = hyper.s_
    out['transformation_matrix'] = hyper.w_
    out['disparity'] = hyper.disparity_
    out['scale'] = hyper.scale_
```

### 3.2 Run Full Test Suite
```bash
# Run all tests to verify backward compatibility
uv run pytest nltools/tests/core/test_stats.py::test_align -xvs
uv run pytest nltools/tests/shell/test_brain_data.py::TestBrainData::test_align -xvs
uv run pytest nltools/tests/core/test_hyperalignment.py -xvs

# Run full suite
uv run pytest nltools/tests/ -x
```

**Expected outcome**: ALL existing tests PASS (no regressions)

---

## Phase 4: Refactor and Optimize (TDD Refactor Phase)

**Goal**: Improve code quality while keeping tests green

### 4.1 Code Quality Improvements
1. Add comprehensive docstrings (NumPy style)
2. Add type hints where appropriate
3. Optimize memory usage (avoid unnecessary copies)
4. Add numerical stability checks

### 4.2 Update __all__ Exports
```python
# In nltools/stats.py
__all__ = [
    # ... existing exports ...
    # Note: procrustes() function stays (public API)
]

# In nltools/algorithms/hyperalignment.py
__all__ = ['HyperAlignment']
```

### 4.3 Run Tests After Each Refactor
```bash
# After each improvement, verify tests still pass
uv run pytest nltools/tests/core/test_hyperalignment.py -x
uv run pytest nltools/tests/core/test_stats.py::test_align -x
```

---

## Phase 5: Documentation and Migration (TDD Complete)

### 5.1 Update Documentation Files

**MIGRATION_v0.5_to_v0.6.md**:
```markdown
### New: HyperAlignment Class

For advanced users who want more control over hyperalignment:

```python
from nltools.algorithms import HyperAlignment

# Direct class usage (Procrustes-based hyperalignment)
hyper = HyperAlignment(n_iter=3, auto_pad=True)
hyper.fit(data_list)
transformed = hyper.transform(data_list)

# Access attributes
common_template = hyper.s_  # or hyper.common_model_
transformations = hyper.w_
quality_metrics = hyper.disparity_
scale_factors = hyper.scale_

# Transform a new subject to the common space
new_transform = hyper.transform_subject(new_subject_data)
```

**Backward compatibility**: `align(method='procrustes')` still works identically.
The new `HyperAlignment` class provides the same functionality with a cleaner
sklearn-style API for users who want more control.

**Note**: The term "hyperalignment" refers to the alignment technique, while
"procrustes" refers to the specific mathematical method used. Both terms are
used in the neuroimaging literature.
```

**REFACTORING_PLAN.md**:
Add new section:
```markdown
### Priority 2.6: Extract HyperAlignment as Separate Class ✅ COMPLETE (2025-10-28)
- ✅ Created HyperAlignment class following SRM pattern
- ✅ Extracted procrustes method from align() function
- ✅ 25-30 new tests in test_hyperalignment.py
- ✅ Backward compatible with align(method='procrustes')
- ✅ Exposed n_iter and auto_pad parameters
- ✅ Dual naming: s_ (primary) + common_model_ (alias)
- ✅ Implements Procrustes-based hyperalignment algorithm
```

### 5.2 Update align() Docstring
Update the docstring in `align()` function to reference HyperAlignment:

```python
def align(data, method="deterministic_srm", n_features=None, axis=0, *args, **kwargs):
    """Align subject data into a common response model.

    ...

    method: (str) alignment method to use
        ['probabilistic_srm','deterministic_srm','procrustes']

        Note: For 'procrustes' method, consider using the HyperAlignment class
        directly for more control:

            from nltools.algorithms import HyperAlignment
            hyper = HyperAlignment(n_iter=2, auto_pad=True)
            hyper.fit(data)
            aligned = hyper.transform(data)

    ...
    """
```

### 5.3 Update CLAUDE.md
Add to "Quick Reference" section:
```markdown
| Extract HyperAlignment | Research → Plan → TDD | "Extract procrustes" → [extract-hyperalignment.md] → [Execute] |
```

---

## Phase 6: Commit and Tag

### 6.1 Stage Changes
```bash
git add nltools/algorithms/hyperalignment.py
git add nltools/tests/core/test_hyperalignment.py
git add nltools/stats.py  # Modified align()
git add nltools/algorithms/__init__.py
git add MIGRATION_v0.5_to_v0.6.md
git add REFACTORING_PLAN.md
git add extract-hyperalignment.md  # This plan document
```

### 6.2 Commit with Detailed Message
```bash
git commit -m "feat: Extract HyperAlignment class from align() function

Following SRM pattern, extract procrustes-based hyperalignment from
align() function into nltools.algorithms.HyperAlignment class.

Design decisions:
- Expose n_iter parameter (default=2) for template refinement
- Use s_ as primary attribute, common_model_ as property alias
- Expose auto_pad kwarg for padding control (default=True)
- Maintain exact numerical backward compatibility

Implementation:
- Created nltools/algorithms/hyperalignment.py with HyperAlignment class
- Follows sklearn BaseEstimator/TransformerMixin pattern
- Implements fit(), transform(), transform_subject() methods
- Stores w_, s_, disparity_, scale_ attributes
- Three-stage refinement: initial template → refined → final

Testing:
- Added 25-30 new tests in test_hyperalignment.py
- Tests cover: basic API, edge cases, padding, numerical correctness
- All existing align() tests pass (backward compatible)
- 100% test coverage for new class

Integration:
- align(method='procrustes') now uses HyperAlignment class internally
- Maintains identical output structure
- No breaking changes to public API

Background:
Hyperalignment refers to the alignment technique, while Procrustes refers
to the specific mathematical method (orthogonal Procrustes problem solved
via SVD). The HyperAlignment class implements Procrustes-based hyperalignment
as described in Haxby et al. (2011).

Refs: #TBD (GitHub issue if created)
"
```

### 6.3 Create Git Tag (Optional)
```bash
git tag -a v0.6.0-hyperalignment-class -m "HyperAlignment extracted as separate class"
```

---

## Success Criteria

### Must Have ✅
- [ ] All new HyperAlignment tests pass (25-30 tests)
- [ ] All existing align() tests pass (backward compatibility)
- [ ] Full test suite passes (no regressions)
- [ ] HyperAlignment class follows SRM pattern (sklearn-style API)
- [ ] Exact numerical match with current align() behavior
- [ ] Clear docstrings with examples and references
- [ ] Updated MIGRATION_v0.5_to_v0.6.md

### Nice to Have ⭐
- [ ] Performance benchmarks (compare to inline implementation)
- [ ] Memory profiling (ensure no regression)
- [ ] Example notebook showing advanced usage
- [ ] Consider exposing via nltools.stats or keeping in algorithms module

---

## Estimated Timeline

**Total**: ~6-8 hours

- **Phase 1** (Write tests): 2-3 hours
  - Design test structure: 30 min
  - Write core API tests: 1 hour
  - Write integration tests: 30 min
  - Write edge case tests: 1 hour

- **Phase 2** (Implement class): 2-3 hours
  - Class skeleton: 15 min
  - Stage 0-3 implementation: 1.5 hours
  - transform() and transform_subject(): 45 min
  - Debug until tests pass: 30-45 min

- **Phase 3** (Integrate into align): 30 min
  - Modify align() function: 15 min
  - Run full test suite: 15 min

- **Phase 4** (Refactor): 1 hour
  - Docstrings and type hints: 30 min
  - Code quality improvements: 30 min

- **Phase 5** (Documentation): 30 min
  - Update migration guide: 15 min
  - Add examples and references: 15 min

- **Phase 6** (Commit): 15 min
  - Stage and review: 10 min
  - Write commit message: 5 min

---

## Risk Mitigation

**Risk 1**: Numerical differences from current implementation
- **Mitigation**: Phase 1 includes exact numerical comparison tests
- **Fallback**: If differences found, use np.testing.assert_allclose() to verify they're within floating point precision

**Risk 2**: Test complexity explosion
- **Mitigation**: Focus on core functionality first, add edge cases incrementally
- **Fallback**: Can skip some edge case tests initially and add later

**Risk 3**: Integration breaks existing functionality
- **Mitigation**: Phase 3 runs full test suite before proceeding
- **Fallback**: Git revert if integration fails, debug in isolation

**Risk 4**: Performance regression
- **Mitigation**: Can add benchmarks if concerned
- **Fallback**: Should be negligible (same algorithm, just refactored)

---

## Technical Background

### What is Hyperalignment?

Hyperalignment is a technique for aligning functional neuroimaging data across subjects by finding a common representational space. The goal is to transform each subject's brain activity patterns so they align with a common template, enabling better cross-subject analysis.

### Procrustes-Based Hyperalignment

The current implementation uses **orthogonal Procrustes alignment**:
1. Finds optimal rotation/reflection matrix to align one matrix to another
2. Minimizes sum of squared differences (Frobenius norm)
3. Constraint: transformation must be orthogonal (preserves distances)

**Three-stage iterative refinement**:
- **Stage 1**: Create initial average template by iteratively aligning subjects
- **Stage 2**: Refine template by aligning all subjects to Stage 1 template
- **Stage 3**: Final alignment of all subjects to refined template

This iterative process improves alignment quality compared to single-pass methods.

### Mathematical Foundation

Given two matrices X₁ and X₂, find orthogonal matrix R that minimizes:
```
||X₁ - X₂R||²_F
```

Solution via SVD:
```
A = X₁ᵀX₂
U, Σ, Vᵀ = SVD(A)
R = UVᵀ
```

See: `scipy.linalg.orthogonal_procrustes` and `nltools.stats.procrustes()`

### References

- Haxby, J. V., et al. (2011). A common, high-dimensional model of the representational space in human ventral temporal cortex. *Neuron*, 72(2), 404-416.
- Gower, J. C., & Dijksterhuis, G. B. (2004). *Procrustes problems* (Vol. 30). Oxford University Press.

---

## Next Immediate Steps (When Approved)

1. Create todo list with TodoWrite tool
2. Create `nltools/tests/core/test_hyperalignment.py`
3. Write first test: `test_hyperalignment_init()`
4. Run test → verify it fails (Red phase)
5. Proceed with Phase 1.2 (write all tests)

Ready to execute when approved! 🚀

---

## Appendix: Current Implementation Analysis

### Current Code Structure (stats.py)

**align() function** (lines 1275-1454):
- Entry point: handles data type conversion, axis parameter
- Routes to SRM classes or procrustes inline code
- Returns dict with consistent structure

**procrustes() helper** (lines 1457-1541):
- Pairwise alignment between two matrices
- Returns: mtx1, mtx2, disparity, R, scale
- Used by inline procrustes code in align()

**Inline procrustes in align()** (lines 1353-1412):
- ~60 lines implementing 3-stage refinement
- Handles zero-padding for different sized matrices
- Accumulates transformation matrices and quality metrics

### Test Coverage (test_stats.py)

**test_align()** (lines 305-494, currently SKIPPED):
- Tests all three methods: deterministic_srm, probabilistic_srm, procrustes
- Tests with numpy arrays and Brain_Data objects
- Tests axis=0 (align features) and axis=1 (align samples)
- Verifies output structure and numerical correctness
- Currently skipped due to ISC dimension bug (unrelated to procrustes)

**Brain_Data.align()** (test_brain_data.py, lines 668-716):
- Tests instance method for aligning single Brain_Data to template
- Uses procrustes method
- Verifies transformation and scale application

### Design Rationale

**Why extract as separate class?**
1. **Modularity**: ~60 lines of inline code → reusable class
2. **Consistency**: Matches SRM/DetSRM pattern already in algorithms/
3. **Extensibility**: Easier to add features (PCA preprocessing, different metrics)
4. **Testability**: Can test HyperAlignment in isolation
5. **Discoverability**: Users can find and use directly via nltools.algorithms

**Why "HyperAlignment" not "Procrustes"?**
1. **Domain terminology**: "Hyperalignment" is the neuroimaging term
2. **Clarity**: Describes what it does (align subjects) vs. how (procrustes math)
3. **Namespace**: procrustes() function already exists for pairwise alignment
4. **Literature**: Matches terminology in Haxby et al. (2011) and related papers

**Why keep align(method='procrustes')?**
1. **Backward compatibility**: Existing user code won't break
2. **Convenience**: High-level function handles Brain_Data integration
3. **Consistency**: Matches existing align() API for SRM methods
