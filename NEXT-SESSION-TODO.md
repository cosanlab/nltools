# Next Session TODO: Complete GPU-Accelerated Inference Modules

**Date Created**: 2025-10-30
**Date Updated**: 2025-10-30 (correctness analysis + phase_randomize fix)
**Branch**: `uv-cleanup`
**Context**: Verified mathematical correctness of all inference implementations. Fixed phase_randomize to use only one variable (correct). All 121 tests passing.

---

## ✅ COMPLETED THIS SESSION

### Mathematical Correctness Analysis (DONE)
- ✅ Comprehensive literature review (Nichols & Holmes 2002, Phipson & Smyth 2010, Theiler et al. 1992)
- ✅ Validated all implementations against published methods
- ✅ Verified cross-backend determinism (0.000% variance)
- ✅ **Found and fixed critical bug**: phase_randomize was randomizing BOTH variables
- ✅ Documented all findings in `inference-correctness-analysis.md` (comprehensive report)

### Phase Randomization Fix (DONE)
- ✅ Fixed `timeseries_correlation_permutation_test()` phase_randomize method
- ✅ Changed from randomizing both variables to only randomizing data1 (correct behavior)
- ✅ Improves statistical power (narrower null distribution)
- ✅ Aligns with standard practice (Good 2000, permutation test literature)
- ✅ Updated docstrings with brief test assumptions
- ✅ Added 3 new validation tests (null distribution centered, detects correlation, consistency)

### Documentation Updates (DONE)
- ✅ Added assumption notes to all user-facing function docstrings
- ✅ Updated DESIGN.md with correctness fix details
- ✅ Created `inference-correctness-analysis.md` with full literature review and findings

**Files modified**:
- `nltools/algorithms/inference/timeseries.py` (fixed phase_randomize, updated docstrings)
- `nltools/algorithms/inference/one_sample.py` (added assumption docstring)
- `nltools/algorithms/inference/two_sample.py` (added assumption docstring)
- `nltools/algorithms/inference/correlation.py` (added assumption docstring)
- `nltools/algorithms/inference/DESIGN.md` (documented fix)
- `nltools/tests/core/test_inference.py` (+3 validation tests, 121 total passing)
- `inference-correctness-analysis.md` (NEW, comprehensive correctness report)

---

## 🚧 TODO FOR NEXT SESSION

### Priority 1: Complete correlation.py Metrics

**Add Spearman correlation support**:
- [ ] Implement `_spearman_correlation()` helper function
  - Use scipy.stats.spearmanr or rank-based computation
  - Ensure GPU compatibility (convert to ranks, then use Pearson on ranks)
- [ ] Update `_correlation_permutation_cpu_parallel()` to handle Spearman
- [ ] Update `_correlation_permutation_gpu_batched()` to handle Spearman
- [ ] Add tests for Spearman metric
- [ ] Remove `NotImplementedError` for 'spearman'

**Add Kendall correlation support**:
- [ ] Implement `_kendall_correlation()` helper function
  - More complex (concordant/discordant pairs)
  - May be challenging on GPU - CPU-parallel may be sufficient
- [ ] Update correlation functions to handle Kendall
- [ ] Add tests for Kendall metric
- [ ] Remove `NotImplementedError` for 'kendall'

**Testing**:
- [ ] Verify Spearman matches scipy.stats.spearmanr
- [ ] Verify Kendall matches scipy.stats.kendalltau
- [ ] Test backend consistency for all metrics

---

### Priority 2: Build timeseries.py Module

**Implement core functions**:
- [ ] `circle_shift(data, shift_amount=None, random_state=None)`
  - Circular time-series shift
  - Preserves autocorrelation
  - For 1D: single shift amount
  - For 2D: shift each feature independently
  - Reference: `nltools/stats.py` lines 1818-1847

- [ ] `phase_randomize(data, backend=None, random_state=None)`
  - FFT-based phase randomization
  - Preserves power spectrum, destroys phase
  - Algorithm:
    1. FFT: `np.fft.rfft(data)` or `torch.fft.rfft(data)`
    2. Generate random phases [0, 2π]
    3. Apply phase shifts while preserving magnitude
    4. Ensure conjugate symmetry for real output
    5. Inverse FFT: `np.fft.irfft()` or `torch.fft.irfft()`
  - **GPU-friendly**: Excellent candidate for GPU acceleration
  - Reference: `nltools/stats.py` lines 1760-1817

- [ ] `timeseries_correlation_permutation_test(...)`
  - Similar to correlation_permutation_test
  - Parameter: `method='circle_shift'` or `method='phase_randomize'`
  - Uses circle_shift or phase_randomize instead of simple permutation
  - CPU-parallel and GPU-batched implementations

**Testing**:
- [ ] `TestCircleShift` class
  - Preserves shape (1D and 2D)
  - Deterministic with seed
  - Preserves values (just reorders)

- [ ] `TestPhaseRandomize` class
  - **CRITICAL**: Preserves power spectrum
    ```python
    power_orig = np.abs(np.fft.rfft(data))**2
    power_rand = np.abs(np.fft.rfft(phase_randomize(data)))**2
    np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)
    ```
  - Changes phase (verify randomization works)
  - Backend consistency (NumPy vs PyTorch FFT)
  - Deterministic with seed

- [ ] `TestTimeseriesCorrelation` class
  - Test both circle_shift and phase_randomize methods
  - Matches stats.py `correlation_permutation(method='circle_shift')`
  - Matches stats.py `correlation_permutation(method='phase_randomize')`
  - CPU-parallel and GPU-batched correctness

**References**:
- Theiler et al. (1991). Testing for nonlinearity in time series
- Lancaster et al. (2018). Surrogate data for hypothesis testing

---

### Priority 3: Build matrix.py Module

**Implement core functions**:
- [ ] Helper: `_extract_matrix_elements(matrix, how='upper', include_diag=False)`
  - Extract elements for comparison
  - `how='upper'`: Upper triangle (default, assumes symmetric)
  - `how='lower'`: Lower triangle
  - `how='full'`: Full matrix
  - `include_diag`: Whether to include diagonal (for `how='full'`)

- [ ] Helper: `_permute_matrix(matrix, permutation)`
  - **KEY OPERATION**: Symmetric row+column permutation
  - `permuted = matrix[perm][:, perm]`
  - This reorders both rows and columns together

- [ ] `matrix_permutation_test(data1, data2, ...)`
  - Mantel test for 2D matrix correlation
  - Input: Two square matrices (n×n)
  - Permutation: Symmetrically permute rows AND columns of data2
  - Statistic: Correlation between matrix elements
  - Parameters:
    - `how='upper'|'lower'|'full'`
    - `include_diag=bool`
    - `metric='pearson'` (start with this)
  - CPU-parallel implementation (primary)
  - GPU-batched implementation (bonus - challenging due to indexing)
  - Reference: `nltools/stats.py` lines 737+

**Testing**:
- [ ] `TestMatrixHelpers` class
  - Extract upper/lower/full triangle
  - Permute matrix symmetrically

- [ ] `TestMatrixPermutation` class
  - Basic functionality (square matrices)
  - Identical matrices (perfect correlation, p < 0.05)
  - Uncorrelated matrices (random, p > 0.05)
  - Different `how` parameters
  - Diagonal inclusion
  - Input validation (non-square, mismatched sizes)
  - Matches stats.py `matrix_permutation()`
  - CPU-parallel correctness
  - GPU-batched correctness (if implemented)

**GPU Challenges**:
- Advanced indexing `matrix[perm][:, perm]` is slow on GPU
- May need creative batching or stick with CPU-parallel only
- CPU-parallel is acceptable primary implementation

**References**:
- Mantel (1967). The detection of disease clustering
- Chen et al. (2016). Untangling the relatedness among correlations

---

### Priority 4: Integration and Documentation

**Update exports**:
- [ ] Add to `nltools/algorithms/inference/__init__.py`:
  - `circle_shift`
  - `phase_randomize`
  - `timeseries_correlation_permutation_test`
  - `matrix_permutation_test`

**Test integration**:
- [ ] Run full tier1 test suite: `uv run pytest -m tier1 -n auto`
- [ ] Verify no regressions in existing tests
- [ ] All new tests passing

**Documentation**:
- [ ] Update module docstring in `__init__.py` with new functions
- [ ] Add examples to each function docstring
- [ ] Update `docs/migration-guide.md` if API changes

**Optional (if time)**:
- [ ] Performance benchmarks (compare to stats.py)
- [ ] Memory profiling for GPU batching
- [ ] Add to refactor-progress.md

---

## 📋 ESTIMATED EFFORT

- **Correlation metrics** (Spearman/Kendall): 1-2 hours
- **timeseries.py**: 2-3 hours
- **matrix.py**: 2-3 hours
- **Integration & testing**: 1 hour

**Total**: 6-9 hours of focused work

---

## 🔍 IMPORTANT NOTES

**TDD Workflow** (follow religiously):
1. Write test first (it will fail)
2. Implement minimal code to pass
3. Run ONLY that test: `uv run pytest path/to/test::TestClass::test_name -x`
4. Iterate until test passes
5. Add next test
6. Run module tests: `uv run pytest path/to/test::TestClass -n auto -x`
7. Run tier1 regression: `uv run pytest -m tier1 -n auto`

**Testing defaults**:
- ALWAYS use `-n auto` for tier1 tests (parallel by default)
- ASK permission before tier2 tests (~7 min)
- Use targeted tests during development (NOT full suite)
- Create log files: `uv run pytest ... 2>&1 | tee test.log`

**Git workflow**:
- Do NOT stage changes automatically
- When ready: Say "Changes ready for review" and WAIT
- Eshin stages manually or says "stage the changes"
- Then commit with detailed message

**Pattern to follow**:
- Study `one_sample.py` and `two_sample.py` as templates
- CPU-parallel implementation (joblib with progress bars)
- GPU-batched implementation (automatic batching)
- Main function routes to appropriate backend
- Comprehensive tests following existing pattern

---

## 📊 CURRENT STATE

**Working directory**: `/Users/esh/Documents/pypackages/nltools`
**Branch**: `uv-cleanup`
**Test status**: 121 inference tests (all passing), 385 total tests (381 passing, 4 skipped)

**Files to continue editing**:
- `nltools/algorithms/inference/correlation.py` (add Spearman/Kendall)
- `nltools/algorithms/inference/timeseries.py` (create new)
- `nltools/algorithms/inference/matrix.py` (create new)
- `nltools/algorithms/inference/__init__.py` (update exports)
- `nltools/tests/core/test_inference.py` (append new test classes)

**Reference files**:
- `nltools/stats.py` - Current implementations to match
- `nltools/algorithms/inference/one_sample.py` - Template pattern
- `nltools/algorithms/inference/two_sample.py` - Template pattern
- `claude-guidelines/inference-expansion-plan.md` - Detailed implementation plan

---

## ✅ VERIFICATION BEFORE RESUMING

When starting next session:
1. Read this TODO file
2. Check `git log -1` to see last commit
3. Run `uv run pytest -m tier1 -n auto` to verify current state
4. Review `claude-guidelines/inference-expansion-plan.md` for detailed guidance
5. Start with highest priority item (Spearman correlation)

---

**Last Updated**: 2025-10-30
**Status**: Mathematical correctness verified, phase_randomize fixed, all 121 tests passing
