# Refactoring Action Plan for nltools v0.6.0

## 📊 Progress Tracker
**Last Updated:** 2025-10-28

### Priority 1: Library Refactoring ✅ COMPLETE (90%)
- ✅ Deleted Priority 3 files (brain_collection, model, specs)
- ✅ Added deprecation stubs for removed methods
- ✅ Implemented `.regress()` with nilearn
- ✅ Added `.compute_contrasts()` method
- ✅ Refactored `.extract_roi()` to use NiftiLabelsMasker
- ⚠️ 11 tests still failing (27/38 passing = 71%)

### Priority 2: Documentation ⏳ UP NEXT
- ⬜ Fix remaining test failures
- ⬜ Migrate from Sphinx to Jupyter Book
- ⬜ Update tutorials for new API

### Priority 3: New Features 🔮 FUTURE
- ⬜ Implement Model class
- ⬜ Implement Brain_Collection
- ⬜ Add advanced ML workflows

---

## Core Strategy
Use v0.5.1 as baseline for functionality that MUST be preserved. Ignore post-v0.5.1 features (brain_collection, model, etc.) which belong to Priority 3.

## Phase 1: Baseline Assessment & Test Stabilization

### 1.1 Identify v0.5.1 Core Functionality
```bash
# Check out v0.5.1 to understand baseline
git checkout v0.5.1
# Document all public APIs
# Return to current branch
git checkout uv-cleanup
```

**Files to PRESERVE functionality from (v0.5.1 baseline):**
- `nltools/data/brain_data.py`
- `nltools/data/adjacency.py`
- `nltools/data/design_matrix.py`
- All functional modules (stats.py, utils.py, plotting.py, etc.)

**Files to IGNORE for now (post-v0.5.1):**
- `nltools/data/brain_collection.py` ❌
- `nltools/data/model.py` ❌
- `nltools/data/_validation.py` (Keep but don't prioritize)
- `nltools/tests/test_brain_collection.py` ❌
- Related research in `claude-research/specs/` ❌

### 1.2 Fix Failing Tests (Make v0.5.1 API Work)
Current failure: `.predict()` method missing but tests expect it

**Immediate Actions:**
1. [ ] Temporarily restore `.predict()` with deprecation warning
2. [ ] Mark methods for removal with clear deprecation plan
3. [ ] Get all v0.5.1 tests passing first, THEN refactor

## Phase 2: Method-by-Method Refactoring

### 2.1 Brain_Data Methods to Refactor

#### Methods to REMOVE (with deprecation):
```python
# Add deprecation warnings first, remove in v0.7.0
.predict()       -> Deprecate, point to scikit-learn
.predict_multi() -> Deprecate, point to scikit-learn
.randomise()     -> Deprecate, point to nilearn.mass_univariate
.ttest()         -> Deprecate, point to scipy.stats or nilearn
```

#### Methods to REFACTOR to use nilearn:
```python
.apply_mask()    -> Use nilearn.masking functions
.extract_roi()   -> Use NiftiLabelsMasker
.smooth()        -> Use nilearn.image.smooth_img
.resample()      -> Use nilearn.image.resample_img
```

#### Attributes to RENAME:
```python
.X -> .design_matrix (only set by .regress())
.Y -> Remove (users should manage labels separately)
```

#### New Methods to ADD:
```python
.compute_contrasts() -> Wrapper for nilearn's compute_contrasts
```

### 2.2 Implementation Order

**Week 1: Stabilization**
1. [ ] Add deprecation warnings to methods being removed
2. [ ] Fix all test failures while maintaining v0.5.1 API
3. [ ] Document breaking changes clearly
4. [ ] Run full test suite to establish baseline

**Week 2: Replace with nilearn**
1. [ ] Audit each method for nilearn equivalent
2. [ ] Create research doc: `claude-research/nilearn-replacements.md`
3. [ ] Implement one method at a time with tests:
   - [ ] .apply_mask() -> nilearn.masking
   - [ ] .smooth() -> nilearn.image.smooth_img
   - [ ] .extract_roi() -> NiftiLabelsMasker
   - [ ] .resample() -> nilearn.image.resample_img

**Week 3: Refactor regress() and GLM**
1. [ ] Update .regress() to require Design_Matrix
2. [ ] Store results as attributes not dict
3. [ ] Implement .compute_contrasts()
4. [ ] Update all GLM-related tests

**Week 4: Clean up and optimize**
1. [ ] Remove code duplication
2. [ ] Improve __init__ to support alternative maskers
3. [ ] Profile memory usage on large datasets
4. [ ] Update all docstrings

## Phase 3: Test Coverage Improvement

### 3.1 Test Audit
```bash
# Generate coverage report for v0.5.1 functionality
uv run pytest --cov=nltools --cov-report=html

# Identify gaps in:
# - Core Brain_Data operations
# - Design_Matrix functionality
# - Adjacency methods
# - Statistical functions
```

### 3.2 Test Cleanup
- [ ] Remove tests that duplicate nilearn's test coverage
- [ ] Add integration tests for common workflows
- [ ] Ensure backwards compatibility tests for deprecations
- [ ] Add performance benchmarks for key operations

## Phase 4: Documentation Update

### 4.1 Update CLAUDE.md
- [ ] Add deprecation timeline
- [ ] Document nilearn function mappings
- [ ] Update code examples

### 4.2 Migration Guide
Create `MIGRATION_v0.5_to_v0.6.md`:
- [ ] List all breaking changes
- [ ] Provide before/after examples
- [ ] Suggest alternatives for removed methods

## Success Criteria

### Must Have (v0.6.0 release blockers):
- ✅ All v0.5.1 public APIs work (even if deprecated)
- ✅ All tests pass
- ✅ Clear deprecation warnings with migration path
- ✅ No performance regressions
- ✅ Documentation updated

### Nice to Have:
- ⭐ Reduced code size by 30%
- ⭐ Improved test coverage to 80%+
- ⭐ Memory usage reduced for large datasets
- ⭐ Cleaner separation of concerns

## Timeline

**Week 1**: Baseline & Stabilize (get tests passing)
**Week 2**: Replace custom implementations with nilearn
**Week 3**: Refactor regress() and related methods
**Week 4**: Clean up, optimize, document
**Week 5**: Testing, review, release prep

## Next Immediate Steps

1. **Check v0.5.1 baseline:**
```bash
git checkout v0.5.1
ls nltools/data/  # Note what existed
git checkout uv-cleanup
```

2. **Fix first test failure:**
```python
# In brain_data.py, temporarily add:
def predict(self, *args, **kwargs):
    warnings.warn(
        "predict() is deprecated and will be removed in v0.7.0. "
        "Use scikit-learn directly for prediction.",
        DeprecationWarning,
        stacklevel=2
    )
    # Temporary implementation or raise NotImplementedError
```

3. **Run tests to find next issue:**
```bash
uv run pytest -x --tb=short
```

## Research Needs

Before implementing each refactor, create/update research docs:

1. [ ] `claude-research/nilearn-replacements.md` - Map each custom method to nilearn equivalent
2. [ ] `claude-research/deprecation-strategy.md` - Best practices for deprecation
3. [ ] `claude-research/glm-refactor.md` - How to best use nilearn's GLM functionality
4. [ ] `claude-research/masker-integration.md` - Supporting multiple masker types

## Notes

- Keep `uv` setup and all tooling improvements
- Focus only on v0.5.1 functionality for now
- Brain_Collection and Model are future work (Priority 3)
- When in doubt, check what nilearn already provides
- Maintain backwards compatibility through deprecation warnings

---

## 🎯 Next Focus Areas (Priority Order)

### Immediate (Fix Critical Test Failures)
1. **Fix `.regress()` test** - TypeError in test
   - Check design_matrix input handling
   - Verify FirstLevelModel integration
   - Test with both Design_Matrix and DataFrame inputs

2. **Fix `.extract_roi()` test** - ValueError
   - Test with binary masks
   - Test with labeled atlases
   - Verify NiftiLabelsMasker compatibility

3. **Investigate mysterious failures**:
   - `test_smooth` - assert 2 == 1 (dimension issue?)
   - `test_decompose` - shape mismatch
   - `test_similarity` - NotImplementedError (check if using removed method)
   - `test_bootstrap` - NotImplementedError

### Secondary (Clean Up Expected Failures)
4. **Update tests for deprecated methods** to expect NotImplementedError:
   - `test_ttest` ✓ (already raises correctly)
   - `test_randomise` ✓ (already raises correctly)
   - `test_predict` ✓ (already raises correctly)
   - `test_predict_multi` - needs AttributeError fix

5. **Fix legacy file loading**:
   - `test_load_legacy_h5` - Handle missing Y attribute

### Testing Strategy for Fixes
```bash
# Test single failing test with verbose output
uv run pytest nltools/tests/test_brain_data_old.py::test_regress -xvs

# After fix, run all brain_data tests
uv run pytest nltools/tests/test_brain_data_old.py -x

# Finally, run full suite
uv run pytest
```

### Expected After Fixes
- Target: 34/38 tests passing (89%)
- 4 tests correctly showing NotImplementedError for deprecated methods
- All core v0.5.1 functionality working