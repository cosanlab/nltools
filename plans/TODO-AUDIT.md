# nltools TODO Audit & Categorization
**Created**: 2025-11-02  
**Last Updated**: 2025-11-02  
**Status**: Complete audit of all remaining TODOs in codebase

---

## 📊 Summary

**Total TODOs Found**: 17  
**Location**: `nltools/stats.py` (9), `nltools/data/brain_data.py` (8)

**Categories**:
- **Efficiency Improvements**: 13 TODOs (can defer, not blockers)
- **Future Migration**: 2 TODOs (refactor to use new Model class)
- **Refactoring Decisions**: 2 TODOs (need investigation)

---

## 📋 Detailed TODO Audit

### `nltools/stats.py` (9 TODOs)

#### Category: Efficiency Improvements (7 TODOs)

**DEFER** - Not blockers, optimization opportunities:

1. **Line 307**: `# TODO: see related comment on _transform_outliers`
   - **Function**: `winsorize()`
   - **Context**: Related to `_transform_outliers()` efficiency
   - **Status**: ✅ Documented below
   - **Action**: Defer - low priority optimization

2. **Line 326**: `# TODO: see related comment on _transform_outliers`
   - **Function**: `trim()`
   - **Context**: Related to `_transform_outliers()` efficiency
   - **Status**: ✅ Documented below
   - **Action**: Defer - low priority optimization

3. **Line 340**: `# TODO: do we need this? can we refactor the function it supports to be more efficient?`
   - **Function**: `_transform_outliers()`
   - **Context**: Internal helper for `winsorize()` and `trim()`
   - **Status**: ✅ Function is used, keep but optimize if needed
   - **Action**: Defer - functional, optimization not critical

4. **Line 432**: `# TODO: ensure efficient`
   - **Function**: `downsample()`
   - **Context**: Time-series downsampling with pandas groupby
   - **Status**: ✅ Functional, may have performance improvements
   - **Action**: Defer - profile if performance issues reported

5. **Line 476**: `# TODO: ensure efficient`
   - **Function**: `upsample()`
   - **Context**: Time-series upsampling with interpolation
   - **Status**: ✅ Functional, may have performance improvements
   - **Action**: Defer - profile if performance issues reported

6. **Line 833**: `# TODO: make efficient`
   - **Function**: `make_cosine_basis()`
   - **Context**: DCT basis function generation
   - **Status**: ✅ Functional, may have vectorization opportunities
   - **Action**: Defer - profile if performance issues reported

7. **Line 889**: `# TODO: make efficient`
   - **Function**: `transform_pairwise()`
   - **Context**: Ranking transformation with itertools.combinations
   - **Status**: ✅ Functional, may have vectorization opportunities
   - **Action**: Defer - profile if performance issues reported

8. **Line 1229**: `# TODO: too slow needs to be made more efficient`
   - **Function**: `find_spikes()`
   - **Context**: Spike detection in fMRI data
   - **Status**: ⚠️ Performance concern noted
   - **Action**: Defer - monitor user feedback, optimize if needed

9. **Line 1764**: `# TODO: improve to avoid pandas type conversion use numpy or polars instead`
   - **Function**: `isps()`
   - **Context**: Intersubject phase synchrony computation
   - **Status**: ✅ Clear improvement path identified
   - **Action**: **MEDIUM PRIORITY** - Good optimization candidate (Tier 4 task)

---

### `nltools/data/brain_data.py` (8 TODOs)

#### Category: Efficiency Improvements (6 TODOs)

**DEFER** - Not blockers, optimization opportunities:

1. **Line 1373**: `# TODO: ensure this implementation is efficient as possible`
   - **Function**: `similarity()`
   - **Context**: Image similarity calculation
   - **Status**: ✅ Functional
   - **Action**: Defer - profile if performance issues reported

2. **Line 1416**: `# TODO: make this more efficient using scipy.spatial.distance.cdist`
   - **Function**: `distance()`
   - **Context**: Distance calculation between images
   - **Status**: ✅ Clear improvement path identified
   - **Action**: **MEDIUM PRIORITY** - Good optimization candidate (Tier 4 task)

3. **Line 1434**: `# TODO: ensure this implementation is efficient as possible`
   - **Function**: `multivariate_similarity()`
   - **Context**: OLS-based pattern expression
   - **Status**: ✅ Functional
   - **Action**: Defer - profile if performance issues reported

4. **Line 1681**: `# TODO: refactor to be more efficient`
   - **Function**: `icc()`
   - **Context**: Intraclass correlation coefficient
   - **Status**: ✅ Functional
   - **Action**: Defer - profile if performance issues reported

5. **Line 2611**: `# TODO: does our function offer something beyond what's in nilearn? If so can we make it more efficient?`
   - **Function**: `find_spikes()`
   - **Context**: Spike detection wrapper
   - **Status**: ✅ Functionality verified (uses stats.py version)
   - **Action**: Defer - already wraps stats.py, which is functional

6. **Line 2667**: `# TODO: refactor for efficiency and verify`
   - **Function**: `predict()`
   - **Context**: Model prediction method
   - **Status**: ✅ Functional, may have optimization opportunities
   - **Action**: Defer - profile if performance issues reported

#### Category: Future Migration (2 TODOs)

**REFACTOR** - Methods should be migrated to new Model class:

7. **Line 2769**: `# TODO: refactor to make use of new Model class`
   - **Function**: `randomise()`
   - **Context**: Permutation-based inference method
   - **Status**: ⚠️ Currently deprecated (raises NotImplementedError)
   - **Current State**: Method raises NotImplementedError directing users to Model class
   - **Future Plan**: Refactor to use new Model class for permutation-based inference
   - **Action**: **DEFER** - Wait for Model class implementation (post-v0.6.0)
   - **Note**: Method is properly deprecated with clear error message. TODO tracks future refactoring.

8. **Line 2776**: `# TODO: refactor to make use of new Model class`
   - **Function**: `ttest()`
   - **Context**: Statistical testing method
   - **Status**: ⚠️ Currently deprecated (raises NotImplementedError)
   - **Current State**: Method raises NotImplementedError directing users to Model class
   - **Future Plan**: Refactor to use new Model class for statistical testing
   - **Action**: **DEFER** - Wait for Model class implementation (post-v0.6.0)
   - **Note**: Method is properly deprecated with clear error message. TODO tracks future refactoring.

---

## 🎯 Recommended Actions

### Immediate (Tier 1 - Easy Wins)
1. ✅ **No immediate actions needed** - All TODOs are properly categorized

### Medium Priority (Tier 4 - Optimization Tasks)
2. ✅ **Optimize `isps()`** - Remove pandas conversion (Line 1764 in stats.py)
   - ✅ **COMPLETED** - Changed `pd.DataFrame(data)` to `np.asarray(data)`
   - Removed unnecessary pandas conversion overhead
   - Test verified: `test_isps()` passes

3. ✅ **Optimize `distance()`** - Use scipy.spatial.distance.cdist (Line 1416 in brain_data.py)
   - ✅ **COMPLETED** - Replaced `pairwise_distances` with direct `cdist` call
   - More efficient for pairwise distance computation
   - Test verified: `test_distance()` passes

### Future Migration (Post-v0.6.0)
4. 🔄 **Refactor `randomise()` and `ttest()`** (Lines 2769, 2776 in brain_data.py)
   - Methods currently deprecated with clear error messages
   - TODO tracks future refactoring to use new Model class
   - **Action**: Defer until Model class for statistical inference is implemented
   - **Note**: Methods are properly deprecated - TODO is for future enhancement

### Defer (Tier 5 - Future Optimization)
5. 📊 **Profile remaining efficiency TODOs** (13 items)
   - Only optimize if performance issues reported
   - Can be addressed in v0.6.1+ based on user feedback

---

## 📝 Notes

- **None of these TODOs are blockers** for v0.6.0 release
- All functions are **functional** - TODOs are optimization opportunities
- **2 deprecated methods** (`randomise()`, `ttest()`) properly raise NotImplementedError
  - TODOs track future refactoring to use new Model class (post-v0.6.0)
  - Methods are correctly deprecated with clear migration messages
- **2 clear optimization paths** identified for future work (`isps()`, `distance()`)
- **13 efficiency TODOs** can be deferred until user feedback indicates need

---

## 🔄 Maintenance

**After v0.6.0 Release**:
- Monitor user feedback for performance issues
- Prioritize optimization based on actual usage patterns
- Consider profiling the "too slow" TODO (find_spikes line 1229)

**Next Review**: After v0.6.0 release, or if performance issues reported

---

**Last Updated**: 2025-11-02 (Updated after user review - randomise/ttest TODOs clarified; medium priority optimizations completed)

