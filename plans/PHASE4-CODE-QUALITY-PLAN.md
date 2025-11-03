# Phase 4: Code Quality & Bug Fixes Plan

**Goal**: Fix warnings and review test deselection to prepare for v0.6.0 release

**Estimated Effort**: 4-6 hours

**Priority**: HIGH (must complete before release)

**Status**: 📋 READY FOR SUB-AGENT

---

## Context

The codebase has several warnings that need to be addressed before release:
1. **Priority 1 (Breaking)**: int64→int32 conversions causing nilearn compatibility warnings
2. **Priority 2 (Future Compatibility)**: nilearn 0.12 compatibility warnings (~80+ occurrences)
3. **Priority 3 (Quality of Life)**: PyTables and sklearn warnings

Additionally, ~710 tests are deselected (mostly tier2 GPU tests). We need to verify these are intentional and document any that need fixing.

---

## Task 1: Fix int64→int32 Conversions (Priority 1)

### Location
- `nltools/data/brain_data.py:1599`
- `nltools/data/brain_data.py:2272`

### Problem
Nilearn warns that 64-bit integers are incompatible with FSL/SPM tools. These need to be converted to 32-bit integers.

### Solution
Change:
```python
ma.data = np.round(ma.data).astype(np.int64)
```

To:
```python
ma.data = np.round(ma.data).astype(np.int32)
```

### Steps
1. Read the file at both locations
2. Identify the exact context (what function/method)
3. Replace `np.int64` with `np.int32`
4. Run tests to verify no regressions:
   ```bash
   uv run pytest nltools/tests/shell/test_brain_data.py -xvs
   ```
5. Verify warnings are gone:
   ```bash
   uv run pytest nltools/tests/shell/test_brain_data.py -W default::UserWarning 2>&1 | grep -i "int64\|int32" || echo "No int64 warnings found"
   ```

### Success Criteria
- [ ] Both locations changed from `np.int64` to `np.int32`
- [ ] All tests still pass
- [ ] No int64 warnings in test output
- [ ] Code review confirms change is correct

---

## Task 2: Add Nilearn 0.12 Compatibility Layer (Priority 2)

### Location
- Multiple files using nilearn maskers (likely `brain_data.py`, `adjacency.py`, etc.)

### Problem
Nilearn 0.12+ warns: "3D images will be transformed to 1D arrays" (~80+ occurrences). We need to handle both nilearn <0.12 and >=0.12.

### Solution
1. Detect nilearn version
2. If >=0.12, ensure we're passing 1D arrays to maskers
3. If <0.12, keep existing behavior

### Steps
1. Find all nilearn masker usage:
   ```bash
   grep -r "NiftiMasker\|NiftiLabelsMasker\|NiftiMapsMasker\|NiftiSpheresMasker" nltools/ --include="*.py"
   ```
2. Check nilearn version:
   ```python
   import nilearn
   print(nilearn.__version__)
   ```
3. Create compatibility helper function in `nltools/utils.py`:
   ```python
   def _get_nilearn_version():
       """Get nilearn version as tuple (major, minor)."""
       import nilearn
       version_str = nilearn.__version__
       return tuple(int(x) for x in version_str.split('.')[:2])
   
   def _ensure_1d_array_for_nilearn(data, nilearn_version=None):
       """Ensure data is 1D for nilearn >= 0.12."""
       if nilearn_version is None:
           nilearn_version = _get_nilearn_version()
       
       if nilearn_version >= (0, 12):
           # Nilearn 0.12+ expects 1D arrays
           if data.ndim > 1:
               return data.flatten()
       return data
   ```
4. Update all masker usage sites to use compatibility helper
5. Test with both nilearn versions if possible:
   ```bash
   uv run pytest nltools/tests/ -k "mask\|brain_data" -W default::UserWarning 2>&1 | grep -i "3D images\|transformed" || echo "No nilearn 0.12 warnings"
   ```

### Success Criteria
- [ ] Compatibility helper function added to `utils.py`
- [ ] All masker usage sites updated
- [ ] No nilearn 0.12 warnings in test output
- [ ] All tests still pass
- [ ] Code review confirms backward compatibility maintained

---

## Task 3: Suppress PyTables Performance Warnings (Priority 3)

### Location
- `nltools/utils.py:44` (approximate)

### Problem
PyTables emits performance warnings that clutter test output.

### Solution
Use `warnings.catch_warnings()` context manager to suppress PyTables warnings.

### Steps
1. Find PyTables warning location:
   ```bash
   grep -r "PyTables\|pytables" nltools/utils.py -A 5 -B 5
   ```
2. Identify the exact line causing warnings
3. Wrap the code with:
   ```python
   import warnings
   with warnings.catch_warnings():
       warnings.filterwarnings('ignore', category=UserWarning, module='tables')
       # ... existing code that uses PyTables ...
   ```
4. Test that warnings are suppressed:
   ```bash
   uv run pytest nltools/tests/ -W default::UserWarning 2>&1 | grep -i "pytables\|tables" || echo "No PyTables warnings found"
   ```

### Success Criteria
- [ ] PyTables warnings suppressed with context manager
- [ ] No PyTables warnings in test output
- [ ] All tests still pass
- [ ] Code review confirms suppression is appropriate

---

## Task 4: Fix sklearn Numerical Stability Warnings (Priority 3)

### Problem
sklearn may emit numerical stability warnings for data with extreme values.

### Solution
Add robust scaling or data validation before sklearn operations.

### Steps
1. Find sklearn warnings:
   ```bash
   uv run pytest nltools/tests/ -W default::UserWarning 2>&1 | grep -i "sklearn\|numerical\|stability" | head -20
   ```
2. Identify which functions/methods trigger warnings
3. Add data validation or robust scaling:
   ```python
   from sklearn.preprocessing import RobustScaler
   # Or validate data ranges before sklearn operations
   ```
4. Test that warnings are resolved:
   ```bash
   uv run pytest nltools/tests/ -W default::UserWarning 2>&1 | grep -i "sklearn\|numerical" || echo "No sklearn warnings found"
   ```

### Success Criteria
- [ ] sklearn warnings identified and fixed
- [ ] No sklearn warnings in test output
- [ ] All tests still pass
- [ ] Data validation/scaling appropriate for use case

---

## Task 5: Review Test Deselection (~710 tests)

### Problem
~710 tests are deselected (mostly tier2 GPU tests). We need to verify these are intentional and document any that should be fixed.

### Steps
1. Get list of deselected tests:
   ```bash
   uv run pytest --co -q 2>&1 | grep -E "deselected|test session"
   ```
2. Categorize deselected tests:
   - Tier2 GPU tests (intentional - skip on systems without GPU)
   - Tier2 benchmark tests (intentional - slow)
   - Tests with `@pytest.mark.skip` (intentional - documented reasons)
   - Tests with `@pytest.mark.skipif` (conditional - verify conditions)
   - Other deselected tests (investigate why)
3. For each category:
   - Verify markers are correct (`@pytest.mark.tier2`, `@pytest.mark.skipif`, etc.)
   - Document reason for deselection in test docstring if not clear
   - Fix any incorrect deselections
4. Create summary document:
   - Total deselected: X
   - Tier2 GPU: Y (intentional)
   - Tier2 benchmarks: Z (intentional)
   - Other: N (need investigation/fix)
5. Fix any incorrect deselections:
   - Remove incorrect `@pytest.mark.skip` decorators
   - Fix `@pytest.mark.skipif` conditions if wrong
   - Add missing tier markers if needed

### Success Criteria
- [ ] All deselected tests categorized
- [ ] Summary document created with counts
- [ ] All incorrect deselections fixed
- [ ] Target: ≤10 deselected tests (excluding intentional tier2 GPU/benchmark tests)
- [ ] All tier1 tests run: `uv run pytest -m tier1 -n auto` (~36s runtime)

---

## Verification Steps

After completing all tasks, run full verification:

1. **Run linting**:
   ```bash
   uv run ruff check nltools/ --fix
   uv run ruff format nltools/
   ```

2. **Run tier1 tests**:
   ```bash
   uv run pytest -m tier1 -n auto --tb=short
   ```
   Expected: ~36s runtime, 303 passed, 6 skipped

3. **Check warnings**:
   ```bash
   uv run pytest -m tier1 -W default::UserWarning 2>&1 | grep -E "Warning|UserWarning" | head -20
   ```
   Expected: Minimal or no warnings

4. **Verify test deselection**:
   ```bash
   uv run pytest --co -q 2>&1 | tail -5
   ```
   Expected: ~309 tier1 tests collected, ~645 deselected (mostly tier2)

5. **Document results**:
   - Update `v0.6.0-VERIFICATION.md` with completed checkboxes
   - Note any warnings that couldn't be fixed (document reason)
   - Update test deselection counts

---

## Success Criteria for Phase 4

- [ ] All Priority 1 warnings fixed (int64→int32)
- [ ] All Priority 2 warnings fixed (nilearn compatibility)
- [ ] All Priority 3 warnings fixed or documented (PyTables, sklearn)
- [ ] Test deselection reviewed and documented
- [ ] ≤10 deselected tests (excluding intentional tier2 GPU/benchmark tests)
- [ ] All tier1 tests passing (~36s runtime)
- [ ] No new warnings introduced
- [ ] Linting passes (`ruff check` and `ruff format`)

---

## Notes

- **Testing**: Always run tests after each change to verify no regressions
- **Documentation**: If a warning can't be fixed, document why in code comments
- **Backward Compatibility**: Ensure nilearn compatibility layer maintains backward compatibility
- **Performance**: Verify warning fixes don't impact performance

---

## Reference Files

- `v0.6.0-VERIFICATION.md` - Full verification checklist
- `v0.6.0-ACTION-PLAN.md` - Overall action plan
- `TODO-AUDIT.md` - TODO audit (already complete)

---

**Last Updated**: 2025-01-03  
**Status**: Ready for sub-agent execution

