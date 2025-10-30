# nltools v0.6.0 Refactoring Progress

**Purpose**: Session context, detailed decisions, and helpful information for continuing work.

For task checklist, see `refactor-todos.md`. For strategic vision, see `refactor-plan.md`.

---

## Current State (2025-10-29)

**Branch**: `uv-cleanup`
**Version Target**: v0.6.0 (breaking release)
**Test Status**: 317 tests (310+ passing, ~4 skipped)
**Last Work**: SRM/DetSRM comprehensive tests + API documentation improvements (commits f134854, 1a4b1d9)

---

## Recent Accomplishments

### Documentation & Organization (2025-10-29)
- Reorganized all planning docs into `claude-guidelines/` directory
- Created focused documentation structure:
  - `refactor-plan.md` - Strategic vision (stable)
  - `refactor-todos.md` - Task checklist (tactical)
  - `refactor-progress.md` - Session context (this file)
  - `docs/migration-guide.md` - User-facing guide
- Updated CLAUDE.md with targeted TDD workflows
- Emphasized: no auto-staging, targeted testing, cleanup practices

### SRM/DetSRM Testing (2025-10-29, commit f134854)
- **Achievement**: Eliminated zero-coverage gap for SRM/DetSRM algorithms
- **Approach**: Property-based tests with mathematical invariants (not golden outputs)
- **Tests**: 34 comprehensive tests covering initialization, contracts, math properties, edge cases
- **Key insight**: Orthogonality check should be W.T @ W ≈ I (column orthonormality for [voxels, features] matrices)
- **Research**: Based on BrainIAK, PyMVPA, Hypertools implementations

### API Documentation Improvements (2025-10-29, commit 1a4b1d9)
- Reorganized API docs by category (Data Objects, Models, Stats, Utilities)
- Removed usage examples from API reference (move to tutorials later)
- Clean, navigable structure in Jupyter Book

### Round 1 Audit Fixes (2025-10-28, commit ce3662d)
- Fixed 4 critical bugs identified in systematic audit
- Added tests for `filter()` and `compute_contrasts()`
- Created `minimal_brain_data` fixture for efficient testing

### Cross-Validation Support (commit 187c210)
- Added comprehensive CV support to `Brain_Data.fit()` for Ridge models
- Features: `cv=int`, `cv='auto'`, custom sklearn splitters
- Auto alpha selection with voxel-wise optimal alpha
- Out-of-fold predictions returned as Brain_Data
- 10 comprehensive tests

### HyperAlignment Class (Multiple commits)
- Extracted Procrustes-based hyperalignment into reusable class
- Sklearn-compatible API: `fit()`, `transform()`, `transform_subject()`
- Maintains backward compatibility via `align(method='procrustes')`
- 27 comprehensive tests

---

## Implementation Details

### Efficient Copying Performance
- Implemented `_shallow_copy_with_data()` helper
- Updated 10 methods to use efficient copying
- **Measured improvement**: ~80% faster for method chaining
- **Pattern**: Share immutable objects (mask, nifti_masker), copy only data arrays

### Property Conversions
- Converted `.shape()`, `.isempty()`, `.dtype()` to properties
- Updated ~90 calls across codebase
- Improves API discoverability and usage

### Test Suite Organization
- **Structure**: shell/ (131 tests), core/ (155 tests), support/ (31 tests)
- **Benefits**: Targeted test running, faster dev cycles, clear architectural separation
- **Pattern**: Shell tests object usage, core tests computational correctness

---

## Key Decisions & Rationale

### Why targeted TDD strategy?
- Full test suite takes minutes; specific tests take seconds
- Faster feedback loop improves development velocity
- Reduces token usage in AI-assisted development
- Matches proven workflow from successful sprints

### Why no auto-staging?
- Manual staging allows careful review before commits
- Prevents accidental inclusion of debug code
- Gives user control over commit granularity
- Reduces cognitive load (one less thing AI manages)

### Why property-based SRM tests instead of golden outputs?
- Golden outputs are brittle and platform-dependent
- Mathematical properties are the true contracts
- Easier to maintain and understand
- Matches best practices from BrainIAK, PyMVPA, Hypertools

### Why separate planning docs?
- `refactor-plan.md`: Strategic vision (stable, rarely changes)
- `refactor-todos.md`: Tactical checklist (changes as tasks complete)
- `refactor-progress.md`: Session context (updated with learnings)
- Clear separation of concerns improves navigation

---

## Workflow Patterns

### Starting a Fresh Session
1. Read `refactor-progress.md` (this file) for context
2. Check `refactor-todos.md` for what's next
3. Run `git log -5` to see recent commits
4. Run targeted tests if continuing specific work
5. Update this file with new learnings/decisions

### During Development
1. Use targeted TDD: write test → run specific test → implement → verify
2. Never run full test suite during development
3. Clean up log files and test artifacts regularly
4. Update `refactor-todos.md` as tasks complete
5. Do NOT stage changes automatically - wait for instructions

### Completing a Task
1. Run targeted tests to verify
2. Run related tests for regression checks
3. Say "Changes ready for review"
4. Wait for staging/commit instructions
5. Update `refactor-todos.md` with completion status
6. Update `refactor-progress.md` with learnings

### Deploying Sub-Agents
- Always instruct: use targeted TDD, no auto-staging
- Specify exact tests to run (never full suite)
- Remind to use `uv run` prefix
- Provide clear success criteria

---

## Helpful References

### Git Tags
- `v0.6.0-test-refactor`: Original test implementations for deprecated methods
- `v0.6.0-docs-removal`: Sphinx docs removal reference point

### Important Commits
- `1a4b1d9`: API documentation improvements
- `f134854`: SRM/DetSRM comprehensive tests (34 tests)
- `ce3662d`: Round 1 audit fixes (4 bugs)
- `c2a0929`: filter() and compute_contrasts() tests
- `187c210`: Cross-validation support
- `472259b`: fit/predict API implementation

### Planning Documents (claude-guidelines/)
- `bootstrap-refactor.md`: Comprehensive bootstrap refactoring plan (~14-18 hrs)
- `fastsrm-tdd-plan.md`: FastSRM implementation TDD plan
- `polars-migration.md`: Design_Matrix Polars migration strategy
- `banded-ridge-plan.md`: Banded ridge regression implementation
- `srm-hyperalignment-testing-strategy.md`: SRM testing research
- `design-philosophy.md`: Key architectural decisions and rationale
- `knowledge-base.md`: Technical patterns and best practices

---

## Current Blockers & Decisions

### Bootstrap Refactoring (Priority 2.8)
- **Status**: Planned, comprehensive design in `claude-guidelines/bootstrap-refactor.md`
- **Scope**: Memory-efficient online statistics, support for fitted models
- **Estimate**: 14-18 hours, 26 tests
- **Decision**: Proceed after completing audit and BrainData rename

### Brain_Data → BrainData Rename (Priority 2.9)
- **Status**: Planned
- **Scope**: Rename class, add deprecation alias, update docs
- **Estimate**: 2-3 hours
- **Decision**: Do after audit is complete (avoid merge conflicts)

### Round 1 Codebase Audit (Priority 2.11)
- **Status**: Partially complete (4 bugs fixed in ce3662d)
- **Next**: Systematic review of all classes and methods
- **Approach**: Can parallelize by module (Brain_Data, Adjacency, Design_Matrix, core)
- **Estimate**: 8-12 hours remaining

### Documentation & Tutorials (Priority 2.12)
- **Status**: API docs reorganized (1a4b1d9), tutorials need rewriting
- **Scope**: Complete migration guide, rewrite tutorials for v0.6.0 API
- **Standard**: Match pymer4 tutorial quality (https://eshinjolly.com/pymer4/)
- **Estimate**: 12-16 hours

---

## Next Steps (Priority Order)

1. **Complete migration guide refactoring** (current session)
   - Create tabular user-facing guide in `docs/migration-guide.md`
   - Update `docs/_toc.yml` to include migration guide
   - Update all references in CLAUDE.md

2. **Bootstrap refactoring** (Priority 2.8)
   - Follow TDD plan in `claude-guidelines/bootstrap-refactor.md`
   - 6 phases, 26 tests
   - Memory-efficient online statistics

3. **Continue codebase audit** (Priority 2.11)
   - Systematic review: docstrings, functionality, test coverage
   - Can parallelize by module
   - Fix bugs as discovered

4. **Brain_Data → BrainData rename** (Priority 2.9)
   - After audit to avoid conflicts
   - Add deprecation alias for backward compatibility

5. **Documentation & tutorials overhaul** (Priority 2.12)
   - Rewrite tutorials for v0.6.0 API
   - Complete migration guide
   - Match pymer4 quality standard

---

## Test Running Patterns

### Targeted Testing Examples
```bash
# Single specific test
uv run pytest nltools/tests/shell/test_brain_data.py::TestBrainData::test_fit -xvs

# Single test file
uv run pytest nltools/tests/core/test_srm.py -x

# Pattern-based
uv run pytest -k "ridge and cv" -x

# Directory (regression checks)
uv run pytest nltools/tests/shell/ -x

# Full suite (SPARINGLY - only before commits)
uv run pytest nltools/tests/ -x
```

### When to Run What
- **During development**: Single test or small subset only
- **After implementation**: Related tests for regression checks
- **Before staging**: Directory-level tests (shell/, core/, or support/)
- **Before committing**: Full suite to verify no regressions

---

## Cleanup Practices

### Regular Cleanup
```bash
# Delete stale log files
rm -f *.log audit/*.log nltools/tests/*.log

# Check for test artifacts (but NOT in nltools/tests/data/)
find . -name "*.csv" -o -name "*.nii.gz" | grep -v "nltools/tests/data"
```

### What to Keep
- `nltools/tests/data/*.h5` - Test fixtures (tracked)
- `nltools/tests/data/matplotlibrc` - Test config (tracked)
- Planning docs in `claude-guidelines/` (tracked)

### What to Delete
- `*.log` files (test output logs)
- `*.csv` and `*.nii.gz` outside `nltools/tests/data/` (test artifacts)
- Old planning docs in root (migrated to `claude-guidelines/`)

---

*This file should be updated each session with new learnings, decisions, and context. It's the "working memory" for the refactoring effort.*

*Last updated: 2025-10-29*
