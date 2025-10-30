# CLAUDE.md - nltools Development Guide

*Quick reference for working on nltools. For detailed context, see `claude-guidelines/` directory.*

---

## ⚠️ CRITICAL REQUIREMENTS

**ALWAYS use `uv run` for ALL Python/pytest commands:**
```bash
✅ uv run pytest nltools/tests/
✅ uv run python script.py
❌ NEVER: pytest nltools/tests/  # Will use wrong environment
❌ NEVER: python script.py        # Will use wrong environment
```

**NEVER stage or commit without explicit approval:**
```python
# Our workflow invariant:
if changes_ready:
    say("Changes ready for review")
    # WAIT - Do NOT stage automatically!
    # Eshin will either:
    # 1. Stage manually, OR
    # 2. Say "stage the changes", THEN you run: git add .

# After staging, WAIT for commit approval
# Eshin says: "Go ahead and commit" → Then commit
```

**ALWAYS update after making changes:**
- `docs/migration-guide.md` - User-facing migration guide
- `refactor-todos.md` - Task checklist with progress
- `refactor-progress.md` - Session context and decisions

**Be TOKEN-EFFICIENT with pytest output:**
```bash
# ✅ FIRST RUN: Always capture to log file
uv run pytest nltools/tests/ -xvs --tb=long 2>&1 | tee pytest.log

# ✅ THEN: Use Read/Grep TOOLS on log file (cheap tokens)
# Each pytest run = 1,000-5,000 tokens
# Each Grep tool search = ~50 tokens
# Searching 5 patterns: 25,000 tokens (wasteful) vs 5,250 tokens (efficient)

# ❌ NEVER: Re-run pytest just to search for different patterns
```

**ALWAYS use PARALLEL testing and require PERMISSION for tier2:**
```bash
# ✅ ALWAYS: Default to -n auto for tier1 (6-7× faster)
uv run pytest -m tier1 -n auto 2>&1 | tee pytest_tier1.log

# ⚠️ TIER 2: NEVER run without explicit permission
# Always ask first: "Should I run tier2 tests (~7 min)?"
# Only after approval: uv run pytest -m tier2 -xvs --tb=long 2>&1 | tee pytest_tier2.log

# ❌ NEVER: Run tier2 without asking permission first
uv run pytest -m tier2  # DON'T DO THIS!

# Why this matters:
# - Parallel tier1: 18s vs 2min = 6-7× speedup
# - Log files: Run once, analyze many times = 10× token savings
# - Tier2 cost: ~7 min should be intentional, not automatic
```

**Use TARGETED TEST-DRIVEN DEVELOPMENT (TDD):**
```bash
# Our proven TDD strategy:
# 1. Write/identify the specific test first
# 2. Run ONLY that test (or small subset)
# 3. Implement minimal code to pass
# 4. Run same targeted test to verify
# 5. Run related tests for regressions
# 6. NEVER run full suite during development

# ✅ DO: Run targeted test subsets (fast, focused, TDD-friendly)
uv run pytest nltools/tests/shell/test_brain_data.py::TestBrainData::test_fit -xvs
uv run pytest nltools/tests/core/test_srm.py::test_srm_fit_transform -x
uv run pytest -k "ridge and cv" -x

# ✅ DO: Run tier1 for quick regression checks (ALWAYS use -n auto!)
uv run pytest -m tier1 -n auto  # ~18s (not 2min!)

# ✅ DO: Run by directory for regression checks (use -n auto when possible)
uv run pytest nltools/tests/shell/ -n auto -x
uv run pytest nltools/tests/core/ -n auto -x

# ⚠️  TIER 2: ASK PERMISSION FIRST, then run (~7 min)
# After getting approval:
uv run pytest -m tier2 -xvs --tb=long 2>&1 | tee pytest_tier2.log

# ❌ AVOID: Running full suite OR tier2 without permission during rapid iteration
# Use tier1 for fast feedback; ask about tier2 before running
```

**CLEAN UP after tests:**
```bash
# Regularly delete stale artifacts to keep repo clean
rm -f *.log                    # Remove old log files
rm -f nltools/tests/*.log      # Remove test log files
rm -f *.csv *.nii.gz           # Remove test data artifacts (NOT in nltools/tests/data/!)
```

**When deploying SUB-AGENTS:**
- Always instruct them to use `-n auto` for tier1 tests (parallel by default)
- Tell them tier2 requires explicit permission (must ask before running)
- Instruct them to create log files for any diagnostic work
- Always instruct them to use targeted TDD strategy
- Never have sub-agents run full test suite unless specifically required
- Remind them to use `uv run` prefix for all commands
- Tell them NOT to stage changes automatically - wait for instructions

---

## 🎯 Current State (October 2025)

**Branch**: `uv-cleanup` (active development)
**Version Target**: v0.6.0 (breaking release, API changes allowed)
**Test Status**: 385 tests (381 passing, 4 skipped) ✅
  - Tier 1 (Fast Core): ~350 tests, <2 min (18s with parallelization)
  - Tier 2 (Comprehensive): ~35 tests, ~7 min
  - **Parallel testing is SAFE**: pytest-xdist verified safe with fixtures, GPU, and random seeds
**Last Work**: Tiered testing implementation with pytest-xdist for parallel execution + safety verification

**Important Git Tags**:
- `v0.6.0-test-refactor`: Test implementations for deprecated methods
- `v0.6.0-docs-removal`: Reference for removed Sphinx docs

**What We're Building**: Python neuroimaging library that wraps nilearn with intuitive APIs. Think "requests library for neuroimaging" - we don't reinvent, we simplify.

**Architecture**: "Functional-core, imperative shell"
- Imperative shell: `nltools/data/` (Brain_Data, Adjacency, DesignMatrix)
- Functional core: `stats.py`, `utils.py`, `algorithms/` (ridge, srm, hyperalignment, inference)
- **New**: `algorithms/inference/` - GPU-accelerated permutation testing module
  - `one_sample.py` - One-sample permutation tests (CPU parallel + GPU batched)
  - `two_sample.py` - Two-sample permutation tests (CPU parallel + GPU batched)
  - `utils.py` - Shared helper functions
- **v0.5.1 = baseline**: Must work or deprecate gracefully

---

## 📋 Quick Command Reference

**All commands must use `uv run` prefix**

### Running Tests (Tiered Strategy)

**Tier 1 (Fast Core)**: ~350 tests, ~18s - Run on every iteration (ALWAYS use `-n auto`)
**Tier 2 (Comprehensive)**: ~35 tests, ~7 min - REQUIRES PERMISSION (ask first!)

```bash
# TIER 1: Fast development loop (DEFAULT - ALWAYS use -n auto!)
uv run pytest -m tier1 -n auto  # ~18s (not 2min!)

# TIER 1: With log file for diagnostic work (PREFERRED)
uv run pytest -m tier1 -n auto -xvs --tb=long 2>&1 | tee pytest_tier1.log

# TIER 2: ⚠️ ASK PERMISSION FIRST! (~7 min)
# After getting approval:
uv run pytest -m tier2 -xvs --tb=long 2>&1 | tee pytest_tier2.log

# BOTH TIERS: ⚠️ ONLY before releases with explicit permission
# After getting approval:
uv run pytest -m "tier1 or tier2" -n auto 2>&1 | tee pytest_full.log

# Run specific file or test (for targeted fixes)
uv run pytest nltools/tests/shell/test_brain_data.py::TestBrainData::test_fit -xvs

# Run last failed tests (quick verification after fix)
uv run pytest --lf -x

# Run tests matching pattern (respects tier filtering, use -n auto)
uv run pytest -k "regress or extract" -n auto -x
```

**When to run what**:
- **Every iteration**: `uv run pytest -m tier1 -n auto` (~18s)
- **Before commit**: `uv run pytest -m tier1 -n auto` → Then ASK about tier2
- **Before push**: ASK "Should I run tier2 or full suite?" → Only if approved
- **Tier2 tests**: NEVER run without asking permission first (~7 min cost)
- **CI/Nightly**: Full suite with timing analysis

**Parallel Testing Safety** ✅
Our test suite is verified safe for parallel execution with pytest-xdist:
- ✅ Fixtures properly isolated (function/module scopes work correctly)
- ✅ GPU/PyTorch backend creates fresh instances per test (no shared state)
- ✅ Random seeds in fixtures ensure reproducibility
- ⚠️ Action item: Audit file writes to use `tmp_path` fixture

**Detailed parallel testing documentation**: See `testing-strategy-analysis.md` section "Parallel Testing Safety & Correctness" and `claude-guidelines/knowledge-base.md` subsection "Parallel Testing with pytest-xdist"

### Test Suite Organization

**Test files organized into subdirectories following "imperative shell, functional core" pattern:**

**Structure**:
```
nltools/tests/
├── conftest.py           # Shared fixtures
├── shell/                # Imperative shell (131 tests)
│   ├── test_brain_data.py        # 71 tests (includes CV, fit/predict tests)
│   ├── test_adjacency.py         # 54 tests
│   ├── test_design_matrix.py     # 10 tests
├── core/                 # Functional core (155 tests)
│   ├── test_backends.py          # 16 tests
│   ├── test_models.py            # 37 tests
│   ├── test_ridge.py             # 16 tests
│   ├── test_hyperalignment.py    # 27 tests
│   ├── test_srm.py               # 34 tests (NEW!)
│   ├── test_stats.py             # 15 tests (1 skipped)
│   ├── test_cross_validation.py, test_utils.py, test_mask.py, etc.
├── support/              # Integration & utilities (31 tests)
│   ├── test_datasets.py          # 9 tests
│   ├── test_efficient_copy.py    # 14 tests
│   ├── test_prefs.py             # 5 tests
│   └── test_simulator.py         # 3 tests
└── data/                 # Centralized test data (h5, nii.gz files)
```

**Total: 385 tests (381 passing, 4 skipped)**

**Running tests by directory**:
```bash
# Run all imperative shell tests
uv run pytest nltools/tests/shell/ -v

# Run all functional core tests
uv run pytest nltools/tests/core/ -v

# Run all support/integration tests
uv run pytest nltools/tests/support/ -v

# Run specific test class
uv run pytest nltools/tests/shell/test_brain_data.py::TestBrainData::test_regress

# Run specific test function
uv run pytest nltools/tests/core/test_stats.py::test_isc
```

**Why this organization?**
- **shell/**: Tests object usage patterns and method interactions (class-based)
- **core/**: Tests computational correctness of pure functions (function-based)
- **support/**: Integration tests, performance tests, and utilities
- **data/**: Centralized test data following pytest best practices
- **Selective running**: Easy to run entire categories or specific tests
- **Clear separation**: Directory structure matches architectural patterns

### Git Workflow
```bash
# Check status and recent changes
git status
git log -1  # Last commit (detailed summary)
git log --grep="keyword"  # Search commits

# Stage changes (WAIT for approval before committing)
git add .
git status  # Verify staged changes

# Only after approval:
git commit -m "Detailed message"
```

### Debugging
```bash
# Search log files instead of re-running tests
grep -n "FAILED\|ERROR" pytest_full.log
grep -A10 -B5 "AttributeError" pytest_full.log

# Interactive debugging
uv run pytest path/to/test.py::test_name --pdb

# See print statements
uv run pytest path/to/test.py::test_name -s
```

---

## 🤝 Working Patterns

### Test-Driven Development Cycle

**Log-first TDD workflow** (preferred for diagnostic work):
```bash
# 1. Capture initial state to log file
uv run pytest -m tier1 -n auto -xvs --tb=long 2>&1 | tee pytest_initial.log

# 2. Identify failures with Grep/Read tools (cheap!)
# Use Grep tool to search pytest_initial.log for patterns

# 3. Implement minimal fix

# 4. Verify fix with targeted test (quick, no log needed)
uv run pytest path/to/test.py::test_name -x

# 5. Regression check with parallel tier1
uv run pytest -m tier1 -n auto

# 6. Update log only if more analysis needed
uv run pytest --lf -xvs --tb=long 2>&1 | tee pytest_updated.log
```

**Quick-fix TDD workflow** (for single obvious failures):
```bash
# 1. Run specific failing test
uv run pytest path/to/test.py::test_name -xvs

# 2. Implement fix

# 3. Verify
uv run pytest --lf -x

# 4. Regression check
uv run pytest -m tier1 -n auto
```

### Primary Workflow: Log Files First

**DEFAULT PATTERN: Always create log files for any diagnostic work**

**Why logs are primary, not optional:**
- **Token efficiency**: 1 test run (5K tokens) + 10 searches (500 tokens) vs 10 test runs (50K tokens) = **90% savings**
- **Speed**: Grep is instant; re-running tests takes 18s-7min
- **Completeness**: Preserves full context for analysis
- **Cost**: Test runs are expensive; log analysis is cheap

**Standard workflow:**
```bash
# STEP 1: Create log file on first run (ALWAYS for diagnostics)
uv run pytest -m tier1 -n auto -xvs --tb=long 2>&1 | tee pytest.log

# STEP 2: Analyze with Grep/Read tools (NOT re-running tests!)
# Use Grep tool: pattern matching in pytest.log (~50 tokens/search)
# Use Read tool: view sections of pytest.log (~200 tokens)

# STEP 3: Make fixes based on log analysis

# STEP 4: Quick verification (targeted, no log needed)
uv run pytest --lf -x

# STEP 5: Only if more analysis needed, update log
uv run pytest --lf -xvs --tb=long 2>&1 | tee pytest_updated.log
```

**When to skip logs** (rare exceptions):
- Single test verification after fix (expect pass, <50 lines output)
- Interactive debugging with --pdb
- Real-time test development (watching output evolve)

**When to ALWAYS use logs** (default):
- First diagnostic run (don't know what will fail)
- Any run expecting >100 lines output
- Any situation where you might search multiple patterns
- Tier2 test runs (always capture these!)

### The Staging Protocol

**Never stage or commit without explicit approval**:

1. Make changes
2. Run targeted tests: `uv run pytest <specific-test> -x`
3. Say: "Changes ready for review"
4. **WAIT** - Eshin will stage manually or say "stage the changes"
5. **WAIT** for Eshin to approve commit
6. Only then: Commit with detailed message

**Note**: Eshin manages staging manually. Do NOT run `git add` automatically.

### Research Before Implementation

**Check these sources BEFORE coding**:
```bash
# 1. Guidelines & research archive
ls claude-guidelines/
grep -r "keyword" claude-guidelines/

# 2. Git history
git log --grep="keyword"
git log uv-refactor  # Reference branch

# 3. Check nilearn docs (we wrap nilearn, don't reimplement)
# Use context7 MCP for current APIs
```

---

## 📚 Knowledge Base & Guidelines

**Quick Reference** (this file):
- Critical requirements (above)
- Common commands
- Working patterns
- Quick debugging tips

**Research & Planning Docs** (`claude-guidelines/`):
- **`design-philosophy.md`**: Why we made key architectural decisions (nilearn integration, regress() design, efficient copying, deprecation strategy)
- **`knowledge-base.md`**: Technical patterns, testing workflows, research methodology, code quality standards
- **`refactoring-context.md`**: Project priorities, implicit context dictionary, workflow patterns, understanding architecture
- **`bootstrap-refactor.md`**: Comprehensive bootstrap refactoring plan
- **`fastsrm-tdd-plan.md`**: FastSRM implementation TDD plan
- **`polars-migration.md`**: DesignMatrix Polars migration strategy
- **`banded-ridge-plan.md`**: Banded ridge regression implementation
- **`srm-hyperalignment-testing-strategy.md`**: SRM/hyperalignment testing research
- **`hypertools-hyperalignment-research.md`**: Hypertools implementation analysis
- **`pymvpa-hyperalignment-research.md`**: PyMVPA implementation analysis

**Active Documents** (root):
- **`refactor-plan.md`**: Strategic vision (stable, rarely changes)
- **`refactor-todos.md`**: Task checklist with progress (updated as tasks complete)
- **`refactor-progress.md`**: Session context and decisions (updated each session)
- **`docs/migration-guide.md`**: User-facing upgrade guide (updated as API changes)

**When to reference what**:
- "Why did we implement X this way?" → `claude-guidelines/design-philosophy.md`
- "How should I test/code this?" → `claude-guidelines/knowledge-base.md`
- "Where did we leave off?" → `refactor-progress.md` + `git log -1`
- "What's the current priority?" → `refactor-todos.md`
- "How to implement X?" → Check `claude-guidelines/` for implementation plans
- "What's the overall plan?" → `refactor-plan.md`

---

## 🔍 Quick Debugging Reference

### "Why is this test failing?"
```bash
# 1. Run with maximum verbosity (capture to log)
uv run pytest path::test -xvs --tb=long 2>&1 | tee test_debug.log

# 2. Check what changed
git diff HEAD~1 path/to/file.py

# 3. Look for patterns
uv run pytest -k similar_pattern -x
```

### "Is this in nilearn?"
```python
# Check nilearn modules
from nilearn import maskers, image, glm, plotting
dir(module)  # List available functions

# Search our guidelines docs
grep -r "function_name" claude-guidelines/
```

### "What did we decide about X?"
```bash
# 1. Check design philosophy
cat claude-guidelines/design-philosophy.md

# 2. Search commit messages
git log --grep="keyword"

# 3. Search code changes
git log -p -S "code_pattern"
```

---

## 🚀 Starting a Fresh Session

**When asked "Where did we leave off?"**:
1. Check `git log -1` for last commit
2. Review `refactor-progress.md` for recent context
3. Check `refactor-todos.md` for next tasks
4. Run `uv run pytest --lf` to see test status (if relevant)
5. Summarize and suggest next steps

**When asked "Continue working on X"**:
1. Check `claude-guidelines/` for existing implementation plans
2. Review relevant docs (design-philosophy, knowledge-base, specific plans)
3. Set up todo list with TodoWrite tool
4. Start with targeted tests (NOT full suite!)

**When asked "Why did we..."**:
1. Check `claude-guidelines/design-philosophy.md` for decision rationale
2. Review git history for context
3. Explain reasoning and trade-offs
4. Offer alternatives if reconsidering

**Best Practices for Every Session**:
- Start by reading `refactor-progress.md` for context
- ALWAYS use `-n auto` for tier1 tests (parallel by default, 6-7× faster)
- Create log files for diagnostic work (log-first, not log-as-optimization)
- ASK permission before running tier2 tests (~7 min cost, be intentional)
- Use targeted TDD strategy (write test → run specific test → implement → verify)
- Never run full test suite during development (tier1 only, unless approved)
- Clean up log files and test artifacts regularly
- Deploy sub-agents with explicit instructions: `-n auto`, tier2 permission, log-first
- Do NOT stage changes automatically - wait for explicit instructions
- Update `refactor-todos.md` as tasks complete
- Update `refactor-progress.md` with learnings and decisions
- Update `docs/migration-guide.md` if API changes

---

## 📝 Meta Notes

**Update this file when**:
- Critical requirements change
- Common commands evolve
- Workflow patterns improve

**Update reference files when**:
- Making significant design decisions → `claude-guidelines/design-philosophy.md`
- Discovering useful patterns → `claude-guidelines/knowledge-base.md`
- Priorities or context shifts → `claude-guidelines/refactoring-context.md`
- Planning new features → Create new plan in `claude-guidelines/`
- Completing tasks → `refactor-todos.md` (mark complete)
- Session learnings → `refactor-progress.md` (add context/decisions)
- API changes → `docs/migration-guide.md` (user-facing guide)

**File organization**:
- `CLAUDE.md` (this file): Quick reference and critical workflows
- `claude-guidelines/`: All research, planning docs, and design decisions
- `refactor-plan.md`: Strategic vision (stable)
- `refactor-todos.md`: Task checklist (tactical)
- `refactor-progress.md`: Session context (working memory)
- `docs/migration-guide.md`: User-facing upgrade guide

**Our collaboration principle**: You (Eshin) provide vision and domain expertise. I provide implementation, research, and push back when appropriate. Together we build pragmatic, user-friendly neuroimaging tools.

---

*Last updated: 2025-10-30*
*Branch: uv-cleanup*
*Version target: v0.6.0*
*Test status: 385 tests (381 passing, 4 skipped)*
*Testing defaults: Parallel tier1 (~18s), tier2 requires permission (~7 min)*
*Workflow: Log-first pattern (primary), -n auto (always), ask before tier2*
*Lines: ~530 (comprehensive quick reference + enforced parallel/log-first/permission patterns)*
