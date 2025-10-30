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

# ✅ DO: Run tier1 for quick regression checks (fast!)
uv run pytest -m tier1 -n auto  # ~18s with parallelization!

# ✅ DO: Run by directory for regression checks
uv run pytest nltools/tests/shell/ -x
uv run pytest nltools/tests/core/ -x

# ⚠️  TIER 2: Run comprehensive tests before commits
uv run pytest -m tier2 -x  # Only when needed (~7 min)

# ❌ AVOID: Running full suite during rapid iteration
# Use tier1 for fast feedback, tier2 before commits
```

**CLEAN UP after tests:**
```bash
# Regularly delete stale artifacts to keep repo clean
rm -f *.log                    # Remove old log files
rm -f nltools/tests/*.log      # Remove test log files
rm -f *.csv *.nii.gz           # Remove test data artifacts (NOT in nltools/tests/data/!)
```

**When deploying SUB-AGENTS:**
- Always instruct them to use targeted TDD strategy
- Never have sub-agents run full test suite unless specifically required
- Remind them to use `uv run` prefix for all commands
- Tell them NOT to stage changes automatically - wait for instructions

---

## 🎯 Current State (October 2025)

**Branch**: `uv-cleanup` (active development)
**Version Target**: v0.6.0 (breaking release, API changes allowed)
**Test Status**: 385 tests (381 passing, 4 skipped) ✅
  - Tier 1 (Fast Core): ~350 tests, <2 min
  - Tier 2 (Comprehensive): ~35 tests, ~7 min
**Last Work**: Tiered testing implementation with pytest-xdist for parallel execution

**Important Git Tags**:
- `v0.6.0-test-refactor`: Test implementations for deprecated methods
- `v0.6.0-docs-removal`: Reference for removed Sphinx docs

**What We're Building**: Python neuroimaging library that wraps nilearn with intuitive APIs. Think "requests library for neuroimaging" - we don't reinvent, we simplify.

**Architecture**: "Functional-core, imperative shell"
- Imperative shell: `nltools/data/` (Brain_Data, Adjacency, Design_Matrix)
- Functional core: `stats.py`, `utils.py`, `external/algorithms.py`
- **v0.5.1 = baseline**: Must work or deprecate gracefully

---

## 📋 Quick Command Reference

**All commands must use `uv run` prefix**

### Running Tests (Tiered Strategy)

**Tier 1 (Fast Core)**: ~350 tests, <2 min - Run on every iteration
**Tier 2 (Comprehensive)**: ~35 tests, ~7 min - Run before commits

```bash
# TIER 1: Fast development loop (DEFAULT - runs automatically!)
uv run pytest  # Runs tier1 only due to default config

# TIER 1: Explicit (with parallelization for speed)
uv run pytest -m tier1 -n auto  # ~18s with 4 cores!

# TIER 2: Comprehensive tests (run before commits)
uv run pytest -m tier2 -x

# BOTH TIERS: Full suite (before releases)
uv run pytest -m "tier1 or tier2" -x
uv run pytest -m "tier1 or tier2" -n auto  # ~2 min with parallelization

# Run specific file or test
uv run pytest nltools/tests/shell/test_brain_data.py::TestBrainData::test_fit -xvs

# Run last failed tests
uv run pytest --lf -x

# Run tests matching pattern (respects tier filtering)
uv run pytest -k "regress or extract" -x

# Capture output to log file (recommended for debugging)
uv run pytest -m tier1 -xvs --tb=long 2>&1 | tee pytest_tier1.log
```

**When to run what**:
- **Every save**: `uv run pytest -m tier1 -n auto` (~18s)
- **Before commit**: `uv run pytest -m tier1 -n auto` (verify tier1) + affected tier2 tests
- **Before push**: `uv run pytest -m "tier1 or tier2" -n auto` (~2 min)
- **CI/Nightly**: Full suite with timing analysis

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

**Total: 317 tests (310+ passing, ~4 skipped)**

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

**Standard TDD workflow**:
```bash
# 1. Identify relevant tests
uv run pytest -k pattern --co

# 2. Run failing test with verbose output
uv run pytest path/to/test.py::test_name -xvs --tb=long

# 3. Implement minimal fix

# 4. Verify fix
uv run pytest --lf

# 5. Check for regressions
uv run pytest path/to/module/
```

### Capture Output to Log Files

**CRITICAL: Save pytest output to avoid token waste and time waste**

**Token impact:**
- Running pytest: 1,000-5,000 tokens per run
- Grep tool on log: ~50 tokens per search
- Read tool on log: ~200 tokens per section
- Example: Searching 5 patterns = 25,000 tokens (wasteful) vs. 5,250 tokens (efficient) = **80% token savings**

**Decision criteria:**

| Save to log file FIRST | Run directly (no log) |
|------------------------|------------------------|
| First diagnostic run | Quick fix verification (<50 lines expected) |
| Unknown failure scope | Interactive debugging (--pdb) |
| Need 2+ pattern searches | Single specific test (likely to pass) |
| Expected >500 lines output | Real-time interaction needed |
| Analyzing test suite patterns | |

**Efficient workflow:**
```bash
# STEP 1: Capture full output ONCE
uv run pytest nltools/tests/ -xvs --tb=long 2>&1 | tee pytest_full.log

# STEP 2: Analyze with Read/Grep TOOLS (not re-running pytest!)
# Use Grep tool to search patterns in pytest_full.log
# Use Read tool to view specific sections of pytest_full.log

# STEP 3: Make fixes based on analysis

# STEP 4: Verify with targeted re-run (much smaller output)
uv run pytest --lf -x

# STEP 5: If still failures, update log and repeat
uv run pytest --lf -xvs --tb=long 2>&1 | tee pytest_remaining.log
```

**Why this matters:**
- **Token efficiency**: 5x-10x reduction in token usage
- **Speed**: Grep is instant; re-running tests takes minutes
- **Completeness**: Preserves full stack traces for analysis

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
- **`polars-migration.md`**: Design_Matrix Polars migration strategy
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
- Use targeted TDD strategy (write test → run specific test → implement → verify)
- Never run full test suite during development (only before final commits)
- Clean up log files and test artifacts regularly
- Deploy sub-agents with explicit instructions about targeted TDD
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

*Last updated: 2025-10-29*
*Branch: uv-cleanup*
*Version target: v0.6.0*
*Test status: 317 tests (310+ passing, ~4 skipped)*
*Lines: ~420 (comprehensive quick reference with targeted testing guidelines)*
