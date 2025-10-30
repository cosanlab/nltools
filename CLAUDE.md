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

**NEVER commit without explicit approval - only stage changes:**
```python
# Our workflow invariant:
if changes_ready:
    git add .
    say("Changes staged and ready for review")
    # WAIT FOR APPROVAL - DO NOT COMMIT

# Eshin responds: "Go ahead and commit" → Then commit
```

**ALWAYS update after making changes:**
- `MIGRATION_v0.5_to_v0.6.md` - Document API changes
- `REFACTORING_PLAN.md` - Update progress tracker

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

---

## 🎯 Current State (October 2025)

**Branch**: `uv-cleanup` (active development)
**Version Target**: v0.6.0 (breaking release, API changes allowed)
**Test Status**: 266 passing, 3 skipped ✅
**Last Work**: Cross-validation support for Brain_Data.fit() (commit 187c210)

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

### Running Tests
```bash
# Run all tests
uv run pytest nltools/tests/ -x

# Run with verbose output and stop on first failure
uv run pytest nltools/tests/test_brain_data_old.py -xvs

# Run last failed tests
uv run pytest --lf -x

# Run tests matching pattern
uv run pytest -k "regress or extract" -x

# Capture output to log file (recommended for debugging)
uv run pytest nltools/tests/ -xvs --tb=long 2>&1 | tee pytest_full.log
```

### Test Suite Organization

**Test files organized into subdirectories following "imperative shell, functional core" pattern:**

**Structure**:
```
nltools/tests/
├── conftest.py           # Shared fixtures
├── shell/                # Imperative shell (93 tests)
│   ├── test_brain_data.py        # 60 tests (includes CV tests)
│   ├── test_adjacency.py         # 30 tests
│   ├── test_design_matrix.py     # 10 tests
│   └── test_analysis.py          # 1 test
├── core/                 # Functional core (115 tests)
│   ├── test_backends.py          # 16 tests
│   ├── test_models.py            # 37 tests
│   ├── test_ridge.py             # 16 tests
│   ├── test_hyperalignment.py    # 27 tests
│   ├── test_stats.py             # 15 tests (1 skipped)
│   ├── test_utils.py, test_mask.py, etc.
├── support/              # Integration & utilities (31 tests)
│   ├── test_datasets.py          # 9 tests
│   ├── test_efficient_copy.py    # 14 tests (1 skipped)
│   ├── test_prefs.py             # 5 tests
│   └── test_simulator.py         # 3 tests
└── data/                 # Centralized test data (10 files)
```

**Total: 266 passing, 3 skipped**

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

**Never commit without explicit approval**:

1. Make changes
2. Run tests: `uv run pytest --lf`
3. Stage: `git add .`
4. Say: "Changes staged and ready for review"
5. **WAIT** for Eshin to approve
6. Only then: Commit with detailed message

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

**Detailed Guidelines** (`claude-guidelines/`):
- **`design-philosophy.md`**: Why we made key architectural decisions (nilearn integration, regress() design, efficient copying, deprecation strategy)
- **`knowledge-base.md`**: Technical patterns, testing workflows, research methodology, code quality standards
- **`refactoring-context.md`**: Project priorities, implicit context dictionary, workflow patterns, understanding architecture

**Active Documents**:
- **`REFACTORING_PLAN.md`**: Current tasks, progress tracker, next steps
- **`MIGRATION_v0.5_to_v0.6.md`**: User-facing upgrade guide

**When to reference what**:
- "Why did we implement X this way?" → `design-philosophy.md`
- "How should I test/code this?" → `knowledge-base.md`
- "Where did we leave off?" → `refactoring-context.md` + `git log -1`
- "What's the current priority?" → `refactoring-context.md`

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
2. Review `REFACTORING_PLAN.md` progress
3. Run `uv run pytest --lf` to see test status
4. Check `refactoring-context.md` for priorities
5. Summarize and suggest next steps

**When asked "Continue working on X"**:
1. Check `claude-guidelines/` for context on X
2. Review relevant docs (design-philosophy, knowledge-base)
3. Set up todo list with TodoWrite tool
4. Start with targeted tests

**When asked "Why did we..."**:
1. Check `design-philosophy.md` for decision rationale
2. Review git history for context
3. Explain reasoning and trade-offs
4. Offer alternatives if reconsidering

---

## 📝 Meta Notes

**Update this file when**:
- Critical requirements change
- Common commands evolve
- Workflow patterns improve

**Update reference files when**:
- Making significant design decisions → `design-philosophy.md`
- Discovering useful patterns → `knowledge-base.md`
- Priorities or context shifts → `refactoring-context.md`

**Our collaboration principle**: You (Eshin) provide vision and domain expertise. I provide implementation, research, and push back when appropriate. Together we build pragmatic, user-friendly neuroimaging tools.

---

*Last updated: 2025-10-29*
*Branch: uv-cleanup*
*Version target: v0.6.0*
*Lines: ~375 (comprehensive quick reference)*
