# CLAUDE.md — nltools Development Guide
*Last updated: 2025-12-19*

**Note**: This project uses Linear for issue tracking.
Use Linear issues instead of markdown TODOs or local tracking files.
See @AGENTS.md for workflow details.

## Critical Rule: Always use the `uv` environment
All test, lint, and run commands must use the `uv` prefix (already enforced globally).  
Example:
```bash
✅ uv run pytest nltools/tests/
✅ uv run python script.py
❌ NEVER: pytest nltools/tests/  # Will use wrong environment
❌ NEVER: python script.py        # Will use wrong environment
```

## Goal
Build and refactor `nltools`, a neuroimaging library with a delightful and intuitive user-facing API that prefers "composition over abstraction" by bringing together the power of libraries like `nilearn` with custom efficient, parallelizable, gpu-enabled algorithms that facilitate a wide-range of basic and advanced neuroimaging analyses. Advanced analyses made easy.  

## ⚙️ Environment & Project Context
- **Active branch:** `uv-cleanup`
- **Version target:** `v0.6.0` (breaking release; API updates allowed)
- **Architecture:** *Functional core, imperative shell*
  - Imperative: `nltools/data/` (`BrainData`, `Adjacency`, `DesignMatrix`)
  - Functional: `stats.py`, `utils.py`, `algorithms/` (`ridge`, `srm`, `hyperalignment`, `inference`)
- **Research & background**: `claude-research/` (use sub-agents to add new research here)
- **Task tracking**: Use Linear for all issue/task management
  - **Find work**: Review the `nltools` Linear project backlog and active issues
  - **Create issues**: Open Linear issues for follow-up work, bugs, and release tasks
  - **Update/close**: Move issues through Linear statuses as work progresses
  - **Reference**: `plans/README.md` for the reconstructed backlog and release-plan notes


## SUB-AGENT usage protocol
- Always instruct them to use targeted TDD strategy
- Never have sub-agents run full test suite unless specifically required
- Tell them slow tests require explicit permission (must ask before running)
- Remind them to use `uv run` prefix for all commands
- Always instruct them to use `-n auto` for default tests (parallel by default)
- Instruct them to create and use log files for any diagnostic work

## Testing Guidance

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

### Testing Protocols

**ALWAYS run ruff BEFORE running tests:**
```bash
# ✅ ALWAYS: Check and format with ruff BEFORE running tests
# Catches linting issues (unused imports, formatting, etc.) in <1 second
# Much faster than discovering issues via test failures

# Step 1: Check for issues
uv run ruff check nltools/

# Step 2: Auto-fix what can be fixed
uv run ruff check --fix nltools/

# Step 3: Format code
uv run ruff format nltools/

# ✅ BEST PRACTICE: Run all three in sequence before testing
uv run ruff check --fix nltools/ && uv run ruff format nltools/

# Why this matters:
# - Ruff runs in <1s vs pytest runs 18s-7min
# - Catches unused imports, formatting, style issues immediately
# - Prevents wasted test runs due to trivial linting failures
# - Ensures consistent code style before review
```

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

**Pytest Markers (simplified):**
```bash
# Markers:
# - @pytest.mark.slow  → Tests taking >3s (skipped by default)
# - @pytest.mark.gpu   → Tests requiring CUDA hardware
# - No marker          → Runs by default

# ✅ DEFAULT: Runs all non-slow tests (~881 tests)
uv run pytest -n auto 2>&1 | tee pytest.log

# ✅ SLOW TESTS: Require explicit permission (~197 tests, ~7 min)
# Always ask first: "Should I run slow tests (~7 min)?"
uv run pytest -m slow -xvs --tb=long 2>&1 | tee pytest_slow.log

# ✅ ALL TESTS: Run everything including slow
uv run pytest -m "" -n auto 2>&1 | tee pytest_all.log

# ✅ GPU TESTS: Tests requiring CUDA (~30 tests)
uv run pytest -m gpu -xvs 2>&1 | tee pytest_gpu.log

# ❌ NEVER: Run slow tests without asking permission first
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

# ✅ DO: Run default tests for quick regression checks (ALWAYS use -n auto!)
uv run pytest -n auto  # ~881 tests, skips slow

# ✅ DO: Run by directory for regression checks
uv run pytest nltools/tests/shell/ -n auto -x
uv run pytest nltools/tests/core/ -n auto -x

# ⚠️ SLOW TESTS: ASK PERMISSION FIRST (~7 min)
uv run pytest -m slow -xvs --tb=long 2>&1 | tee pytest_slow.log

# ❌ AVOID: Running slow tests without permission during rapid iteration
```

**CLEAN UP after tests:**
```bash
# Regularly delete stale artifacts to keep repo clean
rm -f *.log                    # Remove old log files
rm -f nltools/tests/*.log      # Remove test log files
rm -f *.csv *.nii.gz           # Remove test data artifacts (NOT in nltools/tests/data/!)
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create Linear issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished Linear issues and update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
