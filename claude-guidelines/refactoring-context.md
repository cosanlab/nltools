# nltools Refactoring Context & Meta-Patterns

*Understanding context, interpreting implicit requests, and navigating the refactoring process*

---

## Project Context & Priorities

### Current Refactoring Status (October 2025)

**Branch**: `uv-cleanup` (active development)
**Reference**: `uv-refactor` branch has historical changes we can learn from
**Version Target**: v0.6.0 (breaking release, API changes allowed)
**Test Status**: 38/38 passing (100%)

### Three-Tier Priority System

#### Priority 1: Core Refactoring ✅ ~90% Complete
**Goal**: Restore v0.5.1 functionality with cleaner architecture

**Completed**:
- ✅ Removed post-v0.5.1 features (Brain_Collection, Model class)
- ✅ Implemented nilearn-based methods (regress, extract_roi, smooth, etc.)
- ✅ Added deprecation stubs for methods moving to Model
- ✅ Fixed all major compatibility issues
- ✅ Implemented efficient copying (~80% performance improvement)
- ✅ All 38 tests passing

**What this means**: v0.5.1 users can upgrade to v0.6.0 with minimal breaking changes (mostly import paths and some method signatures).

#### Priority 2: Documentation (Next Phase)
**Goal**: Make v0.6.0 usable and understandable

**Upcoming work**:
- Migrate Sphinx → Jupyter Book
- Update tutorials for v0.6.0 API changes
- Write migration guide
- Update docstrings for new methods
- Create examples gallery

**What this means**: Users need clear documentation to understand what changed and how to migrate.

#### Priority 3: New Features (Future)
**Goal**: Add advanced functionality beyond v0.5.1

**Planned features**:
- Implement Model class (holds deprecated ML/stats methods)
- Implement Brain_Collection (multi-subject operations)
- Add GPU-accelerated operations
- Advanced ML workflows
- Enhanced visualization

**What this means**: We're not adding new features until the foundation is solid.

### Important Git Tags

- **`v0.6.0-test-refactor`** (2025-10-28): Reference for test implementations of deprecated methods (using pytest.raises)
- **`v0.6.0-docs-removal`** (2025-10-28): Reference for removed Sphinx documentation code

---

## Implicit Context Dictionary

*How to interpret common questions and phrases in this project*

### "Where did we leave off?"

**Check in this order:**
1. `git log -1` - Last commit message (comprehensive summaries)
2. refactor-todos.md - Task checklist
3. Run tests to see current status: `uv run pytest nltools/tests/test_brain_data_old.py --tb=no`
4. CLAUDE.md - "Current State" section

**What to report:**
- Last significant change
- Current test status
- Active focus area
- Suggested next steps

### "It would be easier for the user if..."

**Interpret as**: UX/API design suggestion

**Response pattern:**
1. Consider the user's workflow perspective
   - What are they trying to accomplish?
   - What boilerplate can we eliminate?
2. Check if nilearn has a relevant pattern to follow
   - nilearn is our UX inspiration
3. Propose API change with code examples
4. Document trade-offs (breaking changes, complexity, etc.)
5. Ask for approval if significant

**Example**:
> "It would be easier for the user if regress stored results as attributes"
→ Propose: `brain.glm_betas` instead of `results['beta']`
→ Trade-off: Breaking change, but cleaner API
→ Decision: Implement for v0.6.0

### "Why did we make this decision?"

**Check in this order:**
1. claude-guidelines/design-philosophy.md
2. Git commit messages (search: `git log --grep="keyword"`)
3. claude-guidelines/ folder for technical investigations
4. refactor.md for original requirements

**Response pattern:**
- Summarize the decision
- Explain the rationale
- Note trade-offs considered
- Offer to reconsider if context has changed

### "Can we simplify this?"

**Usually means one of:**
1. Can nilearn do this for us? (check nilearn docs)
2. Is there unnecessary complexity we can remove?
3. Can we reduce the API surface?
4. Can we reduce boilerplate for users?

**Response pattern:**
```
Yes, we can simplify by [specific approach].

Current approach: [code example]
Simplified approach: [code example]

Nilearn's [function] handles [specific part].
Trade-off: [what we lose, if anything]

Shall I proceed?
```

### "This test is failing"

**Standard debugging workflow:**
1. Run with maximum verbosity: `uv run pytest path::test -xvs --tb=long`
2. Check what changed: `git diff HEAD~1 path/to/file.py`
3. Look for patterns: `uv run pytest -k similar_pattern`
4. Check if it's expected (refactoring often breaks tests intentionally)
5. Fix or document why it's failing

**Don't assume it's a bug**: Sometimes tests fail because we changed the API intentionally.

### "Run the tests"

**Always use `uv run`**:
```bash
uv run pytest nltools/tests/test_brain_data_old.py -xvs
```

**Common patterns**:
- All tests: `uv run pytest nltools/tests/`
- Last failed: `uv run pytest --lf -x`
- Specific pattern: `uv run pytest -k regress -x`
- Stop on first failure: `uv run pytest -x`

---

## Starting a Fresh Session

*Guidelines for resuming work after a break*

### When Asked: "Where did we leave off?"

**Standard response template:**

1. **Last commit summary**:
   - What: [Brief description of last change]
   - When: [Commit date and hash]
   - Status: [Tests passing/failing count]

2. **Current state**:
   - Branch: uv-cleanup
   - Test status: [X/38 passing]
   - Active priority: [Priority 1/2/3]

3. **Next steps** (from refactor-todos.md):
   - [List 2-3 next focus areas]
   - [Any blockers or decisions needed]

4. **Recommendation**:
   - [Suggest specific next action]

### When Asked: "Continue working on X"

**Standard workflow:**

1. **Check context**:
   - Review CLAUDE.md for context on X
   - Check claude-guidelines/ for relevant docs
   - Review git history: `git log --grep="X"`

2. **Set up todo list**:
   - Break X into specific tasks
   - Use TodoWrite tool to track

3. **Start with tests**:
   - Identify relevant tests: `uv run pytest -k pattern --co`
   - Run them to see current status
   - Fix or implement as needed

4. **Update docs**:
   - refactor-todos.md progress
   - docs/migration-guide.md if API changes

### When Asked: "Why did we..."

**Standard response pattern:**

1. **Check decision log**:
   - claude-guidelines/design-philosophy.md
   - Git commit messages
   - This file (refactoring-context.md)

2. **Explain rationale**:
   - What: [The decision]
   - Why: [The reasoning]
   - Trade-offs: [What we gave up]
   - Alternatives: [What we considered]

3. **Offer reconsideration**:
   - "If context has changed or you disagree, I can suggest alternatives"
   - Present options if relevant

---

## Workflow Patterns

### Research Before Implementation

**Check these sources BEFORE coding**:
```python
research_sources = [
    "claude-guidelines/",    # Our guidelines & research docs
    "git log uv-refactor",   # Historical decisions
    "nilearn docs",          # Current APIs
    "pytest docs",           # Testing patterns
    "refactor-todos.md",   # Task checklist
    "refactor-progress.md",  # Session context
]
```

### Staging Protocol

**CRITICAL INVARIANT**: Never commit without explicit approval

```python
# Workflow:
if changes_ready:
    git_add_all()
    say("Changes staged and ready for review")
    # WAIT FOR APPROVAL - DO NOT COMMIT
else:
    dont_stage()
    say("Stuck on X because Y, not staged yet")

# Eshin will respond:
# - "Go ahead and commit" → Commit with detailed message
# - "Let me modify first" → Wait for changes
# - "Let's fix X before committing" → Continue working
```

### Documentation Updates

**After making changes, ALWAYS update**:
1. `docs/migration-guide.md` - If any API changes
2. `refactor-todos.md` - Mark tasks complete
3. `refactor-progress.md` - Document session learnings
3. Relevant `claude-guidelines/*.md` - If discovered new patterns

### Test-Driven Development Cycle

**Standard TDD cycle for nltools**:
1. **Identify test**: `uv run pytest -k pattern --co`
2. **Run failing test**: `uv run pytest path::test -xvs`
3. **Implement minimal fix**: Change just enough to pass
4. **Verify**: `uv run pytest --lf`
5. **Check for regressions**: `uv run pytest path/to/module`
6. **Update docs**: refactor-todos.md, refactor-progress.md, docs/migration-guide.md

---

## Understanding the Architecture

### Functional-Core, Imperative Shell

**Imperative Shell** (`nltools/data/`):
- Classes: Brain_Data, Adjacency, DesignMatrix
- Hold state (data, masks, metadata)
- Coordinate operations
- User-facing API

**Functional Core**:
- Modules: stats.py, utils.py, external/algorithms.py
- Pure functions (no state)
- Computation logic
- Tested independently

**Why this pattern?**:
- Easier to test (pure functions)
- Easier to understand (state isolated)
- Easier to maintain (logic separate from coordination)

### v0.5.1 as Baseline

**Critical constraint**: Everything from v0.5.1 must work or deprecate gracefully

**What this means**:
- Don't remove v0.5.1 features silently
- Deprecate explicitly with helpful messages
- Document breaking changes in MIGRATION guide
- Provide migration path for every breaking change

**Post-v0.5.1 features**: Can be removed/moved to Priority 3

---

## When to Reference This File

**Reference this file when:**
- Starting a fresh session ("Where did we leave off?")
- Unclear about project context or priorities
- Interpreting ambiguous requests
- Understanding workflow patterns
- Need to understand the refactoring roadmap

**Update this file when:**
- Priorities shift
- New workflow patterns emerge
- Common questions arise repeatedly
- Context changes significantly

---

*Last updated: 2025-10-28*
*For current commands, see CLAUDE.md*
*For design decisions, see design-philosophy.md*
*For technical patterns, see knowledge-base.md*
