# CLAUDE.md - nltools Development Knowledge Base

*This document captures the shared context, decisions, and working patterns for the nltools library. It enables efficient collaboration by documenting both what we're building and how we work together.*

---

## 🎯 Current State & Context (October 2025)

### Where We Are
- **Branch**: `uv-cleanup` (active development)
- **Reference**: `uv-refactor` branch has historical changes we can learn from
- **Version Target**: v0.6.0 (breaking release, API changes allowed)
- **Test Status**: 38/38 passing (100%) - all tests now pass properly
- **Last Work**: Refactored tests to use pytest.raises for deprecated methods

### Important Git Tags
- **`v0.6.0-test-refactor`** (2025-10-28): Marks where we simplified test code to properly test deprecated methods with pytest.raises. Reference this tag if you need to see the test implementations we removed for methods that will move to Model class.

### What We're Building
**nltools** is a Python neuroimaging library that makes fMRI analysis more accessible by wrapping lower-level tools (primarily nilearn) with intuitive APIs. Think of it as "the requests library for neuroimaging" - we don't reinvent the wheel, we make the wheel easier to use.

### Core Architecture Decision
**"Functional-core, imperative shell"** pattern:
- **Imperative shell** (`nltools/data/`): Stateful classes (Brain_Data, Adjacency, Design_Matrix) that hold data and coordinate operations
- **Functional core**: Pure functions for computations (stats.py, utils.py, etc.)
- **v0.5.1 = Baseline**: This is our compatibility target - everything from v0.5.1 must work or deprecate gracefully
- **Post-v0.5.1 features**: Deferred to Priority 3 (Model class, Brain_Collection)

---

## 🧠 Key Design Decisions & Rationale

### Why We Implemented `.regress()` This Way

**Context**: "Why did you implement regress in this way?"

We chose to wrap `nilearn.glm.first_level.FirstLevelModel` because:
1. **Don't reinvent**: nilearn already has robust, tested GLM implementation
2. **Store as attributes**: Changed from returning dict to storing `.glm_betas`, `.glm_t`, etc. as attributes for easier access and consistency
3. **Override defaults**: We disable smoothing/scaling/drift (user should control these explicitly)
4. **Design_Matrix required**: Forces explicit experimental design specification

```python
# Old pattern (v0.5.1)
brain.X = design_matrix
results_dict = brain.regress()
betas = results_dict['beta']

# New pattern (v0.6.0)
brain.regress(design_matrix)  # Stores results as attributes
betas = brain.glm_betas  # Direct attribute access
```

**Trade-offs**:
- ✅ Cleaner API, leverages nilearn
- ❌ Breaking change from v0.5.1
- Decision: Worth it for long-term maintainability

### The nilearn Integration Philosophy

**Context**: "Can we do this more easily in nilearn?"

**ALWAYS check nilearn first**. Our integration rules:
1. If nilearn has it → wrap it for better UX
2. If nilearn doesn't have it → consider if we really need it
3. If we must implement → follow nilearn patterns for consistency

**Current nilearn dependencies we leverage**:
- `NiftiMasker`: Core data loading and masking
- `FirstLevelModel`: GLM implementation
- `NiftiLabelsMasker`: ROI extraction (new in our refactor)
- Image functions: `smooth_img`, `resample_to_img`
- Plotting: Most visualization functions

**When to push back**: If I suggest reimplementing something nilearn provides, challenge me. The correct response is to find the nilearn function and wrap it.

### The Deprecation Strategy

**Context**: Methods moved to future Model class

We deprecated rather than removed these methods:
- `.predict()` → Model class (ML workflows)
- `.ttest()` → Model class (statistical testing)
- `.randomise()` → Model class (permutation testing)
- `.predict_multi()` → Model class (searchlight/multi-ROI)

They now raise `NotImplementedError` with a message pointing to the Model class. This:
1. Prevents silent failures
2. Gives clear migration path
3. Documents what Model class needs to implement

---

## 💭 Implicit Context Dictionary

### "Where did we leave off?"
Check in order:
1. `git log -1` - Last commit message has comprehensive summary
2. REFACTORING_PLAN.md - "Next Focus Areas" section
3. Test failures: `uv run pytest nltools/tests/test_brain_data_old.py --tb=no | grep FAILED`
4. This file's "Current State" section

### "It would be easier for the user if..."
I should interpret this as a UX/API design suggestion and:
1. Consider the user's workflow perspective
2. Propose API changes that reduce boilerplate
3. Check if nilearn has a relevant pattern to follow
4. Document trade-offs of the easier approach

### "Why did we make this decision?"
Check:
1. This file's "Key Design Decisions" section
2. Git commit messages (very detailed)
3. `claude-research/` folder for technical investigations
4. `refactor.md` for original requirements

### "Can we simplify this?"
Usually means:
1. Can nilearn do this for us?
2. Is there unnecessary complexity we can remove?
3. Can we reduce the API surface?

My response pattern: "Yes, we can simplify by [specific approach]. Nilearn's [function] handles this. Trade-off: [what we lose]. Shall I proceed?"

---

## 🔬 Technical Expertise & Knowledge Boundaries

### Where I Can Speak Confidently

**nilearn patterns** - I understand:
- Masker types and when to use each (NiftiMasker vs LabelsMasker vs MapsMasker)
- GLM implementation details and parameters
- Image manipulation functions and their trade-offs
- Common pitfalls (dimension mismatches, affine problems)

**pytest best practices** - I know:
- Efficient test selection patterns (`--lf`, `-k`, `-x`)
- Fixture scoping for performance
- Debugging strategies (`--pdb`, `-vv`)
- neuroimaging-specific test patterns

**Refactoring strategies** - I can:
- Identify code duplication patterns
- Suggest appropriate abstraction levels
- Plan incremental migrations
- Maintain backward compatibility

### Where I Need to Research

**Latest API changes** - I always verify:
- nilearn functions (check their docs, version 0.11.1+)
- scikit-learn API changes
- New Python features (3.10+)

**Domain-specific statistics** - I research before suggesting:
- Advanced GLM contrasts
- Permutation testing specifics
- Multiple comparison corrections

**Performance optimizations** - I investigate:
- Memory usage patterns for large datasets
- Parallelization opportunities
- Caching strategies

### How I Research

1. **Check existing knowledge**: `ls claude-research/`
2. **Use sub-agents** for deep research: Create/update markdown files
3. **Verify with source**: Check official docs, not blog posts
4. **Test empirically**: Write small test to verify behavior

---

## 🤝 Our Working Patterns

### Communication Patterns That Work

**When I should be direct**:
- "This is already in nilearn as `function_name`"
- "That would break backward compatibility"
- "The test is failing because [specific reason]"

**When I should offer options**:
- "We could use approach A (faster) or B (more flexible)"
- "nilearn has X, or we could implement Y"
- "Trade-off: simplicity vs performance"

**When I should ask for clarification**:
- Ambiguous requirements
- Multiple valid interpretations
- Breaking changes without clear benefit

### The Staging Protocol

**CRITICAL: NEVER commit without explicit approval. Only stage changes.**

```python
# Our workflow invariant:
if changes_ready:
    git_add_all()
    say("Changes staged and ready for review")
    # WAIT FOR APPROVAL - DO NOT COMMIT
else:
    dont_stage()
    say("Stuck on X, not staged because...")

# You'll respond with either:
# "Go ahead and commit" or
# "Let me modify first" or
# "Let's fix X before committing"
```

**After making changes, ALWAYS update:**
1. `MIGRATION_v0.5_to_v0.6.md` - Document any API changes or compatibility notes
2. `REFACTORING_PLAN.md` - Update progress tracker and completed tasks

### Test-Driven Workflow

Our TDD cycle for nltools:
1. **Identify test**: `uv run pytest -k pattern --co`
2. **Run failing test**: `uv run pytest path::test -xvs`
3. **Implement minimal fix**
4. **Verify**: `uv run pytest --lf`
5. **Check for regressions**: `uv run pytest path/to/module`

### Research Before Implementation

**Always check before coding**:
```python
research_sources = [
    "claude-research/",      # Our research docs
    "git log uv-refactor",   # Historical decisions
    "nilearn docs",          # Current APIs
    "pytest docs"            # Testing patterns
]
```

---

## 📚 Knowledge Base Structure

### Active Documents
- **REFACTORING_PLAN.md**: Current tasks, progress tracker, next steps
- **MIGRATION_v0.5_to_v0.6.md**: User-facing upgrade guide
- **priorities.md**: High-level roadmap from Eshin
- **refactor.md**: Original refactoring requirements

### Research Archive (`claude-research/`)
Research docs that inform our decisions:
- `nilearn_features_analysis.md`: What nilearn can do for us
- `research-nilearn-maskers.md`: Masker types and usage
- `apply_mask_analysis.md`: Masking optimization strategies
- `braindata-refactor.md`: Brain_Data simplification plan

**Important**: Always check these before implementing. Update them when discovering new patterns.

### Code Patterns to Follow

**Input validation**:
```python
# Use validation module, don't inline
from nltools.data._validation import validate_data_type
data = validate_data_type(data)
```

**Leveraging nilearn**:
```python
# GOOD: Use nilearn
from nilearn.image import smooth_img
smoothed = Brain_Data(smooth_img(brain.to_nifti(), fwhm=6))

# BAD: Reimplement
def custom_smooth(data, kernel):  # Don't do this
```

**Memory management**:
```python
# Process large data in chunks
for chunk in np.array_split(indices, n_chunks):
    process(brain_data[chunk])
```

---

## 🎯 Current Priorities & Status

### Priority 1: Core Refactoring ✅ 90% Complete
**What we've done**:
- Removed post-v0.5.1 features (Brain_Collection, Model)
- Implemented nilearn-based methods
- Added deprecation stubs
- Fixed major compatibility issues

**What remains** (7 test failures):
1. `.regress()` test - TypeError with design matrix
2. `.extract_roi()` test - ValueError
3. Mystery failures - smooth, decompose, similarity, bootstrap
4. Legacy h5 loading - handle missing Y attribute

### Priority 2: Documentation (Next)
- Fix remaining test failures
- Migrate Sphinx → Jupyter Book
- Update tutorials for v0.6.0 API

### Priority 3: New Features (Future)
- Implement Model class (holds deprecated methods)
- Implement Brain_Collection
- Add advanced ML workflows

---

## 🔍 Quick Debugging Reference

### Common Issues & Solutions

**"Why is this test failing?"**
```bash
# 1. Run with maximum verbosity
uv run pytest path::test -xvs --tb=long

# 2. Check what changed
git diff HEAD~1 path/to/file.py

# 3. Look for patterns
uv run pytest -k similar_pattern
```

**"Is this in nilearn?"**
```python
# Check nilearn's modules
from nilearn import maskers, image, glm, plotting
dir(module)  # List available functions

# Check research docs
grep -r "function_name" claude-research/
```

**"What did we decide about X?"**
```bash
# Search commit messages
git log --grep="keyword"

# Search code changes
git log -p -S "code_pattern"

# Check research
ls claude-research/*keyword*
```

---

## 🚀 Starting a Fresh Session

When you say **"Where did we leave off?"**, I'll:
1. Check git log for last commit
2. Review REFACTORING_PLAN.md progress
3. Run tests to see current failures
4. Summarize status and suggest next steps

When you say **"Continue working on X"**, I'll:
1. Check this file for context on X
2. Review relevant research docs
3. Set up appropriate todo list
4. Start with targeted tests

When you say **"Why did we..."**, I'll:
1. Check this file's decision log
2. Review git history
3. Explain rationale and trade-offs
4. Suggest alternatives if you want to reconsider

---

## 📝 Meta Notes

**This document is living** - Update it when:
- We make significant design decisions
- We discover useful patterns
- We change our workflow
- We learn something important about the libraries

**Confidence calibration**:
- I speak confidently about: nilearn patterns, pytest, refactoring strategies
- I research before claiming: latest APIs, performance optimizations, statistical methods
- I always verify: specific function signatures, version compatibility

**Our collaboration principle**:
You provide vision and domain expertise. I provide implementation, research, and push back when appropriate. Together we build pragmatic, user-friendly neuroimaging tools.

---

*Last updated: 2025-10-28*
*Branch: uv-cleanup*
*Version target: v0.6.0*