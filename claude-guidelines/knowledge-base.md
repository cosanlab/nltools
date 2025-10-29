# nltools Knowledge Base & Technical Patterns

*Reference for development workflows, research methodology, and technical patterns*

---

## Active Documentation Structure

### Primary Documents
- **CLAUDE.md**: Current status, critical requirements, quick commands
- **REFACTORING_PLAN.md**: Current tasks, progress tracker, next steps
- **MIGRATION_v0.5_to_v0.6.md**: User-facing upgrade guide
- **priorities.md**: High-level roadmap from Eshin
- **refactor.md**: Original refactoring requirements

### Guidelines & Research Archive (`claude-guidelines/`)
Documents that inform our decisions:
- **design-philosophy.md**: Why we made key architectural decisions
- **knowledge-base.md**: This file - technical patterns and workflows
- **refactoring-context.md**: Meta-patterns for understanding context
- **nilearn_features_analysis.md**: What nilearn can do for us (in claude-research/)
- **research-nilearn-maskers.md**: Masker types and usage (in claude-research/)
- **apply_mask_analysis.md**: Masking optimization strategies (in claude-research/)
- **braindata-refactor.md**: Brain_Data simplification plan (in claude-research/)

**Important**: Always check guideline docs before implementing. Update them when discovering new patterns.

---

## Technical Expertise & When to Research

### Research First: Latest API Changes

**ALWAYS verify current documentation for:**
- nilearn functions (version 0.11.1+) - breaking changes common
- scikit-learn APIs - deprecation cycles
- pytest features - new testing patterns
- Python 3.10+ features - new syntax and stdlib additions

**How to verify:**
```python
# Check installed version
uv run python -c "import nilearn; print(nilearn.__version__)"

# Look up current API
# Use context7 MCP or official docs, not blog posts
```

### Research First: Domain-Specific Statistics

**Always investigate before suggesting:**
- Advanced GLM contrasts and design matrices
- Permutation testing specifics (e.g., exchangeability assumptions)
- Multiple comparison corrections (when to use what)
- Bayesian alternatives to frequentist methods

**Why**: Statistical correctness is critical for users' published research. Don't wing it.

### Research First: Performance Optimizations

**Investigate before implementing:**
- Memory usage patterns for large neuroimaging datasets (can be 10GB+)
- Parallelization opportunities (when overhead is worth it)
- Caching strategies (what to cache, when to invalidate)
- Sparse matrix operations (when appropriate)

**Benchmark before claiming improvement**: Run tests, measure actual impact.

---

## Research Methodology

### Step 1: Check Existing Knowledge

```bash
# Search guidelines & research archive
ls -lh claude-guidelines/
grep -r "keyword" claude-guidelines/

# Search git history for decisions
git log --grep="keyword"
git log -p -S "code_pattern"  # Search code changes

# Check for previous implementations
git log uv-refactor  # Reference branch with historical changes
```

### Step 2: Consult Primary Sources

**Priority order:**
1. **Official documentation**: nilearn docs, pytest docs, Python docs
2. **Guidelines archive**: Our previous investigations (claude-guidelines/)
3. **Git history**: Commit messages (very detailed in this project)
4. **Academic papers**: For statistical methods
5. **Blog posts/Stack Overflow**: Only to find pointers to official docs

**Use context7 MCP for:**
- Official library documentation lookups
- API signature verification
- Breaking change identification
- Best practice confirmation from official sources

### Step 3: Document Findings

**For significant research (>1 hour investigation):**
```bash
# Create dated research document
# File: claude-guidelines/YYYY-MM-DD-topic.md

# Structure:
## Question/Problem
## Sources Consulted
## Key Findings
## Recommendation
## Trade-offs
## Example Code
```

**For quick verifications (API checks, syntax):**
- No need to create file
- Reference source in commit message if implementing based on it

### Step 4: Empirical Verification

**When uncertain, test it:**
```python
# Create minimal test case
uv run python -c "
from nilearn import image
# Test the API behavior
"

# Or write quick pytest
uv run pytest -k test_name -xvs
```

---

## Communication Patterns for Development

### When to Be Direct

**Use clear, definitive statements when:**
- Nilearn already provides the functionality
  - "This is already in nilearn as `smooth_img`"
- Breaking backward compatibility
  - "That would break v0.5.1 compatibility"
- Test failures have clear root cause
  - "The test is failing because X is None when it expects an array"

### When to Offer Options

**Present alternatives when:**
- Multiple valid approaches exist
  - "We could use approach A (faster) or B (more flexible). A uses nilearn's X, B uses our custom Y."
- Performance vs. simplicity trade-offs
  - "Option 1: Simple but O(n²). Option 2: Complex but O(n log n). For typical n<100, Option 1 is fine."
- API design decisions
  - "We could return dict (v0.5.1 pattern) or store as attributes (cleaner). I recommend attributes because..."

### When to Ask for Clarification

**Ask questions when:**
- Requirements are ambiguous
  - "Should this work for 4D data or just 3D?"
- Multiple valid interpretations exist
  - "By 'smooth', do you mean spatial smoothing or temporal smoothing?"
- Breaking changes need justification
  - "This would break v0.5.1 compatibility. Is that acceptable for v0.6.0?"
- Statistical methods have multiple approaches
  - "For multiple comparisons: FDR, Bonferroni, or permutation-based correction?"

---

## Code Quality Standards

### Input Validation Pattern

**DO**: Use centralized validation module
```python
from nltools.data._validation import validate_data_type
data = validate_data_type(data)  # Handles Brain_Data, str paths, nifti, etc.
```

**WHY**:
- Consistent error messages
- Single source of truth for validation logic
- Easier to maintain and test

### Error Messages

**DO**: Be specific and actionable
```python
raise ValueError(
    f"Expected 2D array of shape (n_samples, n_features), "
    f"but got shape {data.shape}. "
    f"Hint: Use brain_data.data to access the 2D array."
)
```

**DON'T**: Be vague
```python
raise ValueError("Invalid shape")  # What shape? What's expected?
```

### Docstring Standards

**Follow NumPy style** (consistent with nilearn):
```python
def method_name(self, param1, param2=None):
    """Short one-line description.

    Longer description explaining what the method does,
    when to use it, and important caveats.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2. Default is None.

    Returns
    -------
    result_type
        Description of return value

    Notes
    -----
    Additional information, edge cases, references to papers.

    Examples
    --------
    >>> brain = Brain_Data('data.nii.gz')
    >>> result = brain.method_name(param1)
    """
```

### Type Hints

**Use type hints for function signatures**:
```python
from typing import Optional, Union
import numpy as np
import pandas as pd

def process(
    self,
    data: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Process data with optional mask."""
    ...
```

**Why**: Helps with IDE autocomplete and catches type errors early.

---

## Testing Patterns

### Fixture Organization

**Scope fixtures appropriately**:
```python
@pytest.fixture(scope="session")  # Expensive, share across all tests
def sample_nifti_file():
    """Create sample nifti file once for all tests."""
    ...

@pytest.fixture(scope="function")  # Cheap, create fresh for each test
def brain_data():
    """Create Brain_Data instance for each test."""
    ...
```

### Test Selection for Fast Iteration

```bash
# Run last failed tests
uv run pytest --lf -x

# Run tests matching pattern
uv run pytest -k "regress or extract" -x

# Run specific test with verbose output
uv run pytest nltools/tests/test_brain_data_old.py::test_regress -xvs

# Run tests in parallel (if independent)
uv run pytest -n auto  # Requires pytest-xdist
```

### Testing Brain_Data Methods

**Pattern for testing methods that return Brain_Data**:
```python
def test_method_returns_brain_data(brain_data):
    """Test method returns Brain_Data instance."""
    result = brain_data.method()

    assert isinstance(result, Brain_Data)
    assert result.shape() == expected_shape
    assert not np.shares_memory(result.data, brain_data.data)  # Verify copy
```

### Testing Deprecated Methods

**Use pytest.raises for deprecation stubs**:
```python
def test_predict_deprecated(brain_data):
    """Verify .predict() raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Model class"):
        brain_data.predict()
```

---

## Performance Optimization Patterns

### Memory-Efficient Processing

**DO**: Process in chunks for large datasets
```python
n_samples = len(brain_data)
chunk_size = 100
results = []

for i in range(0, n_samples, chunk_size):
    chunk = brain_data[i:i+chunk_size]
    results.append(process_chunk(chunk))

return np.vstack(results)
```

**DON'T**: Load everything at once
```python
# This can cause OOM for large datasets
all_data = [brain_data[i] for i in range(len(brain_data))]
```

### Copy Optimization

**DO**: Use `_shallow_copy_with_data()` for method chaining
```python
def transform(self):
    out = self._shallow_copy_with_data()  # Share immutables
    out.data = self.data.copy()  # Copy only what changes
    # ... modify out.data ...
    return out
```

**DON'T**: Deep copy entire object every time
```python
def transform(self):
    out = self.copy()  # Expensive: copies everything
    # ... modify out.data ...
    return out
```

### Vectorization

**DO**: Use numpy operations
```python
# Vectorized: fast
result = (data - data.mean(axis=0)) / data.std(axis=0)
```

**DON'T**: Use Python loops for array operations
```python
# Loop: slow for large arrays
result = np.empty_like(data)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        result[i, j] = (data[i, j] - mean[j]) / std[j]
```

---

## Token-Efficient Pytest Debugging

**Critical principle**: Pytest output is expensive in tokens; log files are cheap to analyze.

### The Token Economics

**Typical pytest run costs:**
- Full test suite: 3,000-8,000 tokens
- Single module: 500-2,000 tokens
- Single test with `-xvs --tb=long`: 100-500 tokens

**Searching logged output costs:**
- Grep tool on log: 30-100 tokens per search
- Read tool on log: 50-300 tokens per section
- Bash grep for counts: 10-20 tokens

**The problem - Anti-pattern (wasteful):**
```bash
# DON'T: Re-run pytest for each search pattern
uv run pytest -xvs | grep "FAILED"      # 2000 tokens
uv run pytest -xvs | grep "Error"       # 2000 tokens
uv run pytest -xvs | grep "test_foo"    # 2000 tokens
# Total: 6000 tokens for 3 searches
```

**The solution - Efficient pattern:**
```bash
# DO: Log once, search many times
uv run pytest -xvs 2>&1 | tee test.log  # 2000 tokens (once)
# Then use Grep tool on test.log        # 50 tokens × 3 = 150
# Total: 2150 tokens for 3 searches (64% reduction!)
```

### Decision Framework

**ALWAYS log to file when:**
1. Running full test suite or entire module
2. First diagnostic run (unknown scope of failures)
3. You'll need to search for 2+ different patterns
4. Output expected to be >500 lines
5. Analyzing test failure patterns across suite
6. Debugging across 10+ test failures

**Run directly (no log) when:**
1. Quick verification after a fix (expect <50 lines output)
2. Interactive debugging with `--pdb`
3. Single specific test that's likely to pass
4. Using pytest features requiring real-time interaction
5. Exploring test options (`--co`, `--markers`)

### Recommended Workflow

**Phase 1: Initial Diagnosis (LOG EVERYTHING)**
```bash
# Capture complete output with all details
uv run pytest nltools/tests/ -xvs --tb=long 2>&1 | tee pytest_initial.log

# This logs:
# - All test results
# - Full tracebacks
# - stdout/stderr from tests
# - Summary statistics
```

**Phase 2: Analysis (USE TOOLS ON LOG - TOKEN EFFICIENT)**
```bash
# Use Grep tool to find patterns (each search ~50 tokens)
# Example searches:
# - Pattern: "FAILED" → Find all failed tests
# - Pattern: "AttributeError" → Find attribute errors
# - Pattern: "test_regress" with -A 20 → Get test context
# - Pattern: "ERROR" with output_mode="count" → Count errors

# Use Read tool to examine sections (200-300 tokens)
# - Read pytest_initial.log with offset/limit
# - View specific failure contexts
# - Examine full tracebacks

# Use Bash grep ONLY for quick counts
# grep -c "FAILED" pytest_initial.log  # Just get number
```

**Phase 3: Make Fixes**
```bash
# Edit code based on analysis from log file
# No need to re-run pytest yet
```

**Phase 4: Verify Fix (TARGETED RE-RUN)**
```bash
# Run only what failed (much smaller output)
uv run pytest --lf -x

# If passes: Done! (minimal tokens)
# If fails: Proceed to Phase 5
```

**Phase 5: Iterate (LOG SMALLER SUBSET)**
```bash
# Only failing tests remain, log them
uv run pytest --lf -xvs --tb=long 2>&1 | tee pytest_remaining.log

# Repeat Phase 2 analysis on pytest_remaining.log
# This file is smaller, searches are even cheaper
```

### Tool Selection Guide

**For searching patterns in log files:**
- ✅ **Grep tool** (most token-efficient for pattern searches)
  - Returns only matching lines + context
  - Can use `-A`, `-B`, `-C` for context
  - Can use `output_mode="count"` for counts
  - Example: 30-100 tokens per search

- ⚠️ **Bash grep** (okay for quick counts only)
  - Use only for simple counts: `grep -c "pattern" file`
  - For actual pattern analysis, use Grep tool instead

- ❌ **Re-running pytest** (wasteful)
  - 1000-5000 tokens per run
  - Use only after making code changes

**For viewing sections of log files:**
- ✅ **Read tool** with offset/limit
  - View specific line ranges
  - Extract relevant sections
  - Example: 50-300 tokens per section

- ❌ **Bash cat/head/tail**
  - Use Read tool instead (more efficient)

**For getting counts/summaries:**
- ✅ **Bash grep -c** (if you only need count)
  - `grep -c "FAILED" pytest.log`
  - Returns just a number

- ✅ **Grep tool with output_mode="count"**
  - More detailed count information

### Concrete Examples

**Example 1: Finding all test failures (efficient)**
```bash
# Step 1: Run once and log (2000 tokens)
uv run pytest nltools/tests/ --tb=no 2>&1 | tee pytest_summary.log

# Step 2: Use Grep tool to find failures (~50 tokens)
# Grep tool: pattern="FAILED", file=pytest_summary.log
# Returns: list of failed tests

# Step 3: Get details on specific failure
# Option A: Grep for that specific test with context (~50 tokens)
# Grep tool: pattern="test_regress.*FAILED" with -A 20
#
# Option B: Re-run just that one test (200-300 tokens)
# uv run pytest nltools/tests/test_brain_data_old.py::test_regress -xvs

# Total tokens: 2000 + 50 + 50 = 2100 (efficient!)
# vs. running full suite 3 times = 6000 tokens (wasteful!)
```

**Example 2: Analyzing error patterns across test suite**
```bash
# Step 1: Full test run with details (5000 tokens)
uv run pytest nltools/tests/ -xvs --tb=long 2>&1 | tee pytest_detailed.log

# Step 2: Search for different error types (all cheap - ~50 tokens each)
# Use Grep tool on pytest_detailed.log:
# - Pattern: "AttributeError" → Find attribute access errors
# - Pattern: "TypeError" → Find type mismatches
# - Pattern: "ValueError" → Find value problems
# - Pattern: "ImportError" → Find import issues
# - Pattern: "AssertionError" → Find assertion failures

# Step 3: Count each error type (10 tokens each)
# grep -c "AttributeError" pytest_detailed.log
# grep -c "TypeError" pytest_detailed.log
# (etc.)

# Total: 5000 (initial) + 250 (5 searches) + 50 (counts) = 5300 tokens
# vs. 5× pytest runs = 25,000 tokens
# Savings: 80% reduction!
```

**Example 3: Bisecting where change broke tests**
```bash
# Step 1: Run full suite and log (4000 tokens)
uv run pytest nltools/tests/ -x --tb=short 2>&1 | tee pytest_baseline.log

# Step 2: Make change to code

# Step 3: Run again and log (4000 tokens)
uv run pytest nltools/tests/ -x --tb=short 2>&1 | tee pytest_changed.log

# Step 4: Compare logs efficiently using tools
# Use Grep tool to find failures in each log
# Use diff command to compare: diff pytest_baseline.log pytest_changed.log
# Or use Grep to find new error patterns

# Total: 8000 tokens for comparison
# Much cheaper than re-running suite multiple times while exploring
```

**Example 4: Debugging specific test with multiple hypotheses**
```bash
# Step 1: Run specific test with full details (500 tokens)
uv run pytest nltools/tests/test_brain_data_old.py::test_regress -xvs --tb=long 2>&1 | tee test_regress.log

# Step 2: Search for different clues (each ~30 tokens)
# Use Grep tool:
# - Pattern: "NiftiMasker" → Check masker usage
# - Pattern: "shape" → Check array shapes
# - Pattern: "None" → Check for None values
# - Pattern: "glm" → Check GLM calls

# Step 3: Read specific sections for context
# Use Read tool to view lines 50-100 of test_regress.log

# Total: 500 + 120 (4 searches) + 150 (reading) = 770 tokens
# vs. re-running test 4 times with different print statements = 2000 tokens
```

### Anti-Patterns to Avoid

❌ **Running pytest multiple times for different searches**
```bash
# DON'T DO THIS:
uv run pytest -xvs 2>&1 | grep "Error"     # 2000 tokens
uv run pytest -xvs 2>&1 | grep "FAILED"    # 2000 tokens
uv run pytest -xvs 2>&1 | grep "Warning"   # 2000 tokens
# Total: 6000 tokens for 3 searches

# DO THIS INSTEAD:
uv run pytest -xvs 2>&1 | tee test.log     # 2000 tokens once
# Then use Grep tool 3 times                # 150 tokens total
# Total: 2150 tokens (64% reduction)
```

❌ **Using Bash tools when Claude Code tools are more efficient**
```bash
# DON'T: Pipe entire log through bash
cat pytest.log | grep "pattern"   # Returns full output to me

# DO: Use Grep tool
# Grep tool on pytest.log with pattern
# Returns only matches, much more token-efficient
```

❌ **Not saving logs when output is large**
```bash
# DON'T: Run without logging
uv run pytest nltools/tests/ -xvs  # 5000 tokens
# (Realize you need to search for something)
# (Have to run again)
uv run pytest nltools/tests/ -xvs  # Another 5000 tokens
# Total: 10,000 tokens

# DO: Log first time
uv run pytest nltools/tests/ -xvs 2>&1 | tee test.log  # 5000 tokens
# Then search log file with tools  # 50 tokens per search
# Total: 5000 + searches (much cheaper)
```

❌ **Logging to file but not using tools to analyze it**
```bash
# DON'T: Log but then bash grep everything
uv run pytest -xvs 2>&1 | tee test.log
grep "pattern1" test.log  # Bash command, inefficient
grep "pattern2" test.log  # Bash command, inefficient

# DO: Log and use Grep tool
uv run pytest -xvs 2>&1 | tee test.log
# Then use Grep tool for searches (more token-efficient)
```

### When This Matters Most

**High-value scenarios for logging:**
- Debugging across 10+ test failures
- Bisecting where a change broke things
- Understanding error patterns in test suite
- Working with slow-running tests (>30 seconds)
- Limited token budget in session
- Analyzing complex test interactions

**Lower-value scenarios (direct run okay):**
- Single test, quick verification (<50 lines)
- Tests complete in <5 seconds
- Interactive debugging session (need --pdb)
- Exploring test discovery (--co, --markers)
- Very targeted fix with immediate feedback

### Summary: Token-Efficient Pytest Workflow

1. **First run**: ALWAYS log to file (`tee`)
2. **Analysis**: Use Grep/Read TOOLS on log file (not re-running pytest)
3. **Fix code**: Based on analysis
4. **Verify**: Run pytest again (much smaller, targeted)
5. **Iterate**: If needed, log the smaller failing subset

**Key insight**: Each pytest run costs 1000-5000 tokens. Each tool search costs 30-100 tokens. Use this 10-50× efficiency difference to your advantage!

---

## Debugging Workflows

### Pytest Debugging

**Capture output to logs** (avoid re-running and token waste):
```bash
# Capture full test run
uv run pytest nltools/tests/ -xvs --tb=long 2>&1 | tee pytest_full.log

# Search logs with Grep tool instead of re-running
# Use Grep tool: pattern="FAILED|ERROR" in pytest_full.log
# Use Grep tool: pattern="AttributeError" with -A 10 -B 5
```

### Interactive Debugging

**Use pytest --pdb** for interactive debugging:
```bash
uv run pytest path/to/test.py::test_name --pdb

# When test fails, you get interactive prompt:
# >>> print(variable)
# >>> brain_data.data.shape
```

### Print Debugging

**Use pytest -s** to see print statements:
```bash
uv run pytest path/to/test.py::test_name -s

# Prints will appear in output (not captured)
```

---

## When to Reference This File

**Reference this file when:**
- Implementing new features (check code patterns)
- Writing tests (check testing patterns)
- Investigating failures (check debugging workflows)
- Researching technical questions (check research methodology)
- Optimizing performance (check optimization patterns)

**Update this file when:**
- Discovering new useful patterns
- Solving tricky debugging scenarios
- Learning new pytest features
- Finding performance optimization opportunities

---

*Last updated: 2025-10-28*
*For current status and commands, see CLAUDE.md*
*For design decisions, see design-philosophy.md*
