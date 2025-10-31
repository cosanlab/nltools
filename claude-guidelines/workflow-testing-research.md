# Workflow/Behavioral Testing Research & Recommendations

**Date**: 2025-10-29
**Context**: Exploring modern approaches to workflow/behavioral testing for nltools v0.6.0

---

## Executive Summary

**Recommendation: Enhanced pytest with executable tutorials (Option 2)**

This approach:
- ✅ Leverages existing tutorials as real-world workflow tests
- ✅ Minimal new infrastructure (stay with pytest)
- ✅ Provides double value (documentation + validation)
- ✅ Natural fit for scientific Python ecosystem
- ✅ Can layer on property-based testing (Hypothesis) incrementally

---

## Research Findings

### Current Industry Trends (2024-2025)

**Key Insights:**
1. **pytest remains dominant** for scientific Python (scikit-learn, nilearn, pandas all use it)
2. **Property-based testing (Hypothesis)** gaining traction for numerical code
3. **BDD frameworks** (pytest-bdd, Behave) popular for web/enterprise, less for data science
4. **Tutorial-as-test pattern** increasingly common in scientific libraries (nilearn examples)

### What Similar Libraries Do

**Scikit-learn:**
- Uses pytest exclusively
- Organizes tests by component (estimators, pipeline, meta-estimators, etc.)
- Has dedicated `test_pipeline.py` for workflow/integration tests
- Uses diverse datasets (toy, synthetic, real-world) for validation
- Custom testing utilities in `sklearn/utils/_testing.py`

**Nilearn:**
- pytest + pytest-cov for coverage
- Examples directory serves dual purpose (docs + smoke tests)
- Examples executed during CI to validate workflows
- Tests organized by module functionality

**Common Pattern:**
- Unit tests validate components
- Integration tests validate workflows
- Examples/tutorials serve as real-world validation
- All run through pytest

---

## Options Analysis

### Option 1: Behavior-Driven Development (pytest-bdd)

**What it is:**
- Write scenarios in Gherkin syntax (Given-When-Then)
- Map scenarios to Python step definitions
- Integrates with pytest ecosystem

**Example:**
```gherkin
# features/multivariate_prediction.feature
Feature: Multivariate Prediction Workflow
  As a researcher
  I want to run MVPA analyses
  So that I can predict outcomes from brain data

  Scenario: Ridge regression with k-fold CV
    Given I have loaded pain dataset
    And I set Y to PainLevel
    When I run ridge regression with 5-fold CV
    Then I should get cross-validated predictions
    And I should get a weight map
    And prediction accuracy should be above chance
```

**Pros:**
- ✅ Human-readable scenarios (non-technical stakeholders can understand)
- ✅ Clear separation: scenarios vs. implementation
- ✅ Works within pytest ecosystem
- ✅ Good for documenting expected behavior

**Cons:**
- ❌ Extra abstraction layer (feature files + step definitions)
- ❌ Overhead for scientific workflows (less value than web/enterprise apps)
- ❌ Not widely adopted in scientific Python community
- ❌ Gherkin syntax may feel verbose for technical users
- ❌ Step definitions can become repetitive

**Verdict:** ⚠️ Overkill for nltools
- Scientific users prefer code-first approaches
- Extra ceremony without clear benefit for our use case
- Better suited for stakeholder-driven development

---

### Option 2: Enhanced pytest + Executable Tutorials (RECOMMENDED)

**What it is:**
- Organize workflow tests in `nltools/tests/workflows/`
- Convert existing tutorials into pytest-runnable tests
- Use pytest fixtures for common setup patterns
- Add property-based tests (Hypothesis) for numerical invariants

**Structure:**
```
nltools/tests/
├── shell/              # Object API tests (existing)
├── core/               # Function tests (existing)
├── support/            # Integration tests (existing)
├── workflows/          # NEW: End-to-end workflow tests
│   ├── conftest.py     # Workflow-specific fixtures
│   ├── test_mvpa_workflow.py
│   ├── test_roi_analysis_workflow.py
│   ├── test_hyperalignment_workflow.py
│   ├── test_regression_workflow.py
│   └── test_data_manipulation_workflow.py
└── tutorials/          # NEW: Tutorial validation tests
    └── test_tutorials_executable.py
```

**Example Test:**
```python
# nltools/tests/workflows/test_mvpa_workflow.py
import pytest
from nltools.datasets import fetch_pain

class TestMVPAWorkflow:
    """End-to-end tests for MVPA prediction workflows."""

    def test_ridge_prediction_with_kfold_cv(self, pain_data):
        """
        Workflow: Load data → Set outcomes → Run ridge with CV → Validate results

        Validates:
        - Prediction pipeline completes without errors
        - Returns expected outputs (weight_map, yfit_xval, etc.)
        - Predictions are above chance (correlation > 0)
        - Weight map has correct shape
        """
        # Setup
        data = pain_data
        data.Y = data.X['PainLevel']

        # Execute workflow
        stats = data.predict(
            algorithm='ridge',
            cv_dict={'type': 'kfolds', 'n_folds': 5, 'stratified': data.Y}
        )

        # Validate outputs
        assert 'weight_map' in stats
        assert 'yfit_xval' in stats
        assert stats['weight_map'].shape() == data.shape()

        # Validate performance (above chance)
        from scipy.stats import pearsonr
        r, p = pearsonr(data.Y, stats['yfit_xval'])
        assert r > 0.1, f"Prediction accuracy {r:.3f} not above chance"

    def test_loso_cross_validation_workflow(self, pain_data):
        """Workflow: Leave-one-subject-out cross-validation."""
        data = pain_data
        data.Y = data.X['PainLevel']
        subject_id = data.X['SubjectID']

        stats = data.predict(
            algorithm='ridge',
            cv_dict={'type': 'loso', 'subject_id': subject_id}
        )

        # Validate LOSO behavior
        assert len(stats['yfit_xval']) == len(data)
        # Each fold should hold out different subjects
        # (implementation-specific validation)

# nltools/tests/tutorials/test_tutorials_executable.py
def test_basic_workflow_tutorial_runs():
    """Validates that basic workflow tutorial executes without errors."""
    # Execute notebook using nbconvert or papermill
    # Assert no exceptions raised
    # Optionally validate key outputs
    pass

def test_multivariate_prediction_tutorial():
    """Validates multivariate prediction tutorial."""
    pass
```

**Property-Based Testing Example (Hypothesis):**
```python
# nltools/tests/workflows/test_numerical_properties.py
from hypothesis import given, strategies as st
import numpy as np

@given(
    n_samples=st.integers(min_value=10, max_value=100),
    n_features=st.integers(min_value=5, max_value=50)
)
def test_regression_invariants(n_samples, n_features):
    """
    Property: Regression should work for any valid data shape.
    Tests with randomly generated data dimensions.
    """
    from nltools.data import BrainData

    # Generate random data with given shape
    data = BrainData(np.random.randn(n_samples, n_features))
    data.Y = np.random.randn(n_samples)

    # Regression should not crash
    results = data.regress()

    # Invariants that should always hold
    assert results['beta'].shape() == (n_features,)
    assert results['t'].shape() == (n_features,)
    assert not np.any(np.isnan(results['beta'].data))
```

**Pros:**
- ✅ **Double value**: Tutorials become validated code examples
- ✅ **Minimal overhead**: Stays within pytest (no new framework)
- ✅ **Natural fit**: Matches scientific Python community standards
- ✅ **Flexible**: Can add property-based tests incrementally
- ✅ **Real-world coverage**: Tutorials represent actual user workflows
- ✅ **Catches regressions**: Tutorial failures indicate breaking changes
- ✅ **Living documentation**: Tests enforce that docs stay current

**Cons:**
- ⚠️ Setup required: Need to make tutorials pytest-runnable
- ⚠️ Execution time: Some workflows may be slow (mitigated with markers)
- ⚠️ Data dependencies: May need to mock/cache large datasets

**Implementation Steps:**
1. Create `nltools/tests/workflows/` directory
2. Extract key workflows from tutorials into pytest tests
3. Add Hypothesis for property-based testing of numerical code
4. Use pytest markers (`@pytest.mark.slow`) for long-running tests
5. Optionally: Use `nbconvert` or `papermill` to execute tutorial notebooks in CI

---

### Option 3: Dedicated Integration Test Framework (Robot Framework, Tavern)

**What it is:**
- Separate framework for end-to-end testing
- YAML or table-driven test definitions
- Often used for API/microservice testing

**Pros:**
- ✅ Clear separation from unit tests
- ✅ Non-programmer friendly syntax

**Cons:**
- ❌ **Not common in scientific Python** (no adoption in scikit-learn, nilearn, etc.)
- ❌ Another tool to learn and maintain
- ❌ Worse IDE integration
- ❌ Less Python-native

**Verdict:** ❌ Not recommended
- Poor fit for scientific computing domain
- Adds complexity without clear benefits

---

### Option 4: Doctest-Based Workflows

**What it is:**
- Embed executable examples in docstrings
- pytest can discover and run doctests

**Example:**
```python
def predict(self, algorithm='ridge', cv_dict=None, **kwargs):
    """
    Run multivariate prediction with cross-validation.

    Examples
    --------
    >>> from nltools.datasets import fetch_pain
    >>> data = fetch_pain()
    >>> data.Y = data.X['PainLevel']
    >>> stats = data.predict(algorithm='ridge',
    ...                      cv_dict={'type': 'kfolds', 'n_folds': 5})
    >>> 'weight_map' in stats
    True
    >>> 'yfit_xval' in stats
    True
    """
```

**Pros:**
- ✅ Documentation + validation in one place
- ✅ Encourages good examples in docstrings
- ✅ Zero additional test files

**Cons:**
- ❌ Limited assertions (must be simple)
- ❌ Hard to test complex workflows
- ❌ Verbose in docstrings
- ❌ Difficult to debug failures

**Verdict:** ⚠️ Complementary tool only
- Good for simple API examples
- Not sufficient for workflow testing
- Use alongside Option 2, not instead of

---

## Recommendation: Hybrid Approach

**Core Strategy**: Enhanced pytest (Option 2) + selective Hypothesis

### Phase 1: Workflow Test Infrastructure (v0.6.0)
```bash
# Create new test organization
nltools/tests/
├── workflows/              # NEW: End-to-end workflow tests
│   ├── conftest.py
│   ├── test_mvpa_workflow.py
│   ├── test_roi_analysis_workflow.py
│   ├── test_regression_workflow.py
│   └── test_group_analysis_workflow.py
```

**Test Categories:**
1. **MVPA Workflows**: Full prediction pipelines (ridge, lasso, SVM, etc.)
2. **ROI Analysis**: Extract → analyze → visualize
3. **Regression Workflows**: Design matrix → regress → threshold
4. **Group Analysis**: Load → group comparison → statistics
5. **Hyperalignment/SRM**: Multi-subject alignment workflows

### Phase 2: Tutorial Validation (v0.6.1+)
```bash
nltools/tests/
├── tutorials/              # NEW: Execute tutorials as tests
│   ├── conftest.py
│   └── test_tutorial_notebooks.py
```

Use `papermill` or `nbconvert` to execute tutorial notebooks in CI.

### Phase 3: Property-Based Testing (v0.7.0+)
Add Hypothesis tests for:
- Numerical stability (regression with various data shapes/ranges)
- Mathematical invariants (e.g., correlation symmetry, standardization properties)
- Edge cases (empty data, single sample, high-dimensional, etc.)

---

## Implementation Checklist

### Immediate (v0.6.0)
- [ ] Create `nltools/tests/workflows/` directory
- [ ] Add workflow-specific fixtures in `workflows/conftest.py`
- [ ] Implement 3-5 core workflow tests based on most common tutorials:
  - [ ] MVPA prediction workflow
  - [ ] ROI analysis workflow
  - [ ] Regression workflow
  - [ ] Group comparison workflow
  - [ ] Hyperalignment workflow
- [ ] Add pytest markers for slow tests: `@pytest.mark.workflow`
- [ ] Update CLAUDE.md with workflow testing guidelines
- [ ] Update refactor-todos.md with workflow test task

### Near-term (v0.6.1)
- [ ] Research tutorial execution (papermill vs nbconvert)
- [ ] Create `test_tutorial_notebooks.py` that validates tutorials run
- [ ] Add tutorial execution to CI (may be slow, possibly separate job)
- [ ] Configure caching for tutorial data

### Future (v0.7.0+)
- [ ] Add Hypothesis dependency (dev)
- [ ] Implement property-based tests for numerical code
- [ ] Consider parameterized tests for algorithm variations
- [ ] Add performance benchmarks (optional: pytest-benchmark)

---

## Example: Complete Workflow Test

```python
# nltools/tests/workflows/test_mvpa_workflow.py
"""
End-to-end workflow tests for MVPA prediction analyses.

These tests validate complete user workflows from data loading through
prediction and validation. They ensure the entire pipeline works together
and produces sensible results.
"""

import pytest
import numpy as np
from scipy.stats import pearsonr
from nltools.datasets import fetch_pain


@pytest.fixture(scope="module")
def pain_data():
    """Load pain dataset once for all tests in this module."""
    return fetch_pain()


class TestMVPAPredictionWorkflow:
    """Test complete MVPA prediction workflows."""

    @pytest.mark.workflow
    def test_ridge_kfold_workflow(self, pain_data):
        """
        Complete workflow: Ridge regression with k-fold cross-validation.

        Workflow Steps:
        1. Load pain dataset
        2. Set outcome variable (PainLevel)
        3. Run ridge prediction with 5-fold CV
        4. Validate outputs and performance

        Expected Behavior:
        - Pipeline completes without errors
        - Returns weight_map and cross-validated predictions
        - Prediction correlation is above chance (r > 0.1)
        - Weight map has correct spatial dimensions
        """
        # Step 1 & 2: Setup data and outcome
        data = pain_data.copy()  # Don't modify fixture
        data.Y = data.X['PainLevel']

        # Step 3: Run prediction
        stats = data.predict(
            algorithm='ridge',
            cv_dict={'type': 'kfolds', 'n_folds': 5, 'stratified': data.Y}
        )

        # Step 4: Validate outputs
        assert 'weight_map' in stats, "Missing weight_map in results"
        assert 'yfit_xval' in stats, "Missing cross-validated predictions"
        assert len(stats['yfit_xval']) == len(data), "Predictions length mismatch"

        # Validate weight map
        assert stats['weight_map'].shape() == data[0].shape(), "Weight map shape mismatch"
        assert not np.all(stats['weight_map'].data == 0), "Weight map is all zeros"

        # Validate performance
        r, p = pearsonr(data.Y, stats['yfit_xval'])
        assert r > 0.1, f"Prediction accuracy (r={r:.3f}) not above chance"
        assert p < 0.05, f"Prediction not significant (p={p:.3f})"

    @pytest.mark.workflow
    def test_loso_cv_workflow(self, pain_data):
        """
        Complete workflow: Leave-one-subject-out cross-validation.

        Validates that LOSO CV:
        - Respects subject boundaries
        - Produces predictions for all samples
        - Maintains data integrity
        """
        data = pain_data.copy()
        data.Y = data.X['PainLevel']
        subject_id = data.X['SubjectID']

        stats = data.predict(
            algorithm='ridge',
            cv_dict={'type': 'loso', 'subject_id': subject_id}
        )

        # Validate LOSO behavior
        assert len(stats['yfit_xval']) == len(data)
        assert 'weight_map' in stats

        # Check that we get reasonable predictions
        r, _ = pearsonr(data.Y, stats['yfit_xval'])
        assert r > 0, "LOSO predictions should have positive correlation"

    @pytest.mark.workflow
    @pytest.mark.parametrize("algorithm", ["ridge", "lasso", "svr", "pcr"])
    def test_algorithm_compatibility_workflow(self, pain_data, algorithm):
        """
        Workflow: Test that all algorithms work with standard CV.

        Ensures API consistency across different prediction algorithms.
        """
        data = pain_data.copy()
        data.Y = data.X['PainLevel']

        # Each algorithm should work with standard 5-fold CV
        stats = data.predict(
            algorithm=algorithm,
            cv_dict={'type': 'kfolds', 'n_folds': 5}
        )

        # All should return same output structure
        assert 'weight_map' in stats
        assert 'yfit_xval' in stats
        assert len(stats['yfit_xval']) == len(data)


class TestRegressionWorkflow:
    """Test complete regression analysis workflows."""

    @pytest.mark.workflow
    def test_regression_with_design_matrix_workflow(self, pain_data):
        """
        Complete workflow: Regression with custom design matrix.

        Workflow Steps:
        1. Load data
        2. Create design matrix with intercept + predictor
        3. Run regression
        4. Validate beta maps and statistics
        """
        import pandas as pd

        data = pain_data.copy()

        # Create design matrix
        data.X = pd.DataFrame({
            'intercept': 1,
            'pain_rating': data.X['PainLevel']
        })

        # Run regression
        results = data.regress()

        # Validate results
        assert 'beta' in results
        assert 't' in results
        assert len(results['beta']) == 2  # intercept + pain_rating
        assert len(results['t']) == 2

        # Validate shapes
        for beta_map in results['beta']:
            assert beta_map.shape() == data[0].shape()


class TestROIWorkflow:
    """Test ROI extraction and analysis workflows."""

    @pytest.mark.workflow
    def test_roi_extraction_workflow(self, pain_data):
        """
        Complete workflow: Create ROI from thresholded map and extract data.

        Workflow Steps:
        1. Perform regression
        2. Threshold t-map to create ROI
        3. Extract ROI data
        4. Validate extracted values
        """
        import pandas as pd

        data = pain_data.copy()
        data.X = pd.DataFrame({
            'intercept': 1,
            'pain_rating': data.X['PainLevel']
        })

        # Run regression
        results = data.regress()

        # Create ROI from top 5% of voxels
        roi = results['t'][1].threshold(upper='95%', binarize=True)

        # Extract ROI data
        roi_activity = data.extract_roi(roi)

        # Validate extraction
        assert len(roi_activity) == len(data)
        assert roi_activity.ndim == 1
        assert not np.all(roi_activity == 0)


# Optional: Property-based tests with Hypothesis (future)
"""
from hypothesis import given, strategies as st

@given(
    n_folds=st.integers(min_value=2, max_value=10),
    algorithm=st.sampled_from(['ridge', 'lasso'])
)
def test_cv_folds_property(pain_data, n_folds, algorithm):
    '''
    Property: Prediction should work with any valid number of folds.
    '''
    data = pain_data.copy()
    data.Y = data.X['PainLevel']

    stats = data.predict(
        algorithm=algorithm,
        cv_dict={'type': 'kfolds', 'n_folds': n_folds}
    )

    assert len(stats['yfit_xval']) == len(data)
"""
```

---

## pytest Configuration Updates

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (moderate speed)",
    "workflow: End-to-end workflow tests (may be slow)",
    "slow: Tests that take >5 seconds",
    "tutorial: Tutorial validation tests (very slow)"
]

# Run fast tests by default
addopts = "-m 'not slow and not tutorial'"
```

**Usage:**
```bash
# Run only workflow tests
uv run pytest -m workflow

# Run all tests including slow ones
uv run pytest -m ""

# Run everything except tutorial validation
uv run pytest -m "not tutorial"

# Run fast tests (default)
uv run pytest
```

---

## Resources & References

### Tools
- **pytest**: https://pytest.org
- **Hypothesis**: https://hypothesis.readthedocs.io
- **papermill**: https://papermill.readthedocs.io (notebook execution)
- **nbconvert**: https://nbconvert.readthedocs.io (notebook execution)
- **pytest-bdd**: https://pytest-bdd.readthedocs.io (if considering BDD)

### Examples
- **scikit-learn testing**: https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/tests
- **nilearn testing**: https://github.com/nilearn/nilearn
- **Hypothesis for science**: https://hypothesis.works/articles/intro/

### Articles
- "Testing ML Code: How Scikit-learn Does It" (Medium)
- "Property-Based Testing in Python with Hypothesis" (Semaphore)
- "Getting Started with pytest-bdd" (Real Python)

---

## Conclusion

**Recommended Approach**: Enhanced pytest with workflow tests + executable tutorials

This approach:
1. **Stays within pytest ecosystem** (minimal tooling overhead)
2. **Leverages existing tutorials** (double value for effort)
3. **Matches scientific Python conventions** (follows scikit-learn/nilearn patterns)
4. **Provides flexibility** (can add Hypothesis later for property-based testing)
5. **Clear organization** (`workflows/` directory signals intent)

**Implementation Priority**:
1. **Phase 1 (v0.6.0)**: Core workflow tests (3-5 tests covering most common use cases)
2. **Phase 2 (v0.6.1)**: Tutorial validation (execute notebooks in CI)
3. **Phase 3 (v0.7.0+)**: Property-based testing (Hypothesis for numerical properties)

This strategy provides immediate value with minimal overhead while leaving room for future enhancements.
