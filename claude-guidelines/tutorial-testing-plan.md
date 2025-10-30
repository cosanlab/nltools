# Tutorial-Based Workflow Testing Plan

**Date**: 2025-10-29
**Author**: Research synthesis for nltools v0.6.0+
**Context**: Using tutorials as simultaneous pedagogical materials AND behavioral workflow tests

---

## Executive Summary

**Goal**: Develop comprehensive, executable tutorial notebooks that serve dual purposes:
1. **Pedagogical**: Teach users how to use nltools effectively
2. **Testing**: Validate that common workflows execute correctly (behavioral/integration testing)

**Approach**: Adopt pymer4's pattern of executable tutorials built via jupyter-book in CI/CD, inspired by DartBrains and Naturalistic-Data courses.

**Implementation**: Update `_config.yml` to execute notebooks with caching, reorganize tutorials by complexity, and configure GitHub Actions to build docs and validate tutorials on every PR/push.

---

## Research Findings

### Current nltools State

**Documentation Setup**:
- Uses jupyter-book for documentation
- Currently has `execute_notebooks: off` in `_config.yml` ⚠️
- 18 tutorial notebooks across 3 directories:
  - `tutorials/basic/` (6 notebooks): Fundamentals
  - `tutorials/01_DataOperations/` (6 notebooks): Data manipulation
  - `tutorials/02_Analysis/` (6 notebooks): Statistical analyses
- Tutorials are **disabled** in `_toc.yml` (commented out)
- Test data embedded in tutorial directories (haxbydata, etc.)

**Current Test Suite**:
- 317 tests organized by "imperative shell, functional core"
- Unit tests validate components
- NO workflow/integration tests validating end-to-end user workflows
- NO automated validation that tutorials actually run

### Successful Patterns (pymer4)

**Key Features**:
- Tutorials organized by progressive complexity
- Four core tutorials: linear regression → categorical predictors → mixed models → GLMMs
- Built with jupyter-book
- CI/CD runs `docs-build` to execute notebooks
- Deployed to GitHub Pages automatically
- Uses pixi for environment management (we use uv)

**Why It Works**:
- Tutorials demonstrate real use cases users actually need
- Executing tutorials validates the full workflow, not just individual functions
- Documentation stays current (if code breaks, tutorials fail)
- Double value for effort (docs + tests)

### Scientific Python Ecosystem Insights

**Tools for Notebook Testing**:
1. **jupyter-book execution modes**:
   - `execute_notebooks: off` - Don't execute (current nltools)
   - `execute_notebooks: auto` - Execute notebooks missing outputs
   - `execute_notebooks: cache` - Execute and cache with jupyter-cache (RECOMMENDED)

2. **Additional testing tools** (future enhancements):
   - `nbmake`: pytest plugin for notebook execution (used by Dask, Quansight)
   - `papermill`: Parameterized notebook execution
   - `testbook`: Unit testing framework for notebooks

**Best Practices**:
- Use `execute_notebooks: cache` for intelligent caching (only re-run when code changes)
- Set `nb_execution_timeout: 300` for long-running analyses
- Configure `nb_execution_raise_on_error: true` to fail CI on notebook errors
- Clear outputs before committing to version control
- Use relative paths for data loading (jupyter-book sets cwd to notebook directory)

### Workflows from DartBrains Course

**Core neuroimaging workflows identified**:

1. **Data Loading & Preprocessing**:
   - Load Brain_Data from files
   - Smoothing operations
   - Motion covariate creation
   - Spike detection for outliers

2. **First-Level GLM**:
   - DesignMatrix creation with convolution
   - Adding motion covariates
   - Running regression with `regress()`
   - Extracting beta maps and t-statistics

3. **Group-Level Analysis**:
   - Computing one-sample t-tests with `ttest()`
   - Multiple comparison correction (FDR, FWE)
   - Thresholding statistical maps

4. **ROI Analysis**:
   - Creating ROIs from thresholded maps
   - Extracting ROI timeseries
   - Correlating activity with behavioral measures

5. **Advanced Methods**:
   - Connectivity analysis
   - Multivariate prediction (MVPA)
   - Representational Similarity Analysis (RSA)
   - Hyperalignment for multi-subject analysis

### Workflows from Naturalistic-Data Course

**Advanced workflows**:

1. **Intersubject Correlation (ISC)**
2. **Hyperalignment** (functional alignment across subjects)
3. **Shared Response Model (SRM)**
4. **Event segmentation with HMMs**
5. **Dynamic connectivity analysis**
6. **NLP-based encoding models**
7. **Automated stimulus annotation**

---

## Proposed Tutorial Organization

### Pedagogical Structure (Progressive Complexity)

**Tier 1: Fundamentals** (Getting Started)
- Tutorial focuses on: "I have neuroimaging data, how do I get started?"
- Minimal assumptions about background
- Build confidence with quick wins

**Tier 2: Common Workflows** (Building Skills)
- Tutorial focuses on: "I want to do X analysis, how do I do it?"
- Assumes familiarity with Tier 1 concepts
- Each tutorial = complete, realistic workflow

**Tier 3: Advanced Methods** (Mastery)
- Tutorial focuses on: "How do I do cutting-edge analyses?"
- Assumes expertise with Tier 1 & 2
- Demonstrates state-of-the-art techniques

---

## Recommended Tutorial Structure

```
docs/tutorials/
├── 01_fundamentals/          # Tier 1: Getting Started
│   ├── 01_brain_data_basics.ipynb
│   ├── 02_design_matrix_basics.ipynb
│   ├── 03_adjacency_basics.ipynb
│   └── 04_visualization_basics.ipynb
│
├── 02_workflows/              # Tier 2: Common Workflows
│   ├── 01_first_level_glm.ipynb
│   ├── 02_group_analysis.ipynb
│   ├── 03_roi_analysis.ipynb
│   ├── 04_mvpa_prediction.ipynb
│   ├── 05_searchlight_analysis.ipynb
│   └── 06_connectivity_analysis.ipynb
│
├── 03_advanced/               # Tier 3: Advanced Methods
│   ├── 01_hyperalignment.ipynb
│   ├── 02_shared_response_model.ipynb
│   ├── 03_representational_similarity.ipynb
│   ├── 04_encoding_models.ipynb
│   └── 05_naturalistic_data_analysis.ipynb
│
└── data/                      # Centralized test data
    ├── minimal_brain_data.h5
    ├── example_events.csv
    └── README.md
```

**Total**: ~15 comprehensive tutorials (down from 18, but each more complete)

**Why This Organization**:
- Clear progression from basics → workflows → advanced
- Each tutorial is self-contained (can be run independently)
- Tier 2 workflows map to real research questions users have
- Centralized data directory (don't duplicate across tutorials)

---

## Tutorial Content Specifications

### Every Tutorial Should Include:

1. **Clear Learning Objectives**
   ```markdown
   ## Learning Objectives
   By the end of this tutorial, you will be able to:
   - Load neuroimaging data into Brain_Data
   - Perform basic operations (indexing, arithmetic)
   - Visualize brain images
   ```

2. **Complete, Runnable Code**
   - No broken cells
   - No TODO comments (unless explaining future features)
   - All imports at the top
   - Clear variable names

3. **Pedagogical Narrative**
   - Explain WHY, not just WHAT
   - Connect to research questions
   - Highlight common pitfalls
   - Reference related concepts

4. **Validation/Sanity Checks**
   ```python
   # Sanity check: Data should be centered after standardization
   z_data = data.standardize()
   assert abs(z_data.mean().mean()) < 1e-10, "Data not properly centered!"
   print("✓ Data successfully standardized")
   ```

5. **Visual Outputs**
   - At least one plot per tutorial
   - Demonstrates that code worked correctly
   - Helps users verify their own results

6. **Clear Next Steps**
   ```markdown
   ## Next Steps
   - Try running this workflow on your own data
   - See Tutorial 02 for group-level analysis
   - Check out the API reference for advanced options
   ```

### Avoid in Tutorials:

- ❌ Overly simplistic toy examples (use realistic data)
- ❌ Incomplete workflows (show the full analysis)
- ❌ Commented-out code (clean it up or remove it)
- ❌ External dependencies not in requirements (keep it minimal)
- ❌ Long-running computations (>2 minutes, unless necessary)

---

## Priority Workflows to Implement

### Phase 1: Core Workflows (v0.6.0) - HIGH PRIORITY

**Target**: 5-6 essential tutorials that cover 80% of use cases

1. **Brain_Data Fundamentals** (already exists, needs update)
   - Loading data
   - Basic operations (indexing, arithmetic, statistics)
   - Visualization
   - File I/O
   - **Validates**: Brain_Data class core functionality

2. **First-Level GLM Workflow**
   - Load fMRI timeseries
   - Create DesignMatrix with HRF convolution
   - Add motion covariates
   - Run regression with `regress()`
   - Extract and visualize beta maps
   - **Validates**: DesignMatrix, regress(), convolution, masking

3. **Group-Level Analysis Workflow**
   - Load first-level betas for multiple subjects
   - Perform one-sample t-test with `ttest()`
   - Multiple comparison correction (FDR)
   - Threshold and visualize results
   - **Validates**: ttest(), threshold(), multiple comparisons

4. **ROI Analysis Workflow**
   - Create ROI from atlas or thresholded map
   - Extract ROI timeseries with `extract_roi()`
   - Correlate with behavioral measures
   - Visualize relationship
   - **Validates**: extract_roi(), masking, apply_mask()

5. **MVPA Prediction Workflow**
   - Load multi-subject data
   - Set outcome variable (Y)
   - Run cross-validated prediction with `predict()`
   - Compare algorithms (ridge, lasso, SVM)
   - Visualize weight maps and accuracy
   - **Validates**: predict(), cross-validation, algorithms

6. **DesignMatrix Basics** (already exists, needs update)
   - Creating design matrices
   - Adding columns (nuisance regressors)
   - Convolution with HRF
   - Visualization and diagnostics
   - **Validates**: DesignMatrix class functionality

### Phase 2: Advanced Workflows (v0.6.1) - MEDIUM PRIORITY

**Target**: 4-5 advanced tutorials for power users

7. **Hyperalignment Workflow**
   - Load multi-subject data
   - Run hyperalignment
   - Validate alignment quality
   - Apply to new data
   - **Validates**: Hyperaligner class

8. **Shared Response Model Workflow**
   - Load naturalistic data
   - Fit SRM
   - Extract shared and individual components
   - Reconstruct individual data
   - **Validates**: SRM class

9. **Searchlight Analysis Workflow**
   - Define searchlight parameters
   - Run searchlight with custom function
   - Visualize accuracy maps
   - **Validates**: searchlight functionality

10. **Adjacency & Connectivity Workflow**
    - Create Adjacency object
    - Compute correlations
    - Threshold and visualize networks
    - **Validates**: Adjacency class

### Phase 3: Specialized Workflows (v0.7.0+) - FUTURE

**Target**: Cutting-edge methods and integrations

11. **Representational Similarity Analysis**
12. **Encoding Models with NLP Features**
13. **Time-Varying Connectivity**
14. **Integration with Nilearn Pipelines**
15. **Custom Algorithms and Extensions**

---

## Technical Implementation

### Step 1: Configure jupyter-book Execution

**Update `docs/_config.yml`**:

```yaml
title: NL-Tools
author: Luke J. Chang, Eshin Jolly
logo: ""

# CRITICAL: Enable notebook execution with caching
execute:
  execute_notebooks: cache          # Execute and cache notebooks
  cache: "_build/.jupyter_cache"    # Cache location
  timeout: 300                      # 5 minute timeout per notebook
  run_in_temp: false                # Use notebook directory as cwd
  allow_errors: false               # Fail build on errors
  stderr_output: remove             # Clean up stderr in output

# Rest of config remains the same...
repository:
  url: https://github.com/cosanlab/nltools
  path_to_book: docs
  branch: master

html:
  use_issues_button: true
  use_repository_button: true
  home_page_in_navbar: false

sphinx:
  extra_extensions:
  - 'sphinx_design'
  - 'sphinx.ext.autodoc'
  - 'sphinx.ext.autosummary'
  - 'sphinx.ext.napoleon'
  - 'sphinx.ext.viewcode'
  - 'sphinx.ext.intersphinx'

  config:
    # ... (rest of sphinx config)

    # Add notebook execution config
    nb_execution_mode: cache
    nb_execution_raise_on_error: true
    nb_execution_timeout: 300
```

**Why This Configuration**:
- `execute_notebooks: cache` - Only re-runs notebooks when code cells change (fast!)
- `timeout: 300` - Prevents hanging on long computations (5 min max)
- `allow_errors: false` - Ensures tutorials must pass (test validation)
- `run_in_temp: false` - Notebooks run in their directory (relative paths work)
- `stderr_output: remove` - Cleaner output (removes warnings from display)

### Step 2: Update Table of Contents

**Update `docs/_toc.yml`**:

```yaml
format: jb-book
root: index

parts:
  # ========================================
  # TUTORIALS (Re-enabled!)
  # ========================================
  - caption: Getting Started
    chapters:
      - file: tutorials/01_fundamentals/01_brain_data_basics
      - file: tutorials/01_fundamentals/02_design_matrix_basics
      - file: tutorials/01_fundamentals/03_adjacency_basics
      - file: tutorials/01_fundamentals/04_visualization_basics

  - caption: Common Workflows
    chapters:
      - file: tutorials/02_workflows/01_first_level_glm
      - file: tutorials/02_workflows/02_group_analysis
      - file: tutorials/02_workflows/03_roi_analysis
      - file: tutorials/02_workflows/04_mvpa_prediction
      - file: tutorials/02_workflows/05_searchlight_analysis
      - file: tutorials/02_workflows/06_connectivity_analysis

  - caption: Advanced Methods
    chapters:
      - file: tutorials/03_advanced/01_hyperalignment
      - file: tutorials/03_advanced/02_shared_response_model
      - file: tutorials/03_advanced/03_representational_similarity
      - file: tutorials/03_advanced/04_encoding_models
      - file: tutorials/03_advanced/05_naturalistic_data_analysis

  # ========================================
  # API REFERENCE
  # ========================================
  - caption: Core Data Objects
    chapters:
      - file: api/data/brain_data
      - file: api/data/adjacency
      - file: api/data/design_matrix

  - caption: Statistical Models
    chapters:
      - file: api/models
      - file: api/algorithms

  # ... (rest of API sections)

  # ========================================
  # MIGRATION & DEVELOPMENT
  # ========================================
  - caption: Migration & Upgrading
    chapters:
      - file: migration-guide

  - caption: Development
    chapters:
      - file: contributing
      - file: performance
```

**Why This Organization**:
- Tutorials come FIRST (most users start here)
- Progressive complexity (fundamentals → workflows → advanced)
- API reference for deep dives
- Development docs at the end

### Step 3: Update GitHub Actions Workflow

**Update `.github/workflows/uv_workflow.yml`**:

```yaml
name: Tests, Build, Docs, & Deploy

on:
  push:
    branches:
      - master
      - main
      - uv-refactor
  pull_request:
    branches:
      - main
      - master
  release:
    types: [published]
  workflow_dispatch:
  schedule:
    - cron: "0 0 * * 0"  # Weekly tests

jobs:
  test:
    runs-on: ${{ matrix.os }}
    continue-on-error: ${{ matrix.experimental }}
    defaults:
      run:
        shell: bash -l {0}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']
        experimental: [false]
        include:
          - os: windows-latest
            python-version: '3.10'
            experimental: true

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          version: "0.7.6"
          enable-cache: true
          cache-dependency-glob: "uv.lock"
          python-version: ${{ matrix.python-version }}

      - name: Run unit tests
        run: |
          uv run pytest nltools/tests/

      # NEW: Build docs with tutorial execution
      # This validates all tutorials execute correctly
      - name: Build and validate tutorials
        run: |
          uv run jupyter-book build docs --builder html

      # Optional: Cache the jupyter-cache directory across builds
      - name: Cache Jupyter execution
        uses: actions/cache@v3
        with:
          path: docs/_build/.jupyter_cache
          key: jupyter-cache-${{ runner.os }}-${{ hashFiles('docs/tutorials/**/*.ipynb') }}
          restore-keys: |
            jupyter-cache-${{ runner.os }}-

      - name: Build package
        run: |
          uv build

      # Upload built docs as artifact (for inspection/debugging)
      - name: Upload built docs
        if: ${{ matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10' }}
        uses: actions/upload-artifact@v3
        with:
          name: built-docs
          path: docs/_build/html/

      # Deploy to GitHub Pages (only on main branch push)
      - name: Setup Pages
        if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' && matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10' }}
        uses: actions/configure-pages@v3

      - name: Upload to Pages
        if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' && matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10' }}
        uses: actions/upload-pages-artifact@v3
        with:
          path: "./docs/_build/html"

      - name: Deploy to GitHub Pages
        if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' && matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10' }}
        id: deployment
        uses: actions/deploy-pages@v4

      - name: Upload to PyPI
        if: ${{ github.event_name == 'release' && matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10' }}
        uses: pypa/gh-action-pypi-publish@v1.12.4
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
          verbose: true
          skip-existing: true
```

**Key Changes**:
1. **Tutorial validation**: `jupyter-book build docs` now executes all notebooks
2. **Caching**: Cache jupyter-cache directory to speed up subsequent builds
3. **Artifact upload**: Save built docs for debugging
4. **GitHub Pages**: Automatic deployment on main branch push
5. **Single OS for deployment**: Only deploy from ubuntu-latest + Python 3.10

**Why This Works**:
- Every PR/push validates that tutorials execute correctly
- If a code change breaks a tutorial, CI fails (catches regressions)
- Built docs are automatically deployed to GitHub Pages
- Caching makes subsequent builds fast

### Step 4: Data Management Strategy

**Challenge**: Tutorials need realistic data, but shouldn't bloat the repository

**Solution**: Hybrid approach

1. **Small test data** (committed to repo):
   - `docs/tutorials/data/minimal_brain_data.h5` (~1-5 MB)
   - `docs/tutorials/data/example_events.csv` (< 1 MB)
   - Sufficient for basic tutorials and quick workflows

2. **Larger datasets** (downloaded on first run):
   - Use `nltools.datasets.fetch_*()` functions
   - Cached in user's home directory (`~/.nltools/data/`)
   - Used for realistic, complete workflows

3. **Example data pattern**:
   ```python
   # At start of tutorial
   from nltools.datasets import fetch_pain

   # This downloads ~10 MB data on first run, cached thereafter
   data = fetch_pain()
   ```

**Benefits**:
- Repository stays lean
- Tutorials use realistic data
- Users can run tutorials locally without downloading GB of data
- CI caches downloaded data across builds (fast)

### Step 5: Tutorial Development Workflow

**For each new tutorial**:

1. **Create notebook skeleton**:
   ```bash
   # Create new tutorial file
   touch docs/tutorials/02_workflows/XX_new_workflow.ipynb
   ```

2. **Write pedagogical content**:
   - Learning objectives
   - Narrative explanations
   - Complete, working code
   - Validation/sanity checks
   - Visualizations
   - Summary and next steps

3. **Test locally**:
   ```bash
   # Build just the new tutorial (fast iteration)
   uv run jupyter-book build docs --path-output _build_test docs/tutorials/02_workflows/XX_new_workflow.ipynb

   # Or build all docs
   uv run jupyter-book build docs
   ```

4. **Clear outputs before committing**:
   ```bash
   # jupyter-book can clear outputs automatically
   uv run jupyter-book clean docs --all

   # Or manually with jupyter nbconvert
   jupyter nbconvert --clear-output --inplace docs/tutorials/02_workflows/XX_new_workflow.ipynb
   ```

5. **Commit and push**:
   - GitHub Actions will execute notebook and validate
   - Review rendered output in built docs artifact

6. **Iterate based on CI feedback**

---

## Migration Strategy

### Phase 1: Foundation (Weeks 1-2)

**Goals**:
- Update configuration files
- Reorganize existing tutorials
- Validate current tutorials execute

**Tasks**:
1. Update `_config.yml` to enable `execute_notebooks: cache`
2. Create new directory structure: `01_fundamentals/`, `02_workflows/`, `03_advanced/`
3. Move existing tutorials to appropriate directories
4. Update `_toc.yml` with new organization
5. Clear all notebook outputs: `jupyter-book clean docs --all`
6. Test local build: `uv run jupyter-book build docs`
7. Fix any execution errors in existing notebooks
8. Update GitHub Actions workflow
9. Test CI build

**Deliverables**:
- ✅ jupyter-book configured to execute notebooks
- ✅ Existing tutorials reorganized and validating in CI
- ✅ Automated deployment to GitHub Pages working

### Phase 2: Core Workflows (Weeks 3-6)

**Goals**:
- Rewrite/enhance 5-6 core workflow tutorials
- Ensure comprehensive coverage of common use cases

**Priority Order**:
1. Brain_Data Fundamentals (enhance existing)
2. First-Level GLM Workflow (NEW)
3. Group-Level Analysis Workflow (NEW)
4. ROI Analysis Workflow (NEW)
5. MVPA Prediction Workflow (enhance existing)
6. DesignMatrix Basics (enhance existing)

**For each tutorial**:
- [ ] Write learning objectives
- [ ] Develop complete, narrative workflow
- [ ] Add validation/sanity checks
- [ ] Include visualizations
- [ ] Test locally
- [ ] Get feedback from lab members (optional)
- [ ] Merge to main

**Deliverables**:
- ✅ 5-6 comprehensive workflow tutorials
- ✅ Documentation that teaches AND validates
- ✅ Behavioral testing coverage for 80% of use cases

### Phase 3: Advanced Methods (Weeks 7-10)

**Goals**:
- Develop 4-5 advanced tutorials
- Cover cutting-edge methods

**Priority Order**:
1. Hyperalignment Workflow
2. Shared Response Model Workflow
3. Searchlight Analysis Workflow
4. Adjacency & Connectivity Workflow
5. Representational Similarity Analysis (optional)

**Deliverables**:
- ✅ Advanced tutorials for power users
- ✅ Complete documentation coverage

### Phase 4: Polish & Maintenance (Ongoing)

**Goals**:
- Keep tutorials current
- Address user feedback
- Add new tutorials for new features

**Ongoing Tasks**:
- Monitor CI for tutorial failures (indicates breaking changes)
- Update tutorials when API changes
- Add new tutorials for new features
- Respond to user questions with new tutorials

---

## Success Metrics

### Immediate (v0.6.0)

- [ ] All existing tutorials execute without errors in CI
- [ ] jupyter-book configured to execute with caching
- [ ] GitHub Actions successfully builds and deploys docs
- [ ] 5-6 core workflow tutorials complete and validating

### Medium-term (v0.6.1)

- [ ] 10+ comprehensive tutorials covering common use cases
- [ ] Tutorial execution time < 10 minutes total (with caching)
- [ ] Zero tutorial failures in CI over 1 month
- [ ] Positive user feedback on tutorial quality

### Long-term (v0.7.0+)

- [ ] 15+ tutorials covering fundamentals → advanced methods
- [ ] Tutorials referenced in published papers (evidence of use)
- [ ] Tutorial-first development: new features include tutorial
- [ ] Community contributions of tutorials

---

## Benefits Summary

### For Users

✅ **Learn by doing**: Complete, realistic workflows they can adapt
✅ **Always current**: Executed in CI ensures docs stay up-to-date
✅ **Copy-paste friendly**: All code is tested and ready to use
✅ **Progressive complexity**: Easy to find appropriate tutorial for skill level

### For Developers

✅ **Behavioral testing**: Validates end-to-end workflows, not just units
✅ **Catches regressions**: Breaking changes fail tutorial execution
✅ **Documentation as tests**: Double value for effort
✅ **Living documentation**: Docs can't get out of sync with code

### For Project

✅ **Higher quality**: Ensures workflows actually work as intended
✅ **Better onboarding**: New users can get productive faster
✅ **Reduced support**: Comprehensive tutorials answer common questions
✅ **Improved research**: Better tools → better science

---

## Implementation Checklist

### Configuration Updates

- [ ] Update `docs/_config.yml` to enable notebook execution
- [ ] Update `docs/_toc.yml` with new tutorial organization
- [ ] Update `.github/workflows/uv_workflow.yml` for tutorial validation
- [ ] Add jupyter-cache configuration
- [ ] Configure GitHub Pages deployment

### Tutorial Reorganization

- [ ] Create `docs/tutorials/01_fundamentals/` directory
- [ ] Create `docs/tutorials/02_workflows/` directory
- [ ] Create `docs/tutorials/03_advanced/` directory
- [ ] Create `docs/tutorials/data/` directory for test data
- [ ] Move existing tutorials to appropriate directories
- [ ] Clear all notebook outputs
- [ ] Test local build

### Core Workflow Development (Priority 1)

- [ ] Tutorial 01: Brain_Data Fundamentals (enhance existing)
- [ ] Tutorial 02: First-Level GLM Workflow (NEW)
- [ ] Tutorial 03: Group-Level Analysis Workflow (NEW)
- [ ] Tutorial 04: ROI Analysis Workflow (NEW)
- [ ] Tutorial 05: MVPA Prediction Workflow (enhance existing)
- [ ] Tutorial 06: DesignMatrix Basics (enhance existing)

### Advanced Workflow Development (Priority 2)

- [ ] Tutorial 07: Hyperalignment Workflow
- [ ] Tutorial 08: Shared Response Model Workflow
- [ ] Tutorial 09: Searchlight Analysis Workflow
- [ ] Tutorial 10: Adjacency & Connectivity Workflow

### Documentation Updates

- [ ] Update CLAUDE.md with tutorial testing guidelines
- [ ] Update README.md with link to tutorials
- [ ] Add tutorial development guide to contributing.md
- [ ] Update refactor-plan.md with tutorial testing strategy
- [ ] Update refactor-todos.md with tutorial tasks

### Validation & Testing

- [ ] Verify all tutorials execute locally
- [ ] Verify CI builds docs successfully
- [ ] Verify GitHub Pages deployment works
- [ ] Test caching behavior (fast subsequent builds)
- [ ] Validate on multiple platforms (Linux, macOS)

---

## Resources & References

### Tools

- **Jupyter Book**: https://jupyterbook.org/
- **Jupyter Cache**: https://jupyter-cache.readthedocs.io/
- **nbmake** (future): https://github.com/treebeardtech/nbmake
- **GitHub Actions**: https://docs.github.com/en/actions

### Documentation

- **Jupyter Book Execution**: https://jupyterbook.org/content/execute.html
- **GitHub Pages Deployment**: https://jupyterbook.org/publish/gh-pages.html
- **Notebook Best Practices**: https://github.com/chrisvoncsefalvay/jupyter-best-practices

### Inspiration

- **pymer4**: https://eshinjolly.com/pymer4/
- **DartBrains**: https://dartbrains.org/
- **Naturalistic-Data**: https://naturalistic-data.org/
- **nilearn examples**: https://nilearn.github.io/stable/auto_examples/

---

## Next Steps

**Immediate Action Items**:

1. **Get approval** from Eshin on this plan
2. **Create feature branch**: `git checkout -b tutorial-testing-setup`
3. **Update configuration files** (Phase 1, Week 1)
4. **Test local build** to validate approach
5. **Update GitHub Actions** and verify CI
6. **Begin core workflow development** (Phase 2)

**Questions for Discussion**:

1. **Timeline**: Should tutorial development be part of v0.6.0 refactor or separate v0.6.1 release?
2. **Scope**: Start with 5-6 core workflows or go bigger initially?
3. **Review process**: Should tutorials be reviewed by lab members before merging?
4. **Data strategy**: Commit small test data to repo or always use fetch functions?
5. **Deployment**: GitHub Pages or ReadTheDocs or both?

---

**Last Updated**: 2025-10-29
**Status**: Proposal - Awaiting Approval
**Next Review**: After Phase 1 completion
