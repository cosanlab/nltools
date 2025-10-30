# API Documentation Improvement Plan for nltools v0.6.0

**Created:** 2025-10-29
**Status:** Ready for Implementation
**Priority:** 2.12 (High Priority - Pre-Release Blocker)
**Estimated Effort:** 12-16 hours

---

## Executive Summary

Transform nltools API documentation from inconsistent stubs into deployment-quality, well-organized, easily navigable reference documentation using Jupyter Book's autodoc capabilities.

**Key Goals:**
1. **Consistent quality** across all API modules
2. **Easy navigation** with simple, modular categorization
3. **Professional formatting** suitable for deployment
4. **Google-style docstrings** with Examples sections for auto-generation
5. **No usage examples** in API reference (reserved for tutorials)

---

## Current State Analysis

### Strengths ✅
- **Solid infrastructure:** Jupyter Book with autodoc, napoleon, autosummary configured
- **Napoleon extension:** Supports both NumPy and Google style docstrings
- **Two exemplar modules:** `algorithms.md` (288 lines) and `models.md` (335 lines) show comprehensive approach
- **Working configuration:** viewcode, line numbers, GitHub integration enabled

### Critical Gaps ⚠️

| Issue | Impact | Modules Affected |
|-------|--------|------------------|
| Minimal stubs | Poor user experience | 10 of 12 API files (< 20 lines each) |
| Missing Examples sections | Auto-docs lack usage clarity | All modules |
| Inconsistent docstring style | Mixed NumPy/Google styles | **All source files need conversion to Google style** |
| Inconsistent depth | Confusing navigation | All API docs |
| No categorization | Hard to find relevant docs | API reference structure |
| No cross-references | Isolated documentation | All modules |
| Templates directory missing | Config references non-existent path | Build system |

### Documentation Coverage Map

| Module | Lines | Status | Examples? | Priority |
|--------|-------|--------|-----------|----------|
| `algorithms.md` | 288 | ⭐ Comprehensive | ✅ Yes | Maintain quality |
| `models.md` | 335 | ⭐ Comprehensive | ✅ Yes | Maintain quality |
| `backends.md` | 146 | 🟡 Moderate | ✅ Yes | Polish |
| `prefs.md` | 54 | 🟡 Moderate | ✅ Yes | Polish |
| `data.md` | 19 | 🔴 Minimal | ❌ No | **Critical** |
| `stats.md` | 8 | 🔴 Minimal | ❌ No | **Critical** |
| `utils.md` | 8 | 🔴 Minimal | ❌ No | **Critical** |
| `analysis.md` | 8 | 🔴 Minimal | ❌ No | Enhance |
| `crossval.md` | 11 | 🔴 Minimal | ❌ No | Enhance |
| `dataset.md` | 8 | 🔴 Minimal | ❌ No | Enhance |
| `filereader.md` | 8 | 🔴 Minimal | ❌ No | Enhance |
| `mask.md` | 8 | 🔴 Minimal | ❌ No | Enhance |

---

## Research Findings: Jupyter Book Best Practices

### Key Insights from Web Search

**1. Napoleon Extension for Google Docstrings**
- Already configured in `_config.yml` (needs update for Google style)
- Automatically parses Google-style docstrings
- Converts to reStructuredText for Sphinx rendering
- **Action required:** Enable Google style, disable NumPy style in config

**2. Autodoc Integration with MyST Markdown**
- Use `{eval-rst}` directive to embed autodoc in `.md` files
- Cannot use autodoc directly in Markdown syntax
- Current approach is correct

**3. Autosummary for Summary Tables**
- `autosummary_generate: true` creates stub files automatically
- Can create tables of functions/classes with `:toctree:` option
- Useful for modules with many functions

**4. Best Practice: Separate API from Usage Examples**
- API reference should be concise and auto-generated
- Usage examples belong in tutorials (separate section)
- Keep API docs focused on signatures, parameters, returns

**5. Examples in Docstrings**
- Adding `Examples` sections to docstrings improves auto-generated docs
- Should be minimal (1-2 lines showing basic usage)
- More complex examples go in tutorials

**6. Module Organization**
- Group by category (Data Objects, Models, Stats, Utilities)
- Use Jupyter Book "parts" in `_toc.yml` for sections
- **Keep simple and modular** for easy rearrangement
- Clear navigation hierarchy improves discoverability

---

## Proposed Improvements

### Phase 1: Docstring Conversion & Enhancement (5-7 hours)

**Objective:** Convert all docstrings to Google style and add Examples sections for better auto-generated docs

**CRITICAL:** All docstrings must use Google style, not NumPy style.

**Scope:** Focus on most-used public methods/functions

**Priority Order:**
1. **Critical** (high-usage, core functionality):
   - `Brain_Data` class: `__init__`, `fit`, `predict`, `apply_mask`, `regress`, `smooth`, `threshold`, `filter`, `standardize`, `extract_roi`, `compute_contrasts`
   - `Adjacency` class: `__init__`, `distance`, `similarity`, `plot`
   - `Design_Matrix` class: `__init__`, `append`, `convolve`, `add_poly`, `add_dct_basis`
   - `stats.py`: `regress`, `align`, `one_sample_permutation`, `correlation_permutation`, `isc`, `isc_group`
   - `utils.py`: `get_resource_path`, `check_brain_data`

2. **Important** (frequently used):
   - `mask.py`: `create_sphere`, `expand_mask`, `collapse_mask`
   - `cross_validation.py`: `KFoldStratified`, `set_cv`
   - `datasets.py`: `fetch_pain`, `fetch_emotion`
   - `analysis.py`: `Roc` class methods

3. **Nice-to-have** (less frequently used but should be complete):
   - `file_reader.py`: `onsets_to_dm`
   - Other utility functions

**Docstring Enhancement Guidelines (Google Style):**

```python
# BEFORE (NumPy style - WRONG)
def regress(self, X, mode='ols'):
    """Regress an X matrix on Brain_Data.

    Parameters
    ----------
    X : array-like
        Design matrix
    mode : str
        Type of regression (ols or gls)

    Returns
    -------
    dict
        Regression results
    """
    pass

# AFTER (Google style - CORRECT with Examples)
def regress(self, X, mode='ols'):
    """Regress a design matrix on Brain_Data.

    .. deprecated:: 0.6.0
        `regress()` is deprecated and will be removed in v0.7.0.
        Use `fit(model='glm')` instead.

    Args:
        X (array-like or Design_Matrix): Design matrix with shape
            (n_samples, n_predictors).
        mode ({'ols', 'gls'}, optional): Type of regression. 'ols' for
            ordinary least squares, 'gls' for generalized least squares.
            Defaults to 'ols'.

    Returns:
        dict: Dictionary containing:
            - 'beta': regression coefficients (Brain_Data)
            - 't': t-statistics (Brain_Data)
            - 'p': p-values (Brain_Data)
            - 'sigma': residual standard deviation (Brain_Data)
            - 'residual': residuals (Brain_Data)

    Example:
        >>> from nltools.datasets import fetch_pain
        >>> data = fetch_pain()
        >>> # Deprecated: use fit(model='glm') instead
        >>> results = data.regress(X, mode='ols')
        >>> results['beta'].plot()

    See Also:
        fit: Modern sklearn-style API for regression
    """
    pass
```

**Standards for Examples Sections (Google Style):**
- **Minimal:** 1-3 lines showing basic usage
- **Self-contained:** Use nltools datasets when possible
- **No output:** Don't show output in Examples (auto-generated docs may not execute)
- **Cross-reference:** Use `See Also:` section to link related functions
- **Deprecation:** Mark deprecated methods clearly with `.. deprecated::` directive
- **Section names:** Use `Example:` (singular) for single example, `Examples:` for multiple

**Implementation Approach:**
1. **Convert to Google style first** - This is the most time-consuming part
2. Start with `Brain_Data.__init__` and core methods
3. Test build after each class: `uv run jupyter-book build docs/`
4. Verify Examples render correctly in HTML output
5. Use git commits to track progress by module

**Google Style Quick Reference:**
```python
"""Short description (one line).

Extended description (optional, can be multiple paragraphs).

Args:
    param1 (type): Description of param1.
    param2 (type, optional): Description of param2. Defaults to value.

Returns:
    type: Description of return value.

Raises:
    ErrorType: Description of when this error is raised.

Example:
    >>> code example here

See Also:
    related_function: Description
"""
```

---

### Phase 2: API Reference Reorganization (3-4 hours)

**Objective:** Create simple, modular categorization that's easy to rearrange

**Design Principle:** Keep structure flat and modular - easy to add/remove/reorganize sections

**Current Structure (flat):**
```
API Reference
├── algorithms.md
├── analysis.md
├── backends.md
├── crossval.md
├── data.md
├── dataset.md
├── filereader.md
├── mask.md
├── models.md
├── prefs.md
├── stats.md
└── utils.md
```

**Proposed Structure (simple & modular):**

```
API Reference
├── Core Data Objects
│   ├── Brain_Data
│   ├── Adjacency
│   └── Design_Matrix
│
├── Statistical Models
│   ├── Ridge
│   └── HyperAlignment
│
├── Statistical Functions
│   └── Stats Module
│
├── Data Operations
│   ├── Masking
│   ├── File I/O
│   ├── Datasets
│   └── Cross-Validation
│
├── Computation
│   ├── Backends
│   ├── Algorithms
│   └── Analysis Tools
│
└── Utilities
    ├── Utils
    └── Preferences
```

**Modularity Notes:**
- Each category is a separate "part" in `_toc.yml`
- Easy to move files between categories
- Easy to add new categories
- No deep nesting - maximum 2 levels

**Implementation:**

**File:** `docs/_toc.yml` (Simple & Modular)

```yaml
format: jb-book
root: index

parts:
  # ========================================
  # API REFERENCE (Simple & Modular)
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

  - caption: Statistical Functions
    chapters:
      - file: api/stats
      - file: api/analysis

  - caption: Data Operations
    chapters:
      - file: api/mask
      - file: api/filereader
      - file: api/dataset
      - file: api/crossval

  - caption: Computation
    chapters:
      - file: api/backends

  - caption: Utilities
    chapters:
      - file: api/utils
      - file: api/prefs

  # ========================================
  # DEVELOPMENT
  # ========================================
  - caption: Development
    chapters:
      - file: contributing
      - file: performance
```

**Why This Is Modular:**
- Each `caption` is a standalone section
- No nested `sections` - just flat `chapters`
- Easy to reorder: just move the entire `caption` block
- Easy to rename: just change the `caption` text
- Easy to add: just add a new `caption` block
- No landing pages needed - simpler structure

**New Files to Create:**

1. **Split `docs/api/data.md` into three files:**
   - `docs/api/data/brain_data.md` (enhanced)
   - `docs/api/data/adjacency.md` (enhanced)
   - `docs/api/data/design_matrix.md` (enhanced)

**NO landing pages needed** - keep it simple!

**Benefits of Simple Structure:**
- **Flat hierarchy**: Easy to navigate
- **Reduced maintenance**: No landing pages to update
- **Easy reorganization**: Just move caption blocks
- **Clear sections**: Categories are self-explanatory
- **Future-proof**: Add new files without restructuring

---

### Phase 3: Content Enhancement (4-5 hours)

**Objective:** Expand minimal stub files to moderate depth

**Target:** 10 minimal files need enhancement

**Template for Enhanced API Pages:**

```markdown
# {Title}

{One-sentence description}

## Overview

{2-3 sentences explaining purpose and key use cases}

## Key Functions/Classes

{Brief summary of 2-3 most important items with one-line descriptions}

## Quick Start

```python
{Minimal 3-5 line example showing basic usage}
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.module
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {Related modules}
- {Related tutorials (when available)}
```

**Example: Enhanced `docs/api/mask.md`**

```markdown
# Masking Tools

Tools for creating, manipulating, and applying brain masks.

## Overview

Masking is fundamental in neuroimaging analysis for selecting brain regions
and reducing data dimensionality. These tools extend nilearn's masking
capabilities with convenient functions for sphere creation, mask expansion,
and ROI manipulation.

## Key Functions

**`create_sphere`** - Create spherical ROI masks from coordinates
**`expand_mask`** - Convert masked vector back to full brain image
**`collapse_mask`** - Extract data from image using mask

## Quick Start

```python
from nltools.mask import create_sphere
from nltools.prefs import MNI_Template

# Create 10mm sphere at coordinates
mask = create_sphere([0, 0, 0], radius=10, mask=MNI_Template.mask)
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.mask
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - Apply masks to Brain_Data objects
- {doc}`prefs` - MNI template preferences
```

**Implementation Priority:**

1. **Phase 3A: Critical modules** (2-3 hours)
   - `data.md` → Split into 3 enhanced files
   - `stats.md` → Enhanced overview
   - `utils.md` → Enhanced overview

2. **Phase 3B: Important modules** (1-2 hours)
   - `mask.md`
   - `crossval.md`
   - `dataset.md`

3. **Phase 3C: Supporting modules** (1 hour)
   - `analysis.md`
   - `filereader.md`

---

### Phase 4: Configuration & Infrastructure (1-2 hours)

**Objective:** Fix configuration issues and optimize build

**Tasks:**

**1. Create Templates Directory**

Current config references `templates_path: templates` but directory doesn't exist.

```bash
mkdir -p docs/_templates
```

**File:** `docs/_templates/autosummary/class.rst`

```rst
{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}
```

**File:** `docs/_templates/autosummary/module.rst`

```rst
{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
```

**2. Enhance `_config.yml` with Better Autodoc Options (GOOGLE STYLE)**

```yaml
sphinx:
  extra_extensions:
    - 'sphinx_design'
    - 'sphinx.ext.autodoc'
    - 'sphinx.ext.autosummary'
    - 'sphinx.ext.napoleon'
    - 'sphinx.ext.viewcode'
    - 'sphinx.ext.intersphinx'          # NEW: Cross-reference external docs

  config:
    # Autosummary
    autosummary_generate: true
    autosummary_generate_overwrite: true
    autosummary_imported_members: false  # NEW: Don't document imports

    # Autodoc
    autodoc_default_options:             # NEW: Default options for autodoc
      members: true
      member-order: 'bysource'
      special-members: '__init__'
      undoc-members: true
      exclude-members: '__weakref__'

    # Napoleon (GOOGLE STYLE - CRITICAL)
    napoleon_google_docstring: true      # CHANGED: Enable Google style
    napoleon_numpy_docstring: false      # CHANGED: Disable NumPy style
    napoleon_include_init_with_doc: true
    napoleon_include_private_with_doc: false
    napoleon_include_special_with_doc: true
    napoleon_use_admonition_for_examples: true  # NEW: Nice formatting
    napoleon_use_admonition_for_notes: true
    napoleon_use_admonition_for_references: true
    napoleon_use_ivar: false
    napoleon_use_param: true
    napoleon_use_rtype: true
    napoleon_preprocess_types: true      # NEW: Better type formatting

    # Display
    add_function_parentheses: true
    add_module_names: false              # CHANGED: Shorter names (cleaner)

    # Viewcode
    viewcode_line_numbers: true
    viewcode_follow_imported_members: false  # NEW: Don't follow imports

    # Intersphinx (cross-references)
    intersphinx_mapping:
      python: ['https://docs.python.org/3', null]
      numpy: ['https://numpy.org/doc/stable', null]
      scipy: ['https://docs.scipy.org/doc/scipy', null]
      pandas: ['https://pandas.pydata.org/docs', null]
      sklearn: ['https://scikit-learn.org/stable', null]
      nilearn: ['https://nilearn.github.io/stable', null]

    # Templates
    templates_path: ['_templates']

    # Warnings
    suppress_warnings: ["etoc.toctree"]
```

**3. Add Cross-References in Docstrings (Google Style)**

Use Sphinx cross-reference syntax in Google-style docstrings:

```python
def align(data, method='procrustes', n_iter=2):
    """Align subjects using functional alignment.

    Args:
        data (list of Brain_Data): Subject data to align.
        method ({'procrustes', 'srm', 'deterministic_srm'}): Alignment method.
        n_iter (int): Number of iterations for Procrustes alignment.

    Returns:
        list of Brain_Data: Aligned subject data.

    Note:
        For Procrustes alignment with n_iter > 1, internally uses
        :class:`nltools.algorithms.HyperAlignment` for iterative refinement.

        For SRM methods, wraps :class:`nilearn.decomposition.DictLearning`.

    Example:
        >>> from nltools.stats import align
        >>> aligned = align(subjects, method='procrustes', n_iter=2)

    See Also:
        HyperAlignment: Class-based Procrustes alignment
        SRM: Shared Response Model alignment
        DetSRM: Deterministic Shared Response Model
    """
    pass
```

**4. Update `.gitignore` for Documentation Builds**

```
# Documentation
docs/_build/
docs/_templates/autosummary/
```

**5. Create Documentation Build Script**

**File:** `scripts/build_docs.sh`

```bash
#!/bin/bash
# Build nltools documentation with Jupyter Book

set -e  # Exit on error

echo "Building nltools documentation..."

# Navigate to project root
cd "$(dirname "$0")/.."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf docs/_build

# Build documentation
echo "Building Jupyter Book..."
uv run jupyter-book build docs/

# Check for warnings
echo ""
echo "Build complete! Check for warnings above."
echo "Open docs/_build/html/index.html to view documentation."
```

---

### Phase 5: Quality Assurance (1-2 hours)

**Objective:** Verify all documentation builds correctly and looks professional

**Checklist:**

**Build Quality:**
- [ ] All pages build without errors
- [ ] No Sphinx warnings (except expected suppressed ones)
- [ ] All autodoc directives resolve correctly
- [ ] Examples sections render with proper formatting
- [ ] Cross-references link correctly (internal and external)
- [ ] Source code links work from all documented items

**Navigation:**
- [ ] Table of contents renders correctly
- [ ] Section hierarchy is clear and logical
- [ ] Landing pages provide clear orientation
- [ ] Breadcrumbs show correct path
- [ ] Search functionality works

**Content Quality:**
- [ ] All public classes/functions documented
- [ ] Parameter descriptions are clear and complete
- [ ] Return values are documented
- [ ] Examples are minimal and self-contained
- [ ] Deprecation warnings are clear with migration path
- [ ] "See Also" sections link related functionality

**Visual Quality:**
- [ ] Code blocks formatted correctly
- [ ] Admonitions (Note, Warning, etc.) render properly
- [ ] Tables display correctly
- [ ] Math equations render (if any)
- [ ] No broken formatting or layout issues

**Accessibility:**
- [ ] Heading hierarchy is semantic (h1 → h2 → h3)
- [ ] Links have descriptive text
- [ ] Code examples are readable
- [ ] Color contrast is sufficient (Jupyter Book default theme)

**Testing Procedure:**

```bash
# Build and check
uv run jupyter-book build docs/

# Review build output for warnings
grep -i "warning" docs/_build/.jupyter_cache/*/jupyter_execute/*.log

# Open in browser
open docs/_build/html/index.html

# Test navigation
# - Click through all sections
# - Test search with common terms: "Brain_Data", "Ridge", "regress"
# - Verify external links (GitHub, nilearn docs)
# - Check mobile responsiveness (browser dev tools)

# Validate links
uv run jupyter-book build docs/ --builder linkcheck
```

---

## Implementation Schedule

**Total Effort: 12-16 hours**

| Phase | Focus | Hours | Priority |
|-------|-------|-------|----------|
| **1** | Docstring conversion to Google style + Examples | 5-7 | **Critical** |
| **2** | API reorganization (simple & modular) | 2-3 | High |
| **3** | Content enhancement | 4-5 | High |
| **4** | Configuration fixes (enable Google style) | 1-2 | **Critical** |
| **5** | Quality assurance | 1-2 | High |

**Recommended Approach:**

**Week 1: Foundation (6-9 hours)**
- **Phase 4 FIRST:** Enable Google style in config (30 min) - DO THIS FIRST!
- **Phase 1:** Convert docstrings to Google style for critical classes
  - `Brain_Data`, `Adjacency`, `Design_Matrix`
  - Core `stats.py` functions

**Week 2: Structure & Content (6-8 hours)**
- **Phase 2:** Reorganize API structure (simple & modular, no landing pages)
- **Phase 3:** Enhance minimal stub files
- Continue Phase 1 for remaining modules

**Week 3: Polish (1-2 hours)**
- **Phase 5:** QA testing, build verification, fixes

**Critical Path:**
1. Phase 4 must be done FIRST (config)
2. Phase 1 is the longest - convert docstrings progressively
3. Phases 2 & 3 can overlap once config is set

---

## Success Criteria

### Must Have ✅
- [ ] **All docstrings converted to Google style** (critical!)
- [ ] Configuration updated to enable Google style docstrings
- [ ] All 12 API modules have enhanced content (> 50 lines each)
- [ ] All public methods in `Brain_Data`, `Adjacency`, `Design_Matrix` have Examples sections
- [ ] API organized into 6 simple categories (no landing pages)
- [ ] Documentation builds with zero errors and < 5 warnings
- [ ] Navigation is intuitive and consistent
- [ ] All pages follow consistent formatting template

### Nice to Have ⭐
- [ ] All functions in `stats.py` have Examples sections
- [ ] Cross-references work between nltools and nilearn docs
- [ ] Build script automates common tasks
- [ ] Link checker passes with zero broken links

### Out of Scope (for v0.6.0) 🚫
- Tutorials rewrite (planned separately)
- Usage examples in API reference (belongs in tutorials)
- Comprehensive docstring audit of private functions
- **Plotting module documentation** (skipped per user request)
- **Simulator module documentation** (skipped per user request)
- Video tutorials or interactive demos

---

## Docstring Quality Standards (Google Style)

**Mandatory Elements:**
1. **Short description** (one line)
2. **Args** section with types and descriptions
3. **Returns** section with types and descriptions
4. **Example** or **Examples** section (1-3 lines)

**Recommended Elements:**
5. **Extended description** (2-3 sentences for complex functions)
6. **See Also:** section (cross-references)
7. **Note:** section (mathematical details, caveats)
8. **References:** section (academic papers, URLs)

**Deprecation Handling (Google Style):**
```python
"""Function description.

.. deprecated:: 0.6.0
    `old_function()` is deprecated and will be removed in v0.7.0.
    Use `new_function()` instead.

Args:
    param (type): Description.
```

**Style Compliance:**
- Follow Google docstring convention exactly
- Use `:class:`, `:func:`, `:meth:` for cross-references in descriptions
- Examples should use `>>>` for doctests (but not executed)
- Indent consistently (4 spaces)
- Section names end with colon: `Args:`, `Returns:`, `Example:`, `See Also:`

**Google Style Sections:**
- `Args:` - Function/method arguments
- `Returns:` - Return value(s)
- `Yields:` - For generators
- `Raises:` - Exceptions that may be raised
- `Example:` or `Examples:` - Usage examples
- `Note:` - Additional notes
- `See Also:` - Related functions/classes
- `References:` - Citations

---

## Risk Mitigation

**Risk:** Docstring changes break existing code
**Mitigation:** Docstrings are documentation only; won't affect runtime

**Risk:** Large restructure breaks existing links
**Mitigation:** Keep old API URLs working with redirects (Jupyter Book supports)

**Risk:** Examples in docstrings fail
**Mitigation:** Examples are not executed by default; use simple, verified patterns

**Risk:** Time overruns due to scope creep
**Mitigation:** Strict prioritization; Phase 3C and Phase 5 nice-to-haves can be deferred

**Risk:** Configuration changes break build
**Mitigation:** Test after each change; keep git history clean for easy rollback

---

## Related Documents

- **`refactor-plan.md`**: Overall v0.6.0 strategic vision
- **`refactor-todos.md`**: Task tracking (this is Priority 2.12)
- **`docs/migration-guide.md`**: User-facing migration guide
- **`TODO_TRACKER.md`**: Tutorial tracking (tutorials out of scope)
- **`contributing.md`**: Build and test commands

---

## Next Steps

1. **Review and approve** this plan with Eshin
2. **Create feature branch**: `git checkout -b api-docs-v0.6.0`
3. **Phase 1**: Start with `Brain_Data` docstring enhancements
4. **Iterate**: Build and test after each module
5. **Commit frequently**: One commit per module or logical group
6. **Final review**: Complete QA checklist before merge

---

*Last updated: 2025-10-29*
*Status: Ready for implementation*
*Estimated completion: 2-3 weeks with focused effort*
