# nltools Tutorials

Comprehensive, hands-on tutorials for neuroimaging analysis with nltools.

## Organization

Tutorials are organized by complexity and use case:

### 📚 Getting Started (Fundamentals)

Learn the core data structures and basic operations:

1. **BrainData Basics** - Loading, manipulating, and visualizing neuroimaging data
2. **DesignMatrix Basics** - Creating experimental designs with HRF convolution
3. **Adjacency Basics** - Working with connectivity and similarity matrices

**Target audience**: New users, basic Python knowledge assumed

---

### 🔬 Common Workflows

Complete, realistic analysis pipelines:

1. **First-Level GLM Analysis** - Single-subject General Linear Model
2. **Group-Level Analysis** - Multi-subject inference and statistics
3. **ROI Analysis** - Region-based analyses and brain-behavior correlations
4. **MVPA Prediction** *(planned)* - Multivariate pattern analysis
5. **Connectivity Analysis** *(planned)* - Functional connectivity workflows
6. **Searchlight Analysis** *(planned)* - Whole-brain decoding

**Target audience**: Users with fMRI analysis experience

---

### 🚀 Advanced Methods

Cutting-edge techniques for specialized analyses:

1. **Hyperalignment** - Functional alignment across subjects
2. **Shared Response Model** *(planned)* - Multi-subject dimensionality reduction
3. **Representational Similarity** *(planned)* - RSA and pattern analysis
4. **Encoding Models** *(planned)* - Natural stimulus modeling

**Target audience**: Advanced users, method developers

---

## Tutorial Features

Each tutorial includes:
- ✅ **Clear learning objectives**
- ✅ **Complete, runnable code**
- ✅ **Pedagogical explanations** (why, not just what)
- ✅ **Sanity checks and validation**
- ✅ **Visualizations**
- ✅ **Best practices and pitfalls**
- ✅ **Next steps and related tutorials**

## Using Tutorials

### Format

Tutorials are written in **Markdown with code cells** for easy version control and editing. Jupyter-book can execute these directly when `execute_notebooks` is enabled.

### Data

All tutorials use `nltools.datasets.fetch_*()` functions to download data on-demand:
- **Pain dataset**: `fetch_pain()` - Main dataset for most tutorials
- Data cached locally in `~/.nltools/data/`
- No large files in repository

### Running Locally

```bash
# Install nltools with tutorial dependencies
uv sync

# Run individual tutorial as script (when ready)
uv run python -c "exec(open('tutorials/01_fundamentals/01_brain_data_basics.md').read())"

# Build all docs with jupyter-book
uv run jupyter-book build docs
```

## Contributing Tutorials

See `docs/contributing.md` for guidelines on:
- Tutorial structure and style
- Code quality standards
- Review process

## Roadmap

### Phase 1: Core Workflows ✅ (In Progress)
- [x] BrainData fundamentals
- [x] DesignMatrix fundamentals
- [x] Adjacency fundamentals
- [x] First-level GLM workflow
- [x] Group-level analysis
- [x] ROI analysis
- [ ] MVPA prediction workflow

### Phase 2: Advanced Methods (v0.6.1)
- [x] Hyperalignment
- [ ] Shared Response Model
- [ ] Representational Similarity Analysis
- [ ] Encoding models

### Phase 3: Specialized Topics (v0.7.0+)
- [ ] Time-varying connectivity
- [ ] Naturalistic data analysis
- [ ] Custom algorithms and extensions
- [ ] Integration with other libraries

## Tutorial Index

### By Topic

**GLM & Univariate**:
- First-Level GLM (Workflows #1)
- Group-Level Analysis (Workflows #2)

**ROI & Connectivity**:
- ROI Analysis (Workflows #3)
- Adjacency Basics (Fundamentals #3)

**Multivariate & Patterns**:
- MVPA Prediction (planned)
- Hyperalignment (Advanced #1)

**Fundamentals**:
- BrainData (Fundamentals #1)
- DesignMatrix (Fundamentals #2)
- Adjacency (Fundamentals #3)

### By Difficulty

**Beginner**: Fundamentals #1-3
**Intermediate**: Workflows #1-3
**Advanced**: Advanced #1+

---

**Questions?** See [nltools documentation](https://nltools.org) or [open an issue](https://github.com/cosanlab/nltools/issues).
