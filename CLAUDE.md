# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses `uv` for dependency and environment management:

**Testing**:
- Run all tests: `uv run pytest`
- Run specific test file: `uv run pytest nltools/tests/test_brain_data.py`
- Run specific test: `uv run pytest nltools/tests/test_brain_data.py::test_method_name`

**Linting**:
- Check code: `uv run ruff check`
- Fix issues: `uv run ruff check --fix`
- Ignored rules: W292, E501, E731, E741

**Documentation**:
- Build docs: `uv run jupyter-book build docs/`

**Package Management**:
- Add dependency: `uv add packagename`
- Add dev dependency: `uv add --dev packagename`
- Build package: `uv build`

## Core Architecture

nltools is a neuroimaging analysis package built around three main data classes that work together:

**Brain_Data** (`nltools/data/brain_data.py`):
- Primary class for 3D/4D neuroimaging data stored as vectorized arrays (images × voxels)
- Handles spatial operations via NiftiMasker integration
- Provides statistical methods (ttest, regression) and ML methods (predict, cross-validation)
- Supports data processing pipelines (filtering, smoothing, standardization)

**Design_Matrix** (`nltools/data/design_matrix.py`):
- Enhanced pandas DataFrame for experimental designs with temporal awareness
- Tracks sampling frequency and supports HRF convolution
- Provides temporal operations (resampling) and multicollinearity detection
- Integrates with Brain_Data for GLM analyses

**Adjacency** (`nltools/data/adjacency.py`):
- Handles similarity/connectivity matrices stored efficiently as vectors (upper triangle)
- Supports multiple matrix types (distance, similarity, directed)
- Provides network analysis tools and statistical testing methods

## Key Integration Patterns

- **GLM Workflow**: Brain_Data + Design_Matrix → `regress()` → statistical maps
- **ML Workflow**: Brain_Data → `predict()` → cross-validation via `cross_validation.py`
- **Connectivity**: Brain_Data → similarity methods → Adjacency → network analysis
- **I/O Consistency**: All classes support NIfTI, HDF5, and text formats with similar APIs

## Testing Structure

Tests use pytest with fixtures in `conftest.py` that create simulated data:
- `sim_brain_data`: Brain_Data with synthetic signal
- `sim_design_matrix`: Design matrix with experimental conditions
- `sim_adjacency_*`: Various adjacency matrix configurations
- HDF5 fixtures for backwards compatibility testing

## Important Dependencies

- **nilearn**: Core neuroimaging operations and NiftiMasker
- **scikit-learn**: Machine learning algorithms and cross-validation
- **h5py**: HDF5 file format support
- **seaborn**: Statistical plotting
- Development uses **jupyter-book** for documentation and **tables** for HDF5