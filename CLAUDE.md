# CLAUDE.md

## Project Goals
- We are are developing a Python library called `nltools` which provides an intuitive way to work with and analyze fMRI data
- The overall goal to help improve the library and ease the maintainence burden by more closely following a "functional-core, imperative shell" design pattern, with all imperative classes defined in `nltools/data/` and all other project files primarily serving as functions (and tests). This shouldn't be a prison, but a guiding principle for the library design
- `nltools` builds upon many Python libraries (see pyproject.toml) and aims to to offer a more user-friendly user experience by providing utilities and classes that operate as a more accessible level of abstraction and thus require less boilerplate code for common operations

## Execution guidelines
- Always make sure you have sufficient research on Pythonic best-practices and the latest API syntax of any Python libraries BEFORE you make code changes. You can search previous research by reviewing files within claude-reserach/. If previous research is out-dated or in-sufficient, you should use sub-agents to gather additional research and instruct them to create/update the markdown files in claude-research/ 
- Always run `uv run ruff check` and `uv run ruff check --fix` to lint and check your code for errors
- Always write/edit Python docstrings in google docstring format
- Only add Python type annotations if they improve reliability and understability
- When asked to write or edit Notebooks, make sure they build successfully using `uv run jupyter-book build docs/`
- Follow a test-driven-development pattern using `uv run pytest -k nameoftest` and pytest fixtures defined in `nltools/tests/conftest.py`:
  - Write/update a test for functionality that you intend to verify and immediately run it see how it fails
  - Then incrementally add minimal source code changes until the test passes
  - Then update the test code to test more specific functionality and see how it fails
  - Repeat incrementally adding minimal source code changes until the test passes
  - Repeat until the task is done

## Imperative Classes

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

## Important Run-time Dependencies
- **h5py**: HDF5 file format support
- **nilearn**: Core neuroimaging operations and NiftiMasker
- **seaborn**: Statistical plotting
- **pynv**: for interacting with the [Neurovault](https://neurovault.org/) API
- **scikit-learn**: Machine learning algorithms and cross-validation (dependency of `nilearn`)

## Development Dependencies
-**jupyter-book** for documentation and **tables** for HDF5
-**networkx** for visualizing graphs
-**pytest** for testing
-**ruff** for linting and syntax checking
-**tables** for tables