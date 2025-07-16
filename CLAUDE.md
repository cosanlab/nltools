# CLAUDE.md

## ALWAYS Remember the Following

- Remember the User's Broad Goals which set the bigger picture for all of their queries:
  - We are re developing a Python library called `nltools` which provides an intuitive way to work and analyze fMRI data
  - `nltools` builds upon many Python libraries (see @pyproject.toml) and aims to to offer a more user-friendly user experience by providing utilities and classes that operate as a more accessible level of abstraction and thus require less boilerplate code for common operations
  - The library loosely follows a "functional core, imperative shell" design pattern where imperative classes are defined in nltools/data/ and most other .py files contain utility functions
  - The library should be a robust and reliable while also minimizing maintenance overhead
- Always answer user queries using the following process:
  1. Understand what the query is requesting and use your context7 mcp tool to gather additional information about relevant Python libraries such as those listed in @pyproject.toml. Say "RESEARCHING" when you start this step, however, you can skip step if you have performed it before in the current session and you feel like you have enough context, in which case say "PREVIOUS RESEARCH SUFFICIENT". 
  2. Use that information along with relevant code-base files to make plan for what exact changes needed to be made
  3. Present the plan to the user for review along with any additional questions you have that would help further clarify the plan. Do not write/change any code yet. Say "PLANNING" when you start this step.
  4. Wait for the user's feedback and update the plan. This may take several iteration cycles
  5. Once you have the user's explicit approval, say "EXECUTING" and implement the plan following the [execution guidelines](#execution-guidelines) section below
  6. Provide the user with a succinct summary of what you did and ask them if there are any modifications they would like made
  7. Apply the requested modifications from the user. This may take several iteration cycles.
  
## Execution guidelines
- Always write code that following best Pythonic practices
- Always add types if they improve reliability and understability
- Always write/edit docstrings in google docstring format
- Always run `uv run ruff check` and `uv run ruff check --fix` to lint and check your code for errors
- Always test your code by adding/updating test(s) in ntools/tests using the current tests as examples and running `uv run pytest -k nameoftest`
  - You should use/update pytest fixtures in nltools/tests/conftest.py to avoid re-writing test boilerplate
- When editing or modifying Notebooks make sure they build successfully using `uv run jupyter-book build docs/`

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