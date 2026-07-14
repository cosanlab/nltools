[![Package versioning](https://img.shields.io/pypi/v/nltools.svg)](https://pypi.org/project/nltools/)
[![(Auto-On-Push/PR) Formatting, Tests, and Coverage](https://github.com/cosanlab/nltools/actions/workflows/auto_formatting_tests_and_coverage.yml/badge.svg)](https://github.com/cosanlab/nltools/actions/workflows/auto_formatting_tests_and_coverage.yml)
[![codecov](https://codecov.io/gh/cosanlab/nltools/branch/master/graph/badge.svg)](https://codecov.io/gh/cosanlab/nltools)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/625677967a0749299f38c2bf8ee269c3)](https://www.codacy.com/app/ljchang/nltools?utm_source=github.com&utm_medium=referral&utm_content=ljchang/nltools&utm_campaign=Badge_Grade)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2229813.svg)](https://doi.org/10.5281/zenodo.2229813)
![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

# NLTools

Python toolbox for analyzing neuroimaging data. It is particularly useful for conducting multivariate analyses. It is originally based on Tor Wager's object oriented matlab [canlab core tools](http://wagerlab.colorado.edu/tools) and relies heavily on [nilearn](http://nilearn.github.io) and [scikit learn](http://scikit-learn.org/stable/index.html). Nltools is compatible with Python 3.11+.

## Documentation

Documentation and tutorials are available at https://nltools.org

## Installation

```
uv add nltools
```

or

```
pip install nltools
```

## Development

Project management is handled by `uv` which will automatically install development dependencies along side core dependencies, configure a virtual environment, and install `nltools` into that environment in editable mode.

```
git clone https://github.com/cosanlab/nltools  
uv sync  
```

- Build docs (including the local JupyterLite bundle): `uv run poe docs-build`
- Build JupyterLite for `cosanlab.github.io/nltools/`: `uv run poe docs-jupyterlite-pages`
- Preview built-docs: `uv run poe docs-preview` 
- Run linting: `uv run poe lint`
- Run all tests: `uv run pytest`
- Run specific test: `uv run pytest -k test_name`
- Add or remove dependencies: `uv add/remove packagename`
- Add or remove development dependencies: `uv add/remove --dev packagename`
- Build package locally: `uv build`
