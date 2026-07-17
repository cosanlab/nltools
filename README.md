[![Package versioning](https://img.shields.io/pypi/v/nltools.svg)](https://pypi.org/project/nltools/)
[![CI](https://github.com/cosanlab/nltools/actions/workflows/ci.yml/badge.svg)](https://github.com/cosanlab/nltools/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cosanlab/nltools/branch/master/graph/badge.svg)](https://codecov.io/gh/cosanlab/nltools)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2229813.svg)](https://doi.org/10.5281/zenodo.2229813)
![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

# NLTools

Python toolbox for analyzing neuroimaging data, with a focus on multivariate analyses. It grew out of Tor Wager's object-oriented Matlab [CANlab core tools](http://wagerlab.colorado.edu/tools) and builds on [nilearn](https://nilearn.github.io) and [scikit-learn](https://scikit-learn.org). Requires Python 3.11+.

## Documentation

Documentation and tutorials are available at https://nltools.org

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended, for use in a project):

```
uv add nltools
```

Or with pip:

```
pip install nltools
```

## Development

`uv` manages the whole workflow — it creates the virtual environment, installs core and development dependencies, and installs `nltools` into it in editable mode.

```
git clone https://github.com/cosanlab/nltools
cd nltools
uv sync
```

Common tasks (run via [poe](https://poethepoet.natn.io/), all prefixed with `uv run`):

| Command | What it does |
|---|---|
| `uv run poe lint` | Fix, format, and type-check (ruff + ty) |
| `uv run poe test` | Run the fast test suite in parallel |
| `uv run pytest -k test_name` | Run a specific test |
| `uv run poe docs-preview` | Live-preview the docs site |
| `uv run poe docs-build` | Full docs build (API reference + tutorials + interactive WASM pages) |
| `uv run poe tutorials` | Run every tutorial notebook end-to-end |
| `uv build` | Build the package locally |
| `uv add/remove [--dev] pkg` | Add or remove a (dev) dependency |
