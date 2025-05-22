# Contributing

Project management is handled by [`uv`](https://docs.astral.sh/uv/guides/projects/) which will automatically install development dependencies alongside core dependencies, configure a virtual environment, and install `nltools` into that environment in editable mode.

Run tests: `uv run pytest`

Run linting: `uv run ruff check`

Fix linting: `uv run ruff check --fix`

Build docs locally: `uv run jupyter-book build docs/` 

Add or remove dependencies: `uv add/remove packagename`

Add or remove development dependencies: `uv add/remove --dev packagename`

Build package locally: `uv build`