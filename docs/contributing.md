# Contributing

Project management is handled by [`uv`](https://docs.astral.sh/uv/guides/projects/) which will automatically install development dependencies alongside core dependencies, configure a virtual environment, and install `nltools` into that environment in editable mode.

## Setup

After cloning, enable the shared git hooks:

```bash
uv run poe setup-hooks
```

This enforces [conventional commits](https://www.conventionalcommits.org/) on all commit messages. The format is:

```
<type>[optional scope]: <description>
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`.

## Common Commands

Run tests: `uv run pytest`

Run linting: `uv run ruff check`

Fix linting: `uv run ruff check --fix`

Build docs locally: `uv run jupyter-book build docs/`

Generate changelog: `uv run poe changelog`

Add or remove dependencies: `uv add/remove packagename`

Add or remove development dependencies: `uv add/remove --dev packagename`

Build package locally: `uv build`