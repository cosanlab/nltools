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

Regenerate the generated docs sources (vocabulary tables, tutorial markdown): `uv run poe docs-generate`

Build the site: `uv run poe docs-build` — a strict build into `site/`, which fails on a broken internal link

Preview the site with live reload: `uv run poe docs-serve`

Build the tutorials with their outputs baked in: `uv run poe docs-build-fresh` — the MyST build, which executes every notebook cell, until the notebook rendering pipeline lands

Edit a tutorial: `uv run marimo edit docs/tutorials/<group>/<notebook>.py`, then `uv run poe docs-generate` to re-render its `.md`

Generate changelog: `uv run poe changelog`

Add or remove dependencies: `uv add/remove packagename`

Add or remove development dependencies: `uv add/remove --dev packagename`

Build package locally: `uv build`

## Documentation

Pages under `docs/api/` are hand-written: frontmatter, prose, and `::: dotted.path` directives that mkdocstrings renders at build time. Add a public object to the page that fits it — and to the `nltools.algorithms` A-Z index when it is an algorithm — then add any new page to the `nav` in `zensical.toml`. `uv run poe lint-api` fails when an export has no home, has two, or a directive names something that does not exist.

To link into the API from a guide page, use the object's full dotted path as the anchor — `[BrainData.predict](../api/data/brain_data.md#nltools.data.braindata.BrainData.predict)` — because mkdocstrings gives every heading it emits `id="<full dotted path>"`, never a slug of the displayed name.
