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

Regenerate the generated docs sources (the API vocabulary tables): `uv run poe docs-generate`

Build the site: `uv run poe docs-build` — a strict build into `site/` that fails on a broken internal link or a missing cross-reference

Preview the site with live reload: `uv run poe docs-serve`

Build the site cold, as CI does: `uv run poe docs-build-fresh` — the same build with the tutorials' fit caches dropped first

Edit a tutorial notebook: `uv run marimo edit docs/tutorials/<group>/<notebook>.py`. The tutorial pages are out of the site until #503 restores them

Generate changelog: `uv run poe changelog`

Add or remove dependencies: `uv add/remove packagename`

Add or remove development dependencies: `uv add/remove --dev packagename`

Build package locally: `uv build`

## Documentation

Pages under `docs/api/` are hand-written: frontmatter, prose, and `::: dotted.path` directives that mkdocstrings renders at build time. There is one page per user-facing namespace, and each page's `members:` list mirrors the API table in `CLAUDE.md`. Add a new user-facing object to its namespace's page, and add a new page to the `nav` in `zensical.toml`. The strict build is the only check on these pages: a directive naming something that does not exist fails it.

The site is currently the home page, the migration guide, the Reference and the Development pages. The tutorial notebooks stay under `docs/tutorials/` and the User Guide pages under `docs-staging/guide/`; #503 and #505 bring them back into the nav one page at a time. When a tutorial returns, `docs-generate` writes its `.md` beside the `.py` again (a git-ignored build artifact) and `markdown-exec` executes its cells during the build.

To link into the API from another page, use the object's full dotted path as the anchor — `[BrainData.predict](api/data/brain_data.md#nltools.data.braindata.BrainData.predict)` — because mkdocstrings gives every heading it emits `id="<full dotted path>"`, never a slug of the displayed name.
