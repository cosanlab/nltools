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

Remove the build outputs: `uv run poe docs-clean` — drops `site/`, zensical's page cache, the generated changelog and the generated tutorial pages

Edit a tutorial notebook: `uv run marimo edit docs/tutorials/<group>/<notebook>.py`. The tutorial pages are out of the site until #503 restores them

Generate the changelog on its own: `uv run poe changelog` — `docs-generate` already runs it, writing `docs/changelog.md` from git history; the file is not committed, and a release tag is what turns its Unreleased section into a version heading

Add or remove dependencies: `uv add/remove packagename`

Add or remove development dependencies: `uv add/remove --dev packagename`

Build package locally: `uv build`

## Documentation

The site is built by zensical, configured in `zensical.toml` at the repo root: nav, theme, markdown extensions, and the mkdocstrings and markdown-exec plugins. It is the only docs toolchain.

Pages under `docs/api/` are hand-written: frontmatter, prose, and `::: dotted.path` directives that mkdocstrings renders at build time. There is one page per user-facing namespace, and each page's `members:` list mirrors the API table in `CLAUDE.md`. Add a new user-facing object to its namespace's page, and add a new page to the `nav` in `zensical.toml`. The strict build is the only check on these pages: a directive naming something that does not exist fails it.

The site is the home page, the quickstart, the migration guide, the tutorials, the Reference pages and the Development pages. Tutorial notebooks under `docs/tutorials/` join the nav one at a time; a notebook that is not in the nav is parked source and is not built. For each notebook in the nav, `docs-generate` writes a `.md` beside the `.py` (a git-ignored build artifact) and `markdown-exec` executes its cells during the build. The User Guide pages are parked under `docs-staging/guide/`, outside `docs/`, so zensical does not ship them.

To link into the API from another page, use the object's full dotted path as the anchor — `[BrainData.predict](api/data/brain_data.md#nltools.data.braindata.BrainData.predict)` — because mkdocstrings gives every heading it emits `id="<full dotted path>"`, never a slug of the displayed name.
