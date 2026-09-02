# CLAUDE.md

## Gates: `uv run poe lint` · `uv run poe lint-api`

**All commands must use `uv run` prefix** — bare `pytest`/`python` uses the wrong environment. Run `lint-api` (vocabulary checker + semgrep + kw-only check + docs-drift check) after touching any public signature.

## Project Context
- **v0.6.0**: breaking release — API changes allowed.
- **Breaking commits**: use `!` in the type (e.g. `feat(data)!:`, `refactor!:`) and include a `BREAKING:` line in the body describing the API change.
- **Task tracking**: Linear (project `nltools`, team `Ejolly`).

## Skills

Use the vendored project skills (in `.claude/skills/`) for the domains they cover — don't work from memory:
- **`nilearn`** — before writing, reviewing, or debugging any nilearn code (GLM, masking, plotting, decoding, datasets, connectivity). This codebase builds directly on nilearn; the skill carries current signatures and patterns.
- **`marimo-notebook` / `marimo-pair`** — when authoring or editing the marimo `.py` tutorials under `docs/tutorials/`.

## Architecture: Functional Core, Imperative Shell

Classes are **facades and glue** — all real logic lives in pure functions.

- **Shell** (imperative): `nltools/data/` — `BrainData`, `Adjacency`, `DesignMatrix`, `BrainCollection`. Each is a facade over submodules (io, modeling, plotting, etc.)
- **Core** (functional): `utils`, `cross_validation`, `mask`, `algorithms/` — the single functional entry point (`from nltools.algorithms import fdr, zscore, isc, ...`): `corrections`, `outliers`, `signal`, `similarity`, `regression`, `alignment` [SRM/hyperalignment/LocalAlignment/procrustes], `inference` [permutation/bootstrap + intersubject], `ridge`, `hrf`. (`nltools.stats` was removed in v0.6.0 — everything it held now lives here.)

**Design rules:**
- Pure functions first. Classes compose and delegate to them, never the reverse.
- Use frozen dataclasses for immutable state containers. Prefer modern Python (type hints, `@dataclass(frozen=True)`, `|` unions, etc.).
- Don't repeat logic — extract shared helpers as functions where most useful and import them. Prefer a single source of truth over duplicated code.
- **No underscore-prefixed module names** (e.g. `validation.py` not `_validation.py`). Leading underscores are fine for internal functions/methods, just not filenames.
- **Generated column names use the reserved `.nl_` prefix** (`nltools.utils.RESERVED_PREFIX`): build them with `reserved_name()` / `run_separated_name()`, and recognize them with `is_reserved_name()` / `parse_run_separated()` — never by pattern-matching user-controlled names.
- **One GPU execution layer, run-or-raise**: memory budgets, batch sizing, and OOM recovery live only in `algorithms/backends.py` (`device_memory_budget`, `auto_batch_size`, `compute_oom_safe`, `auto_n_jobs_for_arrays`) — algorithms supply per-item working-set estimates, never their own budget math (source-scan test enforces). `max_gpu_memory_gb=None` = measure the device. Explicit `device='gpu'`/`parallel='gpu'` runs on GPU or raises; `'auto'` is the only graceful fallback.

**Internals reference** (design docs for the subsystems below — read the relevant one before changing that subsystem; keep it in sync when behavior changes):
- `docs/development/execution-model.md` — `BrainCollection` parallel execution: path-backed caching, the `cache=` knob, HDF5 fit bundles, `_ItemTask`/`_DesignContext` pickling, parallel write safety. (Replaces the old `data/collection/SPEC.md`.)
- `docs/development/ridge-internals.md` — the six ridge tricks + the `class Backend` abstraction (`parallel=`/hyphenated names/MPS). (Replaces `algorithms/ridge/{DESIGN,README}.md`.)
- `docs/development/inference-internals.md` — permutation/bootstrap algorithms, deterministic cross-backend RNG, Phipson-Smyth p-values, numerical stability. (Replaces `algorithms/inference/DESIGN.md`.)
- `docs/development/index.md` — the architecture overview + canonical kwarg vocabulary (human-facing entry point; the interactive Design Tour at `docs/public/design-tour.html` links into these).

## API Conventions (v0.6.0)

**Single source of truth: `docs/_data/api-vocabulary.yml`.** Canonical kwarg names, banned aliases, per-kwarg contracts (required defaults, keyword-onlyness), documented exceptions, and enforcement scope all live there — read it before naming or renaming any public kwarg. `scripts/check_api_vocabulary.py` (in `lint-api`) enforces it against every public signature; `scripts/build_api_vocabulary.py` renders it into the docs tables (edit the YAML, never the rendered AUTOGEN blocks). New carve-outs go in that file's `exceptions:` / `enforcement.exemptions:` with a reason — never as inline suppressions.

Mistake-prone distinctions (full detail in the YAML): `method=` (algorithm variant) vs `metric=` (similarity only) vs `summary=` (`'mean'|'median'` central tendency); `n_jobs=` (CPU workers) vs `device=` (cpu/gpu); `n_permute` (permutations) vs `n_samples` (bootstrap).

Conventions the manifest can't express per-kwarg (enforcer in parentheses):

- **Trailing kwarg order** (when any apply): `..., domain_kwargs, return_flags, n_jobs=-1, random_state=None, progress_bar=False` (convention only).
- **`**kwargs`**: permitted **only** when forwarding to an external third-party API (sklearn, matplotlib, nilearn, nibabel, seaborn, pandas); internal nltools delegation must use explicit signatures (semgrep `kwargs-internal-forwarding`).
- **Keyword-only `*` marker**: required in `__init__` after the primary data arg, and in any public method with 3+ kwargs (`scripts/check_kwonly.py`).
- **Facade translation**: the ridge/alignment layers may keep legacy names (`parallel=`, `backend=`, `n_iter=`); facades translate at the boundary, and the checker path-excludes those subsystems.
- **`spatial_scale`**: a given method may support a subset of `'whole_brain'|'roi'|'searchlight'` and raise `NotImplementedError` for the rest; vocabulary follows Jolly & Chang, 2021, *SCAN*.

## Documentation

Jupyter Book v2 (mystmd). API docs auto-generated by `griffe2md` from Google-style docstrings.
Version lives **only** in `pyproject.toml`. A single `BASE_URL` env var parameterizes the build
(unset = root — local and the nltools.org deploy; `/<repo>` only for a subpath deploy). Every push
to `master` rebuilds, executes, and deploys the site to https://nltools.org (`docs-deploy.yml`), so
docs track master and run ahead of the PyPI release.

```bash
uv run poe docs-generate   # regen sources: API docs (griffe2md) + tutorial .md (marimo → MyST-NB)
uv run poe docs-site       # myst build --site --html --execute (bakes tutorial outputs)
uv run poe docs-build      # full: docs-generate → docs-site
uv run poe docs-preview    # myst start --execute (live preview; reuses the execute cache)
uv run poe docs-clean      # rm _build
uv run poe tutorials       # run every tutorial notebook end-to-end (fast; no MyST)
uv run poe changelog       # regenerate docs/changelog.md (git-cliff)
uv run poe release         # bump version, build, smoke-test, changelog, tag, publish
```

Scripts: `build_api_docs.py` (API md), `marimo_to_myst.py` (marimo → MyST-NB), `release.py`.
Config: `docs/myst.yml`, `[tool.griffe2md]` in `pyproject.toml`, `cliff.toml`.

**Tutorials** — plain marimo `.py` notebooks under `docs/tutorials/{basics,workflows}/` are the single
source of truth (edit locally with `uv run marimo edit <nb>.py`; the PEP 723 header lists only
`marimo` + `nltools>=0.6.0` so `uvx marimo edit --sandbox` and molab can run them — the pin fails
loudly until 0.6.0 is on PyPI rather than silently importing 0.5.1). `marimo_to_myst.py`
(in `docs-generate`) renders each to a committed sibling `.md` — with an "Open in molab" badge and
`edit_url`/`source_url`/`downloads` frontmatter pointing at the `.py` — that `docs-site`/`docs-preview` execute
through a `python3` ipykernel; outputs are cached in `docs/_build/execute`, so `myst` **must** be given
`--execute` or the pages render with no outputs. A cell that raises halts execution of every cell after
it in that notebook (the build still exits 0) — grep the build log for `⛔️`. In-browser support (marimo
WASM tutorial pages *and* the library's Pyodide path — `seed_resources`, IDBFS cache, node smoke tests)
is deferred to post-0.6.0 and preserved on the `0.6.1-browser` branch.

### Docstring style — Google-style Markdown, NO RST

Docstrings are consumed by `griffe2md` + mystmd. **RST syntax does not render and leaks into the
docs.** Write Google-style sections with Markdown inline formatting:

- **Sections:** `Args:` / `Returns:` / `Raises:` / `Examples:` / `Note:` (not RST field lists like
  `:param x:` / `:returns:`).
- **Cross-references:** plain Markdown code spans — `` `BrainData.distance` ``, `` `list_atlases` `` —
  **never** RST roles (`` :meth:`...` ``, `` :func:`...` ``, `` :class:`...` ``).
- **Code blocks:** fenced ```` ```python ```` blocks, not RST `::` literal blocks.
- **First line = summary:** griffe uses the first physical line as the one-line summary in tables.
  Keep it a complete, standalone sentence (≤120 chars, ends with a period); put detail in a
  following paragraph. Avoid summaries that wrap mid-phrase or run two sentences together.
- Deprecated members: start the docstring with `Deprecated:` — they are auto-hidden from the API
  reference (documented in the migration guide instead).

## Testing: Red-Green TDD

Always write or identify a **failing test first**, then implement the minimal code to pass it.

1. Write/find the test that demonstrates the desired behavior
2. Run it — confirm it fails (red)
3. Write the minimal implementation to pass (green)
4. Refactor if needed, re-run to confirm still green
5. Run related tests for regressions

**Before tests:** `uv run ruff check --fix nltools/ && uv run ruff format nltools/`

**Capture output:** `uv run pytest ... 2>&1 | tee pytest.log` then search the log file, don't re-run.

**Markers:** `slow` (skipped by default, ~7 min, ask before running), `gpu` (CUDA required)

```bash
# Targeted TDD (preferred during development):
uv run pytest nltools/tests/data/braindata -xvs
uv run pytest -k "ridge and cv" -x

# Per-class / per-module suites (poe wrappers):
uv run poe test-braindata      # (also: test-adjacency, test-designmatrix, test-collection)
uv run poe test-algorithms     # (also: test-core, test-models, test-io, test-plotting, test-support)

# Default (non-slow, parallel):
uv run poe test                # == pytest -n auto
uv run poe test-all            # everything incl. slow + integration (ask first)

# Tests live in nltools/tests/: data/ (the four data classes), core/ (algorithms, incl.
# core/test_algorithms/ for the consolidated modules and core/test_inference/),
# models/ (GLM/ridge/base), io_tests/, plotting/, support/, integration/, fixtures/
```
