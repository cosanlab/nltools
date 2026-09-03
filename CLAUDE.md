# CLAUDE.md

## Gates

`uv run poe lint` (ruff fix → ruff format → ty) · `uv run poe lint-api` (canonical-kwarg checker, semgrep, keyword-only check, docstring check, plus drift checks for the rendered vocabulary and API docs). Run `lint-api` after touching any public signature or docstring.

**Every command needs the `uv run` prefix** — bare `pytest`/`python` uses the wrong environment. `uv run poe` lists every task with help text; the sections below cover only the ones whose behavior isn't obvious from that listing.

## Project Context

- **v0.6.0**: breaking release — API changes allowed.
- **Breaking commits**: `!` in the type (`feat(data)!:`, `refactor!:`) plus a `BREAKING:` line in the body describing the API change.
- **Task tracking**: Linear (project `nltools`, team `Ejolly`).

## Skills

Use the vendored project skills (`.claude/skills/`) for the domains they cover — don't work from memory:

- **`nilearn`** — before writing, reviewing, or debugging any nilearn code (GLM, masking, plotting, decoding, datasets, connectivity). This codebase builds directly on nilearn; the skill carries current signatures and patterns.
- **`marimo-notebook` / `marimo-pair`** — when authoring or editing the marimo `.py` tutorials under `docs/tutorials/`.

## Architecture: Functional Core, Imperative Shell

Classes are **facades and glue** — all real logic lives in pure functions. Classes compose and delegate to them, never the reverse.

- **Shell** (imperative): `nltools/data/` — `BrainData`, `Adjacency`, `DesignMatrix`, `BrainCollection`, each a facade over a package of submodules (io, modeling, plotting, …).
- **Core** (functional): `utils`, `cross_validation`, `mask`, and `nltools/algorithms/` — the single functional entry point, flat: `from nltools.algorithms import fdr, zscore, isc, ...`. Its `__init__.py` docstring maps the submodules.
- **`nltools.stats` was removed in v0.6.0** — everything it held now lives in `nltools.algorithms`.

**Design rules:**

- Frozen dataclasses for immutable state containers; modern Python throughout (type hints, `|` unions).
- **No underscore-prefixed module names** (`validation.py`, not `_validation.py`). Leading underscores are fine for functions and methods, just not filenames.
- **Generated column names use the reserved `.nl_` prefix** (`nltools.utils.RESERVED_PREFIX`): build them with `reserved_name()` / `run_separated_name()`, recognize them with `is_reserved_name()` / `parse_run_separated()` — never by pattern-matching user-controlled names.
- **One GPU execution layer, run-or-raise**: memory budgets, batch sizing, and OOM recovery live only in `algorithms/backends.py` (`device_memory_budget`, `auto_batch_size`, `compute_oom_safe`, `auto_n_jobs_for_arrays`) — algorithms supply per-item working-set estimates, never their own budget math (a source-scan test enforces this). `max_gpu_memory_gb=None` = measure the device. Explicit `device='gpu'` / `parallel='gpu'` runs on GPU or raises; `'auto'` is the only graceful fallback.

**Internals reference** — read the relevant page before changing that subsystem, and keep it in sync when behavior changes:

- `docs/development/execution-model.md` — `BrainCollection` parallel execution: path-backed caching, the `cache=` knob, HDF5 fit bundles, the pickling contract, parallel write safety.
- `docs/development/ridge-internals.md` — the six ridge tricks and the `Backend` abstraction (`parallel=`, hyphenated names, MPS).
- `docs/development/inference-internals.md` — permutation/bootstrap algorithms, deterministic cross-backend RNG, Phipson-Smyth p-values, numerical stability.
- `docs/development/index.md` — architecture overview and the rendered kwarg vocabulary table (the human-facing entry point; the Design Tour at `docs/public/design-tour.html` links into these).

## API Conventions (v0.6.0)

**Single source of truth: `docs/_data/api-vocabulary.yml`** — canonical kwarg names, banned aliases, per-kwarg contracts (required defaults, keyword-onlyness), documented exceptions, and enforcement scope. Read it before naming or renaming any public kwarg. `scripts/check_api_vocabulary.py` enforces it against every public signature; `scripts/build_api_vocabulary.py` renders it into the docs tables (edit the YAML, never the rendered AUTOGEN blocks). New carve-outs go in that file's `exceptions:` / `enforcement.exemptions:` with a reason — never as inline suppressions.

Mistake-prone distinctions: `method=` (algorithm variant) vs `metric=` (similarity only) vs `summary=` (`'mean'|'median'` central tendency); `n_jobs=` (CPU workers) vs `device=` (cpu/gpu); `n_permute` (permutations) vs `n_samples` (bootstrap).

Conventions the manifest can't express per-kwarg (enforcer in parentheses):

- **Trailing kwarg order** (when any apply): `..., domain_kwargs, return_flags, n_jobs=-1, random_state=None, progress_bar=False` (convention only).
- **`**kwargs`**: permitted **only** when forwarding to an external third-party API (sklearn, matplotlib, nilearn, nibabel, seaborn, pandas); internal nltools delegation must use explicit signatures (semgrep `kwargs-internal-forwarding`).
- **Keyword-only `*` marker**: required in `__init__` after the primary data arg, and in any public method with 3+ kwargs (`scripts/check_kwonly.py`).
- **Facade translation**: the ridge/alignment layers keep legacy names (`parallel=`, `backend=`, `n_iter=`); facades translate at the boundary, and the checker path-excludes those subsystems.
- **`spatial_scale`**: a given method may support a subset of `'whole_brain'|'roi'|'searchlight'` and raise `NotImplementedError` for the rest; vocabulary follows Jolly & Chang, 2021, *SCAN*.

## Documentation

Jupyter Book v2 (mystmd), deployed to https://nltools.org on every push to `master` (`docs-deploy.yml`) — so docs track master and run ahead of the PyPI release. The version lives **only** in `pyproject.toml`. A single `BASE_URL` env var parameterizes the build (unset = root, which covers both local builds and the nltools.org deploy; `/<repo>` only for a subpath deploy).

- **`docs/api/` is generated and committed** — `griffe2md` from Google-style docstrings via `scripts/build_api_docs.py`. Never hand-edit those files; `lint-api` fails on the drift. Same for every rendered `AUTOGEN` block.
- `uv run poe docs-build` = `docs-generate` (API md + tutorial md) → `docs-site`. **`myst` must be given `--execute`** or tutorial pages render with no outputs — the poe tasks already pass it. A cell that raises halts every later cell in that notebook while the build still exits 0, so grep the build log for `⛔️`.
- **Tutorials**: the plain marimo `.py` notebooks under `docs/tutorials/{basics,workflows}/` are the single source of truth (`uv run marimo edit <nb>.py`); `scripts/marimo_to_myst.py` renders each into a committed sibling `.md` that the site executes. Edit the `.py`, never the `.md`. The PEP 723 header lists only `marimo` + `nltools>=0.6.0` so `uvx marimo edit --sandbox` and molab can run them.
- In-browser support (marimo WASM tutorial pages and the library's Pyodide path) is deferred post-0.6.0 and preserved on the `0.6.1-browser` branch — don't reintroduce it.

### Docstring style — Google-style Markdown, NO RST

RST syntax does not render and leaks into the published docs.

- **Sections:** `Args:` / `Returns:` / `Raises:` / `Examples:` / `Note:` — not RST field lists (`:param x:`, `:returns:`).
- **Cross-references:** plain Markdown code spans — `` `BrainData.distance` ``, `` `list_atlases` `` — never RST roles (`` :meth:`...` ``, `` :func:`...` ``).
- **Code blocks:** fenced ```` ```python ```` blocks, not RST `::` literal blocks.
- **First line = summary:** griffe uses the first physical line as the one-line summary in tables. Keep it a complete, standalone sentence (≤120 chars, ends with a period) and put detail in a following paragraph.
- **Deprecated members:** start the docstring with `Deprecated:` — they are auto-hidden from the API reference and documented in the migration guide instead.

## Testing: Red-Green TDD

Always write or identify a **failing test first**, then the minimal code to pass it: red → green → refactor → re-run related tests for regressions. Run `uv run poe lint` before running tests.

- **Markers:** `slow` and `integration` are both skipped by default — `test-all` runs them (~7 min; ask first). `gpu` requires CUDA.
- **Capture output:** `uv run pytest ... 2>&1 | tee pytest.log`, then search the log rather than re-running.
- Tests mirror the source layout under `nltools/tests/`: `data/` (the four data classes), `core/` (algorithms, including `core/test_algorithms/` and `core/test_inference/`), `models/`, `io_tests/`, `plotting/`, `support/`, `integration/`, `fixtures/`.

```bash
uv run pytest nltools/tests/data/braindata -xvs   # targeted TDD (preferred)
uv run pytest -k "ridge and cv" -x
uv run poe test                                   # default: fast tests, parallel
uv run poe test-braindata                         # per-class/module wrappers; `uv run poe` lists all
```
