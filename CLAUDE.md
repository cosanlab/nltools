# CLAUDE.md

## Gate: `uv run poe lint`

**All commands must use `uv run` prefix** — bare `pytest`/`python` uses the wrong environment.

## Project Context
- **Branch:** `uv-cleanup` → **v0.6.0** (breaking release; API changes allowed)
- **Task tracking**: Linear (project: `nltools`, team: `Ejolly`)
- **Breaking commits**: use `!` in the type (e.g. `feat(data)!:`, `refactor!:`) and include a `BREAKING:` line in the body describing the API change.

## Architecture: Functional Core, Imperative Shell

Classes are **facades and glue** — all real logic lives in pure functions.

- **Shell** (imperative): `nltools/data/` — `BrainData`, `Adjacency`, `DesignMatrix`, `BrainCollection`. Each is a facade over submodules (io, modeling, plotting, etc.)
- **Core** (functional): `stats`, `utils`, `cross_validation`, `mask`, `algorithms/` (`alignment` [SRM/hyperalignment/LocalAlignment], `inference`, `ridge`)

**Design rules:**
- Pure functions first. Classes compose and delegate to them, never the reverse.
- Use frozen dataclasses for immutable state containers. Prefer modern Python (type hints, `@dataclass(frozen=True)`, `|` unions, etc.).
- Don't repeat logic — extract shared helpers as functions where most useful and import them. Prefer a single source of truth over duplicated code.
- **No underscore-prefixed module names** (e.g. `validation.py` not `_validation.py`). Leading underscores are fine for internal functions/methods, just not filenames.

**Internals reference** (design docs for the subsystems below — read the relevant one before changing that subsystem; keep it in sync when behavior changes):
- `docs/development/execution-model.md` — `BrainCollection` parallel execution: path-backed caching, the `cache=` knob, HDF5 fit bundles, `_ItemTask`/`_DesignContext` pickling, parallel write safety. (Replaces the old `data/collection/SPEC.md`.)
- `docs/development/ridge-internals.md` — the six ridge tricks + the `class Backend` abstraction (`parallel=`/hyphenated names/MPS). (Replaces `algorithms/ridge/{DESIGN,README}.md`.)
- `docs/development/inference-internals.md` — permutation/bootstrap algorithms, deterministic cross-backend RNG, Phipson-Smyth p-values, numerical stability. (Replaces `algorithms/inference/DESIGN.md`.)
- `docs/development/index.md` — the architecture overview + canonical kwarg vocabulary (human-facing entry point; the interactive [Design Tour](docs/design-tour.md) links into these).

## API Conventions (v0.6.0)

Canonical kwarg names across the four data-class facades:

| Concept | Canonical kwarg | Notes |
|---|---|---|
| Algorithm/variant choice | `method` | not `algorithm`, `scheme`, `kind`, `estimator`, `icc_type`, `extract_type`, `perm_type`, `mode` |
| Spatial scale | `spatial_scale` | values: `'whole_brain' \| 'roi' \| 'searchlight'` (a given method may support a subset and raise `NotImplementedError` for the rest — e.g. `BrainData.align` has no `searchlight`; `BrainCollection.align` / `LocalAlignment` are local-only, no `whole_brain`). Used by `BrainData.predict` / `.distance` / `.align` / `.mean`/`.std`/`.median`, `BrainCollection.predict` / `.align`, and `LocalAlignment` (with companion `roi_mask=`). Distinct from `method=` (algorithm choice). Vocabulary follows Jolly & Chang, 2021, *SCAN*. |
| Distance/similarity metric | `metric` | kept separate from `method` |
| Parallel execution | `n_jobs: int = -1` | not `parallel=` (stats-layer internals still use `parallel=` but facades translate) |
| GPU/CPU selection | `device: str = "cpu"` | BrainCollection only; separate from `n_jobs` |
| Progress indicator | `progress_bar: bool = False` | not `show_progress`, `verbose` (`verbose` reserved for log-level only) |
| Threshold pair | `lower`, `upper`, `binarize` | plus convenience `threshold: float` where bidirectional |
| Permutation count | `n_permute` | not `n_perm`, `n_iter` |
| Bootstrap sample count | `n_samples` | semantically distinct from `n_permute` |
| Diagonal flag | `include_diag: bool` | not `ignore_diagonal` |
| Radius (mm) | `radius_mm: float` | units in the name |

**Canonical trailing kwarg order** (when any apply):
`..., domain_kwargs, return_flags, n_jobs=-1, random_state=None, progress_bar=False`

**`**kwargs` rule**: permitted **only** when forwarding to an external third-party API (sklearn estimator, matplotlib, nilearn, nibabel, seaborn, pandas). Internal delegation between nltools modules must use explicit signatures.

**Keyword-only `*` marker**: required in `__init__` after the primary data arg, and in any public method with 3+ kwargs.

**Facade translation**: internal algorithm-layer APIs (e.g. `CVScheme.scheme`, `Glm.noise_model`) may keep legacy names; the class facade translates at the boundary. (`LocalAlignment` was canonicalized in v0.6.0 — it uses `spatial_scale`/`roi_mask` directly, so `.align` facades pass straight through with no translation.)

**Documented naming exceptions** (deliberate deviations from the table, decided for v0.6.0):
- `BrainData.fit(model='glm'|'ridge')` keeps `model=` (not `method=`): it selects an estimator **class**, not an algorithm variant, and reads naturally. (F175)
- `Ridge(n_iter=...)` keeps `n_iter=` (a banned alias in general): it is the random-search iteration count, matching sklearn's `RandomizedSearchCV` name; no canonical name exists for that concept. (F105)
- `compute_contrasts(statistic=...)` uses `statistic=` (not `method=`): it selects an output **statistic** map (t/z/p/beta/…), not an algorithm. (F077)

## Documentation

Jupyter Book v2 (mystmd). API docs auto-generated via `griffe2md` (Google-style docstrings).

```bash
uv run poe docs-generate   # regenerate API docs only
uv run poe docs-build      # generate + myst build --site + marimo/WASM tutorials
uv run poe docs-marimo     # export tutorial notebooks to WASM pages (--base-url for /nltools/)
uv run poe docs-preview    # myst start (live preview of the site + gallery)
uv run poe docs-clean      # rm _build
uv run poe tutorials       # static-check + run every tutorial notebook end-to-end
uv run poe changelog       # regenerate docs/changelog.md from git history (git-cliff)
uv run poe release         # full release: bump version, build, smoke-test, changelog, tag, publish
```

- Config: `docs/myst.yml` (TOC + site), `[tool.griffe2md]` in `pyproject.toml`
- API generation script: `scripts/build_api_docs.py` (postprocess: strips residual RST roles,
  hides deprecated members, fixes griffe heading quirks)
- Changelog: `cliff.toml` (git-cliff config) → `docs/changelog.md`; badge CSS in `docs/_static/custom.css`
- Release: `scripts/release.py` (uv/poe workflow); version lives ONLY in `pyproject.toml`
- Tutorials: **marimo `.py` notebooks** under `docs/tutorials/{basics,workflows}/` are the single
  source of truth. `scripts/build_marimo_wasm.py` exports each to a self-contained interactive
  **WASM page** (`marimo export html-wasm --mode run --show-code --execute`) served under
  `_build/html/tutorials/<group>-<name>.html`; the MyST `tutorials/index.md` is a gallery linking
  to them. Each notebook has a PEP 723 header for the PyPI stack (auto-micropip'd in-browser) and an
  `IN_WASM` cell that installs the nltools dev wheel from a build-hosted URL; `--execute` bakes
  outputs at build time via a `file://` wheel injected into PEP 723. See `marimo-learning.md` for the
  full mechanics. (The old `marimo_to_myst.py` `.md`-render + JupyterLite pipeline is retired.)
  Local check: `uv run poe tutorials`. **In-browser data**: `01_glm`/`03_mvpa` + all basics seed
  trimmed data from HF `nltools/niftis`; `02_encoding`/`04_isc` still need trimmed subsets hosted
  before their data cells run in WASM (they run locally today).

### Docstring style — Google-style Markdown, NO RST

Docstrings are consumed by `griffe2md` + mystmd. **RST syntax does not render and leaks into the
docs.** Write Google-style sections with Markdown inline formatting:

- **Sections:** `Args:` / `Returns:` / `Raises:` / `Examples:` / `Note:` (not RST field lists like
  `:param x:` / `:returns:`).
- **Cross-references:** plain Markdown code spans — `` `BrainData.distance` ``, `` `list_atlases` `` —
  **never** RST roles (`` :meth:`...` ``, `` :func:`...` ``, `` :class:`...` ``).
- **Code blocks:** fenced ```` ```python ```` blocks, not RST `::` literal blocks.
- **Inline literals:** single or double backticks both render fine (`` `x` `` or `` ``x`` ``).
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
uv run poe test-stats          # (also: test-core, test-models, test-io, test-plotting, test-support)

# Default (non-slow, parallel):
uv run poe test                # == pytest -n auto
uv run poe test-all            # everything incl. slow + integration (ask first)

# Tests live in nltools/tests/: data/ (the four data classes), stats/, core/ (algorithms),
# models/ (GLM/ridge/base), io_tests/, plotting/, support/, integration/, fixtures/
```
