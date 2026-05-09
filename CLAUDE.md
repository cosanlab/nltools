# CLAUDE.md

## Gate: `uv run poe lint`

**All commands must use `uv run` prefix** — bare `pytest`/`python` uses the wrong environment.

## Code Search: symbex first

Use `symbex` for exploring Python symbols (classes, functions, signatures, docstrings). It's far more token-efficient than reading entire files. Fall back to `Grep`/`Read` only when symbex can't get what you need.

```bash
symbex 'BrainData.fit' -s        # signature only
symbex 'BrainData.*' -s -d       # all methods with docstrings
symbex '*' -s nltools/stats.py   # all symbols in a file
```

If `symbex` is not found, prompt the user to install it: `uv tool install symbex`

## Project Context
- **Branch:** `uv-cleanup` → **v0.6.0** (breaking release; API changes allowed)
- **Task tracking**: Linear (project: `nltools`, team: `Ejolly`)
- **Breaking commits**: use `!` in the type (e.g. `feat(data)!:`, `refactor!:`) and include a `BREAKING:` line in the body describing the API change.

## Architecture: Functional Core, Imperative Shell

Classes are **facades and glue** — all real logic lives in pure functions.

- **Shell** (imperative): `nltools/data/` — `BrainData`, `Adjacency`, `DesignMatrix`, `BrainCollection`. Each is a facade over submodules (io, modeling, plotting, etc.)
- **Core** (functional): `stats.py`, `utils.py`, `algorithms/` (`ridge`, `srm`, `hyperalignment`, `inference`)

**Design rules:**
- Pure functions first. Classes compose and delegate to them, never the reverse.
- Use frozen dataclasses for immutable state containers. Prefer modern Python (type hints, `@dataclass(frozen=True)`, `|` unions, etc.).
- Don't repeat logic — extract shared helpers as functions where most useful and import them. Prefer a single source of truth over duplicated code.
- **No underscore-prefixed module names** (e.g. `validation.py` not `_validation.py`). Leading underscores are fine for internal functions/methods, just not filenames.

## API Conventions (v0.6.0)

Canonical kwarg names across the four data-class facades:

| Concept | Canonical kwarg | Notes |
|---|---|---|
| Algorithm/variant choice | `method` | not `algorithm`, `scheme`, `kind`, `estimator`, `icc_type`, `extract_type`, `perm_type`, `mode` |
| Spatial scope | `spatial_scale` | values: `'whole_brain' \| 'roi' \| 'searchlight'`. Used by `BrainData.predict`, `BrainData.distance` (and other future spatial-scale-aware methods). Distinct from `method=` (algorithm choice). Companion kwargs `roi_mask=`, `radius_mm=`. Vocabulary follows Jolly & Chang, 2021, *SCAN*. |
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

**Facade translation**: internal algorithm-layer APIs (e.g. `CVScheme.scheme`, `LocalAlignment.scheme`, `Glm.noise_model`) may keep legacy names; the class facade translates at the boundary.

## Documentation

Jupyter Book v2 (mystmd). API docs auto-generated via `griffe2md` (Google-style docstrings).

```bash
uv run poe docs-generate   # regenerate API docs only
uv run poe docs-build      # generate + myst build --site
uv run poe docs-preview    # myst start (live preview)
uv run poe docs-clean      # rm _build
```

- Config: `docs/myst.yml` (TOC + site), `[tool.griffe2md]` in `pyproject.toml`
- API generation script: `scripts/build_api_docs.py`
- Tutorials: `docs/tutorials/` — MyST Markdown with `{code-cell}` directives

## Sub-agents
- Instruct to use `uv run`, `symbex`, targeted TDD, `-n auto`, log files
- Slow tests require explicit user permission

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
uv run pytest nltools/tests/shell/test_brain_data.py::TestBrainData::test_fit -xvs
uv run pytest -k "ridge and cv" -x

# Default (non-slow, parallel):
uv run pytest -n auto

# By directory:
uv run pytest nltools/tests/shell/ -n auto -x

# All including slow (ask first):
uv run pytest -m "" -n auto

# Tests live in: shell/ (data classes), core/ (algorithms/stats), support/ (utilities), data/ (fixtures)
```
