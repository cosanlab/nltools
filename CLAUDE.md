# AGENTS

> `nltools` is a neuroimaging and statistical modeling Python library that focuses on ease-of-use, numerical correctness, and performance

## Operating Rules

- This project uses `uv` for Python environment and task management; run `uv run poe` for the current task list.
- Project work is tracked in `cosanlab/nltools` on GitHub. Use `gh` for issues and pull requests.
- Treat `pyproject.toml`, source code, and tests as authoritative when documentation disagrees.
- Docstrings use Google-style Markdown:
  - Start with a complete summary sentence of at most 120 characters.
  - Use `Args:`, `Returns:`, `Raises:`, `Examples:`, and `Note:`
  - Use Markdown code spans for references and fenced Python blocks for examples.
- Every function and class that is not on the user-facing surface has a leading underscore. An unprefixed function or class anywhere in `nltools/` is user-facing by construction, and a test under `nltools/tests/support` enforces it.
- Module filenames never begin with an underscore.
- Keep `__all__` in each user-facing namespace exactly equal to its designated members; internal modules have no `__all__`.
- Use the vendored `.claude/skills/nilearn` skill before writing, reviewing, or debugging nilearn code.
- Use the vendored marimo skills when editing notebooks under `docs/tutorials/`.

## Architecture

### Two layers

nltools has a functional core wrapped in an imperative shell, and the two serve different readers.

**Layer one, for maintainers.** Internal modules hold the implementation as pure functions and frozen dataclasses; the facade classes delegate to them and carry no numerical or domain logic of their own. Internal modules are the only source of implementation. Their functions and classes are underscore-prefixed and documented only in their docstrings.

**Layer two, for users.** The facade classes with their methods, plus a short designated list of standalone functions. Users are never expected to call an estimator, a delegate, or a helper directly; a return object (`Predict`, `BootstrapResult`, `ContrastResult`, `BrainSpaceConfig`) is data they read, not something they construct. A public-looking name does not make something user-facing; being listed here does.

Namespace | User-facing members
--- | ---
`nltools` | `BrainData`, `Adjacency`, `DesignMatrix`, `Roc`, `Simulator`, `SimulateGrid`, `concatenate`, the four brainspace functions, `__version__`
`nltools.data` | the six classes and the four result types
`nltools.algorithms` | the statistical functions inherited from v0.5.1's `stats` module, `compute_searchlight_neighborhoods`, `check_gpu_available`
`nltools.io` | `events_to_dm`
`nltools.datasets` | the fetchers, resource and atlas lookups
`nltools.mask` | the five mask functions
`nltools.cross_validation` | `KFoldStratified`
`nltools.plotting` | `component_viewer`
`nltools.utils` | the two warning classes

Adding to this table is an API decision, documented in the migration guide when a v0.5.1 user can see it. `docs/_data/api-vocabulary.yml` remains the authority for keyword names, defaults, keyword-only requirements, banned aliases, and documented exceptions on every user-facing signature. Consult it before adding or changing a user-facing signature. Edit the manifest, then regenerate its outputs. Do not edit rendered vocabulary tables.

Use explicit signatures for internal nltools calls. `**kwargs` is allowed only when forwarding arguments to a third-party API such as sklearn, nilearn, matplotlib, seaborn, nibabel, or pandas.

Keep trailing control arguments in this order when they apply:

```python
..., domain_kwargs, return_flags, n_jobs=-1, random_state=None, progress_bar=False
```

Some ridge and alignment internals retain legacy parameter names. Translate them at the facade boundary rather than renaming those internals casually.

Treat public names, signatures, defaults, and semantics as compatibility contracts. Do not introduce an intentional breaking change unless the task explicitly authorizes it. Document an approved break in the migration guide and use a conventional commit with `!` and a `BREAKING:` body entry.

The `uv run poe ok` gate includes the API checks required after changing a public signature or docstring.

### Internal modules

- Separate state from business logic without overengineering: frozen dataclasses for immutable state, pure functions for business logic.
- Domain-specific helpers live with their domain (`nltools/data/designmatrix/utils.py`, not `nltools/utils.py`); the shared helper modules hold only general-purpose code.
- The functional core never imports from a facade package.

## Documentation

The package version lives only in `pyproject.toml`.

The site is a home page, the tutorials, one API reference page per user-facing namespace, the migration guide, contributing and the changelog. Pages under `docs/api/` are mkdocstrings stubs whose `members:` lists mirror the table above; the strict docs build is the only check on them. Design notes and specifications under `docs/development/` are maintainer documents, not part of the site.

Marked `AUTOGEN` blocks are generated and committed. Change their source, then run the generator. Never edit generated output directly.

Marimo notebooks under `docs/tutorials/` are the tutorial sources. Edit the `.py`; the `.md` sibling is generated by `docs-generate` and not committed. `docs-build` executes every notebook and fails on a cell that raises or warns; `docs-serve` replays recorded outputs for notebooks whose cells did not change.

Use `uv run poe docs-generate` after changing the vocabulary manifest or a tutorial `.py` file. Use `uv run poe docs-build` when the change can affect the rendered site or executed tutorials.

## Hard invariants

- The reserved `.nl_` namespace applies only to columns generated inside a `DesignMatrix`. Create, recognize, and parse those names with `_reserved_name()`, `_run_separated_name()`, `_is_reserved_name()`, and `_parse_run_separated()`. Never identify generated `DesignMatrix` columns by matching user-controlled naming patterns.
- GPU execution means Himalaya ridge fitting (`_Ridge(device='gpu')`, `BrainData.fit(ridge_device='gpu')`) and the ridge bootstrap (`BrainData.bootstrap(device='gpu')`). nltools ships no GPU implementation of its own; permutation, ISC and alignment run on CPU workers. Memory budgeting, batch sizing, worker sizing and OOM recovery for those paths live only in `nltools/algorithms/backends.py`; an algorithm supplies its per-item working-set estimate and never its own budget arithmetic.
- On those ridge paths an explicit `device="gpu"` must run on the GPU or raise; there is no silent CPU fallback, and no `"auto"` value.
- Read the relevant design document before changing these subsystems:
  - `docs/development/execution-model.md` preserves deferred 0.6.1 `BrainCollection` execution design; it is not an active 0.6.0 subsystem
  - `docs/development/ridge-internals.md` for ridge backends and numerical behavior
  - `docs/development/inference-internals.md` for permutation tests, bootstrap tests, RNG behavior, and numerical stability
  - `docs/development/index.md` for the overall architecture

Update the corresponding document when an invariant or behavior changes.


## Workflow and gates

Tests cover what nltools itself does: the contract of each user-facing method (shapes, keyword semantics, errors it raises, return conventions), the invariants of the data classes, and the wiring between them. They do not re-derive results that sklearn, nilearn, scipy or Himalaya already guarantee; when nltools adds logic on top (masking, orientation, aggregation, index bookkeeping), one test pins that addition against a hand-computable case. They do not check documentation structure, page presence, export lists or docstring formatting; one designation test guards the user-facing surface. Prefer one test per behaviour over parametrised sweeps, and delete a test when the behaviour it guarded is removed.

Use red-green TDD for behavioral changes:

1. Write or identify a failing test.
2. Make the smallest change that passes it.
3. Rerun the focused test until it passes.
4. Run `uv run poe ok` before considering the change complete.

`uv run poe ok` is the project-wide completion gate. It fails fast in this order:

1. Ruff lint
2. Ruff format check
3. ty type checking
4. Public API checks: the vocabulary manifest, keyword-only markers, Semgrep rules, vocabulary table drift
5. The default fast test suite

Use `uv run poe` to find targeted test tasks during development. The `lint` task remains available when code needs automatic lint and formatting fixes.

For focused debugging, capture the result once and inspect the log:

```bash
uv run pytest <path-or-expression> -xvs 2>&1 | tee pytest.log
```

The default suite skips `slow` and `integration` tests. Do not run `uv run poe test-all` without checking first.
