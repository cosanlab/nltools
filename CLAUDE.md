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
- Always keep module level exports i.e. `__all__` in `__init__.py` up-to-date
- User-facing functions, methods, and class names must never use a `_` prefix
- Use the vendored `.claude/skills/nilearn` skill before writing, reviewing, or debugging nilearn code.
- Use the vendored marimo skills when editing notebooks under `docs/tutorials/`.

## Architecture

`nltools` uses a functional core with an imperative shell that separates public API from internal functionality:

### Public API

- `nltools.data`: contains the stateful class facades: `BrainData`, `Adjacency`, `DesignMatrix`, and `BrainCollection`. These classes and their methods are the **primary** user-facing surface
  - Facade methods *delegate* to internal modules and should not contain numerical or domain logic of their own.
- `nltools.algorithms`: contains statistical functions and models that serve as the **secondary** user-facing surface
- `nltools.{cross-validation, datasets, mask}`: contain additional helper functions also part of the **secondary** user-facing surface

`docs/_data/api-vocabulary.yml` is the authority for public keyword names, defaults, keyword-only requirements, banned aliases, and documented exceptions. Consult it before adding or changing a public signature. Edit the manifest, then regenerate its outputs. Do not edit rendered vocabulary tables.

Use explicit signatures for internal nltools calls. `**kwargs` is allowed only when forwarding arguments to a third-party API such as sklearn, nilearn, matplotlib, seaborn, nibabel, or pandas.

Keep trailing control arguments in this order when they apply:

```python
..., domain_kwargs, return_flags, n_jobs=-1, random_state=None, progress_bar=False
```

Some ridge and alignment internals retain legacy parameter names. Translate them at the facade boundary rather than renaming those internals casually.

Treat public names, signatures, defaults, and semantics as compatibility contracts. Do not introduce an intentional breaking change unless the task explicitly authorizes it. Document an approved break in the migration guide and use a conventional commit with `!` and a `BREAKING:` body entry.

The `uv run poe ok` gate includes the API checks required after changing a public signature or docstring.

### Internal Modules

- All other modules and sub-modules contain _internal_ functionality that supports the user-facing API
- These modules should be the **only** source of implementation; logic should not be duplicated in the user-facing API
- Module filenames must not begin with an underscore, but internal functions and methods may
- Separate state from business logic without overengineering:
  - Use frozen dataclasses for immutable state containers
  - Writing pure functions for business logic

## Documentation

The package version lives only in `pyproject.toml`.

`docs/api/` and marked `AUTOGEN` blocks are generated and committed. Change their source, then run the generator. Never edit generated output directly.

Marimo notebooks under `docs/tutorials/{basics,workflows}/` are tutorial sources. Edit the `.py` notebook, not its generated `.md` sibling.

Use `uv run poe docs-generate` after changing docstrings, the vocabulary manifest, or tutorial `.py` files. Use `uv run poe docs-build` when the change can affect the rendered site or executed tutorials.

## Hard invariants

- The reserved `.nl_` namespace applies only to columns generated inside a `DesignMatrix`. Create, recognize, and parse those names with `reserved_name()`, `run_separated_name()`, `is_reserved_name()`, and `parse_run_separated()`. Never identify generated `DesignMatrix` columns by matching user-controlled naming patterns.
- GPU execution is centralized in `nltools/algorithms/backends.py`. Memory budgeting, batch sizing, worker sizing, and OOM recovery belong there. Algorithms provide working-set estimates but must not implement their own budget calculations.
- An explicit `device="gpu"` or `parallel="gpu"` must run on the GPU or raise. Only `"auto"` may fall back.
- Read the relevant design document before changing these subsystems:
  - `docs/development/execution-model.md` for `BrainCollection` execution, caching, serialization, and parallel writes
  - `docs/development/ridge-internals.md` for ridge backends and numerical behavior
  - `docs/development/inference-internals.md` for permutation tests, bootstrap tests, RNG behavior, and numerical stability
  - `docs/development/index.md` for the overall architecture

Update the corresponding document when an invariant or behavior changes.


## Workflow and gates

Use red-green TDD for behavioral changes:

1. Write or identify a failing test.
2. Make the smallest change that passes it.
3. Rerun the focused test until it passes.
4. Run `uv run poe ok` before considering the change complete.

`uv run poe ok` is the project-wide completion gate. It fails fast in this order:

1. Ruff lint
2. Ruff format check
3. ty type checking
4. Public API and Semgrep checks
5. The default fast test suite

Use `uv run poe` to find targeted test tasks during development. The `lint` task remains available when code needs automatic lint and formatting fixes.

For focused debugging, capture the result once and inspect the log:

```bash
uv run pytest <path-or-expression> -xvs 2>&1 | tee pytest.log
```

The default suite skips `slow` and `integration` tests. Do not run `uv run poe test-all` without checking first.
