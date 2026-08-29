# `0.6.1-browser` — backup of the in-browser (marimo WASM / Pyodide) support

This branch preserves everything that was **removed from v0.6.0** so the release could ship. It forks from `5eefd193`, the last commit with all of it intact, and is not meant to be merged as-is — see the revival plan in the tracking issue.

**Tracking issue: [cosanlab/nltools#487](https://github.com/cosanlab/nltools/issues/487)**

What lives here and nowhere else on `master`:

- `scripts/build_marimo_wasm.py`, the `docs-wasm` poe task, and the WASM steps in `docs-deploy.yml`
- The `IN_WASM` / `wasm_ready` / `seeded` / `browser_*` cells in every notebook under `docs/tutorials/`
- `nltools.templates.seed_resources` and the Pyodide/IDBFS fetch path in `nltools/templates/fetch.py`
- `nltools.datasets.PAIN_RESOURCES`, `EMOTION_METADATA`, `emotion_resources()`
- `nltools/tests/pyodide/`, the `pyodide` CI job, and the `test-pyodide` poe task
