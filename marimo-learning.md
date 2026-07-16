# marimo WASM for nltools docs — spike notes

_Exploration of browser-only marimo notebooks as a possible successor to the
JupyterLite "Try it live" layer. Dated 2026-07-16 (branch `uv-cleanup`)._

## TL;DR

marimo's `export html-wasm` is a **viable replacement for the JupyterLite sandbox**:
the full nltools scientific stack micropip-installs in-browser, the existing
`seed_resources`/IDBFS dataset code ports **with zero changes**, and nilearn
glass-brain plots render. It's a **lateral UX/maintainability move** (nicer editor,
reactive model, git-diffable `.py` source), **not a capability unlock** — JupyterLite
already works. Keep MyST + griffe2md for the docs/API site regardless; marimo only
competes with the interactive layer.

---

## Two different things called "marimo wasm" — don't conflate

1. **`marimo export html-wasm <nb>.py`** — a marimo *core feature*. One `.py`
   notebook → one self-contained page that boots Pyodide and runs live. This is the
   **direct analog to our JupyterLite "Try it live" bundle**, and what the spike tested.
2. **marimo-book** (the tool at <https://marimobook.org>) — a whole static-site
   **generator** built on Material-for-MkDocs. It would replace **MyST itself**, not
   just JupyterLite. Its default is *static render at build time* (bake outputs, no
   kernel), WASM opt-in per chapter. Philosophically the same as our MyST
   `{code-cell}` executed-at-build tutorials, so it buys little on the static side
   while costing us the entire griffe2md/mystmd/git-cliff pipeline. **Not recommended.**

marimo-book's own docs warn Pyodide is "missing nibabel, nilearn, statsmodels." That's
about the *default Pyodide bundle* — micropip installs the pure-Python ones (nibabel,
nilearn) fine, and **nltools has no statsmodels dependency at all**. So that disclaimer
does not bite us. The real ceiling is the wasm32 **2 GB memory cap** (fine for small
tutorial data, a hard no for real fMRI datasets) — identical for JupyterLite and marimo.

---

## What the spike did

Adapted `docs/tutorials/basics/01_brain_data.md` into a marimo `.py` notebook and
exported it to html-wasm, driven in a real browser (playwright-cli) to confirm it
actually executes — not just that it builds.

- marimo 0.23.10, exported via `uvx marimo export html-wasm 01_brain_data.py -o site --mode run`
- Deps installed in-cell via `micropip` (mirroring the JupyterLite piplite list)
- nltools v0.6.0 dev wheel (`uv build --wheel`) served alongside and micropip-installed with `deps=False`
- Data sourced from the seeded MNI template (`seed_resources` → HF `nltools/niftis`),
  **not** `fetch_pain()` (see gotcha #2)

Spike files live in the scratchpad: `marimo-spike/01_brain_data.py` + `site5/`.

## Confirmed working in-browser (marimo run mode, Pyodide / Python 3.14)

- ✅ **Full nltools dependency stack micropip-installs** — `nilearn==0.13.1`,
  `nibabel`, `scipy`, `scikit-learn`, `pandas`, `polars`, `seaborn`, `pynv`,
  `matplotlib`. No resolution errors. (marimo ships a very recent Pyodide — Python
  **3.14**, vs JupyterLite's 3.12.)
- ✅ **`seed_resources`/IDBFS ports as-is, no code change.** `"pyodide" in sys.modules`
  is true under marimo's worker; the async seed + sync `fetch_resource` + IndexedDB
  persistence all worked. This was the biggest de-risking win — the hard part of our
  JupyterLite setup carried over for free.
- ✅ **`BrainData` constructs from a seeded nifti**; `.mean/.std/.threshold/.standardize`
  all ran; in-memory multi-image `BrainData` (indexing, arithmetic) ran.
- ✅ **nilearn glass-brain `.plot()` renders.** It returns a `matplotlib.figure.Figure`,
  which marimo displays as a PNG. Plain matplotlib figures render too. (3 figures
  rendered end-to-end: plain-mpl probe + glass brain + thresholded glass brain.)
- ✅ Zero JS console errors; clean self-contained bundle (~27 MB incl. wheel).

## Gotchas found (each cost a debug cycle — record for next time)

1. **`--mode edit` does NOT auto-run cells.** Edit-mode html-wasm exports open with
   cells stale (`.marimo-cell.needs-run`); they wait for the user to run them. Only
   **`--mode run`** auto-instantiates the whole DAG. For a "try it live" docs page you
   want `run` mode (read-only app, code hidden by default), or you must tell readers to
   hit "Run all". Independent `mo.md` cells render regardless, which masks the problem.
2. **`fetch_pain()` / neurovault is not browser-viable.** It does synchronous HTTP
   (`nilearn.fetch_neurovault_ids`), which Pyodide can't do. Same limitation as
   JupyterLite — it's why the *basics* tutorials aren't in the current JupyterLite
   bundle. Browser tutorials must seed from our HF `nltools/niftis` repo via
   `seed_resources`, or use the in-memory `load_haxby_example()`.
3. **Every in-browser `BrainData(...)` needs the default MNI mask pre-seeded.**
   `BrainData.__init__` → `initialize_mask` auto-loads `default/2mm-MNI152-2009fsl-mask.nii.gz`.
   Seed the mask/brain/T1 triple, mirroring `nltools/tests/pyodide/test_runner.mjs`,
   or construction raises `RuntimeError: Resource ... not in the Pyodide cache`.
4. **Wheel URL must resolve against the origin, not `location.href`.** micropip cells
   run in the Pyodide **web worker**, where `js.location.href` is the *worker script*
   URL (`.../assets/worker-*.js`), not the page. `dirname(href)` → `/assets/…` → 404
   HTML → `zipfile.BadZipFile: File is not a zip file`. Use `js.location.origin` (shared
   between page and worker) + an absolute path, or hardcode the wheel's absolute URL.
5. **Cold install is slow** — several minutes on first load (PyPI wheel downloads),
   ~30 s on warm HTTP cache. Comparable to JupyterLite's ~1 min. Pyodide packages and
   the seeded niftis are cached (the latter in IndexedDB across reloads); the PyPI wheels
   rely on the browser HTTP cache.

## marimo html-wasm vs JupyterLite (the interactive layer)

| | JupyterLite (current) | marimo html-wasm |
|---|---|---|
| Notebook source | `.ipynb` (JSON, merge-hostile) | `.py` (git-diffable — big win) |
| Execution model | classic, hidden out-of-order state | reactive DAG (no stale-cell bugs) |
| Payload | full JupyterLab | per-notebook, lighter |
| UX | IDE-in-browser | clean app/notebook view, `edit`/`run` toggle |
| Dataset seeding | our IDBFS code | **same code, ported for free** |
| Auto-run on load | yes | only in `--mode run` |
| Maturity with our stack | working today | proven by this spike |

## Recommendation

- **Do not** replace MyST with marimo-book. Too immature, MkDocs-based, no story for
  our griffe2md auto-generated API reference. Wholesale swap = discard real investment
  for a marginal static-render gain.
- **marimo html-wasm is a real candidate** to replace the JupyterLite sandbox when we
  want the `.py` source + reactive UX. Not urgent — JupyterLite works — so it's a
  "when we choose to," not a "must."
- **Most interesting middle path:** marimo notebooks can embed as WASM **islands**
  inside an existing site. Keep MyST for prose + the full API reference; drop live
  marimo islands into just the interactive spots. Sidesteps the all-or-nothing framing.

## Productionization findings (2026-07-16, pt 2 — building the real pipeline)

Resolved the two open unknowns from the spike; details for the build script.

1. **`fetch_pain()` IS browser-viable now.** Gotcha #2 above is **stale** — the pain
   & emotion datasets were migrated off Neurovault onto HF `nltools/niftis`
   (`nltools/datasets.py`; commits `0a9e65bb`, `a3889777`). `fetch_pain()` now loads
   via `fetch_resource()` (cache-backed), and its docstring documents the Pyodide path:
   `await seed_resources(PAIN_RESOURCES)` (the 84-image manifest, `datasets.py:46`) then
   call it. So the basics tutorials seed `PAIN_RESOURCES` + the MNI triple at boot and use
   `fetch_pain()` directly.
2. **`--execute` bakes outputs, but only from PEP 723 deps — it ignores the ambient
   env** (even with `--no-sandbox`: `No module named 'nltools'`). To bake, PEP 723 must
   carry `nltools @ file://<abs>/dist/nltools-*.whl` — then `--execute` builds an isolated
   uv env, installs the wheel, runs, and embeds outputs (verified: baked marker in
   `index.html`). Use a `; sys_platform != 'emscripten'` marker so the browser skips the
   file:// dep; the in-browser micropip cell installs nltools from the hosted URL instead.
   marimo warns `nltools` "not bundled in the Pyodide lockfile" — expected/harmless.
3. **`-o <name>.html` is NOT self-contained** — the 24 KB html references a sibling
   `./assets/` dir (~27 MB marimo frontend, 686 content-hashed files). Exporting **all
   notebooks into one shared output dir** yields a single shared `assets/` + N tiny htmls
   (idempotent re-copy since assets are content-hashed). So: flat `tutorials/<name>.html`
   layout, one `assets/`, deploy size ~27 MB for the whole set (+ the wheel served once).
4. Version is `0.5.1` in `pyproject.toml` (bumped only at release) — the build script
   must **glob** the wheel (`dist/nltools-*.whl`, newest by mtime), never hardcode a version.

## If we productionize this

- Author tutorials as marimo `.py`, export with `--mode run`.
- Host one shared nltools wheel at a stable https URL (like JupyterLite's piplite wheel);
  reference it by absolute URL from the install cell (gotcha #4).
- Pre-seed the default MNI triple + any tutorial-specific niftis at notebook boot
  (gotcha #3), reusing `nltools.templates.seed_resources` unchanged.
- Keep the `nilearn==0.13.1` pin (packaging<26 vs bundled 24.2) until Pyodide's bundled
  `packaging` moves — verify against whatever Pyodide version marimo ships at the time.
- Add a browser smoke test analogous to `test_runner.mjs` that loads the exported page
  and asserts a figure renders (the playwright-cli flow used here).
