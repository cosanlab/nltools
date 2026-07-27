---
# AUTO-GENERATED from 01_glm.py by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe docs-generate`.
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

```{code-cell} python3
:tags: [remove-input]
import sys

IN_WASM = sys.platform == "emscripten"
```

```{code-cell} python3
:tags: [remove-input]
# In-browser only: install nltools + its full runtime stack before any nltools import
# runs, then hand `wasm_ready` to every nltools-importing cell to force ordering. We
# can't rely on marimo's PEP 723 header auto-install alone: it races cell execution and
# marimo never re-runs a cell that already failed with ModuleNotFoundError.
wasm_ready = True
if IN_WASM:
    import asyncio

    import micropip
    import js

    async def _pip(reqs, **kw):
        # Install packages ONE AT A TIME instead of a single concurrent
        # micropip.install([...]) call. The big concurrent batch download
        # occasionally returns a truncated wheel (BadZipFile); micropip then
        # caches the corrupt bytes so an in-session retry keeps failing — and
        # marimo never re-runs an errored cell, permanently bricking the
        # page. Sequential installs keep peak download concurrency low and
        # sidestep the corruption; a per-package retry still rides out
        # ordinary network blips. (see nltools#455 investigation)
        items = [reqs] if isinstance(reqs, str) else list(reqs)
        for _item in items:
            for _attempt in range(3):
                try:
                    await micropip.install(_item, **kw)
                    break
                except Exception:  # noqa: BLE001
                    if _attempt == 2:
                        raise
                    await asyncio.sleep(0.75 * (_attempt + 1))

    # Install the stack UNPINNED so micropip takes Pyodide's bundled builds (pinning
    # to nltools' host versions, e.g. joblib>=1.5.3, fails against Pyodide's bundled
    # joblib). nilearn is the exception: 0.14+ needs packaging>=26 (absent in Pyodide
    # 0.27.7), so pin the last 0.13.x. numpy/scipy/pandas/sklearn/matplotlib come in
    # transitively at their bundled versions.
    await _pip(
        [
            "nibabel",
            "nilearn==0.13.1",
            "seaborn",
            "polars",
            "pynv",
            "huggingface-hub",
            "anywidget",
        ]
    )
    # deps=False installs the wheel without re-checking nltools' own version pins.
    await _pip(
        js.location.origin + "__NLTOOLS_WHEEL_URL__", deps=False
    )
```

# GLM Analysis

:::{tip} Interactive version
The outputs below are pre-computed. [**Open this tutorial as a live notebook →**](/tutorials/workflows-01_glm.html) to run and edit every cell in your browser (via marimo + WebAssembly).
:::

**What it answers.** *Where* in the brain does activity track your task design? The general linear model (GLM) is the mass-univariate workhorse of task fMRI: fit one regression per voxel, then test contrasts between conditions. Use it when you have a known design and want a statistical map of effects.

For the underlying theory, see the GLM chapters in [dartbrains](https://dartbrains.org). This tutorial is about *running* the analysis in nltools.
<!---->
**How it works.** A GLM analysis runs in two stages:

- **First level (single subject).** Regress each voxel's timeseries on the design matrix **X** → one β (and t) per regressor. A *contrast* is a linear combination of βs, giving a per-subject effect map.
- **Second level (group).** Stack the per-subject contrast **effect-size** maps and run a one-sample test across subjects.

Feed *effect sizes* (βs), not first-level t-maps, into the group test: a first-level t is `β / SE(β)`, and SE varies across subjects for reasons unrelated to the effect (scan length, motion). Stacking t-maps would conflate effect magnitude with first-level precision.

```{code-cell} python3
_ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
import numpy as np
from joblib import Memory

from nltools.data import BrainData, DesignMatrix
from nltools.stats import fdr, threshold
from nltools.utils import concatenate

# Memoize per-subject fits to disk (.cache/ is git-ignored) so re-running
# the notebook reloads results instead of refitting every voxel.
memory = Memory(".cache/tutorials", verbose=0)
```

## How to do it

We use the **language localizer demo** from `nilearn` — 10 subjects viewing blocks of sentences (`language`) vs. consonant strings (`string`). Each subject's BIDS derivatives give us three files: the preprocessed BOLD, an events TSV, and a confounds TSV. In the browser (Pyodide), the same analysis uses a trimmed copy of the eight subjects fitted below so it runs without a server.

```{code-cell} python3
:tags: [remove-input]
# In-browser only: seed the trimmed BIDS subset into the IDBFS cache and
# resolve each subject's files from it. `browser_get_sub_files` stays None
# locally, where the visible cell below loads from nilearn instead. Imports
# are underscore-aliased to keep them cell-local (marimo defines each name
# once across cells).
_ = wasm_ready  # ensure the nltools wheel is installed first (WASM)
browser_get_sub_files = None
if IN_WASM:
    import json as _json
    from pathlib import Path as _Path

    from nltools.templates import fetch_resource as _fetch, seed_resources as _seed

    _pyodide_subjects = [f"{_subject:02d}" for _subject in range(1, 9)]

    def _resource_paths(sub: str) -> dict:
        stem = f"sub-{sub}_task-languagelocalizer"
        return {
            "bold": f"tutorials/glm/derivatives/sub-{sub}/func/{stem}_desc-preproc_bold.nii.gz",
            "sidecar": f"tutorials/glm/derivatives/sub-{sub}/func/{stem}_desc-preproc_bold.json",
            "confounds": f"tutorials/glm/derivatives/sub-{sub}/func/{stem}_desc-confounds_regressors.tsv",
            "events": f"tutorials/glm/sub-{sub}/func/{stem}_events.tsv",
        }

    _glm_resources = [
        _relpath
        for _sub in _pyodide_subjects
        for _relpath in _resource_paths(_sub).values()
    ] + [
        # MNI templates the resample + slice plots fetch — pre-seed in Pyodide.
        # Both 2mm (BrainData default brainspace) and 3mm are covered.
        "default/2mm-MNI152-2009fsl-mask.nii.gz",
        "default/2mm-MNI152-2009fsl-brain.nii.gz",
        "default/2mm-MNI152-2009fsl-T1.nii.gz",
        "default/3mm-MNI152-2009fsl-mask.nii.gz",
        "default/3mm-MNI152-2009fsl-brain.nii.gz",
        "default/3mm-MNI152-2009fsl-T1.nii.gz",
    ]
    await _seed(_glm_resources)

    _pyodide_files = {}
    for _sub in _pyodide_subjects:
        _relpaths = _resource_paths(_sub)
        _sidecar = _fetch(_relpaths["sidecar"])
        _pyodide_files[_sub] = {
            "bold": _fetch(_relpaths["bold"]),
            "events": _fetch(_relpaths["events"]),
            "confounds": _fetch(_relpaths["confounds"]),
            "TR": _json.loads(_Path(_sidecar).read_text())["RepetitionTime"],
        }

    def browser_get_sub_files(sub: str) -> dict:
        """Resolve one subject's trimmed browser-ready tutorial files."""
        return _pyodide_files[sub]
```

```{code-cell} python3
import json
from pathlib import Path

from nilearn.datasets import fetch_language_localizer_demo_dataset
from nilearn.interfaces.bids import get_bids_files

if IN_WASM:
    get_sub_files = browser_get_sub_files
else:
    DATASET = fetch_language_localizer_demo_dataset(verbose=0)
    DATA_DIR = Path(DATASET["data_dir"])

    def get_sub_files(sub: str) -> dict:
        """Resolve one subject's BOLD, events, confounds, and TR from BIDS."""
        derivatives = DATA_DIR / "derivatives"
        sidecar = get_bids_files(derivatives, file_tag="bold", file_type="json", sub_label=sub)[0]
        return {
            "bold": get_bids_files(derivatives, file_tag="bold", file_type="nii.gz", sub_label=sub)[0],
            "events": get_bids_files(DATA_DIR, file_tag="events", file_type="tsv", sub_label=sub)[0],
            "confounds": get_bids_files(derivatives, file_type="tsv", modality_folder="func", sub_label=sub)[0],
            "TR": json.loads(Path(sidecar).read_text())["RepetitionTime"],
        }
```

### First level (single subject)

The recipe for one subject: load the BOLD (`BrainData` resamples to standard MNI automatically), build the design, and fit. Building a `DesignMatrix` from a BIDS events file creates boxcar regressors and **convolves them with the canonical (Glover) HRF for you** — columns come back as `language_c0` / `string_c0` (pass `hrf_model=None` for raw boxcars to `.convolve()` yourself). We append the motion confounds as nuisance columns and add polynomial drift. Wrapping it in `memory.cache` means each subject is fit once, then reloaded from disk.

```{code-cell} python3
@memory.cache
def first_level(sub: str, contrast: str = "language_c0 - string_c0"):
    """Fit one subject's GLM; return its design and the contrast bundle.

    We return only the lightweight design and contrast maps (not the
    fitted model, which carries residuals and a copy of the data) so the
    on-disk cache stays small.
    """
    f = get_sub_files(sub)
    brain = BrainData(f["bold"])
    events = DesignMatrix(f["events"], run_length=brain.shape[0], TR=f["TR"])
    confounds = DesignMatrix(f["confounds"], run_length="infer", TR=f["TR"])
    brain.fit(X=events.append(confounds, axis=1, as_confounds=True).add_poly(2))
    return brain.design_matrix, brain.compute_contrasts(contrast, statistic="all")
```

```{code-cell} python3
design, contrasts = first_level("01")
design.plot()  # the design we just fit
```

The helper returns the `language > string` contrast as a bundle — `beta`, `t`, `z`, `p`, `se` — computed in one call with `statistic="all"`, so we can threshold the t-map here *and* reuse the β map for the group analysis below.

```{code-cell} python3
contrasts["t"].plot(method="slices", threshold=3.09, title="sub-01: language > string (t)")
```

Even at one subject the left-lateralized fronto-temporal language network is visible (`|t| > 3.09`, two-tailed p ≈ 0.001).

### Second level (group)

The same cached recipe runs per subject, returning one **effect-size** (β) map each. We loop over eight of the ten demo subjects.

```{code-cell} python3
SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08"]
beta_maps = []
for sub in SUBJECTS:
    _, sub_contrasts = first_level(sub)
    beta_maps.append(sub_contrasts["beta"])
```

`concatenate` stacks the per-subject maps into one `(n_subjects, n_voxels)` `BrainData`. `BrainData.ttest` runs a voxelwise one-sample test, returning the effect-size `mean`, the parametric `t`, a signed `z`, and `p`. `nltools.stats.threshold` keeps the `z` values whose `p` clears a cutoff — here voxelwise `p < 0.001`.

```{code-cell} python3
group = concatenate(beta_maps)
group_result = group.ttest()
group_z = threshold(group_result["z"], group_result["p"], thr=0.001)
group_z.plot(method="slices", title="Group: language > string (voxelwise p < 0.001)")
```

### Multiple-comparisons correction

That `p < 0.001` map is *uncorrected* — it ignores that we ran tens of thousands of tests. `nltools.stats.fdr` returns the p-threshold controlling the false-discovery rate. Whole-brain correction is stringent: on a ten-subject demo, far fewer voxels survive than at the uncorrected threshold — exactly the inflation that correction guards against. Restricting the search to an ROI (see the [MVPA tutorial](workflows-03_mvpa.html)) recovers power.

```{code-cell} python3
p_values = np.asarray(group_result["p"].data)
n_voxels = p_values.size
fdr_thr = fdr(p_values, q=0.05)
bonf_thr = 0.05 / n_voxels

n_uncorrected = int((p_values < 0.001).sum())
n_fdr = int((p_values <= fdr_thr).sum()) if fdr_thr > 0 else 0
n_bonferroni = int((p_values < bonf_thr).sum())

print(f"voxels surviving, out of {n_voxels}:")
print(f"  uncorrected (p < 0.001):  {n_uncorrected:5d}")
print(f"  FDR (q = 0.05):           {n_fdr:5d}")
print(f"  Bonferroni (p < 0.05/N):  {n_bonferroni:5d}")
```

## Recap

| Stage | What it does | Key API |
|---|---|---|
| Build design | BIDS events → HRF-convolved regressors + confounds + drift | `DesignMatrix(events, run_length=, TR=)`, `.append(confounds, axis=1, as_confounds=True)`, `.add_poly()` |
| First level | OLS at every voxel | `brain.fit(X=design)` |
| Contrast | Linear combination of βs (effect size + inference) | `brain.compute_contrasts("A - B", statistic="all")` |
| Stack subjects | Concatenate first-level β maps | `concatenate([...])` |
| Group test | Voxelwise one-sample t-test → `{mean, t, z, p}` | `group.ttest()` |
| Correction | FDR threshold | `nltools.stats.fdr`, `nltools.stats.threshold` |

The per-subject loop is the explicit path; `BrainCollection` will wrap multi-subject fitting into a single call once it lands on this branch.

**Next steps**

- [Encoding models](workflows-02_encoding.html) — predict brain activity *from* stimulus features (GLM vs. Ridge).
- [Multivariate pattern analysis](workflows-03_mvpa.html) — decode conditions and compare representational geometry.
- [Inter-subject correlation](workflows-04_isc.html) — shared responses to naturalistic stimuli.
