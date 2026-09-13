# 0.6.0 release verification matrix

Approved through Kata `9kgh` on 2026-09-08 and executed by `q62r` at the final
release revision. Every row records its command, where it runs, and the
outcome that counts as a pass. Publishing, tagging, and pushing remain separate
decisions.

## Scope decisions

- The fast gate runs after every slice. The complete `slow` suite runs once at
  the final revision; slices run only the slow tests in the areas they touch.
- The `integration` marker currently selects no tests, so there is no separate
  integration run. `test-all` is equivalent to the fast gate plus `slow`.
- Retained GPU claims are verified on both supported accelerators: MPS on the
  local Apple M3 and CUDA on `pika` (NVIDIA GB10). Neither substitutes for the
  other. `neuron` has no accelerator and is not used.
- Numerical tolerances are the suite's existing dtype-specific tolerances. No
  new thresholds are introduced for release verification.
- The alignment result container decision is deferred to the `q31x` audit;
  the audit records the contract before this matrix runs.

## Matrix

| Area | Command | Where | Pass condition |
| --- | --- | --- | --- |
| Project gate | `uv run poe ok` | local | Lint, format, types, API checks, and fast suite pass. |
| Slow suite | <code>uv run pytest -m slow -n auto --maxprocesses=4 2&gt;&amp;1 &#124; tee slow.log</code> | local | All selected slow tests pass (432 passed, 5 skipped, about 6.5 minutes on the M3 at `0a4a6d3b`); every skip names an approved limitation below. |
| MPS detection | `uv run python -c "from nltools.algorithms.backends import check_gpu_available as c; print(c())"` | local | Reports `device='mps'`. |
| MPS execution | <code>uv run pytest -m '' nltools/tests/models/test_ridge.py nltools/tests/data/braindata/test_braindata_bootstrap.py -k "gpu or mps or device" 2&gt;&amp;1 &#124; tee mps.log</code> | local | GPU-marked and `skipif`-gated tests run rather than skip, and pass. GPU execution is Ridge fitting and the ridge bootstrap only; explicit `device="gpu"` runs on MPS or raises, and nothing falls back to CPU silently. |
| CUDA detection | Same detection command after `uv sync` in a fresh checkout of the release revision | `pika` | Reports `device='cuda'` with the GB10 device name. |
| CUDA execution | <code>uv run pytest -m '' nltools/tests/models/test_ridge.py nltools/tests/data/braindata/test_braindata_bootstrap.py -k "gpu or cuda or device" 2&gt;&amp;1 &#124; tee cuda.log</code> | `pika` | Same pass condition as MPS. Ridge fitting, banded ridge, and the ridge bootstrap execute on CUDA. |
| Executed tutorials | `uv run poe docs-build` | local | Site builds under `--strict`; all seven tutorials execute with no stderr. |
| Tutorial scripts | `uv run poe tutorials` | local | Static checks pass and every notebook runs end to end as a script. |
| Persistence | `uv run pytest nltools/tests/io_tests nltools/tests/data -k "h5 or hdf or nifti or write or load"` | local | HDF5 and NIfTI round trips pass. |
| Packaging | `uv build`, then install the wheel into a fresh venv and run the smoke test from `scripts/release.py` (`SMOKE_TEST_CODE`) with the expected version | local | Wheel installs, version matches `pyproject.toml`, `DesignMatrix` and `Adjacency` construct. |
| Pyodide | `uv run poe test-pyodide` | local | Needs node and the network. The wheel's declared dependencies all resolve under Pyodide 314, nltools imports, the data classes construct, and `fetch_resource` downloads into a cache that survives an IDBFS remount. |
| Export inventory | `uv run poe lint-api` plus `uv run python -c "import nltools.data as d; assert 'BrainCollection' not in d.__all__"` | local | API checks pass; `BrainCollection` is absent from public exports and `docs/api`; `q31x` has no unresolved findings. |
| Migration examples | Run each code block in `docs/migration-guide.md` that shows 0.6.0 behavior in a scratch script | local | Every example runs and prints or asserts what the guide claims. |
| Migration downloads | Run the one non-executed fence in `docs/migration-guide.md` (`fetch_pain()` and `fetch_neurovault_collection(504)`, under `### IO and datasets`) in a scratch script | local | Both fetchers download; `fetch_pain()` returns a `BrainData` and `fetch_neurovault_collection(504)` returns a `(metadata, files)` pair. The docs build skips this fence because it needs the network. |

Record every log, the executed revision hash, and machine details in the
`q62r` close message.

## CUDA setup on pika

`pika` has no nltools checkout and no `uv`. Verification installs `uv` for the
current user, clones the repository into the user's home, checks out the exact
release revision, and runs `uv sync`. This touches only the user's home
directory and adds no system configuration. Remove the checkout after
verification unless it is still wanted.

## Approved limitations

- `marimo check` reports 54 `markdown-indentation` warnings across the seven
  unchanged tutorial notebooks, with exit status 0. They are a formatting
  preference in the notebook source, never reach the built pages, and are not
  release blockers.
- CUDA Ridge arithmetic is `float32`. CPU/GPU parity is judged within the
  suite's explicit tolerances, never byte for byte.
- Real-data integration workflows beyond the executed tutorials are not part
  of 0.6.0 verification because no `integration` tests exist.
