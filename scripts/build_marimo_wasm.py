#!/usr/bin/env python
"""Export the marimo tutorial notebooks to WASM-powered HTML for the docs site.

The marimo ``.py`` notebooks under ``docs/tutorials/`` are the single source of
truth for the tutorials. Each one is exported to a self-contained, in-browser
notebook via ``marimo export html-wasm`` (Pyodide) and served as a static page
alongside the MyST site — replacing both the old ``marimo_to_myst`` ``.md``
render pipeline and the JupyterLite "Try it live" bundle.

Dependencies in the browser (the hard-won part — see the two failure modes below):

* The notebook's PEP 723 header is deliberately pinned to **Pyodide-compatible**
  versions (``numpy==2.0.2`` = the numpy bundled by marimo's Pyodide 0.27.7 worker;
  ``nilearn==0.13.1``; other compiled packages left *unpinned* so Pyodide/micropip
  resolve them to their bundled versions — never pin numpy/scipy/pandas/sklearn to
  arbitrary versions, Pyodide only ships one build of each). marimo bakes this exact
  header into the browser's micropip install list, so it must stay Pyodide-safe.
* We therefore export with **``--no-execute``**. ``--execute`` bakes cell outputs
  into the page (a nice pre-boot preview), but marimo runs it in an env whose
  *resolved* versions it then freezes into that same browser install list — which
  overwrites the safe pins with the ambient dev env's too-new versions (e.g.
  ``numpy==2.4.3``), and Pyodide can't provide them, so ``import numpy`` fails and
  the kernel dies. ``--no-execute`` embeds the header verbatim. The WASM page runs
  live in-browser instead of showing a baked preview; the static ``.md`` tutorials
  (the default doc build) already carry real baked outputs via MyST execute.
* ``nltools`` itself is not on PyPI at this dev version. The ``IN_WASM`` setup cell
  micropip-installs the freshly-built dev wheel (``deps=False``) from a hosted URL;
  the committed notebook carries a ``__NLTOOLS_WHEEL_URL__`` placeholder we replace
  with the origin-relative path where the wheel is served. (An earlier version
  instead injected ``nltools @ file://<abs-wheel>`` into the PEP 723 header for a
  build-time ``--execute`` env; marimo froze that into the browser list too, and
  Pyodide failed to fetch the local ``file://`` path over HTTP — the #3952 bug.)
* ``--no-sandbox`` keeps the export itself in the ambient env (which already has
  ``marimo``) rather than spinning up a throwaway ``uv run --isolated`` env.

Output layout (flat, one shared ``assets/`` bundle for all notebooks)::

    <html_dir>/tutorials/
        assets/                     # marimo frontend (~27 MB, shared, content-hashed)
        wheels/nltools-*.whl        # the dev wheel, served once
        basics-01_brain_data.html
        workflows-01_glm.html
        ...

Usage::

    python scripts/build_marimo_wasm.py                       # all notebooks, base-url /
    python scripts/build_marimo_wasm.py --base-url /nltools/  # GitHub Pages subpath
    python scripts/build_marimo_wasm.py --filter 01_brain     # just matching notebooks
    python scripts/build_marimo_wasm.py --execute             # bake outputs (BREAKS browser
                                                              #   compat — see module docstring)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HTML_DIR = REPO_ROOT / "docs" / "_build" / "html"
OUT_SUBDIR = "tutorials"  # under <html_dir>
WHEEL_URL_PLACEHOLDER = "__NLTOOLS_WHEEL_URL__"

# (group, glob) — group name prefixes the output filename to avoid 01_* collisions.
TUTORIAL_SOURCES = [
    ("basics", "docs/tutorials/basics/[0-9]*.py"),
    ("workflows", "docs/tutorials/workflows/[0-9]*.py"),
]


def newest_wheel() -> Path:
    """Return the most recently built ``dist/nltools-*.whl`` (never hardcode version)."""
    wheels = sorted(
        (REPO_ROOT / "dist").glob("nltools-*.whl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not wheels:
        raise SystemExit("no nltools wheel in dist/ — run `uv build --wheel` first")
    return wheels[-1]


def build_wheel() -> Path:
    """Build a fresh dev wheel and return its path."""
    subprocess.run(["uv", "build", "--wheel"], cwd=REPO_ROOT, check=True)
    return newest_wheel()


def resolve_targets(filter_str: str | None) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []
    for group, pattern in TUTORIAL_SOURCES:
        for nb in sorted(REPO_ROOT.glob(pattern)):
            if filter_str and filter_str not in nb.name:
                continue
            targets.append((group, nb))
    return targets


def export_one(
    group: str,
    notebook: Path,
    out_dir: Path,
    wheel_url: str,
    execute: bool,
) -> dict:
    """Template + export one notebook to ``<out_dir>/<group>-<name>.html``."""
    src = notebook.read_text()
    src = src.replace(WHEEL_URL_PLACEHOLDER, wheel_url)

    out_name = f"{group}-{notebook.stem}.html"
    out_path = out_dir / out_name

    with tempfile.TemporaryDirectory() as td:
        tmp_nb = Path(td) / notebook.name
        tmp_nb.write_text(src)
        cmd = [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(tmp_nb),
            "-o",
            str(out_path),
            "--mode",
            "run",
            "--show-code",
            # Export in the ambient dev env (which already has marimo) rather than a
            # throwaway `uv run --isolated` PEP 723 env.
            "--no-sandbox",
            # --execute would freeze the ambient env's (too-new) resolved versions into
            # the browser micropip list, breaking Pyodide; --no-execute keeps the
            # Pyodide-safe header pins. See the module docstring.
            "--execute" if execute else "--no-execute",
            "-f",
        ]
        # marimo prints progress to stderr; surface it but don't capture (long).
        subprocess.run(cmd, check=True)

    # marimo drops a stray CLAUDE.md / logo.png next to the PWA assets — clutter we
    # don't want deployed. (The favicons + *.webmanifest are real PWA assets; keep.)
    for junk in ("CLAUDE.md", "logo.png"):
        (out_dir / junk).unlink(missing_ok=True)

    return {"group": group, "name": notebook.stem, "file": out_name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="/",
        help="Site root path the notebooks are served under (e.g. /nltools/). "
        "Must start and end with / (default: /).",
    )
    parser.add_argument(
        "--filter", help="only export notebooks whose filename contains this"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="bake cell outputs as a pre-boot preview. WARNING: breaks in-browser "
        "execution — marimo freezes the ambient env's resolved dep versions into the "
        "Pyodide micropip list, which Pyodide can't satisfy. Default: off (--no-execute).",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=HTML_DIR,
        help="MyST HTML output dir to write tutorials/ into (default: docs/_build/html)",
    )
    args = parser.parse_args()

    base_url = args.base_url
    if not (base_url.startswith("/") and base_url.endswith("/")):
        parser.error("--base-url must start and end with '/' (e.g. / or /nltools/)")

    targets = resolve_targets(args.filter)
    if not targets:
        parser.error("no notebooks matched")

    out_dir = args.html_dir / OUT_SUBDIR
    wheels_dir = out_dir / "wheels"
    out_dir.mkdir(parents=True, exist_ok=True)
    wheels_dir.mkdir(parents=True, exist_ok=True)

    wheel = build_wheel()
    shutil.copy2(wheel, wheels_dir / wheel.name)
    wheel_url = f"{base_url}{OUT_SUBDIR}/wheels/{wheel.name}"
    print(f"wheel: {wheel.name}  →  served at {wheel_url}")

    manifest = []
    for group, nb in targets:
        print(f"\n=== exporting {group}/{nb.name} ===")
        manifest.append(export_one(group, nb, out_dir, wheel_url, args.execute))
        print(f"  → {out_dir}/{group}-{nb.stem}.html")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(manifest)} notebooks + manifest.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
