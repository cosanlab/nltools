#!/usr/bin/env python
"""Convert marimo tutorial notebooks to marimo-free Jupyter notebooks.

The marimo ``.py`` notebooks under ``docs/tutorials/workflows`` are the single
source of truth.  This script asks marimo to export them in topological order,
which turns ``mo.md()`` cells into Jupyter markdown cells and unwraps analysis
cells into plain Python.  It then removes the remaining marimo import cell so
the result runs in a plain Jupyter/Pyodide kernel without marimo installed.

Usage::

    python scripts/tutorials_to_ipynb.py docs/tutorials/workflows/01_glm.py
    python scripts/tutorials_to_ipynb.py docs/tutorials/workflows/*.py
    python scripts/tutorials_to_ipynb.py --all
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs/jupyterlite/files"

JUPYTERLITE_PACKAGES = [
    "numpy==2.0.2",
    "nibabel",
    "nilearn==0.13.1",
    "scikit-learn",
    "scipy",
    "pandas",
    "polars",
    "seaborn",
    "matplotlib",
    "anywidget",
    "ipyniivue",
    "ipywidgets",  # optional extra for installs; required here for iplot()'s slider
    "joblib>=1.5.3",
    "huggingface-hub",
    "pynv",
]

# Notebooks converted by ``--all``.
TUTORIAL_GLOBS = [
    "docs/tutorials/workflows/[0-9]*.py",
]

# Same cleanup used by scripts/marimo_to_myst.py after marimo export.
MARIMO_IMPORT_RE = re.compile(r"^\s*import\s+marimo\b.*$", re.MULTILINE)
FORBIDDEN_CODE_RE = re.compile(r"(?:\bmarimo\b|\bmo\.|@app\.cell)")


def jupyterlite_setup_cells() -> list[nbformat.NotebookNode]:
    """Return the browser-kernel installation instructions and setup cell."""
    packages = "\n".join(f"    {package!r}," for package in JUPYTERLITE_PACKAGES)
    source = (
        "import piplite\n\n"
        "await piplite.install([\n"
        f"{packages}\n"
        "])\n"
        'await piplite.install("nltools", deps=False)'
    )
    return [
        nbformat.v4.new_markdown_cell(
            "Run this cell first — installs nltools and dependencies into the "
            "browser kernel (~1 min)."
        ),
        nbformat.v4.new_code_cell(source),
    ]


def rel_to_repo(path: Path) -> Path:
    """Return a path relative to the repository when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return Path(path.name)


def export_marimo_ipynb(notebook: Path, output: Path) -> None:
    """Export ``notebook`` as IPYNB in dependency-safe topological order."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "ipynb",
            "--sort",
            "topological",
            str(notebook),
            "-o",
            str(output),
        ],
        check=True,
    )


def strip_marimo_runtime(notebook: nbformat.NotebookNode) -> None:
    """Remove marimo imports and runtime-only cells from an exported notebook."""
    cells = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = MARIMO_IMPORT_RE.sub("", cell.source).strip("\n")
            if not cell.source.strip():
                continue
        cells.append(cell)
    notebook.cells = cells


def assign_deterministic_ids(notebook: nbformat.NotebookNode) -> None:
    """Give every cell a stable id so regenerating a notebook is a git no-op.

    ``marimo export`` and ``nbformat.v4.new_*_cell`` both mint random cell ids,
    so an unchanged tutorial otherwise produces a different ``.ipynb`` on every
    build. Index-based ids are deterministic and unique within the notebook.
    """
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{index:02d}"


def validate(notebook: nbformat.NotebookNode, source: Path) -> None:
    """Reject code cells that still depend on the marimo runtime."""
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        match = FORBIDDEN_CODE_RE.search(cell.source)
        if match:
            raise ValueError(
                f"{source}: code cell {index} still contains {match.group(0)!r}"
            )


def convert(notebook: Path) -> Path:
    """Convert one marimo notebook and return its generated IPYNB path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / notebook.with_suffix(".ipynb").name

    with tempfile.TemporaryDirectory() as temp_dir:
        exported = Path(temp_dir) / output.name
        export_marimo_ipynb(notebook, exported)
        jupyter_notebook = nbformat.read(exported, as_version=4)

    strip_marimo_runtime(jupyter_notebook)
    jupyter_notebook.cells = jupyterlite_setup_cells() + jupyter_notebook.cells
    validate(jupyter_notebook, notebook)
    assign_deterministic_ids(jupyter_notebook)
    nbformat.write(jupyter_notebook, output, version=4)
    return output


def resolve_targets(args: argparse.Namespace) -> list[Path]:
    if args.all:
        targets: list[Path] = []
        for pattern in TUTORIAL_GLOBS:
            targets.extend(sorted(REPO_ROOT.glob(pattern)))
        return targets
    return [Path(path).resolve() for path in args.notebooks]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("notebooks", nargs="*", help="marimo .py notebooks to convert")
    parser.add_argument(
        "--all", action="store_true", help="convert every notebook in TUTORIAL_GLOBS"
    )
    args = parser.parse_args()

    targets = resolve_targets(args)
    if not targets:
        parser.error("no notebooks given (pass paths or --all)")

    for notebook in targets:
        if not notebook.exists():
            print(f"  skip (missing): {notebook}", file=sys.stderr)
            continue
        output = convert(notebook)
        print(f"  {rel_to_repo(notebook)} -> {rel_to_repo(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
