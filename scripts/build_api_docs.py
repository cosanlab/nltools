#!/usr/bin/env python3
"""Generate API documentation from Python source using griffe2md.

Runs griffe2md on each module listed in MODULES, postprocesses the Markdown
(see `postprocess_api_docs`) and writes one page per module under docs/api/.
The output filenames match the TOC entries in docs/myst.yml (a test keeps the
two lists in sync).

griffe warnings (missing annotations, unresolved references, ...) are always
surfaced — deduplicated, grouped by source file, with a count — because
griffe2md exits 0 even when it emits them.

Usage:
    python scripts/build_api_docs.py                     # generate all
    python scripts/build_api_docs.py --clean             # rm docs/api/**/*.md first
    python scripts/build_api_docs.py --check             # drift gate: regenerate to a
                                                         # temp dir, diff vs docs/api
    python scripts/build_api_docs.py --fail-on-warnings  # exit 1 on any griffe warning
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Make the sibling postprocess module importable whether run as a script
# (sys.path[0] already covers it) or imported some other way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_api_docs import (  # noqa: E402
    page_prefix,
    page_title,
    postprocess,
    with_frontmatter,
    xref_entries,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_API = PROJECT_ROOT / "docs" / "api"


def _griffe2md_argv() -> list[str]:
    """Build the argv prefix for invoking griffe2md, robust to PATH and shebangs.

    poe's shell tasks don't reliably propagate the uv venv's bin dir onto PATH,
    and a relocated venv (e.g. synced across machines) can leave the console
    script with a stale shebang so exec-ing it directly fails. Running the console
    script *through* the current interpreter sidesteps both: the shebang line is
    ignored and no PATH lookup is needed. Fall back to a bare name if not found.
    """
    candidate = Path(sys.executable).parent / "griffe2md"
    if candidate.exists():
        return [sys.executable, str(candidate)]
    resolved = shutil.which("griffe2md")
    return [resolved] if resolved else ["griffe2md"]


GRIFFE2MD = _griffe2md_argv()

# (import_path, output_path relative to docs/api/)
# Matches the myst.yml TOC structure (tests/support/test_build_api_docs.py).
MODULES: list[tuple[str, str]] = [
    # --- top-level API ---
    ("nltools.plotting", "plotting.md"),
    ("nltools.mask", "mask.md"),
    ("nltools.io", "io.md"),
    ("nltools.datasets", "dataset.md"),
    ("nltools.cross_validation", "crossval.md"),
    ("nltools.data.roc", "analysis.md"),
    ("nltools.utils", "utils.md"),
    ("nltools.templates", "templates.md"),
    ("nltools.data.simulator", "simulator.md"),
    ("nltools.data.braindata.neighborhoods", "neighborhoods.md"),
    ("nltools.data.braindata.cache", "cache.md"),
    ("nltools.models", "models.md"),
    ("nltools.algorithms.backends", "backends.md"),
    # --- data classes ---
    ("nltools.data.braindata.BrainData", "data/brain_data.md"),
    ("nltools.data.braindata.io", "data/braindata_io.md"),
    ("nltools.data.braindata.analysis", "data/braindata_analysis.md"),
    ("nltools.data.braindata.modeling", "data/braindata_modeling.md"),
    ("nltools.data.braindata.prediction", "data/braindata_prediction.md"),
    ("nltools.data.braindata.bootstrap", "data/braindata_bootstrap.md"),
    ("nltools.data.braindata.plotting", "data/braindata_plotting.md"),
    ("nltools.data.adjacency.Adjacency", "data/adjacency.md"),
    ("nltools.data.adjacency.stats", "data/adjacency_stats.md"),
    ("nltools.data.adjacency.modeling", "data/adjacency_modeling.md"),
    ("nltools.data.adjacency.plotting", "data/adjacency_plotting.md"),
    ("nltools.data.adjacency.io", "data/adjacency_io.md"),
    ("nltools.data.adjacency.spatial", "data/adjacency_spatial.md"),
    ("nltools.data.designmatrix.DesignMatrix", "data/design_matrix.md"),
    ("nltools.data.designmatrix.transforms", "data/design_matrix_transforms.md"),
    ("nltools.data.designmatrix.regressors", "data/design_matrix_regressors.md"),
    ("nltools.data.designmatrix.append", "data/design_matrix_append.md"),
    ("nltools.data.designmatrix.diagnostics", "data/design_matrix_diagnostics.md"),
    ("nltools.data.designmatrix.plotting", "data/design_matrix_plotting.md"),
    ("nltools.data.designmatrix.io", "data/design_matrix_io.md"),
    ("nltools.data.collection.BrainCollection", "data/brain_collection.md"),
    ("nltools.data.collection.core", "data/collection_core.md"),
    ("nltools.data.collection.execution", "data/collection_execution.md"),
    ("nltools.data.collection.inference", "data/collection_inference.md"),
    ("nltools.data.collection.io", "data/collection_io.md"),
    ("nltools.data.fitresults", "data/fitresults.md"),
    # --- atlases ---
    ("nltools.data.atlases", "data/atlases.md"),
    ("nltools.data.atlases.registry", "data/atlases_registry.md"),
    ("nltools.data.atlases.loading", "data/atlases_loading.md"),
    ("nltools.data.atlases.labeling", "data/atlases_labeling.md"),
    ("nltools.data.atlases.reporting", "data/atlases_reporting.md"),
    # --- algorithms ---
    ("nltools.algorithms", "algorithms.md"),
    ("nltools.algorithms.corrections", "algorithms/corrections.md"),
    ("nltools.algorithms.outliers", "algorithms/outliers.md"),
    ("nltools.algorithms.signal", "algorithms/signal.md"),
    ("nltools.algorithms.similarity", "algorithms/similarity.md"),
    ("nltools.algorithms.regression", "algorithms/regression.md"),
    ("nltools.algorithms.alignment", "algorithms/alignment.md"),
    ("nltools.algorithms.alignment.procrustes", "algorithms/alignment_procrustes.md"),
    ("nltools.algorithms.hrf", "algorithms/hrf.md"),
    ("nltools.algorithms.ridge", "algorithms/ridge.md"),
    ("nltools.algorithms.inference", "algorithms/inference.md"),
    ("nltools.algorithms.inference.one_sample", "algorithms/inference_one_sample.md"),
    ("nltools.algorithms.inference.two_sample", "algorithms/inference_two_sample.md"),
    ("nltools.algorithms.inference.correlation", "algorithms/inference_correlation.md"),
    ("nltools.algorithms.inference.timeseries", "algorithms/inference_timeseries.md"),
    ("nltools.algorithms.inference.matrix", "algorithms/inference_matrix.md"),
    ("nltools.algorithms.inference.isc", "algorithms/inference_isc.md"),
    (
        "nltools.algorithms.inference.intersubject",
        "algorithms/inference_intersubject.md",
    ),
    ("nltools.algorithms.inference.bootstrap", "algorithms/inference_bootstrap.md"),
]


# ---------------------------------------------------------------------------
# griffe warnings
# ---------------------------------------------------------------------------

# ``path/to/file.py:123: message`` — griffe's warning format.
_WARNING_RE = re.compile(r"^(?P<file>[^\s:]+):(?P<line>\d+): (?P<msg>.+)$")


def parse_warnings(stderr: str) -> list[str]:
    """Return griffe2md's stderr as a list of warning lines (blank lines dropped).

    Nothing is filtered: "No type or annotation" lines are real docstring bugs.
    """
    return [ln.rstrip() for ln in stderr.splitlines() if ln.strip()]


def warning_report(warnings: set[str] | list[str]) -> str:
    """Format warnings deduplicated, grouped by source file, with a count summary.

    The same object rendered on several pages (a facade class and its own page)
    repeats its warnings verbatim; each appears once here. Lines that don't match
    griffe's ``file:line: message`` format are listed under ``(other)``.
    """
    unique = sorted(set(warnings))
    if not unique:
        return "0 griffe warnings"
    by_file: dict[str, list[str]] = defaultdict(list)
    for w in unique:
        m = _WARNING_RE.match(w)
        if m:
            by_file[m["file"]].append(f"{m['line']}: {m['msg']}")
        else:
            by_file["(other)"].append(w)
    out: list[str] = []
    for file in sorted(by_file):
        out.append(file)
        out.extend(f"  {entry}" for entry in by_file[file])
    n_files = len(by_file)
    out.append(
        f"{len(unique)} griffe warning{'s' if len(unique) != 1 else ''} "
        f"in {n_files} file{'s' if n_files != 1 else ''}"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


@dataclass
class BuildReport:
    """What one build produced: page counts and every griffe warning seen."""

    ok: int = 0
    failed: list[str] = field(default_factory=list)
    warnings: set[str] = field(default_factory=set)


def render_module(module: str) -> tuple[str | None, list[str]]:
    """Run griffe2md for one module; return (raw markdown or None on failure, warnings).

    Runs from the project root so griffe2md finds ``[tool.griffe2md]`` in
    pyproject.toml and warning paths are repo-relative.
    """
    result = subprocess.run(
        [*GRIFFE2MD, module],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    warnings = parse_warnings(result.stderr)
    if result.returncode != 0:
        return None, warnings
    return result.stdout, warnings


def build(out_dir: Path, *, clean: bool = False, verbose: bool = True) -> BuildReport:
    """Generate every page in MODULES into ``out_dir`` (normally docs/api).

    Two passes over the raw griffe2md output: the first postprocesses each page
    without cross-page links and indexes the labels it produced (`xref_entries`);
    the second postprocesses again with that index so type annotations and
    ``Bases:`` entries link to the page documenting them.
    """
    report = BuildReport()
    if clean:
        for f in out_dir.rglob("*.md"):
            f.unlink()
        if verbose:
            print(f"Cleaned {out_dir}")

    raw: dict[str, tuple[str, str]] = {}  # out_path -> (module, raw markdown)
    for module, out_path in MODULES:
        if verbose:
            print(f"  {module} → {out_path}")
        text, warnings = render_module(module)
        report.warnings.update(warnings)
        if text is None:
            print(f"  FAILED: {module} → {out_path}", file=sys.stderr)
            report.failed.append(module)
            continue
        raw[out_path] = (module, text)

    xref: dict[str, str] = {}
    prefixes: dict[str, str] = {}
    for out_path, (module, text) in raw.items():
        prefix = page_prefix(out_dir / out_path, out_dir)
        prefixes[out_path] = prefix
        xref.update(xref_entries(module, prefix, postprocess(text, prefix)))

    for out_path, (module, text) in raw.items():
        output = out_dir / out_path
        output.parent.mkdir(parents=True, exist_ok=True)
        prefix = prefixes[out_path]
        page = postprocess(text, prefix, xref)
        output.write_text(with_frontmatter(page, page_title(module), prefix))
        report.ok += 1
    return report


# ---------------------------------------------------------------------------
# drift check
# ---------------------------------------------------------------------------


def diff_trees(generated: Path, committed: Path) -> list[str]:
    """Compare two docs/api trees; one message per changed, stale, or new page."""
    gen = {p.relative_to(generated).as_posix(): p for p in generated.rglob("*.md")}
    com = {p.relative_to(committed).as_posix(): p for p in committed.rglob("*.md")}
    drift: list[str] = []
    for rel in sorted(gen.keys() | com.keys()):
        if rel not in com:
            drift.append(f"new (not committed): {rel}")
        elif rel not in gen:
            drift.append(f"missing (stale, delete): {rel}")
        elif gen[rel].read_text() != com[rel].read_text():
            drift.append(f"changed: {rel}")
    return drift


def check(docs_api: Path = DOCS_API) -> int:
    """Regenerate into a temp dir and diff against ``docs_api``; 0 when in sync."""
    with tempfile.TemporaryDirectory(prefix="nltools-api-docs-") as tmp:
        report = build(Path(tmp), verbose=False)
        if report.failed:
            print(f"build failed for: {', '.join(report.failed)}", file=sys.stderr)
            return 1
        drift = diff_trees(Path(tmp), docs_api)
    if drift:
        print("docs/api is out of date — run `uv run poe docs-generate` and commit:")
        for line in drift:
            print(f"  {line}")
        return 1
    print(f"docs/api in sync ({report.ok} pages)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing generated docs before generating",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into a temp dir and fail if docs/api differs (no writes)",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit non-zero if griffe emitted any warning",
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(check())

    report = build(DOCS_API, clean=args.clean)
    print()
    print(warning_report(report.warnings), file=sys.stderr)
    print(f"\nGenerated {report.ok} API doc pages ({len(report.failed)} failures)")
    if report.failed or (args.fail_on_warnings and report.warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
