#!/usr/bin/env python3
"""Generate API documentation from Python source using griffe2md.

Runs griffe2md on each module listed in MODULES and writes the output
to docs/api/.  The output filenames match the existing TOC entries in
myst.yml so no TOC changes are needed after generation.

Usage:
    python scripts/build_api_docs.py          # generate all
    python scripts/build_api_docs.py --clean   # rm docs/api/**/*.md first
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_API = PROJECT_ROOT / "docs" / "api"

# (import_path, output_path relative to docs/api/)
# Matches the existing myst.yml TOC structure.
MODULES: list[tuple[str, str]] = [
    # --- top-level API ---
    ("nltools.stats", "stats.md"),
    ("nltools.plotting", "plotting.md"),
    ("nltools.mask", "mask.md"),
    ("nltools.io.file_reader", "filereader.md"),
    ("nltools.datasets", "dataset.md"),
    ("nltools.cross_validation", "crossval.md"),
    ("nltools.data.roc", "analysis.md"),
    ("nltools.utils", "utils.md"),
    ("nltools.prefs", "prefs.md"),
    ("nltools.data.simulator", "simulator.md"),
    ("nltools.data.braindata.neighborhoods", "neighborhoods.md"),
    ("nltools.data.braindata.cache", "cache.md"),
    ("nltools.models", "models.md"),
    ("nltools.backends", "backends.md"),
    # --- data classes ---
    ("nltools.data", "data.md"),
    ("nltools.data.braindata", "data/brain_data.md"),
    ("nltools.data.braindata.io", "data/braindata_io.md"),
    ("nltools.data.braindata.analysis", "data/braindata_analysis.md"),
    ("nltools.data.braindata.modeling", "data/braindata_modeling.md"),
    ("nltools.data.braindata.prediction", "data/braindata_prediction.md"),
    ("nltools.data.braindata.bootstrap", "data/braindata_bootstrap.md"),
    ("nltools.data.braindata.plotting", "data/braindata_plotting.md"),
    ("nltools.data.braindata.pipeline", "data/braindata_pipeline.md"),
    ("nltools.data.adjacency", "data/adjacency.md"),
    ("nltools.data.adjacency.stats", "data/adjacency_stats.md"),
    ("nltools.data.adjacency.modeling", "data/adjacency_modeling.md"),
    ("nltools.data.adjacency.plotting", "data/adjacency_plotting.md"),
    ("nltools.data.adjacency.io", "data/adjacency_io.md"),
    ("nltools.data.designmatrix", "data/design_matrix.md"),
    ("nltools.data.designmatrix.transforms", "data/design_matrix_transforms.md"),
    ("nltools.data.designmatrix.regressors", "data/design_matrix_regressors.md"),
    ("nltools.data.designmatrix.append", "data/design_matrix_append.md"),
    ("nltools.data.designmatrix.diagnostics", "data/design_matrix_diagnostics.md"),
    ("nltools.data.designmatrix.io", "data/design_matrix_io.md"),
    ("nltools.data.collection", "data/brain_collection.md"),
    ("nltools.data.collection.constructors", "data/collection_constructors.md"),
    ("nltools.data.collection.transforms", "data/collection_transforms.md"),
    ("nltools.data.collection.inference", "data/collection_inference.md"),
    ("nltools.data.collection.modeling", "data/collection_modeling.md"),
    ("nltools.data.collection.prediction", "data/collection_prediction.md"),
    ("nltools.data.collection.io", "data/collection_io.md"),
    ("nltools.data.collection.pipeline", "data/collection_pipeline.md"),
    # --- algorithms ---
    ("nltools.algorithms", "algorithms.md"),
    ("nltools.algorithms.inference", "algorithms/inference.md"),
    ("nltools.algorithms.inference.one_sample", "algorithms/inference_one_sample.md"),
    ("nltools.algorithms.inference.two_sample", "algorithms/inference_two_sample.md"),
    ("nltools.algorithms.inference.correlation", "algorithms/inference_correlation.md"),
    ("nltools.algorithms.inference.timeseries", "algorithms/inference_timeseries.md"),
    ("nltools.algorithms.inference.matrix", "algorithms/inference_matrix.md"),
    ("nltools.algorithms.inference.isc", "algorithms/inference_isc.md"),
    ("nltools.algorithms.inference.icc", "algorithms/inference_icc.md"),
    ("nltools.algorithms.inference.bootstrap", "algorithms/inference_bootstrap.md"),
    ("nltools.algorithms.inference.utils", "algorithms/inference_utils.md"),
    # --- pipelines ---
    ("nltools.pipelines", "pipelines.md"),
    ("nltools.pipelines.base", "pipelines/pipeline.md"),
    ("nltools.pipelines.multi_subject", "pipelines/multi_subject.md"),
    ("nltools.pipelines.cv", "pipelines/cv.md"),
    ("nltools.pipelines.steps", "pipelines/steps.md"),
    ("nltools.pipelines.terminals", "pipelines/terminals.md"),
    ("nltools.pipelines.results", "pipelines/results.md"),
]


def generate(module: str, output: Path) -> bool:
    """Run griffe2md for a single module and write to output path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["griffe2md", module, "-o", str(output)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Warnings go to stderr but aren't fatal
        if result.stderr:
            # Filter to just error lines (not annotation warnings)
            errors = [
                ln
                for ln in result.stderr.splitlines()
                if "No type or annotation" not in ln
            ]
            if errors:
                print(f"  warnings: {module}", file=sys.stderr)
                for ln in errors:
                    print(f"    {ln}", file=sys.stderr)
        if not output.exists():
            print(f"  FAILED: {module} → {output}", file=sys.stderr)
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing generated docs before generating",
    )
    args = parser.parse_args()

    if args.clean:
        import shutil

        for subdir in ["data", "algorithms", "pipelines"]:
            d = DOCS_API / subdir
            if d.exists():
                shutil.rmtree(d)
        for f in DOCS_API.glob("*.md"):
            f.unlink()
        print("Cleaned docs/api/")

    # Ensure subdirectories exist
    for subdir in ["data", "algorithms", "pipelines"]:
        (DOCS_API / subdir).mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    for module, out_path in MODULES:
        output = DOCS_API / out_path
        print(f"  {module} → {out_path}")
        if generate(module, output):
            ok += 1
        else:
            fail += 1

    print(f"\nGenerated {ok} API doc pages ({fail} failures)")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
