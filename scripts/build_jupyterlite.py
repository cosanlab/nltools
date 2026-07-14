#!/usr/bin/env python3
"""Build the tutorial JupyterLite bundle for local or subpath hosting."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_LITE_DIR = REPO_ROOT / "docs" / "jupyterlite"
OUTPUT_DIR = REPO_ROOT / "docs" / "_build" / "html" / "try-it-jupyterlite"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a build command from the repository root."""
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def newest_wheel() -> Path:
    """Return the wheel most recently written by ``uv build``."""
    wheels = list((REPO_ROOT / "dist").glob("nltools-*.whl"))
    if not wheels:
        raise FileNotFoundError("uv build did not produce dist/nltools-*.whl")
    return max(wheels, key=lambda path: path.stat().st_mtime).resolve()


def validate_base_url(base_url: str) -> str:
    """Validate a local-relative or absolute JupyterLite runtime mount."""
    if base_url != "./" and (
        not base_url.startswith("/") or not base_url.endswith("/")
    ):
        raise ValueError("--base-url must be './' or start and end with '/'")
    return base_url


@contextmanager
def lite_project(base_url: str) -> Iterator[tuple[Path, list[str]]]:
    """Yield a temporary project with an explicit runtime base URL overlay."""
    with tempfile.TemporaryDirectory(prefix="nltools-jupyterlite-") as temp_dir:
        lite_dir = Path(temp_dir)
        shutil.copy2(
            LOCAL_LITE_DIR / "jupyter_lite_config.json",
            lite_dir / "jupyter_lite_config.json",
        )
        runtime_config = {
            "jupyter-lite-schema-version": 0,
            "jupyter-config-data": {"baseUrl": validate_base_url(base_url)},
        }
        (lite_dir / "jupyter-lite.json").write_text(
            json.dumps(runtime_config, indent=2) + "\n"
        )
        yield (
            lite_dir,
            [
                "--contents",
                str((LOCAL_LITE_DIR / "files").resolve()),
                f"--LiteBuildConfig.cache_dir={(LOCAL_LITE_DIR / '.cache').resolve()}",
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="./",
        help=(
            "runtime mount for the bundle; defaults to './' for the root-based "
            "local build"
        ),
    )
    args = parser.parse_args()

    build_env = os.environ.copy()
    build_env.update(UV_CACHE_DIR="/tmp/uv-cache", UV_TOOL_DIR="/tmp/uv-tools")

    run(["uv", "build", "--wheel"], env=build_env)
    run([sys.executable, "scripts/tutorials_to_ipynb.py", "--all"])

    with lite_project(args.base_url) as (lite_dir, extra_args):
        run(
            [
                "uvx",
                "--with",
                "jupyterlite-pyodide-kernel==0.6.1",
                "--with",
                "jupyter-server",
                "--with",
                "anywidget",
                "--with",
                "ipywidgets",
                "--from",
                "jupyterlite-core==0.6.4",
                "jupyter",
                "lite",
                "build",
                "--lite-dir",
                str(lite_dir),
                "--output-dir",
                str(OUTPUT_DIR),
                "--piplite-wheels",
                str(newest_wheel()),
                *extra_args,
            ],
            env=build_env,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
