#!/usr/bin/env python
"""Interactive nltools release orchestrator.

The workflow reviews the latest pytest log, bumps the project version, builds
and smoke-tests distributions, regenerates the changelog, creates a release
commit and tag, and finally publishes the artifacts.

Usage:
    uv run poe release
    uv run poe release --target testpypi

This script intentionally does not build/deploy docs or run the optional
Pyodide suite. Run those separately when a release requires them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
CHANGELOG = PROJECT_ROOT / "docs" / "changelog.md"
DIST_DIR = PROJECT_ROOT / "dist"
TEST_LOG = PROJECT_ROOT / "pytest.log"
UV_LOCK = PROJECT_ROOT / "uv.lock"

VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
RELEASE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?$")

PYPI_URL = "https://pypi.org/pypi/nltools/json"
TESTPYPI_URL = "https://test.pypi.org/pypi/nltools/json"
TESTPYPI_PUBLISH_URL = "https://test.pypi.org/legacy/"
TESTPYPI_CHECK_URL = "https://test.pypi.org/simple/"

SMOKE_TEST_CODE = textwrap.dedent(
    """\
    import sys
    from importlib.metadata import version

    import numpy as np
    from nltools import Adjacency, DesignMatrix

    expected = sys.argv[1]
    installed = version("nltools")
    assert installed == expected, (installed, expected)

    dm = DesignMatrix(
        {"intercept": [1.0, 1.0], "condition": [0.0, 1.0]},
        sampling_freq=1.0,
    )
    assert dm.shape == (2, 2), dm.shape

    adjacency = Adjacency(
        np.array([[0.0, 0.5], [0.5, 0.0]]),
        matrix_type="similarity",
    )
    assert adjacency.shape == (2, 2), adjacency.shape

    print(f"nltools {installed} wheel smoke test passed")
    """
)

console = Console()


def abort(message: str) -> None:
    """Print an error and exit."""
    console.print(f"\n[bold red]Aborted:[/] {message}")
    raise SystemExit(1)


def confirm(prompt: str) -> bool:
    """Ask a yes/no question, defaulting to no."""
    try:
        answer = console.input(f"\n{prompt} [bold]\\[y/N][/] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def run(
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, streaming output unless capture is requested."""
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        command,
        cwd=cwd,
        env=merged_env,
        capture_output=capture,
        text=True,
        check=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("pypi", "testpypi"),
        default="pypi",
        help="Package index to publish to (default: pypi)",
    )
    return parser.parse_args()


def parse_version(text: str) -> str:
    """Extract the project version from pyproject.toml text."""
    match = VERSION_RE.search(text)
    if not match:
        abort("Cannot parse version from pyproject.toml")
    return match.group(1)


def propose_next_version(version: str) -> str:
    """Propose the next minor release, e.g. 0.5.1 -> 0.6.0."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.\d+(?:.*)?", version)
    if not match:
        abort(f"Cannot propose a release after version {version!r}")
    major, minor = (int(part) for part in match.groups())
    return f"{major}.{minor + 1}.0"


def fetch_published_version(index_url: str) -> str | None:
    """Return the latest published version, or None if the index is unavailable."""
    try:
        with urlopen(index_url, timeout=10) as response:
            data = json.loads(response.read())
    except (URLError, json.JSONDecodeError, OSError):
        return None
    return data.get("info", {}).get("version")


def file_age_hours(path: Path) -> float | None:
    """Return hours since a file was modified, or None when it is absent."""
    if not path.exists():
        return None
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return (datetime.now(tz=UTC) - modified).total_seconds() / 3600


def parse_pytest_summary(path: Path) -> tuple[int | None, int | None, str | None]:
    """Extract pass/fail counts and the final pytest summary line."""
    if not path.exists():
        return None, None, None

    text = path.read_text(errors="replace")
    passed_matches = re.findall(r"(\d+) passed", text)
    failed_matches = re.findall(r"(\d+) failed", text)
    passed = int(passed_matches[-1]) if passed_matches else None
    failed = int(failed_matches[-1]) if failed_matches else 0 if passed is not None else None

    summary = None
    for line in reversed(text.splitlines()):
        stripped = line.strip(" =")
        if re.search(r"\b(?:passed|failed)\b", stripped):
            summary = stripped[:140]
            break
    return passed, failed, summary


def ensure_clean_worktree() -> None:
    """Require release inputs to start from a clean Git worktree."""
    result = run(["git", "status", "--porcelain"], capture=True)
    if result.returncode != 0:
        abort("Cannot inspect git worktree")
    if result.stdout.strip():
        abort("Git worktree is not clean. Commit or stash changes before releasing.")


def step_review_test_state() -> str:
    """Review the existing pytest log and optionally run the current test task."""
    console.print(Panel("[bold]Step 1: Review Test State[/]", style="blue"))

    age = file_age_hours(TEST_LOG)
    passed, failed, summary = parse_pytest_summary(TEST_LOG)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Source", style="cyan")
    table.add_column("Age", justify="right")
    table.add_column("Result")
    table.add_column("Status", justify="center")

    if age is None:
        status = "missing"
        table.add_row("pytest.log", "-", "no log file", "[yellow]MISSING[/]")
    elif failed:
        status = "failed"
        table.add_row(
            "pytest.log",
            f"{age:.1f}h",
            f"{passed or 0} passed, {failed} failed",
            "[red]FAIL[/]",
        )
    elif passed is None:
        status = "unknown"
        table.add_row("pytest.log", f"{age:.1f}h", summary or "unparseable", "[yellow]?[/]")
    elif age >= 24:
        status = "stale"
        table.add_row("pytest.log", f"{age:.1f}h", f"{passed} passed", "[yellow]STALE[/]")
    else:
        status = "current"
        table.add_row("pytest.log", f"{age:.1f}h", f"{passed} passed", "[green]OK[/]")

    console.print(table)
    if status == "current":
        return status

    console.print(
        "[yellow]The saved log is advisory only; nltools poe tasks do not maintain logs.[/]"
    )
    if confirm("Run the current default suite now (`uv run poe test`)?"):
        result = run(["uv", "run", "poe", "test"])
        if result.returncode == 0:
            console.print("[bold green]Current default test suite passed.[/]")
            return "rerun-passed"
        console.print(f"[bold red]Current test suite failed (exit {result.returncode}).[/]")
        if not confirm("Continue despite the test failure?"):
            abort("Fix test failures before releasing.")
        return "rerun-failed"

    if not confirm("Continue without a current passing test run?"):
        abort("Run the test suite before releasing.")
    return status


def step_version_review(target: str) -> tuple[str, str]:
    """Show source/index versions and select the release version."""
    console.print(Panel("[bold]Step 2: Version Review[/]", style="blue"))

    old_version = parse_version(PYPROJECT.read_text())
    proposed = propose_next_version(old_version)
    index_url = TESTPYPI_URL if target == "testpypi" else PYPI_URL
    published = fetch_published_version(index_url) or "unavailable / not published"

    table = Table(show_header=True, header_style="bold")
    table.add_column("", style="cyan")
    table.add_column("Version")
    table.add_row(f"Published ({'TestPyPI' if target == 'testpypi' else 'PyPI'})", published)
    table.add_row("Source (current)", old_version)
    table.add_row("Next (proposed)", f"[bold green]{proposed}[/]")
    console.print(table)

    custom = console.input(
        f"\nAccept [bold]{proposed}[/] or enter a version (Enter to accept): "
    ).strip()
    new_version = custom or proposed
    if not RELEASE_VERSION_RE.fullmatch(new_version):
        abort(f"Invalid release version {new_version!r}")
    if new_version == old_version:
        abort("New version matches the current source version")

    tag = f"v{new_version}"
    existing = run(["git", "tag", "--list", tag], capture=True)
    if existing.returncode != 0:
        abort("Cannot inspect existing git tags")
    if existing.stdout.strip():
        abort(f"Git tag {tag} already exists")
    return old_version, new_version


def step_bump_version(old_version: str, new_version: str) -> None:
    """Bump the sole release version field in pyproject.toml and refresh uv.lock."""
    console.print(Panel("[bold]Step 3: Bump Version[/]", style="blue"))

    text = PYPROJECT.read_text()
    old_line = f'version = "{old_version}"'
    new_text = text.replace(old_line, f'version = "{new_version}"', 1)
    if new_text == text:
        abort(f"Failed to replace {old_line!r} in pyproject.toml")
    PYPROJECT.write_text(new_text)
    console.print(f"  pyproject.toml: [red]{old_version}[/] -> [green]{new_version}[/]")

    result = run(["uv", "lock"])
    if result.returncode != 0:
        abort("uv lock failed after the version bump")


def step_build() -> list[Path]:
    """Build exactly one wheel and one source distribution."""
    console.print(Panel("[bold]Step 4: Build Package[/]", style="blue"))

    result = run(["uv", "build", "--clear"])
    if result.returncode != 0:
        abort("uv build failed")

    wheels = sorted(DIST_DIR.glob("*.whl"))
    sdists = sorted(DIST_DIR.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        abort(
            f"Expected 1 wheel + 1 sdist, found {len(wheels)} wheel(s) "
            f"and {len(sdists)} sdist(s)"
        )

    artifacts = [wheels[0], sdists[0]]
    table = Table(show_header=True, header_style="bold")
    table.add_column("Artifact", style="cyan")
    table.add_column("Size", justify="right")
    for artifact in artifacts:
        table.add_row(artifact.name, f"{artifact.stat().st_size / 1024:.0f} KB")
    console.print(table)
    return artifacts


def step_smoke_test(expected_version: str) -> None:
    """Install the wheel into a fresh uv venv and exercise public data classes."""
    console.print(Panel("[bold]Step 5: Fresh Environment Smoke Test[/]", style="blue"))

    wheel = next(DIST_DIR.glob("*.whl"), None)
    if wheel is None:
        abort("No wheel found in dist/")

    temp_dir = Path(tempfile.mkdtemp(prefix="nltools-release-"))
    venv_dir = temp_dir / ".venv"
    smoke_script = temp_dir / "smoke_test.py"
    try:
        result = run(["uv", "venv", str(venv_dir), "--python", "3.11"])
        if result.returncode != 0:
            abort("Failed to create the smoke-test venv")

        python = venv_dir / "bin" / "python"
        result = run(["uv", "pip", "install", str(wheel), "--python", str(python)])
        if result.returncode != 0:
            abort("Failed to install the wheel in the smoke-test venv")

        smoke_script.write_text(SMOKE_TEST_CODE)
        result = run([str(python), str(smoke_script), expected_version])
        if result.returncode != 0:
            abort("Fresh-wheel smoke test failed")
        console.print("[bold green]Fresh-wheel smoke test passed.[/]")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def step_changelog(new_version: str) -> None:
    """Regenerate the changelog and finalize its synthetic release heading."""
    console.print(Panel("[bold]Step 6: Regenerate Changelog[/]", style="blue"))

    result = run(["uv", "run", "poe", "changelog"])
    if result.returncode != 0:
        abort("Changelog generation failed")

    text = CHANGELOG.read_text()
    heading = f"## {new_version} ({datetime.now(tz=UTC):%Y-%m-%d})"
    if "## Unreleased" in text:
        CHANGELOG.write_text(text.replace("## Unreleased", heading, 1))
    elif heading not in text:
        abort("Generated changelog has no Unreleased heading to finalize")
    console.print(f"  [green]Changelog finalized as {heading}[/]")


def step_confirm(
    old_version: str,
    new_version: str,
    target: str,
    artifacts: list[Path],
    test_status: str,
) -> None:
    """Show the final release plan and require explicit approval."""
    console.print(Panel("[bold]Step 7: Release Summary[/]", style="blue"))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Version", f"{old_version} -> [bold green]{new_version}[/]")
    table.add_row("Target", "TestPyPI" if target == "testpypi" else "PyPI")
    table.add_row("Test state", test_status)
    for artifact in artifacts:
        table.add_row("Artifact", artifact.name)
    table.add_row("Git", f"commit release: v{new_version}; tag v{new_version}")
    console.print(table)

    if not confirm("Create the release commit/tag and publish these artifacts?"):
        console.print("[yellow]Release cancelled; generated version/changelog edits remain.[/]")
        raise SystemExit(0)


def step_git_commit_and_tag(new_version: str) -> None:
    """Commit release metadata/changelog and tag that exact commit."""
    console.print(Panel("[bold]Step 8: Git Commit & Tag[/]", style="blue"))

    paths = ["pyproject.toml", "docs/changelog.md"]
    if UV_LOCK.exists():
        paths.append("uv.lock")
    result = run(["git", "add", *paths])
    if result.returncode != 0:
        abort("git add failed")

    tag = f"v{new_version}"
    result = run(["git", "commit", "-m", f"release: {tag}"])
    if result.returncode != 0:
        abort("Release commit failed")
    result = run(["git", "tag", tag])
    if result.returncode != 0:
        abort(f"git tag {tag} failed")
    console.print(f"  [green]Committed and tagged {tag}[/]")


def load_publish_env(target: str) -> dict[str, str]:
    """Load optional .env.pypi/.env.testpypi credentials for uv publish."""
    env_file = PROJECT_ROOT / f".env.{target}"
    if not env_file.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    console.print(f"  [dim]Loaded publish environment from {env_file.name}[/]")
    return values


def step_publish(target: str) -> None:
    """Publish distributions to PyPI or TestPyPI with uv."""
    console.print(Panel("[bold]Step 9: Publish[/]", style="blue"))

    command = ["uv", "publish"]
    if target == "testpypi":
        command.extend(
            [
                "--publish-url",
                TESTPYPI_PUBLISH_URL,
                "--check-url",
                TESTPYPI_CHECK_URL,
            ]
        )
    result = run(command, env=load_publish_env(target))
    if result.returncode != 0:
        abort(
            "uv publish failed. The release commit and tag remain local; "
            "inspect the error before retrying."
        )
    console.print("[bold green]Published successfully.[/]")


def main() -> None:
    args = parse_args()
    target_label = "TestPyPI" if args.target == "testpypi" else "PyPI"
    console.print(
        Panel(f"[bold]nltools release -> {target_label}[/]", style="bold magenta")
    )

    ensure_clean_worktree()
    test_status = step_review_test_state()
    old_version, new_version = step_version_review(args.target)
    step_bump_version(old_version, new_version)
    artifacts = step_build()
    step_smoke_test(new_version)
    step_changelog(new_version)
    step_confirm(old_version, new_version, args.target, artifacts, test_status)
    step_git_commit_and_tag(new_version)
    step_publish(args.target)

    console.print(
        Panel(
            f"[bold green]Release v{new_version} complete![/]\n"
            f"Published to {target_label}, committed, and tagged.",
            style="green",
        )
    )


if __name__ == "__main__":
    main()
