"""Shared plumbing for the lint-api scripts.

One home for what `check_api_vocabulary.py`, `check_kwonly.py`, and
`build_api_vocabulary.py` all need: the project-root / manifest-path
constants, the YAML loader for `docs/_data/api-vocabulary.yml`, and the
source-tree file iterator. Each script previously carried its own copy.

Not a package member — the scripts import it by sibling path
(``sys.path`` already contains ``scripts/`` when a script is executed
directly; the test suite inserts it explicitly).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_YML = PROJECT_ROOT / "docs" / "_data" / "api-vocabulary.yml"


def load_vocab(vocab_yml: Path = VOCAB_YML) -> dict:
    """Load the API-vocabulary manifest (the single source of truth)."""
    with vocab_yml.open() as f:
        return yaml.safe_load(f)


def iter_py_files(
    roots: Iterable[str | Path],
    *,
    root_dir: Path = PROJECT_ROOT,
    exclude_parts: frozenset[str] = frozenset(),
) -> list[Path]:
    """Collect every ``.py`` file under each root, sorted.

    Args:
        roots: Files or directories; relative entries resolve against
            ``root_dir``, and a ``.py`` file root yields itself.
        root_dir: Base for relative roots (default: the project root, so the
            scripts behave the same from any working directory).
        exclude_parts: Path components that exclude a file when any of them
            appears anywhere in its path (e.g. ``{"tests"}``).

    Returns:
        Sorted list of absolute Paths.
    """
    files: list[Path] = []
    for root in roots:
        p = Path(root)
        if not p.is_absolute():
            p = root_dir / p
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(
                f for f in p.rglob("*.py") if not (exclude_parts & set(f.parts))
            )
    return sorted(files)


def rel_posix(path: Path, *, root_dir: Path = PROJECT_ROOT) -> str:
    """Project-relative POSIX form of ``path`` (as-is if outside ``root_dir``)."""
    try:
        return path.relative_to(root_dir).as_posix()
    except ValueError:
        return path.as_posix()
