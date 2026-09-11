#!/usr/bin/env python3
"""Check the hand-written API reference pages under docs/api against the package.

The pages are mkdocstrings stubs: frontmatter, prose, and `::: dotted.path`
directives whose `members:` lists decide what each page documents. Nothing
regenerates them, so this script is what keeps them honest:

- every directive identifier, and every name in a `members:` list, resolves to a
  real object in the installed package;
- every public export of the namespaces in `PUBLIC_NAMESPACES` is documented on
  exactly one user-facing page (the pages outside the "Internal modules" group
  of the zensical nav);
- the A-Z index lists exactly `nltools.algorithms.__all__`, and each row points
  at the page that documents the object.

It reads the pages, not the built site, so it proves a directive *names* an
object, not that mkdocstrings renders one: an export that loses its docstring
still passes here while disappearing from the site under
`show_if_no_docstring: false`. The strict `uv run poe docs-build` is what catches
that, through the links into the API that then fail to resolve.

Usage:
    python scripts/check_api_pages.py
"""

from __future__ import annotations

import importlib
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_API = PROJECT_ROOT / "docs" / "api"
ZENSICAL_TOML = PROJECT_ROOT / "zensical.toml"

# Namespaces whose every ``__all__`` name must have exactly one home among the
# user-facing pages. Helper namespaces (``nltools.utils``, ``nltools.templates``,
# ``nltools.plotting``) are deliberately absent: they are re-cut across several
# task pages and also documented whole on an internal-module page.
PUBLIC_NAMESPACES = (
    "nltools.data",
    "nltools.algorithms",
    "nltools.models",
    "nltools.cross_validation",
    "nltools.datasets",
    "nltools.mask",
)

# The A-Z index page, and the nav group whose pages are not user-facing.
INDEX_PAGE = "algorithms.md"
INTERNAL_NAV_GROUP = "Internal modules"

# ``::: dotted.path`` on its own line, as permissively as mkdocstrings reads it
# (leading indentation allowed, the space after the colons optional), so no page
# can render a directive this checker does not see. The options block that may
# follow is indented four spaces past the directive.
DIRECTIVE_RE = re.compile(r"^(?P<indent> *)::: ?(?P<identifier>\S+) *$")
OPTIONS_INDENT = 4

# One row of the A-Z index: ``[`name`][identifier] | [Label](page.md)``.
INDEX_ROW_RE = re.compile(
    r"^\[`(?P<name>[^`]+)`\]\[(?P<identifier>[^\]]+)\] \| \[[^\]]+\]\((?P<page>[^)]+)\)$"
)


@dataclass(frozen=True)
class Directive:
    """One `::: identifier` block on a page, with its `members:` option if any.

    ``members`` is the explicit list, ``False`` where the page renders the
    object's docstring and no members at all, and ``None`` where the option is
    absent (a module page documents everything the module exports).
    """

    identifier: str
    members: list[str] | bool | None


def parse_directives(text: str) -> list[Directive]:
    """Return every `::: identifier` block on a page, in order.

    A directive's options run until the first line indented less than the block,
    blank lines included: a blank line followed by unindented prose ends the
    block rather than reaching over it for the next indented thing on the page.
    """
    lines = text.splitlines()
    directives: list[Directive] = []
    index = 0
    while index < len(lines):
        match = DIRECTIVE_RE.match(lines[index])
        index += 1
        if not match:
            continue
        prefix = " " * (len(match["indent"]) + OPTIONS_INDENT)
        block: list[str] = []
        pending_blanks: list[str] = []
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                pending_blanks.append("")
            elif line.startswith(prefix):
                block.extend(pending_blanks)
                pending_blanks.clear()
                block.append(line[len(prefix) :])
            else:
                break
            index += 1
        options = (yaml.safe_load("\n".join(block)) or {}).get("options", {})
        directives.append(Directive(match["identifier"], options.get("members")))
    return directives


def resolve(dotted: str) -> Any:
    """The object a dotted path names, importing modules as needed."""
    try:
        return importlib.import_module(dotted)
    except ImportError:
        pass
    parent_path, _, name = dotted.rpartition(".")
    if not parent_path:
        raise LookupError(dotted)
    try:
        return getattr(resolve(parent_path), name)
    except AttributeError as error:
        raise LookupError(dotted) from error


def module_exports(module: ModuleType) -> list[str]:
    """The public names a module page documents: its `__all__`, else its attributes."""
    if hasattr(module, "__all__"):
        return list(module.__all__)
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, ModuleType)
        and getattr(value, "__module__", module.__name__) == module.__name__
    ]


def nav_pages(nav: Any, groups: tuple[str, ...] = ()) -> dict[str, tuple[str, ...]]:
    """Every ``api/*.md`` entry in the zensical nav, mapped to its group path.

    Groups nest ("Internal modules" > "BrainData internals"), so the value is
    every enclosing label, outermost first.
    """
    found: dict[str, tuple[str, ...]] = {}
    if isinstance(nav, str):
        if nav.startswith("api/"):
            found[nav[len("api/") :]] = groups
    elif isinstance(nav, list):
        for entry in nav:
            found.update(nav_pages(entry, groups))
    elif isinstance(nav, dict):
        for label, entry in nav.items():
            nested = (*groups, label) if isinstance(entry, list) else groups
            found.update(nav_pages(entry, nested))
    return found


def api_nav() -> dict[str, tuple[str, ...]]:
    """The nav's API pages (relative to docs/api) and the groups each sits under."""
    with ZENSICAL_TOML.open("rb") as f:
        return nav_pages(tomllib.load(f)["project"]["nav"])


def object_key(value: Any) -> tuple[str | None, str | None] | int:
    """An identity for one documented object, stable across import aliases.

    A function or class is keyed by where it is defined, so
    ``nltools.algorithms.fdr`` and ``nltools.algorithms.corrections.fdr`` are one
    object. Anything without both markers — a module constant, a ``Literal``
    alias — falls back to ``id()``, which is exact for those but would count two
    exports sharing a singleton value (``None``, ``False``, a small int) as one.
    """
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module is None or qualname is None:
        return id(value)
    return (module, qualname)


def documented_objects(directives: list[Directive]) -> dict[Any, str]:
    """Objects a page documents, keyed by `object_key`, valued by the path used.

    A directive with a `members:` list documents those members; `members: false`
    documents none. A directive that names a module and sets no `members:`
    documents the module's public exports. Any other directive documents the
    object it names.
    """
    objects: dict[Any, str] = {}
    for directive in directives:
        target = resolve(directive.identifier)
        names = directive.members
        if names is False:
            continue
        if names is None and isinstance(target, ModuleType):
            names = module_exports(target)
        if names is None:
            objects[object_key(target)] = directive.identifier
            continue
        for name in names:
            path = f"{directive.identifier}.{name}"
            objects[object_key(resolve(path))] = path
    return objects


def index_rows(text: str) -> list[tuple[str, str, str]]:
    """The A-Z index table as ``(name, identifier, page)`` triples."""
    return [
        (m["name"], m["identifier"], m["page"])
        for m in (INDEX_ROW_RE.match(line) for line in text.splitlines())
        if m
    ]


def check() -> int:
    """Run every check; print what failed and return a process exit code."""
    nav = api_nav()
    pages = {p.relative_to(DOCS_API).as_posix(): p for p in DOCS_API.rglob("*.md")}
    errors: list[str] = []

    # Pages and nav must describe the same set of files; the nav-sync test says
    # so too, but this script reads the nav to tell internal pages apart.
    for missing in sorted(set(nav) - set(pages)):
        errors.append(f"nav lists api/{missing}, which does not exist")
    for orphan in sorted(set(pages) - set(nav)):
        errors.append(f"docs/api/{orphan} is not in the zensical nav")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    per_page: dict[str, dict[Any, str]] = {}
    for rel, path in sorted(pages.items()):
        try:
            per_page[rel] = documented_objects(parse_directives(path.read_text()))
        except LookupError as error:
            errors.append(f"docs/api/{rel}: no such object: {error.args[0]}")

    user_facing = {
        rel: objects
        for rel, objects in per_page.items()
        if INTERNAL_NAV_GROUP not in nav[rel]
    }
    homes: dict[Any, list[str]] = defaultdict(list)
    for rel, objects in user_facing.items():
        for key in objects:
            homes[key].append(rel)

    documented = 0
    for namespace in PUBLIC_NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            value = getattr(module, name)
            where = homes.get(object_key(value), [])
            if len(where) != 1:
                errors.append(
                    f"{namespace}.{name}: documented on {len(where)} user-facing "
                    f"pages ({', '.join(where) or 'none'}), expected exactly one"
                )
            documented += 1

    errors.extend(check_index(homes))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"docs/api: {len(pages)} pages, {documented} public exports, "
        "each documented on exactly one user-facing page"
    )
    return 0


def check_index(homes: dict[Any, list[str]]) -> list[str]:
    """The A-Z index must list every `nltools.algorithms` export, and link it home."""
    errors: list[str] = []
    rows = index_rows((DOCS_API / INDEX_PAGE).read_text())
    algorithms = importlib.import_module("nltools.algorithms")
    listed = [name for name, _, _ in rows]
    expected = sorted(algorithms.__all__, key=str.lower)
    if listed != expected:
        missing = sorted(set(expected) - set(listed))
        extra = sorted(set(listed) - set(expected))
        errors.append(
            f"docs/api/{INDEX_PAGE}: rows do not match nltools.algorithms.__all__ "
            f"(missing: {missing}, unexpected: {extra}, or out of order)"
        )
    for name, identifier, page in rows:
        try:
            target = resolve(identifier)
        except LookupError:
            errors.append(
                f"docs/api/{INDEX_PAGE}: {name} links to unknown {identifier}"
            )
            continue
        where = homes.get(object_key(target), [])
        if where != [page]:
            errors.append(
                f"docs/api/{INDEX_PAGE}: {name} points at {page}, "
                f"but it is documented on {where or ['nothing']}"
            )
    return errors


if __name__ == "__main__":
    sys.exit(check())
