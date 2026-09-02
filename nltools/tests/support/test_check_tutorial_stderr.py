"""Tests for scripts/check_tutorial_stderr.py — the docs-site stderr gate.

The script isn't a package member, so it's loaded by file path via importlib.
It pairs each tutorial page with its MyST execute-cache entry (raw kernel
outputs, before `output_stderr: remove-warn` strips them from the site) and
fails on any stderr stream that isn't explicitly allowlisted.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "check_tutorial_stderr.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_tutorial_stderr", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the defining module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PAGE = """---
kernelspec:
  name: python3
  display_name: Python 3
---

# Title

Some prose.

```{code-cell} python3
import numpy as np
x = np.ones(3)
```

More prose with a plain fence that is NOT executed:

```python
print("not a cell")
```

```{code-cell} python3
:tags: [remove-input]
print(x.sum())
```
"""


class TestCodeCells:
    def test_extracts_only_code_cells_and_strips_directive_options(self, mod):
        cells = mod.code_cells(PAGE)
        assert cells == ["import numpy as np\nx = np.ones(3)", "print(x.sum())"]

    def test_kernel_name_from_frontmatter(self, mod):
        assert mod.kernel_name(PAGE) == "python3"

    def test_kernel_name_defaults_when_missing(self, mod):
        assert mod.kernel_name("# no frontmatter\n") == "python3"


class TestCacheKey:
    def test_matches_mystmd_hash_layout(self, mod):
        # mystmd: md5(kernelSpec.name + JSON.stringify([{kind, content, raisesException}]))
        # JSON.stringify emits compact separators, so the expected digest is the
        # md5 of exactly this string.
        expected = hashlib.md5(
            b'python3[{"kind":"block","content":"print(1)","raisesException":false}]'
        ).hexdigest()
        assert mod.cache_key(["print(1)"], "python3") == expected

    def test_non_ascii_source_is_not_escaped(self, mod):
        # JS keeps non-ASCII characters literal; Python must not \\u-escape them.
        expected = hashlib.md5(
            'python3[{"kind":"block","content":"x = \\"—\\"","raisesException":false}]'.encode()
        ).hexdigest()
        assert mod.cache_key(['x = "—"'], "python3") == expected


class TestFindStderr:
    def test_reports_cell_index_and_text(self, mod):
        outputs = [
            [{"output_type": "stream", "name": "stdout", "text": "fine\n"}],
            [
                {"output_type": "execute_result", "data": {"text/plain": "1"}},
                {
                    "output_type": "stream",
                    "name": "stderr",
                    "text": "UserWarning: boo\n",
                },
            ],
            [{"output_type": "error", "ename": "ValueError", "evalue": "x"}],
        ]
        assert mod.find_stderr(outputs) == [(1, "UserWarning: boo\n")]

    def test_joins_list_text(self, mod):
        outputs = [
            [{"output_type": "stream", "name": "stderr", "text": ["a\n", "b\n"]}]
        ]
        assert mod.find_stderr(outputs) == [(0, "a\nb\n")]


def _write_docs(mod, tmp_path: Path, page_md: str, outputs) -> Path:
    """Lay out docs/tutorials/<page>.md plus its execute-cache entry (if any)."""
    docs = tmp_path / "docs"
    page = docs / "tutorials" / "basics" / "01_demo.md"
    page.parent.mkdir(parents=True)
    page.write_text(page_md)
    cache = docs / "_build" / "execute"
    cache.mkdir(parents=True)
    if outputs is not None:
        key = mod.cache_key(mod.code_cells(page_md), mod.kernel_name(page_md))
        (cache / f"{key}.json").write_text(json.dumps(outputs))
    return docs


class TestCheckPages:
    def test_clean_pages_produce_no_hits(self, mod, tmp_path):
        docs = _write_docs(
            mod,
            tmp_path,
            PAGE,
            [[], [{"output_type": "stream", "name": "stdout", "text": "3.0\n"}]],
        )
        hits, missing = mod.check_pages(docs)
        assert hits == [] and missing == []

    def test_stderr_hit_carries_page_cell_and_first_code_line(self, mod, tmp_path):
        docs = _write_docs(
            mod,
            tmp_path,
            PAGE,
            [
                [],
                [
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": "FutureWarning: x\n",
                    }
                ],
            ],
        )
        hits, missing = mod.check_pages(docs)
        assert missing == []
        assert len(hits) == 1
        hit = hits[0]
        assert hit.page == "tutorials/basics/01_demo.md"
        assert hit.cell_index == 1
        assert hit.first_line == "print(x.sum())"
        assert hit.text == "FutureWarning: x\n"

    def test_missing_cache_entry_is_reported(self, mod, tmp_path):
        docs = _write_docs(mod, tmp_path, PAGE, None)
        hits, missing = mod.check_pages(docs)
        assert hits == []
        assert missing == ["tutorials/basics/01_demo.md"]

    def test_allowlist_suppresses_matching_hit_only(self, mod, tmp_path):
        docs = _write_docs(
            mod,
            tmp_path,
            PAGE,
            [
                [
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": "tolerated noise\n",
                    }
                ],
                [{"output_type": "stream", "name": "stderr", "text": "real problem\n"}],
            ],
        )
        hits, _ = mod.check_pages(
            docs, allowlist=[("tutorials/basics/01_demo.md", "tolerated")]
        )
        assert [h.text for h in hits] == ["real problem\n"]

    def test_stale_cache_entries_are_ignored(self, mod, tmp_path):
        # An entry left over from an older version of a page must not be scanned.
        docs = _write_docs(mod, tmp_path, PAGE, [[], []])
        stale = docs / "_build" / "execute" / ("0" * 32 + ".json")
        stale.write_text(
            json.dumps([[{"output_type": "stream", "name": "stderr", "text": "old\n"}]])
        )
        hits, missing = mod.check_pages(docs)
        assert hits == [] and missing == []


class TestMain:
    def test_exit_zero_when_clean(self, mod, tmp_path, capsys):
        docs = _write_docs(mod, tmp_path, PAGE, [[], []])
        assert mod.main(["--docs-dir", str(docs)]) == 0
        assert "no stderr" in capsys.readouterr().out

    def test_exit_one_on_stderr(self, mod, tmp_path, capsys):
        docs = _write_docs(
            mod,
            tmp_path,
            PAGE,
            [
                [
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": "UserWarning: boo\n",
                    }
                ],
                [],
            ],
        )
        assert mod.main(["--docs-dir", str(docs)]) == 1
        out = capsys.readouterr().out
        assert (
            "01_demo.md" in out
            and "UserWarning: boo" in out
            and "import numpy as np" in out
        )

    def test_exit_one_on_missing_cache(self, mod, tmp_path, capsys):
        docs = _write_docs(mod, tmp_path, PAGE, None)
        assert mod.main(["--docs-dir", str(docs)]) == 1
        assert "no execute-cache entry" in capsys.readouterr().out

    def test_default_allowlist_is_empty(self, mod):
        assert mod.ALLOWED_STDERR == ()
