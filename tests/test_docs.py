"""Every ```python block in README.md and docs/*.md must run. Docs that rot fail the build.

design.md is the author's paper and has no code; it is excluded.
"""

import re

import pytest

from .conftest import ROOT

FILES = [ROOT / "README.md", *sorted(p for p in (ROOT / "docs").glob("*.md") if p.name != "design.md")]
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
CASES = [(f, i, code) for f in FILES for i, code in enumerate(BLOCK.findall(f.read_text()))]


def test_docs_have_runnable_examples():
    assert len(CASES) >= 8


@pytest.mark.parametrize("path,index,code", CASES, ids=[f"{f.name}[{i}]" for f, i, _ in CASES])
def test_doc_snippet_runs(path, index, code, monkeypatch):
    monkeypatch.chdir(ROOT)  # snippets use paths relative to the repo root
    exec(compile(code, f"{path.name}[{index}]", "exec"), {"__name__": "__doc_snippet__"})  # noqa: S102
