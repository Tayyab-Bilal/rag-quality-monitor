"""Golden sets live in the repository as JSONL, one file per route (the file name is the route)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ..models import GoldenCase


class GoldenError(ValueError):
    """A golden file is malformed. The message always starts with `file:line`."""


def load_golden_file(path: str | Path) -> list[GoldenCase]:
    path = Path(path)
    route = path.stem
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenError(f"{where}: invalid JSON ({exc.msg})") from exc
        if not isinstance(raw, dict):
            raise GoldenError(f"{where}: each line must be a JSON object")
        if raw.setdefault("route", route) != route:
            raise GoldenError(f"{where}: route {raw['route']!r} does not match the file name {route!r}")
        try:
            case = GoldenCase.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise GoldenError(f"{where}: {problems}") from exc
        if case.case_id in seen:
            raise GoldenError(f"{where}: duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)
    return cases


def load_golden_dir(directory: str | Path) -> list[GoldenCase]:
    files = sorted(Path(directory).glob("*.jsonl"))
    if not files:
        raise GoldenError(f"{directory}: no .jsonl golden files found")
    return [case for f in files for case in load_golden_file(f)]
