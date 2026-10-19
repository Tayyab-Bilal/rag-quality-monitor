"""CLI for Hook 3, the CI gate.

    python -m rag_quality_monitor.suite --golden examples/golden --thresholds examples/thresholds.json

Exit codes: 0 pass, 1 threshold failure or regression, 2 bad input (golden file, thresholds, pipeline).
The pipeline is a Python file that defines `build_pipeline() -> (retriever, generator)`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

from .hook3.golden import GoldenError, load_golden_dir
from .hook3.runner import SuiteReport, run_suite
from .hook3.thresholds import (
    Regression,
    Violation,
    check_thresholds,
    find_regressions,
    judged_comparable,
    load_thresholds,
)
from .judge.env import embedder_from_env, judge_from_env


def load_pipeline(path: str):
    spec = importlib.util.spec_from_file_location("rqm_pipeline", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_pipeline()


def format_report(report: SuiteReport, violations: list[Violation], regressions: list[Regression]) -> str:
    lines = [f"Golden suite (k={report.k}, config {report.config_hash})", ""]
    for route, s in report.routes.items():
        lines.append(f"route {route} ({s.cases} cases)")
        for name, value in {**s.algorithmic, **s.judged}.items():
            lines.append(f"  {name:<20}{value:6.3f}")
    lines.append("")
    if violations:
        lines.append("THRESHOLD FAILURES")
        lines += [f"  {v.route}: {v.metric} = {v.value if v.value is None else round(v.value, 3)} < {v.limit:.3f}" for v in violations]
    if regressions:
        lines.append("REGRESSIONS vs baseline")
        lines += [f"  {r.route}: {r.metric} {r.baseline:.3f} -> {r.current:.3f}" for r in regressions]
    if not violations and not regressions:
        lines.append("PASS: all thresholds met, no regressions")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag_quality_monitor.suite", description=__doc__.split("\n")[0])
    p.add_argument("--golden", required=True, help="directory of <route>.jsonl files")
    p.add_argument("--thresholds", required=True, help="thresholds JSON")
    p.add_argument("--baseline", help="previous report JSON; regressions against it fail the run")
    p.add_argument("--write", help="write this run's report JSON here")
    p.add_argument("--pipeline", default="examples/acme_rag.py", help="file defining build_pipeline()")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)
    baseline = None
    try:
        golden = load_golden_dir(args.golden)
        thresholds = load_thresholds(args.thresholds)
        if args.baseline and Path(args.baseline).exists():
            baseline = SuiteReport.model_validate_json(Path(args.baseline).read_text())
    except (GoldenError, OSError, ValueError) as exc:  # ValidationError and JSONDecodeError are ValueErrors
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        retriever, generator = load_pipeline(args.pipeline)
    except Exception as exc:  # the pipeline is user code: any failure to import or build it is bad input
        print(f"error: cannot load pipeline {args.pipeline}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report = asyncio.run(
        run_suite(golden, retriever, generator, judge_from_env(), embedder_from_env(), k=args.k)
    )
    violations = check_thresholds(report, thresholds)
    regressions = []
    if baseline is not None:
        regressions = find_regressions(report, baseline, thresholds)
        if not judged_comparable(report, baseline):
            print("note: the baseline used a different judge, embedder, prompt or golden set; "
                  "judged metrics were not compared (algorithmic ones were)")  # fmt: skip
    elif args.baseline:
        print(f"note: no baseline at {args.baseline}; skipping the regression check")
    print(format_report(report, violations, regressions))
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(report.model_dump_json(indent=2))
    return 1 if violations or regressions else 0


if __name__ == "__main__":
    sys.exit(main())
