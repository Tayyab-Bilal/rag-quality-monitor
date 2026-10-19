"""Hook 3 gate: strict thresholds for algorithmic metrics, tolerance for judged ones."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from .runner import SuiteReport

EPS = 1e-9


ALGORITHMIC_METRICS = frozenset({"recall_at_k", "precision_at_k", "hit_rate", "mrr", "ndcg_at_k"})
JUDGED_METRICS = frozenset({"faithfulness", "answer_relevance", "context_precision", "context_recall"})


class Thresholds(BaseModel):
    # forbid + known metric names: a typo must fail loudly, because a gate that silently checks
    # nothing is worse than no gate.
    model_config = ConfigDict(extra="forbid")

    algorithmic: dict[str, float] = {}
    judged: dict[str, float] = {}
    # A judge is never fully deterministic, even at temperature 0, so judged metrics get slack.
    judged_tolerance: float = 0.05


    @field_validator("algorithmic")
    @classmethod
    def _known_algorithmic(cls, v: dict[str, float]) -> dict[str, float]:
        if unknown := sorted(set(v) - ALGORITHMIC_METRICS):
            raise ValueError(f"unknown algorithmic metric(s) {unknown}; known: {sorted(ALGORITHMIC_METRICS)}")
        return v

    @field_validator("judged")
    @classmethod
    def _known_judged(cls, v: dict[str, float]) -> dict[str, float]:
        if unknown := sorted(set(v) - JUDGED_METRICS):
            raise ValueError(f"unknown judged metric(s) {unknown}; known: {sorted(JUDGED_METRICS)}")
        return v

    def require_gate(self) -> None:
        if not self.algorithmic and not self.judged:
            raise ValueError("thresholds define no metrics: the gate would check nothing")


class Violation(BaseModel):
    route: str
    metric: str
    value: float | None
    limit: float


class Regression(BaseModel):
    route: str
    metric: str
    baseline: float
    current: float
    tolerance: float


def load_thresholds(path: str | Path) -> Thresholds:
    th = Thresholds.model_validate(json.loads(Path(path).read_text()))
    th.require_gate()
    return th


def check_thresholds(report: SuiteReport, th: Thresholds) -> list[Violation]:
    """Every route must meet every threshold. A threshold on a metric that was not measured fails:
    a gate that silently skips a check is worse than no gate."""
    th.require_gate()
    out = []
    for route, summary in report.routes.items():
        for metric, limit in th.algorithmic.items():
            value = summary.algorithmic.get(metric)
            if value is None or value < limit - EPS:
                out.append(Violation(route=route, metric=metric, value=value, limit=limit))
        for metric, limit in th.judged.items():
            value = summary.judged.get(metric)
            if value is None or value < limit - th.judged_tolerance - EPS:
                out.append(Violation(route=route, metric=metric, value=value, limit=limit - th.judged_tolerance))
    return out


def judged_comparable(report: SuiteReport, baseline: SuiteReport) -> bool:
    return report.config_hash == baseline.config_hash


def find_regressions(report: SuiteReport, baseline: SuiteReport, th: Thresholds) -> list[Regression]:
    """Metrics that got worse than the previous run, even if they are still above their thresholds.
    Algorithmic metrics may not drop at all; judged metrics may drop by the tolerance. Judged metrics are
    only compared when both reports used the same judge, embedder, prompt and golden set."""
    out = []
    for route, now in report.routes.items():
        before = baseline.routes.get(route)
        if before is None:
            continue
        kinds = [("algorithmic", 0.0)] + ([("judged", th.judged_tolerance)] if judged_comparable(report, baseline) else [])
        for kind, tol in kinds:
            now_m, before_m = getattr(now, kind), getattr(before, kind)
            for metric, old in before_m.items():
                new = now_m.get(metric)
                if new is not None and old - new > tol + EPS:
                    out.append(Regression(route=route, metric=metric, baseline=old, current=new, tolerance=tol))
    return out
