"""Hook 2: compare a batch with a rolling baseline of earlier batches."""

from __future__ import annotations

import statistics

from ..models import Alert, BatchRecord

BASELINE_BATCHES = 10
MIN_BASELINE_BATCHES = 3
MAX_RELATIVE_DROP = 0.15
METRICS = ("faithfulness_mean", "answer_relevance_mean", "context_precision_mean")


def check_drift(
    baseline: list[BatchRecord],
    current: BatchRecord,
    *,
    max_drop: float = MAX_RELATIVE_DROP,
    min_baseline: int = MIN_BASELINE_BATCHES,
) -> list[Alert]:
    """Alert when a metric's batch mean falls more than `max_drop` (relative) below the baseline mean.

    `baseline` must not contain `current`. Batches with another config hash are ignored: scores from a
    different judge or prompt are not comparable. Too little history means no verdict, not an alert.
    """
    same = [b for b in baseline if b.config_hash == current.config_hash and b.batch_id != current.batch_id]
    same = same[:BASELINE_BATCHES]
    if len(same) < min_baseline:
        return []
    alerts = []
    for metric in METRICS:
        history = [v for b in same if (v := getattr(b, metric)) is not None]
        now = getattr(current, metric)
        if len(history) < min_baseline or now is None:
            continue
        base = statistics.fmean(history)
        if base > 0 and (base - now) / base > max_drop + 1e-9:  # tolerance: 1.0 -> 0.85 is exactly 15%
            name = metric.removesuffix("_mean")
            alerts.append(
                Alert(
                    kind="quality_drift", tenant_id=current.tenant_id, route=current.route,
                    message=f"{name} {now:.2f} is {(base - now) / base:.0%} below the baseline {base:.2f}"
                    f" (last {len(history)} batches)",
                    value=now, threshold=base * (1 - max_drop),
                )
            )  # fmt: skip
    return alerts
