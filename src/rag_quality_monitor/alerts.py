"""Where alerts go. A Protocol keeps paging/Slack/email out of the library."""

from __future__ import annotations

from typing import Protocol

from .models import Alert


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None: ...


class ListAlertSink:
    """Collects alerts in memory. Used by tests and the demo."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)
