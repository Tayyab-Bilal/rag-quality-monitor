"""Hook 2, step 1: decide which queries get judged."""

from __future__ import annotations

import hashlib


class Sampler:
    """Per-route sampling rates. The decision is a hash of the sample id, not a random draw, so the
    same id always gets the same answer: replays and tests are reproducible."""

    def __init__(self, default_rate: float = 0.02, route_rates: dict[str, float] | None = None) -> None:
        self.default_rate = default_rate
        self.route_rates = {"compliance": 1.0} if route_rates is None else route_rates

    def rate(self, route: str) -> float:
        return self.route_rates.get(route, self.default_rate)

    def should_sample(self, sample_id: str, route: str) -> bool:
        bucket = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) / 2**32  # in [0, 1)
        return bucket < self.rate(route)
