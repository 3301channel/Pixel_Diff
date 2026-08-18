"""Lightweight stage timing for the comparison pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable


class StageTimer:
    """Record consecutive pipeline stages and the overall elapsed time."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started_at = clock()
        self._last_checkpoint = self._started_at
        self._metrics: dict[str, int] = {}

    def checkpoint(self, name: str) -> int:
        """Finish a named stage and return its duration in milliseconds."""
        now = self._clock()
        elapsed_ms = int(round((now - self._last_checkpoint) * 1000))
        self._metrics[f"timing_{name}_ms"] = elapsed_ms
        self._last_checkpoint = now
        return elapsed_ms

    def finish(self) -> dict[str, int]:
        """Return all stage durations plus total elapsed milliseconds."""
        now = self._clock()
        return {
            **self._metrics,
            "elapsed_ms": int(round((now - self._started_at) * 1000)),
        }
