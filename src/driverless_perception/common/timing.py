"""Wall clock timing of named stages, for the latency benchmark.

The pipelines take an optional ``timer`` and wrap their stages in ``with timer("name"):``. Without one they use
``nullcontext`` and cost nothing. GPU work has to be finished inside the stage for the time to mean anything; the
stages here end with a copy to the CPU, which waits for the GPU.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import numpy as np


def no_timer(name: str):
    return nullcontext()


class StageTimer:
    """Collects milliseconds per stage name. ``frame()`` marks where one frame's samples end, so stages that run
    once per cone can be added up per frame."""

    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.current: dict[str, float] = defaultdict(float)
        self.enabled = True

    @contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled:
                self.current[name] += 1000 * (time.perf_counter() - start)

    def add(self, name: str, ms: float) -> None:
        if self.enabled:
            self.current[name] += ms

    def frame(self) -> dict[str, float]:
        """Closes a frame: every stage that ran gets one sample, the sum of its runs in this frame."""
        done = dict(self.current)
        if self.enabled:
            for name, ms in done.items():
                self.samples[name].append(ms)
        self.current.clear()
        return done


def percentiles(values) -> dict[str, float]:
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": len(v), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}
