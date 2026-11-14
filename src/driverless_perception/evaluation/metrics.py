"""Range-binned detection metrics (the paper reports per-distance results: Figs. 8, 9; Dhall Fig. 7)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .matching import match

DEFAULT_BINS = (0.0, 5.0, 10.0, 15.0, 20.0)


@dataclass
class BinnedCounts:
    edges: tuple[float, ...] = DEFAULT_BINS
    tp: np.ndarray = field(default=None)
    fp: np.ndarray = field(default=None)
    fn: np.ndarray = field(default=None)
    errors: list[list[float]] = field(default=None)

    def __post_init__(self):
        n = len(self.edges) - 1
        self.tp, self.fp, self.fn = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        self.errors = [[] for _ in range(n)]

    def _bin(self, r: float) -> int | None:
        i = int(np.searchsorted(self.edges, r, side="right")) - 1
        return i if 0 <= i < len(self.edges) - 1 else None

    def add_frame(self, det_xy: np.ndarray, gt_xy: np.ndarray, gate_m: float = 0.5,
                  ignore_xy: np.ndarray | None = None) -> None:
        """Accumulate one frame; GT should already be restricted to the sensor's field of view.

        ``ignore_xy`` are real cones that don't count as expected, for example just outside the field of view. A
        detection that matches none of the expected cones but lies within the gate of one of these is neither right
        nor wrong.
        """
        det_xy, gt_xy = np.asarray(det_xy).reshape(-1, 2), np.asarray(gt_xy).reshape(-1, 2)
        pairs = match(det_xy, gt_xy, gate_m)
        md, mg = {p[0] for p in pairs}, {p[1] for p in pairs}
        if ignore_xy is not None and len(ignore_xy):
            ignore_xy = np.asarray(ignore_xy).reshape(-1, 2)
            for d in set(range(len(det_xy))) - md:
                if np.min(np.linalg.norm(ignore_xy - det_xy[d], axis=1)) <= gate_m:
                    md.add(d)
        for d, g, dist in pairs:
            b = self._bin(float(np.hypot(*gt_xy[g])))
            if b is not None:
                self.tp[b] += 1
                self.errors[b].append(dist)
        for d in set(range(len(det_xy))) - md:
            b = self._bin(float(np.hypot(*det_xy[d])))
            if b is not None:
                self.fp[b] += 1
        for g in set(range(len(gt_xy))) - mg:
            b = self._bin(float(np.hypot(*gt_xy[g])))
            if b is not None:
                self.fn[b] += 1

    def summary(self) -> list[dict]:
        out = []
        for i in range(len(self.edges) - 1):
            tp, fp, fn = self.tp[i], self.fp[i], self.fn[i]
            out.append({
                "bin_m": [self.edges[i], self.edges[i + 1]],
                "recall": float(tp / (tp + fn)) if tp + fn else None,
                "precision": float(tp / (tp + fp)) if tp + fp else None,
                "pos_rmse_m": float(np.sqrt(np.mean(np.square(self.errors[i])))) if self.errors[i] else None,
                "n_gt": int(tp + fn),
            })
        return out
