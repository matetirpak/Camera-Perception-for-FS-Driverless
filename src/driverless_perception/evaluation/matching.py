"""Match detections to ground-truth cones in the car frame (Hungarian assignment inside a distance gate)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match(detections_xy: np.ndarray, gt_xy: np.ndarray, gate_m: float = 0.5) -> list[tuple[int, int, float]]:
    """Return (det_index, gt_index, distance) for matched pairs with distance <= gate_m."""
    det = np.asarray(detections_xy).reshape(-1, 2)
    gt = np.asarray(gt_xy).reshape(-1, 2)
    if len(det) == 0 or len(gt) == 0:
        return []
    cost = np.linalg.norm(det[:, None, :] - gt[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(np.where(cost <= gate_m, cost, 1e6))
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] <= gate_m]
