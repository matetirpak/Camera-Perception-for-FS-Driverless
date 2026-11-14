"""The error of an estimated cone position, split into the parts that matter.

A position is judged from where the camera stood: how far off the distance is (range, negative means the cone was
put too close), how far off across the line of sight (lateral), and how far off in height (vertical). The 3D error
is all three together, the horizontal error leaves the height out. Dhall et al. and Gosala et al. report the
distance error, which is the range part. The cone base sits on the ground in every pipeline except PnP, which
estimates its height too, so only there is the vertical part not zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

DISTANCE_BINS = (0, 5, 10, 15, 20)


@dataclass(frozen=True)
class PositionError:
    """One estimated cone against the true one, all in metres."""

    distance_m: float
    estimate_m: float
    error_3d_m: float
    error_xy_m: float
    range_m: float
    lateral_m: float
    vertical_m: float


def position_error(estimate, truth_xy, ground_z: float, camera_xy) -> PositionError:
    """``estimate`` is the cone base as (x, y) or (x, y, z), ``truth_xy`` the true base, both in the map frame."""
    estimate = np.asarray(estimate, float)
    z = float(estimate[2]) if len(estimate) > 2 else ground_z
    point, truth = np.array([estimate[0], estimate[1], z]), np.array([truth_xy[0], truth_xy[1], ground_z])
    ray = truth[:2] - np.asarray(camera_xy, float)
    distance = float(np.hypot(*ray))
    across = np.array([-ray[1], ray[0]]) / distance
    delta = point[:2] - truth[:2]
    return PositionError(distance, float(np.hypot(*(point[:2] - camera_xy))), float(np.linalg.norm(point - truth)),
                         float(np.hypot(*delta)), float(np.hypot(*(point[:2] - camera_xy)) - distance),
                         float(delta @ across), float(z - ground_z))


def summarise(errors: list[PositionError], bins=DISTANCE_BINS) -> list[dict]:
    """Per distance bin: how many cones, the horizontal and 3D error, and the three parts of it."""
    rows = np.array([[e.distance_m, e.error_xy_m, e.error_3d_m, e.range_m, e.lateral_m, e.vertical_m]
                     for e in errors], float).reshape(-1, 6)
    out = []
    for lo, hi in pairwise(bins):
        sel = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
        stats = {"distance_m": [lo, hi], "cones": len(sel)}
        if len(sel):
            stats.update(
                median_m=float(np.median(sel[:, 1])), rmse_m=float(np.sqrt(np.mean(sel[:, 1] ** 2))),
                p90_m=float(np.percentile(sel[:, 1], 90)),
                median_3d_m=float(np.median(sel[:, 2])), rmse_3d_m=float(np.sqrt(np.mean(sel[:, 2] ** 2))),
                range_bias_m=float(np.median(sel[:, 3])), range_spread_m=float(np.std(sel[:, 3])),
                range_rmse_m=float(np.sqrt(np.mean(sel[:, 3] ** 2))),
                lateral_rmse_m=float(np.sqrt(np.mean(sel[:, 4] ** 2))),
                vertical_bias_m=float(np.median(sel[:, 5])),
                vertical_rmse_m=float(np.sqrt(np.mean(sel[:, 5] ** 2))),
            )
        out.append(stats)
    return out


def spread_by_distance(errors: list[PositionError], targets, tolerance: float = 0.25) -> list[dict]:
    """The estimated distances of the cones that really stood at each of ``targets`` metres, for the plot of
    Gosala et al. (Fig. 9a): every measurement of a cone at a known distance, not just a summary."""
    rows = np.array([[e.distance_m, e.estimate_m] for e in errors], float).reshape(-1, 2)
    out = []
    for target in targets:
        sel = rows[np.abs(rows[:, 0] - target) <= tolerance]
        out.append({"distance_m": float(target), "cones": len(sel), "estimates_m": sel[:, 1].tolist()})
    return out
