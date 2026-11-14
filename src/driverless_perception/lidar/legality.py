"""Cone legality check: compare a cluster's point count with the number expected for a cone at that distance.

Gosala et al. Eq. 1:  E(n_d) = 1/2 * h_c / (2 d tan(r_v / 2)) * w_c / (2 d tan(r_h / 2))
(the 1/2 accounts for the triangular silhouette). The angular resolutions r_v and r_h come from the LiDAR's
settings, the cone size from the rules.
"""
from __future__ import annotations

import math

CONE_WIDTH_M = 0.228
CONE_HEIGHT_M = 0.325


def expected_points(distance_m: float, r_v_rad: float, r_h_rad: float,
                    cone_h: float = CONE_HEIGHT_M, cone_w: float = CONE_WIDTH_M) -> float:
    rows = cone_h / (2.0 * distance_m * math.tan(r_v_rad / 2.0))
    cols = cone_w / (2.0 * distance_m * math.tan(r_h_rad / 2.0))
    return 0.5 * rows * cols


def is_legal(n_points: int, distance_m: float, r_v_rad: float, r_h_rad: float,
             low: float = 0.3, high: float = 2.0) -> bool:
    """Accept clusters whose count lies within [low, high] times the expected count."""
    e = expected_points(distance_m, r_v_rad, r_h_rad)
    return low * e <= n_points <= high * e
