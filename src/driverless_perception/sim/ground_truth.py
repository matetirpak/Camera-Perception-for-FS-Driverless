"""Ground-truth track geometry and the oracle driver.

This project does no planning or control of its own. The car is driven along the centre line of the ground-truth
track so that the perception pipelines only have to observe.
"""
from __future__ import annotations

import math

import numpy as np

from ..common.types import ConeColor

# FSDS: a steering command of 1.0 turns the inner front wheel by about 25 deg. Positive steering is to the RIGHT.
MAX_STEER_RAD = math.radians(25.0)


def centerline(xy: np.ndarray, colors: np.ndarray, spacing: float = 0.5) -> np.ndarray:
    """Closed centre line through the track, resampled to roughly `spacing` metres between points.

    Each blue cone is paired with its nearest yellow cone and the midpoints are chained by nearest neighbour,
    starting just ahead of the car (the car starts at the origin facing +x).
    """
    blue, yellow = xy[colors == ConeColor.BLUE], xy[colors == ConeColor.YELLOW]
    mids = np.array([(b + yellow[np.argmin(np.linalg.norm(yellow - b, axis=1))]) / 2 for b in blue])
    ahead = mids[:, 0] > 0
    start = int(np.argmin(np.where(ahead, np.linalg.norm(mids, axis=1), np.inf)))
    order, left = [start], set(range(len(mids))) - {start}
    while left:
        last = mids[order[-1]]
        nxt = min(left, key=lambda i: np.linalg.norm(mids[i] - last))
        order.append(nxt)
        left.remove(nxt)
    path = mids[order]
    # Nearest neighbour can pick the cone pair behind the car as the second point. Make sure we drive forwards.
    if path[1, 0] < path[0, 0]:
        path = np.vstack([path[:1], path[:0:-1]])
    return resample_closed(path, spacing)


def resample_closed(path: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed polyline to points spaced evenly along its length."""
    closed = np.vstack([path, path[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / spacing), 3)
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], axis=1)


def curvature(path: np.ndarray) -> np.ndarray:
    """Unsigned curvature of a closed polyline at every point (circle through the neighbouring points)."""
    a, b, c = np.roll(path, 1, axis=0), path, np.roll(path, -1, axis=0)
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return 2.0 * cross / np.maximum(ab * bc * ca, 1e-9)


def speed_profile(path: np.ndarray, v_max: float, a_lat: float = 4.0, a_brake: float = 3.0,
                  smooth_m: float = 3.0) -> np.ndarray:
    """Target speed at every path point: limited by lateral acceleration in corners and by braking ahead of them."""
    spacing = float(np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    k = curvature(path)
    w = max(int(smooth_m / spacing), 1)
    k = np.convolve(np.concatenate([k[-w:], k, k[:w]]), np.ones(2 * w + 1) / (2 * w + 1), mode="valid")
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(k, 1e-6)))
    # Two backward passes around the loop so braking zones wrap across the start.
    n = len(v)
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            v[i] = min(v[i], math.sqrt(v[(i + 1) % n] ** 2 + 2 * a_brake * spacing))
    return v


def start_line(xy: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment across the track between the big orange cones (2 or 4), used to count laps from GT odom."""
    orange = xy[colors == ConeColor.ORANGE_BIG]
    if len(orange) not in (2, 4):
        raise ValueError(f"expected 2 or 4 big orange cones, got {len(orange)}")
    # The two cones furthest apart are on opposite sides of the track. With 4 cones, average each side.
    d = np.linalg.norm(orange[:, None] - orange[None], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    if len(orange) == 2:
        return orange[i], orange[j]
    side_i = orange[np.argsort(d[i])[:2]].mean(axis=0)
    side_j = orange[np.argsort(d[j])[:2]].mean(axis=0)
    return side_i, side_j


def crossed(p0: np.ndarray, p1: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """True if the motion p0 -> p1 crosses the segment a-b."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (side(a, b, p0) * side(a, b, p1) < 0) and (side(p0, p1, a) * side(p0, p1, b) < 0)


def nearest_index(path: np.ndarray, x: float, y: float) -> int:
    return int(np.argmin(np.linalg.norm(path - [x, y], axis=1)))


def pure_pursuit(path: np.ndarray, pose: tuple[float, float, float], lookahead: float = 4.0,
                 wheelbase: float = 1.55) -> float:
    """FSDS steering command in [-1, 1] (+1 is full right) that follows a closed path from pose (x, y, yaw)."""
    x, y, yaw = pose
    i = nearest_index(path, x, y)
    n = len(path)
    j = i
    for k in range(n):
        j = (i + k) % n
        if np.linalg.norm(path[j] - [x, y]) >= lookahead:
            break
    dx, dy = path[j] - [x, y]
    lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy  # offset of the target to the left of the car
    delta = math.atan2(2 * wheelbase * lateral, dx * dx + dy * dy)
    return float(np.clip(-delta / MAX_STEER_RAD, -1.0, 1.0))
