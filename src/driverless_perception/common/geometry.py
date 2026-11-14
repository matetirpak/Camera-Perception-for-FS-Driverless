"""2D rigid transforms and pose interpolation (the track is flat, as in the paper), plus the full 3D pose where
the cameras need it."""
from __future__ import annotations

import math

import numpy as np


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_car(points_xy: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    """Transform (N,2) world (odom) points into the car frame of ``pose`` = (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.atleast_2d(points_xy) - [x, y]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def car_to_world(points_xy: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.atleast_2d(points_xy)
    return np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]], axis=1)


def interpolate_pose(t: int, times: np.ndarray, poses: np.ndarray) -> tuple[float, float, float]:
    """Linear interpolation of (x, y, yaw) at time ``t`` from sorted ``times`` (N,) and ``poses`` (N,3).

    FSDS ground-truth odom only arrives at ~60 Hz, so observations must be scored at an interpolated pose. The
    yaw takes the short way round but is not wrapped again afterwards, so interpolating across +-pi can return a
    value just outside it. Everything here uses the yaw through sin and cos, where that makes no difference.
    """
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    t0, t1 = times[i - 1], times[i]
    a = 0.0 if t1 == t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
    p0, p1 = poses[i - 1], poses[i]
    dyaw = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
    return (p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), p0[2] + a * dyaw)


def interpolate_pose_3d(t: int, times: np.ndarray, positions: np.ndarray,
                        quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Position and rotation matrix at time ``t`` from sorted ``times`` (N,), ``positions`` (N, 3) and
    ``quaternions`` (N, 4, x y z w like ROS), linear and spherical interpolation. None outside the time span."""
    from scipy.spatial.transform import Rotation, Slerp

    if len(times) < 2 or not times[0] <= t <= times[-1]:
        return None
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    t0, t1 = times[i - 1], times[i]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    rotation = Slerp([0.0, 1.0], Rotation.from_quat(quaternions[i - 1:i + 1]))(a)
    return (1 - a) * positions[i - 1] + a * positions[i], rotation.as_matrix()
