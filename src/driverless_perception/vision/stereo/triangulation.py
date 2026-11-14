"""Triangulating matched points with the pose each image was taken from, and the cone position from them.

Every camera gets its own pose, so the two images could come from different moments. Their relative pose has to be
right to a few millimetres though: with the 0.32 m baseline, 2.5 cm of false motion along the driving direction
between the cameras moves a cone 2 m to the side by about 0.16 m in depth. In a race the node therefore only uses
pairs from the same render.
"""
from __future__ import annotations

import cv2
import numpy as np

from ...common.cameras import BODY_TO_OPTICAL, Intrinsics, quat_to_matrix
from ...common.cone_model import BODY_BOTTOM, TOP, body_radius

# with features spread evenly across the cone's width in the image, the median one sits sqrt(3)/2 of the radius
# in front of the axis
FRONT = np.sqrt(3) / 2 * body_radius((BODY_BOTTOM + TOP) / 2)


def projection(intrinsics: Intrinsics, position, orientation) -> np.ndarray:
    """3x4 matrix that projects world points into the image (pixel i spanning i to i + 1)."""
    R = BODY_TO_OPTICAL @ quat_to_matrix(orientation).T
    return intrinsics.matrix @ np.hstack([R, (-R @ np.asarray(position, float))[:, None]])


def fundamental(left: tuple, right: tuple) -> np.ndarray:
    """Fundamental matrix from the left to the right image, each given as (intrinsics, position, orientation)."""
    P1, P2 = projection(*left), projection(*right)
    # camera centre of the left image, then F = [e2]x P2 P1^+
    centre = np.append(np.asarray(left[1], float), 1.0)
    e2 = P2 @ centre
    skew = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    return skew @ P2 @ np.linalg.pinv(P1)


def triangulate(pts_left: np.ndarray, pts_right: np.ndarray, left: tuple, right: tuple) -> np.ndarray:
    """(N, 3) world points from matched pixels, the cameras given as (intrinsics, position, orientation)."""
    if len(pts_left) == 0:
        return np.zeros((0, 3))
    X = cv2.triangulatePoints(projection(*left), projection(*right), np.asarray(pts_left, float).T,
                              np.asarray(pts_right, float).T)
    return (X[:3] / X[3]).T


def cone_from_points(points: np.ndarray, ground_z: float, guess_xy: np.ndarray, camera_xy: np.ndarray,
                     gate: float = 1.0, min_points: int = 3) -> tuple[np.ndarray | None, np.ndarray]:
    """The cone's base (x, y) from its triangulated feature points, None if fewer than ``min_points`` are on the
    cone, and which points are.

    Points count if they lie between the ground and the cone's top and within ``gate`` of the ``guess`` (the PnP
    estimate), which drops features on the tarmac and mismatches. Their median lies on the side of the cone facing
    the camera, so it is moved back along the line of sight by ``FRONT``.
    """
    dz = points[:, 2] - ground_z
    near = np.hypot(points[:, 0] - guess_xy[0], points[:, 1] - guess_xy[1]) <= gate
    keep = (dz > 0.02) & (dz < TOP) & near
    if keep.sum() < min_points:
        return None, keep
    xy = np.median(points[keep, :2], axis=0)
    ray = xy - camera_xy
    return xy + FRONT * ray / np.linalg.norm(ray), keep
