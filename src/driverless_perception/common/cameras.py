"""Camera model and the frames around it.

The map frame is ENU with its origin where the car started, which is what the simulator reports for car and image
poses. A camera's body frame has x forward, y left and z up like the car; its optical frame, the one the bridge
publishes as ``fsds/<camera>``, has x right, y down and z forward. Orientations are quaternions (w, x, y, z).
Pixel i spans i to i + 1, so the centre of the image is at (width / 2, height / 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# the camera body frame expressed in the optical frame
BODY_TO_OPTICAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class Intrinsics:
    """FSDS cameras have square pixels, no distortion and the principal point in the middle of the image."""

    width: int
    height: int
    fov_deg: float

    @property
    def f(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.fov_deg) / 2)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    def project(self, points_opt: np.ndarray) -> np.ndarray:
        """(N, 2) pixels of (N, 3) points in the optical frame."""
        return self.f * points_opt[:, :2] / points_opt[:, 2:3] + [self.cx, self.cy]


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_to_optical(points: np.ndarray, position, orientation) -> np.ndarray:
    """(N, 3) map frame points in the optical frame of a camera at ``position`` with ``orientation``."""
    body = (np.atleast_2d(points) - position) @ quat_to_matrix(orientation)
    return np.stack([-body[:, 1], -body[:, 2], body[:, 0]], axis=1)


def optical_to_world(points: np.ndarray, position, orientation) -> np.ndarray:
    """The inverse of ``world_to_optical``."""
    body = np.stack([points[:, 2], -points[:, 0], -points[:, 1]], axis=1)
    return body @ quat_to_matrix(orientation).T + position


def up_in_optical(orientation) -> np.ndarray:
    """The map frame's upward direction in the optical frame of a camera with ``orientation``, (0, -1, 0) for a
    level camera."""
    return BODY_TO_OPTICAL @ quat_to_matrix(orientation).T @ np.array([0.0, 0.0, 1.0])


def camera_pose(car_position: np.ndarray, car_rotation: np.ndarray, mount_rotation: np.ndarray,
                mount_translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position and orientation of a camera in the map frame, from the car's pose and the camera's mount as the
    bridge publishes it (a transform into the optical frame)."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(car_rotation @ mount_rotation @ BODY_TO_OPTICAL).as_quat()
    return car_position + car_rotation @ mount_translation, np.array([w, x, y, z])
