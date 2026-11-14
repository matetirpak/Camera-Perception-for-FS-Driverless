"""The small cone as FSDS draws it, and its 7 keypoints (Dhall et al. 2019).

The sizes were measured from depth images of cones in front of the cameras, not taken from the rules: FSDS cones
are 1.5 cm shorter (docs/notes/simulator-notes.md). The keypoints are the top (index 0), then the stripe ends and
the bottom of the body down the left edge (1 to 3) and the same down the right (4 to 6). A cone of revolution has
no fixed edges, ``silhouette_keypoints`` computes where the lines of sight touch it.
"""
from __future__ import annotations

import numpy as np

TOP = 0.31
STRIPE_TOP = 0.211
STRIPE_BOTTOM = 0.128
BODY_BOTTOM = 0.035
# heights of the edge keypoints, top to bottom
EDGE_HEIGHTS = (STRIPE_TOP, STRIPE_BOTTOM, BODY_BOTTOM)


def body_radius(z: float) -> float:
    """Radius of the cone body at height z above the ground, fitted to the measurement (valid from 3.5 to 30 cm)."""
    return 0.0835 - 0.188 * z


def model_points() -> np.ndarray:
    """The 7 keypoints in a cone frame for PnP: origin on the ground at the cone's axis, x to the right as seen by the
    camera, y away from the camera, z up. (7, 3) in metres."""
    left = [[-body_radius(z), 0.0, z] for z in EDGE_HEIGHTS]
    right = [[body_radius(z), 0.0, z] for z in EDGE_HEIGHTS]
    return np.array([[0.0, 0.0, TOP], *left, *right])


def cross_ratio_3d() -> float:
    """Cross ratio of the top and the three points down one edge, the value the keypoint loss pulls towards."""
    p = model_points()[[0, 1, 2, 3]]
    d = lambda a, b: np.linalg.norm(p[a] - p[b])
    return float((d(0, 2) / d(0, 3)) / (d(1, 2) / d(1, 3)))


def silhouette_keypoints(cone_xy, ground_z: float, camera_position) -> np.ndarray:
    """The 7 keypoints of a cone standing at ``cone_xy`` as world points (7, 3), for a camera at
    ``camera_position``. The edge points are where the lines of sight touch the body at each height, so their
    projections are the left and right edge of the cone in the image."""
    axis = np.array([cone_xy[0], cone_xy[1]])
    to_camera = np.asarray(camera_position[:2], float) - axis
    d = float(np.linalg.norm(to_camera))
    w = to_camera / d
    # left as seen from the camera: the viewing direction (-w) turned 90 degrees counterclockwise
    left_dir = np.array([w[1], -w[0]])
    points = [[*axis, ground_z + TOP]]
    for side in (1.0, -1.0):
        for z in EDGE_HEIGHTS:
            r = body_radius(z)
            # tangent point of the line of sight on a circle of radius r seen from distance d
            xy = axis + (r * r / d) * w + side * r * np.sqrt(max(0.0, 1 - (r / d) ** 2)) * left_dir
            points.append([*xy, ground_z + z])
    return np.array(points)
