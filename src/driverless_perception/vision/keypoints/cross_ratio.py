"""Cross ratio of the cone's keypoints (Dhall et al. 2019, Eqs. 1-2).

Cr(a, b, c, d) = (|ac| / |ad|) / (|bc| / |bd|) does not change under projection, so the value measured on the 3D
cone has to hold in the image as well. Each edge of the cone gives one: the top together with the three points
down the left edge, and the top with the three down the right. The keypoint loss pulls both towards
``cone_model.cross_ratio_3d()``, which is what makes the network learn the shape instead of an average cone.
"""
from __future__ import annotations

import numpy as np

LEFT_EDGE = (0, 1, 2, 3)
RIGHT_EDGE = (0, 4, 5, 6)


def cross_ratio(points: np.ndarray, edge=LEFT_EDGE) -> np.ndarray:
    """Cross ratio of the four ``edge`` points of (..., 7, D) keypoints, D = 2 or 3."""
    a, b, c, d = (np.asarray(points, float)[..., i, :] for i in edge)
    dist = lambda u, v: np.maximum(np.linalg.norm(u - v, axis=-1), 1e-9)
    return (dist(a, c) / dist(a, d)) / (dist(b, c) / dist(b, d))
