import numpy as np
import pytest

from driverless_perception.common.cameras import Intrinsics, world_to_optical
from driverless_perception.common.cone_model import (
    TOP,
    body_radius,
    cross_ratio_3d,
    model_points,
    silhouette_keypoints,
)
from driverless_perception.vision.keypoints.cross_ratio import RIGHT_EDGE, cross_ratio

CAMERA = np.array([0.0, 0.0, 1.25])
LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
INTR = Intrinsics(1280, 960, 54.5)


def test_model_points_are_symmetric():
    p = model_points()
    assert p[0].tolist() == [0.0, 0.0, TOP]
    assert np.allclose(p[1:4, 0], -p[4:7, 0]) and (p[1:4, 0] < 0).all()
    assert np.allclose(p[1:4, 2], p[4:7, 2])
    # the edge narrows towards the top
    assert body_radius(0.2) < body_radius(0.1)


def test_cross_ratio_is_close_to_the_papers_cone():
    # AMZ's real cone had 1.394, the FSDS cone is a bit shorter with a round top
    assert cross_ratio_3d() == pytest.approx(1.42, abs=0.02)
    p = model_points()
    assert cross_ratio(p, RIGHT_EDGE) == pytest.approx(cross_ratio_3d())


def test_silhouette_keypoints_land_on_the_right_sides():
    for cone in ([6.0, 0.0], [8.0, 2.5], [5.0, -1.5]):
        pts = silhouette_keypoints(cone, 0.25, CAMERA)
        uv = INTR.project(world_to_optical(pts, CAMERA, LEVEL))
        axis_u = INTR.project(world_to_optical(np.array([[*cone, 0.4]]), CAMERA, LEVEL))[0, 0]
        assert (uv[1:4, 0] < axis_u).all() and (uv[4:7, 0] > axis_u).all()
        # top above the stripe above the base, in the image v grows downwards
        assert uv[0, 1] < uv[1, 1] < uv[2, 1] < uv[3, 1]
        # the edge points are at the body radius from the axis
        radial = np.hypot(pts[1:, 0] - cone[0], pts[1:, 1] - cone[1])
        assert np.allclose(radial, [body_radius(z) for z in pts[1:, 2] - 0.25])
