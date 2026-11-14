import math

import numpy as np

from driverless_perception.common.types import ConeColor
from driverless_perception.sim.ground_truth import (
    MAX_STEER_RAD,
    centerline,
    crossed,
    nearest_index,
    pure_pursuit,
    resample_closed,
    speed_profile,
    start_line,
)


def oval(n=40, a=20.0, b=10.0):
    """A closed track as cone pairs 3 m apart, the car starting at the origin facing +x."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centre = np.stack([a * np.sin(t), b - b * np.cos(t)], axis=1)
    tangent = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    xy = np.vstack([centre + 1.5 * normal, centre - 1.5 * normal])
    colors = np.array([ConeColor.BLUE] * n + [ConeColor.YELLOW] * n)
    return xy, colors


def test_resample_keeps_the_length_and_spacing():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.stack([10 * np.cos(t), 10 * np.sin(t)], axis=1)
    perimeter = 200 * np.linalg.norm(circle[1] - circle[0])
    path = resample_closed(circle, spacing=1.0)
    assert len(path) == int(perimeter)
    # points sit along the polyline, so the straight line between two of them is a hair shorter than the step
    gaps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.allclose(gaps, perimeter / len(path), rtol=1e-3)


def test_centerline_runs_between_the_cone_pairs():
    xy, colors = oval()
    path = centerline(xy, colors, spacing=0.5)
    # the centre line keeps the same distance to both sides, the cone pairs themselves are 3 m apart
    blue = np.linalg.norm(path[:, None] - xy[None, colors == ConeColor.BLUE], axis=2).min(axis=1)
    yellow = np.linalg.norm(path[:, None] - xy[None, colors == ConeColor.YELLOW], axis=2).min(axis=1)
    assert np.abs(blue - yellow).max() < 0.35
    assert 1.4 < blue.min() and blue.max() < 2.2
    assert path[1, 0] > path[0, 0]


def test_speed_profile_brakes_before_a_corner():
    straight = np.stack([np.arange(0, 60, 0.5), np.zeros(120)], axis=1)
    t = np.linspace(-np.pi / 2, np.pi / 2, 40)
    corner = np.stack([60 + 5 * np.cos(t), 5 + 5 * np.sin(t)], axis=1)
    back = np.stack([np.arange(60, 0, -0.5), np.full(120, 10.0)], axis=1)
    path = np.vstack([straight, corner, back])
    v = speed_profile(path, v_max=20.0, a_lat=4.0, a_brake=3.0, smooth_m=1.0)
    corner_speed = v[len(straight) + len(corner) // 2]
    assert corner_speed < np.sqrt(4.0 / (1 / 5)) * 1.2  # limited by lateral acceleration in a 5 m radius
    # the car slows down towards the corner instead of braking at the last moment
    assert v[len(straight) - 40] > v[len(straight) - 5] > corner_speed * 0.9
    spacing = 0.5
    for i in range(len(v) - 1):
        assert v[i] <= math.sqrt(v[i + 1] ** 2 + 2 * 3.0 * spacing) + 1e-6


def test_start_line_averages_four_orange_cones():
    xy = np.array([[0.0, 1.5], [1.0, 1.5], [0.0, -1.5], [1.0, -1.5]])
    colors = np.full(4, ConeColor.ORANGE_BIG)
    a, b = start_line(xy, colors)
    assert np.allclose(sorted([a[1], b[1]]), [-1.5, 1.5])
    assert np.isclose(a[0], 0.5) and np.isclose(b[0], 0.5)


def test_crossing_needs_both_segments_to_overlap():
    a, b = np.array([0.0, -1.0]), np.array([0.0, 1.0])
    assert crossed(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), a, b)
    assert not crossed(np.array([-2.0, 0.0]), np.array([-1.0, 0.0]), a, b)  # stops short of the line
    assert not crossed(np.array([-1.0, 5.0]), np.array([1.0, 5.0]), a, b)  # passes beyond its end
    assert not crossed(np.array([-1.0, 0.0]), np.array([-1.0, 2.0]), a, b)  # parallel to it


def test_nearest_index_and_pure_pursuit_signs():
    path = np.stack([np.arange(0.0, 20.0, 1.0), np.zeros(20)], axis=1)
    assert nearest_index(path, 4.4, 0.2) == 4
    # on the line and pointing along it, no steering is needed
    assert abs(pure_pursuit(path, (0.0, 0.0, 0.0))) < 1e-6
    # standing left of the path the car has to steer right, which is positive in FSDS
    assert pure_pursuit(path, (0.0, 1.0, 0.0)) > 0
    assert pure_pursuit(path, (0.0, -1.0, 0.0)) < 0


def test_steering_is_clipped_to_the_command_range():
    # the target sits almost beside the car, so the geometric steering angle is far beyond the maximum
    path = np.array([[0.0, 4.0], [0.1, 4.0]])
    steer = pure_pursuit(path, (0.0, 0.0, 0.0), lookahead=1.0)
    assert steer == -1.0
    wheelbase, lateral = 1.55, 4.0
    assert math.atan2(2 * wheelbase * lateral, 16.0) > MAX_STEER_RAD
