import numpy as np

from driverless_perception.common.types import ConeColor
from driverless_perception.sim.csv_map import read_csv_map, write_csv_map
from driverless_perception.sim.ground_truth import (
    centerline,
    crossed,
    curvature,
    pure_pursuit,
    speed_profile,
    start_line,
)


def test_droneport_csv(droneport_csv, tmp_path):
    xy, col = read_csv_map(droneport_csv)
    assert (col == ConeColor.BLUE).sum() == 67
    assert (col == ConeColor.YELLOW).sum() == 63
    assert (col == ConeColor.ORANGE_BIG).sum() == 2
    out = tmp_path / "copy.csv"
    write_csv_map(out, xy, col)
    xy2, col2 = read_csv_map(out)
    assert np.allclose(xy, xy2, atol=1e-6)
    assert (col == col2).all()


def test_centerline_goes_forward_from_the_start(droneport_csv):
    xy, col = read_csv_map(droneport_csv)
    path = centerline(xy, col, spacing=0.5)
    assert np.linalg.norm(path[0]) < 10
    assert path[1, 0] > path[0, 0]
    gaps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.allclose(gaps, gaps.mean(), rtol=0.05)


def test_start_line_spans_the_track(droneport_csv):
    xy, col = read_csv_map(droneport_csv)
    a, b = start_line(xy, col)
    assert 2.5 < np.linalg.norm(a - b) < 5.0
    assert crossed(np.array([0.0, 0.0]), np.array([6.0, 0.0]), a, b)
    assert not crossed(np.array([-3.0, 0.0]), np.array([-1.0, 0.0]), a, b)


def test_curvature_and_speed_profile_on_a_circle():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.stack([10 * np.cos(t), 10 * np.sin(t)], axis=1)
    assert np.allclose(curvature(circle), 0.1, rtol=1e-3)
    v = speed_profile(circle, v_max=20.0, a_lat=4.0)
    assert np.allclose(v, np.sqrt(4.0 / 0.1), rtol=1e-3)


def test_pure_pursuit_steers_towards_the_target():
    # the lookahead target (10, 5) is to the left, and FSDS steers left with negative values
    steer = pure_pursuit(np.array([[3.0, 0.0], [10.0, 5.0]]), (0.0, 0.0, 0.0))
    assert -1.0 <= steer < 0.0
