import math

import numpy as np

from driverless_perception.evaluation.matching import match
from driverless_perception.evaluation.metrics import BinnedCounts
from driverless_perception.lidar.legality import expected_points
from driverless_perception.vision.keypoints.cross_ratio import cross_ratio


def test_cross_ratio_is_projective_invariant():
    rng = np.random.default_rng(0)
    line = np.array([[0.0, 0.0], [1.0, 0.5], [2.5, 1.25], [4.0, 2.0]])  # four collinear points
    H = np.eye(3) + rng.normal(scale=0.1, size=(3, 3))
    h = np.c_[line, np.ones(4)] @ H.T
    proj = h[:, :2] / h[:, 2:]
    assert math.isclose(cross_ratio(line), cross_ratio(proj), rel_tol=1e-9)


def test_expected_points_inverse_square():
    rv, rh = math.radians(1.0), math.radians(0.2)
    assert math.isclose(expected_points(5, rv, rh) / expected_points(10, rv, rh), 4.0, rel_tol=1e-3)


def test_matching_and_binned_metrics():
    gt = np.array([[3.0, 1.0], [8.0, -1.5], [12.0, 2.0]])
    det = np.array([[3.1, 1.0], [8.0, -1.2], [30.0, 0.0]])
    pairs = match(det, gt, gate_m=0.5)
    assert sorted((d, g) for d, g, _ in pairs) == [(0, 0), (1, 1)]
    m = BinnedCounts(edges=(0, 5, 10, 15))
    m.add_frame(det, gt)
    s = m.summary()
    assert s[0]["recall"] == 1.0 and s[1]["recall"] == 1.0 and s[2]["recall"] == 0.0
