import numpy as np

from driverless_perception.common.geometry import world_to_car
from driverless_perception.common.types import ConeColor
from driverless_perception.evaluation.race import (
    CameraView,
    camera_view,
    expected_cones,
    report_markdown,
    score_frames,
)

CAMERA = CameraView(-0.3, 0.0, np.radians(54.5))
TRACK = np.array([[5.0, 1.5], [5.0, -1.5], [10.0, 1.5], [10.0, -1.5], [25.0, 0.0], [5.0, 8.0], [-5.0, 0.0],
                  [8.0, 0.5]])
COLORS = np.array([ConeColor.BLUE, ConeColor.YELLOW, ConeColor.BLUE, ConeColor.YELLOW, ConeColor.BLUE,
                   ConeColor.BLUE, ConeColor.BLUE, ConeColor.ORANGE_BIG])


def test_expected_cones_are_in_view_close_and_small():
    got = expected_cones(TRACK, COLORS, (0.0, 0.0, 0.0), CAMERA, max_range=20)
    # too far, out of the field of view, behind the car and the big orange one are left out
    assert sorted(map(tuple, got)) == [(5.0, -1.5), (5.0, 1.5), (10.0, -1.5), (10.0, 1.5)]
    # turned by 90 degrees to the left, the car looks at the cone 8 m to the left
    turned = expected_cones(TRACK, COLORS, (0.0, 0.0, np.pi / 2), CAMERA, max_range=20)
    assert np.allclose(turned, [[8.0, -5.0]])


def test_stereo_expects_cones_both_cameras_see():
    left, right = CameraView(-0.3, 0.16, np.radians(64.5)), CameraView(-0.3, -0.16, np.radians(64.5))
    track = np.array([[1.0, 0.75], [1.0, 0.5], [6.0, 1.0], [9.0, 0.0], [12.0, 0.0]])
    colors = np.full(len(track), ConeColor.BLUE)
    # the first cone is only in the left image, the last one too far for stereo
    got = expected_cones(track, colors, (0.0, 0.0, 0.0), (left, right), max_range=10)
    assert sorted(map(tuple, got)) == [(1.0, 0.5), (6.0, 1.0), (9.0, 0.0)]
    assert len(expected_cones(track, colors, (0.0, 0.0, 0.0), left, max_range=10)) == 4


def test_score_counts_hits_misses_and_false_cones():
    times = np.array([0, 1_000_000_000])
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    truth = world_to_car(TRACK[:4], (0.5, 0.0, 0.0))
    # at t = 0.5 s the car is at x = 0.5: two cones found (one 0.3 m off), one false cone, two missed
    detected = np.array([truth[0], truth[2] + [0.3, 0.0], [3.0, 0.0]])
    result = score_frames([(500_000_000, detected), (2_000_000_000, detected)], times, poses, TRACK, COLORS, CAMERA)
    assert result["frames"] == 1
    assert result["all"]["expected"] == 4 and result["all"]["detections"] == 3
    assert result["all"]["recall"] == 0.5 and result["all"]["precision"] == 2 / 3
    near = result["bins"][0]
    assert near["n_gt"] == 2 and near["recall"] == 0.5
    # finding the big orange cone or the one out of view isn't expected, but isn't a false cone either
    extra = world_to_car(TRACK[[5, 7]], (0.5, 0.0, 0.0))
    with_extra = score_frames([(500_000_000, np.vstack([detected, extra]))], times, poses, TRACK, COLORS, CAMERA)
    assert with_extra["all"]["precision"] == 2 / 3 and with_extra["all"]["recall"] == 0.5


def test_report_has_ranges_and_laps():
    times = np.array([0, 1_000_000_000])
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = score_frames([(500_000_000, world_to_car(TRACK[:4], (0.5, 0.0, 0.0)))], times, poses, TRACK, COLORS,
                          CAMERA)
    result.update(latency_ms={"median": 31.0, "p95": 70.0}, rate_hz=14.7,
                  laps=[{"lap": 1, "frames": 1, "recall": 1.0, "precision": 1.0}])
    text = report_markdown(result, "Mono pipeline in a race")
    assert "| 0-5 m | 2 | 1.00 | 1.00 | 0.00 m |" in text and "| 1 | 1 | 1.00 | 1.00 |" in text
    assert "median 31 ms" in text


def test_strict_gate_in_the_report():
    # the cone 0.3 m off counts within 1 m but not within 0.2 m, and the report shows both
    times = np.array([0, 1_000_000_000])
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    truth = world_to_car(TRACK[:4], (0.5, 0.0, 0.0))
    detected = np.array([truth[0], truth[2] + [0.3, 0.0]])
    loose = score_frames([(500_000_000, detected)], times, poses, TRACK, COLORS, CAMERA, gate_m=1.0)
    strict = score_frames([(500_000_000, detected)], times, poses, TRACK, COLORS, CAMERA, gate_m=0.2)
    assert loose["all"]["recall"] == 0.5 and strict["all"]["recall"] == 0.25
    loose.update(latency_ms={"median": 31.0, "p95": 70.0}, rate_hz=14.7,
                 strict={"gate_m": 0.2, "all": strict["all"], "bins": strict["bins"]})
    text = report_markdown(loose, "Mono pipeline in a race")
    assert "## With a 0.2 m gate" in text and "Overall recall 0.25, precision 0.50" in text


def test_camera_view_from_intrinsics():
    view = camera_view(1280, (1280 / 2) / np.tan(np.radians(54.5) / 2), -0.3, 0.0)
    assert np.isclose(view.fov, np.radians(54.5)) and view.x == -0.3


def test_timing_summary_keeps_the_order_and_counts_frames():
    from driverless_perception.evaluation.race import timing_summary

    frames = [[("bridge", 2.0), ("detector", 6.0), ("detector/inference", 4.0), ("stamp to published", 30.0)],
              [("bridge", 4.0), ("detector", 8.0), ("orb match", 5.0), ("stamp to published", 50.0)]]
    t = timing_summary(frames)
    assert list(t) == ["bridge", "detector", "detector/inference", "stamp to published", "orb match"]
    assert t["bridge"]["frames"] == 2 and t["orb match"]["frames"] == 1
    assert t["detector"]["p50"] == 7.0 and t["stamp to published"]["p99"] > 49.0
