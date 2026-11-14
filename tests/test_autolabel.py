import numpy as np
import pytest

from driverless_perception.common.types import ConeColor
from driverless_perception.sim.autolabel import (
    Intrinsics,
    LabelPolicy,
    label_image,
    optical_to_world,
    world_to_optical,
    yolo_lines,
)

INTR = Intrinsics(640, 480, 60.0)
GROUND = 0.25
CAMERA = np.array([0.0, 0.0, GROUND + 1.0])
LEVEL = np.array([1.0, 0.0, 0.0, 0.0])


def render_scene(cylinders, fog=0.0):
    """Depth and a camera image: grey ground, blue cylinders, blended towards the ground colour by ``fog``."""
    depth = render_depth(cylinders)
    ground = render_depth([])
    image = np.full(depth.shape + (3,), 110, np.float32)
    image[depth < ground - 1e-3] = (1 - fog) * np.array([200, 70, 20]) + fog * 110
    return depth, image.astype(np.uint8)


def render_depth(cylinders, position=CAMERA, orientation=LEVEL):
    """Depth along each pixel's ray for flat ground and vertical cylinders (x, y, radius, bottom, top above ground)."""
    uu, vv = np.meshgrid(np.arange(INTR.width), np.arange(INTR.height))
    rays = np.stack([(uu + 0.5 - INTR.cx) / INTR.f, (vv + 0.5 - INTR.cy) / INTR.f, np.ones(uu.shape)], -1)
    rays = (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).reshape(-1, 3)
    d = optical_to_world(rays, np.zeros(3), orientation)
    p = np.asarray(position, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        best = np.where(d[:, 2] < 0, (GROUND - p[2]) / d[:, 2], np.inf)
        for cx, cy, r, bottom, top in cylinders:
            ox, oy = p[0] - cx, p[1] - cy
            a = d[:, 0] ** 2 + d[:, 1] ** 2
            b = 2 * (ox * d[:, 0] + oy * d[:, 1])
            c = ox * ox + oy * oy - r * r
            t_side = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
            z = p[2] + t_side * d[:, 2] - GROUND
            t_side = np.where((t_side > 0) & (z >= bottom) & (z <= top), t_side, np.inf)
            t_top = (GROUND + top - p[2]) / d[:, 2]
            hx, hy = p[0] + t_top * d[:, 0] - cx, p[1] + t_top * d[:, 1] - cy
            t_top = np.where((t_top > 0) & (hx * hx + hy * hy <= r * r), t_top, np.inf)
            best = np.fmin(best, np.fmin(np.nan_to_num(t_side, nan=np.inf), t_top))
    # FSDS reports the sky as 16384 m
    return np.minimum(best, 16384.0).reshape(INTR.height, INTR.width).astype(np.float32)


def expected_box(x, y, r, top):
    a = np.linspace(0, 2 * np.pi, 360)
    ring = np.stack([x + r * np.cos(a), y + r * np.sin(a), np.zeros_like(a)], axis=1)
    pts = np.vstack([ring + [0, 0, GROUND], ring + [0, 0, GROUND + top]])
    uv = INTR.project(world_to_optical(pts, CAMERA, LEVEL))
    return uv.min(axis=0), uv.max(axis=0)


def test_pose_round_trip_and_optical_axes():
    q = np.array([np.cos(0.2), 0.05, -0.03, np.sin(0.2)])
    q /= np.linalg.norm(q)
    pts = np.random.default_rng(0).normal(size=(10, 3)) * 5
    assert np.allclose(optical_to_world(world_to_optical(pts, CAMERA, q), CAMERA, q), pts)
    # a point straight ahead of a level camera is on the optical axis, one to the left has negative x
    assert np.allclose(world_to_optical(CAMERA + [5, 0, 0], CAMERA, LEVEL), [[0, 0, 5]])
    assert world_to_optical(CAMERA + [5, 1, 0], CAMERA, LEVEL)[0, 0] < 0


def test_box_matches_a_fully_visible_cone():
    depth = render_depth([(8.0, 0.5, 0.06, 0.0, 0.3)])
    labels = label_image(depth, CAMERA, LEVEL, INTR, np.array([[8.0, 0.5]]), np.array([ConeColor.BLUE]), GROUND)
    assert len(labels) == 1
    lo, hi = expected_box(8.0, 0.5, 0.06, 0.3)
    x1, y1, x2, y2 = labels[0].box
    assert abs(x1 - lo[0]) <= 1.5 and abs(x2 - hi[0]) <= 1.5
    assert abs(y1 - lo[1]) <= 1.5 and abs(y2 - hi[1]) <= 1.5
    assert labels[0].distance == pytest.approx(np.linalg.norm([8.0, 0.5, 0.325 / 2 - 1.0]), rel=1e-6)
    assert not labels[0].truncated


def test_base_plate_is_part_of_the_box():
    plate_and_body = [(8.0, 0.0, 0.16, 0.0, 0.012), (8.0, 0.0, 0.06, 0.0, 0.3)]
    labels = label_image(render_depth(plate_and_body), CAMERA, LEVEL, INTR, np.array([[8.0, 0.0]]),
                         np.array([ConeColor.YELLOW]), GROUND)
    lo, hi = expected_box(8.0, 0.0, 0.16, 0.012)
    x1, _, x2, _ = labels[0].box
    assert abs(x1 - lo[0]) <= 1.5 and abs(x2 - hi[0]) <= 1.5


def test_hidden_cones_lose_their_label_or_part_of_it():
    cones = np.array([[8.0, 0.0], [12.0, 0.0], [12.0, 3.0]])
    colors = np.array([ConeColor.BLUE, ConeColor.YELLOW, ConeColor.BLUE])
    free = label_image(render_depth([(12.0, 0.0, 0.06, 0.0, 0.3), (12.0, 3.0, 0.06, 0.0, 0.3)]),
                       CAMERA, LEVEL, INTR, cones[1:], colors[1:], GROUND)
    # a wide post right in front of the cone at 12 m hides it completely
    hidden = label_image(render_depth([(8.0, 0.0, 0.3, 0.0, 1.5), (12.0, 0.0, 0.06, 0.0, 0.3),
                                       (12.0, 3.0, 0.06, 0.0, 0.3)]), CAMERA, LEVEL, INTR, cones, colors, GROUND)
    assert sorted(lab.index for lab in free) == [0, 1]
    assert 1 not in [lab.index for lab in hidden] and 2 in [lab.index for lab in hidden]
    # a thin post covers the left half of it
    half = label_image(render_depth([(9.0, 0.05, 0.03, 0.0, 1.5), (12.0, 0.0, 0.06, 0.0, 0.3)]),
                       CAMERA, LEVEL, INTR, cones[1:2], colors[1:2], GROUND)
    assert half and half[0].visible < free[0].visible


def test_cones_out_of_view_are_skipped():
    cones = np.array([[-5.0, 0.0], [8.0, 20.0], [60.0, 0.0]])
    colors = np.array([ConeColor.BLUE] * 3)
    assert label_image(render_depth([]), CAMERA, LEVEL, INTR, cones, colors, GROUND) == []


def test_contrast_tells_cones_lost_in_fog():
    cone = [(8.0, 0.0, 0.06, 0.0, 0.3)]
    xy, colors = np.array([[8.0, 0.0]]), np.array([ConeColor.BLUE])
    depth, image = render_scene(cone)
    clear = label_image(depth, CAMERA, LEVEL, INTR, xy, colors, GROUND, scene=image)
    _, foggy_image = render_scene(cone, fog=0.95)
    foggy = label_image(depth, CAMERA, LEVEL, INTR, xy, colors, GROUND, scene=foggy_image)
    assert clear[0].contrast > 100 and foggy[0].contrast < 10
    policy = LabelPolicy(min_contrast=20)
    assert policy.accepts(clear[0]) and not policy.accepts(foggy[0])
    # without an image the contrast is unknown and doesn't stop a label
    assert policy.accepts(label_image(depth, CAMERA, LEVEL, INTR, xy, colors, GROUND)[0])


def test_policy_filters_hidden_and_tiny_cones():
    depth = render_depth([(8.0, 0.0, 0.06, 0.0, 0.3)])
    lab = label_image(depth, CAMERA, LEVEL, INTR, np.array([[8.0, 0.0]]), np.array([ConeColor.BLUE]), GROUND)[0]
    assert LabelPolicy().accepts(lab)
    assert not LabelPolicy(min_visible=1.5).accepts(lab)
    assert not LabelPolicy(min_height=1000).accepts(lab)


def test_yolo_lines():
    depth = render_depth([(8.0, 0.0, 0.06, 0.0, 0.3)])
    labels = label_image(depth, CAMERA, LEVEL, INTR, np.array([[8.0, 0.0]]), np.array([ConeColor.ORANGE_BIG]),
                         GROUND)
    cls, cx, _, bw, bh = yolo_lines(labels, INTR.width, INTR.height)[0].split()
    assert int(cls) == ConeColor.ORANGE_BIG
    assert abs(float(cx) - 0.5) < 0.01 and 0 < float(bw) < float(bh) < 1
