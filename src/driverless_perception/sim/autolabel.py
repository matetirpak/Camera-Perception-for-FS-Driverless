"""Labels for simulator camera images, from the ground truth cones and a depth image rendered with each image.

Projecting a cone with the camera pose only gives a rough region: FSDS turns every cone by a random angle and its
small cones are a bit shorter than the rules say. So every pixel of that region is turned back into a 3D point with
the depth image, and the pixels whose point lies inside the cone's volume make up the cone. Their bounding box is
tight around what is really visible. A cone hidden behind another cone, the car or scenery loses its pixels, and
with them its label or part of its box.

Frames and the camera model are in ``common.cameras``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..common.cameras import Intrinsics, optical_to_world, world_to_optical
from ..common.cone_model import silhouette_keypoints
from ..common.types import ConeColor


@dataclass(frozen=True)
class ConeShape:
    """Size of a cone by the rules. Used for the search region and for the area a fully visible cone covers."""

    base_width: float
    height: float


SMALL_CONE = ConeShape(0.228, 0.325)
BIG_CONE = ConeShape(0.285, 0.505)


def cone_shape(color: int) -> ConeShape:
    return BIG_CONE if color == ConeColor.ORANGE_BIG else SMALL_CONE


@dataclass
class ConeLabel:
    """One cone found in an image.

    ``box`` is (x1, y1, x2, y2) in pixels with x2 and y2 exclusive. ``distance`` is from the camera to the middle of
    the cone in metres. ``visible`` is the share of the area a fully visible cone of rule size would cover that is
    really visible, a bit below 1 for a free small cone because FSDS makes them shorter. ``contrast`` is the colour
    difference (0 to 441) between the cone's pixels and the ground right around it in the camera image. The depth
    image ignores fog, so this is what tells a cone lost in fog apart from one you can see. ``truncated`` means cut
    by the image border or partly hidden.
    """

    index: int
    color: int
    box: tuple[int, int, int, int]
    distance: float
    visible: float
    contrast: float
    truncated: bool


@dataclass(frozen=True)
class LabelPolicy:
    """Which of the cones found in an image become labels.

    The contrast threshold was chosen by looking at fog images: below about 12 cones are practically invisible, up
    to about 16 they are faint smudges. At 15 the fog map CompetitionMap2 loses 21% of its boxes, nearly all beyond
    40 m, the dusk map CompetitionMap3 loses 2%.
    """

    min_visible: float = 0.3
    min_height: int = 4
    min_contrast: float = 15.0

    def accepts(self, label: ConeLabel) -> bool:
        return (label.visible >= self.min_visible and label.box[3] - label.box[1] >= self.min_height
                and not label.contrast < self.min_contrast)


def _ring(radius: float, z: float, n: int = 24) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, z)], axis=1)


def label_image(
    depth: np.ndarray,
    position,
    orientation,
    intrinsics: Intrinsics,
    cones_xy: np.ndarray,
    colors: np.ndarray,
    ground_z: float,
    scene: np.ndarray | None = None,
    max_range: float = 80.0,
    min_visible: float = 0.05,
    min_height: int = 2,
    plate_height: float = 0.004,
    truncated_below: float = 0.7,
) -> list[ConeLabel]:
    """Every cone that shows in one image, from its depth image and the pose that came with it.

    ``ground_z`` is the ground height near the car (the car's z). Around every cone the ground height is measured
    again from the depth image, and pixels count as cone from ``plate_height`` above that, which keeps the thin base
    plate in the box without letting tarmac in. ``scene`` is the camera image, needed for ``contrast`` (NaN
    without it). The thresholds here only drop hopeless cases, ``LabelPolicy`` decides what becomes a label.
    """
    h, w = depth.shape
    labels = []
    for i, ((x, y), color) in enumerate(zip(cones_xy, colors)):
        shape = cone_shape(int(color))
        base = np.array([x, y, ground_z])
        centre = world_to_optical(base + [0.0, 0.0, shape.height / 2], position, orientation)[0]
        distance = float(np.linalg.norm(centre))
        if centre[2] < 0.3 or distance > max_range:
            continue

        # r_out covers the corners of the square base. The search region reaches half a metre further, so it also
        # holds the ground around the cone, for the local ground height and the contrast.
        r_out = shape.base_width / 2 * math.sqrt(2) + 0.05
        cylinder = np.vstack([_ring(r_out + 0.5, -0.05), _ring(r_out + 0.5, shape.height + 0.05)]) + base
        cyl_opt = world_to_optical(cylinder, position, orientation)
        if cyl_opt[:, 2].min() < 0.1:
            continue
        uv = intrinsics.project(cyl_opt)
        u0, v0 = np.maximum(np.floor(uv.min(axis=0)).astype(int), 0)
        u1, v1 = np.minimum(np.ceil(uv.max(axis=0)).astype(int), [w - 1, h - 1])
        if u0 >= u1 or v0 >= v1:
            continue

        uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        rays = np.stack([(uu + 0.5 - intrinsics.cx) / intrinsics.f, (vv + 0.5 - intrinsics.cy) / intrinsics.f,
                         np.ones(uu.shape)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        points = optical_to_world((rays * depth[v0:v1 + 1, u0:u1 + 1, None]).reshape(-1, 3), position, orientation)
        radial = np.hypot(points[:, 0] - x, points[:, 1] - y)
        dz = points[:, 2] - ground_z
        ground = (radial > r_out) & (radial < r_out + 0.5) & (np.abs(dz) < 0.1)
        if ground.sum() >= 20:
            dz -= np.median(dz[ground])
        radius = r_out * (1 - 0.8 * np.clip(dz / shape.height, 0, 1)) + 0.02
        inside = (dz > plate_height) & (dz < shape.height + 0.03) & (radial < radius)
        if not inside.any():
            continue
        mask = inside.reshape(uu.shape)
        ys, xs = np.nonzero(mask)
        box = (int(u0 + xs.min()), int(v0 + ys.min()), int(u0 + xs.max() + 1), int(v0 + ys.max() + 1))

        # area a fully visible cone of rule size would cover, border included
        model = np.vstack([_ring(shape.base_width / 2, 0.0), [[0.0, 0.0, shape.height]]]) + base
        model_uv = intrinsics.project(world_to_optical(model, position, orientation)).astype(np.float32)
        expected = cv2.contourArea(cv2.convexHull(model_uv))
        if expected < 1:
            continue
        visible = float(mask.sum() / expected)
        if box[3] - box[1] < min_height or visible < min_visible:
            continue
        contrast = float("nan")
        if scene is not None:
            patch = scene[v0:v1 + 1, u0:u1 + 1].reshape(-1, 3).astype(np.float32)
            # what the cone stands against in the image: ground, fog or sky, but nothing nearer than the cone
            behind = ~inside & (depth[v0:v1 + 1, u0:u1 + 1].reshape(-1) > distance - 1.0)
            if behind.sum() >= 5:
                contrast = float(np.linalg.norm(np.median(patch[inside], axis=0) - np.median(patch[behind], axis=0)))
        cut = bool(model_uv[:, 0].min() < 0 or model_uv[:, 1].min() < 0
                   or model_uv[:, 0].max() > w or model_uv[:, 1].max() > h)
        labels.append(ConeLabel(i, int(color), box, distance, visible, contrast, cut or visible < truncated_below))
    return labels


def project_keypoints(cone_xy, ground_z: float, position, orientation, intrinsics: Intrinsics) -> np.ndarray:
    """The 7 keypoints of a small cone in the image, (7, 2) pixels (see ``common.cone_model``)."""
    points = silhouette_keypoints(cone_xy, ground_z, position)
    return intrinsics.project(world_to_optical(points, position, orientation))


def yolo_lines(labels: list[ConeLabel], width: int, height: int) -> list[str]:
    lines = []
    for lab in labels:
        x1, y1, x2, y2 = lab.box
        lines.append(f"{lab.color} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines
