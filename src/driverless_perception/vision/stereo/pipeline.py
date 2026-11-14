"""The stereo pipeline put together: mono on the left image, box propagation, matching and triangulation.

Positions are in the map frame, because each image is triangulated with the pose it was taken from.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ...common.cameras import Intrinsics, optical_to_world, up_in_optical
from ...common.timing import no_timer
from ..mono import MonoPipeline
from .matching import match_features
from .propagation import propagate_box
from .triangulation import cone_from_points, fundamental, triangulate


def position_sigma(distance: float) -> float:
    """Standard deviation of a stereo cone position in metres. Fitted to the errors on the simulator test map
    (RMSE 0.04 m below 5 m, 0.05 m at 5 to 10 m and 0.12 m at 10 to 15 m)."""
    return 0.04 + 0.0005 * distance ** 2


@dataclass
class View:
    """A BGR camera image with the pose it was taken from. ``ground_z`` is the height of the ground under the car."""

    image: np.ndarray
    intrinsics: Intrinsics
    position: np.ndarray
    orientation: np.ndarray
    ground_z: float

    @property
    def camera(self) -> tuple:
        return self.intrinsics, self.position, self.orientation


@dataclass
class Estimate:
    """What became of one box in the left image: PnP position (x, y, z) and stereo position (x, y), either None if
    it failed, the number of matches and how many of them ended up on the cone. The rest is there to draw it: the
    keypoints in the left image, how the matching went, the search box in the right image, the matched pixels in
    both images, their triangulated points (map frame) and which of them lie on the cone."""

    mono: np.ndarray | None
    stereo: np.ndarray | None
    matches: int
    points: int
    keypoints: np.ndarray | None = None
    matching: dict | None = None
    right_box: np.ndarray | None = None
    pts_left: np.ndarray | None = None
    pts_right: np.ndarray | None = None
    on_cone: np.ndarray | None = None
    world: np.ndarray | None = None


@dataclass
class StereoCone:
    """A cone found by the pipeline: base (x, y) in the map frame, detector class and score, box in the left image."""

    position: np.ndarray
    color: int
    score: float
    box: np.ndarray


class StereoPipeline:
    """``match_margin`` is how far beyond ``max_range`` a cone's PnP position may lie and still be matched. Cones
    further out would be thrown away after triangulation anyway, and the matching is the expensive part."""

    def __init__(self, mono: MonoPipeline, max_range: float = 20.0, margin: float = 0.3, ratio: float = 0.8,
                 max_epipolar_px: float = 2.0, gate: float = 1.0, min_points: int = 3, match_margin: float = 1.5):
        self.mono, self.max_range, self.margin = mono, max_range, margin
        self.match_margin = match_margin
        self.ratio, self.max_epipolar_px, self.gate, self.min_points = ratio, max_epipolar_px, gate, min_points

    def __call__(self, left: View, right: View) -> list[StereoCone]:
        """The whole pipeline with the mono pipeline's detector, cones up to ``max_range`` from the left camera."""
        boxes, classes, scores = self.mono.detect(left.image)
        cones = []
        estimates = self.estimate(left, right, boxes, self.max_range + self.match_margin)
        for e, c, s, b in zip(estimates, classes, scores, boxes):
            if e.stereo is not None and np.hypot(*(e.stereo - left.position[:2])) <= self.max_range:
                cones.append(StereoCone(e.stereo, int(c), float(s), b))
        return cones

    def estimate(self, left: View, right: View, boxes: np.ndarray, match_range: float | None = None) -> list[Estimate]:
        """Mono and stereo position for every box in the left image. With ``match_range`` only the cones whose PnP
        position is at most that far from the left camera are matched, the others only get the PnP position."""
        timer = getattr(self.mono, "timer", no_timer)
        keypoints = self.mono.keypoints(left.image, boxes)
        with timer("pnp"):
            height = float(left.position[2] - left.ground_z)
            positions = self.mono.locate(keypoints, left.intrinsics.matrix, up_in_optical(left.orientation), height)
        with timer("grey and F"):
            gray_left = cv2.cvtColor(left.image, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right.image, cv2.COLOR_BGR2GRAY)
            F = fundamental(left.camera, right.camera)
        out = []
        for box, kp, t in zip(boxes, keypoints, positions):
            if t is None:
                out.append(Estimate(None, None, 0, 0, kp))
                continue
            if match_range is not None and np.linalg.norm(t) > match_range:
                out.append(Estimate(optical_to_world(t[None], left.position, left.orientation)[0], None, 0, 0, kp))
                continue
            with timer("propagate"):
                guess = optical_to_world(t[None], left.position, left.orientation)[0]
                right_box = propagate_box(guess, left.ground_z, right.position, right.orientation,
                                          right.intrinsics, self.margin)
            if right_box is None:
                out.append(Estimate(guess, None, 0, 0, kp))
                continue
            with timer("orb match"):
                pts_left, pts_right, matching = match_features(gray_left, box, gray_right, right_box, F,
                                                               self.ratio, self.max_epipolar_px)
            with timer("triangulate"):
                points = triangulate(pts_left, pts_right, left.camera, right.camera)
                xy, on_cone = cone_from_points(points, left.ground_z, guess[:2], left.position[:2], self.gate,
                                               self.min_points)
            out.append(Estimate(guess, xy, len(points), int(on_cone.sum()), kp, matching, right_box, pts_left,
                                pts_right, on_cone, points))
        return out
