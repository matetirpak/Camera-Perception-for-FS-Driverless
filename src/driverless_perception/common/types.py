"""Core data types shared by all pipelines.

A pipeline turns one sensor frame into a :class:`ConeArray` in the car frame (``fsds/FSCar``, ENU: x forward,
y left, z up, origin at ground level below the car centre) stamped with the *sensor* time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class ConeColor(IntEnum):
    """Our colour classes. Values match ``fs_msgs/Cone`` (and ``driverless_perception_msgs/Cone``)."""

    BLUE = 0
    YELLOW = 1
    ORANGE_BIG = 2
    ORANGE_SMALL = 3
    UNKNOWN = 4

    @classmethod
    def from_fsds_rpc(cls, value: int) -> ConeColor:
        """The FSDS RPC / python referee uses a different order: 0 yellow, 1 blue, 2 orange big, 3 orange small."""
        return _RPC_TO_COLOR[int(value)]

    @classmethod
    def from_fsoco(cls, class_title: str) -> ConeColor:
        return _FSOCO_TO_COLOR[class_title]


_RPC_TO_COLOR = {
    0: ConeColor.YELLOW,
    1: ConeColor.BLUE,
    2: ConeColor.ORANGE_BIG,
    3: ConeColor.ORANGE_SMALL,
    4: ConeColor.UNKNOWN,
}

_FSOCO_TO_COLOR = {
    "blue_cone": ConeColor.BLUE,
    "yellow_cone": ConeColor.YELLOW,
    "orange_cone": ConeColor.ORANGE_SMALL,
    "large_orange_cone": ConeColor.ORANGE_BIG,
    "unknown_cone": ConeColor.UNKNOWN,
}

# class names of the detection datasets, their class ids are the ConeColor values
CLASS_NAMES = {c.value: c.name.lower() for c in ConeColor}
# how each class is drawn, as red, green, blue in 0 to 1
CONE_RGB = {ConeColor.BLUE: (0.12, 0.37, 0.84), ConeColor.YELLOW: (0.96, 0.78, 0.16),
            ConeColor.ORANGE_BIG: (0.97, 0.43, 0.0), ConeColor.ORANGE_SMALL: (1.0, 0.65, 0.0),
            ConeColor.UNKNOWN: (0.67, 0.67, 0.67)}


class Source(IntEnum):
    """Which pipeline produced an observation. Values match ``driverless_perception_msgs/ConeArray``."""

    LIDAR = 0
    MONO = 1
    STEREO = 2
    GROUND_TRUTH = 3


@dataclass
class Cone:
    """One cone observation in the car frame.

    ``color_probs`` are probabilities for [blue, yellow, orange (big or small), unknown], the four outcomes the
    paper's SLAM consumes (Gosala et al., Sec. III-D). ``covariance`` is the 2x2 xy position covariance.
    """

    x: float
    y: float
    color_probs: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.1)

    @property
    def color(self) -> ConeColor:
        i = int(np.argmax(self.color_probs))
        return (ConeColor.BLUE, ConeColor.YELLOW, ConeColor.ORANGE_BIG, ConeColor.UNKNOWN)[i]

    @property
    def range(self) -> float:
        return float(np.hypot(self.x, self.y))


@dataclass
class ConeArray:
    """All cones from one sensor frame. ``stamp_ns`` is the sensor timestamp (unix ns, FSDS sim clock)."""

    stamp_ns: int
    source: Source
    cones: list[Cone] = field(default_factory=list)
    frame_id: str = "fsds/FSCar"

    def xy(self) -> np.ndarray:
        """(N, 2) array of cone positions."""
        return np.array([[c.x, c.y] for c in self.cones]).reshape(-1, 2)
