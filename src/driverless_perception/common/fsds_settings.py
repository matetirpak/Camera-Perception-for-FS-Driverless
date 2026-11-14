"""Read sensor definitions from an FSDS ``settings.json`` so the algorithms use the same sensors as the simulator.

How FSDS interprets the file (checked against v2.2.0): sensor positions are x forward, y left, z up relative to the
car's origin, which is on the ground. ``FOV_Degrees`` is the horizontal field of view and pixels are square, so
fx = fy = (width / 2) / tan(fov / 2). LiDAR lasers are spread evenly between VerticalFOVLower and VerticalFOVUpper,
and ``PointsPerScan`` counts the points of all lasers together.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SETTINGS = Path(os.path.expanduser("~/Formula-Student-Driverless-Simulator/settings.json"))
VEHICLE = "FSCar"
SENSOR_TYPE_LIDAR = 6


@dataclass(frozen=True)
class Mount:
    x: float
    y: float
    z: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int
    fov_deg: float
    mount: Mount
    image_type: int = 0  # 0 RGB scene, 2 DepthPerspective

    @property
    def fx(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.fov_deg) / 2)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2


@dataclass(frozen=True)
class LidarSpec:
    name: str
    lasers: int
    points_per_scan: int
    vfov_lower_deg: float
    vfov_upper_deg: float
    hfov_start_deg: float
    hfov_end_deg: float
    rotations_per_second: float
    range_m: float
    mount: Mount

    @property
    def vertical_resolution_rad(self) -> float:
        """r_v of Gosala et al. Eq. 1."""
        if self.lasers < 2:
            return 0.0
        return math.radians(self.vfov_upper_deg - self.vfov_lower_deg) / (self.lasers - 1)

    @property
    def horizontal_resolution_rad(self) -> float:
        """r_h of Gosala et al. Eq. 1."""
        firings_per_laser = self.points_per_scan / self.lasers
        return math.radians(self.hfov_end_deg - self.hfov_start_deg) / firings_per_laser


def _mount(d: dict) -> Mount:
    return Mount(d.get("X", 0.0), d.get("Y", 0.0), d.get("Z", 0.0),
                 d.get("Roll", 0.0), d.get("Pitch", 0.0), d.get("Yaw", 0.0))


class FsdsSettings:
    """Parsed ``settings.json``. Use :meth:`load` for a file or :meth:`from_json` for ``getSettingsString()``."""

    def __init__(self, raw: dict):
        self.raw = raw
        vehicle = raw["Vehicles"][VEHICLE]
        self.cameras: dict[str, CameraSpec] = {}
        for name, cam in vehicle.get("Cameras", {}).items():
            cap = cam["CaptureSettings"][0]
            self.cameras[name] = CameraSpec(name, int(cap["Width"]), int(cap["Height"]),
                                            float(cap.get("FOV_Degrees", 90.0)), _mount(cam),
                                            int(cap.get("ImageType", 0)))
        self.lidars: dict[str, LidarSpec] = {}
        for name, s in vehicle.get("Sensors", {}).items():
            if s.get("SensorType") != SENSOR_TYPE_LIDAR:
                continue
            self.lidars[name] = LidarSpec(
                name, int(s.get("NumberOfLasers", 16)), int(s.get("PointsPerScan", 10000)),
                float(s["VerticalFOVLower"]), float(s["VerticalFOVUpper"]),
                float(s.get("HorizontalFOVStart", 0.0)), float(s.get("HorizontalFOVEnd", 359.0)),
                float(s.get("RotationsPerSecond", 10)), float(s.get("Range", 100.0)), _mount(s))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SETTINGS) -> FsdsSettings:
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def from_json(cls, text: str) -> FsdsSettings:
        return cls(json.loads(text))
