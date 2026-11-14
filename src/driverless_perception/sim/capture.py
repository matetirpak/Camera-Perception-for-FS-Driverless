"""Capturing labelled camera images from a running simulator.

A thread drives the car along the ground truth centre line on its own connection, while the main loop grabs
camera and depth images at a fixed rate and labels them with ``autolabel``. Every lap can have its own speed and
lateral offset from the centre line, so the cones are seen from different places. The capture stops as soon as a
cone gets hit, because the referee only knows where the cones stood at the start.

Only cameras with a DepthPerspective entry in their ``CaptureSettings`` are captured (see configs/fsds/gotthard.json).
"""
from __future__ import annotations

import csv
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..common.cameras import Intrinsics
from .autolabel import LabelPolicy, label_image, yolo_lines
from .fsds_client import DEPTH_PERSPECTIVE, SCENE, FsdsClient, RpcError, referee_cones
from .ground_truth import centerline, nearest_index, pure_pursuit, speed_profile


@dataclass
class CaptureConfig:
    laps: int = 3
    rate_hz: float = 2.0
    # one entry per lap, repeated if there are more laps
    speeds: list[float] = field(default_factory=lambda: [5.0, 6.0, 7.0])
    offsets: list[float] = field(default_factory=lambda: [0.0, 0.4, -0.4])
    min_speed: float = 1.0
    max_range: float = 80.0
    jpeg_quality: int = 95
    time_limit_s: float = 900.0
    # boxes.csv keeps every cone found, the label files only what the policy accepts
    policy: LabelPolicy = field(default_factory=LabelPolicy)


def _yaw(q: dict) -> float:
    return math.atan2(2 * (q["w_val"] * q["z_val"] + q["x_val"] * q["y_val"]), 1 - 2 * (q["y_val"] ** 2 + q["z_val"] ** 2))


def offset_path(path: np.ndarray, offset: float) -> np.ndarray:
    """The closed path moved sideways, positive to the left."""
    tangent = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return path + offset * np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


class Driver(threading.Thread):
    """Follows the centre line at 20 Hz on its own connection and keeps track of the distance driven."""

    def __init__(self, host: str, path: np.ndarray):
        super().__init__(daemon=True)
        self.host = host
        self.centre = path
        self.travelled = 0.0
        self.speed = 0.0
        self._target = (5.0, 0.0)
        self._halt = threading.Event()
        self.error: Exception | None = None

    def set_lap(self, speed: float, offset: float) -> None:
        self._target = (speed, offset)

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        try:
            self._drive()
        except (OSError, RpcError) as e:
            # the capture loop checks this and stops
            self.error = e

    def _drive(self) -> None:
        client = FsdsClient(self.host, timeout=10)
        client.enable_api_control(True)
        current, path, v_target, last = None, None, None, None
        while not self._halt.is_set():
            t0 = time.time()
            if self._target != current:
                current = self._target
                path = offset_path(self.centre, current[1])
                v_target = speed_profile(path, current[0], 4.0)
            state = client.car_state()
            k = state["kinematics_estimated"]
            x, y = k["position"]["x_val"], k["position"]["y_val"]
            v = self.speed = state["speed"]
            if last is not None:
                self.travelled += math.hypot(x - last[0], y - last[1])
            last = (x, y)
            steering = pure_pursuit(path, (x, y, _yaw(k["orientation"])), lookahead=max(3.0, 0.4 * v))
            error = float(v_target[nearest_index(path, x, y)]) - v
            client.set_controls(float(np.clip(0.25 * error, 0.0, 0.6)), steering,
                                float(np.clip(-0.3 * error, 0.0, 1.0)) if error < -0.5 else 0.0)
            time.sleep(max(0.0, 0.05 - (time.time() - t0)))
        client.set_controls(0.0, 0.0, 1.0)
        client.close()


def write_cones(path: Path, xy: np.ndarray, colors: np.ndarray) -> None:
    """The ground truth cones of the map, their row number is the ``cone`` column of boxes.csv."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cone", "x", "y", "class"])
        writer.writerows([i, round(float(x), 4), round(float(y), 4), int(c)] for i, ((x, y), c) in enumerate(zip(xy, colors)))


def capture_static(out: Path, name: str, map_name: str, frames: int = 100, rate_hz: float = 5.0,
                   jpeg_quality: int = 95, host: str = "127.0.0.1") -> dict:
    """Takes ``frames`` pictures with every camera while the car stands still, for the depth test of Gosala et al.
    (Fig. 9a): the cones sit at known distances and each frame is another measurement of the same cones. Writes
    ``images/``, ``frames.csv``, ``cones.csv`` and ``run.yaml``, without labels."""
    client = FsdsClient(host, timeout=60)
    cameras = client.settings()["Vehicles"]["FSCar"]["Cameras"]
    intrinsics = {n: Intrinsics(c["CaptureSettings"][0]["Width"], c["CaptureSettings"][0]["Height"],
                                c["CaptureSettings"][0]["FOV_Degrees"]) for n, c in cameras.items()}
    xy, colors = referee_cones(client.referee_state())
    (out / "images").mkdir(parents=True, exist_ok=True)
    write_cones(out / "cones.csv", xy, colors)
    rows = []
    start = time.time()
    try:
        for frame in range(frames):
            tick = time.time()
            for scene in client.images([(n, SCENE) for n in cameras]):
                stem = f"{name}_{frame:05d}_{scene.camera}"
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), scene.data, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                rows.append([stem, scene.camera, 1, scene.stamp, 0.0, *np.round(scene.position, 4),
                             *np.round(scene.orientation, 6)])
            time.sleep(max(0.0, 1.0 / rate_hz - (time.time() - tick)))
    finally:
        client.close()

    with open(out / "frames.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "camera", "lap", "stamp", "speed", "x", "y", "z", "qw", "qx", "qy", "qz"])
        writer.writerows(rows)
    summary = {
        "name": name,
        "map": map_name,
        "standing": True,
        "cameras": {n: asdict(i) for n, i in intrinsics.items()},
        "mounts": {n: {k: c[k] for k in ("X", "Y", "Z", "Roll", "Pitch", "Yaw")} for n, c in cameras.items()},
        "cones": len(xy),
        "frames": frames,
        "images": len(rows),
        "duration_s": round(time.time() - start, 1),
    }
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


def capture(out: Path, name: str, map_name: str, config: CaptureConfig, host: str = "127.0.0.1") -> dict:
    """Drives ``config.laps`` laps and writes ``images/``, ``labels/``, ``boxes.csv``, ``frames.csv`` and
    ``run.yaml`` to ``out``. Returns the summary that also goes into ``run.yaml``."""
    client = FsdsClient(host, timeout=60)
    cameras = client.settings()["Vehicles"]["FSCar"]["Cameras"]
    cameras = {n: c for n, c in cameras.items() if any(s["ImageType"] == DEPTH_PERSPECTIVE for s in c["CaptureSettings"])}
    if not cameras:
        raise ValueError("no camera has a DepthPerspective entry in its CaptureSettings")
    intrinsics = {}
    for n, c in cameras.items():
        s = c["CaptureSettings"][0]
        intrinsics[n] = Intrinsics(s["Width"], s["Height"], s["FOV_Degrees"])

    referee = client.referee_state()
    xy, colors = referee_cones(referee)
    doo = referee["doo_counter"]
    path = centerline(xy, colors)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    write_cones(out / "cones.csv", xy, colors)
    frames, boxes = [], []
    stop_reason = "done"
    driver = Driver(host, path)
    driver.start()
    start, frame = time.time(), 0
    requests = [(n, kind) for n in cameras for kind in (SCENE, DEPTH_PERSPECTIVE)]
    try:
        while True:
            tick = time.time()
            lap = int(driver.travelled // length)
            if lap >= config.laps:
                break
            if driver.error is not None:
                raise driver.error
            if not driver.is_alive():
                raise RuntimeError("the driving thread stopped, see the traceback above")
            if tick - start > config.time_limit_s:
                stop_reason = "time limit"
                break
            driver.set_lap(config.speeds[lap % len(config.speeds)], config.offsets[lap % len(config.offsets)])
            if driver.speed < config.min_speed:
                time.sleep(0.1)
                continue
            images = client.images(requests)
            if client.referee_state()["doo_counter"] != doo:
                stop_reason = f"cone hit in lap {lap + 1}, frame {frame} dropped"
                break
            for scene, depth in zip(images[0::2], images[1::2]):
                cam = cameras[scene.camera]
                # the car's origin is on the ground, so the ground is the mount height below the camera
                ground_z = float(depth.position[2] - cam["Z"])
                found = label_image(depth.data, depth.position, depth.orientation, intrinsics[scene.camera], xy,
                                    colors, ground_z, scene.data, config.max_range)
                accepted = [config.policy.accepts(lab) for lab in found]
                labels = [lab for lab, ok in zip(found, accepted) if ok]
                stem = f"{name}_{frame:05d}_{scene.camera}"
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), scene.data,
                            [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
                h, w = scene.data.shape[:2]
                (out / "labels" / f"{stem}.txt").write_text("".join(line + "\n" for line in yolo_lines(labels, w, h)))
                frames.append([stem, scene.camera, lap + 1, scene.stamp, round(driver.speed, 3),
                               *np.round(scene.position, 4), *np.round(scene.orientation, 6)])
                for lab, ok in zip(found, accepted):
                    boxes.append([stem, lab.color, *lab.box, round(lab.distance, 3), round(lab.visible, 3),
                                  round(lab.contrast, 1), int(lab.truncated), lab.index, int(ok)])
            frame += 1
            time.sleep(max(0.0, 1.0 / config.rate_hz - (time.time() - tick)))
    finally:
        driver.stop()
        driver.join()
        client.close()

    with open(out / "frames.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "camera", "lap", "stamp", "speed", "x", "y", "z", "qw", "qx", "qy", "qz"])
        writer.writerows(frames)
    with open(out / "boxes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "x1", "y1", "x2", "y2", "distance", "visible", "contrast", "truncated",
                         "cone", "labelled"])
        writer.writerows(boxes)
    summary = {
        "name": name,
        "map": map_name,
        "cameras": {n: asdict(i) for n, i in intrinsics.items()},
        "mounts": {n: {k: c[k] for k in ("X", "Y", "Z", "Roll", "Pitch", "Yaw")} for n, c in cameras.items()},
        "config": asdict(config),
        "track_length_m": round(length, 1),
        "cones": len(xy),
        "frames": frame,
        "images": len(frames),
        "cones_found": len(boxes),
        "labels": sum(row[-1] for row in boxes),
        "laps_driven": round(driver.travelled / length, 2),
        "duration_s": round(time.time() - start, 1),
        "stop_reason": stop_reason,
    }
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary
