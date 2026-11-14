"""The standing depth test of Gosala et al. (Fig. 9a): cones at known distances, measured over and over.

The car stands still on a map whose cones sit at fixed distances (`maps/depth_test.csv`), so every frame is another
measurement of the same cones. For each cone this collects the distance PnP and stereo came up with, which shows
how far each of them scatters and how that grows with distance. Three series are collected: PnP on the mono camera
(the narrow one the mono pipeline uses), PnP on the left camera and stereo, so PnP and stereo can be compared on
the same image as in the paper.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..common.cameras import Intrinsics, optical_to_world, up_in_optical
from ..sim.dataset import read_cones
from ..vision.stereo.pipeline import StereoPipeline, View

SERIES = ("mono camera PnP", "left camera PnP", "stereo")


def _view(capture: Path, run: dict, row: dict) -> View:
    camera = row["camera"]
    spec = run["cameras"][camera]
    position = np.array([float(row[k]) for k in ("x", "y", "z")])
    orientation = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
    image = cv2.imread(str(capture / "images" / f"{row['image']}.jpg"))
    return View(image, Intrinsics(spec["width"], spec["height"], spec["fov_deg"]), position, orientation,
                position[2] - run["mounts"][camera]["Z"])


def evaluate_depth(capture: Path, pipeline: StereoPipeline, mono_camera: str = "cam_mono",
                   gate: float = 1.0) -> dict:
    """Runs both pipelines on every frame of a standing capture and groups the estimated distances per cone."""
    run = yaml.safe_load((capture / "run.yaml").read_text())
    cones = read_cones(capture / "cones.csv")
    with open(capture / "cones.csv", newline="") as f:
        classes = {int(r["cone"]): int(r["class"]) for r in csv.DictReader(f)}
    frames = defaultdict(dict)
    with open(capture / "frames.csv", newline="") as f:
        for row in csv.DictReader(f):
            frames[row["image"].removesuffix("_" + row["camera"])][row["camera"]] = row

    estimates = defaultdict(lambda: defaultdict(list))
    truth = {}
    for frame in frames.values():
        views = {camera: _view(capture, run, row) for camera, row in frame.items()}
        for name, series in ((mono_camera, "mono camera PnP"), ("cam_left", "left camera PnP")):
            view = views.get(name)
            if view is None:
                continue
            boxes, _, _ = pipeline.mono.detect(view.image)
            boxes = boxes[pipeline.mono.inside(boxes, view.intrinsics.width)]
            keypoints = pipeline.mono.keypoints(view.image, boxes)
            up, height = up_in_optical(view.orientation), float(view.position[2] - view.ground_z)
            for t in pipeline.mono.locate(keypoints, view.intrinsics.matrix, up, height):
                if t is not None:
                    _add(estimates, truth, series, optical_to_world(t[None], view.position, view.orientation)[0],
                         cones, view, gate)
        if "cam_left" in views and "cam_right" in views:
            left = views["cam_left"]
            boxes, _, _ = pipeline.mono.detect(left.image)
            for e in pipeline.estimate(left, views["cam_right"], boxes):
                if e.stereo is not None:
                    _add(estimates, truth, "stereo", e.stereo, cones, left, gate)

    rows = []
    for index in sorted(truth):
        row = {"cone": index, "class": classes.get(index), "distance_m": truth[index],
               "series": {s: estimates[s][index] for s in SERIES if estimates[s][index]}}
        row["stats"] = {s: {"measurements": len(v), "median_m": float(np.median(v)), "std_m": float(np.std(v)),
                            "bias_m": float(np.median(v) - truth[index])} for s, v in row["series"].items()}
        rows.append(row)
    return {"frames": len(frames), "cones": rows}


def _add(estimates, truth, series: str, estimate, cones: np.ndarray, view: View, gate: float) -> None:
    """Files one estimate under the cone it belongs to, if it is close enough to one."""
    distances = np.hypot(*(cones - np.asarray(estimate, float)[:2]).T)
    index = int(np.argmin(distances))
    if distances[index] > gate:
        return
    truth.setdefault(index, float(np.hypot(*(cones[index] - view.position[:2]))))
    estimates[series][index].append(float(np.hypot(*(np.asarray(estimate, float)[:2] - view.position[:2]))))


def report_markdown(result: dict, capture: str) -> str:
    """The table behind the figure: every cone with the spread of its measurements."""
    fmt = lambda v: "-" if v is None else f"{v:+.2f}"
    lines = [
        "# Standing depth test",
        "",
        f"Written by `scripts/depth_test.py` from `{capture}`, {result['frames']} frames of a standing car on",
        "`maps/depth_test.csv`. Every frame measures the same cones again, so the spread of the estimates is the",
        "repeatability of each method, and the bias says how far off they are on average. This is the experiment",
        "behind Fig. 9a of Gosala et al., who found the triangulated depths to scatter much less than the PnP ones.",
        "",
        "| cone | true distance | " + " | ".join(f"{s} bias / std" for s in SERIES) + " |",
        "|---|---|" + "---|" * len(SERIES),
    ]
    for cone in result["cones"]:
        cells = []
        for series in SERIES:
            stats = cone["stats"].get(series)
            cells.append("-" if stats is None else f"{fmt(stats['bias_m'])} / {stats['std_m']:.3f} m")
        lines.append(f"| {cone['cone']} | {cone['distance_m']:.1f} m | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
