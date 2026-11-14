"""Scoring the cones a pipeline published during a recorded race against the ground truth track.

Each cone array is compared at the car's ground truth pose at its stamp. Expected are the small cones inside the
field of view of every camera the pipeline uses and closer than ``max_range``; occlusion is not modelled, so recall
comes out slightly pessimistic. Detections are matched within ``gate_m``, and a detection of a real cone that is
not expected, at the image edge or just out of range, counts neither way.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..common.geometry import interpolate_pose, world_to_car
from ..common.types import ConeColor
from .metrics import BinnedCounts

SMALL = (ConeColor.BLUE, ConeColor.YELLOW, ConeColor.ORANGE_SMALL)


@dataclass(frozen=True)
class CameraView:
    """Where the camera sits in the car frame and how wide it sees (horizontal field of view in radians)."""

    x: float
    y: float
    fov: float


def split_cones(track_xy: np.ndarray, colors: np.ndarray, pose, cameras: CameraView | tuple[CameraView, ...],
                max_range: float, classes=SMALL) -> tuple[np.ndarray, np.ndarray]:
    """Track cones in the car frame seen from ``pose`` (x, y, yaw): the (M, 2) ones all ``cameras`` could see, and
    all others. A detection of one of the others (at the edge of an image, just out of range, a big orange cone) is
    not held against the pipeline."""
    cameras = (cameras,) if isinstance(cameras, CameraView) else tuple(cameras)
    car = world_to_car(track_xy, pose)
    keep = np.isin(colors, [int(c) for c in classes])
    for camera in cameras:
        rel = car - [camera.x, camera.y]
        keep &= (rel[:, 0] > 0) & (np.abs(np.arctan2(rel[:, 1], rel[:, 0])) <= camera.fov / 2)
    keep &= np.hypot(*(car - [cameras[0].x, cameras[0].y]).T) <= max_range
    return car[keep], car[~keep]


def expected_cones(track_xy: np.ndarray, colors: np.ndarray, pose, cameras: CameraView | tuple[CameraView, ...],
                   max_range: float, classes=SMALL) -> np.ndarray:
    """(M, 2) track cones in the car frame that the cameras could see from ``pose`` (x, y, yaw)."""
    return split_cones(track_xy, colors, pose, cameras, max_range, classes)[0]


def score_frames(frames: list[tuple[int, np.ndarray]], odom_times: np.ndarray, odom_poses: np.ndarray,
                 track_xy: np.ndarray, colors: np.ndarray, cameras: CameraView | tuple[CameraView, ...],
                 max_range: float = 20.0, gate_m: float = 1.0, edges=(0.0, 5.0, 10.0, 15.0, 20.0)) -> dict:
    """Recall, precision and position error per range bin over all ``frames`` (stamp in ns, (N, 2) cones in the
    car frame). Frames outside the odometry's time span are skipped."""
    counts = BinnedCounts(edges=tuple(edges))
    used = 0
    for stamp, detected in frames:
        if not odom_times[0] <= stamp <= odom_times[-1]:
            continue
        pose = interpolate_pose(stamp, odom_times, odom_poses)
        expected, others = split_cones(track_xy, colors, pose, cameras, max_range)
        counts.add_frame(detected, expected, gate_m, ignore_xy=others)
        used += 1
    total_tp, total_fp, total_fn = int(counts.tp.sum()), int(counts.fp.sum()), int(counts.fn.sum())
    return {
        "frames": used,
        "gate_m": gate_m,
        "max_range_m": max_range,
        "all": {"recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else None,
                "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else None,
                "detections": total_tp + total_fp, "expected": total_tp + total_fn},
        "bins": counts.summary(),
    }


def timing_summary(frames: list[list[tuple[str, float]]]) -> dict:
    """Per stage of the nodes' timing messages: in how many frames it ran and its median, 95th and 99th
    percentile in ms, in the order the stages ran."""
    order, values = [], defaultdict(list)
    for stages in frames:
        for name, ms in stages:
            if name not in values:
                order.append(name)
            values[name].append(ms)
    return {name: {"frames": len(values[name]), "p50": float(np.percentile(values[name], 50)),
                   "p95": float(np.percentile(values[name], 95)), "p99": float(np.percentile(values[name], 99))}
            for name in order}


def camera_view(width: int, fx: float, x: float, y: float) -> CameraView:
    return CameraView(x, y, 2 * math.atan(width / 2 / fx))


def report_markdown(result: dict, title: str) -> str:
    """The scores of one race as markdown: per range, per lap and timing."""
    fmt = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    a = result["all"]
    overall = (f"{result['frames']} images at {fmt(result.get('rate_hz'), 1)} Hz. Overall recall {fmt(a['recall'])}, "
               f"precision {fmt(a['precision'])}.")
    latency = (f"Latency from the image to the recorded cones: median {fmt(result['latency_ms']['median'], 0)} ms, "
               f"95% within {fmt(result['latency_ms']['p95'], 0)} ms.")
    names = [f"`{n}`" for n in result.get("cameras", {})]
    cameras = ("both " if len(names) > 1 else "") + (" and ".join(names) or "the camera")
    lines = [
        f"# {title}",
        "",
        "Written by `scripts/score_run.py`. Cones the pipeline published during the race against the ground truth",
        f"track at each image's time: small cones in the field of view of {cameras} up to",
        (f"{result['max_range_m']:.0f} m are expected, a cone counts as found within {result['gate_m']:.1f} m. "
         "Finding a real cone that isn't expected counts neither way."),
        "",
        overall,
        latency,
        "",
    ]
    if result.get("cones_hit"):
        hit = (f"The car hit {result['cones_hit']} cone(s) during the race. The ground truth still has them where "
               "they stood, so near them a found cone can count as missed and false.")
        lines += [hit, ""]
    lines += [
        "## By range",
        "",
        "| range | expected | recall | precision | position RMSE |",
        "|---|---|---|---|---|",
    ]
    for b in result["bins"]:
        lines.append(f"| {b['bin_m'][0]:.0f}-{b['bin_m'][1]:.0f} m | {b['n_gt']} | {fmt(b['recall'])} | "
                     f"{fmt(b['precision'])} | {fmt(b['pos_rmse_m'])} m |")
    if result.get("strict"):
        strict = result["strict"]
        lines += ["", f"## With a {strict['gate_m']:.1f} m gate", "",
                  (f"The same frames, but a cone only counts as found within {strict['gate_m']:.1f} m. Overall recall "
                   f"{fmt(strict['all']['recall'])}, precision {fmt(strict['all']['precision'])}."), "",
                  "| range | recall | precision | position RMSE |", "|---|---|---|---|"]
        for b in strict["bins"]:
            lines.append(f"| {b['bin_m'][0]:.0f}-{b['bin_m'][1]:.0f} m | {fmt(b['recall'])} | {fmt(b['precision'])} | "
                         f"{fmt(b['pos_rmse_m'])} m |")
    if result.get("laps"):
        lines += ["", "## By lap", "", "| lap | images | recall | precision |", "|---|---|---|---|"]
        for lap in result["laps"]:
            lines.append(f"| {lap['lap']} | {lap['frames']} | {fmt(lap['recall'])} | {fmt(lap['precision'])} |")
    if result.get("timing"):
        explanation = [
            "Measured inside the node for every frame it published. Bridge is the time from the simulator's stamp",
            "until the bridge sent the image, transport until it arrived in the node, queue until the node took it",
            "up. Indented stages are details of the one above or ran alongside the others, they are not part of",
            "the total.",
        ]
        lines += ["", "## Latency per stage", "", *explanation, "", "| stage | frames | median | 95% | 99% |",
                  "|---|---|---|---|---|"]
        for name, t in result["timing"].items():
            label = f"&nbsp;&nbsp;{name.split('/', 1)[1]}" if "/" in name else name
            if name.startswith("right image/"):
                label = f"&nbsp;&nbsp;right image {name.split('/', 1)[1]}"
            lines.append(f"| {label} | {t['frames']} | {t['p50']:.1f} ms | {t['p95']:.1f} ms | {t['p99']:.1f} ms |")
    return "\n".join(lines) + "\n"
