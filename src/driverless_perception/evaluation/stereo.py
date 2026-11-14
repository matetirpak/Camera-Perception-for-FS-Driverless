"""How far off the stereo pipeline's cone positions are, next to mono PnP on the same cones.

Every frame of a capture has a left and a right image with the pose each was rendered from, so nothing here
depends on calibration or timing. Two variants as in ``evaluation.mono``:

- ``boxes``: ground truth boxes in the left image, for the cones the keypoint network was trained on (small, fully
  visible, at least ``min_height`` pixels high)
- ``detector``: the detector's boxes matched to the ground truth at IoU 0.5, the whole pipeline

Errors are split by ``evaluation.positions``. Mono and stereo are compared on the cones where both gave a
position, at every distance.
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..common.cameras import Intrinsics
from ..sim.autolabel import LabelPolicy
from ..sim.dataset import KEYPOINT_MIN_VISIBLE, label_from_row, read_cones
from ..vision.mono import SMALL_CONES
from ..vision.stereo.pipeline import StereoPipeline, View
from .boxes import box_iou, match
from .positions import position_error, summarise

DISTANCE_BINS = (0, 5, 10, 15, 20)
VARIANTS = ("boxes", "detector")


def view_from_row(capture: Path, run: dict, intrinsics: dict, row: dict) -> View:
    """One camera image of a capture with the pose it was rendered from, from its ``frames.csv`` row."""
    position = np.array([float(row[k]) for k in ("x", "y", "z")])
    orientation = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
    image = cv2.imread(str(capture / "images" / f"{row['image']}.jpg"))
    return View(image, intrinsics[row["camera"]], position, orientation,
                position[2] - run["mounts"][row["camera"]]["Z"])


def capture_frames(capture: Path) -> tuple[dict, dict, list[dict]]:
    """The run description, the intrinsics per camera and the ``frames.csv`` rows grouped by frame, each a dict
    from camera name to row."""
    run = yaml.safe_load((capture / "run.yaml").read_text())
    intrinsics = {cam: Intrinsics(c["width"], c["height"], c["fov_deg"]) for cam, c in run["cameras"].items()}
    frames = defaultdict(dict)
    with open(capture / "frames.csv", newline="") as f:
        for row in csv.DictReader(f):
            frames[row["image"].removesuffix("_" + row["camera"])][row["camera"]] = row
    return run, intrinsics, list(frames.values())


def evaluate_stereo(captures: list[Path], pipeline: StereoPipeline, policy: LabelPolicy | None = None,
                    detector: bool = True) -> dict:
    """Runs the variants on every left and right pair of ``captures``. The detector variant needs the mono
    pipeline's detector."""
    policy = policy or LabelPolicy()
    records = defaultdict(list)
    stages = []
    timing = defaultdict(float)
    for capture in captures:
        run = yaml.safe_load((capture / "run.yaml").read_text())
        cones = read_cones(capture / "cones.csv")
        intrinsics = {cam: Intrinsics(c["width"], c["height"], c["fov_deg"]) for cam, c in run["cameras"].items()}
        boxes = defaultdict(list)
        with open(capture / "boxes.csv", newline="") as f:
            for row in csv.DictReader(f):
                boxes[row["image"]].append(row)
        pairs = defaultdict(dict)
        with open(capture / "frames.csv", newline="") as f:
            for row in csv.DictReader(f):
                pairs[row["image"].removesuffix("_" + row["camera"])][row["camera"]] = row

        def view(row, capture=capture, run=run, intrinsics=intrinsics):
            return view_from_row(capture, run, intrinsics, row)

        for pair in pairs.values():
            if "cam_left" not in pair or "cam_right" not in pair:
                continue
            left, right = view(pair["cam_left"]), view(pair["cam_right"])
            labelled = [r for r in boxes[pair["cam_left"]["image"]] if policy.accepts(label_from_row(r))]
            gt_boxes = np.array([[float(r[k]) for k in ("x1", "y1", "x2", "y2")] for r in labelled]).reshape(-1, 4)
            gt_xy = np.array([cones[int(r["cone"])] for r in labelled]).reshape(-1, 2)
            gt_distance = np.array([float(r["distance"]) for r in labelled])
            gt_cone = [int(r["cone"]) for r in labelled]
            right_boxes = {int(r["cone"]): [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
                           for r in boxes[pair["cam_right"]["image"]]}
            clean = np.array([int(r["class"]) in SMALL_CONES and r["truncated"] == "0"
                              and float(r["visible"]) >= KEYPOINT_MIN_VISIBLE
                              and pipeline.mono.usable(gt_boxes[i], int(r["class"]))
                              for i, r in enumerate(labelled)], bool)

            for i, e in zip(np.flatnonzero(clean), pipeline.estimate(left, right, gt_boxes[clean])):
                records["boxes"].append(_record(e, gt_distance[i], gt_xy[i], left))
                stages.append(_stage(e, gt_distance[i], gt_xy[i], right_boxes.get(gt_cone[i])))

            if not detector:
                continue
            det_boxes, det_cls, det_score = pipeline.mono.detect(left.image)
            m = match(det_boxes, det_cls, det_score, gt_boxes, np.zeros(len(gt_boxes), int), class_aware=False)
            timing["detector_boxes"] += len(det_boxes)
            timing["detector_unmatched"] += int((m.det_gt < 0).sum())
            start = time.perf_counter()
            estimates = pipeline.estimate(left, right, det_boxes)
            timing["estimate_s"] += time.perf_counter() - start
            timing["frames"] += 1
            for g, e in zip(m.det_gt, estimates):
                if g >= 0:
                    records["detector"].append(_record(e, gt_distance[g], gt_xy[g], left))
    counts = {k: v for k, v in timing.items() if k != "estimate_s"}
    if timing["frames"]:
        counts["ms_per_frame"] = 1000 * timing["estimate_s"] / timing["frames"]
        counts["ms_per_box"] = 1000 * timing["estimate_s"] / max(timing["detector_boxes"], 1)
    samples = {name: [[round(r[name].distance_m, 3), round(r[name].estimate_m, 3)]
                      for r in records["detector"] if r[name] is not None] for name in ("mono", "stereo")}
    return {"bins": {v: _binned(records[v]) for v in VARIANTS}, "stages": _staged(stages), "counts": counts,
            "samples": samples}


def _record(e, distance: float, truth_xy, left) -> dict:
    """One cone: how the matching went and what mono and stereo made of it."""
    errors = {}
    for name, estimate in (("mono", e.mono), ("stereo", e.stereo)):
        errors[name] = (None if estimate is None
                        else position_error(estimate, truth_xy, left.ground_z, left.position[:2]))
    return {"distance": distance, "matches": e.matches, "points": e.points, **errors}


def _stage(e, distance: float, truth_xy, right_box) -> dict:
    """What every step made of one cone: how well the box was propagated, how the matching went and how far the
    points that ended up on the cone are from its axis."""
    row = {"distance": distance, "matches": e.matches, "on_cone": e.points}
    if right_box is not None and e.right_box is not None:
        row["iou"] = float(box_iou(np.array(e.right_box), np.array(right_box)))
        row["contains"] = bool(e.right_box[0] <= right_box[0] and e.right_box[1] <= right_box[1]
                               and e.right_box[2] >= right_box[2] and e.right_box[3] >= right_box[3])
    if e.matching is not None:
        row["features"] = min(e.matching["features_left"], e.matching["features_right"])
        row["ratio_test"] = e.matching["ratio_test"]
        if len(e.matching["epipolar_px"]):
            row["epipolar_px"] = float(np.median(e.matching["epipolar_px"]))
    if e.on_cone is not None and e.on_cone.any():
        on = e.world[e.on_cone]
        row["point_error"] = float(np.median(np.hypot(*(on[:, :2] - truth_xy).T)))
        row["median_minus_mean"] = float(np.hypot(*(np.median(on[:, :2], axis=0) - np.mean(on[:, :2], axis=0))))
    return row


def _staged(stages: list[dict]) -> list[dict]:
    """The stage numbers per distance bin, medians and shares."""
    out = []
    for lo, hi in pairwise(DISTANCE_BINS):
        sel = [r for r in stages if lo <= r["distance"] < hi]
        row = {"distance_m": [lo, hi], "cones": len(sel)}
        for key, name in (("iou", "iou_median"), ("features", "features_median"),
                          ("ratio_test", "ratio_test_median"), ("matches", "matches_median"),
                          ("on_cone", "on_cone_median"), ("epipolar_px", "epipolar_px_median"),
                          ("point_error", "point_error_median_m"), ("median_minus_mean", "median_minus_mean_m")):
            values = [r[key] for r in sel if key in r]
            if values:
                row[name] = float(np.median(values))
        contains = [r["contains"] for r in sel if "contains" in r]
        if contains:
            row["contains_share"] = float(np.mean(contains))
        errors = [r["point_error"] for r in sel if "point_error" in r]
        if errors:
            row["point_error_p90_m"] = float(np.percentile(errors, 90))
        matches = [r["matches"] for r in sel if r["matches"]]
        if matches:
            row["on_cone_share"] = float(np.mean([r["on_cone"] / r["matches"] for r in sel if r["matches"]]))
        out.append(row)
    return out


def _binned(records: list) -> list[dict]:
    out = []
    for lo, hi in pairwise(DISTANCE_BINS):
        sel = [r for r in records if lo <= r["distance"] < hi]
        with_pnp = [r for r in sel if r["mono"] is not None]
        both = [r for r in sel if r["mono"] is not None and r["stereo"] is not None]
        stats = {"distance_m": [lo, hi], "cones": len(sel), "with_pnp": len(with_pnp),
                 "with_stereo": sum(1 for r in sel if r["stereo"] is not None)}
        if with_pnp:
            stats["matches_median"] = float(np.median([r["matches"] for r in with_pnp]))
            stats["points_median"] = float(np.median([r["points"] for r in with_pnp]))
        for name in ("mono", "stereo"):
            if both:
                stats[name] = summarise([r[name] for r in both], (lo, hi))[0]
        out.append(stats)
    return out


def report_markdown(results: dict) -> str:
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    lines = [
        "# Stereo cone positions on simulator images",
        "",
        "Written by `scripts/evaluate_stereo.py`. Test map CompetitionMap3 at dusk, left and right image of each",
        f"frame with the pose each was rendered from. Detector `{results['detector']}` (threshold",
        f"{results['conf']:.2f}), keypoint network `{results['keypoints']}`. The error is the horizontal distance",
        "between estimated and true cone base, split into the range part (estimated minus true distance from the",
        "camera, median = bias, standard deviation = spread) and the lateral part across the line of sight. Mono",
        "and stereo are compared on the cones where both gave a position.",
        "",
    ]
    titles = {"boxes": "Ground truth boxes", "detector": "Detector boxes (whole pipeline)"}
    for variant in VARIANTS:
        bins = results["bins"][variant]
        if not any(b["cones"] for b in bins):
            continue
        header = ("| distance | cones | PnP | stereo | matches | on cone | median mono / stereo | 90% mono / stereo "
                  "| range bias mono / stereo | range spread mono / stereo | lateral RMSE mono / stereo |")
        lines += [f"## {titles[variant]}", "", header, "|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in bins:
            m, s = b.get("mono", {}), b.get("stereo", {})
            pair = lambda k, m=m, s=s: f"{fmt(m.get(k))} / {fmt(s.get(k))} m"
            lines.append(f"| {b['distance_m'][0]}-{b['distance_m'][1]} m | {b['cones']} | {b['with_pnp']} | "
                         f"{b['with_stereo']} | {b.get('matches_median', 0):.0f} | {b.get('points_median', 0):.0f} | "
                         f"{pair('median_m')} | {pair('p90_m')} | {pair('range_bias_m')} | "
                         f"{pair('range_spread_m')} | {pair('lateral_rmse_m')} |")
        lines.append("")
    if results.get("stages") and any(b["cones"] for b in results["stages"]):
        stage_header = ("| distance | cones | box IoU | box holds the cone | features | after ratio test | matches |"
                        " on cone | epipolar | point error | median vs mean |")
        lines += [
            "## Stage by stage (ground truth boxes)",
            "",
            "How each step did its job: the propagated box against the true box of the same cone in the right",
            "image, what the matching found, and how far the points that ended up on the cone lie from its axis.",
            "",
            stage_header,
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        count = lambda v: "-" if v is None else f"{v:.0f}"
        for b in results["stages"]:
            share = b.get("contains_share")
            lines.append(f"| {b['distance_m'][0]}-{b['distance_m'][1]} m | {b['cones']} | "
                         f"{fmt(b.get('iou_median'))} | {'-' if share is None else f'{share:.0%}'} | "
                         f"{count(b.get('features_median'))} | {count(b.get('ratio_test_median'))} | "
                         f"{count(b.get('matches_median'))} | {count(b.get('on_cone_median'))} | "
                         f"{fmt(b.get('epipolar_px_median'))} px | {fmt(b.get('point_error_median_m'))} m | "
                         f"{fmt(b.get('median_minus_mean_m'))} m |")
        lines.append("")
    c = results["counts"]
    if c.get("frames"):
        lines += ["## Counts and time", ""]
        lines.append(f"- detector: {c['detector_boxes']:.0f} usable boxes in {c['frames']:.0f} left images, "
                     f"{c['detector_unmatched']:.0f} of them matched no labelled cone")
        lines.append(f"- keypoints, PnP and stereo together: {c['ms_per_frame']:.1f} ms per frame, "
                     f"{c['ms_per_box']:.1f} ms per box, without the detector")
    return "\n".join(lines) + "\n"
