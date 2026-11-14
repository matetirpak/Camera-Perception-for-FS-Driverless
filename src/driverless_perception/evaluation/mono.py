"""How far off the mono pipeline's cone positions are, per distance, on simulator captures.

Three variants show where the error comes from:

- ``exact``: keypoints projected from the ground truth, so only PnP and the cone model can be wrong
- ``boxes``: ground truth boxes, keypoints from the network
- ``detector``: the detector's boxes matched to the ground truth at IoU 0.5, keypoints from the network, the whole
  pipeline

The first two use the cones the keypoint network was trained on: small, fully visible, at least ``min_height``
pixels high. Errors are split into range, lateral and vertical by ``evaluation.positions``; PnP is the only step
that estimates the height of the cone base, so the vertical part only says something here. Dhall et al. (2019)
report about 0.5 m at 10 m and 1 m at 16 m.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..common.cameras import Intrinsics, optical_to_world, up_in_optical
from ..sim.autolabel import LabelPolicy, project_keypoints
from ..sim.dataset import KEYPOINT_MIN_VISIBLE, label_from_row, read_cones
from ..vision.mono import SMALL_CONES, MonoPipeline
from .boxes import match
from .positions import position_error, summarise

DISTANCE_BINS = (0, 5, 10, 15, 20, 30)
VARIANTS = ("exact", "boxes", "detector")
# the same detector keypoints turned into positions four ways, for the report's comparison: PnP of the full pose,
# the upright cone, and the cone held on the ground plane with the true attitude and with the car taken as level
PNP_COMPARISON = (("sqpnp", "true"), ("upright", "true"), ("ground", "true"), ("ground", "level"))
LEVEL_UP = np.array([0.0, -1.0, 0.0])


def evaluate_mono(captures: list[Path], pipeline: MonoPipeline, detector=None, conf: float = 0.25,
                  imgsz: int = 1280, policy: LabelPolicy | None = None) -> dict:
    """Runs the variants on every image of ``captures``. ``detector`` is an ultralytics model or None."""
    policy = policy or LabelPolicy()
    errors = defaultdict(list)
    comparison = defaultdict(list)
    counts = defaultdict(int)
    for capture in captures:
        run = yaml.safe_load((capture / "run.yaml").read_text())
        cones = read_cones(capture / "cones.csv")
        intrinsics = {cam: Intrinsics(c["width"], c["height"], c["fov_deg"]) for cam, c in run["cameras"].items()}
        boxes = defaultdict(list)
        with open(capture / "boxes.csv", newline="") as f:
            for row in csv.DictReader(f):
                boxes[row["image"]].append(row)
        with open(capture / "frames.csv", newline="") as f:
            frames = list(csv.DictReader(f))

        for frame in frames:
            name, camera = frame["image"], frame["camera"]
            intr = intrinsics[camera]
            K = intr.matrix
            position = np.array([float(frame[k]) for k in ("x", "y", "z")])
            orientation = np.array([float(frame[k]) for k in ("qw", "qx", "qy", "qz")])
            ground_z = position[2] - run["mounts"][camera]["Z"]
            up, height = up_in_optical(orientation), position[2] - ground_z
            image = cv2.imread(str(capture / "images" / f"{name}.jpg"))

            labelled = [r for r in boxes[name] if policy.accepts(label_from_row(r))]
            gt_boxes = np.array([[float(r[k]) for k in ("x1", "y1", "x2", "y2")] for r in labelled]).reshape(-1, 4)
            gt_cone = np.array([int(r["cone"]) for r in labelled], int)
            clean = np.array([int(r["class"]) in SMALL_CONES and r["truncated"] == "0"
                              and float(r["visible"]) >= KEYPOINT_MIN_VISIBLE
                              and pipeline.usable(gt_boxes[i], int(r["class"]))
                              for i, r in enumerate(labelled)], bool)

            def world(t, position=position, orientation=orientation):
                return None if t is None else optical_to_world(t[None], position, orientation)[0]

            for i in np.flatnonzero(clean):
                kp = project_keypoints(cones[gt_cone[i]], ground_z, position, orientation, intr)
                estimate = world(pipeline.locate(kp[None], K, up, height)[0])
                _record(errors, counts, "exact", estimate, cones[gt_cone[i]], ground_z, position)
            if clean.any():
                kps = pipeline.keypoints(image, gt_boxes[clean])
                for i, t in zip(np.flatnonzero(clean), pipeline.locate(kps, K, up, height)):
                    _record(errors, counts, "boxes", world(t), cones[gt_cone[i]], ground_z, position)

            if detector is None:
                continue
            size = pipeline.input_size() if pipeline.detector is detector else imgsz
            result = detector.predict(image, imgsz=size, conf=conf, verbose=False,
                                      **pipeline.detector_options())[0].boxes
            det_boxes = result.xyxy.cpu().numpy()
            det_cls = result.cls.cpu().numpy().astype(int)
            det_score = result.conf.cpu().numpy()
            use = np.array([pipeline.usable(b, c) for b, c in zip(det_boxes, det_cls)], bool)
            use &= pipeline.inside(det_boxes, intr.width)
            det_boxes, det_cls, det_score = det_boxes[use], det_cls[use], det_score[use]
            m = match(det_boxes, det_cls, det_score, gt_boxes, np.zeros(len(gt_boxes), int), class_aware=False)
            counts["detector_boxes"] += len(det_boxes)
            counts["detector_unmatched"] += int((m.det_gt < 0).sum())
            matched = m.det_gt >= 0
            if matched.any():
                kps = pipeline.keypoints(image, det_boxes[matched])
                for g, t in zip(m.det_gt[matched], pipeline.locate(kps, K, up, height)):
                    _record(errors, counts, "detector", world(t), cones[gt_cone[g]], ground_z, position)
                for mode, attitude in PNP_COMPARISON:
                    solver = MonoPipeline(None, pnp=mode)
                    for g, t in zip(m.det_gt[matched], solver.locate(kps, K, up if attitude == "true" else LEVEL_UP,
                                                                       height)):
                        _record(comparison, counts, f"{mode} {attitude}", world(t), cones[gt_cone[g]], ground_z,
                                position)
    samples = {v: [[round(e.distance_m, 3), round(e.estimate_m, 3)] for e in errors.get(v, [])] for v in VARIANTS}
    names = [f"{mode} {attitude}" for mode, attitude in PNP_COMPARISON]
    return {"bins": _binned(errors), "counts": dict(counts), "samples": samples, "pnp": pipeline.pnp,
            "edge_px": pipeline.edge_px,
            "pnp_comparison": {n: summarise(comparison.get(n, []), DISTANCE_BINS) for n in names}}


def _record(errors, counts, variant: str, estimate, truth_xy, ground_z: float, camera) -> None:
    """Adds one cone, or counts it as a failure if PnP found no position."""
    counts[f"{variant}_cones"] += 1
    if estimate is None:
        counts[f"{variant}_failed"] += 1
        return
    errors[variant].append(position_error(estimate, truth_xy, ground_z, camera[:2]))


def _binned(errors: dict) -> dict:
    return {variant: summarise(errors.get(variant, []), DISTANCE_BINS) for variant in VARIANTS}


def report_markdown(results: dict, detector_name: str, keypoint_name: str) -> str:
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    setup = (f"detector `{detector_name}`, PnP `{results.get('pnp', 'sqpnp')}`, detector boxes within "
             f"{results.get('edge_px', 0):g} px of the left or right border left out.")
    lines = [
        "# Mono cone positions on simulator images",
        "",
        f"Written by `scripts/evaluate_mono.py`. Test map CompetitionMap3 at dusk. Keypoint network `{keypoint_name}`,",
        setup,
        "The error is the distance between the estimated and the true cone base,",
        "horizontally and in 3D, and split into its parts: range (estimated minus true distance from the camera,",
        "negative means too close), lateral (across the line of sight) and vertical. Dhall et al. (2019) report",
        "about 0.5 m at 10 m and 1 m at 16 m.",
        "",
    ]
    if "keypoints_detail" in results:
        from .keypoints import report_markdown as keypoint_report

        lines += keypoint_report(results["keypoints_detail"]).split("\n")
    titles = {"exact": "Exact keypoints (PnP and cone model only)", "boxes": "Ground truth boxes, keypoint network",
              "detector": "Detector boxes, keypoint network (whole pipeline)"}
    header = ("| distance | cones | median | RMSE | 90% | 3D median | range bias | range spread | lateral RMSE |"
              " vertical bias |")
    for variant in VARIANTS:
        lines += [f"## {titles[variant]}", "", header, "|---|---|---|---|---|---|---|---|---|---|"]
        for b in results["bins"][variant]:
            lines.append(f"| {b['distance_m'][0]}-{b['distance_m'][1]} m | {b['cones']} | {fmt(b.get('median_m'))} m | "
                         f"{fmt(b.get('rmse_m'))} m | {fmt(b.get('p90_m'))} m | {fmt(b.get('median_3d_m'))} m | "
                         f"{fmt(b.get('range_bias_m'))} m | {fmt(b.get('range_spread_m'))} m | "
                         f"{fmt(b.get('lateral_rmse_m'))} m | {fmt(b.get('vertical_bias_m'))} m |")
        lines.append("")
    if results.get("pnp_comparison"):
        lines += [
            "## Solver comparison",
            "",
            "The detector's keypoints of the whole pipeline above, solved four ways: PnP of the full pose (SQPnP, the",
            "paper's way), an upright cone with only its base position unknown (what the pipeline uses), and that",
            "cone held on the ground plane, once with the camera's true attitude and once with the car taken as",
            "level. Median error and range bias per distance.",
            "",
            "| distance | SQPnP | upright | ground, true attitude | ground, car level |",
            "|---|---|---|---|---|",
        ]
        comp = results["pnp_comparison"]
        names = ["sqpnp true", "upright true", "ground true", "ground level"]
        for i, b in enumerate(comp["sqpnp true"]):
            cells = [f"{fmt(comp[n][i].get('median_m'))} / {fmt(comp[n][i].get('range_bias_m'))} m" for n in names]
            lines.append(f"| {b['distance_m'][0]}-{b['distance_m'][1]} m | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Held on the ground the position comes from where the cone stands in the image rather than from its size,",
            "which is only better while the attitude is known to about a tenth of a degree. The simulator has a flat",
            "ground and gives the exact attitude, a car doesn't, so the pipeline uses the upright cone.",
            "",
        ]
    c = results["counts"]
    lines += ["## Counts", ""]
    for variant in VARIANTS:
        if c.get(f"{variant}_cones"):
            lines.append(f"- {variant}: {c[f'{variant}_cones']} cones, PnP failed for {c.get(f'{variant}_failed', 0)}")
    if c.get("detector_boxes"):
        lines.append(f"- detector: {c['detector_boxes']} usable boxes, {c.get('detector_unmatched', 0)} of them "
                     "matched no labelled cone")
    return "\n".join(lines) + "\n"
