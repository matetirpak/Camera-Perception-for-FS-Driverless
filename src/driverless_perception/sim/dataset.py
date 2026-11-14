"""Turning simulator captures (``capture.capture``) into one YOLO dataset, split by map.

The labels are written again from each capture's ``boxes.csv`` with the given ``LabelPolicy``, so the policy can
change without driving again. Images are hard linked instead of copied, the dataset takes no extra space.

``keypoints.csv`` lists the 7 keypoints of every labelled small cone that is fully in view, for the keypoint
network. They are pure geometry (camera pose, cone position, cone model), so they are computed here and not while
capturing.
"""
from __future__ import annotations

import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..common.cameras import Intrinsics
from ..common.types import CLASS_NAMES, ConeColor
from .autolabel import ConeLabel, LabelPolicy, project_keypoints, yolo_lines

SPLITS = ("train", "val", "test")
# the paper's keypoints are for the small cones with one stripe, and only cones that are fully in view get them
KEYPOINT_CLASSES = (ConeColor.BLUE, ConeColor.YELLOW, ConeColor.ORANGE_SMALL)
KEYPOINT_MIN_VISIBLE = 0.9


def read_split(path: str | Path) -> dict[str, str]:
    """Map name to split from a file with train, val and test lists."""
    data = yaml.safe_load(Path(path).read_text())
    split = {}
    for name in SPLITS:
        for m in data[name]:
            if m in split:
                raise ValueError(f"{path}: {m} is in more than one split")
            split[m] = name
    return split


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_cones(path: Path) -> np.ndarray | None:
    """(N, 2) cone positions from a capture's cones.csv, None if the capture has none."""
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return np.array([[float(r["x"]), float(r["y"])] for r in csv.DictReader(f)]).reshape(-1, 2)


def export_captures(captures: list[Path], split: dict[str, str], out: Path, policy: LabelPolicy) -> dict:
    """Writes ``images/<split>``, ``labels/<split>``, ``boxes.csv``, ``keypoints.csv`` and ``fsds.yaml`` below
    ``out``.

    Returns per split the maps, the number of images, the number of labels per class and the number of cones with
    keypoints.
    """
    summary = {name: {"maps": set(), "images": 0, "labels": np.zeros(len(CLASS_NAMES), int), "keypoint_cones": 0}
               for name in SPLITS}
    merged, keypoint_rows = [], []
    for capture in captures:
        run = yaml.safe_load((capture / "run.yaml").read_text())
        if run["map"] not in split:
            raise ValueError(f"{capture.name}: map {run['map']} is not in the split")
        part = split[run["map"]]
        intrinsics = {cam: Intrinsics(c["width"], c["height"], c["fov_deg"]) for cam, c in run["cameras"].items()}
        cones = read_cones(capture / "cones.csv")
        boxes = defaultdict(list)
        with open(capture / "boxes.csv", newline="") as f:
            for row in csv.DictReader(f):
                boxes[row["image"]].append(row)
        with open(capture / "frames.csv", newline="") as f:
            frames = {row["image"]: row for row in csv.DictReader(f)}

        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        for image, frame in frames.items():
            camera = frame["camera"]
            intr = intrinsics[camera]
            labels = []
            for row in boxes[image]:
                lab = ConeLabel(int(row["cone"]), int(row["class"]),
                                tuple(int(row[k]) for k in ("x1", "y1", "x2", "y2")), float(row["distance"]),
                                float(row["visible"]), float(row["contrast"]), bool(int(row["truncated"])))
                ok = policy.accepts(lab)
                if ok:
                    labels.append(lab)
                merged.append({"split": part, "map": run["map"], **row, "labelled": int(ok)})
                if (ok and cones is not None and lab.color in KEYPOINT_CLASSES and not lab.truncated
                        and lab.visible >= KEYPOINT_MIN_VISIBLE):
                    position = np.array([float(frame[k]) for k in ("x", "y", "z")])
                    orientation = np.array([float(frame[k]) for k in ("qw", "qx", "qy", "qz")])
                    ground_z = position[2] - run["mounts"][camera]["Z"]
                    uv = project_keypoints(cones[lab.index], ground_z, position, orientation, intr)
                    keypoint_rows.append({"split": part, "map": run["map"], "image": image, "class": lab.color,
                                          "x1": lab.box[0], "y1": lab.box[1], "x2": lab.box[2], "y2": lab.box[3],
                                          "distance": round(lab.distance, 3),
                                          **{f"{a}{i}": round(float(uv[i, j]), 2)
                                             for i in range(7) for j, a in enumerate("uv")}})
                    summary[part]["keypoint_cones"] += 1
            _link(capture / "images" / f"{image}.jpg", out / "images" / part / f"{image}.jpg")
            lines = yolo_lines(labels, intr.width, intr.height)
            (out / "labels" / part / f"{image}.txt").write_text("".join(line + "\n" for line in lines))
            summary[part]["images"] += 1
            for lab in labels:
                summary[part]["labels"][lab.color] += 1
        summary[part]["maps"].add(run["map"])

    for name, rows in (("boxes.csv", merged), ("keypoints.csv", keypoint_rows)):
        if rows:
            with open(out / name, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    dataset = {name: f"images/{name}" for name in SPLITS}
    dataset["names"] = CLASS_NAMES
    with open(out / "fsds.yaml", "w") as f:
        f.write("# Simulator images labelled from ground truth, split by map. Written by scripts/prepare_fsds.py.\n")
        yaml.safe_dump(dataset, f, sort_keys=False)
    return summary


def label_from_row(row: dict) -> ConeLabel:
    """A ``boxes.csv`` row of a capture as a ``ConeLabel``."""
    return ConeLabel(int(row["cone"]), int(row["class"]), tuple(int(row[k]) for k in ("x1", "y1", "x2", "y2")),
                     float(row["distance"]), float(row["visible"]), float(row["contrast"]), row["truncated"] == "1")
