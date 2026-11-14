"""The FSOCO bounding box dataset: reading it, splitting it by camera and exporting it in YOLO format.

FSOCO is in the Supervisely format, one folder per team, every image with a 140 px watermark border that is
cropped off like the FSOCO devkit does. Class ids follow ``ConeColor`` so the detector needs no mapping.

The folders are the teams that labelled the images, not the ones that recorded them, so a recording can be spread
over many folders. The file name says where an image comes from and the split goes by that, otherwise frames of
one recording end up in train and test (docs/notes/datasets.md).
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from ...common.types import CLASS_NAMES, ConeColor

BORDER = 140
SPLITS = ("train", "val", "test")


@dataclass
class Box:
    """A box in pixels of the cropped image (border removed, before any resizing)."""

    color: ConeColor
    x1: float
    y1: float
    x2: float
    y2: float
    tags: tuple[str, ...] = ()

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class Sample:
    team: str
    image: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Output file name without extension: labelling team and original name, e.g. ``ugr_amz_00804``."""
        return f"{self.team}_{Path(self.image.name).stem}"

    @property
    def origin(self) -> str:
        return image_origin(self.image.name)


_FILE_NAME = re.compile(r"([A-Za-z]+)_?(\d+)")


def image_origin(file_name: str) -> str:
    """The team whose camera took an image, from its file name. ``amz_00805.png`` is AMZ's, whoever labelled it.

    AMZ's images are two unrelated sets. Numbers 0 to 799 are in AMZ's own folder, 1880x1080 and 2872x2328 frames
    with a black car in view. From 804 on they are 2872x920 frames with a white car, labelled by 19 other teams.
    We keep them apart as ``amz`` and ``amz_wide``.
    """
    m = _FILE_NAME.match(file_name)
    if m is None:
        raise ValueError(f"unexpected FSOCO file name: {file_name}")
    prefix, number = m.group(1).lower(), int(m.group(2))
    if prefix == "amz" and number >= 800:
        return "amz_wide"
    return prefix


def parse_export_name(name: str) -> tuple[str, str]:
    """(team, origin) of an exported image, e.g. ``ugr_amz_00804`` was labelled by ugr and taken by AMZ."""
    team, original = name.split("_", 1)
    return team, image_origin(original)


def read_annotation(path: str | Path) -> Sample:
    """Reads one ``<team>/ann/<image>.json`` file. Boxes are moved into the cropped image and clipped to it."""
    path = Path(path)
    data = json.loads(path.read_text())
    width = data["size"]["width"] - 2 * BORDER
    height = data["size"]["height"] - 2 * BORDER
    boxes = []
    for obj in data["objects"]:
        (xa, ya), (xb, yb) = obj["points"]["exterior"]
        x1, x2 = np.clip(sorted((xa - BORDER, xb - BORDER)), 0, width)
        y1, y2 = np.clip(sorted((ya - BORDER, yb - BORDER)), 0, height)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        tags = tuple(sorted(t["name"] for t in obj.get("tags", [])))
        boxes.append(Box(ConeColor.from_fsoco(obj["classTitle"]), float(x1), float(y1), float(x2), float(y2), tags))
    team_dir = path.parent.parent
    # the annotation of image.png is image.png.json
    return Sample(team_dir.name, team_dir / "img" / path.stem, width, height, boxes)


def load_fsoco(root: str | Path) -> list[Sample]:
    """All samples of an unpacked FSOCO bounding box download, sorted by team and image name."""
    anns = sorted(Path(root).glob("*/ann/*.json"))
    if not anns:
        raise FileNotFoundError(f"no FSOCO annotations found in {root}")
    return [read_annotation(a) for a in anns]


def yolo_lines(sample: Sample) -> list[str]:
    """One ``class cx cy w h`` line per box, normalised to the image size, as YOLO expects."""
    lines = []
    for b in sample.boxes:
        cx = (b.x1 + b.x2) / 2 / sample.width
        cy = (b.y1 + b.y2) / 2 / sample.height
        w = (b.x2 - b.x1) / sample.width
        h = (b.y2 - b.y1) / sample.height
        lines.append(f"{int(b.color)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_image(src: Path, dst: Path, size: tuple[int, int], long_side: int = 1920, quality: int = 95) -> None:
    """Crops the border, shrinks the image so its longer side is at most ``long_side`` and saves it as JPEG.

    ``size`` is the (width, height) the annotation expects after cropping, as a check that image and labels match.
    """
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read {src}")
    image = image[BORDER:-BORDER, BORDER:-BORDER]
    h, w = image.shape[:2]
    if (w, h) != tuple(size):
        raise ValueError(f"{src} is {w}x{h} after cropping, the annotation says {size[0]}x{size[1]}")
    scale = long_side / max(w, h)
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"could not write {dst}")


# Splitting

def origin_counts(samples: list[Sample]) -> dict[str, np.ndarray]:
    """Per origin: [images, boxes of class 0, ..., boxes of class 4]."""
    counts: dict[str, np.ndarray] = {}
    for s in samples:
        c = counts.setdefault(s.origin, np.zeros(1 + len(CLASS_NAMES), int))
        c[0] += 1
        for b in s.boxes:
            c[1 + int(b.color)] += 1
    return counts


def group_split(
    counts: dict[str, np.ndarray],
    val: tuple[str, ...] = (),
    test: tuple[str, ...] = (),
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
    trials: int = 20000,
) -> dict[str, list[str]]:
    """Assigns whole groups (origins) to train, val and test.

    Groups listed in ``val`` and ``test`` go there first. If that isn't enough images yet, the rest is found by
    random search: shuffle the remaining groups, fill val and then test up to their share of the images, and keep
    the assignment whose image shares and class proportions come closest to the targets. Class proportions are
    compared relative to each class's overall share, so the rare classes count as much as blue and yellow.
    """
    fixed = {"val": list(val), "test": list(test)}
    unknown = (set(val) | set(test)) - set(counts)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    if set(val) & set(test):
        raise ValueError(f"groups in both val and test: {sorted(set(val) & set(test))}")
    total = sum(counts.values())
    overall = total[1:] / total[1:].sum()
    targets = {"val": val_fraction * total[0], "test": test_fraction * total[0]}

    def score(assignment: dict[str, list[str]]) -> float:
        s = 0.0
        for name in ("val", "test"):
            c = sum((counts[g] for g in assignment[name]), np.zeros_like(total))
            s += abs(c[0] - targets[name]) / targets[name]
            share = c[1:] / max(c[1:].sum(), 1)
            s += np.mean(np.abs(share - overall) / overall)
        return s

    rng = np.random.default_rng(seed)
    free = [g for g in sorted(counts) if g not in set(val) | set(test)]
    best, best_score = None, np.inf
    for _ in range(trials):
        assignment = {"train": [], "val": list(fixed["val"]), "test": list(fixed["test"])}
        filled = {name: sum(counts[g][0] for g in fixed[name]) for name in ("val", "test")}
        for g in rng.permutation(free):
            for name in ("val", "test"):
                if filled[name] < targets[name]:
                    assignment[name].append(str(g))
                    filled[name] += counts[g][0]
                    break
            else:
                assignment["train"].append(str(g))
        s = score(assignment)
        if s < best_score:
            best, best_score = assignment, s
    return {name: sorted(best[name]) for name in SPLITS}


def read_split(path: str | Path) -> dict[str, list[str]]:
    data = yaml.safe_load(Path(path).read_text())
    split = {name: list(data[name]) for name in SPLITS}
    seen = [g for name in SPLITS for g in split[name]]
    if len(seen) != len(set(seen)):
        raise ValueError(f"{path}: an origin appears in more than one split")
    return split


def split_table(counts: dict[str, np.ndarray], split: dict[str, list[str]]) -> str:
    """A small text table with images and boxes per class for each split."""
    total = sum(counts.values())
    header = ["split", "origins", "images", "share"] + [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    rows = [header]
    for name in SPLITS:
        c = sum((counts[t] for t in split[name]), np.zeros_like(total))
        rows.append([name, str(len(split[name])), str(c[0]), f"{c[0] / total[0]:.1%}"] + [str(v) for v in c[1:]])
    rows.append(["all", str(len(counts)), str(total[0]), "100%"] + [str(v) for v in total[1:]])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join("  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in rows)


# Export

def _export_one(job: tuple[Sample, Path, str, int, int]) -> None:
    sample, out, split_name, long_side, quality = job
    image = out / "images" / split_name / f"{sample.name}.jpg"
    convert_image(sample.image, image, (sample.width, sample.height), long_side, quality)
    label = out / "labels" / split_name / f"{sample.name}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("".join(line + "\n" for line in yolo_lines(sample)))


def export_yolo(
    samples: list[Sample],
    split: dict[str, list[str]],
    out: str | Path,
    long_side: int = 1920,
    quality: int = 95,
    workers: int | None = None,
) -> dict[str, int]:
    """Writes ``images/<split>/``, ``labels/<split>/``, ``boxes.csv`` and ``fsoco.yaml`` below ``out``.

    ``boxes.csv`` has one row per box with its line in the label file, its size in the cropped image before resizing
    and its tags, which is what the evaluation needs to report results by box size, origin and tag.
    """
    out = Path(out)
    origin_to_split = {g: name for name in SPLITS for g in split[name]}
    missing = sorted({s.origin for s in samples} - set(origin_to_split))
    if missing:
        raise ValueError(f"origins not in the split: {missing}")

    jobs = [(s, out, origin_to_split[s.origin], long_side, quality) for s in samples]
    cv2.setNumThreads(1)
    with Pool(workers or os.cpu_count()) as pool:
        for done, _ in enumerate(pool.imap_unordered(_export_one, jobs, chunksize=8), start=1):
            if done % 500 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} images", flush=True)

    with open(out / "boxes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "team", "origin", "line", "class", "x1", "y1", "x2", "y2", "image_width",
                    "image_height", "tags"])
        for s in samples:
            for i, b in enumerate(s.boxes):
                w.writerow([origin_to_split[s.origin], s.name, s.team, s.origin, i, int(b.color), f"{b.x1:.1f}",
                            f"{b.y1:.1f}", f"{b.x2:.1f}", f"{b.y2:.1f}", s.width, s.height, " ".join(b.tags)])

    dataset = {name: f"images/{name}" for name in SPLITS}
    dataset["names"] = CLASS_NAMES
    with open(out / "fsoco.yaml", "w") as f:
        f.write("# FSOCO bounding boxes, border removed, split by camera origin. Written by scripts/prepare_fsoco.py.\n")
        yaml.safe_dump(dataset, f, sort_keys=False)

    counts = {name: 0 for name in SPLITS}
    for s in samples:
        counts[origin_to_split[s.origin]] += 1
    return counts
