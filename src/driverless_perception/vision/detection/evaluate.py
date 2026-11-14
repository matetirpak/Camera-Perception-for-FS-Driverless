"""Evaluating YOLO cone detectors on the FSOCO export (``fsoco.export_yolo``) or the simulator dataset
(``sim.dataset.export_captures``).

``evaluate()`` returns one dictionary per model:

- ``coco``: mAP50 and mAP50-95 per class from ultralytics' own validation
- ``threshold``: the confidence threshold with the best F1 on val, used for everything below
- ``test``: precision, recall and F1 per class at that threshold, and how often a found cone gets the right colour
- ``paper``: the same with both orange classes merged and unknown cones ignored, the way Dhall et al. (2019) count
- ``by_height``, ``by_group``, ``by_tag``: recall per box height at the network input, per group (camera origin for
  FSOCO, map for the simulator) and per tag
- ``by_distance``: recall per metre range, only for the simulator, FSOCO has no distances
- ``latency``: milliseconds per image at the size of the simulator cameras

Boxes match at IoU 0.5 (see ``evaluation.boxes``).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from ...common.types import CLASS_NAMES, ConeColor
from ...evaluation.boxes import IGNORED, TP, f1, match, precision_recall
from .fsoco import parse_export_name

HEIGHT_BINS = (0, 8, 16, 32, 64, np.inf)
DISTANCE_BINS = (0, 10, 20, 30, 40, 60, np.inf)
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
TAGS = ("truncated", "knocked_over", "sticker_band_removed")
PAPER_CLASSES = {int(ConeColor.BLUE): "blue", int(ConeColor.YELLOW): "yellow", int(ConeColor.ORANGE_BIG): "orange"}


@dataclass
class ImageLabels:
    """Ground truth of one image. ``group`` is the camera origin (FSOCO) or the map (simulator), ``distance`` has
    one entry per box in metres, NaN where it isn't known."""

    name: str
    group: str
    width: int
    height: int
    xyxy: np.ndarray
    cls: np.ndarray
    tags: list[tuple[str, ...]]
    distance: np.ndarray


def dataset_name(dataset: Path) -> str:
    """``fsoco`` or ``fsds``, from the dataset file in the folder."""
    files = sorted(dataset.glob("*.yaml"))
    if len(files) != 1:
        raise ValueError(f"expected one dataset yaml in {dataset}, found {[f.name for f in files]}")
    return files[0].stem


def load_labels(dataset: Path, split: str, limit: int | None = None) -> list[ImageLabels]:
    """Ground truth of one split in pixels of the exported images.

    FSOCO's ``boxes.csv`` has every box with its tags, the camera origin comes from the file name. The simulator's
    has the map and a distance per box, and only the boxes marked as labelled are in the label files.
    """
    from PIL import Image

    extra = defaultdict(list)
    groups = {}
    with open(dataset / "boxes.csv", newline="") as f:
        reader = csv.DictReader(f)
        simulator = "distance" in reader.fieldnames
        for row in reader:
            if row["split"] != split:
                continue
            if simulator:
                if row["labelled"] == "1":
                    tags = ("truncated",) if row["truncated"] == "1" else ()
                    extra[row["image"]].append((tags, float(row["distance"])))
                groups[row["image"]] = row["map"]
            else:
                extra[row["image"]].append((tuple(row["tags"].split()), np.nan))

    images = sorted((dataset / "images" / split).glob("*.jpg"))[:limit]
    labels = []
    for image in images:
        with Image.open(image) as im:
            w, h = im.size
        text = (dataset / "labels" / split / f"{image.stem}.txt").read_text().split()
        rows = np.array(text, float).reshape(-1, 5)
        info = extra[image.stem]
        if len(info) != len(rows):
            raise ValueError(f"{image.stem}: {len(rows)} labels but {len(info)} rows in boxes.csv")
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        if simulator:
            group = groups.get(image.stem, image.stem.split("_")[0])
        else:
            group = parse_export_name(image.stem)[1]
        labels.append(ImageLabels(image.stem, group, w, h, xyxy, rows[:, 0].astype(int), [t for t, _ in info],
                                  np.array([d for _, d in info], float)))
    return labels


def _precision(device) -> int | None:
    """Half precision on the GPU, full precision on the CPU (ultralytics' ``quantize`` argument)."""
    return None if str(device) == "cpu" else 16


def predict(model, dataset: Path, split: str, labels: list[ImageLabels], imgsz: int, device, conf: float = 0.01,
            batch: int = 8):
    """Detections per image name as (xyxy, class, score), in pixels of the exported images."""
    paths = [str(dataset / "images" / split / f"{lab.name}.jpg") for lab in labels]
    out = {}
    # ultralytics loads a list of paths into memory at once and runs it as one batch, whatever batch says
    for i in range(0, len(paths), batch):
        for r in model.predict(paths[i:i + batch], imgsz=imgsz, conf=conf, device=device,
                               quantize=_precision(device), verbose=False):
            b = r.boxes
            out[Path(r.path).stem] = (b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy())
    return out


@dataclass
class Matched:
    """All detections and ground truth boxes of a split after matching, one entry per box.

    ``pairs`` has one row (ground truth class, detected class, score) per colour agnostic match.
    """

    det_result: np.ndarray
    det_score: np.ndarray
    det_cls: np.ndarray
    det_height: np.ndarray
    det_group: np.ndarray
    gt_score: np.ndarray
    gt_cls: np.ndarray
    gt_ignore: np.ndarray
    gt_height: np.ndarray
    gt_group: np.ndarray
    gt_distance: np.ndarray
    gt_tags: list[tuple[str, ...]]
    pairs: np.ndarray

    @property
    def n_gt(self) -> int:
        return int((~self.gt_ignore).sum())


def match_split(labels: list[ImageLabels], detections: dict, imgsz: int, paper: bool = False) -> Matched:
    """Matches every image. ``paper`` merges the orange classes and ignores unknown cones."""
    cols = {k: [] for k in Matched.__dataclass_fields__}
    for lab in labels:
        d_xyxy, d_cls, d_score = detections.get(lab.name, (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)))
        g_cls = lab.cls.copy()
        g_ignore = np.zeros(len(g_cls), bool)
        if paper:
            keep = d_cls != ConeColor.UNKNOWN
            d_xyxy, d_cls, d_score = d_xyxy[keep], d_cls[keep], d_score[keep]
            d_cls = np.where(d_cls == ConeColor.ORANGE_SMALL, int(ConeColor.ORANGE_BIG), d_cls)
            g_cls = np.where(g_cls == ConeColor.ORANGE_SMALL, int(ConeColor.ORANGE_BIG), g_cls)
            g_ignore = g_cls == ConeColor.UNKNOWN
        m = match(d_xyxy, d_cls, d_score, lab.xyxy, g_cls, g_ignore)
        agnostic = match(d_xyxy, d_cls, d_score, lab.xyxy, g_cls, g_ignore, class_aware=False)
        found = agnostic.det_gt >= 0
        scale = imgsz / max(lab.width, lab.height)
        cols["det_result"].append(m.det_result)
        cols["det_score"].append(d_score)
        cols["det_cls"].append(d_cls)
        cols["det_height"].append((d_xyxy[:, 3] - d_xyxy[:, 1]) * scale)
        cols["det_group"].append(np.full(len(d_score), lab.group))
        cols["gt_score"].append(m.gt_score)
        cols["gt_cls"].append(g_cls)
        cols["gt_ignore"].append(g_ignore)
        cols["gt_height"].append((lab.xyxy[:, 3] - lab.xyxy[:, 1]) * scale)
        cols["gt_group"].append(np.full(len(g_cls), lab.group))
        cols["gt_distance"].append(lab.distance)
        cols["gt_tags"].append(lab.tags)
        cols["pairs"].append(np.stack([g_cls[agnostic.det_gt[found]], d_cls[found], d_score[found]], axis=1))
    joined = {k: np.concatenate(v) if k != "gt_tags" else [t for tags in v for t in tags] for k, v in cols.items()}
    joined["pairs"] = joined["pairs"].reshape(-1, 3)
    return Matched(**joined)


def best_threshold(m: Matched) -> float:
    p, r = precision_recall(m.det_result, m.det_score, m.n_gt, THRESHOLDS)
    return float(THRESHOLDS[int(np.argmax(f1(p, r)))])


def _prf(det_result, det_score, gt_score, gt_ignore, t: float) -> dict:
    """Precision, recall and F1 at threshold ``t`` for a subset of detections and ground truth boxes."""
    kept = (det_score >= t) & (det_result != IGNORED)
    tp = int((det_result[kept] == TP).sum())
    n_det = int(kept.sum())
    n_gt = int((~gt_ignore).sum())
    found = int(((gt_score >= t) & ~gt_ignore).sum())
    p = tp / n_det if n_det else None
    r = found / n_gt if n_gt else None
    return {"precision": p, "recall": r, "f1": float(f1(p, r)) if p is not None and r is not None else None,
            "detections": n_det, "cones": n_gt}


def _recall(gt_score, gt_ignore, mask, t: float) -> dict:
    g = mask & ~gt_ignore
    return {"recall": float((gt_score[g] >= t).sum() / g.sum()) if g.sum() else None, "cones": int(g.sum())}


def summary_at(m: Matched, t: float, names: dict[int, str]) -> dict:
    out = {"all": _prf(m.det_result, m.det_score, m.gt_score, m.gt_ignore, t)}
    for c, name in names.items():
        d, g = m.det_cls == c, m.gt_cls == c
        out[name] = _prf(m.det_result[d], m.det_score[d], m.gt_score[g], m.gt_ignore[g], t)
    pairs = m.pairs[m.pairs[:, 2] >= t]
    confusion = np.zeros((len(names), len(names)), int)
    index = {c: i for i, c in enumerate(names)}
    for g, d, _ in pairs:
        if int(g) in index and int(d) in index:
            confusion[index[int(g)], index[int(d)]] += 1
    out["colour_accuracy"] = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else None
    out["confusion"] = {"rows_are_ground_truth": list(names.values()), "counts": confusion.tolist()}
    return out


def breakdowns(m: Matched, t: float) -> dict:
    by_height = []
    for lo, hi in pairwise(HEIGHT_BINS):
        g = (m.gt_height >= lo) & (m.gt_height < hi)
        d = (m.det_height >= lo) & (m.det_height < hi)
        stats = _prf(m.det_result[d], m.det_score[d], m.gt_score[g], m.gt_ignore[g], t)
        by_height.append({"height_px": [lo, None if np.isinf(hi) else hi], **stats})
    by_group = {}
    for group in sorted(set(m.gt_group)):
        g, d = m.gt_group == group, m.det_group == group
        by_group[group] = _prf(m.det_result[d], m.det_score[d], m.gt_score[g], m.gt_ignore[g], t)
    present = {tag for tags in m.gt_tags for tag in tags}
    by_tag = {}
    for tag in [t for t in TAGS if t in present] + [None]:
        mask = np.array([(tag in tags) if tag else not tags for tags in m.gt_tags], bool)
        by_tag[tag or "no_tag"] = _recall(m.gt_score, m.gt_ignore, mask, t)
    out = {"by_height": by_height, "by_group": by_group, "by_tag": by_tag}
    if np.isfinite(m.gt_distance).any():
        out["by_distance"] = [{"distance_m": [lo, None if np.isinf(hi) else hi],
                               **_recall(m.gt_score, m.gt_ignore, (m.gt_distance >= lo) & (m.gt_distance < hi), t)}
                              for lo, hi in pairwise(DISTANCE_BINS)]
    return out


def coco_metrics(model, data_yaml: Path, split: str, imgsz: int, device, project: Path, name: str) -> dict:
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=8, device=device,
                  quantize=_precision(device), plots=True, project=str(project), name=name, exist_ok=True,
                  verbose=False)
    per_class = {CLASS_NAMES[int(c)]: {"ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i])}
                 for i, c in enumerate(m.box.ap_class_index)}
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class}


def latency(model, image: Path, device, size: tuple[int, int] = (1280, 960), runs: int = 200) -> dict:
    """Median milliseconds per image, on one image resized to ``size`` (the gotthard.json cameras by default)."""
    import cv2

    im = cv2.resize(cv2.imread(str(image)), size)
    kwargs = {"imgsz": max(size), "device": device, "quantize": _precision(device), "verbose": False}
    for _ in range(20):
        model.predict(im, **kwargs)
    speed = [model.predict(im, **kwargs)[0].speed for _ in range(runs)]
    out = {k: float(np.median([s[k] for s in speed])) for k in ("preprocess", "inference", "postprocess")}
    out["total"] = sum(out.values())
    out["image_size"] = list(size)
    return out


def evaluate(weights: Path, dataset: Path, split: str = "test", imgsz: int = 1280, device=0,
             limit: int | None = None, project: Path | None = None) -> dict:
    """Runs everything listed in the module docstring for one model."""
    from ultralytics import YOLO

    name = dataset_name(dataset)
    model = YOLO(weights)
    val_labels = load_labels(dataset, "val", limit)
    test_labels = load_labels(dataset, split, limit)
    val = match_split(val_labels, predict(model, dataset, "val", val_labels, imgsz, device), imgsz)
    detections = predict(model, dataset, split, test_labels, imgsz, device)
    test = match_split(test_labels, detections, imgsz)
    paper = match_split(test_labels, detections, imgsz, paper=True)
    t = best_threshold(val)

    results = {
        "model": Path(weights).stem,
        "dataset": name,
        "split": split,
        "imgsz": imgsz,
        "images": len(test_labels),
        "threshold": t,
        "val_f1_at_threshold": _prf(val.det_result, val.det_score, val.gt_score, val.gt_ignore, t)["f1"],
        "test": summary_at(test, t, CLASS_NAMES),
        "paper": summary_at(paper, t, PAPER_CLASSES),
        **breakdowns(test, t),
    }
    if limit is None:
        if project is None:
            raise ValueError("project is needed for ultralytics' validation plots")
        results["coco"] = coco_metrics(model, dataset / f"{name}.yaml", split, imgsz, device, project,
                                       f"{name}_{split}")
    results["latency"] = latency(model, dataset / "images" / split / f"{test_labels[0].name}.jpg", device)
    return _rounded(results)


def _rounded(value, digits: int = 4):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    return value


# Report

INTRO = {
    "fsoco": [
        "# Cone detection on FSOCO",
        "",
        "Written by `scripts/evaluate_detector.py`. The test split holds the images from AMZ's own folder and from",
        "FRT, cameras the models never saw during training (see `configs/detection/fsoco_split.yaml`).",
    ],
    "fsds": [
        "# Cone detection on simulator images",
        "",
        "Written by `scripts/evaluate_detector.py`. Simulator images labelled from the ground truth (see",
        "`docs/detection.md`), tested on CompetitionMap3 at dusk, a map no model saw during training (see",
        "`configs/detection/fsds_split.yaml`). The threshold comes from val, CompetitionMapTestday2 in fog.",
    ],
}
GROUPS = {"fsoco": "camera origin", "fsds": "map"}


def _fmt(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def _range(bounds: list, unit: str) -> str:
    lo, hi = bounds
    return f"{lo}-{hi} {unit}" if hi is not None else f"{lo}+ {unit}"


def report_markdown(results: list[dict]) -> str:
    """A comparison of several models on one dataset as markdown, from the dictionaries ``evaluate()`` returns."""
    results = sorted(results, key=lambda r: r["model"])
    kind = results[0].get("dataset", "fsoco")
    lines = INTRO[kind] + [
        "Precision and recall use one confidence threshold per model, the one with the best F1 on val. A",
        "detection counts if its box overlaps a labelled cone of the same class with an IoU of at least 0.5.",
        "",
        "## Overall",
        "",
    ]
    rows = []
    for r in results:
        t, coco, lat = r["test"]["all"], r.get("coco", {}), r.get("latency", {})
        rows.append([r["model"], _fmt(r["threshold"]), _fmt(t["precision"]), _fmt(t["recall"]), _fmt(t["f1"]),
                     _fmt(r["test"]["colour_accuracy"], 3), _fmt(coco.get("map50"), 3), _fmt(coco.get("map50_95"), 3),
                     _fmt(lat.get("total"), 1)])
    lines += _table(["model", "threshold", "precision", "recall", "F1", "colour accuracy", "mAP50", "mAP50-95",
                     "ms per image"], rows)
    lines += [
        "Colour accuracy is the share of found cones that got the right class. Latency is the median over 200 runs",
        "on one 1280x960 image (the size of the gotthard.json cameras) with PyTorch in half precision, including",
        "pre- and postprocessing.",
        "",
        "## Compared with Dhall et al. (2019)",
        "",
        "Their YOLOv2 had three classes, so here both orange classes are merged and cones labelled as unknown are",
        "ignored. Their test set was their own, so this is only a rough comparison.",
        "",
    ]
    rows = [[r["model"], _fmt(r["paper"]["all"]["precision"]), _fmt(r["paper"]["all"]["recall"])] for r in results]
    rows.append(["Dhall et al. 2019, YOLOv2", "0.84", "0.71"])
    lines += _table(["model", "precision", "recall"], rows)

    lines += ["## Per class", "", "AP50 from ultralytics, recall at the chosen threshold.", ""]
    names = [n for n in CLASS_NAMES.values() if any(r["test"][n]["cones"] for r in results)]
    rows = []
    for r in results:
        per_class = r.get("coco", {}).get("per_class", {})
        rows.append([r["model"]] + [f"{_fmt(per_class.get(n, {}).get('ap50'))} / {_fmt(r['test'][n]['recall'])}"
                                    for n in names])
    lines += _table(["model"] + names, rows)

    if "by_distance" in results[0]:
        lines += ["## Recall by distance", "", "Distance from the camera to the cone.", ""]
        bins = results[0]["by_distance"]
        rows = [[r["model"]] + [_fmt(b["recall"]) for b in r["by_distance"]] for r in results]
        rows.append(["cones"] + [str(b["cones"]) for b in bins])
        lines += _table(["model"] + [_range(b["distance_m"], "m") for b in bins], rows)

    lines += ["## Recall by box height", "", "Height of the labelled box in pixels at the network input.", ""]
    bins = results[0]["by_height"]
    rows = [[r["model"]] + [_fmt(b["recall"]) for b in r["by_height"]] for r in results]
    rows.append(["cones"] + [str(b["cones"]) for b in bins])
    lines += _table(["model"] + [_range(b["height_px"], "px") for b in bins], rows)

    lines += [f"## By {GROUPS[kind]}", "", "Precision / recall.", ""]
    groups = sorted(results[0]["by_group"])
    rows = [[r["model"]] + [f"{_fmt(r['by_group'][g]['precision'])} / {_fmt(r['by_group'][g]['recall'])}"
                            for g in groups] for r in results]
    lines += _table(["model"] + groups, rows)

    lines += ["## Recall by tag", ""]
    tags = list(results[0]["by_tag"])
    rows = [[r["model"]] + [_fmt(r["by_tag"][t]["recall"]) for t in tags] for r in results]
    rows.append(["cones"] + [str(results[0]["by_tag"][t]["cones"]) for t in tags])
    lines += _table(["model"] + [t.replace("_", " ") for t in tags], rows)
    return "\n".join(lines).rstrip() + "\n"


def find_threshold(detector: str | Path, *report_dirs: Path) -> float:
    """The confidence threshold a detector's evaluation on the simulator images chose, from the first of
    ``report_dirs`` (each a ``reports``-like folder) that has ``detection/fsds/<detector>.json``. Raises if none
    has, because a silent default would make the numbers of a new model incomparable."""
    import json

    name = Path(detector).stem
    for directory in report_dirs:
        path = Path(directory) / "detection" / "fsds" / f"{name}.json"
        if path.exists():
            return float(json.loads(path.read_text())["threshold"])
    searched = ", ".join(str(Path(d) / "detection" / "fsds") for d in report_dirs)
    raise FileNotFoundError(f"no threshold for {name} in {searched}: evaluate the detector first "
                            f"(make eval-detector DETECTOR=...) or pass --conf")

