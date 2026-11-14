"""Pictures for going through a dataset by hand.

Every label is drawn with the numbers behind it, and every candidate that was thrown away in grey with the reason,
so a box that looks wrong can be traced to its row in ``boxes.csv``. Keypoint crops get the seven true points and,
if a network is given, what it made of them.

The functions take the rows as they come out of the CSV, as dicts of strings, and a policy object with
``min_visible``, ``min_height`` and ``min_contrast`` (``sim.autolabel.LabelPolicy``), so nothing here has to know
how a dataset was built.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..common.types import CLASS_NAMES
from .draw import CONE_BGR, GREEN, RED, draw_points

GREY = (145, 145, 145)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# the seven keypoints in the order the network predicts them
KEYPOINT_NAMES = ("top", "left stripe top", "left stripe bottom", "left bottom", "right stripe top",
                  "right stripe bottom", "right bottom")


def read_boxes(path: str | Path, key: str = "image") -> dict[str, list[dict]]:
    """``boxes.csv`` as a dict from image name to its rows, in the order they were written."""
    rows: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row[key]].append(row)
    return dict(rows)


def box_of(row: dict) -> tuple[int, int, int, int]:
    return tuple(round(float(row[k])) for k in ("x1", "y1", "x2", "y2"))


def box_height(row: dict) -> int:
    box = box_of(row)
    return box[3] - box[1]


def is_labelled(row: dict) -> bool:
    """Rows of a dataset carry the policy's verdict, rows of anything else count as labels."""
    return row.get("labelled", "1") == "1"


def reject_reason(row: dict, policy) -> str:
    """Which threshold dropped this candidate, as a short text for the picture."""
    height = box_height(row)
    if height < policy.min_height:
        return f"h {height} px"
    if "visible" in row and float(row["visible"]) < policy.min_visible:
        return f"vis {float(row['visible']):.2f}"
    if "contrast" in row and float(row["contrast"]) < policy.min_contrast:
        return f"con {float(row['contrast']):.0f}"
    return "dropped"


def label_text(row: dict) -> list[str]:
    """One or two short lines for a labelled box, from whatever the dataset knows about it."""
    name = CLASS_NAMES.get(int(row["class"]), "cone")
    first = f"{name} {float(row['distance']):.1f} m" if row.get("distance") else name
    second = ""
    if row.get("visible") and row.get("contrast"):
        second = f"vis {float(row['visible']):.2f} con {float(row['contrast']):.0f}"
    elif row.get("tags", "").strip():
        second = row["tags"].strip()
    if row.get("truncated") == "1" and "truncated" not in second:
        second = (second + " truncated").strip()
    return [first] + ([second] if second else [])


def overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def put_text(image: np.ndarray, lines: list[str], x: int, y: int, colour, scale: float = 0.4,
             taken: list | None = None) -> bool:
    """Small text with a dark outline, so it stays readable on tarmac and on sky. With ``taken`` the text is
    skipped when it would sit on text that is already there, and its rectangle is added to the list."""
    step = round(16 * scale / 0.4)
    top = y if y - step * len(lines) >= 0 else y + step * (len(lines) + 1)
    left = max(x, 1)
    width = max(cv2.getTextSize(line, FONT, scale, 1)[0][0] for line in lines)
    rect = (left, top - step * len(lines), left + width, top + 3)
    if taken is not None:
        if any(overlap(rect, other) for other in taken):
            return False
        taken.append(rect)
    for i, line in enumerate(lines):
        at = (left, top - step * (len(lines) - 1 - i))
        cv2.putText(image, line, at, FONT, scale, BLACK, 3, cv2.LINE_AA)
        cv2.putText(image, line, at, FONT, scale, colour, 1, cv2.LINE_AA)
    return True


def annotate(image: np.ndarray, rows: list[dict], policy, min_text_height: int = 20) -> np.ndarray:
    """Draws the rows onto the image in place: labels in their cone colour with their numbers, rejected
    candidates thin and grey with the reason.

    Only boxes at least ``min_text_height`` high get text, and the near cones get it first: a text that would land
    on one that is already drawn is left out, so a crowded image stays readable."""
    labels = [r for r in rows if is_labelled(r)]
    dropped = [r for r in rows if not is_labelled(r)]
    for row in dropped:
        x1, y1, x2, y2 = box_of(row)
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), GREY, 1, cv2.LINE_AA)
    for row in labels:
        x1, y1, x2, y2 = box_of(row)
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), CONE_BGR.get(int(row["class"]), CONE_BGR[4]), 2,
                      cv2.LINE_AA)

    taken: list = []
    for row in sorted(labels, key=box_height, reverse=True):
        if box_height(row) >= min_text_height:
            x1, y1, _, _ = box_of(row)
            put_text(image, label_text(row), x1, y1 - 4, CONE_BGR.get(int(row["class"]), CONE_BGR[4]), taken=taken)
    for row in sorted(dropped, key=box_height, reverse=True):
        if box_height(row) >= min_text_height:
            x1, _, _, y2 = box_of(row)
            put_text(image, [reject_reason(row, policy)], x1, y2 + 14, GREY, 0.35, taken=taken)
    return image


def band(image: np.ndarray, rows: list[dict], margin: int = 30) -> np.ndarray:
    """The rows of the image cut down to the strip the boxes live in, which makes small cones visible."""
    if not rows:
        return image
    boxes = np.array([box_of(r) for r in rows], float)
    top = max(int(boxes[:, 1].min()) - margin, 0)
    bottom = min(int(boxes[:, 3].max()) + margin, image.shape[0])
    return image[top:bottom]


def yolo_rows(label_file: Path, width: int, height: int) -> list[dict]:
    """A YOLO txt as rows, for images whose dataset has no boxes.csv next to it."""
    if not label_file.exists():
        return []
    numbers = np.array(label_file.read_text().split(), float).reshape(-1, 5)
    return [{"class": str(int(c)), "x1": str((cx - bw / 2) * width), "y1": str((cy - bh / 2) * height),
             "x2": str((cx + bw / 2) * width), "y2": str((cy + bh / 2) * height)}
            for c, cx, cy, bw, bh in numbers]


def keypoint_panel(crop: np.ndarray, truth: np.ndarray, predicted: np.ndarray | None = None, scale: int = 4,
                   numbered: bool = True) -> np.ndarray:
    """One crop blown up, with the true keypoints as green dots and the network's as red crosses."""
    big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    draw_points(big, np.asarray(truth) * scale, GREEN, 3)
    if numbered:
        for i, (u, v) in enumerate(np.asarray(truth, float) * scale):
            y = round(v) - 4 if v > 16 else round(v) + 16
            cv2.putText(big, str(i + 1), (round(u) + 6, y), FONT, 0.4, GREEN, 1, cv2.LINE_AA)
    if predicted is not None:
        draw_points(big, np.asarray(predicted) * scale, RED, 4, cross=True)
    return big


def keypoint_error(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Mean distance between the true and the predicted points, in pixels of the crop."""
    return float(np.mean(np.linalg.norm(np.asarray(truth, float) - np.asarray(predicted, float), axis=1)))


def pick(items: list, count: int, seed: int = 0) -> list:
    """A deterministic sample that keeps the original order, so two runs show the same examples."""
    if count >= len(items):
        return list(items)
    chosen = random.Random(seed).sample(range(len(items)), count)
    return [items[i] for i in sorted(chosen)]


def spread(values: np.ndarray, count: int) -> np.ndarray:
    """Indices spread evenly over the sorted values, to cover the whole distance range instead of the crowd."""
    order = np.argsort(values)
    if count >= len(order):
        return order
    return order[np.linspace(0, len(order) - 1, count).astype(int)]


def write_index(directory: Path, title: str, command: str, entries: list[dict], notes: list[str] = ()) -> Path:
    """The index.md next to the pictures: what was drawn, from where, and the command that made it."""
    lines = [f"# {title}", "", "Written by", "", f"    {command}", ""]
    lines += [*notes, ""] if notes else []
    if entries:
        columns = list(entries[0])
        lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        lines += ["| " + " | ".join(str(e[c]) for c in columns) + " |" for e in entries]
        lines.append("")
    path = directory / "index.md"
    path.write_text("\n".join(lines))
    return path
