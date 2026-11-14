"""FSDS custom map CSV files, one cone per line: ``tag,x,y,direction,x_variance,y_variance,xy_covariance``.

Tags are ``blue``, ``yellow`` and ``big_orange``, the simulator ignores direction and variances. Coordinates are in
metres. The car spawns at the origin facing +x and /fsds/testing_only/track reports exactly these positions, so a
CSV map is also the ground truth. The start line goes between the big orange cones, there have to be 2 or 4 of them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common.types import ConeColor

_TAG_TO_COLOR = {"blue": ConeColor.BLUE, "yellow": ConeColor.YELLOW, "big_orange": ConeColor.ORANGE_BIG}
_COLOR_TO_TAG = {v: k for k, v in _TAG_TO_COLOR.items()}


def read_csv_map(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (N,2) xy [m] and (N,) ConeColor values."""
    xy, colors = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        tag, x, y = line.split(",")[:3]
        xy.append((float(x), float(y)))
        colors.append(_TAG_TO_COLOR[tag.strip()])
    return np.array(xy).reshape(-1, 2), np.array(colors, dtype=int)


def write_csv_map(path: str | Path, xy: np.ndarray, colors: np.ndarray) -> None:
    n_orange = int(np.sum(colors == ConeColor.ORANGE_BIG))
    if n_orange not in (2, 4):
        raise ValueError(f"FSDS needs exactly 2 or 4 big orange cones for the start line, got {n_orange}")
    lines = [f"{_COLOR_TO_TAG[ConeColor(c)]},{x:.6f},{y:.6f},0.0,0.0,0.0,0.0" for (x, y), c in zip(xy, colors)]
    Path(path).write_text("\n".join(lines) + "\n")
