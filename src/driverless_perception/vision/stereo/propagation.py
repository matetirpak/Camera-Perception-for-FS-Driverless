"""Moving a cone's box from the left into the right image (Gosala et al., Sec. III-B.4).

The left camera's PnP estimate says where the cone stands. Projecting a cylinder of the cone's size at that spot
with the right camera's pose gives its box in the right image. PnP is least sure about the distance, which moves the
cone sideways in the right image, so the box is widened by a margin.
"""
from __future__ import annotations

import numpy as np

from ...common.cameras import Intrinsics, world_to_optical
from ...common.cone_model import TOP

# the base plate reaches 0.16 m from the axis
RADIUS = 0.16


def propagate_box(position_world: np.ndarray, ground_z: float, right_position, right_orientation,
                  intrinsics: Intrinsics, margin: float = 0.3) -> np.ndarray | None:
    """(x1, y1, x2, y2) search box in the right image for a cone at ``position_world``, widened on all sides by
    ``margin`` of its width, clipped to the image. None if the cone isn't in front of the camera or not in the
    image. A cone that is really further away sits further right and higher up."""
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.stack([RADIUS * np.cos(a), RADIUS * np.sin(a), np.zeros_like(a)], axis=1)
    base = np.array([position_world[0], position_world[1], ground_z])
    pts = world_to_optical(np.vstack([base + ring, base + ring + [0, 0, TOP]]), right_position, right_orientation)
    if pts[:, 2].min() < 0.5:
        return None
    uv = intrinsics.project(pts)
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    pad = margin * (x2 - x1)
    box = np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad])
    box = np.clip(box, 0, [intrinsics.width, intrinsics.height, intrinsics.width, intrinsics.height])
    return box if box[2] - box[0] >= 4 and box[3] - box[1] >= 4 else None
