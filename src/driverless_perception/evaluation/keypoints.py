"""How well the keypoint network hits the 7 points, broken down by keypoint, by distance and by cone colour.

The network works on an 80x80 crop, so its error is in crop pixels; what PnP feels is the error back in the image,
which is the crop error times the box size over 80. The cross ratio of the top and the three points down one edge
is the quantity the loss pulls towards ``cone_model.cross_ratio_3d()``, so how far the predictions land from it
says whether the network learned the shape of the cone rather than just the average cone.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from ..common.cone_model import cross_ratio_3d
from ..common.types import CLASS_NAMES
from ..vision.keypoints.crops import CROP, plain_crop
from ..vision.keypoints.cross_ratio import LEFT_EDGE, RIGHT_EDGE, cross_ratio

KEYPOINT_NAMES = ("top", "left stripe top", "left stripe bottom", "left bottom", "right stripe top",
                  "right stripe bottom", "right bottom")
DISTANCE_BINS = (0, 5, 10, 15, 20, 30)


def evaluate_keypoints(cache, model, device: str, batch: int = 256) -> dict:
    """Runs the network on every crop of ``cache`` and compares the 7 points with the true ones."""
    import torch

    from ..vision.keypoints.network import to_tensor

    crops, truths = [], []
    for i in range(len(cache)):
        crop, keypoints, _ = plain_crop(cache.image(i), cache.boxes[i], cache.keypoints[i])
        crops.append(crop)
        truths.append(keypoints)
    # the crop is the box resized to 80 px, so one crop pixel is this much of the box in the camera image
    boxes = np.asarray(cache.image_boxes, float).reshape(-1, 4)
    scale = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) / (CROP * np.sqrt(2))
    predictions = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = to_tensor(np.stack(crops[i:i + batch])).to(device)
            predictions.append(model(x).float().cpu().numpy())
    predicted = np.concatenate(predictions)
    truth = np.stack(truths)
    distance = np.asarray(cache.distances, float)
    classes = np.asarray(cache.classes, int)

    error = np.linalg.norm(predicted - truth, axis=-1)
    squared = (error ** 2).mean(axis=1)
    result = {
        "crops": len(crops),
        "mse_px2": float(squared.mean()),
        "median_px": float(np.median(error)),
        "median_image_px": float(np.median(error.mean(axis=1) * scale)),
        "per_keypoint": [{"index": i, "name": KEYPOINT_NAMES[i], "mse_px2": float((error[:, i] ** 2).mean()),
                          "median_px": float(np.median(error[:, i]))} for i in range(error.shape[1])],
        "by_distance": [],
        "by_class": [],
        "cross_ratio": {"model": cross_ratio_3d()},
    }
    for lo, hi in pairwise(DISTANCE_BINS):
        sel = (distance >= lo) & (distance < hi)
        row = {"distance_m": [lo, hi], "crops": int(sel.sum())}
        if sel.any():
            row.update(mse_px2=float(squared[sel].mean()), median_px=float(np.median(error[sel])),
                       median_image_px=float(np.median(error[sel].mean(axis=1) * scale[sel])),
                       box_height_px=float(np.median(boxes[sel, 3] - boxes[sel, 1])))
        result["by_distance"].append(row)
    for c in sorted(set(classes.tolist())):
        sel = classes == c
        result["by_class"].append({"class": CLASS_NAMES[c], "crops": int(sel.sum()),
                                   "mse_px2": float(squared[sel].mean()),
                                   "median_px": float(np.median(error[sel]))})
    for name, points in (("predicted", predicted), ("labels", truth)):
        ratios = np.concatenate([cross_ratio(points, LEFT_EDGE), cross_ratio(points, RIGHT_EDGE)])
        result["cross_ratio"][name] = {"median": float(np.median(ratios)),
                                       "deviation": float(np.median(np.abs(ratios - cross_ratio_3d())))}
    result["cross_ratio"]["predicted_values"] = np.concatenate(
        [cross_ratio(predicted, LEFT_EDGE), cross_ratio(predicted, RIGHT_EDGE)]).round(4).tolist()
    return result


def report_markdown(result: dict) -> str:
    """The keypoint network's error, per keypoint and per distance."""
    lines = [
        "## Keypoint network",
        "",
        f"On {result['crops']} test crops the mean squared error is {result['mse_px2']:.2f} px^2 in the 80x80 crop",
        f"(median {result['median_px']:.2f} px per point, {result['median_image_px']:.2f} px back in the camera",
        "image). Dhall et al. reached 3.78 px^2 on their test crops.",
        "",
        "| keypoint | " + " | ".join(k["name"] for k in result["per_keypoint"]) + " |",
        "|---|" + "---|" * len(result["per_keypoint"]),
        "| median error | " + " | ".join(f"{k['median_px']:.2f} px" for k in result["per_keypoint"]) + " |",
        "",
        "| distance | crops | box height | median error | in the image | MSE |",
        "|---|---|---|---|---|---|",
    ]
    for b in result["by_distance"]:
        if not b["crops"]:
            continue
        lines.append(f"| {b['distance_m'][0]}-{b['distance_m'][1]} m | {b['crops']} | {b['box_height_px']:.0f} px | "
                     f"{b['median_px']:.2f} px | {b['median_image_px']:.2f} px | {b['mse_px2']:.2f} px^2 |")
    cr = result["cross_ratio"]
    lines += [
        "",
        f"The cross ratio of the cone model is {cr['model']:.4f}. The network's keypoints give",
        f"{cr['predicted']['median']:.4f} (median, typically {cr['predicted']['deviation']:.4f} away from the",
        f"model), the true keypoints {cr['labels']['median']:.4f}.",
        "",
    ]
    return "\n".join(lines) + "\n"
