"""A cone map built from what a pipeline saw during a race, against the true track.

Every published cone is moved into the map frame with the ground truth pose at its stamp, the observations are
grouped by the cone they belong to, and each cone's position is the median of its observations. Gosala et al. build
the same map with their SLAM and report an RMS landmark error of 0.25 m from stereo only (Fig. 9b) and 0.20 m from
the LiDAR (Fig. 12). Ours uses the ground truth pose instead of SLAM, so it is a lower bound on what their system
would do with our observations.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..common.geometry import car_to_world, interpolate_pose
from ..common.types import CLASS_NAMES


def build_map(frames: list[tuple[int, np.ndarray, np.ndarray]], odom_times: np.ndarray, odom_poses: np.ndarray,
              track_xy: np.ndarray, colors: np.ndarray, gate: float = 1.0, min_observations: int = 3,
              max_range: float | None = None) -> dict:
    """``frames`` are (stamp, (N, 2) cones in the car frame, (N,) colours). Returns the mapped cones, how far each
    one is from the true one, and what was missed or invented."""
    observations = defaultdict(list)
    colours = defaultdict(list)
    spurious = []
    used = 0
    for stamp, cones, cone_colours in frames:
        if not odom_times[0] <= stamp <= odom_times[-1] or not len(cones):
            continue
        used += 1
        pose = interpolate_pose(stamp, odom_times, odom_poses)
        if max_range is not None:
            keep = np.hypot(*np.asarray(cones, float).reshape(-1, 2).T) <= max_range
            cones, cone_colours = np.asarray(cones)[keep], np.asarray(cone_colours)[keep]
        world = car_to_world(np.asarray(cones, float).reshape(-1, 2), pose)
        for point, colour in zip(world, cone_colours):
            distances = np.hypot(*(track_xy - point).T)
            index = int(np.argmin(distances))
            if distances[index] <= gate:
                observations[index].append(point)
                colours[index].append(int(colour))
            else:
                spurious.append(point)

    # the camera pipelines only report small cones, so only those count as landmarks
    expected = {i for i in range(len(track_xy)) if int(colors[i]) in (0, 1, 3)}
    cones, errors = [], []
    for index, points in sorted(observations.items()):
        if len(points) < min_observations or index not in expected:
            continue
        estimate = np.median(np.array(points), axis=0)
        error = float(np.hypot(*(estimate - track_xy[index])))
        seen = np.bincount(colours[index], minlength=5)
        cones.append({"cone": index, "observations": len(points), "x": float(estimate[0]), "y": float(estimate[1]),
                      "true_x": float(track_xy[index][0]), "true_y": float(track_xy[index][1]),
                      "error_m": error, "colour": CLASS_NAMES[int(np.argmax(seen))],
                      "true_colour": CLASS_NAMES[int(colors[index])],
                      "spread_m": float(np.median(np.hypot(*(np.array(points) - estimate).T)))})
        errors.append(error)
    errors = np.array(errors)
    seen_cones = {c["cone"] for c in cones}
    return {
        "frames": used,
        "gate_m": gate,
        "min_observations": min_observations,
        "mapped": len(cones),
        "expected": len(expected),
        "missed": len(expected - seen_cones),
        "spurious_observations": len(spurious),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
        "median_m": float(np.median(errors)) if len(errors) else None,
        "p90_m": float(np.percentile(errors, 90)) if len(errors) else None,
        "colour_accuracy": (float(np.mean([c["colour"] == c["true_colour"] for c in cones])) if cones else None),
        "cones": cones,
    }


def report_markdown(maps: dict[str, dict], title: str) -> str:
    """One row per pipeline, with the paper's numbers named next to ours."""
    fmt = lambda v, unit=" m": "-" if v is None else f"{v:.2f}{unit}"
    lines = [
        f"# {title}",
        "",
        "Written by `scripts/build_map.py`. Every cone a pipeline published is placed in the map frame with the",
        "ground truth pose at its stamp; a cone of the map is the median of the observations that belong to it.",
        "Gosala et al. report an RMS landmark error of 0.25 m from stereo and 0.20 m from the LiDAR, both built",
        "with their SLAM, so with the pose estimated rather than known.",
        "",
        "| pipeline | frames | cones mapped | missed | RMSE | median | 90% | colour | observations per cone |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, result in maps.items():
        per_cone = int(np.median([c["observations"] for c in result["cones"]])) if result["cones"] else 0
        colour = result["colour_accuracy"]
        lines.append(f"| {name} | {result['frames']} | {result['mapped']} of {result['expected']} | "
                     f"{result['missed']} | {fmt(result['rmse_m'])} | {fmt(result['median_m'])} | "
                     f"{fmt(result['p90_m'])} | {'-' if colour is None else f'{colour:.3f}'} | {per_cone} |")
    lines.append("")
    return "\n".join(lines) + "\n"
