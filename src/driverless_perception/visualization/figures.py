"""Figures of the offline results as PNG bytes, from the report JSON files, training logs and datasets."""
from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from ..common.types import CONE_RGB
from ..evaluation.latency import MONO_STAGES, precision_label

MODEL_COLOURS = {"yolo26s_fsoco": "#8d99ae", "yolo26s_fsoco_fsds": "#3a86ff"}
MODEL_NAMES = {"yolo26s_fsoco": "FSOCO model", "yolo26s_fsoco_fsds": "fine-tuned on sim"}
MONO, STEREO = "#e76f51", "#2a9d8f"
CLASS_COLOURS = {int(c): "#{:02x}{:02x}{:02x}".format(*(round(255 * v) for v in rgb))
                 for c, rgb in CONE_RGB.items()}
CLASS_NAMES = {0: "blue", 1: "yellow", 2: "big orange", 3: "small orange", 4: "unknown"}
SPLIT_COLOURS = {"train": "#adb5bd", "val": "#ffb703", "test": "#e63946"}


def _figure(rows: int, cols: int, title: str, size=(15, 8)):
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    # the system's matplotlib (from ROS) is visible next to the venv's, only its 3D axes complain
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return fig, axes


def _png(fig) -> bytes:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def _centres(bins: list[dict], key: str) -> np.ndarray:
    """Middle of each bin, open or very wide bins count as 20 wide."""
    out = []
    for b in bins:
        lo, hi = b[key]
        out.append((lo + (lo + 20 if hi is None or hi > lo + 20 else hi)) / 2)
    return np.array(out)


def _label(edges, unit: str = "") -> str:
    lo, hi = edges
    return f"{lo:g}-{hi:g}{unit}" if hi is not None else f"{lo:g}+{unit}"


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def training_curves(detector_runs: dict[str, Path], keypoints: Path | None = None) -> bytes:
    """Loss and validation mAP per epoch of each detector run (results.csv), and the keypoint network's error."""
    fig, axes = _figure(1, 3, "Training")
    for name, path in detector_runs.items():
        r = _read_csv(path)
        c = MODEL_COLOURS.get(name, "#444444")
        axes[0, 0].plot(r["epoch"], r["train/box_loss"], color=c, label=f"{MODEL_NAMES.get(name, name)} train")
        axes[0, 0].plot(r["epoch"], r["val/box_loss"], color=c, ls="--", label=f"{MODEL_NAMES.get(name, name)} val")
        axes[0, 1].plot(r["epoch"], r["metrics/mAP50-95(B)"], color=c, marker="o", ms=3,
                        label=MODEL_NAMES.get(name, name))
    axes[0, 0].set(title="Detector box loss", xlabel="epoch")
    axes[0, 1].set(title="Detector mAP50-95 on its val set", xlabel="epoch")
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].legend(fontsize=9)
    ax = axes[0, 2]
    if keypoints is not None and keypoints.exists():
        r = _read_csv(keypoints)
        ax.plot(r["epoch"], r["val_rmse_px"], color="#6a4c93", label="val RMSE")
        ax.plot(r["epoch"], r["val_median_px"], color="#6a4c93", ls="--", label="val median")
        ax.set_yscale("log")
        ax.set(title="Keypoint network, error in the 80x80 crop", xlabel="epoch", ylabel="px")
        lr = ax.twinx()
        lr.step(r["epoch"], r["lr"], color="#999999", lw=1)
        lr.set_yscale("log")
        lr.set_ylabel("learning rate", color="#777777")
        lr.grid(False)
        ax.legend(fontsize=9)
    return _png(fig)


def detection_summary(reports: dict[tuple[str, str], dict]) -> bytes:
    """Test results of the detectors, ``reports`` keyed by (dataset, model) as in reports/detection/."""
    fig, axes = _figure(2, 2, "Detection on the test sets", size=(15, 10))
    names = ["precision", "recall", "mAP50", "mAP50-95", "colour"]
    models = sorted({m for _, m in reports}, key=list(MODEL_COLOURS).index)
    for ax, dataset, title in ((axes[0, 0], "fsoco", "FSOCO test (AMZ and FRT cameras)"),
                               (axes[0, 1], "fsds", "Simulator test (CompetitionMap3, dusk)")):
        for j, model in enumerate(models):
            r = reports.get((dataset, model))
            if r is None:
                continue
            values = [r["test"]["all"]["precision"], r["test"]["all"]["recall"], r["coco"]["map50"],
                      r["coco"]["map50_95"], r["test"]["colour_accuracy"]]
            x = np.arange(len(names)) + (j - (len(models) - 1) / 2) * 0.38
            bars = ax.bar(x, values, 0.36, color=MODEL_COLOURS[model], label=MODEL_NAMES[model])
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(np.arange(len(names)), names)
        ax.set(title=title, ylim=(0, 1.22))
        ax.legend(fontsize=9, loc="upper center", ncol=2)
    ax = axes[1, 0]
    for model in models:
        r = reports.get(("fsds", model))
        if r and r.get("by_distance"):
            bins = [b for b in r["by_distance"] if b["cones"]]
            ax.plot(_centres(bins, "distance_m"), [b["recall"] for b in bins], marker="o",
                    color=MODEL_COLOURS[model], label=MODEL_NAMES[model])
            for b, x in zip(bins, _centres(bins, "distance_m")):
                ax.annotate(_label(b["distance_m"], " m"), (x, 0.02), fontsize=8, ha="center", color="#666666")
    ax.set(title="Simulator test: recall by distance", xlabel="distance [m]", ylabel="recall", ylim=(0, 1.05))
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    for (dataset, model), r in sorted(reports.items()):
        bins = [b for b in r["by_height"] if b["cones"]]
        labels = [_label(b["height_px"]) for b in bins]
        ax.plot(range(len(bins)), [b["recall"] for b in bins], marker="o", color=MODEL_COLOURS[model],
                ls="-" if dataset == "fsds" else ":", label=f"{MODEL_NAMES[model]} on {dataset.upper()}")
        ax.set_xticks(range(len(bins)), labels)
    ax.set(title="Recall by box height", xlabel="box height [px]", ylabel="recall", ylim=(0, 1.05))
    ax.legend(fontsize=9)
    return _png(fig)


def dataset_overview(fsoco_boxes: Path, fsds_boxes: Path) -> bytes:
    """What the two datasets hold: FSOCO images per camera origin and split, boxes per class, simulator labels per
    map and by distance."""
    fig, axes = _figure(2, 2, "Datasets", size=(15, 10))
    images, classes = defaultdict(set), Counter()
    with open(fsoco_boxes, newline="") as f:
        for row in csv.DictReader(f):
            images[(row["origin"], row["split"])].add(row["image"])
            classes[(row["split"], int(row["class"]))] += 1
    origins = sorted({o for o, _ in images}, key=lambda o: -sum(len(images[(o, s)]) for s in SPLIT_COLOURS))
    ax = axes[0, 0]
    left = np.zeros(len(origins))
    for split, c in SPLIT_COLOURS.items():
        n = np.array([len(images[(o, split)]) for o in origins])
        ax.barh(origins, n, left=left, color=c, label=split)
        left += n
    ax.invert_yaxis()
    ax.set(title="FSOCO: images per camera origin", xlabel="images")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for j, split in enumerate(SPLIT_COLOURS):
        n = [classes[(split, c)] for c in CLASS_NAMES]
        ax.bar(np.arange(5) + (j - 1) * 0.27, n, 0.26, color=SPLIT_COLOURS[split], label=split)
    ax.set_xticks(range(5), list(CLASS_NAMES.values()))
    ax.set(title="FSOCO: boxes per class", yscale="log")
    ax.legend(fontsize=9)

    per_map, distance, split_of = Counter(), defaultdict(list), {}
    with open(fsds_boxes, newline="") as f:
        for row in csv.DictReader(f):
            if row["labelled"] != "1":
                continue
            per_map[(row["map"], int(row["class"]))] += 1
            distance[int(row["class"])].append(float(row["distance"]))
            split_of[row["map"]] = row["split"]
    maps = sorted(split_of, key=lambda m: (list(SPLIT_COLOURS).index(split_of[m]), m))
    ax = axes[1, 0]
    bottom = np.zeros(len(maps))
    for c in (0, 1, 2):
        n = np.array([per_map[(m, c)] for m in maps])
        ax.bar(range(len(maps)), n, bottom=bottom, color=CLASS_COLOURS[c], label=CLASS_NAMES[c])
        bottom += n
    ax.set_xticks(range(len(maps)), [f"{m.replace('CompetitionMap', 'Comp')}\n({split_of[m]})" for m in maps],
                  fontsize=8)
    ax.set(title="Simulator: labels per map")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    for c in (0, 1, 2):
        ax.hist(distance[c], bins=np.arange(0, 82, 2), histtype="step", lw=2, color=CLASS_COLOURS[c],
                label=CLASS_NAMES[c])
    ax.set(title="Simulator: labels by distance", xlabel="distance [m]", ylabel="labels")
    ax.legend(fontsize=9)
    return _png(fig)


def mono_errors(report: dict) -> bytes:
    """Position error of the mono pipeline per distance for its three variants (reports/mono/fsds.json)."""
    fig, axes = _figure(1, 2, "Mono pipeline on the simulator test map", size=(15, 6))
    colours = {"exact": "#adb5bd", "boxes": "#f4a261", "detector": MONO}
    titles = {"exact": "exact keypoints", "boxes": "true boxes, keypoint network", "detector": "whole pipeline"}
    for variant, bins in report["bins"].items():
        bins = [b for b in bins if b.get("median_m") is not None]
        x = _centres(bins, "distance_m")
        axes[0, 0].plot(x, [b["median_m"] for b in bins], marker="o", color=colours[variant],
                        label=f"{titles[variant]} median")
        axes[0, 0].plot(x, [b["p90_m"] for b in bins], ls="--", color=colours[variant], label=f"{titles[variant]} 90%")
        axes[0, 1].plot(x, [b["range_bias_m"] for b in bins], marker="o", color=colours[variant],
                        label=titles[variant])
    axes[0, 0].plot([10, 16], [0.5, 1.0], "k*", ms=12, label="Dhall et al. (real data)")
    axes[0, 0].set(title="Horizontal position error", xlabel="distance [m]", ylabel="error [m]")
    axes[0, 1].axhline(0, color="k", lw=0.8)
    axes[0, 1].set(title="Range bias (negative: too close)", xlabel="distance [m]", ylabel="median range error [m]")
    for ax in axes[0]:
        ax.legend(fontsize=9)
    return _png(fig)


def stereo_vs_mono(report: dict) -> bytes:
    """Stereo against PnP on the same cones (reports/stereo/fsds.json)."""
    fig, axes = _figure(1, 3, "Stereo against mono PnP on the simulator test map (whole pipeline)", size=(16, 6))
    bins = [b for b in report["bins"]["detector"] if "mono" in b]
    x = _centres(bins, "distance_m")
    for key, colour, name in (("mono", MONO, "PnP"), ("stereo", STEREO, "stereo")):
        axes[0, 0].plot(x, [b[key]["median_m"] for b in bins], marker="o", color=colour, label=f"{name} median")
        axes[0, 0].plot(x, [b[key]["p90_m"] for b in bins], ls="--", color=colour, label=f"{name} 90%")
        axes[0, 1].plot(x, [b[key]["range_spread_m"] for b in bins], marker="o", color=colour,
                        label=f"{name} spread")
        axes[0, 1].plot(x, [b[key]["range_bias_m"] for b in bins], ls=":", marker="s", color=colour,
                        label=f"{name} bias")
    axes[0, 0].set(title="Horizontal position error", xlabel="distance [m]", ylabel="error [m]")
    axes[0, 1].axhline(0, color="k", lw=0.8)
    axes[0, 1].set(title="Range error: spread (std) and bias (median)", xlabel="distance [m]", ylabel="[m]")
    found = [b["with_stereo"] / max(b["with_pnp"], 1) for b in report["bins"]["detector"]]
    labels = [_label(b["distance_m"], " m") for b in report["bins"]["detector"]]
    bars = axes[0, 2].bar(labels, found, color=STEREO)
    axes[0, 2].bar_label(bars, [f"{f:.0%}\n{b.get('matches_median', 0):.0f} matches"
                                for f, b in zip(found, report["bins"]["detector"])], fontsize=8)
    axes[0, 2].set(title="Cones that got a stereo position", ylim=(0, 1.2))
    for ax in axes[0, :2]:
        ax.legend(fontsize=9)
    return _png(fig)


def race_scores(mono: dict, stereo: dict | None) -> bytes:
    """Recall, precision and position error by range and by lap in the 10 lap race (the *_metrics.json files)."""
    fig, axes = _figure(1, 3, "The 10 lap race on CompetitionMap3", size=(16, 6))
    runs = [("mono", mono, MONO)] + ([("stereo", stereo, STEREO)] if stereo else [])
    for name, r, colour in runs:
        bins = [b for b in r["bins"] if b["n_gt"]]
        x = _centres(bins, "bin_m")
        axes[0, 0].plot(x, [b["recall"] for b in bins], marker="o", color=colour, label=f"{name} recall")
        axes[0, 0].plot(x, [b["precision"] for b in bins], marker="s", ls="--", color=colour,
                        label=f"{name} precision")
        axes[0, 1].plot(x, [b["pos_rmse_m"] for b in bins], marker="o", color=colour, label=name)
        laps = r.get("laps", [])
        axes[0, 2].plot([lap["lap"] for lap in laps], [lap["recall"] for lap in laps], marker="o", color=colour,
                        label=f"{name} recall")
        axes[0, 2].plot([lap["lap"] for lap in laps], [lap["precision"] for lap in laps], marker="s", ls="--",
                        color=colour, label=f"{name} precision")
    axes[0, 0].set(title="By range", xlabel="range [m]", ylim=(0, 1.05))
    axes[0, 1].set(title="Position RMSE", xlabel="range [m]", ylabel="[m]")
    axes[0, 2].set(title="By lap", xlabel="lap", ylim=(0.7, 1.02), xticks=range(1, 11))
    for ax in axes[0]:
        ax.legend(fontsize=9)
    return _png(fig)


def pairing(gaps_ms: np.ndarray, errors: dict[str, list[tuple[float, float]]],
            bins=(0, 5, 12, 20, 35, 50)) -> bytes:
    """Stamp gap between each left and the nearest right image, and the stereo error per gap with each image at
    its own stamp's pose or both at the left one's. ``errors`` maps a policy to (gap in ms, error in m) pairs."""
    fig, axes = _figure(1, 2, "Pairing the left and right images of a race", size=(15, 6))
    ax = axes[0, 0]
    ax.hist(np.clip(gaps_ms, 0, 80), bins=np.arange(0, 82, 2), color="#577590")
    ax.axvline(25, color="k", ls="--")
    ax.text(26, ax.get_ylim()[1] * 0.9, "same render below 25 ms", fontsize=9)
    ax.set(title="Gap to the nearest right image", xlabel="stamp gap [ms]", ylabel="left images")
    ax = axes[0, 1]
    labels = [f"{lo}-{hi} ms" for lo, hi in pairwise(bins)]
    colours = {"own stamps": "#bc4749", "one pose": STEREO}
    for j, (policy, pairs) in enumerate(errors.items()):
        pairs = np.array(pairs, float).reshape(-1, 2)
        rmse = []
        for lo, hi in pairwise(bins):
            e = pairs[(pairs[:, 0] >= lo) & (pairs[:, 0] < hi), 1]
            rmse.append(np.sqrt(np.mean(e ** 2)) if len(e) else 0.0)
        bars = ax.bar(np.arange(len(labels)) + (j - 0.5) * 0.4, rmse, 0.38, color=colours.get(policy), label=policy)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set(title="Stereo position RMSE by stamp gap", ylabel="RMSE [m]")
    ax.legend(fontsize=9)
    return _png(fig)


def depth_spread(result: dict) -> bytes:
    """The standing depth test (Gosala et al., Fig. 9a): every measurement of each cone, and how much the estimates
    of one cone scatter."""
    fig, axes = _figure(1, 2, "Standing depth test: cones at known distances, measured every frame", size=(15, 6))
    colours = {"mono camera PnP": "#c99a86", "left camera PnP": MONO, "stereo": STEREO}
    offsets = {"mono camera PnP": -0.45, "left camera PnP": 0.0, "stereo": 0.45}
    ax = axes[0, 0]
    truth = [c["distance_m"] for c in result["cones"]]
    ax.plot([0, max(truth) + 2], [0, max(truth) + 2], color="#999999", ls="--", lw=1, label="true distance")
    for series, colour in colours.items():
        data = [c["series"][series] for c in result["cones"] if series in c["series"]]
        positions = [c["distance_m"] + offsets[series] for c in result["cones"] if series in c["series"]]
        if not data:
            continue
        box = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True, manage_ticks=False,
                         flierprops={"marker": ".", "markersize": 3, "markerfacecolor": colour,
                                     "markeredgecolor": colour})
        for part in box["boxes"]:
            part.set(facecolor=colour, alpha=0.65, edgecolor=colour)
        for key in ("whiskers", "caps", "medians"):
            for part in box[key]:
                part.set(color="#333333" if key == "medians" else colour)
        ax.plot([], [], color=colour, lw=6, alpha=0.65, label=series)
    ax.set(title="Estimated distance per cone", xlabel="true distance [m]", ylabel="estimated distance [m]",
           xticks=[round(t) for t in truth])
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for series, colour in colours.items():
        points = sorted((c["distance_m"], c["stats"][series]["std_m"]) for c in result["cones"]
                        if series in c["stats"])
        if points:
            ax.plot(*zip(*points), marker="o", color=colour, label=series)
    ax.set(title="Spread of the measurements of one cone", xlabel="true distance [m]",
           ylabel="standard deviation [m]", yscale="log")
    ax.legend(fontsize=9)
    return _png(fig)


def depth_scatter(samples: dict[str, list], bins=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20)) -> bytes:
    """How far the estimated distance is off while driving, grouped by true distance: the moving version of the
    standing depth test. ``samples`` maps a name to (true distance, estimated distance) pairs. Inside a bin the
    cones stand at different distances, so the error is plotted, not the estimate itself."""
    fig, axes = _figure(1, 2, "Distance error while driving, by true distance", size=(15, 6))
    colours = {"PnP": MONO, "stereo": STEREO}
    offsets = {"PnP": -0.35, "stereo": 0.35}
    centres = [(lo + hi) / 2 for lo, hi in pairwise(bins)]
    ax = axes[0, 0]
    ax.axhline(0, color="#999999", ls="--", lw=1)
    for name, rows in samples.items():
        rows = np.array(rows, float).reshape(-1, 2)
        grouped, positions = [], []
        for (lo, hi), centre in zip(pairwise(bins), centres):
            sel = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
            if len(sel) >= 10:
                grouped.append(sel[:, 1] - sel[:, 0])
                positions.append(centre + offsets.get(name, 0.0))
        if not grouped:
            continue
        colour = colours.get(name, "#777777")
        box = ax.boxplot(grouped, positions=positions, widths=0.6, patch_artist=True, manage_ticks=False,
                         flierprops={"marker": ".", "markersize": 2, "markerfacecolor": colour,
                                     "markeredgecolor": colour, "alpha": 0.4})
        for part in box["boxes"]:
            part.set(facecolor=colour, alpha=0.65, edgecolor=colour)
        for key in ("whiskers", "caps", "medians"):
            for part in box[key]:
                part.set(color="#333333" if key == "medians" else colour)
        ax.plot([], [], color=colour, lw=6, alpha=0.65, label=name)
        axes[0, 1].plot(positions, [float(np.std(g)) for g in grouped], marker="o", color=colour, label=name)
    ax.set(title="Estimated minus true distance (box, whiskers and outliers)", xlabel="true distance [m]",
           ylabel="error [m]", xticks=centres)
    ax.legend(fontsize=9)
    axes[0, 1].set(title="Spread of the error in each bin", xlabel="true distance [m]",
                   ylabel="standard deviation [m]")
    axes[0, 1].legend(fontsize=9)
    return _png(fig)


def stereo_stages(stages: list[dict]) -> bytes:
    """What every step of the stereo pipeline did, per distance: propagation, matching and triangulation."""
    fig, axes = _figure(1, 3, "Stereo pipeline stage by stage (ground truth boxes)", size=(16, 6))
    bins = [b for b in stages if b["cones"]]
    x = _centres(bins, "distance_m")
    ax = axes[0, 0]
    ax.plot(x, [b.get("iou_median") for b in bins], marker="o", color=STEREO, label="IoU with the true box")
    ax.plot(x, [b.get("contains_share") for b in bins], marker="s", ls="--", color="#577590",
            label="holds the whole cone")
    ax.set(title="Propagated box in the right image", xlabel="distance [m]", ylim=(0, 1.1))
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for key, name, colour in (("features_median", "ORB features (smaller image)", "#8d99ae"),
                              ("ratio_test_median", "after the ratio test", "#577590"),
                              ("matches_median", "after the epipolar test", MONO),
                              ("on_cone_median", "on the cone", STEREO)):
        ax.plot(x, [b.get(key) for b in bins], marker="o", color=colour, label=name)
    ax.set(title="Features and matches per cone", xlabel="distance [m]", yscale="log")
    ax.legend(fontsize=9)
    ax = axes[0, 2]
    ax.plot(x, [b.get("point_error_median_m") for b in bins], marker="o", color=STEREO,
            label="one point, median")
    ax.plot(x, [b.get("point_error_p90_m") for b in bins], ls="--", color=STEREO, label="one point, 90%")
    ax.plot(x, [b.get("median_minus_mean_m") for b in bins], marker="s", color="#bc4749",
            label="median minus mean of the points")
    ax.set(title="Triangulated points against the true cone", xlabel="distance [m]", ylabel="[m]")
    ax.legend(fontsize=9)
    return _png(fig)


def keypoint_errors(result: dict) -> bytes:
    """Where the keypoint network is off: per keypoint, with distance, and how close its cross ratio stays to the
    cone model's."""
    fig, axes = _figure(1, 3, "Keypoint network on the test crops", size=(16, 6))
    ax = axes[0, 0]
    names = [k["name"] for k in result["per_keypoint"]]
    bars = ax.bar(range(len(names)), [k["median_px"] for k in result["per_keypoint"]], color="#6a4c93")
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(names)), [n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set(title="Median error per keypoint", ylabel="px in the 80x80 crop")
    ax = axes[0, 1]
    bins = [b for b in result["by_distance"] if b["crops"]]
    x = _centres(bins, "distance_m")
    ax.plot(x, [b["median_px"] for b in bins], marker="o", color="#6a4c93", label="in the crop [px]")
    ax.plot(x, [b["median_image_px"] for b in bins], marker="s", color=MONO, label="in the camera image [px]")
    ax.plot(x, [b["box_height_px"] for b in bins], ls="--", color="#adb5bd", label="box height [px]")
    ax.set(title="Error with distance", xlabel="distance [m]", yscale="log")
    ax.legend(fontsize=9)
    ax = axes[0, 2]
    values = np.array(result["cross_ratio"].get("predicted_values", []), float)
    if len(values):
        ax.hist(np.clip(values, 0.8, 2.2), bins=60, color="#6a4c93")
    ax.axvline(result["cross_ratio"]["model"], color="#bc4749", ls="--",
               label=f"cone model {result['cross_ratio']['model']:.3f}")
    ax.set(title="Cross ratio of the predicted keypoints", xlabel="cross ratio", ylabel="edges")
    ax.legend(fontsize=9)
    return _png(fig)


def cone_map(maps: dict[str, dict]) -> bytes:
    """The map each pipeline built against the true track, and how far its cones are off."""
    fig, axes = _figure(1, 2, "Cone map from a race, with the ground truth pose", size=(15, 7))
    colours = {"mono": MONO, "stereo": STEREO}
    ax = axes[0, 0]
    first = next(iter(maps.values()))
    ax.scatter([c["true_x"] for c in first["cones"]], [c["true_y"] for c in first["cones"]], s=14, marker="o",
               facecolors="none", edgecolors="#555555", label="true cone")
    for name, result in maps.items():
        colour = colours.get(name, "#777777")
        ax.scatter([c["x"] for c in result["cones"]], [c["y"] for c in result["cones"]], s=8, color=colour,
                   label=f"{name} (RMSE {result['rmse_m']:.2f} m)")
    ax.set(title="The track as mapped", xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for name, result in maps.items():
        errors = np.sort([c["error_m"] for c in result["cones"]])
        if len(errors):
            ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors), color=colours.get(name, "#777777"),
                    label=f"{name}, median {np.median(errors):.2f} m")
    ax.axvline(0.25, color="#bc4749", ls="--", label="stereo with SLAM, Gosala et al. (0.25 m)")
    ax.set(title="How far the mapped cones are off", xlabel="error [m]", ylabel="share of cones", ylim=(0, 1.02))
    ax.legend(fontsize=9)
    return _png(fig)


STAGE_COLOURS = {"detector": "#3a86ff", "jpeg": "#cdb4db", "crops": "#8ecae6", "keypoint net": "#ffb703",
                 "pnp": "#fb8500", "grey and F": "#adb5bd", "propagate": "#6c757d", "orb match": "#2a9d8f",
                 "triangulate": "#264653"}


def latency_breakdown(result: dict) -> bytes:
    """Median time per stage of both pipelines against the cycle of a real time loop, and the detector at other
    input sizes and precisions."""
    sweep = result.get("sweep") or []
    fig, axes = _figure(1, 2 if sweep else 1, "Time per stage", size=(15, 6))
    ax = axes[0, 0]
    rows = [("mono", result["mono"]), ("stereo", result["stereo"])]
    for y, (name, p) in enumerate(rows):
        left = 0.0
        for stage in p["stages"]:
            # every stage once in the legend: the mono stages from the first bar, the rest from the second
            first = y == 0 or stage["stage"] not in MONO_STAGES
            ax.barh(y, stage["p50"], left=left, color=STAGE_COLOURS.get(stage["stage"], "#999999"),
                    label=stage["stage"] if first else None)
            left += stage["p50"]
        # what the stage medians do not add up to: result objects, copies, Python between the stages
        rest = p["total"]["p50"] - left
        if rest > 0:
            ax.barh(y, rest, left=left, color="#e9ecef", edgecolor="#adb5bd", label="between stages" if y else None)
        ax.plot([p["total"]["p99"]] * 2, [y - 0.35, y + 0.35], color="k", lw=2)
    both = result["both"]
    ax.barh(2, both["p50"], color="#495057", label="both back to back")
    ax.plot([both["p99"]] * 2, [1.65, 2.35], color="k", lw=2, label="p99")
    for b in result["budget"]:
        ax.axvline(b["period_ms"], color="#e63946", ls="--", lw=1)
        ax.text(b["period_ms"] + 1, 2.45, f"{b['rate_hz']:g} Hz", color="#e63946", fontsize=9)
    race = result.get("race", {})
    for y, name in ((0, "mono"), (1, "stereo")):
        if name in race:
            ax.plot(race[name]["p95"], y, marker="D", color="#e63946", ms=7,
                    label="race p95, image to cones" if y else None)
    ax.set_yticks([0, 1, 2], ["mono", "stereo", "both"])
    ax.set_xlim(0, max(b["period_ms"] for b in result["budget"]) * 1.1)
    ax.set(title="Median ms per frame by stage, p99 as a black tick", xlabel="ms")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    if sweep:
        ax = axes[0, 1]
        labels = [f"{r['imgsz']}\n" + "\n".join(precision_label(r).split(", ")) for r in sweep]
        x = np.arange(len(sweep))
        bottom = np.zeros(len(sweep))
        for key, colour in (("preprocess", "#8ecae6"), ("inference", "#3a86ff"), ("postprocess", "#023047")):
            heights = np.array([r[key]["p50"] for r in sweep])
            ax.bar(x, heights, 0.6, bottom=bottom, color=colour, label=key)
            bottom += heights
        ax.errorbar(x, [r["total"]["p50"] for r in sweep],
                    yerr=[[0] * len(sweep), [r["total"]["p99"] - r["total"]["p50"] for r in sweep]],
                    fmt="none", ecolor="k", capsize=4, label="up to p99")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set(title="Detector alone by input size and precision", ylabel="ms per image")
        ax.legend(fontsize=8)
    return _png(fig)
