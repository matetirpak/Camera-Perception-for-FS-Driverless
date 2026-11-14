"""One file for Foxglove with everything there is to see: a lap of a recorded race with what the pipelines made of
it, and a figure of every offline result. ``scripts/build_results.py`` is the command line around ``build``.

The bags are read with the mcap library rather than rosbag2, so nothing here needs ROS: their messages are copied
into the new file unchanged, and what we add uses Foxglove's own schemas (``visualization.foxglove``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# up in the optical frame of a level camera
LEVEL_UP = np.array([0.0, -1.0, 0.0])

CAMERAS = ("cam_left", "cam_mono", "cam_right")
# copied from the race bag: what the 3D view, the plots and the camera panels need
COPY = ["/fsds/testing_only/odom", "/fsds/testing_only/extra_info", "/fsds/testing_only/track", "/fsds/gss",
        "/fsds/control_command", "/tf", "/tf_static", "/ground_truth/cones", "/ground_truth/cone_markers",
        "/race/lap", "/race/state", "/perception/mono/cones", "/perception/mono/markers",
        *[f"/fsds/{c}/image_color/compressed" for c in CAMERAS], *[f"/fsds/{c}/camera_info" for c in CAMERAS]]
STEREO_COPY = ["/perception/stereo/cones", "/perception/stereo/markers"]
LIDAR = "/fsds/lidar/Lidar"
FIGURE_PERIOD_NS = 5_000_000_000
SAME_RENDER_NS = 12_000_000
PAIR_LIMIT_NS = 50_000_000


def bag(run: Path) -> Path:
    return next(iter(sorted(run.glob("*.mcap"))))


def stamp_ns(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def decode(msg) -> np.ndarray:
    import cv2

    return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)


def read(path: Path, topics: list[str], start: int | None = None, end: int | None = None) -> dict[str, list]:
    """Decoded messages per topic as (log time, message)."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    out = {t: [] for t in topics}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, msg in reader.iter_decoded_messages(topics=topics, start_time=start, end_time=end):
            out[channel.topic].append((message.log_time, msg))
    return out


def lap_window(path: Path, lap: int) -> tuple[int, int]:
    """Start and end time of one lap, from the lap counter the race supervisor published."""
    counts = {msg.data: t for t, msg in read(path, ["/race/lap"])["/race/lap"]}
    if lap - 1 not in counts or lap not in counts:
        raise SystemExit(f"lap {lap} is not in the bag, it counted {sorted(counts)}")
    return counts[lap - 1], counts[lap]


def copy_window(writer, path: Path, topics: list[str], start: int, end: int) -> None:
    """Copies the messages of ``topics`` in the window, and for topics that only have older messages (the latched
    track and cones) the last one before it."""
    from mcap.reader import make_reader

    with open(path, "rb") as f:
        reader = make_reader(f)
        seen = set()
        for schema, channel, message in reader.iter_messages(topics=topics, start_time=start, end_time=end):
            writer.copy(schema, channel, message)
            seen.add(channel.topic)
        missing = [t for t in topics if t not in seen]
        if missing:
            last = {}
            for schema, channel, message in reader.iter_messages(topics=missing, end_time=start):
                last[channel.topic] = (schema, channel, message)
            for schema, channel, message in last.values():
                writer.copy(schema, channel, message, log_time=start)


def camera_setup(path: Path, start: int):
    """Intrinsics per camera from camera_info and the mount of each from the static transforms."""
    from scipy.spatial.transform import Rotation

    from ..common.cameras import Intrinsics

    info = read(path, [f"/fsds/{c}/camera_info" for c in CAMERAS], end=start + FIGURE_PERIOD_NS)
    intrinsics = {}
    for camera in CAMERAS:
        msg = info[f"/fsds/{camera}/camera_info"][-1][1]
        fov = np.degrees(2 * np.arctan(msg.width / 2 / msg.k[0]))
        intrinsics[camera] = Intrinsics(msg.width, msg.height, float(fov))
    mounts = {}
    for _, msg in read(path, ["/tf_static"], end=start + FIGURE_PERIOD_NS)["/tf_static"]:
        for t in msg.transforms:
            for camera in CAMERAS:
                if t.child_frame_id.endswith(camera):
                    q, p = t.transform.rotation, t.transform.translation
                    mounts[camera] = (Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                                      np.array([p.x, p.y, p.z]))
    return intrinsics, mounts


def odometry(path: Path, start: int, end: int):
    """Stamps, positions and quaternions of the ground truth odometry, a second around the window included."""
    rows = read(path, ["/fsds/testing_only/odom"], start - 1_000_000_000, end + 1_000_000_000)
    messages = [m for _, m in rows["/fsds/testing_only/odom"]]
    times = np.array([stamp_ns(m.header) for m in messages])
    positions = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z]
                          for m in messages])
    quaternions = np.array([[m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z,
                             m.pose.pose.orientation.w] for m in messages])
    order = np.argsort(times)
    return times[order], positions[order], quaternions[order]


def mono_annotations(writer, images, pipeline, intrinsics, topic: str, rows_topic: str, world, track_xy) -> None:
    """Runs the detector and the keypoint network on every image, draws boxes, keypoints and the PnP range, and
    writes a row per cone for the table."""
    from ..common.types import CLASS_NAMES
    from .foxglove import CONE_RGBA, Annotations

    for log_time, msg in images:
        image = decode(msg)
        stamp = stamp_ns(msg.header)
        boxes, classes, scores = pipeline.detect(image)
        inside = pipeline.inside(boxes, intrinsics.width)
        boxes, classes, scores = boxes[inside], classes[inside], scores[inside]
        keypoints = pipeline.keypoints(image, boxes)
        # the camera is mounted level, and the upright cone doesn't depend on the car's pitch and roll
        positions = pipeline.locate(keypoints, intrinsics.matrix, LEVEL_UP)
        ann = Annotations(stamp)
        rows = []
        for box, cls, score, kp, t in zip(boxes, classes, scores, keypoints, positions):
            ann.box(box, CONE_RGBA[int(cls)])
            ann.points(kp, (1.0, 1.0, 1.0, 0.9), 3.0)
            label = f"{CLASS_NAMES[int(cls)]} {score:.2f}"
            row = {"colour": CLASS_NAMES[int(cls)], "score": round(float(score), 3),
                   "box height px": round(float(box[3] - box[1]), 1), "PnP distance m": None, "PnP error m": None}
            if t is not None:
                label += f", {np.linalg.norm(t):.1f} m"
                row["PnP distance m"] = round(float(np.linalg.norm(t)), 2)
                point = world(stamp, t)
                if point is not None and len(track_xy):
                    row["PnP error m"] = round(float(np.min(np.hypot(*(track_xy - point[:2]).T))), 2)
            ann.text(box[0], max(box[1] - 16, 0), label, CONE_RGBA[int(cls)], 13.0)
            rows.append(row)
        writer.proto(topic, ann.msg, log_time)
        writer.json(rows_topic, "Cones", {"cones": rows}, log_time)


def pair_images(left: list, right: list) -> list[tuple]:
    """Each left image with the right one closest in time: (left, right, gap in ns)."""
    times = np.array([stamp_ns(m.header) for _, m in right])
    pairs = []
    for log_time, msg in left:
        if not len(times):
            break
        j = int(np.argmin(np.abs(times - stamp_ns(msg.header))))
        pairs.append(((log_time, msg), right[j], int(abs(times[j] - stamp_ns(msg.header)))))
    return pairs


def stereo_pass(writer, pairs, pipeline, intrinsics, mounts, odom, track_xy, max_range: float):
    """Stereo on every pair from the same render: annotations for both images and the points in the 3D view. Also
    measures the error of both pose policies per stamp gap, for the pairing figure."""
    from ..common.cameras import camera_pose
    from ..common.geometry import interpolate_pose_3d
    from ..vision.stereo.pipeline import View

    times, positions, quaternions = odom
    errors = {"own stamps": [], "one pose": []}
    best = None

    def view(camera, pose, image):
        position, orientation = camera_pose(*pose, *mounts[camera])
        return View(image, intrinsics[camera], position, orientation, float(pose[0][2]))

    def nearest(xy):
        d = np.hypot(*(track_xy - xy).T)
        return float(d.min())

    for (log_time, left_msg), (_, right_msg), gap in pairs:
        t_left, t_right = stamp_ns(left_msg.header), stamp_ns(right_msg.header)
        pose = interpolate_pose_3d(t_left, times, positions, quaternions)
        own = interpolate_pose_3d(t_right, times, positions, quaternions)
        if pose is None or own is None or gap > PAIR_LIMIT_NS:
            continue
        left_image, right_image = decode(left_msg), decode(right_msg)
        left_view = view("cam_left", pose, left_image)
        boxes, classes, scores = pipeline.mono.detect(left_image)
        for policy, right_pose in (("one pose", pose), ("own stamps", own)):
            estimates = pipeline.estimate(left_view, view("cam_right", right_pose, right_image), boxes)
            for e in estimates:
                if e.stereo is not None and np.hypot(*(e.stereo - left_view.position[:2])) <= max_range:
                    error = nearest(e.stereo)
                    if error < 1.0:
                        errors[policy].append((gap / 1e6, error))
            if policy == "one pose" and gap <= SAME_RENDER_NS:
                draw_pair(writer, log_time, left_msg, right_msg, boxes, classes, scores, estimates, left_view,
                          track_xy)
                on_cone = sum(e.points for e in estimates)
                if best is None or on_cone > best[0]:
                    best = (on_cone, left_image, right_image, boxes, classes, estimates)
    return errors, best


def draw_pair(writer, log_time, left_msg, right_msg, boxes, classes, scores, estimates, left_view,
              track_xy=None) -> None:
    """Annotations for the left and the right image of one pair, and the triangulated points in the map frame."""
    from ..common.types import CLASS_NAMES
    from .foxglove import CONE_RGBA, Annotations, sphere_cloud

    left = Annotations(stamp_ns(left_msg.header))
    right = Annotations(stamp_ns(right_msg.header))
    points, colours = [], []
    for box, cls, score, e in zip(boxes, classes, scores, estimates):
        colour = CONE_RGBA[int(cls)]
        left.box(box, colour)
        if e.keypoints is not None:
            left.points(e.keypoints, (1.0, 1.0, 1.0, 0.9), 3.0)
        label = f"{CLASS_NAMES[int(cls)]} {score:.2f}"
        if e.stereo is not None:
            label += f", stereo {np.hypot(*(e.stereo - left_view.position[:2])):.1f} m ({e.points} points)"
        elif e.mono is not None:
            label += ", no stereo"
        left.text(box[0], max(box[1] - 16, 0), label, colour, 13.0)
        if e.right_box is not None:
            right.box(e.right_box, (0.8, 0.8, 0.8, 0.9), 1.0)
        if e.pts_right is not None and len(e.pts_right):
            right.points(e.pts_right[~e.on_cone], (0.6, 0.6, 0.6, 0.8), 3.0)
            right.points(e.pts_right[e.on_cone], colour, 4.0)
            points.append(e.world[e.on_cone])
            colours += [colour] * int(e.on_cone.sum())
    rows = []
    for box, cls, score, e in zip(boxes, classes, scores, estimates):
        row = {"colour": CLASS_NAMES[int(cls)], "score": round(float(score), 3),
               "box height px": round(float(box[3] - box[1]), 1), "matches": e.matches, "on cone": e.points,
               "PnP distance m": None, "stereo distance m": None, "PnP error m": None, "stereo error m": None}
        for name, estimate in (("PnP", e.mono), ("stereo", e.stereo)):
            if estimate is None:
                continue
            row[f"{name} distance m"] = round(float(np.hypot(*(estimate[:2] - left_view.position[:2]))), 2)
            if track_xy is not None and len(track_xy):
                row[f"{name} error m"] = round(float(np.min(np.hypot(*(track_xy - estimate[:2]).T))), 2)
        rows.append(row)
    writer.json("/results/cones", "Cones", {"cones": rows}, log_time)
    writer.proto("/results/cam_left/annotations", left.msg, log_time)
    writer.proto("/results/cam_right/annotations", right.msg, log_time)
    if points:
        writer.proto("/results/stereo/points",
                     sphere_cloud(stamp_ns(left_msg.header), "fsds/map", "matches", np.vstack(points),
                                  np.array(colours)), log_time)


def frame_metrics(writer, arrays, topic: str, odom2d, track_xy, colors, views, max_range: float, gate: float = 1.0):
    """Recall, precision and error of every published cone array against the ground truth, for the plots."""
    from ..common.geometry import interpolate_pose
    from ..evaluation.metrics import BinnedCounts
    from ..evaluation.race import split_cones

    times, poses = odom2d
    for log_time, msg in arrays:
        t = stamp_ns(msg.header)
        if not times[0] <= t <= times[-1]:
            continue
        pose = interpolate_pose(t, times, poses)
        expected, others = split_cones(track_xy, colors, pose, views, max_range)
        detected = np.array([[c.position.x, c.position.y] for c in msg.cones]).reshape(-1, 2)
        counts = BinnedCounts(edges=(0.0, max_range))
        counts.add_frame(detected, expected, gate, ignore_xy=others)
        tp, fp, fn = int(counts.tp[0]), int(counts.fp[0]), int(counts.fn[0])
        errors = counts.errors[0]
        writer.json(topic, "FrameMetrics", {
            "expected": tp + fn, "found": tp, "missed": fn, "false": fp,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "mean_error_m": float(np.mean(errors)) if errors else None,
            "max_error_m": float(np.max(errors)) if errors else None,
        }, log_time)


def build_figures(root: Path, race: Path, stereo_run: Path, models, keypoint_model, device: str, conf: float,
                  pairing_data=None) -> dict[str, bytes]:
    """Every offline result as a picture: datasets, training, detection, keypoints, mono, stereo and the race.

    Everything comes from the reports, the training logs and the datasets, except the pairing figure, which needs
    the gaps and errors measured on a race bag. Without them that one figure is left out. ``models`` maps model
    names to loaded detectors, the FSOCO model first and the fine-tuned one the pipelines use second."""
    import cv2

    from ..vision.keypoints.crops import CropCache
    from . import examples, figures
    from .draw import caption, png, tile

    baseline, detector = list(models)
    reports = {}
    for dataset in ("fsoco", "fsds"):
        for model in models:
            path = root / "reports/detection" / dataset / f"{model}.json"
            if path.exists():
                reports[(dataset, model)] = json.loads(path.read_text())
    out = {}
    out["datasets"] = figures.dataset_overview(root / "data/processed/fsoco_yolo/boxes.csv",
                                               root / "data/processed/fsds_yolo/boxes.csv")
    out["training"] = figures.training_curves({m: root / "experiments/detection" / m / "results.csv"
                                               for m in models if (root / "experiments/detection" / m).exists()},
                                              root / "experiments/keypoints/fsds/results.csv")
    out["detection"] = figures.detection_summary(reports)
    curves = [(root / "experiments/detection" / baseline / "fsoco_test/BoxPR_curve.png", "FSOCO model on FSOCO test"),
              (root / "experiments/detection" / detector / "fsds_test/BoxPR_curve.png",
               "fine-tuned model on the simulator test map")]
    if all(p.exists() for p, _ in curves):
        out["detection_pr"] = png(tile([caption(cv2.imread(str(p)), t) for p, t in curves], 2, 700))
    mono_report = json.loads((root / "reports/mono/fsds.json").read_text())
    stereo_report = json.loads((root / "reports/stereo/fsds.json").read_text())
    out["mono"] = figures.mono_errors(mono_report)
    out["stereo"] = figures.stereo_vs_mono(stereo_report)
    out["stereo_stages"] = figures.stereo_stages(stereo_report["stages"])
    out["depth_driving"] = figures.depth_scatter({"PnP": stereo_report["samples"]["mono"],
                                                  "stereo": stereo_report["samples"]["stereo"]})
    if "keypoints_detail" in mono_report:
        out["keypoint_errors"] = figures.keypoint_errors(mono_report["keypoints_detail"])
    standing = root / "reports/depth/fsds.json"
    if standing.exists():
        out["depth_standing"] = figures.depth_spread(json.loads(standing.read_text()))
    latency = root / "reports/latency/fsds.json"
    if latency.exists():
        out["latency"] = figures.latency_breakdown(json.loads(latency.read_text()))
    cone_map = root / "reports/map" / f"{race.name}.json"
    if cone_map.exists():
        out["map"] = figures.cone_map(json.loads(cone_map.read_text()))
    mono_race, stereo_race = race / "mono_metrics.json", stereo_run / "stereo_metrics.json"
    if mono_race.exists():
        out["race"] = figures.race_scores(json.loads(mono_race.read_text()),
                                          json.loads(stereo_race.read_text()) if stereo_race.exists() else None)
    if pairing_data is not None:
        gaps, errors = pairing_data
        out["pairing"] = figures.pairing(gaps, errors)
    out["fsoco_examples"] = examples.fsoco_examples(root / "data/processed/fsoco_yolo",
                                                    {"FSOCO model": models[baseline],
                                                     "fine-tuned on sim": models[detector]}, conf)
    captures = sorted(root.glob("data/interim/fsds_captures/*/run.yaml"))
    wanted = [c.parent for c in captures if c.parent.name.split("_")[0] in
              ("TrainingMap", "CompetitionMapTestday2", "CompetitionMap3")]
    out["sim_examples"] = examples.sim_examples(wanted, models[detector], conf)
    crops = root / "data/processed/fsds_keypoints/test.npz"
    if crops.exists():
        out["keypoint_crops"] = examples.keypoint_examples(CropCache.load(crops), keypoint_model, device)
    return out


def summary_tables(root: Path, race: Path, stereo_run: Path, models) -> dict:
    """The numbers of every report in one message, for the table panels."""
    rows = {"detection": [], "mono_offline": [], "stereo_offline": [], "stereo_stages": [], "race_by_range": [],
            "race_by_lap": [], "map": []}
    for dataset, name in (("fsoco", "FSOCO test"), ("fsds", "simulator test")):
        for model in models:
            path = root / "reports/detection" / dataset / f"{model}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            rows["detection"].append({"test set": name, "model": model, "precision": r["test"]["all"]["precision"],
                                      "recall": r["test"]["all"]["recall"], "mAP50-95": r["coco"]["map50_95"],
                                      "colour accuracy": r["test"]["colour_accuracy"],
                                      "latency ms": r["latency"]["total"]})
    mono = json.loads((root / "reports/mono/fsds.json").read_text())
    for b in mono["bins"]["detector"]:
        if b.get("median_m") is not None:
            rows["mono_offline"].append({"distance": f"{b['distance_m'][0]}-{b['distance_m'][1]} m",
                                         "cones": b["cones"], "median m": b["median_m"], "RMSE m": b["rmse_m"],
                                         "90% m": b["p90_m"], "range bias m": b["range_bias_m"]})
    stereo = json.loads((root / "reports/stereo/fsds.json").read_text())
    for b in stereo["bins"]["detector"]:
        if "stereo" in b:
            rows["stereo_offline"].append({"distance": f"{b['distance_m'][0]}-{b['distance_m'][1]} m",
                                           "cones": b["cones"], "with stereo": b["with_stereo"],
                                           "median PnP m": b["mono"]["median_m"],
                                           "median stereo m": b["stereo"]["median_m"],
                                           "spread PnP m": b["mono"]["range_spread_m"],
                                           "spread stereo m": b["stereo"]["range_spread_m"]})
    for b in stereo["stages"]:
        if b["cones"]:
            rows["stereo_stages"].append({"distance": f"{b['distance_m'][0]}-{b['distance_m'][1]} m",
                                          "cones": b["cones"], "box IoU": b.get("iou_median"),
                                          "box holds the cone": b.get("contains_share"),
                                          "features": b.get("features_median"), "matches": b.get("matches_median"),
                                          "on cone": b.get("on_cone_median"),
                                          "epipolar px": b.get("epipolar_px_median"),
                                          "point error m": b.get("point_error_median_m")})
    cone_map = root / "reports/map" / f"{race.name}.json"
    if cone_map.exists():
        for name, result in json.loads(cone_map.read_text()).items():
            rows["map"].append({"pipeline": name, "mapped": result["mapped"], "expected": result["expected"],
                                "missed": result["missed"], "RMSE m": result["rmse_m"],
                                "median m": result["median_m"], "90% m": result["p90_m"],
                                "colour accuracy": result["colour_accuracy"],
                                "observations per cone": int(np.median([c["observations"] for c in result["cones"]]))
                                if result["cones"] else 0})
    races = {"mono": race / "mono_metrics.json", "stereo": stereo_run / "stereo_metrics.json"}
    laps = {}
    for pipeline, path in races.items():
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        for b in r["bins"]:
            if b["n_gt"]:
                rows["race_by_range"].append({"pipeline": pipeline,
                                              "range": f"{b['bin_m'][0]:.0f}-{b['bin_m'][1]:.0f} m",
                                              "expected": b["n_gt"], "recall": b["recall"],
                                              "precision": b["precision"], "RMSE m": b["pos_rmse_m"]})
        for lap in r.get("laps", []):
            row = laps.setdefault(lap["lap"], {"lap": lap["lap"]})
            row[f"{pipeline} recall"] = lap["recall"]
            row[f"{pipeline} precision"] = lap["precision"]
    rows["race_by_lap"] = [laps[k] for k in sorted(laps)]
    return rows


def write_figures(pictures: dict[str, bytes], directory: Path) -> list[str]:
    """Drops every figure as a PNG into ``directory`` and returns the names in the order they were written."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in pictures.items():
        (directory / f"{name}.png").write_bytes(data)
    return list(pictures)


def build(root: Path, race_run: Path, stereo_run: Path, lap: int, out: Path, mono, stereo, models,
          keypoint_model, device: str, conf: float, max_range: float = 20.0, lidar: bool = True,
          figures_dir: Path | None = None) -> Path:
    """Writes ``out/<name>.mcap``: one lap of ``race_run`` with the pipelines' work drawn on it, the cones of
    ``stereo_run`` (the same run for a live race), and the figures of every report under ``root``."""
    from .foxglove import ResultsWriter, compressed_image

    race, stereo_bag = bag(race_run), bag(stereo_run) if stereo_run.exists() else None
    start, end = lap_window(race, lap)
    print(f"lap {lap}: {(end - start) / 1e9:.1f} s of {race_run.name}")
    out.mkdir(parents=True, exist_ok=True)
    writer = ResultsWriter(out / f"{out.name}.mcap")

    topics = COPY + ([LIDAR] if lidar else [])
    copy_window(writer, race, topics, start, end)
    if stereo_bag is not None:
        copy_window(writer, stereo_bag, STEREO_COPY, start, end)

    intrinsics, mounts = camera_setup(race, start)
    odom = odometry(race, start, end)
    images = read(race, [f"/fsds/{c}/image_color/compressed" for c in CAMERAS], start, end)
    track = read(race, ["/ground_truth/cones"], end=end)["/ground_truth/cones"][-1][1]
    track_xy = np.array([[c.position.x, c.position.y] for c in track.cones])
    colors = np.array([c.color for c in track.cones])

    from ..common.cameras import camera_pose, optical_to_world
    from ..common.geometry import interpolate_pose_3d

    def mono_world(stamp, t, camera="cam_mono"):
        """A point in the camera's optical frame moved into the map frame, for the error against the true cones."""
        pose = interpolate_pose_3d(stamp, *odom)
        if pose is None:
            return None
        position, orientation = camera_pose(*pose, *mounts[camera])
        return optical_to_world(np.asarray(t, float)[None], position, orientation)[0]

    print(f"mono pipeline on {len(images['/fsds/cam_mono/image_color/compressed'])} images")
    mono_annotations(writer, images["/fsds/cam_mono/image_color/compressed"], mono, intrinsics["cam_mono"],
                     "/results/cam_mono/annotations", "/results/cam_mono/cones", mono_world, track_xy)
    pairs = pair_images(images["/fsds/cam_left/image_color/compressed"],
                        images["/fsds/cam_right/image_color/compressed"])
    print(f"stereo pipeline on {sum(1 for *_, gap in pairs if gap <= SAME_RENDER_NS)} pairs from the same render "
          f"({len(pairs)} left images)")
    errors, best = stereo_pass(writer, pairs, stereo, intrinsics, mounts, odom, track_xy, stereo.max_range)

    times2d = odom[0]
    from ..common.geometry import yaw_from_quaternion
    from ..evaluation.race import camera_view

    poses2d = np.array([[p[0], p[1], yaw_from_quaternion(*q)] for p, q in zip(odom[1], odom[2])])
    views = {"mono": (camera_view(intrinsics["cam_mono"].width, intrinsics["cam_mono"].f, *mounts["cam_mono"][1][:2]),),
             "stereo": tuple(camera_view(intrinsics[c].width, intrinsics[c].f, *mounts[c][1][:2])
                             for c in ("cam_left", "cam_right"))}
    for pipeline, path, ranges in (("mono", race, max_range), ("stereo", stereo_bag, stereo.max_range)):
        if path is None:
            continue
        arrays = read(path, [f"/perception/{pipeline}/cones"], start, end)[f"/perception/{pipeline}/cones"]
        frame_metrics(writer, arrays, f"/results/{pipeline}/frame", (times2d, poses2d), track_xy, colors,
                      views[pipeline], ranges)

    gaps = np.array([gap / 1e6 for *_, gap in pairs])
    pictures = build_figures(root, race_run, stereo_run, models, keypoint_model, device, conf,
                             (gaps, errors))
    if best is not None:
        from .examples import match_example

        _, left_image, right_image, boxes, classes, estimates = best
        pictures["stereo_matches"] = match_example(left_image, right_image, boxes, classes, estimates)
    summary = summary_tables(root, race_run, stereo_run, models)
    for t in range(start, end, FIGURE_PERIOD_NS):
        for name, data in pictures.items():
            writer.proto(f"/results/figures/{name}", compressed_image(t, data), t)
        writer.json("/results/summary", "Summary", summary, t)
    writer.close()

    if figures_dir is not None:
        write_figures(pictures, figures_dir)
    path = out / f"{out.name}.mcap"
    print(f"wrote {path} ({path.stat().st_size / 1e6:.0f} MB), {len(pictures)} figures")
    return path
