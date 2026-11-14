"""Scores the cones a camera pipeline published during a recorded race against the ground truth track.

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    uv run scripts/score_run.py runs/<name>
    uv run scripts/score_run.py runs/<name> --pipeline stereo

Needs a run recorded with `scripts/record_run.sh --settings configs/fsds/gotthard.json --mono --stereo`, or a
stereo replay from `scripts/replay_stereo.sh`. Reads the bag, writes <run>/<pipeline>_metrics.json and prints
recall, precision and position error per range bin, plus how long the cones took from the image to the recorder.
A cone counts as found within --gate (1 m); the scores are also given for --strict-gate (0.5 m). Mono is scored on
cam_mono up to 20 m, stereo on cones both cam_left and cam_right see, also up to 20 m. --report also writes them as
markdown, together with where every frame's time went if the nodes published their timing.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from driverless_perception.common.geometry import yaw_from_quaternion
from driverless_perception.evaluation.race import camera_view, report_markdown, score_frames, timing_summary

# cameras that have to see a cone for it to be expected, and up to which distance
PIPELINES = {"mono": (["cam_mono"], 20.0), "stereo": (["cam_left", "cam_right"], 20.0)}


def read_bag(path: Path, topics: list[str]) -> dict[str, list]:
    """Messages per topic as (receive time in ns, message)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in topics}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(types)))
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, t = reader.read_next()
        out[topic].append((t, deserialize_message(data, types[topic])))
    return out


def stamp(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="mono")
    parser.add_argument("--cameras", nargs="+", help="cameras that have to see a cone, default by pipeline")
    parser.add_argument("--max-range", type=float, help="expected cones up to this far away, default by pipeline")
    parser.add_argument("--gate", type=float, default=1.0, help="metres within which a cone counts as found")
    parser.add_argument("--strict-gate", type=float, default=0.5,
                        help="a second, stricter gate the scores are also given for, 0 to leave it out")
    parser.add_argument("--report", type=Path, help="also write the scores as markdown to this file")
    parser.add_argument("--title", help="heading of the report, default names the run")
    args = parser.parse_args()

    cameras, max_range = PIPELINES[args.pipeline]
    cameras = args.cameras or cameras
    max_range = args.max_range or max_range
    edges = sorted({*range(0, int(max_range), 5), max_range})
    cones_topic = f"/perception/{args.pipeline}/cones"
    odom_topic, track_topic = "/fsds/testing_only/odom", "/ground_truth/cones"
    info_topics = [f"/fsds/{c}/camera_info" for c in cameras]
    timing_topic = f"/perception/{args.pipeline}/timing"
    msgs = read_bag(args.run, [cones_topic, odom_topic, track_topic, *info_topics, "/tf_static", "/race/lap",
                               "/fsds/testing_only/extra_info", timing_topic])
    if not msgs[cones_topic]:
        raise SystemExit(f"no {cones_topic} in {args.run}")

    track = msgs[track_topic][-1][1]
    track_xy = np.array([[c.position.x, c.position.y] for c in track.cones])
    colors = np.array([c.color for c in track.cones])
    odom = [m for _, m in msgs[odom_topic]]
    odom_times = np.array([stamp(m.header) for m in odom])
    odom_poses = np.array([[m.pose.pose.position.x, m.pose.pose.position.y,
                            yaw_from_quaternion(m.pose.pose.orientation.x, m.pose.pose.orientation.y,
                                                m.pose.pose.orientation.z, m.pose.pose.orientation.w)]
                           for m in odom])
    order = np.argsort(odom_times)
    views = []
    for c, topic in zip(cameras, info_topics):
        info = msgs[topic][-1][1]
        mount = next(t.transform.translation for _, m in msgs["/tf_static"] for t in m.transforms
                     if t.child_frame_id.endswith(c))
        views.append(camera_view(info.width, info.k[0], mount.x, mount.y))

    frames = [(stamp(m.header), np.array([[c.position.x, c.position.y] for c in m.cones]).reshape(-1, 2))
              for _, m in msgs[cones_topic]]
    result = score_frames(frames, odom_times[order], odom_poses[order], track_xy, colors, views, max_range,
                          args.gate, edges)
    if args.strict_gate:
        strict = score_frames(frames, odom_times[order], odom_poses[order], track_xy, colors, views, max_range,
                              args.strict_gate, edges)
        result["strict"] = {"gate_m": args.strict_gate, "all": strict["all"], "bins": strict["bins"]}
    latency = np.array([(t - stamp(m.header)) / 1e6 for t, m in msgs[cones_topic]])
    span = (msgs[cones_topic][-1][0] - msgs[cones_topic][0][0]) / 1e9
    result["latency_ms"] = {"median": float(np.median(latency)), "p95": float(np.percentile(latency, 95))}
    result["rate_hz"] = len(frames) / span if span > 0 else None
    result["cameras"] = {c: float(np.degrees(v.fov)) for c, v in zip(cameras, views)}
    if msgs[timing_topic]:
        # recorded since the nodes publish where their time goes, older runs don't have it
        result["timing"] = timing_summary([[(st.name, st.ms) for st in m.stages] for _, m in msgs[timing_topic]])
        result["boxes_per_frame"] = float(np.mean([m.boxes for _, m in msgs[timing_topic]]))
    if msgs["/fsds/testing_only/extra_info"]:
        # the referee's count of cones hit, their ground truth position is wrong from then on
        result["cones_hit"] = int(msgs["/fsds/testing_only/extra_info"][-1][1].doo_counter)

    # the race supervisor publishes the number of completed laps, a frame belongs to the lap after the last count.
    # Frames after the last completed lap (braking after the finish) get no row of their own.
    changes = [(t, m.data) for t, m in msgs["/race/lap"]]
    if changes:
        receive = [t for t, _ in msgs[cones_topic]]
        lap_of = [1 + max([n for t0, n in changes if t0 <= t], default=0) for t in receive]
        result["laps"] = []
        for lap in sorted({n for n in lap_of if n <= max(n for _, n in changes)}):
            part = [f for f, n in zip(frames, lap_of) if n == lap]
            s = score_frames(part, odom_times[order], odom_poses[order], track_xy, colors, views, max_range,
                             args.gate, edges)
            if s["frames"]:
                result["laps"].append({"lap": lap, "frames": s["frames"], "recall": s["all"]["recall"],
                                       "precision": s["all"]["precision"]})
    metrics = args.run / f"{args.pipeline}_metrics.json"
    metrics.write_text(json.dumps(result, indent=2) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        title = args.title or f"{args.pipeline.capitalize()} pipeline in a race ({args.run.name})"
        args.report.write_text(report_markdown(result, title))
        print(f"wrote {args.report}")

    a = result["all"]
    print(f"{result['frames']} frames at {result['rate_hz']:.1f} Hz, latency {result['latency_ms']['median']:.0f} ms "
          f"(95%: {result['latency_ms']['p95']:.0f} ms), recall {a['recall']:.2f}, precision {a['precision']:.2f}")
    print("range      expected  recall  precision  position RMSE")
    for b in result["bins"]:
        fmt = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
        print(f"{b['bin_m'][0]:>4.0f}-{b['bin_m'][1]:<4.0f} m {b['n_gt']:>8}  {fmt(b['recall']):>6}  "
              f"{fmt(b['precision']):>9}  {fmt(b['pos_rmse_m'])} m")
    print(f"wrote {metrics}")


if __name__ == "__main__":
    main()
