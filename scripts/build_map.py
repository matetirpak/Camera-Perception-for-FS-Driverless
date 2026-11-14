"""Builds a cone map from a recorded race and compares it with the true track (Gosala et al., Fig. 9b).

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    uv run scripts/build_map.py runs/live_competitionmap3_10laps

Both pipelines' cones are read from the race, or the stereo ones from a replay with --stereo. Every cone a pipeline
published is placed in the map frame with the ground truth pose at its time; each cone of the
map is the median of its observations. Writes reports/map/<run>.md and .json. The paper reaches 0.25 m with stereo
and 0.20 m with the LiDAR, both with its own SLAM pose.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_run import read_bag, stamp

from driverless_perception.common.geometry import yaw_from_quaternion
from driverless_perception.evaluation.mapping import build_map, report_markdown

ROOT = Path(__file__).resolve().parents[1]
RANGES = {"mono": 20.0, "stereo": 20.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the race with the ground truth and the pipelines' cones")
    parser.add_argument("--stereo", type=Path, help="take the stereo cones from this replay of the race instead")
    parser.add_argument("--gate", type=float, default=1.0, help="metres within which an observation belongs to a cone")
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/map")
    args = parser.parse_args()

    msgs = read_bag(args.run, ["/perception/mono/cones", "/perception/stereo/cones", "/fsds/testing_only/odom",
                               "/ground_truth/cones"])
    track = msgs["/ground_truth/cones"][-1][1]
    track_xy = np.array([[c.position.x, c.position.y] for c in track.cones])
    colors = np.array([c.color for c in track.cones])
    odom = [m for _, m in msgs["/fsds/testing_only/odom"]]
    times = np.array([stamp(m.header) for m in odom])
    poses = np.array([[m.pose.pose.position.x, m.pose.pose.position.y,
                       yaw_from_quaternion(m.pose.pose.orientation.x, m.pose.pose.orientation.y,
                                           m.pose.pose.orientation.z, m.pose.pose.orientation.w)] for m in odom])
    order = np.argsort(times)
    times, poses = times[order], poses[order]

    sources = {"mono": msgs["/perception/mono/cones"], "stereo": msgs["/perception/stereo/cones"]}
    if args.stereo:
        sources["stereo"] = read_bag(args.stereo, ["/perception/stereo/cones"])["/perception/stereo/cones"]
    sources = {name: arrays for name, arrays in sources.items() if arrays}
    maps = {}
    for name, arrays in sources.items():
        frames = [(stamp(m.header), np.array([[c.position.x, c.position.y] for c in m.cones]).reshape(-1, 2),
                   np.array([c.color for c in m.cones], int)) for _, m in arrays]
        maps[name] = build_map(frames, times, poses, track_xy, colors, args.gate, args.min_observations,
                               RANGES.get(name))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.run.name}.json").write_text(json.dumps(maps, indent=2) + "\n")
    report = report_markdown(maps, f"Cone map of {args.run.name}")
    (args.out / f"{args.run.name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
