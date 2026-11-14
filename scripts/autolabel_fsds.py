"""Drives a few laps in a fresh simulator and captures automatically labelled camera images.

    uv run scripts/autolabel_fsds.py --map Competition/CompetitionMap1
    uv run scripts/autolabel_fsds.py --map TrainingMap --laps 1 --speeds 5

Every lap uses the next speed and lateral offset from --speeds and --offsets. The run goes to
data/interim/fsds_captures/<map>_<date>_<time>/ with images, YOLO labels, boxes.csv (distance, visibility and
truncation per box), frames.csv (camera pose per image) and run.yaml. scripts/prepare_fsds.py turns the captures
into a dataset. The simulator is started with configs/fsds/gotthard.json, which has the depth images the labels need.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from driverless_perception.sim.capture import CaptureConfig, capture

ROOT = Path(__file__).resolve().parents[1]


def sim(*args: str) -> None:
    result = subprocess.run([str(ROOT / "scripts/sim.sh"), *args], capture_output=True, text=True, check=False)
    print(result.stdout.strip())
    if result.returncode != 0:
        sys.exit(result.stderr.strip() or f"sim.sh {' '.join(args)} failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, help="TrainingMap, Competition/CompetitionMap1 and so on")
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--rate", type=float, default=2.0, help="frames per second, each with all cameras")
    parser.add_argument("--speeds", type=float, nargs="+", default=[5.0, 6.0, 7.0], help="m/s, one per lap")
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.4, -0.4],
                        help="metres left of the centre line, one per lap")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/fsds/gotthard.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/interim/fsds_captures")
    parser.add_argument("--name", help="run name, default <map>_<date>_<time>")
    args = parser.parse_args()

    map_name = args.map.split("/")[-1]
    name = args.name or f"{map_name}_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
    out = args.out / name
    if out.exists():
        sys.exit(f"{out} exists already")
    config = CaptureConfig(laps=args.laps, rate_hz=args.rate, speeds=args.speeds, offsets=args.offsets)

    # a fresh simulator for every run, the referee (cones hit) is not reset otherwise
    sim("stop")
    sim("start", args.map, str(args.settings.resolve()))
    time.sleep(3)
    try:
        summary = capture(out, name, map_name, config)
    finally:
        sim("stop")
    print(f"{summary['images']} images with {summary['labels']} labels ({summary['cones_found']} cones found) "
          f"in {summary['duration_s']} s, {summary['laps_driven']} laps, stopped: {summary['stop_reason']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
