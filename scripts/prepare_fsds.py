"""Turns the simulator captures into a YOLO dataset in data/processed/fsds_yolo, split by map.

    uv run scripts/prepare_fsds.py
    uv run scripts/prepare_fsds.py --min-contrast 12 --overwrite

The split is in configs/detection/fsds_split.yaml. The labels are written again from every capture's boxes.csv
with the thresholds given here, so they can change without driving again.
"""
import argparse
import shutil
from pathlib import Path

from driverless_perception.common.types import CLASS_NAMES
from driverless_perception.sim.autolabel import LabelPolicy
from driverless_perception.sim.dataset import export_captures, read_split

ROOT = Path(__file__).resolve().parents[1]


def main():
    default = LabelPolicy()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, default=ROOT / "data/interim/fsds_captures")
    parser.add_argument("--split", type=Path, default=ROOT / "configs/detection/fsds_split.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/fsds_yolo")
    parser.add_argument("--min-visible", type=float, default=default.min_visible)
    parser.add_argument("--min-height", type=int, default=default.min_height)
    parser.add_argument("--min-contrast", type=float, default=default.min_contrast)
    parser.add_argument("--overwrite", action="store_true", help="delete --out first if it exists")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.out} is not empty, pass --overwrite to replace it")
        shutil.rmtree(args.out)
    captures = sorted(p.parent for p in args.captures.glob("*/run.yaml"))
    if not captures:
        raise SystemExit(f"no captures in {args.captures}")
    policy = LabelPolicy(args.min_visible, args.min_height, args.min_contrast)
    summary = export_captures(captures, read_split(args.split), args.out, policy)

    names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    print(f"{len(captures)} captures, {policy}")
    for part, s in summary.items():
        labels = ", ".join(f"{n} {c}" for n, c in zip(names, s["labels"]) if c)
        print(f"{part}: {s['images']} images from {', '.join(sorted(s['maps'])) or 'no maps'}. Labels: {labels}. "
              f"Cones with keypoints: {s['keypoint_cones']}")
    print(f"dataset file: {args.out / 'fsds.yaml'}")


if __name__ == "__main__":
    main()
