"""Splits FSOCO into train, val and test by camera origin and writes configs/detection/fsoco_split.yaml.

An image's origin is the team whose car recorded it, taken from the file name. The folder it sits in is only the
team that labelled it, so splitting by folder would put frames of the same recording into train and test.

The split file is part of the repo, so this only has to run again to change the split. The defaults reproduce it:
AMZ's own images and FRT for test, BME for val. Those are the only origins besides AMZ's 2872x920 frames with a
reasonable number of unknown cones, so this way every split has all five classes.
"""
import argparse
from pathlib import Path

import yaml

from driverless_perception.vision.detection.fsoco import SPLITS, group_split, load_fsoco, origin_counts, split_table

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=ROOT / "data/raw/fsoco_bounding_boxes")
    parser.add_argument("--out", type=Path, default=ROOT / "configs/detection/fsoco_split.yaml")
    parser.add_argument("--val-origins", nargs="*", default=["bme"])
    parser.add_argument("--test-origins", nargs="*", default=["amz", "frt"])
    parser.add_argument("--val", type=float, default=0.1, help="share of the images for val")
    parser.add_argument("--test", type=float, default=0.1, help="share of the images for test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    counts = origin_counts(load_fsoco(args.src))
    split = group_split(counts, tuple(args.val_origins), tuple(args.test_origins), args.val, args.test, args.seed)
    table = split_table(counts, split)
    print(table)

    command = (f"--val-origins {' '.join(args.val_origins)} --test-origins {' '.join(args.test_origins)} "
               f"--val {args.val} --test {args.test} --seed {args.seed}")
    header = ["FSOCO split by camera origin (see scripts/split_fsoco.py), written with", command, "",
              *table.splitlines()]
    with open(args.out, "w") as f:
        f.write("".join(f"# {line}".rstrip() + "\n" for line in header))
        yaml.safe_dump({name: split[name] for name in SPLITS}, f, sort_keys=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
