"""Converts FSOCO into a YOLO dataset, split by camera origin as given in configs/detection/fsoco_split.yaml.

The watermark border is cropped, images are shrunk to at most --long-side pixels and saved as JPEG. The result in
data/processed/fsoco_yolo is what the detection configs point to.
"""
import argparse
import shutil
import time
from pathlib import Path

from driverless_perception.vision.detection.fsoco import (
    export_yolo,
    load_fsoco,
    origin_counts,
    read_split,
    split_table,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=ROOT / "data/raw/fsoco_bounding_boxes")
    parser.add_argument("--split", type=Path, default=ROOT / "configs/detection/fsoco_split.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/fsoco_yolo")
    parser.add_argument("--long-side", type=int, default=1920)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="delete --out first if it exists")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.out} is not empty, pass --overwrite to replace it")
        shutil.rmtree(args.out)

    samples = load_fsoco(args.src)
    split = read_split(args.split)
    print(split_table(origin_counts(samples), split))

    start = time.time()
    counts = export_yolo(samples, split, args.out, args.long_side, args.quality, args.workers)
    print(f"done in {time.time() - start:.0f} s: " + ", ".join(f"{n} {name}" for name, n in counts.items()))
    print(f"dataset file: {args.out / 'fsoco.yaml'}")


if __name__ == "__main__":
    main()
